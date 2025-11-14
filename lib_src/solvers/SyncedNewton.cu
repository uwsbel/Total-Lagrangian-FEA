#include <cooperative_groups.h>
#include <cublas_v2.h>  // Add this line for cuBLAS
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataFunc.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataFunc.cuh"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "SyncedNewton.cuh"

namespace cg = cooperative_groups;

// =====================================================
// SPARSE HESSIAN: Bitset-based Sparsity Analysis
// =====================================================

// Helper: Binary search to find column index in sorted array
__device__ int binary_search_column(const int *cols, int n_cols, int target) {
  int left = 0, right = n_cols - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (cols[mid] == target)
      return mid;
    if (cols[mid] < target)
      left = mid + 1;
    else
      right = mid - 1;
  }
  return -1;  // Not found
}
// Step 1: Analyze sparsity pattern using bitsets (union of all contributions)
__global__ void analyze_hessian_sparsity_bitset(GPU_FEAT10_Data *d_data,
                                                SyncedNewtonSolver *d_solver,
                                                unsigned int *d_col_bitset,
                                                int bitset_size) {
  int row_dof = blockIdx.x * blockDim.x + threadIdx.x;
  int n_dofs  = 3 * d_solver->get_n_coef();

  if (row_dof >= n_dofs)
    return;

  unsigned int *my_bitset = &d_col_bitset[row_dof * bitset_size];

  int node_i = row_dof / 3;
  int dof_i  = row_dof % 3;  // 0=x, 1=y, 2=z

  // ===== Contribution 1: Mass Matrix =====
  // M is block-diagonal 3x3 for each node pair (identity structure)
  const int *mass_offsets = d_data->csr_offsets();
  const int *mass_columns = d_data->csr_columns();

  int row_start = mass_offsets[node_i];
  int row_end   = mass_offsets[node_i + 1];

  for (int idx = row_start; idx < row_end; idx++) {
    int node_j = mass_columns[idx];

    // FIXED: Only add diagonal entries of 3×3 block (consistent mass identity)
    // row_dof = node_i*3 + dof_i only couples with col_dof = node_j*3 + dof_i
    int col_dof = node_j * 3 + dof_i;  // Same DOF index

    int word_idx = col_dof / 32;
    int bit_idx  = col_dof % 32;
    atomicOr(&my_bitset[word_idx], 1u << bit_idx);
  }

  // ===== Contribution 2: Tangent Stiffness (Element Connectivity) =====
  // Each element contributes a 30x30 block (10 nodes × 3 DOFs)
  // We need element connectivity: which elements contain node_i
  int n_elems = d_solver->get_n_beam();

  for (int elem = 0; elem < n_elems; elem++) {
    // Check if this element contains node_i
    bool node_in_elem = false;
    for (int local_node = 0; local_node < 10; local_node++) {
      int global_node = d_data->element_connectivity()(elem, local_node);
      if (global_node == node_i) {
        node_in_elem = true;
        break;
      }
    }

    if (node_in_elem) {
      // This element affects row_dof, add all element nodes to sparsity
      for (int local_node = 0; local_node < 10; local_node++) {
        int node_j = d_data->element_connectivity()(elem, local_node);
        for (int dof_j = 0; dof_j < 3; dof_j++) {
          int col_dof  = node_j * 3 + dof_j;
          int word_idx = col_dof / 32;
          int bit_idx  = col_dof % 32;
          atomicOr(&my_bitset[word_idx], 1u << bit_idx);
        }
      }
    }
  }

  // ===== Contribution 3: Constraint Jacobian (J^T * J) =====
  int n_constraints = d_solver->gpu_n_constraints();
  if (n_constraints > 0) {
    // J^T stored in CSR format
    const int *cjT_offsets = d_data->cj_csr_offsets();
    const int *cjT_columns = d_data->cj_csr_columns();

    // Find constraints that affect this DOF
    int col_start = cjT_offsets[row_dof];
    int col_end   = cjT_offsets[row_dof + 1];

    for (int idx1 = col_start; idx1 < col_end; idx1++) {
      int c_idx = cjT_columns[idx1];

      // For this constraint, find all DOFs it couples to (via J^T @ J)
      // Scan all DOFs to find which ones are affected by constraint c_idx
      for (int other_dof = 0; other_dof < n_dofs; other_dof++) {
        int other_start = cjT_offsets[other_dof];
        int other_end   = cjT_offsets[other_dof + 1];

        // Check if constraint c_idx affects other_dof
        for (int idx2 = other_start; idx2 < other_end; idx2++) {
          if (cjT_columns[idx2] == c_idx) {
            // Add coupling between row_dof and other_dof
            int word_idx = other_dof / 32;
            int bit_idx  = other_dof % 32;
            atomicOr(&my_bitset[word_idx], 1u << bit_idx);
            break;
          }
        }
      }
    }
  }
}

// Step 2: Count non-zeros from bitset
__global__ void count_nnz_from_bitset(unsigned int *d_col_bitset,
                                      int *d_nnz_per_row, int n_dofs,
                                      int bitset_size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_dofs)
    return;

  unsigned int *my_bitset = &d_col_bitset[row * bitset_size];
  int count               = 0;

  for (int w = 0; w < bitset_size; w++) {
    count += __popc(my_bitset[w]);
  }

  d_nnz_per_row[row] = count;
}

// Step 3: Prefix sum to compute row offsets (call thrust::exclusive_scan on
// host)
__global__ void extract_columns_from_bitset(unsigned int *d_col_bitset,
                                            int *d_csr_row_offsets,
                                            int *d_csr_col_indices, int n_dofs,
                                            int bitset_size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_dofs)
    return;

  unsigned int *my_bitset = &d_col_bitset[row * bitset_size];
  int write_pos           = d_csr_row_offsets[row];

  // Scan all words in order
  for (int w = 0; w < bitset_size; w++) {
    unsigned int word = my_bitset[w];

    // Process bits from LSB to MSB (gives sorted order)
    for (int bit = 0; bit < 32; bit++) {
      if (word & (1u << bit)) {
        int col = w * 32 + bit;

        if (col < n_dofs) {
          d_csr_col_indices[write_pos++] = col;
        }
      }
    }
  }

  // DEBUG: Print for first few rows
  if (row < 3) {
    int row_start = d_csr_row_offsets[row];
    int row_end   = d_csr_row_offsets[row + 1];
    printf("DEBUG extract: Row %d wrote %d entries (expected %d)\n", row,
           write_pos - row_start, row_end - row_start);
  }
}

__global__ void debug_check_sparsity_pattern(int *d_csr_row_offsets,
                                             int *d_csr_col_indices,
                                             int n_dofs) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= min(n_dofs, 10))
    return;

  int row_start = d_csr_row_offsets[row];
  int row_end   = d_csr_row_offsets[row + 1];

  printf("Row %d: nnz=%d, cols=[", row, row_end - row_start);
  for (int idx = row_start; idx < min(row_end, row_start + 10); idx++) {
    printf("%d ", d_csr_col_indices[idx]);
  }
  printf("...]\n");

  for (int idx = row_start; idx < row_end - 1; idx++) {
    if (d_csr_col_indices[idx] >= d_csr_col_indices[idx + 1]) {
      printf("ERROR: Row %d NOT SORTED at idx %d: %d >= %d\n", row,
             idx - row_start, d_csr_col_indices[idx],
             d_csr_col_indices[idx + 1]);
    }
  }
}

__global__ void debug_mass_assembly_lookups(GPU_FEAT10_Data *d_data,
                                            SyncedNewtonSolver *d_solver,
                                            int *d_csr_row_offsets,
                                            int *d_csr_col_indices) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= min(d_solver->get_n_coef(), 5))
    return;

  int node_i = tid;

  const int *mass_offsets = d_data->csr_offsets();
  const int *mass_columns = d_data->csr_columns();

  int row_start = mass_offsets[node_i];
  int row_end   = mass_offsets[node_i + 1];

  printf("\nNode %d has %d mass connections:\n", node_i, row_end - row_start);

  for (int idx = row_start; idx < row_end; idx++) {
    int node_j = mass_columns[idx];

    for (int dof_i = 0; dof_i < 3; dof_i++) {
      for (int dof_j = 0; dof_j < 3; dof_j++) {
        if (dof_i == dof_j) {
          int row_dof = node_i * 3 + dof_i;
          int col_dof = node_j * 3 + dof_j;

          int row_begin  = d_csr_row_offsets[row_dof];
          int row_length = d_csr_row_offsets[row_dof + 1] - row_begin;

          int pos = binary_search_column(&d_csr_col_indices[row_begin],
                                         row_length, col_dof);

          if (pos < 0) {
            printf(
                "  MISSING: row_dof=%d wants col_dof=%d (node_i=%d dof=%d -> "
                "node_j=%d dof=%d)\n",
                row_dof, col_dof, node_i, dof_i, node_j, dof_j);
            printf("    Row %d has cols: ", row_dof);
            for (int p = 0; p < min(row_length, 20); p++) {
              printf("%d ", d_csr_col_indices[row_begin + p]);
            }
            printf("\n");
          }
        }
      }
    }
  }
}

// =====================================================
// SPARSE HESSIAN: Assembly Kernels
// =====================================================

// Assemble Mass Contribution: (M/h) into sparse H
// CORRECTED: Handles full 3x3 blocks for consistent mass
__global__ void assemble_sparse_hessian_mass(GPU_FEAT10_Data *d_data,
                                             SyncedNewtonSolver *d_solver,
                                             int *d_csr_row_offsets,
                                             int *d_csr_col_indices,
                                             double *d_csr_values) {
  int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  int n_nodes = d_solver->get_n_coef();

  if (tid >= n_nodes)
    return;

  const double inv_h = 1.0 / d_solver->solver_time_step();
  int node_i         = tid;

  const int *mass_offsets   = d_data->csr_offsets();
  const int *mass_columns   = d_data->csr_columns();
  const double *mass_values = d_data->csr_values();

  int row_start = mass_offsets[node_i];
  int row_end   = mass_offsets[node_i + 1];

  for (int idx = row_start; idx < row_end; idx++) {
    int node_j     = mass_columns[idx];
    double mass_ij = mass_values[idx];
    double contrib = mass_ij * inv_h;

    // Add to FULL 3x3 block (i, j) in sparse Hessian
    // Consistent mass: M_ij = mass_ij * I_3x3
    for (int dof_i = 0; dof_i < 3; dof_i++) {
      for (int dof_j = 0; dof_j < 3; dof_j++) {
        int row_dof = node_i * 3 + dof_i;
        int col_dof = node_j * 3 + dof_j;

        // Only add to diagonal of the 3x3 block (identity structure)
        if (dof_i == dof_j) {
          // Find position in CSR structure
          int row_begin  = d_csr_row_offsets[row_dof];
          int row_length = d_csr_row_offsets[row_dof + 1] - row_begin;

          int pos = binary_search_column(&d_csr_col_indices[row_begin],
                                         row_length, col_dof);
          // printf("pos: %d\n", pos);

          if (pos >= 0) {
            atomicAdd(&d_csr_values[row_begin + pos], contrib);
          }
        }
      }
    }
  }
}

// Assemble Tangent Stiffness: (h * Kt) into sparse H
__global__ void assemble_sparse_hessian_tangent(GPU_FEAT10_Data *d_data,
                                                SyncedNewtonSolver *d_solver,
                                                int *d_csr_row_offsets,
                                                int *d_csr_col_indices,
                                                double *d_csr_values) {
  int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int n_elem = d_data->gpu_n_elem();
  int n_qp   = Quadrature::N_QP_T10_5;

  int elem_idx = tid / n_qp;
  int qp_idx   = tid % n_qp;

  if (elem_idx >= n_elem)
    return;

  const double h = d_solver->solver_time_step();

  // Get element connectivity
  int global_node_indices[10];
#pragma unroll
  for (int node = 0; node < 10; node++) {
    global_node_indices[node] = d_data->element_connectivity()(elem_idx, node);
  }

  // Get current nodal positions for this element
  double x_nodes[10][3];
#pragma unroll
  for (int node = 0; node < 10; node++) {
    int global_node_idx = global_node_indices[node];
    x_nodes[node][0]    = d_data->x12()(global_node_idx);
    x_nodes[node][1]    = d_data->y12()(global_node_idx);
    x_nodes[node][2]    = d_data->z12()(global_node_idx);
  }

  // Get precomputed shape function gradients
  double grad_N[10][3];
#pragma unroll
  for (int a = 0; a < 10; a++) {
    grad_N[a][0] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 0);
    grad_N[a][1] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 1);
    grad_N[a][2] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 2);
  }

  // Compute deformation gradient F
  double F[3][3] = {{0.0}};
#pragma unroll
  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        F[i][j] += x_nodes[a][i] * grad_N[a][j];
      }
    }
  }

  // Compute C = F^T * F
  double C[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
#pragma unroll
      for (int k = 0; k < 3; k++) {
        C[i][j] += F[k][i] * F[k][j];
      }
    }
  }

  // Compute tr(E)
  double trC = C[0][0] + C[1][1] + C[2][2];
  double trE = 0.5 * (trC - 3.0);

  // Compute F * F^T
  double FFT[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
#pragma unroll
      for (int k = 0; k < 3; k++) {
        FFT[i][j] += F[i][k] * F[j][k];
      }
    }
  }

  // Precompute Fh[i] = F @ grad_N[i]
  double Fh[10][3];
#pragma unroll
  for (int i = 0; i < 10; i++) {
#pragma unroll
    for (int row = 0; row < 3; row++) {
      Fh[i][row] = 0.0;
#pragma unroll
      for (int col = 0; col < 3; col++) {
        Fh[i][row] += F[row][col] * grad_N[i][col];
      }
    }
  }

  // Material and quadrature
  double lambda = d_data->lambda();
  double mu     = d_data->mu();
  double detJ   = d_data->detJ_ref(elem_idx, qp_idx);
  double wq     = d_data->tet5pt_weights(qp_idx);
  double dV     = detJ * wq;

// Build local 30x30 tangent and scatter to global sparse H
#pragma unroll
  for (int i = 0; i < 10; i++) {
#pragma unroll
    for (int j = 0; j < 10; j++) {
      // Compute scalars
      double hij = grad_N[j][0] * grad_N[i][0] + grad_N[j][1] * grad_N[i][1] +
                   grad_N[j][2] * grad_N[i][2];

      double Fhj_dot_Fhi =
          Fh[j][0] * Fh[i][0] + Fh[j][1] * Fh[i][1] + Fh[j][2] * Fh[i][2];

// Fill 3x3 block (i,j)
#pragma unroll
      for (int d = 0; d < 3; d++) {
#pragma unroll
        for (int e = 0; e < 3; e++) {
          double A_de    = lambda * Fh[i][d] * Fh[j][e];
          double B_de    = lambda * trE * hij * (d == e ? 1.0 : 0.0);
          double C1_de   = mu * Fhj_dot_Fhi * (d == e ? 1.0 : 0.0);
          double D_de    = mu * Fh[j][d] * Fh[i][e];
          double Etrm_de = mu * hij * FFT[d][e];
          double Ftrm_de = -mu * hij * (d == e ? 1.0 : 0.0);

          double K_ij_de =
              (A_de + B_de + C1_de + D_de + Etrm_de + Ftrm_de) * dV;

          // Map to global DOFs
          int global_dof_i = global_node_indices[i] * 3 + d;
          int global_dof_j = global_node_indices[j] * 3 + e;

          // Find position in CSR and add h * K_ij_de
          int row_begin  = d_csr_row_offsets[global_dof_i];
          int row_length = d_csr_row_offsets[global_dof_i + 1] - row_begin;

          int pos = binary_search_column(&d_csr_col_indices[row_begin],
                                         row_length, global_dof_j);

          if (pos >= 0) {
            atomicAdd(&d_csr_values[row_begin + pos], h * K_ij_de);
          }
        }
      }
    }
  }
}

// Assemble Constraint Contribution: (h^2 * rho * J^T * J) into sparse H
// NO CHANGES NEEDED - already correct
__global__ void assemble_sparse_hessian_constraints(
    GPU_FEAT10_Data *d_data, SyncedNewtonSolver *d_solver,
    int *d_csr_row_offsets, int *d_csr_col_indices, double *d_csr_values) {
  int tid           = blockIdx.x * blockDim.x + threadIdx.x;
  int n_constraints = d_solver->gpu_n_constraints();

  if (tid >= n_constraints)
    return;

  const double h      = d_solver->solver_time_step();
  const double rho    = *d_solver->solver_rho();
  const double factor = h * h * rho;

  int c_idx  = tid;
  int n_dofs = 3 * d_solver->get_n_coef();

  const int *cjT_offsets   = d_data->cj_csr_offsets();
  const int *cjT_columns   = d_data->cj_csr_columns();
  const double *cjT_values = d_data->cj_csr_values();

  // Find all DOFs affected by this constraint
  for (int dof_i = 0; dof_i < n_dofs; dof_i++) {
    int col_start_i = cjT_offsets[dof_i];
    int col_end_i   = cjT_offsets[dof_i + 1];

    double J_ic  = 0.0;
    bool found_i = false;

    // Check if constraint c_idx affects dof_i
    for (int idx = col_start_i; idx < col_end_i; idx++) {
      if (cjT_columns[idx] == c_idx) {
        J_ic    = cjT_values[idx];
        found_i = true;
        break;
      }
    }

    if (!found_i)
      continue;

    // Now find all dof_j that this constraint couples with
    for (int dof_j = 0; dof_j < n_dofs; dof_j++) {
      int col_start_j = cjT_offsets[dof_j];
      int col_end_j   = cjT_offsets[dof_j + 1];

      double J_jc = 0.0;

      for (int idx = col_start_j; idx < col_end_j; idx++) {
        if (cjT_columns[idx] == c_idx) {
          J_jc = cjT_values[idx];
          break;
        }
      }

      if (J_jc != 0.0) {
        // Add factor * J_ic * J_jc to H(dof_i, dof_j)
        int row_begin  = d_csr_row_offsets[dof_i];
        int row_length = d_csr_row_offsets[dof_i + 1] - row_begin;

        int pos = binary_search_column(&d_csr_col_indices[row_begin],
                                       row_length, dof_j);

        if (pos >= 0) {
          atomicAdd(&d_csr_values[row_begin + pos], factor * J_ic * J_jc);
        }
      }
    }
  }
}

template <typename ElementType>
__device__ double solver_grad_L(int tid, ElementType *data,
                                SyncedNewtonSolver *d_solver) {
  double res = 0.0;

  const int node_i = tid / 3;
  const int dof_i  = tid % 3;

  const double inv_dt = 1.0 / d_solver->solver_time_step();
  const double dt     = d_solver->solver_time_step();

  // Cache pointers once
  const double *__restrict__ v_g    = d_solver->v_guess().data();
  const double *__restrict__ v_p    = d_solver->v_prev().data();
  const int *__restrict__ offsets   = data->csr_offsets();
  const int *__restrict__ columns   = data->csr_columns();
  const double *__restrict__ values = data->csr_values();

  // Mass matrix contribution: (M @ (v_loc - v_prev)) / h
  int row_start = offsets[node_i];
  int row_end   = offsets[node_i + 1];

  for (int idx = row_start; idx < row_end; idx++) {
    int node_j     = columns[idx];
    double mass_ij = values[idx];
    int tid_j      = node_j * 3 + dof_i;
    double v_diff  = v_g[tid_j] - v_p[tid_j];
    res += mass_ij * v_diff * inv_dt;
  }

  // Mechanical force contribution: - (-f_int + f_ext) = f_int - f_ext
  res -= (-data->f_int()(tid));  // Add f_int
  res -= data->f_ext()(tid);     // Subtract f_ext

  const int n_constraints = d_solver->gpu_n_constraints();

  if (n_constraints > 0) {
    // Python: h * (J.T @ (lam_mult + rho_bb * cA))
    const double rho = *d_solver->solver_rho();

    const double *__restrict__ lam = d_solver->lambda_guess().data();
    const double *__restrict__ con = data->constraint().data();

    // CSR format stores J^T (transpose of constraint Jacobian)
    const int *__restrict__ cjT_offsets   = data->cj_csr_offsets();
    const int *__restrict__ cjT_columns   = data->cj_csr_columns();
    const double *__restrict__ cjT_values = data->cj_csr_values();

    // Get all constraints that affect this DOF (tid)
    const int col_start = cjT_offsets[tid];
    const int col_end   = cjT_offsets[tid + 1];

    for (int idx = col_start; idx < col_end; idx++) {
      const int constraint_idx        = cjT_columns[idx];
      const double constraint_jac_val = cjT_values[idx];
      const double constraint_val     = con[constraint_idx];

      // Add constraint contribution: h * J^T * (lambda + rho*c)
      res += dt * constraint_jac_val *
             (lam[constraint_idx] + rho * constraint_val);
    }
  }

  return res;
}

// ===============================================
// cusparse and cudss one step newton
// ===============================================

__global__ void cudss_solve_update_pos_prev(
    GPU_FEAT10_Data *d_data, SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < d_newton_solver->get_n_coef()) {
    d_newton_solver->x12_prev()(tid) = d_data->x12()(tid);
    d_newton_solver->y12_prev()(tid) = d_data->y12()(tid);
    d_newton_solver->z12_prev()(tid) = d_data->z12()(tid);
  }
}

__global__ void cudss_solve_compute_p(GPU_FEAT10_Data *d_data,
                                      SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < d_newton_solver->get_n_beam() * d_newton_solver->gpu_n_total_qp()) {
    int idx      = tid;
    int elem_idx = idx / d_newton_solver->gpu_n_total_qp();
    int qp_idx   = idx % d_newton_solver->gpu_n_total_qp();
    compute_p(elem_idx, qp_idx, d_data);
  }
}

__global__ void cudss_solve_clear_internal_force(GPU_FEAT10_Data *d_data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < d_data->n_coef * 3) {
    clear_internal_force(d_data);
  }
}

__global__ void cudss_solve_compute_internal_force(
    GPU_FEAT10_Data *d_data, SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < d_newton_solver->get_n_beam() * d_newton_solver->gpu_n_shape()) {
    int idx      = tid;
    int elem_idx = idx / d_newton_solver->gpu_n_shape();
    int node_idx = idx % d_newton_solver->gpu_n_shape();
    compute_internal_force(elem_idx, node_idx, d_data);
  }
}

__global__ void cudss_solve_constraints_eval(
    GPU_FEAT10_Data *d_data, SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < d_newton_solver->gpu_n_constraints() / 3) {
    compute_constraint_data(d_data);
  }
}

__global__ void cudss_solve_update_dual_var(
    GPU_FEAT10_Data *d_data, SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int n_constraints = d_newton_solver->gpu_n_constraints();
  if (tid < n_constraints) {
    double constraint_val = d_data->constraint()[tid];
    d_newton_solver->lambda_guess()[tid] +=
        *d_newton_solver->solver_rho() * constraint_val;
  }
}

__global__ void cudss_solve_compute_grad_l(
    GPU_FEAT10_Data *d_data, SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < d_newton_solver->get_n_coef() * 3) {
    double g                  = solver_grad_L(tid, d_data, d_newton_solver);
    d_newton_solver->g()[tid] = g;
  }
}

__global__ void cudss_solve_initialize_prehess(
    GPU_FEAT10_Data *d_data, SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_newton_solver->get_n_coef() * 3) {
    d_newton_solver->delta_v()[tid] = 0.0;
    d_newton_solver->r()[tid]       = -d_newton_solver->g()[tid];
  }
}

__global__ void cudss_solve_update_pos(SyncedNewtonSolver *d_newton_solver,
                                       GPU_FEAT10_Data *d_data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_newton_solver->get_n_coef()) {
    d_data->x12()(tid) = d_newton_solver->x12_prev()(tid) +
                         d_newton_solver->v_guess()(tid * 3 + 0) *
                             d_newton_solver->solver_time_step();
    d_data->y12()(tid) = d_newton_solver->y12_prev()(tid) +
                         d_newton_solver->v_guess()(tid * 3 + 1) *
                             d_newton_solver->solver_time_step();
    d_data->z12()(tid) = d_newton_solver->z12_prev()(tid) +
                         d_newton_solver->v_guess()(tid * 3 + 2) *
                             d_newton_solver->solver_time_step();
  }
}

__global__ void cudss_solve_update_v_guess(
    SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_newton_solver->get_n_coef() * 3) {
    d_newton_solver->v_guess()[tid] += d_newton_solver->delta_v()[tid];
  }
}

__global__ void cudss_solve_update_v_prev(SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_newton_solver->get_n_coef() * 3) {
    d_newton_solver->v_prev()[tid] = d_newton_solver->v_guess()[tid];
  }
}

// Replace lines 744-767 with this optimized version using member handles:
double SyncedNewtonSolver::compute_l2_norm_cublas(double *d_vec, int n_dofs) {
  // Reuse persistent cublas_handle_ and d_norm_temp_ (no
  // create/destroy/malloc/free!)
  cublasDnrm2(cublas_handle_, n_dofs, d_vec, 1, d_norm_temp_);

  double h_norm;
  HANDLE_ERROR(cudaMemcpy(&h_norm, d_norm_temp_, sizeof(double),
                          cudaMemcpyDeviceToHost));
  return h_norm;
}

void SyncedNewtonSolver::AnalyzeHessianSparsity() {
  if (sparse_hessian_initialized_)
    return;  // Already done

  std::cout << "Analyzing Hessian sparsity pattern..." << std::endl;

  int n_dofs      = 3 * n_coef_;
  int bitset_size = (n_dofs + 31) / 32;

  // Allocate temporary bitset workspace
  HANDLE_ERROR(
      cudaMalloc(&d_col_bitset_, n_dofs * bitset_size * sizeof(unsigned int)));
  HANDLE_ERROR(cudaMemset(d_col_bitset_, 0,
                          n_dofs * bitset_size * sizeof(unsigned int)));

  HANDLE_ERROR(cudaMalloc(&d_nnz_per_row_, n_dofs * sizeof(int)));
  HANDLE_ERROR(cudaMemset(d_nnz_per_row_, 0, n_dofs * sizeof(int)));

  // Step 1: Analyze sparsity (bitset union)
  int numBlocks = (n_dofs + 255) / 256;
  analyze_hessian_sparsity_bitset<<<numBlocks, 256>>>(
      static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_, d_col_bitset_,
      bitset_size);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Step 2: Count nnz per row
  count_nnz_from_bitset<<<numBlocks, 256>>>(d_col_bitset_, d_nnz_per_row_,
                                            n_dofs, bitset_size);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Step 3: Allocate row_offsets and compute prefix sum
  HANDLE_ERROR(cudaMalloc(&d_csr_row_offsets_, (n_dofs + 1) * sizeof(int)));

  // Option 1: Use inclusive_scan then prepend 0
  // We'll compute cumulative sum: offset[i] = sum of nnz[0..i-1]

  thrust::device_ptr<int> nnz_ptr(d_nnz_per_row_);
  thrust::device_ptr<int> offset_ptr(d_csr_row_offsets_);

  // First, set offset[0] = 0
  int zero = 0;
  HANDLE_ERROR(cudaMemcpy(d_csr_row_offsets_, &zero, sizeof(int),
                          cudaMemcpyHostToDevice));

  // Then compute inclusive scan into a temp array
  int *d_temp_scan;
  HANDLE_ERROR(cudaMalloc(&d_temp_scan, n_dofs * sizeof(int)));
  thrust::device_ptr<int> temp_ptr(d_temp_scan);
  thrust::inclusive_scan(nnz_ptr, nnz_ptr + n_dofs, temp_ptr);

  // Copy the inclusive scan result to offset[1..n_dofs]
  HANDLE_ERROR(cudaMemcpy(d_csr_row_offsets_ + 1, d_temp_scan,
                          n_dofs * sizeof(int), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaFree(d_temp_scan));

  HANDLE_ERROR(cudaDeviceSynchronize());

  // Get total nnz
  HANDLE_ERROR(cudaMemcpy(&h_nnz_, &d_csr_row_offsets_[n_dofs], sizeof(int),
                          cudaMemcpyDeviceToHost));

  std::cout << "Sparse Hessian: " << n_dofs << " x " << n_dofs
            << ", nnz = " << h_nnz_ << " ("
            << (100.0 * h_nnz_) / (n_dofs * n_dofs) << "%)" << std::endl;

  // Step 4: Allocate col_indices and values
  HANDLE_ERROR(cudaMalloc(&d_csr_col_indices_, h_nnz_ * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_csr_values_, h_nnz_ * sizeof(double)));

  // Step 5: Extract column indices from bitset
  extract_columns_from_bitset<<<numBlocks, 256>>>(
      d_col_bitset_, d_csr_row_offsets_, d_csr_col_indices_, n_dofs,
      bitset_size);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // DEBUG: Check sparsity pattern
  std::cout << "DEBUG: Checking sparsity pattern..." << std::endl;
  debug_check_sparsity_pattern<<<(10 + 255) / 256, 256>>>(
      d_csr_row_offsets_, d_csr_col_indices_, n_dofs);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // DEBUG: Check what mass assembly will look for
  std::cout << "DEBUG: Checking mass assembly lookups..." << std::endl;
  debug_mass_assembly_lookups<<<(5 + 255) / 256, 256>>>(
      static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_,
      d_csr_row_offsets_, d_csr_col_indices_);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Free temporary workspace
  cudaFree(d_col_bitset_);
  d_col_bitset_ = nullptr;
  cudaFree(d_nnz_per_row_);
  d_nnz_per_row_ = nullptr;

  sparse_hessian_initialized_ = true;
  std::cout << "Sparsity analysis complete." << std::endl;
}

// Replace your OneStepNewtonCuDSS() function with this debugged version:
// Replace the entire OneStepNewtonCuDSS function (lines 900-1128):
void SyncedNewtonSolver::OneStepNewtonCuDSS() {
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  int threadsPerBlock = 256;
  int numBlocks_compute_p =
      (n_beam_ * n_total_qp_ + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocks_clear_internal_force =
      (n_coef_ * 3 + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocks_internal_force =
      (n_beam_ * n_shape_ + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocks_grad_l = (n_coef_ * 3 + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocks_constraints_eval =
      (n_constraints_ / 3 + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocks_initialize_prehess =
      (n_coef_ * 3 + threadsPerBlock - 1) / threadsPerBlock;

  // SPARSE HESSIAN: New block sizes
  int numBlocks_sparse_mass = (n_coef_ + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocks_sparse_tangent =
      (n_beam_ * Quadrature::N_QP_T10_5 + threadsPerBlock - 1) /
      threadsPerBlock;
  int numBlocks_sparse_constraint =
      (n_constraints_ + threadsPerBlock - 1) / threadsPerBlock;

  int numBlocks_update_v_guess =
      (n_coef_ * 3 + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocks_update_pos = (n_coef_ + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocks_update_pos_prev =
      (n_coef_ + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocks_update_dual_var =
      (n_constraints_ + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocks_update_prev_v =
      (n_coef_ * 3 + threadsPerBlock - 1) / threadsPerBlock;
  int n_dofs = 3 * n_coef_;

  // ===== CREATE cuDSS HANDLES ONCE PER TIMESTEP =====
  if (!cudss_handle_) {
    CUDSS_OK(cudssCreate(&cudss_handle_));
    CUDSS_OK(cudssConfigCreate(&cudss_config_));
    CUDSS_OK(cudssDataCreate(cudss_handle_, &cudss_data_));
  }

  // Create matrix/vector descriptors ONCE (reuse pointers, sparsity pattern
  // fixed)
  cudssMatrix_t dssA, dssB, dssX;
  CUDSS_OK(cudssMatrixCreateCsr(
      &dssA, n_dofs, n_dofs, h_nnz_, d_csr_row_offsets_, nullptr,
      d_csr_col_indices_, d_csr_values_, CUDA_R_32I, CUDA_R_64F,
      CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
  CUDSS_OK(cudssMatrixCreateDn(&dssB, n_dofs, 1, n_dofs, d_r_, CUDA_R_64F,
                               CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_OK(cudssMatrixCreateDn(&dssX, n_dofs, 1, n_dofs, d_delta_v_, CUDA_R_64F,
                               CUDSS_LAYOUT_COL_MAJOR));

  // Analysis phase ONCE per timestep (sparsity pattern doesn't change)
  CUDSS_OK(cudssExecute(cudss_handle_, CUDSS_PHASE_ANALYSIS, cudss_config_,
                        cudss_data_, dssA, dssX, dssB));

  HANDLE_ERROR(cudaEventRecord(start));

  cudss_solve_update_pos_prev<<<numBlocks_update_pos_prev, threadsPerBlock>>>(
      static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_);

  for (int outer_iter = 0; outer_iter < h_max_outer_; ++outer_iter) {
    std::cout << "Outer iter " << outer_iter << std::endl;

    for (int newton_iter = 0; newton_iter < h_max_inner_; ++newton_iter) {
      std::cout << "  Newton iter " << newton_iter << std::endl;

      // Compute gradient
      cudss_solve_compute_p<<<numBlocks_compute_p, threadsPerBlock>>>(
          static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_);

      cudss_solve_clear_internal_force<<<numBlocks_clear_internal_force,
                                         threadsPerBlock>>>(
          static_cast<GPU_FEAT10_Data *>(d_data_));
      cudss_solve_compute_internal_force<<<numBlocks_internal_force,
                                           threadsPerBlock>>>(
          static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_);

      cudss_solve_constraints_eval<<<numBlocks_constraints_eval,
                                     threadsPerBlock>>>(
          static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_);
      cudss_solve_compute_grad_l<<<numBlocks_grad_l, threadsPerBlock>>>(
          static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_);

      cudss_solve_initialize_prehess<<<numBlocks_initialize_prehess,
                                       threadsPerBlock>>>(
          static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_);

      // ===== SPARSE HESSIAN ASSEMBLY =====
      HANDLE_ERROR(cudaMemset(d_csr_values_, 0, h_nnz_ * sizeof(double)));

      assemble_sparse_hessian_mass<<<numBlocks_sparse_mass, threadsPerBlock>>>(
          static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_,
          d_csr_row_offsets_, d_csr_col_indices_, d_csr_values_);

      assemble_sparse_hessian_tangent<<<numBlocks_sparse_tangent,
                                        threadsPerBlock>>>(
          static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_,
          d_csr_row_offsets_, d_csr_col_indices_, d_csr_values_);

      if (n_constraints_ > 0) {
        assemble_sparse_hessian_constraints<<<numBlocks_sparse_constraint,
                                              threadsPerBlock>>>(
            static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_,
            d_csr_row_offsets_, d_csr_col_indices_, d_csr_values_);
      }

      HANDLE_ERROR(cudaDeviceSynchronize());

      // ===== SOLVE USING cuDSS (REUSE HANDLES!) =====

      // Factorization (Hessian values updated, but sparsity unchanged)
      CUDSS_OK(cudssExecute(cudss_handle_, CUDSS_PHASE_FACTORIZATION,
                            cudss_config_, cudss_data_, dssA, dssX, dssB));

      // Solve
      HANDLE_ERROR(cudaMemset(d_delta_v_, 0, n_dofs * sizeof(double)));
      CUDSS_OK(cudssExecute(cudss_handle_, CUDSS_PHASE_SOLVE, cudss_config_,
                            cudss_data_, dssA, dssX, dssB));

      // Update position
      cudss_solve_update_v_guess<<<numBlocks_update_v_guess, threadsPerBlock>>>(
          d_newton_solver_);
      cudss_solve_update_pos<<<numBlocks_update_pos, threadsPerBlock>>>(
          d_newton_solver_, static_cast<GPU_FEAT10_Data *>(d_data_));

      // Check convergence
      HANDLE_ERROR(cudaDeviceSynchronize());
      double norm_g = compute_l2_norm_cublas(d_g_, n_dofs);
      std::cout << "    ||g|| = " << std::scientific << norm_g << std::endl;

      if (norm_g < h_inner_tol_) {
        break;
      }
    }

    cudss_solve_update_v_prev<<<numBlocks_update_prev_v, threadsPerBlock>>>(
        d_newton_solver_);

    cudss_solve_constraints_eval<<<numBlocks_constraints_eval,
                                   threadsPerBlock>>>(
        static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_);

    cudss_solve_update_dual_var<<<numBlocks_update_dual_var, threadsPerBlock>>>(
        static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_);

    // Check constraint convergence
    HANDLE_ERROR(cudaDeviceSynchronize());

    if (n_constraints_ > 0) {
      double norm_constraint =
          compute_l2_norm_cublas(d_constraint_ptr_, n_constraints_);
      std::cout << "  Outer iter " << outer_iter
                << ": ||c|| = " << std::scientific << norm_constraint
                << std::endl;

      if (norm_constraint < h_outer_tol_) {
        std::cout << "  Outer loop converged at iteration " << outer_iter
                  << std::endl;
        break;
      }
    }
  }

  // Cleanup matrix descriptors (keep handles for next timestep!)
  CUDSS_OK(cudssMatrixDestroy(dssA));
  CUDSS_OK(cudssMatrixDestroy(dssB));
  CUDSS_OK(cudssMatrixDestroy(dssX));

  HANDLE_ERROR(cudaDeviceSynchronize());
  float milliseconds = 0;
  HANDLE_ERROR(cudaEventRecord(stop));
  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

  std::cout << "OneStepNewtonCuDSS kernel time: " << milliseconds << " ms"
            << std::endl;

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
}