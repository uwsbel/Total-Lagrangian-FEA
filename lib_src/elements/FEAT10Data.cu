#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    FEAT10Data.cu
 * Brief:   Implements GPU-side data management and element kernels for
 *          10-node tetrahedral FEAT10 elements. Handles allocation,
 *          initialization, mass and stiffness assembly, internal/external
 *          force evaluation, and optional constraint coupling.
 *==============================================================
 *==============================================================*/

#include "FEAT10Data.cuh"
#include "FEAT10DataFunc.cuh"

namespace cg = cooperative_groups;

__global__ void build_mass_keys_feat10_kernel(GPU_FEAT10_Data *d_data,
                                              unsigned long long *d_keys) {
  const int total = d_data->gpu_n_elem() * Quadrature::N_NODE_T10_10 *
                    Quadrature::N_NODE_T10_10;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }

  const int elem =
      tid / (Quadrature::N_NODE_T10_10 * Quadrature::N_NODE_T10_10);
  const int item_local =
      tid % (Quadrature::N_NODE_T10_10 * Quadrature::N_NODE_T10_10);
  const int i_local = item_local / Quadrature::N_NODE_T10_10;
  const int j_local = item_local % Quadrature::N_NODE_T10_10;

  const int i_global = d_data->element_connectivity()(elem, i_local);
  const int j_global = d_data->element_connectivity()(elem, j_local);

  const unsigned long long key =
      (static_cast<unsigned long long>(static_cast<unsigned int>(i_global))
       << 32) |
      static_cast<unsigned long long>(static_cast<unsigned int>(j_global));
  d_keys[tid] = key;
}

__global__ void decode_mass_keys_kernel(const unsigned long long *d_keys,
                                        int nnz, int *d_csr_columns,
                                        int *d_row_counts) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }

  const unsigned long long key = d_keys[tid];
  const int row                = static_cast<int>(key >> 32);
  const int col                = static_cast<int>(key & 0xffffffffULL);
  d_csr_columns[tid]           = col;
  atomicAdd(d_row_counts + row, 1);
}

__global__ void set_last_offset_kernel(int *d_offsets, int n_rows, int nnz) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_offsets[n_rows] = nnz;
  }
}

__device__ __forceinline__ int binary_search_column_csr_mass(const int *cols,
                                                             int n_cols,
                                                             int target) {
  int left  = 0;
  int right = n_cols - 1;
  while (left <= right) {
    int mid = left + ((right - left) >> 1);
    int v   = cols[mid];
    if (v == target)
      return mid;
    if (v < target)
      left = mid + 1;
    else
      right = mid - 1;
  }
  return -1;
}

__global__ void dn_du_pre_kernel(GPU_FEAT10_Data *d_data) {
  // Get global thread index
  int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate element index and quadrature point index
  int elem_idx = global_thread_idx / Quadrature::N_QP_T10_5;
  int qp_idx   = global_thread_idx % Quadrature::N_QP_T10_5;

  // Bounds check
  if (elem_idx >= d_data->gpu_n_elem() || qp_idx >= Quadrature::N_QP_T10_5) {
    return;
  }

  // Get quadrature point coordinates (xi, eta, zeta)
  double xi   = d_data->tet5pt_x(qp_idx);  // L2 in Python code
  double eta  = d_data->tet5pt_y(qp_idx);  // L3 in Python code
  double zeta = d_data->tet5pt_z(qp_idx);  // L4 in Python code

  // Compute barycentric coordinates
  double L1   = 1.0 - xi - eta - zeta;
  double L2   = xi;
  double L3   = eta;
  double L4   = zeta;
  double L[4] = {L1, L2, L3, L4};

  // Derivatives of barycentric coordinates (dL matrix from Python)
  double dL[4][3] = {
      {-1.0, -1.0, -1.0},  // dL1/dxi, dL1/deta, dL1/dzeta
      {1.0, 0.0, 0.0},     // dL2/dxi, dL2/deta, dL2/dzeta
      {0.0, 1.0, 0.0},     // dL3/dxi, dL3/deta, dL3/dzeta
      {0.0, 0.0, 1.0}      // dL4/dxi, dL4/deta, dL4/dzeta
  };

  // Compute shape function derivatives dN_dxi for all 10 nodes
  double dN_dxi[10][3];

  // Corner nodes (0-3): dN_dxi[i, :] = (4*L[i]-1)*dL[i, :]
  for (int i = 0; i < 4; i++) {
    double factor = 4.0 * L[i] - 1.0;
    for (int j = 0; j < 3; j++) {
      dN_dxi[i][j] = factor * dL[i][j];
    }
  }

  // Edge nodes (4-9): dN_dxi[k, :] = 4*(L[i]*dL[j, :] + L[j]*dL[i, :])
  // Edge connectivity: [(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]
  int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};

  for (int k = 0; k < 6; k++) {
    int i = edges[k][0];
    int j = edges[k][1];

    for (int d = 0; d < 3; d++) {
      dN_dxi[k + 4][d] = 4.0 * (L[i] * dL[j][d] + L[j] * dL[i][d]);
    }
  }

  // Get element node coordinates for this element
  double X_elem[10][3];  // 10 nodes × 3 coordinates
  for (int node = 0; node < 10; node++) {
    int global_node_idx = d_data->element_connectivity()(elem_idx, node);
    X_elem[node][0]     = d_data->x12()(global_node_idx);  // x coordinate
    X_elem[node][1]     = d_data->y12()(global_node_idx);  // y coordinate
    X_elem[node][2]     = d_data->z12()(global_node_idx);  // z coordinate
  }

  // Compute Jacobian matrix J = Σ(X_node ⊗ dN_dxi)
  double J[3][3] = {{0.0}};  // Initialize to zero
  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {    // Node coordinate components
      for (int j = 0; j < 3; j++) {  // Natural coordinate derivatives
        J[i][j] += X_elem[a][i] * dN_dxi[a][j];  // Outer product
      }
    }
  }

  // Compute determinant of J (3x3 matrix)
  double detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

  // Store the determinant in d_detJ_ref
  d_data->detJ_ref(elem_idx, qp_idx) = detJ;

  // Compute J^T (transpose)
  double JT[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      JT[i][j] = J[j][i];
    }
  }

  // Solve JT * grad_N = dN_dxi for each shape function
  double grad_N[10][3];
  for (int a = 0; a < 10; a++) {
    // Solve 3×3 system: JT * grad_N[a] = dN_dxi[a]
    // You'll need a 3×3 linear solver here (LU decomposition, Gaussian
    // elimination, etc.)
    solve_3x3_system(JT, dN_dxi[a], grad_N[a]);
  }

  // Store the PHYSICAL gradients in grad_N_ref
  for (int i = 0; i < 10; i++) {
    d_data->grad_N_ref(elem_idx, qp_idx)(i, 0) = grad_N[i][0];  // ∂N_i/∂x
    d_data->grad_N_ref(elem_idx, qp_idx)(i, 1) = grad_N[i][1];  // ∂N_i/∂y
    d_data->grad_N_ref(elem_idx, qp_idx)(i, 2) = grad_N[i][2];  // ∂N_i/∂z
  }
}

__global__ void mass_matrix_qp_kernel(GPU_FEAT10_Data *d_data) {
  int n_qp_per_elem = Quadrature::N_QP_T10_5;  // 5 quadrature points
  int thread_global = blockIdx.x * blockDim.x + threadIdx.x;

  // Decode: which element and which (i, j) node pair?
  int elem       = thread_global / (10 * 10);  // 10 nodes per element
  int item_local = thread_global % (10 * 10);

  if (elem >= d_data->gpu_n_elem())
    return;

  // Decode item_local into (i_local, j_local) node indices
  int i_local = item_local / 10;  // Local node i (0-9)
  int j_local = item_local % 10;  // Local node j (0-9)

  // Get global node indices
  int i_global = d_data->element_connectivity()(elem, i_local);
  int j_global = d_data->element_connectivity()(elem, j_local);

  // Get material density
  double rho = d_data->rho0();

  // Accumulator for this (i, j) pair across all QPs
  double mass_contribution = 0.0;

  // Loop over all quadrature points
  for (int qp = 0; qp < n_qp_per_elem; qp++) {
    // Get quadrature point coordinates
    double xi   = d_data->tet5pt_x(qp);
    double eta  = d_data->tet5pt_y(qp);
    double zeta = d_data->tet5pt_z(qp);
    double wq   = d_data->tet5pt_weights(qp);
    // printf("xi: %f, eta: %f, zeta: %f, wq: %f\n", xi, eta, zeta, wq);

    // Compute barycentric coordinates
    double L1   = 1.0 - xi - eta - zeta;
    double L2   = xi;
    double L3   = eta;
    double L4   = zeta;
    double L[4] = {L1, L2, L3, L4};

    // Compute shape functions
    double N[10];

    // Corner nodes (0-3)
    for (int k = 0; k < 4; k++) {
      N[k] = L[k] * (2.0 * L[k] - 1.0);
    }

    // Edge nodes (4-9)
    int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
    for (int k = 0; k < 6; k++) {
      int ii   = edges[k][0];
      int jj   = edges[k][1];
      N[k + 4] = 4.0 * L[ii] * L[jj];
    }

    // Get determinant (pre-computed)
    double detJ = d_data->detJ_ref(elem, qp);

    // Accumulate: rho * N[i] * N[j] * detJ * wq
    mass_contribution += rho * N[i_local] * N[j_local] * detJ * wq;
  }

  const int row_start = d_data->csr_offsets()[i_global];
  const int row_end   = d_data->csr_offsets()[i_global + 1];
  const int n_cols    = row_end - row_start;
  const int local_idx = binary_search_column_csr_mass(
      d_data->csr_columns() + row_start, n_cols, j_global);
  if (local_idx >= 0) {
    atomicAdd(d_data->csr_values() + row_start + local_idx, mass_contribution);
  }
}

__global__ void calc_constraint_kernel(GPU_FEAT10_Data *d_data) {
  compute_constraint_data(d_data);
}

// HRZ (Hinton-Rock-Zienkiewicz) lumped mass kernel
// One thread per element - computes lumped mass for all 10 nodes of the element
// and atomicAdds to global mass vector
__global__ void compute_hrz_lumped_mass_kernel(GPU_FEAT10_Data *d_data,
                                                double *d_mass_lumped) {
  int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (elem_idx >= d_data->gpu_n_elem()) return;

  double rho = d_data->rho0();

  // Accumulators for HRZ algorithm
  double vol_elem = 0.0;
  double diag_consistent[10] = {0.0};

  // Loop over quadrature points
  for (int qp = 0; qp < Quadrature::N_QP_T10_5; qp++) {
    double xi   = d_data->tet5pt_x(qp);
    double eta  = d_data->tet5pt_y(qp);
    double zeta = d_data->tet5pt_z(qp);
    double wq   = d_data->tet5pt_weights(qp);
    double detJ = d_data->detJ_ref(elem_idx, qp);

    // Compute barycentric coordinates
    double L1 = 1.0 - xi - eta - zeta;
    double L2 = xi;
    double L3 = eta;
    double L4 = zeta;
    double L[4] = {L1, L2, L3, L4};

    // Compute shape functions for T10 element
    double N[10];

    // Corner nodes (0-3): N_i = L_i * (2*L_i - 1)
    for (int k = 0; k < 4; k++) {
      N[k] = L[k] * (2.0 * L[k] - 1.0);
    }

    // Edge nodes (4-9): N_k = 4 * L_i * L_j
    // Edge connectivity: [(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]
    int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
    for (int k = 0; k < 6; k++) {
      int ii = edges[k][0];
      int jj = edges[k][1];
      N[k + 4] = 4.0 * L[ii] * L[jj];
    }

    double dV = detJ * wq;
    vol_elem += dV;

    // Accumulate diagonal of consistent mass: ∫ N_i² dV
    for (int i = 0; i < 10; i++) {
      diag_consistent[i] += N[i] * N[i] * dV;
    }
  }

  // HRZ scaling: preserve total element mass
  double total_mass = rho * vol_elem;
  double sum_diag = 0.0;
  for (int i = 0; i < 10; i++) {
    sum_diag += diag_consistent[i];
  }

  // Avoid division by zero
  if (sum_diag < 1e-30) return;

  double scale = total_mass / sum_diag;

  // Assemble to global mass vector
  for (int i_local = 0; i_local < 10; i_local++) {
    int i_global = d_data->element_connectivity()(elem_idx, i_local);
    double m_lumped = diag_consistent[i_local] * scale;
    atomicAdd(&d_mass_lumped[i_global], m_lumped);
  }
}

void GPU_FEAT10_Data::CalcDnDuPre() {
  int total_threads = n_elem * Quadrature::N_QP_T10_5;

  int threads_per_block = 128;  // or another suitable block size
  int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  dn_du_pre_kernel<<<blocks, threads_per_block>>>(d_data);
  cudaDeviceSynchronize();
}

__global__ void calc_p_kernel(GPU_FEAT10_Data *d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx   = thread_idx / Quadrature::N_QP_T10_5;
  int qp_idx     = thread_idx % Quadrature::N_QP_T10_5;

  if (elem_idx >= d_data->gpu_n_elem() || qp_idx >= Quadrature::N_QP_T10_5)
    return;

  // No solver context for standalone CalcP: pass null v_guess (no viscous
  // contribution)
  compute_p(elem_idx, qp_idx, d_data, nullptr, 0.0);
}

void GPU_FEAT10_Data::CalcP() {
  int threads = 128;
  int blocks  = (n_elem * Quadrature::N_QP_T10_5 + threads - 1) / threads;
  calc_p_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}

__global__ void compute_internal_force_kernel(GPU_FEAT10_Data *d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx =
      thread_idx / Quadrature::N_NODE_T10_10;  // 10 nodes per element
  int node_local =
      thread_idx % Quadrature::N_NODE_T10_10;  // Local node index (0-9)

  if (elem_idx >= d_data->gpu_n_elem() ||
      node_local >= Quadrature::N_NODE_T10_10)
    return;

  compute_internal_force(elem_idx, node_local, d_data);
}

void GPU_FEAT10_Data::CalcInternalForce() {
  int threads = 128;
  int blocks  = (n_elem * Quadrature::N_NODE_T10_10 + threads - 1) / threads;
  compute_internal_force_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}

void GPU_FEAT10_Data::CalcConstraintData() {
  if (!is_constraints_setup) {
    std::cerr << "constraint is not set up" << std::endl;
    return;
  }
  if (n_constraint == 0) {
    return;
  }
  int total_threads     = n_constraint / 3;
  int threads_per_block = 128;
  int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  calc_constraint_kernel<<<blocks, threads_per_block>>>(d_data);
  cudaDeviceSynchronize();
}

void GPU_FEAT10_Data::CalcLumpedMassHRZ() {
  if (is_lumped_mass_computed) {
    // Already computed, just return
    return;
  }

  // Zero out the lumped mass vector
  HANDLE_ERROR(cudaMemset(d_mass_lumped, 0, n_coef * sizeof(double)));

  // Launch kernel: one thread per element
  int threads_per_block = 128;
  int blocks = (n_elem + threads_per_block - 1) / threads_per_block;

  compute_hrz_lumped_mass_kernel<<<blocks, threads_per_block>>>(d_data,
                                                                 d_mass_lumped);
  HANDLE_ERROR(cudaDeviceSynchronize());

  is_lumped_mass_computed = true;

  // Update device copy of struct
  HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data),
                          cudaMemcpyHostToDevice));
}

void GPU_FEAT10_Data::CalcMassMatrix() {
  if (!is_csr_setup) {
    BuildMassCSRPattern();
  }

  int h_nnz = 0;
  HANDLE_ERROR(cudaMemcpy(&h_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_nnz > 0) {
    HANDLE_ERROR(cudaMemset(d_csr_values, 0,
                            static_cast<size_t>(h_nnz) * sizeof(double)));
  }

  // Launch: n_elem × 10 × 10 threads
  int total_threads     = n_elem * 10 * 10;
  int threads_per_block = 128;
  int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  mass_matrix_qp_kernel<<<blocks, threads_per_block>>>(d_data);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void GPU_FEAT10_Data::BuildMassCSRPattern() {
  if (is_csr_setup) {
    return;
  }

  const int total_keys =
      n_elem * Quadrature::N_NODE_T10_10 * Quadrature::N_NODE_T10_10;
  unsigned long long *d_keys = nullptr;
  HANDLE_ERROR(cudaMalloc(
      &d_keys, static_cast<size_t>(total_keys) * sizeof(unsigned long long)));

  {
    constexpr int threads = 256;
    const int blocks      = (total_keys + threads - 1) / threads;
    build_mass_keys_feat10_kernel<<<blocks, threads>>>(d_data, d_keys);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  thrust::device_ptr<unsigned long long> keys_begin(d_keys);
  thrust::device_ptr<unsigned long long> keys_end = keys_begin + total_keys;
  thrust::sort(thrust::device, keys_begin, keys_end);
  thrust::device_ptr<unsigned long long> keys_unique_end =
      thrust::unique(thrust::device, keys_begin, keys_end);

  const int nnz = static_cast<int>(keys_unique_end - keys_begin);

  HANDLE_ERROR(cudaMalloc((void **)&d_csr_offsets,
                          static_cast<size_t>(n_coef + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&d_csr_columns,
                          static_cast<size_t>(nnz) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&d_csr_values,
                          static_cast<size_t>(nnz) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc((void **)&d_nnz, sizeof(int)));

  int *d_row_counts = nullptr;
  HANDLE_ERROR(
      cudaMalloc(&d_row_counts, static_cast<size_t>(n_coef) * sizeof(int)));
  HANDLE_ERROR(
      cudaMemset(d_row_counts, 0, static_cast<size_t>(n_coef) * sizeof(int)));

  {
    constexpr int threads = 256;
    const int blocks      = (nnz + threads - 1) / threads;
    decode_mass_keys_kernel<<<blocks, threads>>>(d_keys, nnz, d_csr_columns,
                                                 d_row_counts);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  thrust::device_ptr<int> row_counts_begin(d_row_counts);
  thrust::device_ptr<int> offsets_begin(d_csr_offsets);
  thrust::exclusive_scan(thrust::device, row_counts_begin,
                         row_counts_begin + n_coef, offsets_begin);

  {
    set_last_offset_kernel<<<1, 1>>>(d_csr_offsets, n_coef, nnz);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  HANDLE_ERROR(cudaMemcpy(d_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemset(d_csr_values, 0, static_cast<size_t>(nnz) * sizeof(double)));

  HANDLE_ERROR(cudaFree(d_row_counts));
  HANDLE_ERROR(cudaFree(d_keys));

  is_csr_setup = true;
  HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data),
                          cudaMemcpyHostToDevice));
}

namespace {
__global__ void build_constraint_j_csr_kernel(int n_constraint,
                                              const int *fixed_nodes,
                                              int *j_offsets, int *j_columns,
                                              double *j_values) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_constraint) {
    return;
  }

  j_offsets[tid] = tid;

  int fixed_idx  = tid / 3;
  int dof        = tid % 3;
  int node       = fixed_nodes[fixed_idx];
  j_columns[tid] = node * 3 + dof;
  j_values[tid]  = 1.0;
}

__global__ void build_constraint_jt_row_counts_kernel(int n_constraint,
                                                      const int *fixed_nodes,
                                                      int *row_counts) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_constraint) {
    return;
  }

  int fixed_idx = tid / 3;
  int dof       = tid % 3;
  int node      = fixed_nodes[fixed_idx];
  int row       = node * 3 + dof;
  atomicAdd(&row_counts[row], 1);
}

__global__ void build_constraint_jt_fill_kernel(int n_constraint,
                                                const int *fixed_nodes,
                                                const int *offsets,
                                                int *row_positions,
                                                int *columns, double *values) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_constraint) {
    return;
  }

  int fixed_idx = tid / 3;
  int dof       = tid % 3;
  int node      = fixed_nodes[fixed_idx];
  int row       = node * 3 + dof;

  int pos = atomicAdd(&row_positions[row], 1);
  int out = offsets[row] + pos;

  columns[out] = tid;
  values[out]  = 1.0;
}
}  // namespace

// This function converts the Constraint Jacobian matrix J to CSR format
// (Rows = Constraints, Cols = DOFs)
void GPU_FEAT10_Data::ConvertToCSR_ConstraintJac() {
  if (is_j_csr_setup) {
    return;
  }
  if (!is_constraints_setup || n_constraint == 0) {
    return;
  }

  const int nnz = n_constraint;

  HANDLE_ERROR(cudaMalloc((void **)&d_j_csr_offsets,
                          static_cast<size_t>(n_constraint + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&d_j_csr_columns,
                          static_cast<size_t>(nnz) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&d_j_csr_values,
                          static_cast<size_t>(nnz) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc((void **)&d_j_nnz, sizeof(int)));

  {
    constexpr int threads = 256;
    const int blocks      = (n_constraint + threads - 1) / threads;
    build_constraint_j_csr_kernel<<<blocks, threads>>>(
        n_constraint, d_fixed_nodes, d_j_csr_offsets, d_j_csr_columns,
        d_j_csr_values);
    HANDLE_ERROR(cudaDeviceSynchronize());
    set_last_offset_kernel<<<1, 1>>>(d_j_csr_offsets, n_constraint, nnz);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  HANDLE_ERROR(cudaMemcpy(d_j_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
  is_j_csr_setup = true;
  HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data),
                          cudaMemcpyHostToDevice));
}

// This function converts the TRANSPOSE of the constraint Jacobian matrix to CSR
// format
void GPU_FEAT10_Data::ConvertToCSR_ConstraintJacT() {
  if (is_cj_csr_setup) {
    return;
  }
  if (!is_constraints_setup || n_constraint == 0) {
    return;
  }

  const int num_rows = n_coef * 3;
  const int nnz      = n_constraint;

  HANDLE_ERROR(cudaMalloc((void **)&d_cj_csr_offsets,
                          static_cast<size_t>(num_rows + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&d_cj_csr_columns,
                          static_cast<size_t>(nnz) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&d_cj_csr_values,
                          static_cast<size_t>(nnz) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc((void **)&d_cj_nnz, sizeof(int)));

  int *d_row_counts    = nullptr;
  int *d_row_positions = nullptr;
  HANDLE_ERROR(
      cudaMalloc(&d_row_counts, static_cast<size_t>(num_rows) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_row_positions,
                          static_cast<size_t>(num_rows) * sizeof(int)));
  HANDLE_ERROR(
      cudaMemset(d_row_counts, 0, static_cast<size_t>(num_rows) * sizeof(int)));

  {
    constexpr int threads = 256;
    const int blocks      = (n_constraint + threads - 1) / threads;
    build_constraint_jt_row_counts_kernel<<<blocks, threads>>>(
        n_constraint, d_fixed_nodes, d_row_counts);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  thrust::device_ptr<int> counts_begin(d_row_counts);
  thrust::device_ptr<int> offsets_begin(d_cj_csr_offsets);
  thrust::exclusive_scan(thrust::device, counts_begin, counts_begin + num_rows,
                         offsets_begin);

  set_last_offset_kernel<<<1, 1>>>(d_cj_csr_offsets, num_rows, nnz);
  HANDLE_ERROR(cudaDeviceSynchronize());

  HANDLE_ERROR(cudaMemset(d_row_positions, 0,
                          static_cast<size_t>(num_rows) * sizeof(int)));

  {
    constexpr int threads = 256;
    const int blocks      = (n_constraint + threads - 1) / threads;
    build_constraint_jt_fill_kernel<<<blocks, threads>>>(
        n_constraint, d_fixed_nodes, d_cj_csr_offsets, d_row_positions,
        d_cj_csr_columns, d_cj_csr_values);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  HANDLE_ERROR(cudaFree(d_row_counts));
  HANDLE_ERROR(cudaFree(d_row_positions));

  HANDLE_ERROR(cudaMemcpy(d_cj_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
  is_cj_csr_setup = true;
  HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data),
                          cudaMemcpyHostToDevice));
}

void GPU_FEAT10_Data::RetrieveDetJToCPU(
    std::vector<std::vector<double>> &detJ) {
  detJ.resize(n_elem);
  for (int elem_idx = 0; elem_idx < n_elem; elem_idx++) {
    detJ[elem_idx].resize(Quadrature::N_QP_T10_5);
    HANDLE_ERROR(cudaMemcpy(
        detJ[elem_idx].data(), d_detJ_ref + elem_idx * Quadrature::N_QP_T10_5,
        Quadrature::N_QP_T10_5 * sizeof(double), cudaMemcpyDeviceToHost));
  }
}

void GPU_FEAT10_Data::RetrieveDnDuPreToCPU(
    std::vector<std::vector<Eigen::MatrixXd>> &dn_du_pre) {
  // Resize to [n_elem][N_QP_T10_5]
  dn_du_pre.resize(n_elem);

  for (int elem_idx = 0; elem_idx < n_elem; elem_idx++) {
    dn_du_pre[elem_idx].resize(Quadrature::N_QP_T10_5);

    for (int qp_idx = 0; qp_idx < Quadrature::N_QP_T10_5; qp_idx++) {
      // Each QP matrix: 10 × 3 (10 shape functions × 3 derivatives)
      dn_du_pre[elem_idx][qp_idx].resize(10, 3);

      // Calculate offset for this specific element + QP
      int offset = (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 10 * 3;
      int size   = 10 * 3 * sizeof(double);

      HANDLE_ERROR(cudaMemcpy(dn_du_pre[elem_idx][qp_idx].data(),
                              d_grad_N_ref + offset, size,
                              cudaMemcpyDeviceToHost));
    }
  }
}

void GPU_FEAT10_Data::RetrieveMassCSRToCPU(std::vector<int> &offsets,
                                           std::vector<int> &columns,
                                           std::vector<double> &values) {
  offsets.assign(static_cast<size_t>(n_coef) + 1, 0);
  columns.clear();
  values.clear();

  if (!is_csr_setup) {
    return;
  }

  int h_nnz = 0;
  HANDLE_ERROR(cudaMemcpy(&h_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));

  columns.resize(static_cast<size_t>(h_nnz));
  values.resize(static_cast<size_t>(h_nnz));

  HANDLE_ERROR(cudaMemcpy(offsets.data(), d_csr_offsets,
                          static_cast<size_t>(n_coef + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(columns.data(), d_csr_columns,
                          static_cast<size_t>(h_nnz) * sizeof(int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(values.data(), d_csr_values,
                          static_cast<size_t>(h_nnz) * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void GPU_FEAT10_Data::RetrievePFromFToCPU(
    std::vector<std::vector<Eigen::MatrixXd>> &p_from_F) {
  // Resize to [n_elem][N_QP_T10_5]
  p_from_F.resize(n_elem);

  for (int elem_idx = 0; elem_idx < n_elem; elem_idx++) {
    p_from_F[elem_idx].resize(Quadrature::N_QP_T10_5);

    for (int qp_idx = 0; qp_idx < Quadrature::N_QP_T10_5; qp_idx++) {
      // Each P matrix: 3 × 3 (first Piola-Kirchhoff stress tensor)
      p_from_F[elem_idx][qp_idx].resize(3, 3);

      // Calculate offset for this specific element + QP
      // P is stored as [elem][qp](i,j) where i,j are 0,1,2
      int offset = (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 3 * 3;
      int size   = 3 * 3 * sizeof(double);

      HANDLE_ERROR(cudaMemcpy(p_from_F[elem_idx][qp_idx].data(), d_P + offset,
                              size, cudaMemcpyDeviceToHost));
    }
  }
}

void GPU_FEAT10_Data::RetrieveInternalForceToCPU(
    Eigen::VectorXd &internal_force) {
  // Resize to total DOFs (3 * number of nodes)
  int total_dofs = 3 * n_coef;
  internal_force.resize(total_dofs);

  // Copy from device to host
  HANDLE_ERROR(cudaMemcpy(internal_force.data(), d_f_int,
                          total_dofs * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_FEAT10_Data::RetrieveExternalForceToCPU(
    Eigen::VectorXd &external_force) {
  // Resize to total DOFs (3 * number of nodes)
  int total_dofs = 3 * n_coef;
  external_force.resize(total_dofs);

  // Copy from device to host
  HANDLE_ERROR(cudaMemcpy(external_force.data(), d_f_ext,
                          total_dofs * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_FEAT10_Data::RetrieveLumpedMassToCPU(Eigen::VectorXd &lumped_mass) {
  // Resize to number of nodes (one scalar per node)
  lumped_mass.resize(n_coef);

  // Copy from device to host
  HANDLE_ERROR(cudaMemcpy(lumped_mass.data(), d_mass_lumped,
                          n_coef * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_FEAT10_Data::RetrievePositionToCPU(Eigen::VectorXd &x12,
                                            Eigen::VectorXd &y12,
                                            Eigen::VectorXd &z12) {
  // Resize to total number of nodes
  int total_nodes = n_coef;
  x12.resize(total_nodes);
  y12.resize(total_nodes);
  z12.resize(total_nodes);

  // Copy from device to host
  HANDLE_ERROR(cudaMemcpy(x12.data(), d_h_x12, total_nodes * sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(y12.data(), d_h_y12, total_nodes * sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(z12.data(), d_h_z12, total_nodes * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void GPU_FEAT10_Data::SetNodalFixed(const Eigen::VectorXi &fixed_nodes) {
  if (is_constraints_setup) {
    std::cerr << "GPU_FEAT10_Data CONSTRAINT is already set up." << std::endl;
    return;
  }

  n_constraint = fixed_nodes.size() * 3;

  HANDLE_ERROR(cudaMalloc(&d_constraint, n_constraint * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_fixed_nodes, fixed_nodes.size() * sizeof(int)));

  HANDLE_ERROR(cudaMemset(d_constraint, 0, n_constraint * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_fixed_nodes, fixed_nodes.data(),
                          fixed_nodes.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

  is_constraints_setup = true;
  if (d_data) {
    HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data),
                            cudaMemcpyHostToDevice));
  }
}

void GPU_FEAT10_Data::UpdateNodalFixed(const Eigen::VectorXi &fixed_nodes) {
  int new_n_constraint = fixed_nodes.size() * 3;

  // If constraints not set up yet, just call SetNodalFixed
  if (!is_constraints_setup) {
    SetNodalFixed(fixed_nodes);
    return;
  }

  // If number of constraints changed, reallocate
  if (new_n_constraint != n_constraint) {
    // Free old buffers
    HANDLE_ERROR(cudaFree(d_constraint));
    HANDLE_ERROR(cudaFree(d_fixed_nodes));

    // Free old CSR buffers if they exist
    if (is_cj_csr_setup) {
      HANDLE_ERROR(cudaFree(d_cj_csr_offsets));
      HANDLE_ERROR(cudaFree(d_cj_csr_columns));
      HANDLE_ERROR(cudaFree(d_cj_csr_values));
      HANDLE_ERROR(cudaFree(d_cj_nnz));
      d_cj_csr_offsets = nullptr;
      d_cj_csr_columns = nullptr;
      d_cj_csr_values  = nullptr;
      d_cj_nnz         = nullptr;
      is_cj_csr_setup  = false;
    }

    if (is_j_csr_setup) {
      HANDLE_ERROR(cudaFree(d_j_csr_offsets));
      HANDLE_ERROR(cudaFree(d_j_csr_columns));
      HANDLE_ERROR(cudaFree(d_j_csr_values));
      HANDLE_ERROR(cudaFree(d_j_nnz));
      d_j_csr_offsets = nullptr;
      d_j_csr_columns = nullptr;
      d_j_csr_values  = nullptr;
      d_j_nnz         = nullptr;
      is_j_csr_setup  = false;
    }

    n_constraint = new_n_constraint;

    // Allocate new buffers
    HANDLE_ERROR(cudaMalloc(&d_constraint, n_constraint * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_fixed_nodes, fixed_nodes.size() * sizeof(int)));
  }

  // Clear and update constraint data
  HANDLE_ERROR(cudaMemset(d_constraint, 0, n_constraint * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_fixed_nodes, fixed_nodes.data(),
                          fixed_nodes.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

  // Invalidate Jacobian CSR caches: fixed nodes may have changed even if the
  // constraint count stayed the same.
  if (is_cj_csr_setup) {
    HANDLE_ERROR(cudaFree(d_cj_csr_offsets));
    HANDLE_ERROR(cudaFree(d_cj_csr_columns));
    HANDLE_ERROR(cudaFree(d_cj_csr_values));
    HANDLE_ERROR(cudaFree(d_cj_nnz));
    d_cj_csr_offsets = nullptr;
    d_cj_csr_columns = nullptr;
    d_cj_csr_values  = nullptr;
    d_cj_nnz         = nullptr;
    is_cj_csr_setup  = false;
  }

  if (is_j_csr_setup) {
    HANDLE_ERROR(cudaFree(d_j_csr_offsets));
    HANDLE_ERROR(cudaFree(d_j_csr_columns));
    HANDLE_ERROR(cudaFree(d_j_csr_values));
    HANDLE_ERROR(cudaFree(d_j_nnz));
    d_j_csr_offsets = nullptr;
    d_j_csr_columns = nullptr;
    d_j_csr_values  = nullptr;
    d_j_nnz         = nullptr;
    is_j_csr_setup  = false;
  }

  HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data),
                          cudaMemcpyHostToDevice));
}

void GPU_FEAT10_Data::RetrieveConnectivityToCPU(Eigen::MatrixXi &connectivity) {
  connectivity.resize(n_elem, Quadrature::N_NODE_T10_10);
  HANDLE_ERROR(cudaMemcpy(connectivity.data(), d_element_connectivity,
                          n_elem * Quadrature::N_NODE_T10_10 * sizeof(int),
                          cudaMemcpyDeviceToHost));
}

void GPU_FEAT10_Data::WriteOutputVTK(const std::string &filename) {
  Eigen::VectorXd x12, y12, z12;
  this->RetrievePositionToCPU(x12, y12, z12);

  // Retrieve connectivity
  Eigen::MatrixXi connectivity;
  this->RetrieveConnectivityToCPU(connectivity);

  std::ofstream out(filename);
  out << "# vtk DataFile Version 3.0\n";
  out << "T10 mesh output\n";
  out << "ASCII\n";
  out << "DATASET UNSTRUCTURED_GRID\n";

  // Write points
  out << "POINTS " << x12.size() << " float\n";
  for (int i = 0; i < x12.size(); ++i) {
    out << x12(i) << " " << y12(i) << " " << z12(i) << "\n";
  }

  // Write cells (elements)
  out << "CELLS " << connectivity.rows() << " " << connectivity.rows() * 11
      << "\n";
  for (int i = 0; i < connectivity.rows(); ++i) {
    out << "10 ";
    for (int j = 0; j < 10; ++j)
      out << connectivity(i, j) << " ";
    out << "\n";
  }

  // Write cell types (24 = VTK_QUADRATIC_TETRA)
  out << "CELL_TYPES " << connectivity.rows() << "\n";
  for (int i = 0; i < connectivity.rows(); ++i)
    out << "24\n";

  out.close();
}
