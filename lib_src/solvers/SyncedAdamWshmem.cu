#include <cooperative_groups.h>

#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "SyncedAdamWshmem.cuh"
namespace cg = cooperative_groups;

// ============================================================================
// Shared memory structure for T10 element computations
// Each block processes ELEMS_PER_BLOCK (12) elements with 128 threads
// 10 threads per element, 8 threads idle
// ============================================================================

// Shared memory layout per element:
// FLOAT arrays:
// - grad_N_ref: 10 nodes * 3 dims * 5 QPs = 150 floats
// - detJ_ref: 5 floats
// - qp_weights: 5 floats
// - P_qp: 5 QPs * 9 = 45 floats (first Piola-Kirchhoff stress at each QP)
// Total floats per element: 205
//
// DOUBLE arrays (need double precision for stress computation):
// - x_nodes: 10 * 3 = 30 doubles
// - v_nodes: 10 * 3 = 30 doubles
// Total doubles per element: 60

// Connectivity stored separately as ints: 10 per element
// Material constants: 4 doubles (shared across all elements)

constexpr int SHMEM_GRAD_N_SIZE  = N_NODES_T10 * 3 * N_QP_T10;  // 150
constexpr int SHMEM_DETJ_SIZE    = N_QP_T10;                     // 5
constexpr int SHMEM_WEIGHTS_SIZE = N_QP_T10;                     // 5
constexpr int SHMEM_P_SIZE       = N_QP_T10 * 9;                 // 45 (P matrices)

// Offsets within per-element FLOAT shared memory
constexpr int OFFSET_GRAD_N  = 0;
constexpr int OFFSET_DETJ    = OFFSET_GRAD_N + SHMEM_GRAD_N_SIZE;   // 150
constexpr int OFFSET_WEIGHTS = OFFSET_DETJ + SHMEM_DETJ_SIZE;       // 155
constexpr int OFFSET_P       = OFFSET_WEIGHTS + SHMEM_WEIGHTS_SIZE; // 160
constexpr int FLOATS_PER_ELEM = OFFSET_P + SHMEM_P_SIZE;            // 205

// Sizes for DOUBLE arrays (per element)
constexpr int SHMEM_XNODES_SIZE_D  = N_NODES_T10 * 3;              // 30
constexpr int SHMEM_VNODES_SIZE_D  = N_NODES_T10 * 3;              // 30
constexpr int DOUBLES_PER_ELEM = SHMEM_XNODES_SIZE_D + SHMEM_VNODES_SIZE_D;  // 60

// Offsets within per-element DOUBLE shared memory
constexpr int OFFSET_XNODES_D = 0;
constexpr int OFFSET_VNODES_D = SHMEM_XNODES_SIZE_D;  // 30

// Material constants size (use double for precision)
constexpr int SHMEM_MATERIAL_SIZE_D = 4;  // lambda, mu, eta_damp, lambda_damp

// Total static shared memory sizes
constexpr int TOTAL_SHMEM_FLOATS = ELEMS_PER_BLOCK * FLOATS_PER_ELEM;
constexpr int TOTAL_SHMEM_DOUBLES = ELEMS_PER_BLOCK * DOUBLES_PER_ELEM + SHMEM_MATERIAL_SIZE_D;
constexpr int TOTAL_SHMEM_INTS   = ELEMS_PER_BLOCK * N_NODES_T10;

// Total shared memory: 
// - Floats: 12 * 205 * 4 = 9,840 bytes
// - Doubles: (12 * 60 + 4) * 8 = 5,792 bytes  
// - Ints: 12 * 10 * 4 = 480 bytes
// - Total: ~16 KB per block (well within 48KB limit)

// ============================================================================
// Device functions for shared memory operations
// ============================================================================

// Load element data into shared memory (called by all 10 threads of an element)
__device__ __forceinline__ void load_element_to_shmem(
    int elem_idx, int local_node_idx, GPU_FEAT10_Data *d_data,
    const double *__restrict__ v_guess, float *shmem_elem_f, double *shmem_elem_d, int *shmem_conn) {
  
  // Load connectivity (each thread loads one node index)
  int global_node_idx        = d_data->element_connectivity()(elem_idx, local_node_idx);
  shmem_conn[local_node_idx] = global_node_idx;

  // Load nodal positions as DOUBLE (critical for precision in P computation)
  shmem_elem_d[OFFSET_XNODES_D + local_node_idx * 3 + 0] = d_data->x12()(global_node_idx);
  shmem_elem_d[OFFSET_XNODES_D + local_node_idx * 3 + 1] = d_data->y12()(global_node_idx);
  shmem_elem_d[OFFSET_XNODES_D + local_node_idx * 3 + 2] = d_data->z12()(global_node_idx);

  // Load velocities as DOUBLE if available
  if (v_guess != nullptr) {
    shmem_elem_d[OFFSET_VNODES_D + local_node_idx * 3 + 0] = v_guess[global_node_idx * 3 + 0];
    shmem_elem_d[OFFSET_VNODES_D + local_node_idx * 3 + 1] = v_guess[global_node_idx * 3 + 1];
    shmem_elem_d[OFFSET_VNODES_D + local_node_idx * 3 + 2] = v_guess[global_node_idx * 3 + 2];
  } else {
    shmem_elem_d[OFFSET_VNODES_D + local_node_idx * 3 + 0] = 0.0;
    shmem_elem_d[OFFSET_VNODES_D + local_node_idx * 3 + 1] = 0.0;
    shmem_elem_d[OFFSET_VNODES_D + local_node_idx * 3 + 2] = 0.0;
  }

  // Load grad_N_ref as FLOAT: each thread loads data for its node across all QPs
  for (int qp = 0; qp < N_QP_T10; qp++) {
    int shmem_idx = OFFSET_GRAD_N + qp * N_NODES_T10 * 3 + local_node_idx * 3;
    shmem_elem_f[shmem_idx + 0] = (float)d_data->grad_N_ref(elem_idx, qp)(local_node_idx, 0);
    shmem_elem_f[shmem_idx + 1] = (float)d_data->grad_N_ref(elem_idx, qp)(local_node_idx, 1);
    shmem_elem_f[shmem_idx + 2] = (float)d_data->grad_N_ref(elem_idx, qp)(local_node_idx, 2);
  }

  // Only first 5 threads load detJ and weights as FLOAT
  if (local_node_idx < N_QP_T10) {
    shmem_elem_f[OFFSET_DETJ + local_node_idx] = (float)d_data->detJ_ref(elem_idx, local_node_idx);
    shmem_elem_f[OFFSET_WEIGHTS + local_node_idx] = (float)d_data->tet5pt_weights(local_node_idx);
  }
}

// Compute P at one QP and store to shared memory (called by thread with local_node_idx < 5)
__device__ __forceinline__ void compute_P_at_qp_shmem(
    int qp_idx, float *shmem_elem_f, double *shmem_elem_d,
    double lambda, double mu, double eta_damp, double lambda_damp) {
  
  // Compute F and Fdot from shared memory
  // Positions/velocities are stored as DOUBLE, grad_N as FLOAT
  double F[3][3] = {{0.0}};
  double Fdot[3][3] = {{0.0}};

  for (int a = 0; a < N_NODES_T10; a++) {
    double x_a[3], v_a[3], grad_N_a[3];

    // Load position and velocity from DOUBLE shared memory
    x_a[0] = shmem_elem_d[OFFSET_XNODES_D + a * 3 + 0];
    x_a[1] = shmem_elem_d[OFFSET_XNODES_D + a * 3 + 1];
    x_a[2] = shmem_elem_d[OFFSET_XNODES_D + a * 3 + 2];

    v_a[0] = shmem_elem_d[OFFSET_VNODES_D + a * 3 + 0];
    v_a[1] = shmem_elem_d[OFFSET_VNODES_D + a * 3 + 1];
    v_a[2] = shmem_elem_d[OFFSET_VNODES_D + a * 3 + 2];

    // Load grad_N for this node at this QP (cast float to double)
    int grad_idx = OFFSET_GRAD_N + qp_idx * N_NODES_T10 * 3 + a * 3;
    grad_N_a[0]  = (double)shmem_elem_f[grad_idx + 0];
    grad_N_a[1]  = (double)shmem_elem_f[grad_idx + 1];
    grad_N_a[2]  = (double)shmem_elem_f[grad_idx + 2];

    // Accumulate F and Fdot
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        F[i][j] += x_a[i] * grad_N_a[j];
        Fdot[i][j] += v_a[i] * grad_N_a[j];
      }
    }
  }

  // Compute F^T * F
  double FtF[3][3] = {{0.0}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FtF[i][j] += F[k][i] * F[k][j];
      }
    }
  }
  double trFtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

  // Compute F * F^T
  double FFt[3][3] = {{0.0}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FFt[i][j] += F[i][k] * F[j][k];
      }
    }
  }

  // Compute F * F^T * F
  double FFtF[3][3] = {{0.0}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FFtF[i][j] += FFt[i][k] * F[k][j];
      }
    }
  }

  // Compute viscous contribution
  double Ft[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Ft[i][j] = F[j][i];
    }
  }

  double FdotT_F[3][3] = {{0.0}};
  double Ft_Fdot[3][3] = {{0.0}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FdotT_F[i][j] += Fdot[k][i] * F[k][j];
        Ft_Fdot[i][j] += Ft[i][k] * Fdot[k][j];
      }
    }
  }

  double Edot[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Edot[i][j] = 0.5 * (FdotT_F[i][j] + Ft_Fdot[i][j]);
    }
  }
  double trEdot = Edot[0][0] + Edot[1][1] + Edot[2][2];

  double S_vis[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      S_vis[i][j] = 2.0 * eta_damp * Edot[i][j] +
                    lambda_damp * trEdot * (i == j ? 1.0 : 0.0);
    }
  }

  double P_vis[3][3] = {{0.0}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        P_vis[i][j] += F[i][k] * S_vis[k][j];
      }
    }
  }

  // Compute elastic P = lambda * (0.5*tr(F^T*F) - 1.5) * F + mu * (F*F^T*F - F)
  double lambda_factor = lambda * (0.5 * trFtF - 1.5);

  // Store P to FLOAT shared memory (cast back to float)
  int P_offset = OFFSET_P + qp_idx * 9;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double P_ij = lambda_factor * F[i][j] + mu * (FFtF[i][j] - F[i][j]) + P_vis[i][j];
      shmem_elem_f[P_offset + i * 3 + j] = (float)P_ij;
    }
  }
}

// Compute internal force for one node using P from shared memory
__device__ __forceinline__ void compute_internal_force_from_P_shmem(
    int local_node_idx, const float *shmem_elem_f, double f_node[3]) {
  
  f_node[0] = 0.0;
  f_node[1] = 0.0;
  f_node[2] = 0.0;

  // Loop over all quadrature points
  for (int qp_idx = 0; qp_idx < N_QP_T10; qp_idx++) {
    // Read P from shared memory
    float P[3][3];
    int P_offset = OFFSET_P + qp_idx * 9;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        P[i][j] = shmem_elem_f[P_offset + i * 3 + j];
      }
    }

    // Get grad_N for this node at this QP
    float grad_N[3];
    int grad_idx = OFFSET_GRAD_N + qp_idx * N_NODES_T10 * 3 + local_node_idx * 3;
    grad_N[0]    = shmem_elem_f[grad_idx + 0];
    grad_N[1]    = shmem_elem_f[grad_idx + 1];
    grad_N[2]    = shmem_elem_f[grad_idx + 2];

    // Get determinant and weight
    float detJ = shmem_elem_f[OFFSET_DETJ + qp_idx];
    float wq   = shmem_elem_f[OFFSET_WEIGHTS + qp_idx];
    float dV   = detJ * wq;

    // Compute P @ grad_N
    float f_contribution[3];
    for (int i = 0; i < 3; i++) {
      f_contribution[i] = 0.0f;
      for (int j = 0; j < 3; j++) {
        f_contribution[i] += P[i][j] * grad_N[j];
      }
    }

    // Accumulate with implicit cast to double
    f_node[0] += (double)(f_contribution[0] * dV);
    f_node[1] += (double)(f_contribution[1] * dV);
    f_node[2] += (double)(f_contribution[2] * dV);
  }
}

// Store P to global memory for residual computation compatibility
__device__ __forceinline__ void store_P_to_global(
    int elem_idx, int qp_idx, const float *shmem_elem_f, GPU_FEAT10_Data *d_data) {
  
  int P_offset = OFFSET_P + qp_idx * 9;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      d_data->P(elem_idx, qp_idx)(i, j) = (double)shmem_elem_f[P_offset + i * 3 + j];
    }
  }
}

// ============================================================================
// Templated solver_grad_L for shmem solver
// ============================================================================
__device__ double solver_grad_L_shmem(int tid, GPU_FEAT10_Data *d_data,
                                       SyncedAdamWshmemSolver *d_solver) {
  double res = 0.0;

  const int node_i = tid / 3;
  const int dof_i  = tid % 3;

  const double inv_dt = 1.0 / d_solver->solver_time_step();
  const double dt     = d_solver->solver_time_step();

  // Cache pointers once
  const double *__restrict__ v_g    = d_solver->v_guess().data();
  const double *__restrict__ v_p    = d_solver->v_prev().data();
  const int *__restrict__ offsets   = d_data->csr_offsets();
  const int *__restrict__ columns   = d_data->csr_columns();
  const double *__restrict__ values = d_data->csr_values();

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
  res -= (-d_data->f_int()(tid));
  res -= d_data->f_ext()(tid);

  const int n_constraints = d_solver->gpu_n_constraints();

  if (n_constraints > 0) {
    const double rho = *d_solver->solver_rho();

    const double *__restrict__ lam = d_solver->lambda_guess().data();
    const double *__restrict__ con = d_data->constraint().data();

    const int *__restrict__ cjT_offsets   = d_data->cj_csr_offsets();
    const int *__restrict__ cjT_columns   = d_data->cj_csr_columns();
    const double *__restrict__ cjT_values = d_data->cj_csr_values();

    const int col_start = cjT_offsets[tid];
    const int col_end   = cjT_offsets[tid + 1];

    for (int idx = col_start; idx < col_end; idx++) {
      const int constraint_idx        = cjT_columns[idx];
      const double constraint_jac_val = cjT_values[idx];
      const double constraint_val     = con[constraint_idx];

      res += dt * constraint_jac_val *
             (lam[constraint_idx] + rho * constraint_val);
    }
  }

  return res;
}

// ============================================================================
// Device function: Load ONLY positions/velocities to shared memory (for step updates)
// Assumes connectivity and grad_N are already loaded
// ============================================================================
__device__ __forceinline__ void update_positions_in_shmem(
    int local_node_idx, GPU_FEAT10_Data *d_data,
    const double *__restrict__ v_guess, double *shmem_elem_d, const int *shmem_conn) {
  
  int global_node_idx = shmem_conn[local_node_idx];

  // Update nodal positions from global memory
  shmem_elem_d[OFFSET_XNODES_D + local_node_idx * 3 + 0] = d_data->x12()(global_node_idx);
  shmem_elem_d[OFFSET_XNODES_D + local_node_idx * 3 + 1] = d_data->y12()(global_node_idx);
  shmem_elem_d[OFFSET_XNODES_D + local_node_idx * 3 + 2] = d_data->z12()(global_node_idx);

  // Update velocities from global memory
  if (v_guess != nullptr) {
    shmem_elem_d[OFFSET_VNODES_D + local_node_idx * 3 + 0] = v_guess[global_node_idx * 3 + 0];
    shmem_elem_d[OFFSET_VNODES_D + local_node_idx * 3 + 1] = v_guess[global_node_idx * 3 + 1];
    shmem_elem_d[OFFSET_VNODES_D + local_node_idx * 3 + 2] = v_guess[global_node_idx * 3 + 2];
  } else {
    shmem_elem_d[OFFSET_VNODES_D + local_node_idx * 3 + 0] = 0.0;
    shmem_elem_d[OFFSET_VNODES_D + local_node_idx * 3 + 1] = 0.0;
    shmem_elem_d[OFFSET_VNODES_D + local_node_idx * 3 + 2] = 0.0;
  }
}

// ============================================================================
// Device function: Load reference configuration data (connectivity, grad_N, detJ, weights)
// This only needs to be called ONCE at the start of the multi-step kernel
// ============================================================================
__device__ __forceinline__ void load_reference_data_to_shmem(
    int elem_idx, int local_node_idx, GPU_FEAT10_Data *d_data,
    float *shmem_elem_f, int *shmem_conn) {
  
  // Load connectivity (each thread loads one node index)
  int global_node_idx = d_data->element_connectivity()(elem_idx, local_node_idx);
  shmem_conn[local_node_idx] = global_node_idx;

  // Load grad_N_ref as FLOAT: each thread loads data for its node across all QPs
  for (int qp = 0; qp < N_QP_T10; qp++) {
    int shmem_idx = OFFSET_GRAD_N + qp * N_NODES_T10 * 3 + local_node_idx * 3;
    shmem_elem_f[shmem_idx + 0] = (float)d_data->grad_N_ref(elem_idx, qp)(local_node_idx, 0);
    shmem_elem_f[shmem_idx + 1] = (float)d_data->grad_N_ref(elem_idx, qp)(local_node_idx, 1);
    shmem_elem_f[shmem_idx + 2] = (float)d_data->grad_N_ref(elem_idx, qp)(local_node_idx, 2);
  }

  // Only first 5 threads load detJ and weights as FLOAT
  if (local_node_idx < N_QP_T10) {
    shmem_elem_f[OFFSET_DETJ + local_node_idx] = (float)d_data->detJ_ref(elem_idx, local_node_idx);
    shmem_elem_f[OFFSET_WEIGHTS + local_node_idx] = (float)d_data->tet5pt_weights(local_node_idx);
  }
}

// ============================================================================
// Main MULTI-STEP kernel with STATIC shared memory optimization
// Performs num_steps simulation steps, loading reference data only once
// ============================================================================
__global__ void multi_step_adamw_shmem_kernel(GPU_FEAT10_Data *d_data,
                                              SyncedAdamWshmemSolver *d_adamw_solver) {
  cg::grid_group grid = cg::this_grid();
  int tid             = blockIdx.x * blockDim.x + threadIdx.x;

  // Static shared memory allocation
  __shared__ float shmem_floats[TOTAL_SHMEM_FLOATS];
  __shared__ double shmem_doubles[TOTAL_SHMEM_DOUBLES];
  __shared__ int shmem_ints[TOTAL_SHMEM_INTS];

  // Material constants pointer (at end of double array for full precision)
  double *shmem_material = shmem_doubles + ELEMS_PER_BLOCK * DOUBLES_PER_ELEM;

  const int n_coef = d_adamw_solver->get_n_coef();
  const int n_elem = d_adamw_solver->get_n_beam();
  const int num_steps = d_adamw_solver->solver_num_steps();

  // ========================================================================
  // ONCE AT START: Load material constants to shared memory
  // ========================================================================
  if (threadIdx.x == 0) {
    shmem_material[0] = d_data->lambda();
    shmem_material[1] = d_data->mu();
    shmem_material[2] = d_data->eta_damp();
    shmem_material[3] = d_data->lambda_damp();
  }
  __syncthreads();

  // Get material constants from shared memory (valid for all steps)
  double lambda      = shmem_material[0];
  double mu          = shmem_material[1];
  double eta_damp    = shmem_material[2];
  double lambda_damp = shmem_material[3];

  // Thread mapping within block (valid for all steps)
  int local_elem_idx = threadIdx.x / THREADS_PER_ELEM;  // 0-11
  int local_node_idx = threadIdx.x % THREADS_PER_ELEM;  // 0-9
  bool is_active_thread = (local_elem_idx < ELEMS_PER_BLOCK);

  // Number of element chunks
  int total_elem_chunks = (n_elem + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;

  // ========================================================================
  // MULTI-STEP LOOP: Run num_steps simulation steps
  // ========================================================================
  for (int step = 0; step < num_steps; step++) {
    
    if (tid == 0) {
      printf("\n===== Simulation Step %d / %d =====\n", step + 1, num_steps);
    }
    grid.sync();

    // Save previous positions for this step
    if (tid < n_coef) {
      d_adamw_solver->x12_prev()(tid) = d_data->x12()(tid);
      d_adamw_solver->y12_prev()(tid) = d_data->y12()(tid);
      d_adamw_solver->z12_prev()(tid) = d_data->z12()(tid);
    }
    grid.sync();

    // Reset flags for this step
    if (tid == 0) {
      *d_adamw_solver->inner_flag() = 0;
      *d_adamw_solver->outer_flag() = 0;
    }
    grid.sync();

    // ======================================================================
    // OUTER ITERATION LOOP (Augmented Lagrangian)
    // ======================================================================
    for (int outer_iter = 0; outer_iter < d_adamw_solver->solver_max_outer();
         outer_iter++) {
      if (*d_adamw_solver->outer_flag() == 0) {
        // Initialize per-thread AdamW state
        double t   = 1.0;
        double m_t = 0.0;
        double v_t = 0.0;

        double lr           = d_adamw_solver->solver_lr();
        double beta1        = d_adamw_solver->solver_beta1();
        double beta2        = d_adamw_solver->solver_beta2();
        double eps          = d_adamw_solver->solver_eps();
        double weight_decay = d_adamw_solver->solver_weight_decay();
        double lr_decay     = d_adamw_solver->solver_lr_decay();
        int conv_check_interval =
            d_adamw_solver->solver_convergence_check_interval();

        if (tid == 0) {
          *d_adamw_solver->norm_g() = 0.0;
        }
        grid.sync();

        if (tid < n_coef * 3) {
          d_adamw_solver->g()(tid) = 0.0;
          t = 1.0;
        }

        // ====================================================================
        // INNER ITERATION LOOP (AdamW optimization)
        // ====================================================================
        for (int inner_iter = 0; inner_iter < d_adamw_solver->solver_max_inner();
             inner_iter++) {
          grid.sync();

          if (*d_adamw_solver->inner_flag() == 0) {
            if (tid == 0 && inner_iter % conv_check_interval == 0) {
              printf("outer iter: %d, inner iter: %d\n", outer_iter, inner_iter);
            }

            // Step 1: AdamW velocity update
            double y = 0.0;
            if (tid < n_coef * 3) {
              double g_tid       = d_adamw_solver->g()(tid);
              double v_guess_tid = d_adamw_solver->v_guess()(tid);
              lr                 = lr * lr_decay;
              t += 1;

              m_t          = beta1 * m_t + (1 - beta1) * g_tid;
              v_t          = beta2 * v_t + (1 - beta2) * g_tid * g_tid;
              double m_hat = m_t / (1 - pow(beta1, t));
              double v_hat = v_t / (1 - pow(beta2, t));
              y            = v_guess_tid -
                  lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * v_guess_tid);

              d_adamw_solver->v_guess()(tid) = y;
            }
            grid.sync();

            // Step 2: Update global positions
            if (tid < n_coef) {
              d_data->x12()(tid) = d_adamw_solver->x12_prev()(tid) +
                                   d_adamw_solver->solver_time_step() *
                                       d_adamw_solver->v_guess()(tid * 3 + 0);
              d_data->y12()(tid) = d_adamw_solver->y12_prev()(tid) +
                                   d_adamw_solver->solver_time_step() *
                                       d_adamw_solver->v_guess()(tid * 3 + 1);
              d_data->z12()(tid) = d_adamw_solver->z12_prev()(tid) +
                                   d_adamw_solver->solver_time_step() *
                                       d_adamw_solver->v_guess()(tid * 3 + 2);
            }
            grid.sync();

            // ==============================================================
            // SHARED MEMORY: Compute P and internal forces
            // Load all element data (including reference data) for each chunk
            // ==============================================================

            // Clear internal force
            if (tid < n_coef * 3) {
              d_data->f_int()[tid] = 0.0;
            }
            grid.sync();

            // Start timing for shmem section
            unsigned long long shmem_start = clock64();
            unsigned long long load_cycles = 0, P_cycles = 0, fint_cycles = 0;
            int chunks_processed = 0;

            // Process all elements using shared memory
            for (int chunk = blockIdx.x; chunk < total_elem_chunks;
                 chunk += gridDim.x) {
              int base_elem = chunk * ELEMS_PER_BLOCK;
              int elem_idx  = base_elem + local_elem_idx;
              bool is_valid_elem = is_active_thread && (elem_idx < n_elem);

              // Pointers to this element's shared memory
              float *shmem_elem_f  = shmem_floats + local_elem_idx * FLOATS_PER_ELEM;
              double *shmem_elem_d = shmem_doubles + local_elem_idx * DOUBLES_PER_ELEM;
              int *shmem_conn      = shmem_ints + local_elem_idx * N_NODES_T10;

              // Phase 1: Load ALL element data to shared memory
              unsigned long long t0 = clock64();
              if (is_valid_elem) {
                load_element_to_shmem(elem_idx, local_node_idx, d_data,
                                      d_adamw_solver->v_guess().data(),
                                      shmem_elem_f, shmem_elem_d, shmem_conn);
              }
              __syncthreads();
              unsigned long long t1 = clock64();

              // Phase 2: Compute P at each QP using shmem (threads 0-4)
              if (is_valid_elem && local_node_idx < N_QP_T10) {
                compute_P_at_qp_shmem(local_node_idx, shmem_elem_f, shmem_elem_d,
                                      lambda, mu, eta_damp, lambda_damp);
              }
              __syncthreads();
              unsigned long long t2 = clock64();

              // Phase 3: Compute internal force using shmem (all 10 threads)
              if (is_valid_elem) {
                double f_node[3];
                compute_internal_force_from_P_shmem(local_node_idx, shmem_elem_f, f_node);

                // Atomic add to global f_int
                int global_node_idx = shmem_conn[local_node_idx];
                atomicAdd(&(d_data->f_int()[global_node_idx * 3 + 0]), f_node[0]);
                atomicAdd(&(d_data->f_int()[global_node_idx * 3 + 1]), f_node[1]);
                atomicAdd(&(d_data->f_int()[global_node_idx * 3 + 2]), f_node[2]);
              }
              __syncthreads();
              unsigned long long t3 = clock64();

              if (threadIdx.x == 0) {
                load_cycles += (t1 - t0);
                P_cycles += (t2 - t1);
                fint_cycles += (t3 - t2);
                chunks_processed++;
              }
            }

            // End timing for shmem section
            unsigned long long shmem_end = clock64();
            unsigned long long shmem_cycles = shmem_end - shmem_start;

            if (tid == 0 && inner_iter % conv_check_interval == 0) {
              printf("Shmem P+fint: %llu cycles (%d chunks), load=%llu, P=%llu, fint=%llu\n", 
                     shmem_cycles, chunks_processed, load_cycles, P_cycles, fint_cycles);
            }
            grid.sync();

            // Compute constraints
            if (tid < d_adamw_solver->gpu_n_constraints() / 3) {
              compute_constraint_data(d_data);
            }
            grid.sync();

            // Compute gradient
            if (tid < n_coef * 3) {
              double g = solver_grad_L_shmem(tid, d_data, d_adamw_solver);
              d_adamw_solver->g()[tid] = g;
            }
            grid.sync();

            // Check convergence
            if (tid == 0 && inner_iter % conv_check_interval == 0) {
              double norm_g = 0.0;
              for (int i = 0; i < 3 * n_coef; i++) {
                norm_g += d_adamw_solver->g()(i) * d_adamw_solver->g()(i);
              }
              *d_adamw_solver->norm_g() = sqrt(norm_g);

              double norm_v_curr = 0.0;
              for (int i = 0; i < 3 * n_coef; i++) {
                norm_v_curr +=
                    d_adamw_solver->v_guess()(i) * d_adamw_solver->v_guess()(i);
              }
              norm_v_curr = sqrt(norm_v_curr);

              printf("norm_g: %.17f, norm_v_curr: %.17f\n",
                     *d_adamw_solver->norm_g(), norm_v_curr);

              if (*d_adamw_solver->norm_g() <=
                  d_adamw_solver->solver_inner_tol() * (1.0 + norm_v_curr)) {
                printf("Converged: gnorm=%.17f <= tol*(1+||v||)=%.17f\n",
                       *d_adamw_solver->norm_g(),
                       d_adamw_solver->solver_inner_tol() * (1.0 + norm_v_curr));
                *d_adamw_solver->inner_flag() = 1;
              }
            }
            grid.sync();
          }
        }  // End inner iteration loop

        // Update v_prev
        if (tid < n_coef * 3) {
          d_adamw_solver->v_prev()[tid] = d_adamw_solver->v_guess()[tid];
        }
        grid.sync();

        // Update positions
        if (tid < n_coef) {
          d_data->x12()(tid) = d_adamw_solver->x12_prev()(tid) +
                               d_adamw_solver->v_guess()(tid * 3 + 0) *
                                   d_adamw_solver->solver_time_step();
          d_data->y12()(tid) = d_adamw_solver->y12_prev()(tid) +
                               d_adamw_solver->v_guess()(tid * 3 + 1) *
                                   d_adamw_solver->solver_time_step();
          d_data->z12()(tid) = d_adamw_solver->z12_prev()(tid) +
                               d_adamw_solver->v_guess()(tid * 3 + 2) *
                                   d_adamw_solver->solver_time_step();
        }
        grid.sync();

        // Compute constraints at new position
        if (tid < d_adamw_solver->gpu_n_constraints() / 3) {
          compute_constraint_data(d_data);
        }
        grid.sync();

        // Dual variable update
        int n_constraints = d_adamw_solver->gpu_n_constraints();
        for (int i = tid; i < n_constraints; i += grid.size()) {
          double constraint_val = d_data->constraint()[i];
          d_adamw_solver->lambda_guess()[i] +=
              *d_adamw_solver->solver_rho() * d_adamw_solver->solver_time_step() *
              constraint_val;
        }
        grid.sync();

        if (tid == 0) {
          double norm_constraint = 0.0;
          for (int i = 0; i < d_adamw_solver->gpu_n_constraints(); i++) {
            double constraint_val = d_data->constraint()[i];
            norm_constraint += constraint_val * constraint_val;
          }
          norm_constraint = sqrt(norm_constraint);
          printf("norm_constraint: %.17f\n", norm_constraint);

          if (norm_constraint < d_adamw_solver->solver_outer_tol()) {
            printf("Converged constraint: %.17f\n", norm_constraint);
            *d_adamw_solver->outer_flag() = 1;
          }
        }
        grid.sync();
      }
    }  // End outer iteration loop

    // Final position update for this step
    if (tid < n_coef) {
      d_data->x12()(tid) = d_adamw_solver->x12_prev()(tid) +
                           d_adamw_solver->v_guess()(tid * 3 + 0) *
                               d_adamw_solver->solver_time_step();
      d_data->y12()(tid) = d_adamw_solver->y12_prev()(tid) +
                           d_adamw_solver->v_guess()(tid * 3 + 1) *
                               d_adamw_solver->solver_time_step();
      d_data->z12()(tid) = d_adamw_solver->z12_prev()(tid) +
                           d_adamw_solver->v_guess()(tid * 3 + 2) *
                               d_adamw_solver->solver_time_step();
    }
    grid.sync();

    // Reset inner_flag for next step
    if (tid == 0) {
      *d_adamw_solver->inner_flag() = 0;
    }
    grid.sync();

  }  // End multi-step loop
}

// ============================================================================
// Host function to launch the multi-step kernel
// ============================================================================
void SyncedAdamWshmemSolver::MultiStepAdamWshmem(int num_steps) {
  // Copy num_steps to device
  cudaMemcpy(d_num_steps_, &num_steps, sizeof(int), cudaMemcpyHostToDevice);

  // Update device solver object with new num_steps pointer
  HANDLE_ERROR(cudaMemcpy(d_adamw_solver_, this,
                          sizeof(SyncedAdamWshmemSolver),
                          cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  constexpr int threads = THREADS_PER_BLOCK_SHMEM;  // 128

  cudaDeviceProp props;
  HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

  // Static shared memory - no dynamic allocation needed
  size_t shmem_size = 0;  // Using static shared memory

  std::cout << "Multi-step kernel with " << num_steps << " steps" << std::endl;
  std::cout << "Static shared memory: " 
            << (TOTAL_SHMEM_FLOATS * sizeof(float) + TOTAL_SHMEM_DOUBLES * sizeof(double) + TOTAL_SHMEM_INTS * sizeof(int)) 
            << " bytes" << std::endl;

  // Calculate number of blocks needed
  int N            = 3 * n_coef_;
  int blocksNeeded = (N + threads - 1) / threads;

  int maxBlocksPerSm = 0;
  HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocksPerSm, multi_step_adamw_shmem_kernel, threads, shmem_size));
  int blocks = std::min(blocksNeeded, maxBlocksPerSm * props.multiProcessorCount);

  std::cout << "Launching " << blocks << " blocks with " << threads
            << " threads each" << std::endl;

  HANDLE_ERROR(cudaEventRecord(start));

  // Launch cooperative kernel (no dynamic shared memory)
  void *args[] = {&d_data_, &d_adamw_solver_};
  HANDLE_ERROR(cudaLaunchCooperativeKernel(
      (void *)multi_step_adamw_shmem_kernel, blocks, threads, args, shmem_size));

  HANDLE_ERROR(cudaEventRecord(stop));
  HANDLE_ERROR(cudaDeviceSynchronize());

  float milliseconds = 0;
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

  std::cout << "MultiStepAdamWshmem kernel time for " << num_steps << " steps: " 
            << milliseconds << " ms" << std::endl;
  std::cout << "Average per step: " << milliseconds / num_steps << " ms" << std::endl;

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
}
