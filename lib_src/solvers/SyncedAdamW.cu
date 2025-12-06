#include <cooperative_groups.h>
#include <type_traits>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataFunc.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataFunc.cuh"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "SyncedAdamW.cuh"
namespace cg = cooperative_groups;

// =============================================================================
// Shared memory grad accessor helpers
// =============================================================================

// For FEAT10: grad_N is per (elem, qp), size = 10 * 3 = 30 doubles per qp
//             Total: 5 QPs * 30 = 150 doubles (but varies per element, so we can't preload all)
// For ANCF3243: grad_s is per qp only, size = 8 * 3 = 24 doubles per qp (same for all elements)
//              Total: 12 QPs * 24 = 288 doubles
// For ANCF3443: grad_s is per qp only, size = 16 * 3 = 48 doubles per qp (same for all elements)
//              Total: 48 QPs * 48 = 2304 doubles

// Get total shared memory size for ALL QPs based on element type
// For FEAT10: each block needs grad_N for all threads in that block
//             Each thread needs 10 * 3 = 30 doubles, so 256 threads * 30 = 7680 doubles
// For ANCF3243: grad_s is per qp only, size = 8 * 3 = 24 doubles per qp (same for all elements)
//              Total: 12 QPs * 24 = 288 doubles
// For ANCF3443: grad_s is per qp only, size = 16 * 3 = 48 doubles per qp (same for all elements)
//              Total: 48 QPs * 48 = 2304 doubles

// Get per-thread shared memory size
template <typename ElementType>
__host__ __device__ __forceinline__ int get_per_thread_grad_size();

template <>
__host__ __device__ __forceinline__ int get_per_thread_grad_size<GPU_FEAT10_Data>() {
  return Quadrature::N_NODE_T10_10 * 3;  // 30 doubles per thread
}

template <>
__host__ __device__ __forceinline__ int get_per_thread_grad_size<GPU_ANCF3243_Data>() {
  return 0;  // Uses QP-based shared memory, not per-thread
}

template <>
__host__ __device__ __forceinline__ int get_per_thread_grad_size<GPU_ANCF3443_Data>() {
  return 0;  // Uses QP-based shared memory, not per-thread
}

// Get total shared memory size for ALL QPs (for ANCF types)
template <typename ElementType>
__host__ __device__ __forceinline__ int get_total_grad_smem_size();

template <>
__host__ __device__ __forceinline__ int get_total_grad_smem_size<GPU_FEAT10_Data>() {
  // FEAT10: Uses per-thread allocation, not this function
  return 0;
}

template <>
__host__ __device__ __forceinline__ int get_total_grad_smem_size<GPU_ANCF3243_Data>() {
  // 12 QPs * 8 shapes * 3 components = 288 doubles
  return Quadrature::N_TOTAL_QP_3_2_2 * Quadrature::N_SHAPE_3243 * 3;
}

template <>
__host__ __device__ __forceinline__ int get_total_grad_smem_size<GPU_ANCF3443_Data>() {
  // 48 QPs * 16 shapes * 3 components = 2304 doubles
  return Quadrature::N_TOTAL_QP_4_4_3 * Quadrature::N_SHAPE_3443 * 3;
}

// Get per-QP grad size
template <typename ElementType>
__device__ __forceinline__ int get_per_qp_grad_size();

template <>
__device__ __forceinline__ int get_per_qp_grad_size<GPU_FEAT10_Data>() {
  return Quadrature::N_NODE_T10_10 * 3;  // 30
}

template <>
__device__ __forceinline__ int get_per_qp_grad_size<GPU_ANCF3243_Data>() {
  return Quadrature::N_SHAPE_3243 * 3;  // 24
}

template <>
__device__ __forceinline__ int get_per_qp_grad_size<GPU_ANCF3443_Data>() {
  return Quadrature::N_SHAPE_3443 * 3;  // 48
}

// Load ALL grad_s data into shared memory at kernel start (for ANCF types)
// Using FLOAT to reduce shared memory usage
// For FEAT10: we load per-block, per-phase (see load_grad_for_block_feat10)
template <typename ElementType>
__device__ __forceinline__ void load_all_grad_to_smem(ElementType *d_data,
                                                       float *s_grad_f,
                                                       int total_threads,
                                                       int global_tid);

template <>
__device__ __forceinline__ void load_all_grad_to_smem<GPU_FEAT10_Data>(
    GPU_FEAT10_Data *d_data, float *s_grad_f, int total_threads, int global_tid) {
  // No-op at kernel start for FEAT10
  // We load per-block during compute_p and compute_internal_force phases
}

template <>
__device__ __forceinline__ void load_all_grad_to_smem<GPU_ANCF3243_Data>(
    GPU_ANCF3243_Data *d_data, float *s_grad_f, int total_threads, int global_tid) {
  // Load all 12 QPs * 24 values = 288 floats
  constexpr int total_size = Quadrature::N_TOTAL_QP_3_2_2 * Quadrature::N_SHAPE_3243 * 3;
  for (int i = global_tid; i < total_size; i += total_threads) {
    int qp_idx    = i / (Quadrature::N_SHAPE_3243 * 3);
    int remainder = i % (Quadrature::N_SHAPE_3243 * 3);
    int shape_idx = remainder / 3;
    int comp      = remainder % 3;
    s_grad_f[i]   = static_cast<float>(d_data->ds_du_pre(qp_idx)(shape_idx, comp));
  }
}

template <>
__device__ __forceinline__ void load_all_grad_to_smem<GPU_ANCF3443_Data>(
    GPU_ANCF3443_Data *d_data, float *s_grad_f, int total_threads, int global_tid) {
  // Load all 48 QPs * 48 values = 2304 floats
  constexpr int total_size = Quadrature::N_TOTAL_QP_4_4_3 * Quadrature::N_SHAPE_3443 * 3;
  for (int i = global_tid; i < total_size; i += total_threads) {
    int qp_idx    = i / (Quadrature::N_SHAPE_3443 * 3);
    int remainder = i % (Quadrature::N_SHAPE_3443 * 3);
    int shape_idx = remainder / 3;
    int comp      = remainder % 3;
    s_grad_f[i]   = static_cast<float>(d_data->ds_du_pre(qp_idx)(shape_idx, comp));
  }
}

// Load grad_N for all threads in this block (FEAT10 only)
// Each thread loads its own 30 FLOATS into s_grad_f[threadIdx.x * 30 ... threadIdx.x * 30 + 29]
// Using float to reduce shared memory usage: 128 threads * 30 floats * 4 bytes = 15360 bytes
// Called cooperatively by all threads in the block
__device__ __forceinline__ void load_grad_for_block_feat10(
    GPU_FEAT10_Data *d_data, float *s_grad_f, int block_start_tid,
    int n_qp, int total_work) {
  // Each thread in the block loads its own grad_N data
  int my_global_tid = block_start_tid + threadIdx.x;
  
  if (my_global_tid < total_work) {
    int elem_idx = my_global_tid / n_qp;
    int qp_idx   = my_global_tid % n_qp;
    
    // Load 30 floats for this thread's (elem, qp) into shared memory
    float* my_grad = s_grad_f + threadIdx.x * 30;
#pragma unroll
    for (int i = 0; i < 10; i++) {
      my_grad[i * 3 + 0] = static_cast<float>(d_data->grad_N_ref(elem_idx, qp_idx)(i, 0));
      my_grad[i * 3 + 1] = static_cast<float>(d_data->grad_N_ref(elem_idx, qp_idx)(i, 1));
      my_grad[i * 3 + 2] = static_cast<float>(d_data->grad_N_ref(elem_idx, qp_idx)(i, 2));
    }
  }
}

// Get pointer to grad data for a specific QP from shared memory (float)
template <typename ElementType>
__device__ __forceinline__ const float* get_grad_for_qp_f(int qp_idx, const float *s_grad_f);

// For FEAT10: use float shared memory
__device__ __forceinline__ const float* get_grad_for_qp_feat10(const float *s_grad_f) {
  // For FEAT10: each thread's grad_N is at s_grad_f[threadIdx.x * 30]
  return s_grad_f + threadIdx.x * 30;
}

template <>
__device__ __forceinline__ const float* get_grad_for_qp_f<GPU_FEAT10_Data>(int qp_idx, const float *s_grad_f) {
  // For FEAT10: each thread's grad_N is at s_grad_f[threadIdx.x * 30]
  return s_grad_f + threadIdx.x * 30;
}

template <>
__device__ __forceinline__ const float* get_grad_for_qp_f<GPU_ANCF3243_Data>(int qp_idx, const float *s_grad_f) {
  return s_grad_f + qp_idx * Quadrature::N_SHAPE_3243 * 3;
}

template <>
__device__ __forceinline__ const float* get_grad_for_qp_f<GPU_ANCF3443_Data>(int qp_idx, const float *s_grad_f) {
  return s_grad_f + qp_idx * Quadrature::N_SHAPE_3443 * 3;
}

// =============================================================================
// Modified compute_p functions that use shared memory grad
// For ANCF types: reads from shared memory pointer (all QPs loaded at kernel start)
// For FEAT10: reads from shared memory pointer (per-block loading)
// =============================================================================

// FEAT10 version using float shared memory for grad_N
__device__ __forceinline__ void compute_p_smem_feat10(int elem_idx, int qp_idx,
                                                GPU_FEAT10_Data *d_data,
                                                const double *__restrict__ v_guess,
                                                double dt,
                                                const float *s_grad_N_f) {
  // Get current nodal positions for this element
  double x_nodes[10][3];  // 10 nodes × 3 coordinates

#pragma unroll
  for (int node = 0; node < 10; node++) {
    int global_node_idx = d_data->element_connectivity()(elem_idx, node);
    x_nodes[node][0]    = d_data->x12()(global_node_idx);
    x_nodes[node][1]    = d_data->y12()(global_node_idx);
    x_nodes[node][2]    = d_data->z12()(global_node_idx);
  }

  // Initialize deformation gradient F to zero
  double F[3][3] = {{0.0}};

  // Use shared memory grad_N directly (float auto-promotes to double)
#pragma unroll
  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        F[i][j] += x_nodes[a][i] * s_grad_N_f[a * 3 + j];
      }
    }
  }

  // Compute Fdot = sum_a (v_nodes[a] ⊗ grad_N[a])
  double Fdot[3][3] = {{0.0}};
#pragma unroll
  for (int a = 0; a < 10; a++) {
    double v_a[3] = {0.0, 0.0, 0.0};
    if (v_guess != nullptr) {
      int global_node_idx = d_data->element_connectivity()(elem_idx, a);
      v_a[0] = v_guess[global_node_idx * 3 + 0];
      v_a[1] = v_guess[global_node_idx * 3 + 1];
      v_a[2] = v_guess[global_node_idx * 3 + 2];
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        Fdot[i][j] += v_a[i] * s_grad_N_f[a * 3 + j];
      }
    }
  }

  // Compute viscous Piola
  double Edot[3][3] = {{0.0}};
  double Ft[3][3];
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Ft[i][j] = F[j][i];
    }
  }
  double FdotT_F[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FdotT_F[i][j] += Fdot[k][i] * F[k][j];
      }
    }
  }
  double Ft_Fdot[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        Ft_Fdot[i][j] += Ft[i][k] * Fdot[k][j];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Edot[i][j] = 0.5 * (FdotT_F[i][j] + Ft_Fdot[i][j]);
    }
  }

  double trEdot = Edot[0][0] + Edot[1][1] + Edot[2][2];
  double eta = d_data->eta_damp();
  double lambda_d = d_data->lambda_damp();

  double S_vis[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      S_vis[i][j] = 2.0 * eta * Edot[i][j] + lambda_d * trEdot * (i == j ? 1.0 : 0.0);
    }
  }

  double P_vis[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        P_vis[i][j] += F[i][k] * S_vis[k][j];
      }
    }
  }

  // store Fdot and P_vis
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      d_data->Fdot(elem_idx, qp_idx)(i, j) = Fdot[i][j];
      d_data->P_vis(elem_idx, qp_idx)(i, j) = P_vis[i][j];
    }
  }

  // Compute F^T * F
  double FtF[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FtF[i][j] += F[k][i] * F[k][j];
      }
    }
  }

  double trFtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

  double FFt[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FFt[i][j] += F[i][k] * F[j][k];
      }
    }
  }

  double FFtF[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FFtF[i][j] += FFt[i][k] * F[k][j];
      }
    }
  }

  double lambda = d_data->lambda();
  double mu     = d_data->mu();
  double lambda_factor = lambda * (0.5 * trFtF - 1.5);

#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      d_data->P(elem_idx, qp_idx)(i, j) =
          lambda_factor * F[i][j] + mu * (FFtF[i][j] - F[i][j]) + P_vis[i][j];
    }
  }
}

__device__ __forceinline__ void compute_p_smem(int elem_idx, int qp_idx,
                                                GPU_ANCF3243_Data *d_data,
                                                const double *__restrict__ v_guess,
                                                double dt,
                                                const float *s_grad_s) {
  // Initialize F to zero
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      d_data->F(elem_idx, qp_idx)(i, j) = 0.0;
    }
  }

  double e[8][3];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int node_local  = (i < 4) ? 0 : 1;
    const int dof_local   = i % 4;
    const int node_global = d_data->element_node(elem_idx, node_local);
    const int coef_idx    = node_global * 4 + dof_local;
    e[i][0] = d_data->x12()(coef_idx);
    e[i][1] = d_data->y12()(coef_idx);
    e[i][2] = d_data->z12()(coef_idx);
  }

  // Use shared memory grad_s directly (float auto-promotes to double)
#pragma unroll
  for (int i = 0; i < 8; i++) {
#pragma unroll
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        d_data->F(elem_idx, qp_idx)(row, col) += e[i][row] * s_grad_s[i * 3 + col];
      }
    }
  }

  double FtF[3][3] = {0.0};
#pragma unroll
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        FtF[i][j] += d_data->F(elem_idx, qp_idx)(k, i) * d_data->F(elem_idx, qp_idx)(k, j);

  double tr_FtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

  double Ft[3][3] = {0};
#pragma unroll
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      Ft[i][j] = d_data->F(elem_idx, qp_idx)(j, i);

  double G[3][3] = {0};
#pragma unroll
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        G[i][j] += d_data->F(elem_idx, qp_idx)(i, k) * Ft[k][j];

  double FFF[3][3] = {0};
#pragma unroll
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        FFF[i][j] += G[i][k] * d_data->F(elem_idx, qp_idx)(k, j);

  double factor = d_data->lambda() * (0.5 * tr_FtF - 1.5);
#pragma unroll
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      d_data->P(elem_idx, qp_idx)(i, j) = factor * d_data->F(elem_idx, qp_idx)(i, j) + d_data->mu() * (FFF[i][j] - d_data->F(elem_idx, qp_idx)(i, j));

  // Compute Fdot using shared memory grad_s (float -> double)
  double Fdot[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 8; i++) {
    double v_i[3] = {0.0};
    const int node_local  = (i < 4) ? 0 : 1;
    const int dof_local   = i % 4;
    const int node_global = d_data->element_node(elem_idx, node_local);
    const int coef_idx    = node_global * 4 + dof_local;
    if (v_guess != nullptr) {
      v_i[0] = v_guess[coef_idx * 3 + 0];
      v_i[1] = v_guess[coef_idx * 3 + 1];
      v_i[2] = v_guess[coef_idx * 3 + 2];
    }
#pragma unroll
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        Fdot[row][col] += v_i[row] * s_grad_s[i * 3 + col];
      }
    }
  }

  // Edot and viscous stress
  double FdotT_F[3][3] = {{0.0}};
  double Ft_Fdot[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Ft[i][j] = d_data->F(elem_idx, qp_idx)(j, i);
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FdotT_F[i][j] += Fdot[k][i] * d_data->F(elem_idx, qp_idx)(k, j);
        Ft_Fdot[i][j] += Ft[i][k] * Fdot[k][j];
      }
    }
  }
  double Edot[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Edot[i][j] = 0.5 * (FdotT_F[i][j] + Ft_Fdot[i][j]);

  double trEdot = Edot[0][0] + Edot[1][1] + Edot[2][2];
  double eta = d_data->eta_damp();
  double lambda_d = d_data->lambda_damp();
  double S_vis[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      S_vis[i][j] = 2.0 * eta * Edot[i][j] + lambda_d * trEdot * (i == j ? 1.0 : 0.0);
    }
  }
  double P_vis[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        P_vis[i][j] += d_data->F(elem_idx, qp_idx)(i, k) * S_vis[k][j];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
    d_data->Fdot(elem_idx, qp_idx)(i, j) = Fdot[i][j];
    d_data->P_vis(elem_idx, qp_idx)(i, j) = P_vis[i][j];
  }
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      d_data->P(elem_idx, qp_idx)(i, j) += P_vis[i][j];
    }
  }
}

__device__ __forceinline__ void compute_p_smem(int elem_idx, int qp_idx,
                                                GPU_ANCF3443_Data *d_data,
                                                const double *__restrict__ v_guess,
                                                double dt,
                                                const float *s_grad_s) {
  // Initialize F to zero
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      d_data->F(elem_idx, qp_idx)(i, j) = 0.0;
    }
  }

  double x_local_arr[16], y_local_arr[16], z_local_arr[16];
  d_data->x12_elem(elem_idx, x_local_arr);
  d_data->y12_elem(elem_idx, y_local_arr);
  d_data->z12_elem(elem_idx, z_local_arr);

  double e[16][3];
#pragma unroll
  for (int i = 0; i < 16; i++) {
    e[i][0] = x_local_arr[i];
    e[i][1] = y_local_arr[i];
    e[i][2] = z_local_arr[i];
  }

  // Use shared memory grad_s directly (float auto-promotes to double)
#pragma unroll
  for (int i = 0; i < 16; i++) {
#pragma unroll
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        d_data->F(elem_idx, qp_idx)(row, col) += e[i][row] * s_grad_s[i * 3 + col];
      }
    }
  }

  // Precompute Ft
  double Ft[3][3];
#pragma unroll
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      Ft[i][j] = d_data->F(elem_idx, qp_idx)(j, i);

  // Compute Fdot using shared memory directly (float auto-promotes to double)
  double Fdot[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 16; i++) {
    double v_i[3] = {0.0, 0.0, 0.0};
    int node_local = i / 4;
    int dof_local  = i % 4;
    int node_global = d_data->element_connectivity()(elem_idx, node_local);
    int coef_idx = node_global * 4 + dof_local;
    if (v_guess != nullptr) {
      v_i[0] = v_guess[coef_idx * 3 + 0];
      v_i[1] = v_guess[coef_idx * 3 + 1];
      v_i[2] = v_guess[coef_idx * 3 + 2];
    }
#pragma unroll
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        Fdot[row][col] += v_i[row] * s_grad_s[i * 3 + col];
      }
    }
  }

  // Edot and viscous stress
  double FdotT_F[3][3] = {{0.0}};
  double Ft_Fdot[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Ft[i][j] = d_data->F(elem_idx, qp_idx)(j, i);
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FdotT_F[i][j] += Fdot[k][i] * d_data->F(elem_idx, qp_idx)(k, j);
        Ft_Fdot[i][j] += Ft[i][k] * Fdot[k][j];
      }
    }
  }
  double Edot[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Edot[i][j] = 0.5 * (FdotT_F[i][j] + Ft_Fdot[i][j]);

  double trEdot = Edot[0][0] + Edot[1][1] + Edot[2][2];
  double eta = d_data->eta_damp();
  double lambda_d = d_data->lambda_damp();
  double S_vis[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      S_vis[i][j] = 2.0 * eta * Edot[i][j] + lambda_d * trEdot * (i == j ? 1.0 : 0.0);
    }
  }
  double P_vis[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        P_vis[i][j] += d_data->F(elem_idx, qp_idx)(i, k) * S_vis[k][j];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
    d_data->Fdot(elem_idx, qp_idx)(i, j) = Fdot[i][j];
    d_data->P_vis(elem_idx, qp_idx)(i, j) = P_vis[i][j];
  }

  double FtF[3][3] = {0.0};
#pragma unroll
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        FtF[i][j] += d_data->F(elem_idx, qp_idx)(k, i) * d_data->F(elem_idx, qp_idx)(k, j);

  double tr_FtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

  double G[3][3] = {0};
#pragma unroll
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        G[i][j] += d_data->F(elem_idx, qp_idx)(i, k) * Ft[k][j];

  double FFF[3][3] = {0};
#pragma unroll
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        FFF[i][j] += G[i][k] * d_data->F(elem_idx, qp_idx)(k, j);

  double factor = d_data->lambda() * (0.5 * tr_FtF - 1.5);
#pragma unroll
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      d_data->P(elem_idx, qp_idx)(i, j) = factor * d_data->F(elem_idx, qp_idx)(i, j) + d_data->mu() * (FFF[i][j] - d_data->F(elem_idx, qp_idx)(i, j)) + P_vis[i][j];
}

// =============================================================================
// Modified compute_internal_force_per_qp functions that use shared memory grad
// =============================================================================

// FEAT10 version using float shared memory for grad_N
__device__ __forceinline__ void compute_internal_force_per_qp_smem_feat10(
    int elem_idx, int qp_idx, GPU_FEAT10_Data *d_data, const float *s_grad_N_f) {
  
  double P[3][3];
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      P[i][j] = d_data->P(elem_idx, qp_idx)(i, j);
    }
  }

  double dV = d_data->detJ_ref(elem_idx, qp_idx) * d_data->tet5pt_weights(qp_idx);

#pragma unroll
  for (int node_local = 0; node_local < 10; node_local++) {
    int global_node_idx = d_data->element_connectivity()(elem_idx, node_local);

#pragma unroll
    for (int i = 0; i < 3; i++) {
      double f_i = 0.0;
#pragma unroll
      for (int j = 0; j < 3; j++) {
        f_i += P[i][j] * s_grad_N_f[node_local * 3 + j];
      }
      f_i *= dV;
      int global_dof_idx = 3 * global_node_idx + i;
      atomicAdd(&(d_data->f_int()(global_dof_idx)), f_i);
    }
  }
}

__device__ __forceinline__ void compute_internal_force_per_qp_smem(
    int elem_idx, int qp_idx, GPU_ANCF3243_Data *d_data, const float *s_grad_s) {
  
  double P[3][3];
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      P[i][j] = d_data->P(elem_idx, qp_idx)(i, j);
    }
  }

  double geom = (d_data->L() * d_data->W() * d_data->H()) / 8.0;
  double scale = d_data->weight_xi()(qp_idx / (Quadrature::N_QP_2 * Quadrature::N_QP_2)) *
                 d_data->weight_eta()((qp_idx / Quadrature::N_QP_2) % Quadrature::N_QP_2) *
                 d_data->weight_zeta()(qp_idx % Quadrature::N_QP_2);
  double dV = scale * geom;

#pragma unroll
  for (int node_idx = 0; node_idx < 8; node_idx++) {
    const int node_local      = node_idx / 4;
    const int dof_local       = node_idx % 4;
    const int node_global     = d_data->element_node(elem_idx, node_local);
    const int coef_idx_global = node_global * 4 + dof_local;

#pragma unroll
    for (int r = 0; r < 3; ++r) {
      double f_r = 0.0;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        f_r += P[r][c] * s_grad_s[node_idx * 3 + c];
      }
      f_r *= dV;
      atomicAdd(&d_data->f_int(coef_idx_global)(r), f_r);
    }
  }
}

__device__ __forceinline__ void compute_internal_force_per_qp_smem(
    int elem_idx, int qp_idx, GPU_ANCF3443_Data *d_data, const float *s_grad_s) {
  
  double P[3][3];
#pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      P[i][j] = d_data->P(elem_idx, qp_idx)(i, j);
    }
  }

  double geom = (d_data->L() * d_data->W() * d_data->H()) / 8.0;
  double scale = d_data->weight_xi()(qp_idx / (Quadrature::N_QP_4 * Quadrature::N_QP_3)) *
                 d_data->weight_eta()((qp_idx / Quadrature::N_QP_3) % Quadrature::N_QP_4) *
                 d_data->weight_zeta()(qp_idx % Quadrature::N_QP_3);
  double dV = scale * geom;

#pragma unroll
  for (int node_idx = 0; node_idx < 16; node_idx++) {
    int global_node_idx =
        d_data->element_connectivity()(elem_idx, node_idx / 4) * 4 +
        (node_idx % 4);

#pragma unroll
    for (int r = 0; r < 3; ++r) {
      double f_r = 0.0;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        f_r += P[r][c] * s_grad_s[node_idx * 3 + c];
      }
      f_r *= dV;
      atomicAdd(&d_data->f_int(global_node_idx)(r), f_r);
    }
  }
}

// Templated solver_grad_L
template <typename ElementType>
__device__ double solver_grad_L(int tid, ElementType *data,
                                SyncedAdamWSolver *d_solver) {
  double res = 0.0;

  const int node_i = tid / 3;
  const int dof_i  = tid % 3;

  const int n_coef    = d_solver->get_n_coef();
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

    // If you have CSC format (or build it once):
    const int *__restrict__ cjT_offsets   = data->cj_csr_offsets();
    const int *__restrict__ cjT_columns   = data->cj_csr_columns();
    const double *__restrict__ cjT_values = data->cj_csr_values();

    // Get all constraints that affect this DOF (tid)
    const int col_start = cjT_offsets[tid];
    const int col_end   = cjT_offsets[tid + 1];

    for (int idx = col_start; idx < col_end; idx++) {
      const int constraint_idx = cjT_columns[idx];  // Which constraint
      const double constraint_jac_val =
          cjT_values[idx];  // J^T[tid, constraint_idx] = J[constraint_idx, tid]
      const double constraint_val = con[constraint_idx];

      // Add constraint contribution: h * J^T * (lambda + rho*c)
      res += dt * constraint_jac_val *
             (lam[constraint_idx] + rho * constraint_val);
    }
  }

  return res;
}

// Templated kernel
template <typename ElementType>
__global__ void one_step_adamw_kernel_impl(ElementType *d_data,
                                           SyncedAdamWSolver *d_adamw_solver) {
  // Dynamic shared memory for grad_N / grad_s (stored as float to save space)
  // For ANCF types: load ALL QP grad data once per block at kernel start
  // For FEAT10: loaded per-phase (before compute_p and compute_internal_force)
  extern __shared__ float s_grad[];
  
  cg::grid_group grid = cg::this_grid();
  int tid             = blockIdx.x * blockDim.x + threadIdx.x;

  // Load all grad data into shared memory at kernel start
  // This only needs to be done ONCE since grad never changes (reference configuration)
  // For ANCF types: load all QP grad_s data
  // For FEAT10: load grad_N for all threads in this block
  unsigned long long load_grad_start = 0ull;
  if (tid == 0) load_grad_start = clock64();
  
  if constexpr (std::is_same_v<ElementType, GPU_FEAT10_Data>) {
    const int n_qp = d_adamw_solver->gpu_n_total_qp();
    const int n_elem = d_adamw_solver->get_n_beam();
    const int total_work = n_elem * n_qp;
    int block_start_tid = blockIdx.x * blockDim.x;
    load_grad_for_block_feat10(d_data, s_grad, block_start_tid, n_qp, total_work);
  } else {
    load_all_grad_to_smem<ElementType>(d_data, s_grad, blockDim.x, threadIdx.x);
  }
  __syncthreads();
  
  if (tid == 0) {
    unsigned long long load_grad_end = clock64();
    unsigned long long load_grad_cycles = load_grad_end - load_grad_start;
    printf("load_grad_to_smem time (clock cycles): %llu\n", load_grad_cycles);
  }

  // Save previous positions
  if (tid < d_adamw_solver->get_n_coef()) {
    d_adamw_solver->x12_prev()(tid) = d_data->x12()(tid);
    d_adamw_solver->y12_prev()(tid) = d_data->y12()(tid);
    d_adamw_solver->z12_prev()(tid) = d_data->z12()(tid);
  }

  grid.sync();

  if (tid == 0) {
    *d_adamw_solver->inner_flag() = 0;
    *d_adamw_solver->outer_flag() = 0;
  }

  grid.sync();

  for (int outer_iter = 0; outer_iter < d_adamw_solver->solver_max_outer();
       outer_iter++) {
    if (*d_adamw_solver->outer_flag() == 0) {
      // Initialize per-thread variables
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

      if (tid < d_adamw_solver->get_n_coef() * 3) {
        d_adamw_solver->g()(tid) = 0.0;
        t                        = 1.0;
      }

      for (int inner_iter = 0; inner_iter < d_adamw_solver->solver_max_inner();
           inner_iter++) {
        grid.sync();

        if (*d_adamw_solver->inner_flag() == 0) {
          if (tid == 0 && inner_iter % conv_check_interval == 0) {
            printf("outer iter: %d, inner iter: %d\n", outer_iter, inner_iter);
          }

          // Step 1: Compute look-ahead velocity
          double y = 0.0;
          if (tid < d_adamw_solver->get_n_coef() * 3) {
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

          // Step 2: Update scratch positions
          if (tid < d_adamw_solver->get_n_coef()) {
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

          // Compute P (stress) with timing using clock64()
          unsigned long long compute_p_start = 0ull;
          if (tid == 0) compute_p_start = clock64();
          grid.sync();

          // Each thread handles one (elem, qp) pair - original parallel pattern
          // Grad data already loaded at kernel start - no reload needed
          {
            const int n_elem = d_adamw_solver->get_n_beam();
            const int n_qp = d_adamw_solver->gpu_n_total_qp();
            const int total_work = n_elem * n_qp;
            
            if (tid < total_work) {
              int elem_idx = tid / n_qp;
              int qp_idx   = tid % n_qp;
              
              if constexpr (std::is_same_v<ElementType, GPU_FEAT10_Data>) {
                const float* grad_ptr_f = get_grad_for_qp_feat10(s_grad);
                compute_p_smem_feat10(elem_idx, qp_idx, d_data,
                               d_adamw_solver->v_guess().data(),
                               d_adamw_solver->solver_time_step(),
                               grad_ptr_f);
              } else {
                const float* grad_ptr = get_grad_for_qp_f<ElementType>(qp_idx, s_grad);
                compute_p_smem(elem_idx, qp_idx, d_data,
                               d_adamw_solver->v_guess().data(),
                               d_adamw_solver->solver_time_step(),
                               grad_ptr);
              }
            }
          }

          grid.sync();

          // if (tid == 0) {
          //   unsigned long long compute_p_end = clock64();
          //   unsigned long long compute_p_cycles = compute_p_end - compute_p_start;
          //   printf("compute_p time (clock cycles): %llu\n", compute_p_cycles);
          // }

          // Clear internal force
          if (tid < d_adamw_solver->get_n_coef() * 3) {
            clear_internal_force(d_data);
          }

          grid.sync();

          // Compute internal force (timed) - per QP parallelization
          unsigned long long internal_force_start = 0ull;
          if (tid == 0) internal_force_start = clock64();
          grid.sync();

          // Each thread handles one (elem, qp) pair - original parallel pattern
          // Grad data already loaded at kernel start - no reload needed
          {
            const int n_elem = d_adamw_solver->get_n_beam();
            const int n_qp = d_adamw_solver->gpu_n_total_qp();
            const int total_work = n_elem * n_qp;
            
            if (tid < total_work) {
              int elem_idx = tid / n_qp;
              int qp_idx   = tid % n_qp;
              
              if constexpr (std::is_same_v<ElementType, GPU_FEAT10_Data>) {
                const float* grad_ptr_f = get_grad_for_qp_feat10(s_grad);
                compute_internal_force_per_qp_smem_feat10(elem_idx, qp_idx, d_data, grad_ptr_f);
              } else {
                const float* grad_ptr = get_grad_for_qp_f<ElementType>(qp_idx, s_grad);
                compute_internal_force_per_qp_smem(elem_idx, qp_idx, d_data, grad_ptr);
              }
            }
          }

          grid.sync();
          // if (tid == 0) {
          //   unsigned long long internal_force_end = clock64();
          //   unsigned long long internal_force_cycles =
          //       internal_force_end - internal_force_start;
          //   printf("compute_internal_force_per_qp time (clock cycles): %llu\n",
          //          internal_force_cycles);
          // }

          // Compute constraints
          if (tid < d_adamw_solver->gpu_n_constraints() / 3) {
            compute_constraint_data(d_data);
          }

          grid.sync();

          // Compute gradient with timing using clock64()
          unsigned long long grad_start = 0ull;
          if (tid == 0) grad_start = clock64();
          grid.sync();

          if (tid < d_adamw_solver->get_n_coef() * 3) {
            double g = solver_grad_L(tid, d_data, d_adamw_solver);
            d_adamw_solver->g()[tid] = g;
          }

          grid.sync();

          // if (tid == 0) {
          //   unsigned long long grad_end = clock64();
          //   unsigned long long grad_cycles = grad_end - grad_start;
          //   printf("solver_grad_L time (clock cycles): %llu\n", grad_cycles);
          // }

          // Check convergence
          if (tid == 0 && inner_iter % conv_check_interval == 0) {
            double norm_g = 0.0;
            for (int i = 0; i < 3 * d_adamw_solver->get_n_coef(); i++) {
              norm_g += d_adamw_solver->g()(i) * d_adamw_solver->g()(i);
            }
            *d_adamw_solver->norm_g() = sqrt(norm_g);

            double norm_v_curr = 0.0;
            for (int i = 0; i < 3 * d_adamw_solver->get_n_coef(); i++) {
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
      }

      // Update v_prev
      if (tid < d_adamw_solver->get_n_coef() * 3) {
        d_adamw_solver->v_prev()[tid] = d_adamw_solver->v_guess()[tid];
      }

      grid.sync();

      // Update positions
      if (tid < d_adamw_solver->get_n_coef()) {
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

      // dual variable update
      int n_constraints = d_adamw_solver->gpu_n_constraints();
      for (int i = tid; i < n_constraints; i += grid.size()) {
        double constraint_val = d_data->constraint()[i];
        d_adamw_solver->lambda_guess()[i] +=
            *d_adamw_solver->solver_rho() * d_adamw_solver->solver_time_step() *
            constraint_val;
      }
      grid.sync();

      if (tid == 0) {
        // check constraint convergence
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
  }

  // Final position update
  if (tid < d_adamw_solver->get_n_coef()) {
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
}

// Explicit instantiations
template __global__ void one_step_adamw_kernel_impl<GPU_ANCF3243_Data>(
    GPU_ANCF3243_Data *, SyncedAdamWSolver *);
template __global__ void one_step_adamw_kernel_impl<GPU_ANCF3443_Data>(
    GPU_ANCF3443_Data *, SyncedAdamWSolver *);
template __global__ void one_step_adamw_kernel_impl<GPU_FEAT10_Data>(
    GPU_FEAT10_Data *, SyncedAdamWSolver *);

void SyncedAdamWSolver::OneStepAdamW() {
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  cudaDeviceProp props;
  HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

  HANDLE_ERROR(cudaEventRecord(start));

  // Launch appropriate templated kernel based on element type
  if (type_ == TYPE_3243) {
    int threads = 256;
    int N            = 3 * n_coef_;
    int blocksNeeded = (N + threads - 1) / threads;
    
    // Shared memory size: ALL QPs * 8 shapes * 3 components = 12 * 24 = 288 floats
    size_t smem_size = Quadrature::N_TOTAL_QP_3_2_2 * Quadrature::N_SHAPE_3243 * 3 * sizeof(float);
    
    int maxBlocksPerSm = 0;
    HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSm, one_step_adamw_kernel_impl<GPU_ANCF3243_Data>, threads,
        smem_size));
    int blocks =
        std::min(blocksNeeded, maxBlocksPerSm * props.multiProcessorCount);

    void *args[] = {&d_data_, &d_adamw_solver_};
    HANDLE_ERROR(cudaLaunchCooperativeKernel(
        (void *)one_step_adamw_kernel_impl<GPU_ANCF3243_Data>, blocks, threads,
        args, smem_size));
  } else if (type_ == TYPE_3443) {
    int threads = 256;
    int N            = 3 * n_coef_;
    int blocksNeeded = (N + threads - 1) / threads;
    
    // Shared memory size: ALL QPs * 16 shapes * 3 components = 48 * 48 = 2304 floats
    size_t smem_size = Quadrature::N_TOTAL_QP_4_4_3 * Quadrature::N_SHAPE_3443 * 3 * sizeof(float);
    
    int maxBlocksPerSm = 0;
    HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSm, one_step_adamw_kernel_impl<GPU_ANCF3443_Data>, threads,
        smem_size));
    int blocks =
        std::min(blocksNeeded, maxBlocksPerSm * props.multiProcessorCount);

    void *args[] = {&d_data_, &d_adamw_solver_};
    HANDLE_ERROR(cudaLaunchCooperativeKernel(
        (void *)one_step_adamw_kernel_impl<GPU_ANCF3443_Data>, blocks, threads,
        args, smem_size));
  } else if (type_ == TYPE_T10) {
    // FEAT10: use 128 threads per block with float shared memory
    // 128 threads * 30 floats * 4 bytes = 15360 bytes
    int threads = 128;
    int N            = 3 * n_coef_;
    int blocksNeeded = (N + threads - 1) / threads;
    
    size_t smem_size = threads * Quadrature::N_NODE_T10_10 * 3 * sizeof(float);
    
    int maxBlocksPerSm = 0;
    HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSm, one_step_adamw_kernel_impl<GPU_FEAT10_Data>, threads,
        smem_size));
    int blocks =
        std::min(blocksNeeded, maxBlocksPerSm * props.multiProcessorCount);

    void *args[] = {&d_data_, &d_adamw_solver_};
    HANDLE_ERROR(cudaLaunchCooperativeKernel(
        (void *)one_step_adamw_kernel_impl<GPU_FEAT10_Data>, blocks, threads,
        args, smem_size));
  }

  HANDLE_ERROR(cudaEventRecord(stop));
  HANDLE_ERROR(cudaDeviceSynchronize());

  float milliseconds = 0;
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

  std::cout << "OneStepAdamW kernel time: " << milliseconds << " ms"
            << std::endl;

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
}