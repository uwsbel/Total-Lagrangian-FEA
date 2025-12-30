/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    SyncedVBD.cu
 * Brief:   Implements the GPU-synchronized VBD (Vertex Block Descent) solver.
 *          Contains kernels for parallel per-node block updates using graph
 *          coloring, with ALM outer loop for constraint handling.
 *==============================================================
 *==============================================================*/

#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <vector>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataFunc.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataFunc.cuh"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "SyncedVBD.cuh"

namespace cg = cooperative_groups;

// =====================================================
// Device Helper Functions
// =====================================================

// Solve 3x3 linear system using Cramer's rule (optimized for small systems)
__device__ __forceinline__ void solve_3x3_vbd(const double H[3][3],
                                               const double R[3],
                                               double dv[3]) {
  // Compute determinant of H
  double det = H[0][0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1]) -
               H[0][1] * (H[1][0] * H[2][2] - H[1][2] * H[2][0]) +
               H[0][2] * (H[1][0] * H[2][1] - H[1][1] * H[2][0]);

  if (fabs(det) < 1e-30) {
    // Singular matrix - return zero update
    dv[0] = dv[1] = dv[2] = 0.0;
    return;
  }

  double inv_det = 1.0 / det;

  // Compute inverse of H using cofactor matrix
  double H_inv[3][3];
  H_inv[0][0] = (H[1][1] * H[2][2] - H[1][2] * H[2][1]) * inv_det;
  H_inv[0][1] = (H[0][2] * H[2][1] - H[0][1] * H[2][2]) * inv_det;
  H_inv[0][2] = (H[0][1] * H[1][2] - H[0][2] * H[1][1]) * inv_det;
  H_inv[1][0] = (H[1][2] * H[2][0] - H[1][0] * H[2][2]) * inv_det;
  H_inv[1][1] = (H[0][0] * H[2][2] - H[0][2] * H[2][0]) * inv_det;
  H_inv[1][2] = (H[0][2] * H[1][0] - H[0][0] * H[1][2]) * inv_det;
  H_inv[2][0] = (H[1][0] * H[2][1] - H[1][1] * H[2][0]) * inv_det;
  H_inv[2][1] = (H[0][1] * H[2][0] - H[0][0] * H[2][1]) * inv_det;
  H_inv[2][2] = (H[0][0] * H[1][1] - H[0][1] * H[1][0]) * inv_det;

  // dv = -H_inv * R
  dv[0] = -(H_inv[0][0] * R[0] + H_inv[0][1] * R[1] + H_inv[0][2] * R[2]);
  dv[1] = -(H_inv[1][0] * R[0] + H_inv[1][1] * R[1] + H_inv[1][2] * R[2]);
  dv[2] = -(H_inv[2][0] * R[0] + H_inv[2][1] * R[1] + H_inv[2][2] * R[2]);
}

// =====================================================
// Residual/gradient computation (shared with AdamW/Newton formulation)
// =====================================================

template <typename ElementType>
__device__ __forceinline__ double solver_grad_L_vbd(int tid, ElementType *data,
                                                    SyncedVBDSolver *d_solver) {
  double res = 0.0;

  const int node_i = tid / 3;
  const int dof_i  = tid % 3;

  const double inv_dt = 1.0 / d_solver->solver_time_step();
  const double dt     = d_solver->solver_time_step();

  const double *__restrict__ v_g    = d_solver->v_guess().data();
  const double *__restrict__ v_p    = d_solver->v_prev().data();
  const int *__restrict__ offsets   = data->csr_offsets();
  const int *__restrict__ columns   = data->csr_columns();
  const double *__restrict__ values = data->csr_values();

  // Mass term: (M @ (v - v_prev)) / h
  const int row_start = offsets[node_i];
  const int row_end   = offsets[node_i + 1];
  for (int idx = row_start; idx < row_end; ++idx) {
    const int node_j     = columns[idx];
    const double mass_ij = values[idx];
    const int tid_j      = node_j * 3 + dof_i;
    const double v_diff  = v_g[tid_j] - v_p[tid_j];
    res += mass_ij * v_diff * inv_dt;
  }

  // Mechanical force: f_int - f_ext
  res -= (-data->f_int()(tid));
  res -= data->f_ext()(tid);

  const int n_constraints = d_solver->gpu_n_constraints();
  if (n_constraints > 0) {
    const double rho = *d_solver->solver_rho();

    const double *__restrict__ lam = d_solver->lambda_guess().data();
    const double *__restrict__ con = data->constraint().data();

    // J^T stored in CSR by DOF-column (same as Newton/AdamW).
    const int *__restrict__ cjT_offsets   = data->cj_csr_offsets();
    const int *__restrict__ cjT_columns   = data->cj_csr_columns();
    const double *__restrict__ cjT_values = data->cj_csr_values();

    const int col_start = cjT_offsets[tid];
    const int col_end   = cjT_offsets[tid + 1];
    for (int idx = col_start; idx < col_end; ++idx) {
      const int constraint_idx        = cjT_columns[idx];
      const double constraint_jac_val = cjT_values[idx];
      const double constraint_val     = con[constraint_idx];
      res += dt * constraint_jac_val *
             (lam[constraint_idx] + rho * constraint_val);
    }
  }

  return res;
}

// =====================================================
// VBD Update Kernel for FEAT10 Elements
// =====================================================

template <typename ElementType>
__global__ void vbd_build_fixed_map_kernel(ElementType *d_data,
                                          int *d_fixed_map) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int n_fixed = d_data->gpu_n_constraint() / 3;
  if (k < n_fixed) {
    int node_idx = d_data->fixed_nodes()[k];
    d_fixed_map[node_idx] = k;
  }
}

// Per-color VBD update with 1 block per node. Threads parallelize over the
// mass-row scan and the (incident element × quadrature point) contributions.
// Reduction and the 3×3 solve happen in shared memory.
//
// Uses a fixed block size so shared memory can be statically allocated.
template <int kBlockSize>
__global__ void vbd_update_color_block_kernel(GPU_FEAT10_Data *d_data,
                                              SyncedVBDSolver *d_solver,
                                              int color_start,
                                              int color_count) {
  static_assert(kBlockSize > 0, "kBlockSize must be > 0");
  static_assert((kBlockSize & (kBlockSize - 1)) == 0,
                "kBlockSize must be power-of-two for reduction");

  const int tid = threadIdx.x;
  if (tid >= kBlockSize) return;
  const int node_slot = blockIdx.x;
  if (node_slot >= color_count) return;

  const int node_i = d_solver->color_nodes()[color_start + node_slot];

  const double h        = d_solver->solver_time_step();
  const double inv_h    = 1.0 / h;
  const double omega    = d_solver->solver_omega();
  const double hess_eps = d_solver->solver_hess_eps();
  const double rho      = *d_solver->solver_rho();

  double r0 = 0.0, r1 = 0.0, r2 = 0.0;
  double h00 = 0.0, h01 = 0.0, h02 = 0.0;
  double h10 = 0.0, h11 = 0.0, h12 = 0.0;
  double h20 = 0.0, h21 = 0.0, h22 = 0.0;

  // Mass-row contribution: (M_row_i/h) * (v - v_prev)
  {
    const int *__restrict__ offsets = d_data->csr_offsets();
    const int *__restrict__ columns = d_data->csr_columns();
    const double *__restrict__ values = d_data->csr_values();
    const int row_start = offsets[node_i];
    const int row_end   = offsets[node_i + 1];
    for (int idx = row_start + tid; idx < row_end; idx += blockDim.x) {
      const int node_j  = columns[idx];
      const double m_ij = values[idx];
      const int dof_j   = node_j * 3;
      r0 += m_ij * (d_solver->v_guess()[dof_j + 0] -
                    d_solver->v_prev()[dof_j + 0]) *
            inv_h;
      r1 += m_ij * (d_solver->v_guess()[dof_j + 1] -
                    d_solver->v_prev()[dof_j + 1]) *
            inv_h;
      r2 += m_ij * (d_solver->v_guess()[dof_j + 2] -
                    d_solver->v_prev()[dof_j + 2]) *
            inv_h;
    }
  }

  // Element contributions for this node: cached F/P at (elem, qp).
  {
    const int inc_start = d_solver->incidence_offsets()[node_i];
    const int inc_end   = d_solver->incidence_offsets()[node_i + 1];
    const int n_inc     = inc_end - inc_start;

    const double lambda = d_data->lambda();
    const double mu     = d_data->mu();

    constexpr int n_qp = Quadrature::N_QP_T10_5;
    const int work_count = n_inc * n_qp;

    for (int w = tid; w < work_count; w += blockDim.x) {
      const int inc_idx = inc_start + (w / n_qp);
      const int qp_idx  = w - (w / n_qp) * n_qp;

      const int2 inc       = d_solver->incidence_data()[inc_idx];
      const int elem_idx   = inc.x;
      const int local_node = inc.y;

      const double ha0 = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 0);
      const double ha1 = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 1);
      const double ha2 = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 2);

      const double detJ = d_data->detJ_ref(elem_idx, qp_idx);
      const double wq   = d_data->tet5pt_weights(qp_idx);
      const double dV   = detJ * wq;

      // Internal force: (P @ h_a) * dV
      const double P00 = d_data->P(elem_idx, qp_idx)(0, 0);
      const double P01 = d_data->P(elem_idx, qp_idx)(0, 1);
      const double P02 = d_data->P(elem_idx, qp_idx)(0, 2);
      const double P10 = d_data->P(elem_idx, qp_idx)(1, 0);
      const double P11 = d_data->P(elem_idx, qp_idx)(1, 1);
      const double P12 = d_data->P(elem_idx, qp_idx)(1, 2);
      const double P20 = d_data->P(elem_idx, qp_idx)(2, 0);
      const double P21 = d_data->P(elem_idx, qp_idx)(2, 1);
      const double P22 = d_data->P(elem_idx, qp_idx)(2, 2);

      r0 += (P00 * ha0 + P01 * ha1 + P02 * ha2) * dV;
      r1 += (P10 * ha0 + P11 * ha1 + P12 * ha2) * dV;
      r2 += (P20 * ha0 + P21 * ha1 + P22 * ha2) * dV;

      // Diagonal tangent block K_ii contribution (SVK diagonal block).
      const double F00 = d_data->F(elem_idx, qp_idx)(0, 0);
      const double F01 = d_data->F(elem_idx, qp_idx)(0, 1);
      const double F02 = d_data->F(elem_idx, qp_idx)(0, 2);
      const double F10 = d_data->F(elem_idx, qp_idx)(1, 0);
      const double F11 = d_data->F(elem_idx, qp_idx)(1, 1);
      const double F12 = d_data->F(elem_idx, qp_idx)(1, 2);
      const double F20 = d_data->F(elem_idx, qp_idx)(2, 0);
      const double F21 = d_data->F(elem_idx, qp_idx)(2, 1);
      const double F22 = d_data->F(elem_idx, qp_idx)(2, 2);

      const double trFtF = F00 * F00 + F01 * F01 + F02 * F02 + F10 * F10 +
                           F11 * F11 + F12 * F12 + F20 * F20 + F21 * F21 +
                           F22 * F22;

      const double FFT00 = F00 * F00 + F01 * F01 + F02 * F02;
      const double FFT01 = F00 * F10 + F01 * F11 + F02 * F12;
      const double FFT02 = F00 * F20 + F01 * F21 + F02 * F22;
      const double FFT10 = FFT01;
      const double FFT11 = F10 * F10 + F11 * F11 + F12 * F12;
      const double FFT12 = F10 * F20 + F11 * F21 + F12 * F22;
      const double FFT20 = FFT02;
      const double FFT21 = FFT12;
      const double FFT22 = F20 * F20 + F21 * F21 + F22 * F22;

      const double g0 = F00 * ha0 + F01 * ha1 + F02 * ha2;
      const double g1 = F10 * ha0 + F11 * ha1 + F12 * ha2;
      const double g2 = F20 * ha0 + F21 * ha1 + F22 * ha2;

      const double s    = ha0 * ha0 + ha1 * ha1 + ha2 * ha2;
      const double g_sq = g0 * g0 + g1 * g1 + g2 * g2;
      const double trE  = 0.5 * (trFtF - 3.0);

      const double t1 = (lambda + mu);
      const double t2 = lambda * trE * s;
      const double t3 = mu * g_sq;
      const double t4 = mu * s;
      const double wK = h * dV;

      h00 += wK * (t1 * g0 * g0 + (t2 + t3) + t4 * (FFT00 - 1.0));
      h01 += wK * (t1 * g0 * g1 + t4 * (FFT01));
      h02 += wK * (t1 * g0 * g2 + t4 * (FFT02));
      h10 += wK * (t1 * g1 * g0 + t4 * (FFT10));
      h11 += wK * (t1 * g1 * g1 + (t2 + t3) + t4 * (FFT11 - 1.0));
      h12 += wK * (t1 * g1 * g2 + t4 * (FFT12));
      h20 += wK * (t1 * g2 * g0 + t4 * (FFT20));
      h21 += wK * (t1 * g2 * g1 + t4 * (FFT21));
      h22 += wK * (t1 * g2 * g2 + (t2 + t3) + t4 * (FFT22 - 1.0));
    }
  }

  // Shared-memory reduction (12 accumulators).
  __shared__ double smem[12 * kBlockSize];
  double *s_r0 = smem;
  double *s_r1 = s_r0 + kBlockSize;
  double *s_r2 = s_r1 + kBlockSize;
  double *s_h00 = s_r2 + kBlockSize;
  double *s_h01 = s_h00 + kBlockSize;
  double *s_h02 = s_h01 + kBlockSize;
  double *s_h10 = s_h02 + kBlockSize;
  double *s_h11 = s_h10 + kBlockSize;
  double *s_h12 = s_h11 + kBlockSize;
  double *s_h20 = s_h12 + kBlockSize;
  double *s_h21 = s_h20 + kBlockSize;
  double *s_h22 = s_h21 + kBlockSize;

  s_r0[tid] = r0;
  s_r1[tid] = r1;
  s_r2[tid] = r2;
  s_h00[tid] = h00;
  s_h01[tid] = h01;
  s_h02[tid] = h02;
  s_h10[tid] = h10;
  s_h11[tid] = h11;
  s_h12[tid] = h12;
  s_h20[tid] = h20;
  s_h21[tid] = h21;
  s_h22[tid] = h22;

  __syncthreads();

  for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const int peer = tid + stride;
      s_r0[tid] += s_r0[peer];
      s_r1[tid] += s_r1[peer];
      s_r2[tid] += s_r2[peer];
      s_h00[tid] += s_h00[peer];
      s_h01[tid] += s_h01[peer];
      s_h02[tid] += s_h02[peer];
      s_h10[tid] += s_h10[peer];
      s_h11[tid] += s_h11[peer];
      s_h12[tid] += s_h12[peer];
      s_h20[tid] += s_h20[peer];
      s_h21[tid] += s_h21[peer];
      s_h22[tid] += s_h22[peer];
    }
    __syncthreads();
  }

  if (tid != 0) return;

  r0 = s_r0[0];
  r1 = s_r1[0];
  r2 = s_r2[0];
  h00 = s_h00[0];
  h01 = s_h01[0];
  h02 = s_h02[0];
  h10 = s_h10[0];
  h11 = s_h11[0];
  h12 = s_h12[0];
  h20 = s_h20[0];
  h21 = s_h21[0];
  h22 = s_h22[0];

  double R_i[3] = {r0, r1, r2};
  R_i[0] -= d_data->f_ext()(node_i * 3 + 0);
  R_i[1] -= d_data->f_ext()(node_i * 3 + 1);
  R_i[2] -= d_data->f_ext()(node_i * 3 + 2);

  double H_i[3][3];
  const double *mass_block = d_solver->mass_diag_blocks() + node_i * 9;
  H_i[0][0] = mass_block[0] * inv_h + h00;
  H_i[0][1] = mass_block[1] * inv_h + h01;
  H_i[0][2] = mass_block[2] * inv_h + h02;
  H_i[1][0] = mass_block[3] * inv_h + h10;
  H_i[1][1] = mass_block[4] * inv_h + h11;
  H_i[1][2] = mass_block[5] * inv_h + h12;
  H_i[2][0] = mass_block[6] * inv_h + h20;
  H_i[2][1] = mass_block[7] * inv_h + h21;
  H_i[2][2] = mass_block[8] * inv_h + h22;

  // Handle constraints (pin constraints)
  int k = -1;
  if (d_solver->gpu_n_constraints() > 0) {
    k = d_solver->fixed_map()[node_i];
  }
  if (k >= 0) {
    // Current position = q_prev + h*v (computed on-the-fly for this node).
    const double v_i0 = d_solver->v_guess()[node_i * 3 + 0];
    const double v_i1 = d_solver->v_guess()[node_i * 3 + 1];
    const double v_i2 = d_solver->v_guess()[node_i * 3 + 2];

    const double x_i0 = d_solver->x12_prev()[node_i] + h * v_i0;
    const double x_i1 = d_solver->y12_prev()[node_i] + h * v_i1;
    const double x_i2 = d_solver->z12_prev()[node_i] + h * v_i2;

    const double X_i0 = d_data->x12_jac()(node_i);
    const double X_i1 = d_data->y12_jac()(node_i);
    const double X_i2 = d_data->z12_jac()(node_i);

    const double c0 = x_i0 - X_i0;
    const double c1 = x_i1 - X_i1;
    const double c2 = x_i2 - X_i2;

    const double lam0 = d_solver->lambda_guess()[k * 3 + 0];
    const double lam1 = d_solver->lambda_guess()[k * 3 + 1];
    const double lam2 = d_solver->lambda_guess()[k * 3 + 2];

    R_i[0] += h * (lam0 + rho * c0);
    R_i[1] += h * (lam1 + rho * c1);
    R_i[2] += h * (lam2 + rho * c2);

    const double h2_rho = h * h * rho;
    H_i[0][0] += h2_rho;
    H_i[1][1] += h2_rho;
    H_i[2][2] += h2_rho;
  }

  // Symmetrize and regularize Hessian
  {
    const double avg01 = 0.5 * (H_i[0][1] + H_i[1][0]);
    const double avg02 = 0.5 * (H_i[0][2] + H_i[2][0]);
    const double avg12 = 0.5 * (H_i[1][2] + H_i[2][1]);
    H_i[0][1] = H_i[1][0] = avg01;
    H_i[0][2] = H_i[2][0] = avg02;
    H_i[1][2] = H_i[2][1] = avg12;
  }

  const double trace_H = H_i[0][0] + H_i[1][1] + H_i[2][2];
  const double eps_reg = hess_eps * fmax(1.0, trace_H);
  H_i[0][0] += eps_reg;
  H_i[1][1] += eps_reg;
  H_i[2][2] += eps_reg;

  double dv[3];
  solve_3x3_vbd(H_i, R_i, dv);

  d_solver->v_guess()[node_i * 3 + 0] += omega * dv[0];
  d_solver->v_guess()[node_i * 3 + 1] += omega * dv[1];
  d_solver->v_guess()[node_i * 3 + 2] += omega * dv[2];
}

template <typename ElementType>
__global__ void vbd_update_pos_from_vel_color(SyncedVBDSolver *d_solver,
                                             ElementType *d_data,
                                             int color_start,
                                             int color_count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= color_count) return;

  const int node_i = d_solver->color_nodes()[color_start + idx];
  const double h   = d_solver->solver_time_step();
  d_data->x12()(node_i) =
      d_solver->x12_prev()(node_i) + d_solver->v_guess()(node_i * 3 + 0) * h;
  d_data->y12()(node_i) =
      d_solver->y12_prev()(node_i) + d_solver->v_guess()(node_i * 3 + 1) * h;
  d_data->z12()(node_i) =
      d_solver->z12_prev()(node_i) + d_solver->v_guess()(node_i * 3 + 2) * h;
}

// =====================================================
// Position Update Kernels
// =====================================================

template <typename ElementType>
__global__ void vbd_update_pos_prev(ElementType *d_data,
                                    SyncedVBDSolver *d_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_solver->get_n_coef()) {
    d_solver->x12_prev()(tid) = d_data->x12()(tid);
    d_solver->y12_prev()(tid) = d_data->y12()(tid);
    d_solver->z12_prev()(tid) = d_data->z12()(tid);
  }
}

template <typename ElementType>
__global__ void vbd_update_pos_from_vel(SyncedVBDSolver *d_solver,
                                        ElementType *d_data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_solver->get_n_coef()) {
    double h = d_solver->solver_time_step();
    d_data->x12()(tid) =
        d_solver->x12_prev()(tid) + d_solver->v_guess()(tid * 3 + 0) * h;
    d_data->y12()(tid) =
        d_solver->y12_prev()(tid) + d_solver->v_guess()(tid * 3 + 1) * h;
    d_data->z12()(tid) =
        d_solver->z12_prev()(tid) + d_solver->v_guess()(tid * 3 + 2) * h;
  }
}

__global__ void vbd_update_v_prev(SyncedVBDSolver *d_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_solver->get_n_coef() * 3) {
    d_solver->v_prev()[tid] = d_solver->v_guess()[tid];
  }
}

// =====================================================
// Constraint Evaluation and Dual Update
// =====================================================

template <typename ElementType>
__global__ void vbd_compute_constraint(ElementType *d_data,
                                       SyncedVBDSolver *d_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n_fixed = d_solver->gpu_n_constraints() / 3;
  if (tid < n_fixed) {
    int node_idx = d_data->fixed_nodes()[tid];
    d_data->constraint()[tid * 3 + 0] =
        d_data->x12()(node_idx) - d_data->x12_jac()(node_idx);
    d_data->constraint()[tid * 3 + 1] =
        d_data->y12()(node_idx) - d_data->y12_jac()(node_idx);
    d_data->constraint()[tid * 3 + 2] =
        d_data->z12()(node_idx) - d_data->z12_jac()(node_idx);
  }
}

template <typename ElementType>
__global__ void vbd_update_dual(ElementType *d_data,
                                SyncedVBDSolver *d_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n_constraints = d_solver->gpu_n_constraints();
  if (tid < n_constraints) {
    double rho = *d_solver->solver_rho();
    d_solver->lambda_guess()[tid] += rho * d_data->constraint()[tid];
  }
}

// =====================================================
// Compute Residual Norm (for convergence check)
// This computes the full residual R_i for each node:
//   R_i = (M_ii/h)(v_i - v_prev_i) + f_int_i - f_ext_i + h*(lam + rho*c)
// =====================================================

 template <typename ElementType>
 __global__ void vbd_compute_full_residual(ElementType *d_data,
                                          SyncedVBDSolver *d_solver,
                                          double *d_residual) {
  int node_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (node_i >= d_solver->get_n_coef())
    return;

  const double h = d_solver->solver_time_step();
  const double inv_h = 1.0 / h;
  const double rho = *d_solver->solver_rho();

  // Get velocities
  double v_i[3];
  v_i[0] = d_solver->v_guess()[node_i * 3 + 0];
  v_i[1] = d_solver->v_guess()[node_i * 3 + 1];
  v_i[2] = d_solver->v_guess()[node_i * 3 + 2];

  // Initialize residual with full consistent mass row:
  //   (M_row_i/h) * (v - v_prev)
  double R_i[3] = {0.0, 0.0, 0.0};
  {
    const int *__restrict__ offsets = d_data->csr_offsets();
    const int *__restrict__ columns = d_data->csr_columns();
    const double *__restrict__ values = d_data->csr_values();

    int row_start = offsets[node_i];
    int row_end = offsets[node_i + 1];

    for (int idx = row_start; idx < row_end; ++idx) {
      int node_j = columns[idx];
      double m_ij = values[idx];
      R_i[0] += m_ij * (d_solver->v_guess()[node_j * 3 + 0] -
                       d_solver->v_prev()[node_j * 3 + 0]) *
              inv_h;
      R_i[1] += m_ij * (d_solver->v_guess()[node_j * 3 + 1] -
                       d_solver->v_prev()[node_j * 3 + 1]) *
              inv_h;
      R_i[2] += m_ij * (d_solver->v_guess()[node_j * 3 + 2] -
                       d_solver->v_prev()[node_j * 3 + 2]) *
              inv_h;
    }
  }

  // Get incidence data for this node
  int inc_start = d_solver->incidence_offsets()[node_i];
  int inc_end = d_solver->incidence_offsets()[node_i + 1];

  double lambda = d_data->lambda();
  double mu = d_data->mu();

  // Loop over incident elements to compute f_int_i
  for (int inc_idx = inc_start; inc_idx < inc_end; ++inc_idx) {
    int2 inc = d_solver->incidence_data()[inc_idx];
    int elem_idx = inc.x;
    int local_node = inc.y;

    int global_nodes[10];
    for (int n = 0; n < 10; ++n) {
      global_nodes[n] = d_data->element_connectivity()(elem_idx, n);
    }

    for (int qp_idx = 0; qp_idx < Quadrature::N_QP_T10_5; ++qp_idx) {
      // Compute F
      double F[3][3] = {{0.0}};
      for (int a = 0; a < 10; ++a) {
        int gn = global_nodes[a];
        double x_a0 = d_solver->x12_prev()[gn] + h * d_solver->v_guess()[gn * 3 + 0];
        double x_a1 = d_solver->y12_prev()[gn] + h * d_solver->v_guess()[gn * 3 + 1];
        double x_a2 = d_solver->z12_prev()[gn] + h * d_solver->v_guess()[gn * 3 + 2];

        double g0 = d_data->grad_N_ref(elem_idx, qp_idx)(a, 0);
        double g1 = d_data->grad_N_ref(elem_idx, qp_idx)(a, 1);
        double g2 = d_data->grad_N_ref(elem_idx, qp_idx)(a, 2);

        F[0][0] += x_a0 * g0;
        F[0][1] += x_a0 * g1;
        F[0][2] += x_a0 * g2;
        F[1][0] += x_a1 * g0;
        F[1][1] += x_a1 * g1;
        F[1][2] += x_a1 * g2;
        F[2][0] += x_a2 * g0;
        F[2][1] += x_a2 * g1;
        F[2][2] += x_a2 * g2;
      }

      // F^T * F
      double FtF[3][3] = {{0.0}};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) {
            FtF[i][j] += F[k][i] * F[k][j];
          }
        }
      }
      double trFtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

      // F * F^T
      double FFT[3][3] = {{0.0}};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) {
            FFT[i][j] += F[i][k] * F[j][k];
          }
        }
      }

      // F * F^T * F
      double FFtF[3][3] = {{0.0}};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) {
            FFtF[i][j] += FFT[i][k] * F[k][j];
          }
        }
      }

      // P = lambda*(0.5*trFtF - 1.5)*F + mu*(FFtF - F)
      double P[3][3];
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          P[i][j] = lambda * (0.5 * trFtF - 1.5) * F[i][j] +
                    mu * (FFtF[i][j] - F[i][j]);
        }
      }

      double h_a[3];
      h_a[0] = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 0);
      h_a[1] = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 1);
      h_a[2] = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 2);

      double detJ = d_data->detJ_ref(elem_idx, qp_idx);
      double wq = d_data->tet5pt_weights(qp_idx);
      double dV = detJ * wq;

      // f_int contribution
      for (int i = 0; i < 3; ++i) {
        double f_int_i = 0.0;
        for (int j = 0; j < 3; ++j) {
          f_int_i += P[i][j] * h_a[j];
        }
        R_i[i] += f_int_i * dV;
      }
    }
  }

  // Subtract external force
  R_i[0] -= d_data->f_ext()(node_i * 3 + 0);
  R_i[1] -= d_data->f_ext()(node_i * 3 + 1);
  R_i[2] -= d_data->f_ext()(node_i * 3 + 2);

  // Add constraint contribution for fixed nodes
  int k2 = -1;
  if (d_solver->gpu_n_constraints() > 0) {
    k2 = d_solver->fixed_map()[node_i];
  }
  if (k2 >= 0) {
    double X_i[3];
    X_i[0] = d_data->x12_jac()(node_i);
    X_i[1] = d_data->y12_jac()(node_i);
    X_i[2] = d_data->z12_jac()(node_i);

    double x_i[3];
    x_i[0] = d_solver->x12_prev()[node_i] + h * v_i[0];
    x_i[1] = d_solver->y12_prev()[node_i] + h * v_i[1];
    x_i[2] = d_solver->z12_prev()[node_i] + h * v_i[2];

    double c_i[3];
    c_i[0] = x_i[0] - X_i[0];
    c_i[1] = x_i[1] - X_i[1];
    c_i[2] = x_i[2] - X_i[2];

    double lam_k[3];
    lam_k[0] = d_solver->lambda_guess()[k2 * 3 + 0];
    lam_k[1] = d_solver->lambda_guess()[k2 * 3 + 1];
    lam_k[2] = d_solver->lambda_guess()[k2 * 3 + 2];

    R_i[0] += h * (lam_k[0] + rho * c_i[0]);
    R_i[1] += h * (lam_k[1] + rho * c_i[1]);
    R_i[2] += h * (lam_k[2] + rho * c_i[2]);
  }

  // Store residual
  d_residual[node_i * 3 + 0] = R_i[0];
  d_residual[node_i * 3 + 1] = R_i[1];
  d_residual[node_i * 3 + 2] = R_i[2];
}

template <typename ElementType>
__global__ void vbd_compute_residual_norm(ElementType *d_data,
                                          SyncedVBDSolver *d_solver,
                                          double *d_res_sq) {
  extern __shared__ double sdata[];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n_dofs = d_solver->get_n_coef() * 3;

  double local_sq = 0.0;
  if (tid < n_dofs) {
    double g = d_solver->g()[tid];
    local_sq = g * g;
  }

  // Reduce within block
  int local_tid = threadIdx.x;
  sdata[local_tid] = local_sq;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (local_tid < s) {
      sdata[local_tid] += sdata[local_tid + s];
    }
    __syncthreads();
  }

  if (local_tid == 0) {
    atomicAdd(d_res_sq, sdata[0]);
  }
}

// =====================================================
// Update element stress caches (F/P) for current x12
// =====================================================
template <typename ElementType>
__global__ void vbd_compute_p_kernel(ElementType *d_data,
                                     SyncedVBDSolver *d_solver) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int n_qp = d_solver->gpu_n_total_qp();
  const int total = d_solver->get_n_beam() * n_qp;
  for (int idx = tid; idx < total; idx += stride) {
    const int elem_idx = idx / n_qp;
    const int qp_idx   = idx - elem_idx * n_qp;
    compute_p(elem_idx, qp_idx, d_data, d_solver->v_guess().data(),
              d_solver->solver_time_step());
  }
}

template <typename ElementType>
__global__ void vbd_clear_internal_force_kernel(ElementType *d_data) {
  clear_internal_force(d_data);
}

template <typename ElementType>
__global__ void vbd_compute_internal_force_kernel(ElementType *d_data,
                                                  SyncedVBDSolver *d_solver) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int n_shape = d_solver->gpu_n_shape();
  const int total = d_solver->get_n_beam() * n_shape;
  for (int idx = tid; idx < total; idx += stride) {
    const int elem_idx = idx / n_shape;
    const int node_idx = idx - elem_idx * n_shape;
    compute_internal_force(elem_idx, node_idx, d_data);
  }
}

template <typename ElementType>
__global__ void vbd_constraints_eval_kernel(ElementType *d_data,
                                           SyncedVBDSolver *d_solver) {
  (void)d_solver;
  compute_constraint_data(d_data);
}

template <typename ElementType>
__global__ void vbd_compute_grad_l_kernel(ElementType *d_data,
                                         SyncedVBDSolver *d_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_dofs = d_solver->get_n_coef() * 3;
  if (tid < n_dofs) {
    const double g = solver_grad_L_vbd(tid, d_data, d_solver);
    d_solver->g()[tid] = g;
  }
}

// =====================================================
// Initialize Coloring
// =====================================================

void SyncedVBDSolver::InitializeColoring() {
  if (coloring_initialized_)
    return;

  std::cout << "Initializing VBD coloring..." << std::endl;

  // Get element connectivity from CPU
  Eigen::MatrixXi h_connectivity;
  int nodes_per_elem = 0;

  if (type_ == TYPE_T10) {
    auto *typed_data = static_cast<GPU_FEAT10_Data *>(h_data_);
    typed_data->RetrieveConnectivityToCPU(h_connectivity);
    nodes_per_elem = 10;
  } else if (type_ == TYPE_3243) {
    // For ANCF elements, connectivity is different
    // Build connectivity from element_node
    // ANCF3243 has 2 physical nodes per element, 4 DOFs each = 8 shape functions
    nodes_per_elem = 8;
    h_connectivity.resize(n_beam_, nodes_per_elem);
    // Need to reconstruct connectivity - for now use simplified version
    for (int e = 0; e < n_beam_; ++e) {
      for (int i = 0; i < 4; ++i) {
        h_connectivity(e, i) = e * 4 + i;  // Node 0's DOFs
        h_connectivity(e, 4 + i) = (e + 1) * 4 + i;  // Node 1's DOFs
      }
    }
  } else if (type_ == TYPE_3443) {
    // ANCF3443 has 4 physical nodes per element, 4 DOFs each = 16 shape functions
    nodes_per_elem = 16;
    h_connectivity.resize(n_beam_, nodes_per_elem);
    // Simplified connectivity
    for (int e = 0; e < n_beam_; ++e) {
      for (int n = 0; n < 4; ++n) {
        for (int d = 0; d < 4; ++d) {
          h_connectivity(e, n * 4 + d) = n * 4 + d;  // Placeholder
        }
      }
    }
  }

  // Build adjacency graph on CPU
  auto adjacency = ANCFCPUUtils::BuildVertexAdjacency(h_connectivity, n_coef_);

  // Compute greedy coloring
  Eigen::VectorXi colors = ANCFCPUUtils::GreedyVertexColoring(adjacency);

  // Validate coloring
  bool valid = ANCFCPUUtils::ValidateColoring(h_connectivity, colors);
  if (!valid) {
    std::cerr << "Warning: Invalid coloring detected!" << std::endl;
  }

  n_colors_ = colors.maxCoeff() + 1;
  std::cout << "VBD coloring: " << n_colors_ << " colors for " << n_coef_
            << " nodes" << std::endl;

  // Build color_to_nodes mapping
  auto color_to_nodes = ANCFCPUUtils::BuildColorToNodes(colors, n_colors_);

  // Build incidence mapping
  auto incidence = ANCFCPUUtils::BuildNodeIncidence(h_connectivity, n_coef_);

  // Allocate and copy colors to device
  HANDLE_ERROR(cudaMalloc(&d_colors_, n_coef_ * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_colors_, colors.data(), n_coef_ * sizeof(int),
                          cudaMemcpyHostToDevice));

  // Build flat color_nodes array and offsets
  std::vector<int> h_color_offsets(n_colors_ + 1);
  std::vector<int> h_color_nodes;
  h_color_offsets[0] = 0;
  for (int c = 0; c < n_colors_; ++c) {
    for (int node : color_to_nodes[c]) {
      h_color_nodes.push_back(node);
    }
    h_color_offsets[c + 1] = static_cast<int>(h_color_nodes.size());
  }

  HANDLE_ERROR(cudaMalloc(&d_color_offsets_, (n_colors_ + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_color_offsets_, h_color_offsets.data(),
                          (n_colors_ + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

  // Cache on host so we don't cudaMemcpy offsets every inner iteration
  if (h_color_offsets_cache_) {
    delete[] h_color_offsets_cache_;
    h_color_offsets_cache_ = nullptr;
    h_color_offsets_cache_size_ = 0;
  }
  h_color_offsets_cache_size_ = static_cast<int>(h_color_offsets.size());
  h_color_offsets_cache_ = new int[static_cast<size_t>(h_color_offsets_cache_size_)];
  std::memcpy(h_color_offsets_cache_, h_color_offsets.data(),
              static_cast<size_t>(h_color_offsets_cache_size_) * sizeof(int));

  HANDLE_ERROR(cudaMalloc(&d_color_nodes_, h_color_nodes.size() * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_color_nodes_, h_color_nodes.data(),
                          h_color_nodes.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

  // Build color grouping schedule (host-only). We group colors such that no two
  // colors in the same group share an element, allowing a single P refresh per
  // group (reduces compute_p launches).
  {
    if (h_color_group_offsets_cache_) {
      delete[] h_color_group_offsets_cache_;
      h_color_group_offsets_cache_ = nullptr;
      h_color_group_offsets_cache_size_ = 0;
    }
    if (h_color_group_colors_cache_) {
      delete[] h_color_group_colors_cache_;
      h_color_group_colors_cache_ = nullptr;
      h_color_group_colors_cache_size_ = 0;
    }

    const int group_size = std::max(1, h_color_group_size_);

    // Build conflict matrix between colors from element connectivity.
    const int n_words = (n_colors_ + 63) / 64;
    std::vector<uint64_t> conflict_bits(
        static_cast<size_t>(n_colors_) * static_cast<size_t>(n_words), 0ULL);

    auto set_conflict = [&](int a, int b) {
      if (a == b) return;
      const int word = b >> 6;
      const int bit = b & 63;
      conflict_bits[static_cast<size_t>(a) * static_cast<size_t>(n_words) +
                    static_cast<size_t>(word)] |= (1ULL << bit);
    };

    const int n_elem = h_connectivity.rows();
    for (int e = 0; e < n_elem; ++e) {
      for (int i = 0; i < nodes_per_elem; ++i) {
        const int node_i = h_connectivity(e, i);
        const int c_i = colors[node_i];
        for (int j = i + 1; j < nodes_per_elem; ++j) {
          const int node_j = h_connectivity(e, j);
          const int c_j = colors[node_j];
          set_conflict(c_i, c_j);
          set_conflict(c_j, c_i);
        }
      }
    }

    auto conflicts = [&](int a, int b) -> bool {
      const int word = b >> 6;
      const int bit = b & 63;
      const uint64_t mask =
          conflict_bits[static_cast<size_t>(a) * static_cast<size_t>(n_words) +
                        static_cast<size_t>(word)];
      return (mask & (1ULL << bit)) != 0ULL;
    };

    std::vector<std::vector<int>> groups;
    groups.reserve(static_cast<size_t>(n_colors_));
    for (int c = 0; c < n_colors_; ++c) {
      bool placed = false;
      if (group_size > 1) {
        for (auto &g : groups) {
          if (static_cast<int>(g.size()) >= group_size) continue;
          bool ok = true;
          for (int c2 : g) {
            if (conflicts(c2, c)) {
              ok = false;
              break;
            }
          }
          if (ok) {
            g.push_back(c);
            placed = true;
            break;
          }
        }
      }
      if (!placed) {
        groups.push_back({c});
      }
    }

    n_color_groups_ = static_cast<int>(groups.size());
    h_color_group_offsets_cache_size_ = n_color_groups_ + 1;
    h_color_group_colors_cache_size_ = n_colors_;
    h_color_group_offsets_cache_ = new int[static_cast<size_t>(h_color_group_offsets_cache_size_)];
    h_color_group_colors_cache_ = new int[static_cast<size_t>(h_color_group_colors_cache_size_)];

    int cursor = 0;
    h_color_group_offsets_cache_[0] = 0;
    for (int g = 0; g < n_color_groups_; ++g) {
      for (int c : groups[static_cast<size_t>(g)]) {
        h_color_group_colors_cache_[cursor++] = c;
      }
      h_color_group_offsets_cache_[g + 1] = cursor;
    }

    if (cursor != n_colors_) {
      std::cerr << "Color grouping internal error: expected " << n_colors_
                << " colors, got " << cursor << std::endl;
    }

    if (group_size > 1) {
      std::cout << "VBD color grouping: group_size=" << group_size
                << " => " << n_color_groups_ << " groups (from " << n_colors_
                << " colors)" << std::endl;
    }
  }

  // Build incidence data arrays
  std::vector<int> h_incidence_offsets(n_coef_ + 1);
  std::vector<int2> h_incidence_data;
  h_incidence_offsets[0] = 0;
  for (int i = 0; i < n_coef_; ++i) {
    for (const auto &p : incidence[i]) {
      h_incidence_data.push_back(make_int2(p.first, p.second));
    }
    h_incidence_offsets[i + 1] = static_cast<int>(h_incidence_data.size());
  }

  HANDLE_ERROR(
      cudaMalloc(&d_incidence_offsets_, (n_coef_ + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_incidence_offsets_, h_incidence_offsets.data(),
                          (n_coef_ + 1) * sizeof(int), cudaMemcpyHostToDevice));

  HANDLE_ERROR(
      cudaMalloc(&d_incidence_data_, h_incidence_data.size() * sizeof(int2)));
  HANDLE_ERROR(cudaMemcpy(d_incidence_data_, h_incidence_data.data(),
                          h_incidence_data.size() * sizeof(int2),
                          cudaMemcpyHostToDevice));

  coloring_initialized_ = true;

  // Update device copy
  HANDLE_ERROR(cudaMemcpy(d_vbd_solver_, this, sizeof(SyncedVBDSolver),
                          cudaMemcpyHostToDevice));
}

// =====================================================
// Initialize Diagonal Mass Blocks
// =====================================================

void SyncedVBDSolver::InitializeMassDiagBlocks() {
  std::cout << "Initializing VBD diagonal mass blocks..." << std::endl;

  // Allocate diagonal mass blocks (3x3 per node, stored as 9 doubles)
  HANDLE_ERROR(cudaMalloc(&d_mass_diag_blocks_, n_coef_ * 9 * sizeof(double)));
  HANDLE_ERROR(cudaMemset(d_mass_diag_blocks_, 0, n_coef_ * 9 * sizeof(double)));

  // For consistent mass, we need to extract diagonal blocks from the CSR mass matrix
  // The mass matrix has already been built by CalcMassMatrix()

  if (type_ == TYPE_T10) {
    auto *typed_data = static_cast<GPU_FEAT10_Data *>(h_data_);

    // Retrieve mass CSR to CPU
    std::vector<int> offsets, columns;
    std::vector<double> values;
    typed_data->RetrieveMassCSRToCPU(offsets, columns, values);

    // Extract diagonal blocks (M_ii for each node)
    std::vector<double> h_mass_diag(n_coef_ * 9, 0.0);

    for (int i = 0; i < n_coef_; ++i) {
      // Find diagonal entry M(i,i)
      for (int idx = offsets[i]; idx < offsets[i + 1]; ++idx) {
        if (columns[idx] == i) {
          // This is the diagonal entry - mass is scalar * I_3
          double m_ii = values[idx];
          // Set 3x3 block as m_ii * I_3
          h_mass_diag[i * 9 + 0] = m_ii;  // (0,0)
          h_mass_diag[i * 9 + 4] = m_ii;  // (1,1)
          h_mass_diag[i * 9 + 8] = m_ii;  // (2,2)
          break;
        }
      }
    }

    HANDLE_ERROR(cudaMemcpy(d_mass_diag_blocks_, h_mass_diag.data(),
                            n_coef_ * 9 * sizeof(double),
                            cudaMemcpyHostToDevice));
  } else {
    // For ANCF elements, use simplified lumped mass
    // TODO: Implement proper consistent mass extraction for ANCF
    std::vector<double> h_mass_diag(n_coef_ * 9, 0.0);
    // Use lumped mass approximation
    double total_mass_approx = 1.0;  // Placeholder
    double m_per_node = total_mass_approx / n_coef_;
    for (int i = 0; i < n_coef_; ++i) {
      h_mass_diag[i * 9 + 0] = m_per_node;
      h_mass_diag[i * 9 + 4] = m_per_node;
      h_mass_diag[i * 9 + 8] = m_per_node;
    }
    HANDLE_ERROR(cudaMemcpy(d_mass_diag_blocks_, h_mass_diag.data(),
                            n_coef_ * 9 * sizeof(double),
                            cudaMemcpyHostToDevice));
  }

  // Update device copy
  HANDLE_ERROR(cudaMemcpy(d_vbd_solver_, this, sizeof(SyncedVBDSolver),
                          cudaMemcpyHostToDevice));
}

// =====================================================
// Initialize Fixed Map (node -> fixed index, or -1)
// =====================================================

void SyncedVBDSolver::InitializeFixedMap() {
  if (fixed_map_initialized_) {
    return;
  }

  HANDLE_ERROR(cudaMalloc(&d_fixed_map_, n_coef_ * sizeof(int)));
  HANDLE_ERROR(cudaMemset(d_fixed_map_, 0xFF, n_coef_ * sizeof(int)));  // -1

  if (n_constraints_ > 0) {
    const int threads = 256;
    int n_fixed = n_constraints_ / 3;
    int blocks = (n_fixed + threads - 1) / threads;

    if (type_ == TYPE_T10) {
      vbd_build_fixed_map_kernel<<<blocks, threads>>>(
          static_cast<GPU_FEAT10_Data *>(d_data_), d_fixed_map_);
      HANDLE_ERROR(cudaGetLastError());
    }
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  fixed_map_initialized_ = true;
  HANDLE_ERROR(cudaMemcpy(d_vbd_solver_, this, sizeof(SyncedVBDSolver),
                          cudaMemcpyHostToDevice));
}

// =====================================================
// L2 Norm Computation
// =====================================================

double SyncedVBDSolver::compute_l2_norm_cublas(double *d_vec, int n_dofs) {
  cublasDnrm2(cublas_handle_, n_dofs, d_vec, 1, d_norm_temp_);
  double h_norm;
  HANDLE_ERROR(
      cudaMemcpy(&h_norm, d_norm_temp_, sizeof(double), cudaMemcpyDeviceToHost));
  return h_norm;
}

// =====================================================
// CUDA Graphs (required)
// =====================================================

void SyncedVBDSolver::DestroyCudaGraphs() {
  if (inner_sweep_graph_exec_) {
    HANDLE_ERROR(cudaGraphExecDestroy(inner_sweep_graph_exec_));
    inner_sweep_graph_exec_ = nullptr;
  }
  if (post_outer_graph_exec_) {
    HANDLE_ERROR(cudaGraphExecDestroy(post_outer_graph_exec_));
    post_outer_graph_exec_ = nullptr;
  }

  graph_threads_ = 0;
  graph_blocks_coef_ = 0;
  graph_blocks_p_ = 0;
  graph_n_colors_ = 0;
  graph_n_constraints_ = 0;
  graph_color_group_size_ = 1;
  graph_n_color_groups_ = 0;
}

bool SyncedVBDSolver::EnsureInnerSweepGraph(int threads, int blocks_p) {
  if (type_ != TYPE_T10) {
    std::cerr << "EnsureInnerSweepGraph is only implemented for TYPE_T10."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!coloring_initialized_ || !fixed_map_initialized_) {
    std::cerr << "EnsureInnerSweepGraph requires InitializeColoring() and "
                 "InitializeFixedMap() first."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  const bool signature_match =
      inner_sweep_graph_exec_ && graph_threads_ == threads &&
      graph_blocks_p_ == blocks_p && graph_n_colors_ == n_colors_ &&
      graph_n_constraints_ == n_constraints_ &&
      graph_color_group_size_ == h_color_group_size_ &&
      graph_n_color_groups_ == n_color_groups_;
  if (signature_match) return false;

  // If a previously captured graph exists but the signature changed, destroy it.
  if (inner_sweep_graph_exec_) {
    const bool shared_mismatch =
        graph_threads_ != threads || graph_n_constraints_ != n_constraints_;
    const bool inner_mismatch =
        shared_mismatch || graph_blocks_p_ != blocks_p ||
        graph_n_colors_ != n_colors_ ||
        graph_color_group_size_ != h_color_group_size_ ||
        graph_n_color_groups_ != n_color_groups_;
    if (inner_mismatch) {
      HANDLE_ERROR(cudaGraphExecDestroy(inner_sweep_graph_exec_));
      inner_sweep_graph_exec_ = nullptr;
    }
    if (shared_mismatch && post_outer_graph_exec_) {
      HANDLE_ERROR(cudaGraphExecDestroy(post_outer_graph_exec_));
      post_outer_graph_exec_ = nullptr;
    }
  } else if (post_outer_graph_exec_) {
    // If post-outer graph exists but shared signature fields changed, drop it.
    const bool shared_mismatch =
        graph_threads_ != threads || graph_n_constraints_ != n_constraints_;
    if (shared_mismatch) {
      HANDLE_ERROR(cudaGraphExecDestroy(post_outer_graph_exec_));
      post_outer_graph_exec_ = nullptr;
    }
  }

  if (!h_color_offsets_cache_ || h_color_offsets_cache_size_ != n_colors_ + 1) {
    std::cerr << "CUDA graph init requires host color-offset cache. "
                 "Call InitializeColoring() first."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!h_color_group_offsets_cache_ ||
      h_color_group_offsets_cache_size_ != n_color_groups_ + 1 ||
      !h_color_group_colors_cache_ ||
      h_color_group_colors_cache_size_ != n_colors_) {
    std::cerr << "CUDA graph init requires host color-group schedule cache."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  constexpr int kVbdBlockThreads = 64;
  if (threads != kVbdBlockThreads) {
    std::cerr << "CUDA graph init expects threads=" << kVbdBlockThreads
              << " for vbd_update_color_block_kernel<" << kVbdBlockThreads
              << ">."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!graph_capture_stream_) {
    std::cerr << "CUDA graph capture stream is null; aborting."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  auto *typed_data = static_cast<GPU_FEAT10_Data *>(d_data_);

  // Stream capture executes the recorded kernels once. We only call this at the
  // first inner sweep so that execution counts as sweep 0.
  cudaGraph_t graph = nullptr;
  cudaError_t err = cudaStreamSynchronize(0);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamSynchronize(default) failed during graph init: "
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  err = cudaStreamSynchronize(graph_capture_stream_);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamSynchronize(capture) failed during graph init: "
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  err = cudaStreamBeginCapture(graph_capture_stream_,
                               cudaStreamCaptureModeRelaxed);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamBeginCapture failed: "
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  for (int g = 0; g < n_color_groups_; ++g) {
    const int group_begin = h_color_group_offsets_cache_[g];
    const int group_end = h_color_group_offsets_cache_[g + 1];
    for (int gi = group_begin; gi < group_end; ++gi) {
      const int c = h_color_group_colors_cache_[gi];
      const int color_start = h_color_offsets_cache_[c];
      const int color_count =
          h_color_offsets_cache_[c + 1] - h_color_offsets_cache_[c];
      if (color_count <= 0) continue;

      const int blocks = color_count;
      vbd_update_color_block_kernel<kVbdBlockThreads>
          <<<blocks, threads, 0, graph_capture_stream_>>>(
              typed_data, d_vbd_solver_, color_start, color_count);

      const int blocks_color = (color_count + threads - 1) / threads;
      vbd_update_pos_from_vel_color<<<blocks_color, threads, 0,
                                     graph_capture_stream_>>>(
          d_vbd_solver_, typed_data, color_start, color_count);
    }

    // Refresh global element cache once per group.
    vbd_compute_p_kernel<<<blocks_p, threads, 0, graph_capture_stream_>>>(
        typed_data, d_vbd_solver_);
  }

  err = cudaStreamEndCapture(graph_capture_stream_, &graph);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamEndCapture(inner sweep) failed: "
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  // Ensure the sweep executed during capture completes before continuing on the
  // default stream.
  err = cudaStreamSynchronize(graph_capture_stream_);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamSynchronize(capture) failed after capture: "
              << cudaGetErrorString(err) << std::endl;
    HANDLE_ERROR(cudaGraphDestroy(graph));
    exit(EXIT_FAILURE);
  }

  cudaGraphNode_t error_node = nullptr;
  char log_buffer[4096] = {0};
  const cudaError_t inst_err =
      cudaGraphInstantiate(&inner_sweep_graph_exec_, graph, &error_node,
                           log_buffer, sizeof(log_buffer));
  if (inst_err != cudaSuccess) {
    std::cerr << "cudaGraphInstantiate(inner sweep) failed: "
              << cudaGetErrorString(inst_err) << "\n"
              << log_buffer << std::endl;
    HANDLE_ERROR(cudaGraphDestroy(graph));
    exit(EXIT_FAILURE);
  }
  HANDLE_ERROR(cudaGraphDestroy(graph));

  graph_threads_ = threads;
  graph_blocks_p_ = blocks_p;
  graph_n_colors_ = n_colors_;
  graph_n_constraints_ = n_constraints_;
  graph_color_group_size_ = h_color_group_size_;
  graph_n_color_groups_ = n_color_groups_;
  return true;
}

bool SyncedVBDSolver::EnsurePostOuterGraph(int threads, int blocks_coef) {
  if (type_ != TYPE_T10) {
    std::cerr << "EnsurePostOuterGraph is only implemented for TYPE_T10."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!coloring_initialized_ || !fixed_map_initialized_) {
    std::cerr << "EnsurePostOuterGraph requires InitializeColoring() and "
                 "InitializeFixedMap() first."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  const bool signature_match =
      post_outer_graph_exec_ && graph_threads_ == threads &&
      graph_blocks_coef_ == blocks_coef && graph_n_colors_ == n_colors_ &&
      graph_n_constraints_ == n_constraints_;
  if (signature_match) return false;

  // If a previously captured graph exists but the signature changed, destroy it.
  if (post_outer_graph_exec_) {
    const bool shared_mismatch =
        graph_threads_ != threads || graph_n_constraints_ != n_constraints_;
    const bool post_mismatch =
        shared_mismatch || graph_blocks_coef_ != blocks_coef;
    if (post_mismatch) {
      HANDLE_ERROR(cudaGraphExecDestroy(post_outer_graph_exec_));
      post_outer_graph_exec_ = nullptr;
    }
    if (shared_mismatch && inner_sweep_graph_exec_) {
      HANDLE_ERROR(cudaGraphExecDestroy(inner_sweep_graph_exec_));
      inner_sweep_graph_exec_ = nullptr;
    }
  } else if (inner_sweep_graph_exec_) {
    // If inner-sweep graph exists but shared signature fields changed, drop it.
    const bool shared_mismatch =
        graph_threads_ != threads || graph_n_constraints_ != n_constraints_;
    if (shared_mismatch) {
      HANDLE_ERROR(cudaGraphExecDestroy(inner_sweep_graph_exec_));
      inner_sweep_graph_exec_ = nullptr;
    }
  }

  if (!graph_capture_stream_) {
    std::cerr << "CUDA graph capture stream is null; aborting."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  auto *typed_data = static_cast<GPU_FEAT10_Data *>(d_data_);

  cudaGraph_t graph = nullptr;
  cudaError_t err = cudaStreamSynchronize(0);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamSynchronize(default) failed during graph init: "
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  err = cudaStreamSynchronize(graph_capture_stream_);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamSynchronize(capture) failed during graph init: "
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  err = cudaStreamBeginCapture(graph_capture_stream_,
                               cudaStreamCaptureModeRelaxed);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamBeginCapture failed: "
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  vbd_update_pos_from_vel<<<blocks_coef, threads, 0, graph_capture_stream_>>>(
      d_vbd_solver_, typed_data);
  if (n_constraints_ > 0) {
    const int n_fixed = n_constraints_ / 3;
    const int blocks_c = (n_fixed + threads - 1) / threads;
    vbd_compute_constraint<<<blocks_c, threads, 0, graph_capture_stream_>>>(
        typed_data, d_vbd_solver_);
  }

  err = cudaStreamEndCapture(graph_capture_stream_, &graph);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamEndCapture(post outer) failed: "
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }

  err = cudaStreamSynchronize(graph_capture_stream_);
  if (err != cudaSuccess) {
    std::cerr << "cudaStreamSynchronize(capture) failed after capture: "
              << cudaGetErrorString(err) << std::endl;
    HANDLE_ERROR(cudaGraphDestroy(graph));
    exit(EXIT_FAILURE);
  }

  cudaGraphNode_t error_node = nullptr;
  char log_buffer[4096] = {0};
  const cudaError_t inst_err =
      cudaGraphInstantiate(&post_outer_graph_exec_, graph, &error_node,
                           log_buffer, sizeof(log_buffer));
  if (inst_err != cudaSuccess) {
    std::cerr << "cudaGraphInstantiate(post outer) failed: "
              << cudaGetErrorString(inst_err) << "\n"
              << log_buffer << std::endl;
    HANDLE_ERROR(cudaGraphDestroy(graph));
    exit(EXIT_FAILURE);
  }
  HANDLE_ERROR(cudaGraphDestroy(graph));

  graph_threads_ = threads;
  graph_blocks_coef_ = blocks_coef;
  graph_n_colors_ = n_colors_;
  graph_n_constraints_ = n_constraints_;
  graph_color_group_size_ = h_color_group_size_;
  graph_n_color_groups_ = n_color_groups_;
  return true;
}

// =====================================================
// Main VBD Solve Step
// =====================================================

void SyncedVBDSolver::OneStepVBD() {
  cudaDeviceProp props;
  HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));
  const int max_grid_stride_blocks = 32 * props.multiProcessorCount;

  // Ensure coloring is initialized
  if (!coloring_initialized_) {
    InitializeColoring();
    InitializeMassDiagBlocks();
  }
  if (!fixed_map_initialized_) {
    InitializeFixedMap();
  }

  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  constexpr int kVbdBlockThreads = 64;
  const int threads = kVbdBlockThreads;
  const int n_dofs = n_coef_ * 3;

  GPU_FEAT10_Data *typed_data = nullptr;
  int blocks_coef = 0;
  int blocks_dof = 0;
  int blocks_p = 0;
  int blocks_if = 0;
  int blocks_fixed = 0;

  if (type_ == TYPE_T10) {
    typed_data = static_cast<GPU_FEAT10_Data *>(d_data_);
    blocks_coef = (n_coef_ + threads - 1) / threads;
    blocks_dof = (n_dofs + threads - 1) / threads;

    const int total_qp = n_beam_ * n_total_qp_;
    blocks_p = (total_qp + threads - 1) / threads;
    blocks_p = std::max(1, std::min(blocks_p, max_grid_stride_blocks));

    const int total_if = n_beam_ * n_shape_;
    blocks_if = (total_if + threads - 1) / threads;
    blocks_if = std::max(1, std::min(blocks_if, max_grid_stride_blocks));

    blocks_fixed =
        ((n_constraints_ > 0 ? (n_constraints_ / 3) : 1) + threads - 1) /
        threads;
  }

  if (type_ != TYPE_T10) {
    std::cerr << "SyncedVBDSolver::OneStepVBD is only implemented for TYPE_T10."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  auto launch_grad_l_residual_check = [&]() {
    // Evaluate the same residual/gradient used by AdamW/Newton formulation,
    // using cached F/P and (re)assembled f_int.
    vbd_clear_internal_force_kernel<<<blocks_dof, threads>>>(typed_data);
    vbd_compute_internal_force_kernel<<<blocks_if, threads>>>(typed_data,
                                                              d_vbd_solver_);
    if (n_constraints_ > 0) {
      vbd_constraints_eval_kernel<<<blocks_fixed, threads>>>(typed_data,
                                                             d_vbd_solver_);
    }
    vbd_compute_grad_l_kernel<<<blocks_dof, threads>>>(typed_data, d_vbd_solver_);
  };

  // Store previous positions
  HANDLE_ERROR(cudaEventRecord(start));
  vbd_update_pos_prev<<<blocks_coef, threads>>>(
      static_cast<GPU_FEAT10_Data *>(d_data_), d_vbd_solver_);

  // ALM outer loop
  for (int outer_iter = 0; outer_iter < h_max_outer_; ++outer_iter) {
    double R0 = -1.0;

    // Build initial x12/F/P caches for the current v_guess before running the
    // colored inner sweeps in this outer iteration.
    vbd_update_pos_from_vel<<<blocks_coef, threads>>>(d_vbd_solver_, typed_data);
    vbd_compute_p_kernel<<<blocks_p, threads>>>(typed_data, d_vbd_solver_);

    if (h_conv_check_interval_ > 0 && type_ == TYPE_T10) {
      launch_grad_l_residual_check();
      R0 = compute_l2_norm_cublas(d_g_, n_dofs);
      std::cout << "    [VBD check] init: ||g||=" << std::scientific << R0
                << std::endl;
    }

    // VBD inner loop (colored Gauss-Seidel sweeps)
    const bool sweep0_already_executed =
        EnsureInnerSweepGraph(threads, blocks_p);
    for (int inner_iter = 0; inner_iter < h_max_inner_; ++inner_iter) {
      if (!(inner_iter == 0 && sweep0_already_executed)) {
        HANDLE_ERROR(cudaGraphLaunch(inner_sweep_graph_exec_, 0));
      }

      // Compute and print residual every convergence_check_interval sweeps (like Python)
      // Python: if verbose and (sweep % 5 == 0 or sweep == max_sweeps-1)
      if (h_conv_check_interval_ > 0 &&
          (inner_iter % h_conv_check_interval_ == 0 ||
           inner_iter == h_max_inner_ - 1)) {
        if (type_ == TYPE_T10) {
          launch_grad_l_residual_check();
          const double R_norm = compute_l2_norm_cublas(d_g_, n_dofs);

          std::cout << "    VBD sweep " << std::setw(3) << inner_iter 
                    << ": ||g|| = " << std::scientific << R_norm;
          std::cout << std::endl;

          const double stop_thresh =
              fmax(h_inner_tol_, h_inner_rtol_ * (R0 >= 0.0 ? R0 : R_norm));
          if (R_norm <= stop_thresh) {
            break;
          }
        }
      }
    }

    // Update positions from final velocities after all sweeps
    if (type_ == TYPE_T10) {
      const bool post_already_executed =
          EnsurePostOuterGraph(threads, blocks_coef);
      if (!post_already_executed) {
        HANDLE_ERROR(cudaGraphLaunch(post_outer_graph_exec_, 0));
      }
    }

    // Compute constraint violation and update multipliers
    if (n_constraints_ > 0) {
      // Check constraint norm
      double c_norm = 0.0;
      if (d_constraint_ptr_ != nullptr) {
        c_norm = compute_l2_norm_cublas(d_constraint_ptr_, n_constraints_);
      } else if (type_ == TYPE_T10) {
        auto *typed_data = static_cast<GPU_FEAT10_Data *>(h_data_);
        c_norm = compute_l2_norm_cublas(typed_data->Get_Constraint_Ptr(),
                                        n_constraints_);
      }

      std::cout << "VBD outer " << outer_iter << ": ||c|| = " << c_norm
                << std::endl;

      if (c_norm < h_outer_tol_) {
        std::cout << "VBD converged at outer iteration " << outer_iter
                  << std::endl;
        break;
      }

      // Update dual variables
      int blocks_d = (n_constraints_ + threads - 1) / threads;
      if (type_ == TYPE_T10) {
        vbd_update_dual<<<blocks_d, threads>>>(
            static_cast<GPU_FEAT10_Data *>(d_data_), d_vbd_solver_);
      }
    }
  }

  // Update v_prev for next time step
  {
    int blocks = (n_dofs + threads - 1) / threads;
    vbd_update_v_prev<<<blocks, threads>>>(d_vbd_solver_);
  }

  // Refresh global F/P caches for postprocessing/debug (matches updated x12)
  if (type_ == TYPE_T10) {
    int total = n_beam_ * n_total_qp_;
    int blocks = (total + threads - 1) / threads;
    vbd_compute_p_kernel<<<blocks, threads>>>(
        static_cast<GPU_FEAT10_Data *>(d_data_), d_vbd_solver_);
  }

  HANDLE_ERROR(cudaEventRecord(stop));
  HANDLE_ERROR(cudaDeviceSynchronize());

  float milliseconds = 0.0f;
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "OneStepVBD kernel time: " << milliseconds << " ms" << std::endl;

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
}
