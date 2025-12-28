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
#include <cstring>
#include <iomanip>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

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

 template <typename ElementType>
 __global__ void vbd_update_color_kernel(
     ElementType *d_data, SyncedVBDSolver *d_solver, int color_start,
     int color_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= color_count)
    return;

  // Get the actual node index for this color
  int node_i = d_solver->color_nodes()[color_start + tid];

  const double h       = d_solver->solver_time_step();
  const double inv_h   = 1.0 / h;
  const double omega   = d_solver->solver_omega();
  const double hess_eps = d_solver->solver_hess_eps();
  const double rho     = *d_solver->solver_rho();

  // Get current velocity for this node
  double v_i[3];
  v_i[0] = d_solver->v_guess()[node_i * 3 + 0];
  v_i[1] = d_solver->v_guess()[node_i * 3 + 1];
  v_i[2] = d_solver->v_guess()[node_i * 3 + 2];

  // Get current position = q_prev + h*v
  double x_prev_i[3];
  x_prev_i[0] = d_solver->x12_prev()[node_i];
  x_prev_i[1] = d_solver->y12_prev()[node_i];
  x_prev_i[2] = d_solver->z12_prev()[node_i];

  double x_i[3];
  x_i[0] = x_prev_i[0] + h * v_i[0];
  x_i[1] = x_prev_i[1] + h * v_i[1];
  x_i[2] = x_prev_i[2] + h * v_i[2];

  // Get mass diagonal block for this node (3x3)
  double M_ii[3][3];
  const double *mass_block = d_solver->mass_diag_blocks() + node_i * 9;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      M_ii[i][j] = mass_block[i * 3 + j];
    }
  }

  // Initialize residual and local Hessian block (3x3) for node_i.
  // Residual/cost uses the full problem (matches other solvers' grad_L):
  //   R_i = (M_row_i/h)(v - v_prev) + f_int_i - f_ext_i + h * J_i^T (lam + rho*c)
  // Local Hessian uses only the diagonal 3x3 block approximation:
  //   H_i ≈ (M_ii/h) + h*K_ii + h^2*rho*I   (pins only)
  double R_i[3] = {0.0, 0.0, 0.0};
  double H_i[3][3] = {{0.0}};

  // Mass contribution to residual: full consistent mass row
  //   (M_row_i/h) * (v - v_prev)
  {
    const int *__restrict__ offsets = d_data->csr_offsets();
    const int *__restrict__ columns = d_data->csr_columns();
    const double *__restrict__ values = d_data->csr_values();

    int row_start = offsets[node_i];
    int row_end = offsets[node_i + 1];

    for (int idx = row_start; idx < row_end; ++idx) {
      int node_j = columns[idx];
      double m_ij = values[idx];

      R_i[0] += m_ij *
                (d_solver->v_guess()[node_j * 3 + 0] -
                 d_solver->v_prev()[node_j * 3 + 0]) *
                inv_h;
      R_i[1] += m_ij *
                (d_solver->v_guess()[node_j * 3 + 1] -
                 d_solver->v_prev()[node_j * 3 + 1]) *
                inv_h;
      R_i[2] += m_ij *
                (d_solver->v_guess()[node_j * 3 + 2] -
                 d_solver->v_prev()[node_j * 3 + 2]) *
                inv_h;
    }
  }

  // Mass contribution to Hessian: M_ii / h
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      H_i[i][j] = M_ii[i][j] * inv_h;
    }
  }

  // Get incidence data for this node - iterate over all incident elements
  int inc_start = d_solver->incidence_offsets()[node_i];
  int inc_end = d_solver->incidence_offsets()[node_i + 1];

  // Material parameters
  double lambda = d_data->lambda();
  double mu = d_data->mu();

  // Loop over incident elements
  for (int inc_idx = inc_start; inc_idx < inc_end; ++inc_idx) {
    int2 inc = d_solver->incidence_data()[inc_idx];
    int elem_idx = inc.x;
    int local_node = inc.y;

    // Get element connectivity
    int global_nodes[10];
#pragma unroll
    for (int n = 0; n < 10; ++n) {
      global_nodes[n] = d_data->element_connectivity()(elem_idx, n);
    }

    // Loop over quadrature points for this element
    for (int qp_idx = 0; qp_idx < Quadrature::N_QP_T10_5; ++qp_idx) {
      // Compute deformation gradient F
      double F[3][3] = {{0.0}};
#pragma unroll
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

      // Compute F^T * F
      double FtF[3][3] = {{0.0}};
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
#pragma unroll
          for (int k = 0; k < 3; ++k) {
            FtF[i][j] += F[k][i] * F[k][j];
          }
        }
      }

      double trFtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

      // Compute F * F^T
      double FFT[3][3] = {{0.0}};
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
#pragma unroll
          for (int k = 0; k < 3; ++k) {
            FFT[i][j] += F[i][k] * F[j][k];
          }
        }
      }

      // Compute F * F^T * F
      double FFtF[3][3] = {{0.0}};
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) {
            FFtF[i][j] += FFT[i][k] * F[k][j];
          }
        }
      }

      // Compute first Piola-Kirchhoff stress P (SVK material)
      // P = lambda*(0.5*tr(F^T*F) - 1.5)*F + mu*(F*F^T*F - F)
      double P[3][3];
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          P[i][j] = lambda * (0.5 * trFtF - 1.5) * F[i][j] +
                    mu * (FFtF[i][j] - F[i][j]);
        }
      }

      // Get shape gradient for node_i at this QP
      double h_a[3];
      h_a[0] = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 0);
      h_a[1] = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 1);
      h_a[2] = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 2);

      // Integration weight
      double detJ = d_data->detJ_ref(elem_idx, qp_idx);
      double wq = d_data->tet5pt_weights(qp_idx);
      double dV = detJ * wq;

      // Internal force contribution: f_int_i += (P @ h_a) * dV
      double f_int_contrib[3];
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        f_int_contrib[i] = 0.0;
        for (int j = 0; j < 3; ++j) {
          f_int_contrib[i] += P[i][j] * h_a[j];
        }
        f_int_contrib[i] *= dV;
      }

      // Add to residual (internal force goes positive)
      R_i[0] += f_int_contrib[0];
      R_i[1] += f_int_contrib[1];
      R_i[2] += f_int_contrib[2];

      // Compute diagonal tangent block K_aa for this quadrature point
      // From Python: Kaa = (lam+mu)*outer(g,g) + lam*trE*s*I + mu*g2*I + mu*s*(FFT-I)
      // where g = F @ h_a, s = h_a^T @ h_a, g2 = g^T @ g, trE = 0.5*(tr(F^T*F)-3)

      double g[3];  // g = F @ h_a
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        g[i] = 0.0;
        for (int j = 0; j < 3; ++j) {
          g[i] += F[i][j] * h_a[j];
        }
      }

      double s = h_a[0] * h_a[0] + h_a[1] * h_a[1] + h_a[2] * h_a[2];
      double g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
      double trE = 0.5 * (trFtF - 3.0);

      // Compute K_aa block
      double K_aa[3][3];
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          double I_ij = (i == j) ? 1.0 : 0.0;
          K_aa[i][j] = (lambda + mu) * g[i] * g[j] +
                       lambda * trE * s * I_ij +
                       mu * g2 * I_ij +
                       mu * s * (FFT[i][j] - I_ij);
          K_aa[i][j] *= dV;
        }
      }

      // Add tangent contribution to Hessian: H += h * K_aa
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          H_i[i][j] += h * K_aa[i][j];
        }
      }
    }  // end qp loop
  }    // end incidence loop

  // Subtract external force from residual
  R_i[0] -= d_data->f_ext()(node_i * 3 + 0);
  R_i[1] -= d_data->f_ext()(node_i * 3 + 1);
  R_i[2] -= d_data->f_ext()(node_i * 3 + 2);

  // Handle constraints (pin constraints)
  int k = -1;
  if (d_solver->gpu_n_constraints() > 0) {
    k = d_solver->fixed_map()[node_i];
  }
  if (k >= 0) {
    // Get reference position
    double X_i[3];
    X_i[0] = d_data->x12_jac()(node_i);
    X_i[1] = d_data->y12_jac()(node_i);
    X_i[2] = d_data->z12_jac()(node_i);

    // Constraint: c_i = x_i - X_i
    double c_i[3];
    c_i[0] = x_i[0] - X_i[0];
    c_i[1] = x_i[1] - X_i[1];
    c_i[2] = x_i[2] - X_i[2];

    // Get Lagrange multiplier for this constraint
    double lam_k[3];
    lam_k[0] = d_solver->lambda_guess()[k * 3 + 0];
    lam_k[1] = d_solver->lambda_guess()[k * 3 + 1];
    lam_k[2] = d_solver->lambda_guess()[k * 3 + 2];

    // Add to residual: h * (lambda + rho * c)
    R_i[0] += h * (lam_k[0] + rho * c_i[0]);
    R_i[1] += h * (lam_k[1] + rho * c_i[1]);
    R_i[2] += h * (lam_k[2] + rho * c_i[2]);

    // Add to Hessian: h^2 * rho * I
    double h2_rho = h * h * rho;
    H_i[0][0] += h2_rho;
    H_i[1][1] += h2_rho;
    H_i[2][2] += h2_rho;
  }

  // Symmetrize and regularize Hessian
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = i + 1; j < 3; ++j) {
      double avg = 0.5 * (H_i[i][j] + H_i[j][i]);
      H_i[i][j] = avg;
      H_i[j][i] = avg;
    }
  }

  // Add regularization
  double trace_H = H_i[0][0] + H_i[1][1] + H_i[2][2];
  double eps_reg = hess_eps * fmax(1.0, trace_H);
  H_i[0][0] += eps_reg;
  H_i[1][1] += eps_reg;
  H_i[2][2] += eps_reg;

  // Solve for velocity update: dv = -H^{-1} * R
  double dv[3];
  solve_3x3_vbd(H_i, R_i, dv);

  // Apply relaxation and update velocity
  d_solver->v_guess()[node_i * 3 + 0] = v_i[0] + omega * dv[0];
  d_solver->v_guess()[node_i * 3 + 1] = v_i[1] + omega * dv[1];
  d_solver->v_guess()[node_i * 3 + 2] = v_i[2] + omega * dv[2];
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
// Main VBD Solve Step
// =====================================================

void SyncedVBDSolver::OneStepVBD() {
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

  const int threads = 256;
  const int n_dofs = n_coef_ * 3;

  // Store previous positions
  HANDLE_ERROR(cudaEventRecord(start));
  if (type_ == TYPE_T10) {
    int blocks = (n_coef_ + threads - 1) / threads;
    vbd_update_pos_prev<<<blocks, threads>>>(
        static_cast<GPU_FEAT10_Data *>(d_data_), d_vbd_solver_);
  }
  // ALM outer loop
  for (int outer_iter = 0; outer_iter < h_max_outer_; ++outer_iter) {
    double R0 = -1.0;

    if (h_conv_check_interval_ > 0 && type_ == TYPE_T10) {
      int blocks = (n_coef_ + threads - 1) / threads;
      vbd_compute_full_residual<<<blocks, threads>>>(
          static_cast<GPU_FEAT10_Data *>(d_data_), d_vbd_solver_, d_g_);
      HANDLE_ERROR(cudaDeviceSynchronize());
      R0 = compute_l2_norm_cublas(d_g_, n_dofs);
    }

    // VBD inner loop (colored Gauss-Seidel sweeps)
    for (int inner_iter = 0; inner_iter < h_max_inner_; ++inner_iter) {
      // Process each color sequentially (nodes within a color in parallel)
      // Note: Positions are computed on-the-fly in the kernel from x_prev + h*v
      for (int c = 0; c < n_colors_; ++c) {
        int color_start = 0;
        int color_count = 0;
        if (h_color_offsets_cache_ && h_color_offsets_cache_size_ == n_colors_ + 1) {
          color_start = h_color_offsets_cache_[c];
          color_count = h_color_offsets_cache_[c + 1] - color_start;
        } else {
          int h_color_offsets[2];
          HANDLE_ERROR(cudaMemcpy(h_color_offsets, d_color_offsets_ + c,
                                  2 * sizeof(int), cudaMemcpyDeviceToHost));
          color_start = h_color_offsets[0];
          color_count = h_color_offsets[1] - h_color_offsets[0];
        }

        if (color_count > 0) {
          int blocks = (color_count + threads - 1) / threads;

          if (type_ == TYPE_T10) {
            vbd_update_color_kernel<<<blocks, threads>>>(
                static_cast<GPU_FEAT10_Data *>(d_data_), d_vbd_solver_,
                color_start, color_count);
          }
          HANDLE_ERROR(cudaGetLastError());
        }
      }

      // Compute and print residual every convergence_check_interval sweeps (like Python)
      // Python: if verbose and (sweep % 5 == 0 or sweep == max_sweeps-1)
      if (h_conv_check_interval_ > 0 &&
          (inner_iter % h_conv_check_interval_ == 0 ||
           inner_iter == h_max_inner_ - 1)) {
        if (type_ == TYPE_T10) {
          int blocks = (n_coef_ + threads - 1) / threads;
          vbd_compute_full_residual<<<blocks, threads>>>(
              static_cast<GPU_FEAT10_Data *>(d_data_), d_vbd_solver_, d_g_);
          HANDLE_ERROR(cudaDeviceSynchronize());
          
          double R_norm = compute_l2_norm_cublas(d_g_, n_dofs);
          std::cout << "    VBD sweep " << std::setw(3) << inner_iter 
                    << ": ||R|| = " << std::scientific << R_norm << std::endl;

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
      int blocks = (n_coef_ + threads - 1) / threads;
      vbd_update_pos_from_vel<<<blocks, threads>>>(
          d_vbd_solver_, static_cast<GPU_FEAT10_Data *>(d_data_));
    }

	    // Compute constraint violation and update multipliers
	    if (n_constraints_ > 0) {
	      int n_fixed = n_constraints_ / 3;
	      int blocks_c = (n_fixed + threads - 1) / threads;

	      if (type_ == TYPE_T10) {
	        vbd_compute_constraint<<<blocks_c, threads>>>(
	            static_cast<GPU_FEAT10_Data *>(d_data_), d_vbd_solver_);
	      }

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
      HANDLE_ERROR(cudaDeviceSynchronize());
    }
  }

  // Update v_prev for next time step
  {
    int blocks = (n_dofs + threads - 1) / threads;
    vbd_update_v_prev<<<blocks, threads>>>(d_vbd_solver_);
  }

  HANDLE_ERROR(cudaEventRecord(stop));
  HANDLE_ERROR(cudaDeviceSynchronize());

  float milliseconds = 0.0f;
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "OneStepVBD kernel time: " << milliseconds << " ms" << std::endl;

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
}
