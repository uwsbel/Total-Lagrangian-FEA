#pragma once
/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    FEAT10DataFunc.cuh
 * Brief:   Defines CUDA device utilities and kernels for FEAT10 elements,
 *          including shape function evaluation, quadrature integration,
 *          internal force and stress computation, and element contribution
 *          assembly into global mass and stiffness structures.
 *==============================================================
 *==============================================================*/

#include <cuda_runtime.h>

#include <cmath>

#include "FEAT10Data.cuh"
#include "../materials/SVK.cuh"
#include "../materials/MooneyRivlin.cuh"

// Forward declaration for device helper templates.
struct SyncedNewtonSolver;

// Solve 3x3 linear system: A * x = b
// A: 3x3 coefficient matrix (row-major)
// b: right-hand side vector
// x: solution vector (output)
__device__ __forceinline__ void solve_3x3_system(double A[3][3], double b[3],
                                                 double x[3]) {
  // Create augmented matrix [A|b] for Gaussian elimination

  double aug[3][4];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      aug[i][j] = A[i][j];
    }
    aug[i][3] = b[i];
  }

  // Forward elimination with partial pivoting
  for (int k = 0; k < 3; k++) {
    // Find pivot (largest element in column k)
    int pivot_row  = k;
    double max_val = fabs(aug[k][k]);
    for (int i = k + 1; i < 3; i++) {
      if (fabs(aug[i][k]) > max_val) {
        max_val   = fabs(aug[i][k]);
        pivot_row = i;
      }
    }

    // Swap rows if needed
    if (pivot_row != k) {
      for (int j = 0; j < 4; j++) {
        double temp       = aug[k][j];
        aug[k][j]         = aug[pivot_row][j];
        aug[pivot_row][j] = temp;
      }
    }

    // Check for singular matrix
    if (fabs(aug[k][k]) < 1e-14) {
      // Handle singular case - set solution to zero
      x[0] = x[1] = x[2] = 0.0;
      return;
    }

    // Eliminate column k in rows below
    for (int i = k + 1; i < 3; i++) {
      double factor = aug[i][k] / aug[k][k];
      for (int j = k; j < 4; j++) {
        aug[i][j] -= factor * aug[k][j];
      }
    }
  }

  // Back substitution
  x[2] = aug[2][3] / aug[2][2];
  x[1] = (aug[1][3] - aug[1][2] * x[2]) / aug[1][1];
  x[0] = (aug[0][3] - aug[0][2] * x[2] - aug[0][1] * x[1]) / aug[0][0];
}

__device__ __forceinline__ void compute_p(int elem_idx, int qp_idx,
                                          GPU_FEAT10_Data* d_data,
                                          const double* __restrict__ v_guess,
                                          double dt) {
  // Get current nodal positions for this element
  double x_nodes[10][3];  // 10 nodes × 3 coordinates

  // clang-format off

  #pragma unroll
  for (int node = 0; node < 10; node++) {
    int global_node_idx = d_data->element_connectivity()(elem_idx, node);
    x_nodes[node][0]    = d_data->x12()(global_node_idx);  // x coordinate
    x_nodes[node][1]    = d_data->y12()(global_node_idx);  // y coordinate
    x_nodes[node][2]    = d_data->z12()(global_node_idx);  // z coordinate
  }

  // Get precomputed shape function gradients for this element and QP
  // grad_N[a][i] = ∂N_a/∂x_i (physical coordinates)
  double grad_N[10][3];
  #pragma unroll
  for (int a = 0; a < 10; a++) {
    grad_N[a][0] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 0);  // ∂N_a/∂x
    grad_N[a][1] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 1);  // ∂N_a/∂y
    grad_N[a][2] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 2);  // ∂N_a/∂z
  }

  // Initialize deformation gradient F to zero
  double F[3][3] = {{0.0}};

  // Compute F = sum_a (x_nodes[a] ⊗ grad_N[a])
  // F[i][j] = sum_a (x_nodes[a][i] * grad_N[a][j])
  #pragma unroll
  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {    // Current position components
      for (int j = 0; j < 3; j++) {  // Gradient components
        F[i][j] += x_nodes[a][i] * grad_N[a][j];
      }
    }
  }

  // Store deformation gradient
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      d_data->F(elem_idx, qp_idx)(i, j) = F[i][j];
    }
  }

  double eta = d_data->eta_damp();
  double lambda_d = d_data->lambda_damp();
  const bool do_damp = (v_guess != nullptr) && (eta != 0.0 || lambda_d != 0.0);

  double P_vis[3][3] = {{0.0}};
  if (do_damp) {
    // Compute Fdot = sum_a (v_nodes[a] ⊗ grad_N[a])
    double Fdot[3][3] = {{0.0}};
    #pragma unroll
    for (int a = 0; a < 10; a++) {
      double v_a[3] = {0.0, 0.0, 0.0};
      int global_node_idx = d_data->element_connectivity()(elem_idx, a);
      v_a[0] = v_guess[global_node_idx * 3 + 0];
      v_a[1] = v_guess[global_node_idx * 3 + 1];
      v_a[2] = v_guess[global_node_idx * 3 + 2];
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          Fdot[i][j] += v_a[i] * grad_N[a][j];
        }
      }
    }

    // Compute viscous Piola: P_vis = F * S_vis, where
    // S_vis = 2*eta*Edot + lambda_damp*tr(Edot)*I
    double Edot[3][3] = {{0.0}};
    // Edot = 0.5*(Fdot^T * F + F^T * Fdot)
    double Ft[3][3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        Ft[i][j] = F[j][i];
      }
    }
    // compute Fdot^T * F
    double FdotT_F[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          FdotT_F[i][j] += Fdot[k][i] * F[k][j];
        }
      }
    }
    // compute F^T * Fdot
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

    double S_vis[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        S_vis[i][j] = 2.0 * eta * Edot[i][j] +
                      lambda_d * trEdot * (i == j ? 1.0 : 0.0);
      }
    }

    // P_vis = F * S_vis
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
  } else {
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        d_data->Fdot(elem_idx, qp_idx)(i, j) = 0.0;
        d_data->P_vis(elem_idx, qp_idx)(i, j) = 0.0;
      }
    }
  }

  // Compute F^T * F
  double FtF[3][3] = {{0.0}};
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      #pragma unroll
      for (int k = 0; k < 3; k++) {
        FtF[i][j] += F[k][i] * F[k][j];  // F^T[i][k] * F[k][j]
      }
    }
  }

  // Compute trace(F^T * F)
  double trFtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

  // Compute F * F^T
  double FFt[3][3] = {{0.0}};
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      #pragma unroll
      for (int k = 0; k < 3; k++) {
        FFt[i][j] += F[i][k] * F[j][k];  // F[i][k] * F^T[k][j]
      }
    }
  }

  // Compute F * F^T * F
  double FFtF[3][3] = {{0.0}};
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FFtF[i][j] += FFt[i][k] * F[k][j];
      }
    }
  }

  double P_el[3][3];
  if (d_data->material_model() == MATERIAL_MODEL_MOONEY_RIVLIN) {
    mr_compute_P(F, d_data->mu10(), d_data->mu01(), d_data->kappa(), P_el);
  } else {
    // Get material parameters
    double lambda = d_data->lambda();
    double mu     = d_data->mu();
    svk_compute_P_from_trFtF_and_FFtF(F, trFtF, FFtF, lambda, mu, P_el);
  }

  #pragma unroll
  for (int i = 0; i < 3; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      // total P = elastic + viscous
      d_data->P(elem_idx, qp_idx)(i, j) = P_el[i][j] + P_vis[i][j];
    }
  }
  // clang-format on
}

// Template version for compile-time material model dispatch
template <int Mat>
__device__ __forceinline__ void compute_p_impl(int elem_idx, int qp_idx,
                                               GPU_FEAT10_Data* d_data,
                                               const double* __restrict__ v_guess,
                                               double dt) {
  // Get current nodal positions for this element
  double x_nodes[10][3];  // 10 nodes × 3 coordinates

  // clang-format off

  #pragma unroll
  for (int node = 0; node < 10; node++) {
    int global_node_idx = d_data->element_connectivity()(elem_idx, node);
    x_nodes[node][0]    = d_data->x12()(global_node_idx);  // x coordinate
    x_nodes[node][1]    = d_data->y12()(global_node_idx);  // y coordinate
    x_nodes[node][2]    = d_data->z12()(global_node_idx);  // z coordinate
  }

  // Get precomputed shape function gradients for this element and QP
  // grad_N[a][i] = ∂N_a/∂x_i (physical coordinates)
  double grad_N[10][3];
  #pragma unroll
  for (int a = 0; a < 10; a++) {
    grad_N[a][0] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 0);  // ∂N_a/∂x
    grad_N[a][1] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 1);  // ∂N_a/∂y
    grad_N[a][2] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 2);  // ∂N_a/∂z
  }

  // Initialize deformation gradient F to zero
  double F[3][3] = {{0.0}};

  // Compute F = sum_a (x_nodes[a] ⊗ grad_N[a])
  // F[i][j] = sum_a (x_nodes[a][i] * grad_N[a][j])
  #pragma unroll
  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {    // Current position components
      for (int j = 0; j < 3; j++) {  // Gradient components
        F[i][j] += x_nodes[a][i] * grad_N[a][j];
      }
    }
  }

  // Store deformation gradient
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      d_data->F(elem_idx, qp_idx)(i, j) = F[i][j];
    }
  }

  double eta = d_data->eta_damp();
  double lambda_d = d_data->lambda_damp();
  const bool do_damp = (v_guess != nullptr) && (eta != 0.0 || lambda_d != 0.0);

  double P_vis[3][3] = {{0.0}};
  if (do_damp) {
    // Compute Fdot = sum_a (v_nodes[a] ⊗ grad_N[a])
    double Fdot[3][3] = {{0.0}};
    #pragma unroll
    for (int a = 0; a < 10; a++) {
      double v_a[3] = {0.0, 0.0, 0.0};
      int global_node_idx = d_data->element_connectivity()(elem_idx, a);
      v_a[0] = v_guess[global_node_idx * 3 + 0];
      v_a[1] = v_guess[global_node_idx * 3 + 1];
      v_a[2] = v_guess[global_node_idx * 3 + 2];
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          Fdot[i][j] += v_a[i] * grad_N[a][j];
        }
      }
    }

    // Compute viscous Piola: P_vis = F * S_vis, where
    // S_vis = 2*eta*Edot + lambda_damp*tr(Edot)*I
    double Edot[3][3] = {{0.0}};
    // Edot = 0.5*(Fdot^T * F + F^T * Fdot)
    double Ft[3][3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        Ft[i][j] = F[j][i];
      }
    }
    // compute Fdot^T * F
    double FdotT_F[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          FdotT_F[i][j] += Fdot[k][i] * F[k][j];
        }
      }
    }
    // compute F^T * Fdot
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

    double S_vis[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        S_vis[i][j] = 2.0 * eta * Edot[i][j] +
                      lambda_d * trEdot * (i == j ? 1.0 : 0.0);
      }
    }

    // P_vis = F * S_vis
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
  } else {
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        d_data->Fdot(elem_idx, qp_idx)(i, j) = 0.0;
        d_data->P_vis(elem_idx, qp_idx)(i, j) = 0.0;
      }
    }
  }

  // Compute F^T * F
  double FtF[3][3] = {{0.0}};
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      #pragma unroll
      for (int k = 0; k < 3; k++) {
        FtF[i][j] += F[k][i] * F[k][j];  // F^T[i][k] * F[k][j]
      }
    }
  }

  // Compute trace(F^T * F)
  double trFtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

  // Compute F * F^T
  double FFt[3][3] = {{0.0}};
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      #pragma unroll
      for (int k = 0; k < 3; k++) {
        FFt[i][j] += F[i][k] * F[j][k];  // F[i][k] * F^T[k][j]
      }
    }
  }

  // Compute F * F^T * F
  double FFtF[3][3] = {{0.0}};
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FFtF[i][j] += FFt[i][k] * F[k][j];
      }
    }
  }

  // Compile-time material model dispatch
  double P_el[3][3];
  if constexpr (Mat == MATERIAL_MODEL_MOONEY_RIVLIN) {
    mr_compute_P(F, d_data->mu10(), d_data->mu01(), d_data->kappa(), P_el);
  } else {
    // Get material parameters
    double lambda = d_data->lambda();
    double mu     = d_data->mu();
    svk_compute_P_from_trFtF_and_FFtF(F, trFtF, FFtF, lambda, mu, P_el);
  }

  #pragma unroll
  for (int i = 0; i < 3; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      // total P = elastic + viscous
      d_data->P(elem_idx, qp_idx)(i, j) = P_el[i][j] + P_vis[i][j];
    }
  }
  // clang-format on
}

__device__ __forceinline__ void vbd_accumulate_residual_and_hessian_diag(
    int elem_idx, int qp_idx, int local_node, GPU_FEAT10_Data* d_data,
    double dt, double& r0, double& r1, double& r2, double& h00, double& h01,
    double& h02, double& h10, double& h11, double& h12, double& h20,
    double& h21, double& h22) {
  const double ha0 = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 0);
  const double ha1 = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 1);
  const double ha2 = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 2);

  const double detJ = d_data->detJ_ref(elem_idx, qp_idx);
  const double wq   = d_data->tet5pt_weights(qp_idx);
  const double dV   = detJ * wq;

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
  const double trE = 0.5 * (trFtF - 3.0);

  const double FFT00 = F00 * F00 + F01 * F01 + F02 * F02;
  const double FFT01 = F00 * F10 + F01 * F11 + F02 * F12;
  const double FFT02 = F00 * F20 + F01 * F21 + F02 * F22;
  const double FFT10 = FFT01;
  const double FFT11 = F10 * F10 + F11 * F11 + F12 * F12;
  const double FFT12 = F10 * F20 + F11 * F21 + F12 * F22;
  const double FFT20 = FFT02;
  const double FFT21 = FFT12;
  const double FFT22 = F20 * F20 + F21 * F21 + F22 * F22;

  const double Fh0 = F00 * ha0 + F01 * ha1 + F02 * ha2;
  const double Fh1 = F10 * ha0 + F11 * ha1 + F12 * ha2;
  const double Fh2 = F20 * ha0 + F21 * ha1 + F22 * ha2;

  const double hij        = ha0 * ha0 + ha1 * ha1 + ha2 * ha2;
  const double Fh_dot_Fh  = Fh0 * Fh0 + Fh1 * Fh1 + Fh2 * Fh2;
  const double weight_k   = dt * dV;

  double Kblock[3][3];
  if (d_data->material_model() == MATERIAL_MODEL_MOONEY_RIVLIN) {
    double F_local[3][3] = {{F00, F01, F02}, {F10, F11, F12}, {F20, F21, F22}};
    double A_mr[3][3][3][3];
    mr_compute_tangent_tensor(F_local, d_data->mu10(), d_data->mu01(),
                              d_data->kappa(), A_mr);
#pragma unroll
    for (int d = 0; d < 3; ++d) {
#pragma unroll
      for (int e = 0; e < 3; ++e) {
        double sum = 0.0;
#pragma unroll
        for (int J = 0; J < 3; ++J) {
#pragma unroll
          for (int L = 0; L < 3; ++L) {
            const double giJ = (J == 0 ? ha0 : (J == 1 ? ha1 : ha2));
            const double giL = (L == 0 ? ha0 : (L == 1 ? ha1 : ha2));
            sum += A_mr[d][J][e][L] * giJ * giL;
          }
        }
        Kblock[d][e] = sum * weight_k;
      }
    }
  } else {
    const double Fh_vec[3] = {Fh0, Fh1, Fh2};
    const double FFT[3][3] = {{FFT00, FFT01, FFT02},
                              {FFT10, FFT11, FFT12},
                              {FFT20, FFT21, FFT22}};
    svk_compute_tangent_block(Fh_vec, Fh_vec, hij, trE, Fh_dot_Fh, FFT,
                             d_data->lambda(), d_data->mu(), weight_k, Kblock);
  }

  h00 += Kblock[0][0];
  h01 += Kblock[0][1];
  h02 += Kblock[0][2];
  h10 += Kblock[1][0];
  h11 += Kblock[1][1];
  h12 += Kblock[1][2];
  h20 += Kblock[2][0];
  h21 += Kblock[2][1];
  h22 += Kblock[2][2];
}

__device__ __forceinline__ void compute_internal_force(
    int elem_idx, int node_local, GPU_FEAT10_Data* d_data) {
  // Get global node index for this local node
  int global_node_idx = d_data->element_connectivity()(elem_idx, node_local);

  // Initialize force accumulator for this node (3 components)
  double f_node[3] = {0.0, 0.0, 0.0};

  // clang-format off

  // Loop over all quadrature points for this element
  #pragma unroll
  for (int qp_idx = 0; qp_idx < Quadrature::N_QP_T10_5; qp_idx++) {
    // Get precomputed P matrix for this element and quadrature point
    // P is the 3x3 first Piola-Kirchhoff stress tensor
    double P[3][3];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        P[i][j] = d_data->P(elem_idx, qp_idx)(i, j);
      }
    }

    // Get precomputed shape function gradients for this node at this QP
    // grad_N[node_local] = [∂N/∂x, ∂N/∂y, ∂N/∂z] (physical coordinates)
    double grad_N[3];
    grad_N[0] = d_data->grad_N_ref(elem_idx, qp_idx)(node_local, 0);  // ∂N/∂x
    grad_N[1] = d_data->grad_N_ref(elem_idx, qp_idx)(node_local, 1);  // ∂N/∂y
    grad_N[2] = d_data->grad_N_ref(elem_idx, qp_idx)(node_local, 2);  // ∂N/∂z

    // Get determinant and quadrature weight
    double detJ = d_data->detJ_ref(elem_idx, qp_idx);
    double wq   = d_data->tet5pt_weights(qp_idx);
    double dV   = detJ * wq;

    // Compute P @ grad_N (matrix-vector multiply)
    // f_contribution[i] = sum_j(P[i][j] * grad_N[j])
    double f_contribution[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      f_contribution[i] = 0.0;
      for (int j = 0; j < 3; j++) {
        f_contribution[i] += P[i][j] * grad_N[j];
      }
    }

    // Accumulate: f[node_local] += (P @ grad_N[node_local]) * dV
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      f_node[i] += f_contribution[i] * dV;
    }
  }

  // Assemble into global internal force vector using atomic operations
  // Each thread handles one node, so we need atomicAdd for thread safety
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    int global_dof_idx = 3 * global_node_idx + i;
    atomicAdd(&(d_data->f_int()(global_dof_idx)), f_node[i]);
  }

  // clang-format on
}

__device__ __forceinline__ void clear_internal_force(GPU_FEAT10_Data* d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < d_data->n_coef * 3) {
    d_data->f_int()[thread_idx] = 0.0;
  }
}

__device__ __forceinline__ void compute_constraint_data(
    GPU_FEAT10_Data* d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < d_data->gpu_n_constraint() / 3) {
    d_data->constraint()[thread_idx * 3 + 0] =
        d_data->x12()(d_data->fixed_nodes()[thread_idx]) -
        d_data->x12_jac()(d_data->fixed_nodes()[thread_idx]);
    d_data->constraint()[thread_idx * 3 + 1] =
        d_data->y12()(d_data->fixed_nodes()[thread_idx]) -
        d_data->y12_jac()(d_data->fixed_nodes()[thread_idx]);
    d_data->constraint()[thread_idx * 3 + 2] =
        d_data->z12()(d_data->fixed_nodes()[thread_idx]) -
        d_data->z12_jac()(d_data->fixed_nodes()[thread_idx]);
  }
}

// --- CSR-version Hessian assembly for FEAT10 ---
// Local binary-search helper (self-contained)
static __device__ __forceinline__ int binary_search_column_csr(const int* cols,
                                                               int n_cols,
                                                               int target) {
  int left = 0, right = n_cols - 1;
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

// Template declaration
template <typename ElementType>
__device__ __forceinline__ void compute_hessian_assemble_csr(
    ElementType* d_data, SyncedNewtonSolver* d_solver, int elem_idx, int qp_idx,
    int* d_csr_row_offsets, int* d_csr_col_indices, double* d_csr_values,
    double h);

// Explicit specialization for FEAT10
template <>
__device__ __forceinline__ void compute_hessian_assemble_csr<GPU_FEAT10_Data>(
    GPU_FEAT10_Data* d_data, SyncedNewtonSolver* d_solver, int elem_idx,
    int qp_idx, int* d_csr_row_offsets, int* d_csr_col_indices,
    double* d_csr_values, double h) {
  // Reuse all local computations from compute_hessian_assemble and then
  // scatter by searching the CSR row for each (global_row, global_col).

  // Get element connectivity
  int global_node_indices[10];
#pragma unroll
  for (int node = 0; node < 10; node++) {
    global_node_indices[node] = d_data->element_connectivity()(elem_idx, node);
  }

  // Read current nodal positions
  double x_nodes[10][3];
#pragma unroll
  for (int node = 0; node < 10; node++) {
    int gn           = global_node_indices[node];
    x_nodes[node][0] = d_data->x12()(gn);
    x_nodes[node][1] = d_data->y12()(gn);
    x_nodes[node][2] = d_data->z12()(gn);
  }

  // grad_N
  double grad_N[10][3];
#pragma unroll
  for (int a = 0; a < 10; a++) {
    grad_N[a][0] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 0);
    grad_N[a][1] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 1);
    grad_N[a][2] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 2);
  }

  // Compute F, C, FFT, Fh etc. (same math as existing compute_hessian_assemble)
  double F[3][3] = {{0.0}};
#pragma unroll
  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        F[i][j] += x_nodes[a][i] * grad_N[a][j];
      }
    }
  }

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

  double trC = C[0][0] + C[1][1] + C[2][2];
  double trE = 0.5 * (trC - 3.0);

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

  double lambda = d_data->lambda();
  double mu     = d_data->mu();
  double detJ   = d_data->detJ_ref(elem_idx, qp_idx);
  double wq     = d_data->tet5pt_weights(qp_idx);
  double dV     = detJ * wq;

  const bool use_mr = (d_data->material_model() == MATERIAL_MODEL_MOONEY_RIVLIN);
  double A_mr[3][3][3][3];
  if (use_mr) {
    mr_compute_tangent_tensor(F, d_data->mu10(), d_data->mu01(), d_data->kappa(),
                              A_mr);
  }

  // Local K_elem 30x30
  double K_elem[30][30];
#pragma unroll
  for (int ii = 0; ii < 30; ii++)
    for (int jj = 0; jj < 30; jj++)
      K_elem[ii][jj] = 0.0;

#pragma unroll
  for (int i = 0; i < 10; i++) {
#pragma unroll
    for (int j = 0; j < 10; j++) {
      double hij = grad_N[j][0] * grad_N[i][0] + grad_N[j][1] * grad_N[i][1] +
                   grad_N[j][2] * grad_N[i][2];
      double Fhj_dot_Fhi =
          Fh[j][0] * Fh[i][0] + Fh[j][1] * Fh[i][1] + Fh[j][2] * Fh[i][2];

      double Kblock[3][3];
      if (use_mr) {
#pragma unroll
        for (int d = 0; d < 3; d++) {
#pragma unroll
          for (int e = 0; e < 3; e++) {
            double sum = 0.0;
#pragma unroll
            for (int J = 0; J < 3; J++) {
#pragma unroll
              for (int L = 0; L < 3; L++) {
                sum += A_mr[d][J][e][L] * grad_N[i][J] * grad_N[j][L];
              }
            }
            Kblock[d][e] = sum * dV;
          }
        }
      } else {
        svk_compute_tangent_block(Fh[i], Fh[j], hij, trE, Fhj_dot_Fhi, FFT,
                                 lambda, mu, dV, Kblock);
      }

#pragma unroll
      for (int d = 0; d < 3; d++) {
#pragma unroll
        for (int e = 0; e < 3; e++) {
          int row          = 3 * i + d;
          int col          = 3 * j + e;
          K_elem[row][col] = Kblock[d][e];
        }
      }
    }
  }

  // Scatter to CSR
  for (int local_row_node = 0; local_row_node < 10; local_row_node++) {
    int global_node_row = global_node_indices[local_row_node];
    for (int r_dof = 0; r_dof < 3; r_dof++) {
      int global_row = 3 * global_node_row + r_dof;
      int local_row  = 3 * local_row_node + r_dof;

      int row_begin = d_csr_row_offsets[global_row];
      int row_end   = d_csr_row_offsets[global_row + 1];
      int row_len   = row_end - row_begin;

      for (int local_col_node = 0; local_col_node < 10; local_col_node++) {
        int global_node_col = global_node_indices[local_col_node];
        for (int c_dof = 0; c_dof < 3; c_dof++) {
          int global_col = 3 * global_node_col + c_dof;
          int local_col  = 3 * local_col_node + c_dof;

          int pos = binary_search_column_csr(&d_csr_col_indices[row_begin],
                                             row_len, global_col);
          if (pos >= 0) {
            atomicAdd(&d_csr_values[row_begin + pos],
                      h * K_elem[local_row][local_col]);
          }
        }
      }
    }
  }

  double eta_d    = d_data->eta_damp();
  double lambda_d = d_data->lambda_damp();
  if (eta_d == 0.0 && lambda_d == 0.0) {
    return;
  }

  // --- Viscous tangent (Kelvin-Voigt) assembly: C_elem (30x30) ---
  // C_ab = (eta * outer(Fh_b, Fh_a) + eta * FFT * (h_a·h_b) + lambda_d *
  // outer(Fh_a, Fh_b)) * dV
  double C_elem[30][30];
#pragma unroll
  for (int ii = 0; ii < 30; ii++)
    for (int jj = 0; jj < 30; jj++)
      C_elem[ii][jj] = 0.0;

#pragma unroll
  for (int a = 0; a < 10; a++) {
    double* h_a  = grad_N[a];
    double Fh_a0 = Fh[a][0];
    double Fh_a1 = Fh[a][1];
    double Fh_a2 = Fh[a][2];
#pragma unroll
    for (int b = 0; b < 10; b++) {
      double* h_b  = grad_N[b];
      double Fh_b0 = Fh[b][0];
      double Fh_b1 = Fh[b][1];
      double Fh_b2 = Fh[b][2];

      double hdot = h_a[0] * h_b[0] + h_a[1] * h_b[1] + h_a[2] * h_b[2];

      // Compute block C_ab (3x3)
      double Cblock00 = (eta_d * (Fh_b0 * Fh_a0) + eta_d * FFT[0][0] * hdot +
                         lambda_d * (Fh_a0 * Fh_b0)) *
                        dV;
      double Cblock01 = (eta_d * (Fh_b0 * Fh_a1) + eta_d * FFT[0][1] * hdot +
                         lambda_d * (Fh_a0 * Fh_b1)) *
                        dV;
      double Cblock02 = (eta_d * (Fh_b0 * Fh_a2) + eta_d * FFT[0][2] * hdot +
                         lambda_d * (Fh_a0 * Fh_b2)) *
                        dV;

      double Cblock10 = (eta_d * (Fh_b1 * Fh_a0) + eta_d * FFT[1][0] * hdot +
                         lambda_d * (Fh_a1 * Fh_b0)) *
                        dV;
      double Cblock11 = (eta_d * (Fh_b1 * Fh_a1) + eta_d * FFT[1][1] * hdot +
                         lambda_d * (Fh_a1 * Fh_b1)) *
                        dV;
      double Cblock12 = (eta_d * (Fh_b1 * Fh_a2) + eta_d * FFT[1][2] * hdot +
                         lambda_d * (Fh_a1 * Fh_b2)) *
                        dV;

      double Cblock20 = (eta_d * (Fh_b2 * Fh_a0) + eta_d * FFT[2][0] * hdot +
                         lambda_d * (Fh_a2 * Fh_b0)) *
                        dV;
      double Cblock21 = (eta_d * (Fh_b2 * Fh_a1) + eta_d * FFT[2][1] * hdot +
                         lambda_d * (Fh_a2 * Fh_b1)) *
                        dV;
      double Cblock22 = (eta_d * (Fh_b2 * Fh_a2) + eta_d * FFT[2][2] * hdot +
                         lambda_d * (Fh_a2 * Fh_b2)) *
                        dV;

      int row0                   = 3 * a;
      int col0                   = 3 * b;
      C_elem[row0 + 0][col0 + 0] = Cblock00;
      C_elem[row0 + 0][col0 + 1] = Cblock01;
      C_elem[row0 + 0][col0 + 2] = Cblock02;
      C_elem[row0 + 1][col0 + 0] = Cblock10;
      C_elem[row0 + 1][col0 + 1] = Cblock11;
      C_elem[row0 + 1][col0 + 2] = Cblock12;
      C_elem[row0 + 2][col0 + 0] = Cblock20;
      C_elem[row0 + 2][col0 + 1] = Cblock21;
      C_elem[row0 + 2][col0 + 2] = Cblock22;
    }
  }

  // Scatter viscous C_elem to CSR (no h scaling)
  for (int local_row_node = 0; local_row_node < 10; local_row_node++) {
    int global_node_row = global_node_indices[local_row_node];
    for (int r_dof = 0; r_dof < 3; r_dof++) {
      int global_row = 3 * global_node_row + r_dof;
      int local_row  = 3 * local_row_node + r_dof;

      int row_begin = d_csr_row_offsets[global_row];
      int row_end   = d_csr_row_offsets[global_row + 1];
      int row_len   = row_end - row_begin;

      for (int local_col_node = 0; local_col_node < 10; local_col_node++) {
        int global_node_col = global_node_indices[local_col_node];
        for (int c_dof = 0; c_dof < 3; c_dof++) {
          int global_col = 3 * global_node_col + c_dof;
          int local_col  = 3 * local_col_node + c_dof;

          int pos = binary_search_column_csr(&d_csr_col_indices[row_begin],
                                             row_len, global_col);
          if (pos >= 0) {
            atomicAdd(&d_csr_values[row_begin + pos],
                      C_elem[local_row][local_col]);
          }
        }
      }
    }
  }
}
