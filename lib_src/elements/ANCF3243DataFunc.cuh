#pragma once
/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    ANCF3243DataFunc.cuh
 * Brief:   Provides CUDA device functions and kernels used by ANCF 3243 beam
 *          elements, including shape function evaluation, quadrature loops,
 *          internal force and stress computation, and element-level assembly
 *          helpers.
 *==============================================================
 *==============================================================*/

#include "ANCF3243Data.cuh"
#include "../materials/SVK.cuh"
#include "../materials/MooneyRivlin.cuh"

// forward-declare solver type (pointer-only used here)
struct SyncedNewtonSolver;

__device__ __forceinline__ void compute_p(int, int, GPU_ANCF3243_Data *,
                                          const double *, double);
__device__ __forceinline__ void compute_internal_force(int, int,
                                                       GPU_ANCF3243_Data *);
__device__ __forceinline__ void compute_constraint_data(GPU_ANCF3243_Data *);

// Device function: matrix-vector multiply (8x8 * 8x1)
__device__ __forceinline__ void ancf3243_mat_vec_mul8(
    Eigen::Map<Eigen::MatrixXd> A, const double *x, double *out) {
  // clang-format off
    #pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3243; ++i)
    {
        out[i] = 0.0;
        #pragma unroll
        for (int j = 0; j < Quadrature::N_SHAPE_3243; ++j)
        {
            out[i] += A(i, j) * x[j];
        }
    }
  // clang-format on
}

// Device function to compute determinant of 3x3 matrix
__device__ __forceinline__ double ancf3243_det3x3(const double *J) {
  return J[0] * (J[4] * J[8] - J[5] * J[7]) -
         J[1] * (J[3] * J[8] - J[5] * J[6]) +
         J[2] * (J[3] * J[7] - J[4] * J[6]);
}

__device__ __forceinline__ void ancf3243_b_vec(double u, double v, double w,
                                               double *out) {
  out[0] = 1.0;
  out[1] = u;
  out[2] = v;
  out[3] = w;
  out[4] = u * v;
  out[5] = u * w;
  out[6] = u * u;
  out[7] = u * u * u;
}

__device__ __forceinline__ void ancf3243_b_vec_xi(double xi, double eta,
                                                  double zeta, double L,
                                                  double W, double H,
                                                  double *out) {
  double u = L * xi / 2.0;
  double v = W * eta / 2.0;
  double w = H * zeta / 2.0;
  ancf3243_b_vec(u, v, w, out);
}

// Device function for Jacobian determinant in normalized coordinates
__device__ __forceinline__ void ancf3243_calc_det_J_xi(
    double xi, double eta, double zeta, Eigen::Map<Eigen::MatrixXd> B_inv,
    Eigen::Map<Eigen::VectorXd> x12_jac, Eigen::Map<Eigen::VectorXd> y12_jac,
    Eigen::Map<Eigen::VectorXd> z12_jac, double L, double W, double H,
    double *J_out) {
  double db_dxi[Quadrature::N_SHAPE_3243]  = {0.0,
                                              L / 2,
                                              0.0,
                                              0.0,
                                              (L * W / 4) * eta,
                                              (L * H / 4) * zeta,
                                              (L * L / 2) * xi,
                                              (3 * L * L * L / 8) * xi * xi};
  double db_deta[Quadrature::N_SHAPE_3243] = {
      0.0, 0.0, W / 2, 0.0, (L * W / 4) * xi, 0.0, 0.0, 0.0};
  double db_dzeta[Quadrature::N_SHAPE_3243] = {
      0.0, 0.0, 0.0, H / 2, 0.0, (L * H / 4) * xi, 0.0, 0.0};

  double ds_dxi[Quadrature::N_SHAPE_3243], ds_deta[Quadrature::N_SHAPE_3243],
      ds_dzeta[Quadrature::N_SHAPE_3243];
  ancf3243_mat_vec_mul8(B_inv, db_dxi, ds_dxi);
  ancf3243_mat_vec_mul8(B_inv, db_deta, ds_deta);
  ancf3243_mat_vec_mul8(B_inv, db_dzeta, ds_dzeta);

  // Nodal matrix: 3 × 8
  // J = N_mat_jac @ np.column_stack([ds_dxi, ds_deta, ds_dzeta])

  // clang-format off
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            J_out[i * 3 + j] = 0.0;

    #pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3243; ++i)
    {
        J_out[0 * 3 + 0] += x12_jac(i) * ds_dxi[i];
        J_out[1 * 3 + 0] += y12_jac(i) * ds_dxi[i];
        J_out[2 * 3 + 0] += z12_jac(i) * ds_dxi[i];

        J_out[0 * 3 + 1] += x12_jac(i) * ds_deta[i];
        J_out[1 * 3 + 1] += y12_jac(i) * ds_deta[i];
        J_out[2 * 3 + 1] += z12_jac(i) * ds_deta[i];

        J_out[0 * 3 + 2] += x12_jac(i) * ds_dzeta[i];
        J_out[1 * 3 + 2] += y12_jac(i) * ds_dzeta[i];
        J_out[2 * 3 + 2] += z12_jac(i) * ds_dzeta[i];
    }
  // clang-format on
}

__device__ __forceinline__ void compute_p(int elem_idx, int qp_idx,
                                          GPU_ANCF3243_Data *d_data,
                                          const double *__restrict__ v_guess,
                                          double dt) {
  // clang-format off
    // --- Compute C = F^T * F ---

    // Initialize F to zero
    #pragma unroll
    for (int i = 0; i < 3; i++)
    {
        #pragma unroll
        for (int j = 0; j < 3; j++)
        {
            d_data->F(elem_idx, qp_idx)(i, j) = 0.0;
        }
    }

    // Extract local nodal coordinates (e vectors) using element connectivity
    double e[8][3]; // 2 nodes × 4 DOFs = 8 entries
    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        const int node_local  = (i < 4) ? 0 : 1;
        const int dof_local   = i % 4;
        const int node_global = d_data->element_node(elem_idx, node_local);
        const int coef_idx    = node_global * 4 + dof_local;

        e[i][0] = d_data->x12()(coef_idx); // x coordinate
        e[i][1] = d_data->y12()(coef_idx); // y coordinate
        e[i][2] = d_data->z12()(coef_idx); // z coordinate
    }

    // Compute F = sum_i e_i ⊗ ∇s_i
    // F is 3x3 matrix stored in row-major order
    #pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3243; i++)
    { // Loop over nodes
        // Get gradient of shape function i (∇s_i) - this needs proper indexing
        // Assuming ds_du_pre is laid out as [qp_total][8][3]
        // You'll need to provide the correct qp_idx for the current quadrature
        // point
        double grad_s_i[3];
        grad_s_i[0] = d_data->ds_du_pre(qp_idx)(i, 0); // ∂s_i/∂u
        grad_s_i[1] = d_data->ds_du_pre(qp_idx)(i, 1); // ∂s_i/∂v
        grad_s_i[2] = d_data->ds_du_pre(qp_idx)(i, 2); // ∂s_i/∂w

        // Compute outer product: e_i ⊗ ∇s_i and add to F
        #pragma unroll
        for (int row = 0; row < 3; row++)
        { // e_i components
            #pragma unroll
            for (int col = 0; col < 3; col++)
            { // ∇s_i components
                d_data->F(elem_idx, qp_idx)(row, col) +=
                    e[i][row] * grad_s_i[col];
            }
        }
    }

    double FtF[3][3] = {0.0};
    
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j)
            #pragma unroll
            for (int k = 0; k < 3; ++k)
                FtF[i][j] += d_data->F(elem_idx, qp_idx)(k, i) * d_data->F(elem_idx, qp_idx)(k, j);

    // --- trace(F^T F) ---
    double tr_FtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

    // 1. Compute Ft (transpose of F)
    double Ft[3][3] = {0};
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            Ft[i][j] = d_data->F(elem_idx, qp_idx)(j, i); // transpose
        }

    // 2. Compute G = F * Ft
    double G[3][3] = {0}; // G = F * F^T
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j)
            #pragma unroll
            for (int k = 0; k < 3; ++k)
            {
                G[i][j] += d_data->F(elem_idx, qp_idx)(i, k) * Ft[k][j];
            }

    // 3. Compute FFF = G * F = (F * Ft) * F
    double FFF[3][3] = {0};
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j)
            #pragma unroll
            for (int k = 0; k < 3; ++k)
            {
                FFF[i][j] += G[i][k] * d_data->F(elem_idx, qp_idx)(k, j);
            }

    // --- Compute P ---
    double F_local[3][3];
    #pragma unroll
    for (int i = 0; i < 3; ++i)
      #pragma unroll
      for (int j = 0; j < 3; ++j)
        F_local[i][j] = d_data->F(elem_idx, qp_idx)(i, j);

    double P_el[3][3];
    if (d_data->material_model() == MATERIAL_MODEL_MOONEY_RIVLIN) {
      mr_compute_P(F_local, d_data->mu10(), d_data->mu01(), d_data->kappa(),
                   P_el);
    } else {
      svk_compute_P_from_trFtF_and_FFtF(F_local, tr_FtF, FFF, d_data->lambda(),
                                       d_data->mu(), P_el);
    }

    #pragma unroll
    for (int i = 0; i < 3; ++i)
      #pragma unroll
      for (int j = 0; j < 3; ++j)
        d_data->P(elem_idx, qp_idx)(i, j) = P_el[i][j];
 
    double eta = d_data->eta_damp();
    double lambda_d = d_data->lambda_damp();
    const bool do_damp = (v_guess != nullptr) && (eta != 0.0 || lambda_d != 0.0);

    if (do_damp) {
      // Compute Fdot = sum_i v_i ⊗ ∇s_i
      double Fdot[3][3] = {{0.0}};
      #pragma unroll
      for (int i = 0; i < Quadrature::N_SHAPE_3243; i++) {
        double v_i[3] = {0.0};
        // coef index mapping used above
        const int node_local  = (i < 4) ? 0 : 1;
        const int dof_local   = i % 4;
        const int node_global = d_data->element_node(elem_idx, node_local);
        const int coef_idx    = node_global * 4 + dof_local;
        v_i[0] = v_guess[coef_idx * 3 + 0];
        v_i[1] = v_guess[coef_idx * 3 + 1];
        v_i[2] = v_guess[coef_idx * 3 + 2];
        #pragma unroll
        for (int row = 0; row < 3; row++) {
          #pragma unroll
          for (int col = 0; col < 3; col++) {
            // grad_s not stored locally here; read from precomputed ds_du_pre
            double grad_si_col = d_data->ds_du_pre(qp_idx)(i, col);
            Fdot[row][col] += v_i[row] * grad_si_col;
          }
        }
      }

      // Edot = 0.5*(Fdot^T * F + F^T * Fdot)
      double FdotT_F[3][3] = {{0.0}};
      double Ft_Fdot[3][3] = {{0.0}};
      // reuse Ft declared earlier (transpose of F)
      #pragma unroll
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          Ft[i][j] = d_data->F(elem_idx, qp_idx)(j, i);
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
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          Edot[i][j] = 0.5 * (FdotT_F[i][j] + Ft_Fdot[i][j]);

      double trEdot = Edot[0][0] + Edot[1][1] + Edot[2][2];
      double S_vis[3][3] = {{0.0}};
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          S_vis[i][j] = 2.0 * eta * Edot[i][j] +
                        lambda_d * trEdot * (i == j ? 1.0 : 0.0);
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
      // store Fdot and P_vis
      #pragma unroll
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
          d_data->Fdot(elem_idx, qp_idx)(i, j) = Fdot[i][j];
          d_data->P_vis(elem_idx, qp_idx)(i, j) = P_vis[i][j];
        }
      // Add viscous Piola to total Piola so internal force uses elastic + viscous
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        #pragma unroll
        for (int j = 0; j < 3; j++) {
          d_data->P(elem_idx, qp_idx)(i, j) += P_vis[i][j];
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
          d_data->Fdot(elem_idx, qp_idx)(i, j) = 0.0;
          d_data->P_vis(elem_idx, qp_idx)(i, j) = 0.0;
        }
    }
  // clang-format on
}

__device__ __forceinline__ void compute_internal_force(
    int elem_idx, int node_idx, GPU_ANCF3243_Data *d_data) {
  double f_i[3] = {0};
  // Map local node_idx (0-7) to global coefficient index using connectivity
  const int node_local      = node_idx / 4;
  const int dof_local       = node_idx % 4;
  const int node_global     = d_data->element_node(elem_idx, node_local);
  const int coef_idx_global = node_global * 4 + dof_local;
  double geom               = (d_data->L() * d_data->W() * d_data->H()) / 8.0;

  // clang-format off

    #pragma unroll
    for (int qp_idx = 0; qp_idx < Quadrature::N_TOTAL_QP_3_2_2; qp_idx++)
    {
        double grad_s[3];
        grad_s[0] = d_data->ds_du_pre(qp_idx)(node_idx, 0);
        grad_s[1] = d_data->ds_du_pre(qp_idx)(node_idx, 1);
        grad_s[2] = d_data->ds_du_pre(qp_idx)(node_idx, 2);

        double scale = d_data->weight_xi()(qp_idx / (Quadrature::N_QP_2 * Quadrature::N_QP_2)) *
                       d_data->weight_eta()((qp_idx / Quadrature::N_QP_2) % Quadrature::N_QP_2) *
                       d_data->weight_zeta()(qp_idx % Quadrature::N_QP_2);
        #pragma unroll
        for (int r = 0; r < 3; ++r)
        {
            #pragma unroll
            for (int c = 0; c < 3; ++c)
            {
                f_i[r] += (d_data->P(elem_idx, qp_idx)(r, c) * grad_s[c]) * scale * geom;
            }
        }
    }

    #pragma unroll
    for (int d = 0; d < 3; ++d)
    {
        atomicAdd(&d_data->f_int(coef_idx_global)(d), f_i[d]);
    }

  // clang-format on
}

__device__ __forceinline__ void compute_constraint_data(
    GPU_ANCF3243_Data *d_data) {
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

__device__ __forceinline__ void vbd_accumulate_residual_and_hessian_diag(
    int elem_idx, int qp_idx, int local_node, GPU_ANCF3243_Data *d_data,
    double dt, double &r0, double &r1, double &r2, double &h00, double &h01,
    double &h02, double &h10, double &h11, double &h12, double &h20,
    double &h21, double &h22) {
  const double ha0 = d_data->ds_du_pre(qp_idx)(local_node, 0);
  const double ha1 = d_data->ds_du_pre(qp_idx)(local_node, 1);
  const double ha2 = d_data->ds_du_pre(qp_idx)(local_node, 2);

  const double geom = (d_data->L() * d_data->W() * d_data->H()) / 8.0;
  const double scale =
      d_data->weight_xi()(qp_idx / (Quadrature::N_QP_2 * Quadrature::N_QP_2)) *
      d_data->weight_eta()((qp_idx / Quadrature::N_QP_2) % Quadrature::N_QP_2) *
      d_data->weight_zeta()(qp_idx % Quadrature::N_QP_2);
  const double dV = scale * geom;

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

__device__ __forceinline__ void clear_internal_force(
    GPU_ANCF3243_Data *d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < d_data->n_coef * 3) {
    d_data->f_int()[thread_idx] = 0.0;
  }
}

// --- CSR-version Hessian assembly for ANCF3243 ---
static __device__ __forceinline__ int binary_search_column_csr_3243(
    const int *cols, int n_cols, int target) {
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

template <typename ElementType>
__device__ __forceinline__ void compute_hessian_assemble_csr(
    ElementType *d_data, SyncedNewtonSolver *d_solver, int elem_idx, int qp_idx,
    int *d_csr_row_offsets, int *d_csr_col_indices, double *d_csr_values,
    double h);

// Explicit specialization for GPU_ANCF3243_Data
template <>
__device__ __forceinline__ void compute_hessian_assemble_csr<GPU_ANCF3243_Data>(
    GPU_ANCF3243_Data *d_data, SyncedNewtonSolver *d_solver, int elem_idx,
    int qp_idx, int *d_csr_row_offsets, int *d_csr_col_indices,
    double *d_csr_values, double h) {
  // Copy the element-local K construction (24×24) from
  // compute_hessian_assemble, then scatter to CSR using local mapping: coef_idx
  // = node_global * 4 + dof_local

  // Extract e[8][3]
  double e[Quadrature::N_SHAPE_3243][3];
#pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3243; i++) {
    const int node_local  = (i < 4) ? 0 : 1;
    const int dof_local   = i % 4;
    const int node_global = d_data->element_node(elem_idx, node_local);
    const int coef_idx    = node_global * 4 + dof_local;

    e[i][0] = d_data->x12()(coef_idx);
    e[i][1] = d_data->y12()(coef_idx);
    e[i][2] = d_data->z12()(coef_idx);
  }

  double grad_s[Quadrature::N_SHAPE_3243][3];
#pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3243; i++) {
    grad_s[i][0] = d_data->ds_du_pre(qp_idx)(i, 0);
    grad_s[i][1] = d_data->ds_du_pre(qp_idx)(i, 1);
    grad_s[i][2] = d_data->ds_du_pre(qp_idx)(i, 2);
  }

  double F[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3243; i++) {
#pragma unroll
    for (int row = 0; row < 3; row++) {
#pragma unroll
      for (int col = 0; col < 3; col++) {
        F[row][col] += e[i][row] * grad_s[i][col];
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

  double Fh[Quadrature::N_SHAPE_3243][3];
#pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3243; i++) {
#pragma unroll
    for (int row = 0; row < 3; row++) {
      Fh[i][row] = 0.0;
#pragma unroll
      for (int col = 0; col < 3; col++) {
        Fh[i][row] += F[row][col] * grad_s[i][col];
      }
    }
  }

  double lambda = d_data->lambda();
  double mu     = d_data->mu();
  double geom   = (d_data->L() * d_data->W() * d_data->H()) / 8.0;
  double scale =
      d_data->weight_xi()(qp_idx / (Quadrature::N_QP_2 * Quadrature::N_QP_2)) *
      d_data->weight_eta()((qp_idx / Quadrature::N_QP_2) % Quadrature::N_QP_2) *
      d_data->weight_zeta()(qp_idx % Quadrature::N_QP_2);
  double dV = scale * geom;

  const bool use_mr = (d_data->material_model() == MATERIAL_MODEL_MOONEY_RIVLIN);
  double A_mr[3][3][3][3];
  if (use_mr) {
    mr_compute_tangent_tensor(F, d_data->mu10(), d_data->mu01(), d_data->kappa(),
                              A_mr);
  }

  // Local K_elem 24x24
  double K_elem[24][24];
#pragma unroll
  for (int ii = 0; ii < 24; ii++)
    for (int jj = 0; jj < 24; jj++)
      K_elem[ii][jj] = 0.0;

#pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3243; i++) {
#pragma unroll
    for (int j = 0; j < Quadrature::N_SHAPE_3243; j++) {
      double h_ij = grad_s[j][0] * grad_s[i][0] + grad_s[j][1] * grad_s[i][1] +
                    grad_s[j][2] * grad_s[i][2];
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
                sum += A_mr[d][J][e][L] * grad_s[i][J] * grad_s[j][L];
              }
            }
            Kblock[d][e] = sum * dV;
          }
        }
      } else {
        svk_compute_tangent_block(Fh[i], Fh[j], h_ij, trE, Fhj_dot_Fhi, FFT,
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

  // Scatter to CSR using mapping coef_idx = node_global * 4 + dof_local
  for (int local_row_idx = 0; local_row_idx < Quadrature::N_SHAPE_3243;
       local_row_idx++) {
    const int node_local_row  = (local_row_idx < 4) ? 0 : 1;
    const int dof_local_row   = local_row_idx % 4;
    const int node_global_row = d_data->element_node(elem_idx, node_local_row);
    const int coef_idx_row    = node_global_row * 4 + dof_local_row;

    for (int r_dof = 0; r_dof < 3; r_dof++) {
      int global_row = 3 * coef_idx_row + r_dof;
      int local_row  = 3 * local_row_idx + r_dof;

      int row_begin = d_csr_row_offsets[global_row];
      int row_len   = d_csr_row_offsets[global_row + 1] - row_begin;

      for (int local_col_idx = 0; local_col_idx < Quadrature::N_SHAPE_3243;
           local_col_idx++) {
        const int node_local_col = (local_col_idx < 4) ? 0 : 1;
        const int dof_local_col  = local_col_idx % 4;
        const int node_global_col =
            d_data->element_node(elem_idx, node_local_col);
        const int coef_idx_col = node_global_col * 4 + dof_local_col;

        for (int c_dof = 0; c_dof < 3; c_dof++) {
          int global_col = 3 * coef_idx_col + c_dof;
          int local_col  = 3 * local_col_idx + c_dof;

          int pos = binary_search_column_csr_3243(&d_csr_col_indices[row_begin],
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

  // --- Viscous tangent (Kelvin-Voigt) assembly: C_elem (Nloc*3 x Nloc*3) ---
  const int Nloc = Quadrature::N_SHAPE_3243;
  const int Ndof = Nloc * 3;
  double C_elem_loc[24][24];
#pragma unroll
  for (int ii = 0; ii < Ndof; ii++)
    for (int jj = 0; jj < Ndof; jj++)
      C_elem_loc[ii][jj] = 0.0;

#pragma unroll
  for (int a = 0; a < Nloc; a++) {
    double h_a0  = grad_s[a][0];
    double h_a1  = grad_s[a][1];
    double h_a2  = grad_s[a][2];
    double Fh_a0 = Fh[a][0];
    double Fh_a1 = Fh[a][1];
    double Fh_a2 = Fh[a][2];
#pragma unroll
    for (int b = 0; b < Nloc; b++) {
      double h_b0  = grad_s[b][0];
      double h_b1  = grad_s[b][1];
      double h_b2  = grad_s[b][2];
      double Fh_b0 = Fh[b][0];
      double Fh_b1 = Fh[b][1];
      double Fh_b2 = Fh[b][2];

      double hdot = h_a0 * h_b0 + h_a1 * h_b1 + h_a2 * h_b2;

      // build 3x3 block
      double B00 = (eta_d * (Fh_b0 * Fh_a0) + eta_d * FFT[0][0] * hdot +
                    lambda_d * (Fh_a0 * Fh_b0)) *
                   dV;
      double B01 = (eta_d * (Fh_b0 * Fh_a1) + eta_d * FFT[0][1] * hdot +
                    lambda_d * (Fh_a0 * Fh_b1)) *
                   dV;
      double B02 = (eta_d * (Fh_b0 * Fh_a2) + eta_d * FFT[0][2] * hdot +
                    lambda_d * (Fh_a0 * Fh_b2)) *
                   dV;
      double B10 = (eta_d * (Fh_b1 * Fh_a0) + eta_d * FFT[1][0] * hdot +
                    lambda_d * (Fh_a1 * Fh_b0)) *
                   dV;
      double B11 = (eta_d * (Fh_b1 * Fh_a1) + eta_d * FFT[1][1] * hdot +
                    lambda_d * (Fh_a1 * Fh_b1)) *
                   dV;
      double B12 = (eta_d * (Fh_b1 * Fh_a2) + eta_d * FFT[1][2] * hdot +
                    lambda_d * (Fh_a1 * Fh_b2)) *
                   dV;
      double B20 = (eta_d * (Fh_b2 * Fh_a0) + eta_d * FFT[2][0] * hdot +
                    lambda_d * (Fh_a2 * Fh_b0)) *
                   dV;
      double B21 = (eta_d * (Fh_b2 * Fh_a1) + eta_d * FFT[2][1] * hdot +
                    lambda_d * (Fh_a2 * Fh_b1)) *
                   dV;
      double B22 = (eta_d * (Fh_b2 * Fh_a2) + eta_d * FFT[2][2] * hdot +
                    lambda_d * (Fh_a2 * Fh_b2)) *
                   dV;

      int row0                       = 3 * a;
      int col0                       = 3 * b;
      C_elem_loc[row0 + 0][col0 + 0] = B00;
      C_elem_loc[row0 + 0][col0 + 1] = B01;
      C_elem_loc[row0 + 0][col0 + 2] = B02;
      C_elem_loc[row0 + 1][col0 + 0] = B10;
      C_elem_loc[row0 + 1][col0 + 1] = B11;
      C_elem_loc[row0 + 1][col0 + 2] = B12;
      C_elem_loc[row0 + 2][col0 + 0] = B20;
      C_elem_loc[row0 + 2][col0 + 1] = B21;
      C_elem_loc[row0 + 2][col0 + 2] = B22;
    }
  }

  // Scatter viscous C_elem_loc to CSR (no h scaling)
  for (int local_row_idx2 = 0; local_row_idx2 < Nloc; local_row_idx2++) {
    const int node_local_row  = (local_row_idx2 < 4) ? 0 : 1;
    const int dof_local_row   = local_row_idx2 % 4;
    const int node_global_row = d_data->element_node(elem_idx, node_local_row);
    const int coef_idx_row    = node_global_row * 4 + dof_local_row;

    for (int r_dof = 0; r_dof < 3; r_dof++) {
      int global_row = 3 * coef_idx_row + r_dof;
      int local_row  = 3 * local_row_idx2 + r_dof;

      int row_begin = d_csr_row_offsets[global_row];
      int row_len   = d_csr_row_offsets[global_row + 1] - row_begin;

      for (int local_col_idx = 0; local_col_idx < Nloc; local_col_idx++) {
        const int node_local_col = (local_col_idx < 4) ? 0 : 1;
        const int dof_local_col  = local_col_idx % 4;
        const int node_global_col =
            d_data->element_node(elem_idx, node_local_col);
        const int coef_idx_col = node_global_col * 4 + dof_local_col;

        for (int c_dof = 0; c_dof < 3; c_dof++) {
          int global_col = 3 * coef_idx_col + c_dof;
          int local_col  = 3 * local_col_idx + c_dof;

          int pos = binary_search_column_csr_3243(&d_csr_col_indices[row_begin],
                                                  row_len, global_col);
          if (pos >= 0) {
            atomicAdd(&d_csr_values[row_begin + pos],
                      C_elem_loc[local_row][local_col]);
          }
        }
      }
    }
  }
}
