#pragma once
#include "ANCF3243Data.cuh"

__device__ __forceinline__ void compute_p(int, int, GPU_ANCF3243_Data *);
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
                                          GPU_ANCF3243_Data *d_data) {
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

    // Extract local nodal coordinates (e vectors)
    double e[8][3]; // 8 nodes, each with 3 coordinates
    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        e[i][0] = d_data->x12(elem_idx)(i); // x coordinate
        e[i][1] = d_data->y12(elem_idx)(i); // y coordinate
        e[i][2] = d_data->z12(elem_idx)(i); // z coordinate
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
    double factor = d_data->lambda() * (0.5 * tr_FtF - 1.5);
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            d_data->P(elem_idx, qp_idx)(i, j) = factor * d_data->F(elem_idx, qp_idx)(i, j) + d_data->mu() * (FFF[i][j] - d_data->F(elem_idx, qp_idx)(i, j));
        }
  // clang-format on
}

__device__ __forceinline__ void compute_internal_force(
    int elem_idx, int node_idx, GPU_ANCF3243_Data *d_data) {
  double f_i[3] = {0};
  int node_base = d_data->offset_start()(elem_idx);
  double geom   = (d_data->L() * d_data->W() * d_data->H()) / 8.0;

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
        atomicAdd(&d_data->f_int(node_base + node_idx)(d), f_i[d]);
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

    d_data->constraint_jac()(thread_idx * 3,
                             d_data->fixed_nodes()[thread_idx] * 3)     = 1.0;
    d_data->constraint_jac()(thread_idx * 3 + 1,
                             d_data->fixed_nodes()[thread_idx] * 3 + 1) = 1.0;
    d_data->constraint_jac()(thread_idx * 3 + 2,
                             d_data->fixed_nodes()[thread_idx] * 3 + 2) = 1.0;
  }
}

__device__ __forceinline__ void clear_internal_force(
    GPU_ANCF3243_Data *d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < d_data->n_coef * 3) {
    d_data->f_int()[thread_idx] = 0.0;
  }
}

// Add at the end of the file, after clear_internal_force()

__device__ __forceinline__ void compute_hessian_assemble(
    int elem_idx, int qp_idx, GPU_ANCF3243_Data *d_data,
    Eigen::Map<Eigen::MatrixXd>
        H_global,  // Eigen Map to global Hessian (n_dofs x n_dofs)
    double h) {    // Output: Hessian-vector product (3*n_coef)

  // TODO: Implement Hessian-vector product for ANCF3243 elements
  // Element has 8 nodes × 3 DOFs = 24 DOFs per element
  // For now, this is a placeholder stub
}