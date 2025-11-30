#pragma once
#include "ANCF3243Data.cuh"

// forward-declare solver type (pointer-only used here)
struct SyncedNewtonSolver;

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


// --- CSR-version Hessian assembly for ANCF3243 ---
static __device__ __forceinline__ int binary_search_column_csr_3243(const int *cols, int n_cols, int target) {
  int left = 0, right = n_cols - 1;
  while (left <= right) {
    int mid = left + ((right - left) >> 1);
    int v = cols[mid];
    if (v == target) return mid;
    if (v < target) left = mid + 1;
    else right = mid - 1;
  }
  return -1;
}

template<typename ElementType>
__device__ __forceinline__ void compute_hessian_assemble_csr(ElementType* d_data,
    SyncedNewtonSolver* d_solver,
    int elem_idx,
    int qp_idx,
    int* d_csr_row_offsets,
    int* d_csr_col_indices,
    double* d_csr_values,
    double h);

// Explicit specialization for GPU_ANCF3243_Data
template<>
__device__ __forceinline__ void compute_hessian_assemble_csr<GPU_ANCF3243_Data>(
    GPU_ANCF3243_Data* d_data,
    SyncedNewtonSolver* d_solver,
    int elem_idx,
    int qp_idx,
    int* d_csr_row_offsets,
    int* d_csr_col_indices,
    double* d_csr_values,
    double h) {

  // Copy the element-local K construction (24×24) from compute_hessian_assemble,
  // then scatter to CSR using local mapping: coef_idx = node_global * 4 + dof_local

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
  double scale  = d_data->weight_xi()(qp_idx / (Quadrature::N_QP_2 * Quadrature::N_QP_2)) *
                  d_data->weight_eta()((qp_idx / Quadrature::N_QP_2) % Quadrature::N_QP_2) *
                  d_data->weight_zeta()(qp_idx % Quadrature::N_QP_2);
  double dV = scale * geom;

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
      double h_ij = grad_s[j][0] * grad_s[i][0] + grad_s[j][1] * grad_s[i][1] + grad_s[j][2] * grad_s[i][2];
      double Fhj_dot_Fhi = Fh[j][0]*Fh[i][0] + Fh[j][1]*Fh[i][1] + Fh[j][2]*Fh[i][2];

      #pragma unroll
      for (int d = 0; d < 3; d++) {
        #pragma unroll
        for (int e = 0; e < 3; e++) {
          double A_de    = lambda * Fh[i][d] * Fh[j][e];
          double B_de    = lambda * trE * h_ij * (d == e ? 1.0 : 0.0);
          double C1_de   = mu * Fhj_dot_Fhi * (d == e ? 1.0 : 0.0);
          double D_de    = mu * Fh[j][d] * Fh[i][e];
          double Etrm_de = mu * h_ij * FFT[d][e];
          double Ftrm_de = -mu * h_ij * (d == e ? 1.0 : 0.0);

          double K_ij_de = (A_de + B_de + C1_de + D_de + Etrm_de + Ftrm_de) * dV;

          int row = 3 * i + d;
          int col = 3 * j + e;
          K_elem[row][col] = K_ij_de;
        }
      }
    }
  }

  // Scatter to CSR using mapping coef_idx = node_global * 4 + dof_local
  for (int local_row_idx = 0; local_row_idx < Quadrature::N_SHAPE_3243; local_row_idx++) {
    const int node_local_row  = (local_row_idx < 4) ? 0 : 1;
    const int dof_local_row   = local_row_idx % 4;
    const int node_global_row = d_data->element_node(elem_idx, node_local_row);
    const int coef_idx_row    = node_global_row * 4 + dof_local_row;

    for (int r_dof = 0; r_dof < 3; r_dof++) {
      int global_row = 3 * coef_idx_row + r_dof;
      int local_row = 3 * local_row_idx + r_dof;

      int row_begin = d_csr_row_offsets[global_row];
      int row_len   = d_csr_row_offsets[global_row + 1] - row_begin;

      for (int local_col_idx = 0; local_col_idx < Quadrature::N_SHAPE_3243; local_col_idx++) {
        const int node_local_col  = (local_col_idx < 4) ? 0 : 1;
        const int dof_local_col   = local_col_idx % 4;
        const int node_global_col = d_data->element_node(elem_idx, node_local_col);
        const int coef_idx_col    = node_global_col * 4 + dof_local_col;

        for (int c_dof = 0; c_dof < 3; c_dof++) {
          int global_col = 3 * coef_idx_col + c_dof;
          int local_col  = 3 * local_col_idx + c_dof;

          int pos = binary_search_column_csr_3243(&d_csr_col_indices[row_begin], row_len, global_col);
          if (pos >= 0) {
            atomicAdd(&d_csr_values[row_begin + pos], h * K_elem[local_row][local_col]);
          }
        }
      }
    }
  }
}