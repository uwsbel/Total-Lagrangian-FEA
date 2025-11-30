#pragma once
#include "ANCF3443Data.cuh"

// forward-declare solver type (pointer-only used here)
struct SyncedNewtonSolver;

__device__ __forceinline__ void compute_p(int, int, GPU_ANCF3443_Data *);
__device__ __forceinline__ void compute_internal_force(int, int,
                                                       GPU_ANCF3443_Data *);
__device__ __forceinline__ void compute_constraint_data(GPU_ANCF3443_Data *);

__device__ __forceinline__ void ancf3443_mat_vec_mul(
    Eigen::Map<Eigen::MatrixXd> A, const double *x, double *out) {
  // clang-format off
    #pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; ++i)
    {
        out[i] = 0.0;
        #pragma unroll
        for (int j = 0; j < Quadrature::N_SHAPE_3443; ++j)
        {
            out[i] += A(i, j) * x[j];
        }
    }
  // clang-format on
}

// Device function to compute determinant of 3x3 matrix
__device__ __forceinline__ double ancf3443_det3x3(const double *J) {
  return J[0] * (J[4] * J[8] - J[5] * J[7]) -
         J[1] * (J[3] * J[8] - J[5] * J[6]) +
         J[2] * (J[3] * J[7] - J[4] * J[6]);
}

__device__ __forceinline__ void ancf3443_b_vec(double u, double v, double w,
                                               double *out) {
  out[0]  = 1.0;
  out[1]  = u;
  out[2]  = v;
  out[3]  = w;
  out[4]  = u * v;
  out[5]  = u * w;
  out[6]  = v * w;
  out[7]  = u * v * w;
  out[8]  = u * u;
  out[9]  = v * v;
  out[10] = (u * u) * v;
  out[11] = u * (v * v);
  out[12] = u * u * u;
  out[13] = v * v * v;
  out[14] = (u * u * u) * v;
  out[15] = u * (v * v * v);
}

__device__ __forceinline__ void ancf3443_b_vec_xi(double xi, double eta,
                                                  double zeta, double L,
                                                  double W, double H,
                                                  double *out) {
  double u = L * xi / 2.0;
  double v = W * eta / 2.0;
  double w = H * zeta / 2.0;
  ancf3443_b_vec(u, v, w, out);
}

__device__ __forceinline__ void ancf3443_db_dxi(double xi, double eta,
                                                double zeta, double L, double W,
                                                double H, double *out) {
  // Map to physical coordinates
  double u = 0.5 * L * xi;
  double v = 0.5 * W * eta;
  double w = 0.5 * H * zeta;

  // db_du as in your Python code
  out[0]  = 0.0;
  out[1]  = 1.0;
  out[2]  = 0.0;
  out[3]  = 0.0;
  out[4]  = v;
  out[5]  = w;
  out[6]  = 0.0;
  out[7]  = v * w;
  out[8]  = 2.0 * u;
  out[9]  = 0.0;
  out[10] = 2.0 * u * v;
  out[11] = v * v;
  out[12] = 3.0 * u * u;
  out[13] = 0.0;
  out[14] = 3.0 * u * u * v;
  out[15] = v * v * v;

// Chain rule: db/dxi = 0.5 * L * db/du
#pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3443; ++i)
    out[i] *= 0.5 * L;
}

__device__ __forceinline__ void ancf3443_db_deta(double xi, double eta,
                                                 double zeta, double L,
                                                 double W, double H,
                                                 double *out) {
  double u = 0.5 * L * xi;
  double v = 0.5 * W * eta;
  double w = 0.5 * H * zeta;

  // db_dv as in your Python code
  out[0]  = 0.0;
  out[1]  = 0.0;
  out[2]  = 1.0;
  out[3]  = 0.0;
  out[4]  = u;
  out[5]  = 0.0;
  out[6]  = w;
  out[7]  = u * w;
  out[8]  = 0.0;
  out[9]  = 2.0 * v;
  out[10] = u * u;
  out[11] = 2.0 * u * v;
  out[12] = 0.0;
  out[13] = 3.0 * v * v;
  out[14] = u * u * u;
  out[15] = 3.0 * u * v * v;

// Chain rule: db/deta = 0.5 * W * db/dv
#pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3443; ++i)
    out[i] *= 0.5 * W;
}

__device__ __forceinline__ void ancf3443_db_dzeta(double xi, double eta,
                                                  double zeta, double L,
                                                  double W, double H,
                                                  double *out) {
  double u = 0.5 * L * xi;
  double v = 0.5 * W * eta;
  // NOTE: removed for performance and suppress warning
  // double w = 0.5 * H * zeta;

  // db_dw as in your Python code
  out[0]  = 0.0;
  out[1]  = 0.0;
  out[2]  = 0.0;
  out[3]  = 1.0;
  out[4]  = 0.0;
  out[5]  = u;
  out[6]  = v;
  out[7]  = u * v;
  out[8]  = 0.0;
  out[9]  = 0.0;
  out[10] = 0.0;
  out[11] = 0.0;
  out[12] = 0.0;
  out[13] = 0.0;
  out[14] = 0.0;
  out[15] = 0.0;

// Chain rule: db/dzeta = 0.5 * H * db/dw
#pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3443; ++i)
    out[i] *= 0.5 * H;
}

// Device function for Jacobian determinant in normalized coordinates
__device__ __forceinline__ void ancf3443_calc_det_J_xi(
    double xi, double eta, double zeta, Eigen::Map<Eigen::MatrixXd> B_inv,
    Eigen::Map<Eigen::VectorXd> x12_jac, Eigen::Map<Eigen::VectorXd> y12_jac,
    Eigen::Map<Eigen::VectorXd> z12_jac, double L, double W, double H,
    double *J_out) {
  double db_dxi[Quadrature::N_SHAPE_3443], db_deta[Quadrature::N_SHAPE_3443],
      db_dzeta[Quadrature::N_SHAPE_3443];
  ancf3443_db_dxi(xi, eta, zeta, L, W, H, db_dxi);
  ancf3443_db_deta(xi, eta, zeta, L, W, H, db_deta);
  ancf3443_db_dzeta(xi, eta, zeta, L, W, H, db_dzeta);

  double ds_dxi[Quadrature::N_SHAPE_3443], ds_deta[Quadrature::N_SHAPE_3443],
      ds_dzeta[Quadrature::N_SHAPE_3443];
  ancf3443_mat_vec_mul(B_inv, db_dxi, ds_dxi);
  ancf3443_mat_vec_mul(B_inv, db_deta, ds_deta);
  ancf3443_mat_vec_mul(B_inv, db_dzeta, ds_dzeta);

  // clang-format off
    #pragma unroll
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            J_out[i * 3 + j] = 0.0;

    #pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; ++i)
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
                                          GPU_ANCF3443_Data *d_data) {
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

    // Get local nodal coordinates for this element
    double x_local_arr[Quadrature::N_SHAPE_3443], y_local_arr[Quadrature::N_SHAPE_3443], z_local_arr[Quadrature::N_SHAPE_3443];
    d_data->x12_elem(elem_idx, x_local_arr);
    d_data->y12_elem(elem_idx, y_local_arr);
    d_data->z12_elem(elem_idx, z_local_arr);
    Eigen::Map<Eigen::VectorXd> x_local(x_local_arr, Quadrature::N_SHAPE_3443);
    Eigen::Map<Eigen::VectorXd> y_local(y_local_arr, Quadrature::N_SHAPE_3443);
    Eigen::Map<Eigen::VectorXd> z_local(z_local_arr, Quadrature::N_SHAPE_3443);

    double e[Quadrature::N_SHAPE_3443][3]; 
    #pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; i++)
    {
        e[i][0] = x_local[i]; // x position
        e[i][1] = y_local[i]; // y position
        e[i][2] = z_local[i]; // z position
        // If you need derivatives, use x_local[i*4+1], etc.
    }
    // Compute F = sum_i e_i ⊗ ∇s_i
    // F is 3x3 matrix stored in row-major order
    #pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; i++)
    { // Loop over nodes
        // Get gradient of shape function i (∇s_i) - this needs proper indexing
        // Assuming ds_du_pre is laid out as [qp_total][16][3]
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
    int elem_idx, int node_idx, GPU_ANCF3443_Data *d_data) {
  double f_i[3] = {0};
  // int node_base = d_data->offset_start()(elem_idx);
  int global_node_idx =
      d_data->element_connectivity()(elem_idx, node_idx / 4) * 4 +
      (node_idx % 4);
  double geom = (d_data->L() * d_data->W() * d_data->H()) / 8.0;

  // clang-format off
    #pragma unroll
    for (int qp_idx = 0; qp_idx < Quadrature::N_TOTAL_QP_4_4_3; qp_idx++)
    {
        double grad_s[3];
        grad_s[0] = d_data->ds_du_pre(qp_idx)(node_idx, 0);
        grad_s[1] = d_data->ds_du_pre(qp_idx)(node_idx, 1);
        grad_s[2] = d_data->ds_du_pre(qp_idx)(node_idx, 2);

        double scale = d_data->weight_xi()(qp_idx / (Quadrature::N_QP_4 * Quadrature::N_QP_3)) *
                       d_data->weight_eta()((qp_idx / Quadrature::N_QP_3) % Quadrature::N_QP_4) *
                       d_data->weight_zeta()(qp_idx % Quadrature::N_QP_3);
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
        atomicAdd(&d_data->f_int(global_node_idx)(d), f_i[d]);
    }

  // clang-format on
}

__device__ __forceinline__ void compute_constraint_data(
    GPU_ANCF3443_Data *d_data) {
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
    GPU_ANCF3443_Data *d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < d_data->n_coef * 3) {
    d_data->f_int()[thread_idx] = 0.0;
  }
}

// --- CSR-version Hessian assembly for ANCF3443 ---
static __device__ __forceinline__ int binary_search_column_csr_3443(const int *cols, int n_cols, int target) {
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

// Explicit specialization for GPU_ANCF3443_Data
template<>
__device__ __forceinline__ void compute_hessian_assemble_csr<GPU_ANCF3443_Data>(
    GPU_ANCF3443_Data* d_data,
    SyncedNewtonSolver* d_solver,
    int elem_idx,
    int qp_idx,
    int* d_csr_row_offsets,
    int* d_csr_col_indices,
    double* d_csr_values,
    double h) {

  // Build local element K_elem (48x48) similarly to FEAT10/ANCF3243.
  // Use existing local nodal extraction provided in compute_hessian_assemble
  // (if you need the full constitutive formulas I can copy them in).

  double e[Quadrature::N_SHAPE_3443][3];
  #pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {
    int node_local = i / 4;           // which physical node (0..3)
    int dof_local  = i % 4;           // which local DOF
    int node_global = d_data->element_connectivity()(elem_idx, node_local);
    int coef_idx = node_global * 4 + dof_local;
    e[i][0] = d_data->x12()(coef_idx);
    e[i][1] = d_data->y12()(coef_idx);
    e[i][2] = d_data->z12()(coef_idx);
  }

  double grad_s[Quadrature::N_SHAPE_3443][3];
  #pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {
    grad_s[i][0] = d_data->ds_du_pre(qp_idx)(i, 0);
    grad_s[i][1] = d_data->ds_du_pre(qp_idx)(i, 1);
    grad_s[i][2] = d_data->ds_du_pre(qp_idx)(i, 2);
  }

  double F[3][3] = {{0.0}};
  #pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {
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

  double Fh[Quadrature::N_SHAPE_3443][3];
  #pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {
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
  double scale  = d_data->weight_xi()(qp_idx / (Quadrature::N_QP_4 * Quadrature::N_QP_3)) *
                  d_data->weight_eta()((qp_idx / Quadrature::N_QP_3) % Quadrature::N_QP_4) *
                  d_data->weight_zeta()(qp_idx % Quadrature::N_QP_3);
  double dV = scale * geom;

  // Local K_elem (48 x 48)
  const int Nloc = Quadrature::N_SHAPE_3443 * 3; // 16*3 = 48
  // Because large stack arrays can be heavy, but follow pattern from other DataFunc
  // Implement K_elem as a local array
  double K_elem[48][48];
  #pragma unroll
  for (int ii = 0; ii < Nloc; ii++)
    for (int jj = 0; jj < Nloc; jj++)
      K_elem[ii][jj] = 0.0;

  // Build K_elem using the same constitutive building block pattern
  #pragma unroll
  for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {
    #pragma unroll
    for (int j = 0; j < Quadrature::N_SHAPE_3443; j++) {
      double h_ij = grad_s[j][0] * grad_s[i][0] + grad_s[j][1] * grad_s[i][1] + grad_s[j][2] * grad_s[i][2];
      double Fhj_dot_Fhi = Fh[j][0]*Fh[i][0] + Fh[j][1]*Fh[i][1] + Fh[j][2]*Fh[i][2];

      for (int d = 0; d < 3; d++) {
        for (int eidx = 0; eidx < 3; eidx++) {
          double A_de    = lambda * Fh[i][d] * Fh[j][eidx];
          double B_de    = lambda * trE * h_ij * (d == eidx ? 1.0 : 0.0);
          double C1_de   = mu * Fhj_dot_Fhi * (d == eidx ? 1.0 : 0.0);
          double D_de    = mu * Fh[j][d] * Fh[i][eidx];
          double Etrm_de = mu * h_ij * FFT[d][eidx];
          double Ftrm_de = -mu * h_ij * (d == eidx ? 1.0 : 0.0);

          double K_ij_de = (A_de + B_de + C1_de + D_de + Etrm_de + Ftrm_de) * dV;

          int row = 3 * i + d;
          int col = 3 * j + eidx;
          K_elem[row][col] = K_ij_de;
        }
      }
    }
  }

  // Scatter to CSR
  for (int local_row = 0; local_row < Quadrature::N_SHAPE_3443; local_row++) {
    int node_local_row = local_row / 4;
    int dof_local_row  = local_row % 4;
    int node_global_row = d_data->element_connectivity()(elem_idx, node_local_row);
    int coef_idx_row = node_global_row * 4 + dof_local_row;

    for (int r_dof = 0; r_dof < 3; r_dof++) {
      int global_row = 3 * coef_idx_row + r_dof;
      int local_row_index = 3 * local_row + r_dof;

      int row_begin = d_csr_row_offsets[global_row];
      int row_len = d_csr_row_offsets[global_row + 1] - row_begin;

      for (int local_col = 0; local_col < Quadrature::N_SHAPE_3443; local_col++) {
        int node_local_col = local_col / 4;
        int dof_local_col  = local_col % 4;
        int node_global_col = d_data->element_connectivity()(elem_idx, node_local_col);
        int coef_idx_col = node_global_col * 4 + dof_local_col;

        for (int c_dof = 0; c_dof < 3; c_dof++) {
          int global_col = 3 * coef_idx_col + c_dof;
          int local_col_index = 3 * local_col + c_dof;

          int pos = binary_search_column_csr_3443(&d_csr_col_indices[row_begin], row_len, global_col);
          if (pos >= 0) {
            atomicAdd(&d_csr_values[row_begin + pos], h * K_elem[local_row_index][local_col_index]);
          }
        }
      }
    }
  }
}

