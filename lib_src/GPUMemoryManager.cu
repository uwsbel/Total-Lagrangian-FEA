#include "GPUMemoryManager.cuh"
#include <iomanip>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Device function: matrix-vector multiply (8x8 * 8x1)
__device__ void mat_vec_mul8(Eigen::Map<Eigen::MatrixXd> A, const double *x, double *out)
{
  for (int i = 0; i < Quadrature::N_SHAPE; ++i)
  {
    out[i] = 0.0;
    for (int j = 0; j < Quadrature::N_SHAPE; ++j)
    {
      out[i] += A(i, j) * x[j];
    }
  }
}

// Device function to compute determinant of 3x3 matrix
__device__ double det3x3(const double *J)
{
  return J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) + J[2] * (J[3] * J[7] - J[4] * J[6]);
}

// Kernel: one thread per quadrature point, computes 8x3 ds_du_pre
__global__ void ds_du_pre_kernel(double L, double W, double H, GPU_ANCF3243_Data *d_data)
{
  int ixi = blockIdx.x;
  int ieta = blockIdx.y;
  int izeta = threadIdx.x;
  int idx = ixi * Quadrature::N_QP_2 * Quadrature::N_QP_2 + ieta * Quadrature::N_QP_2 + izeta;

  double xi = d_data->gauss_xi()(ixi);
  double eta = d_data->gauss_eta()(ieta);
  double zeta = d_data->gauss_zeta()(izeta);

  double u = L * xi / 2.0;
  double v = W * eta / 2.0;
  double w = H * zeta / 2.0;

  double db_du[Quadrature::N_SHAPE] = {0, 1, 0, 0, v, w, 2 * u, 3 * u * u};
  double db_dv[Quadrature::N_SHAPE] = {0, 0, 1, 0, u, 0, 0, 0};
  double db_dw[Quadrature::N_SHAPE] = {0, 0, 0, 1, 0, u, 0, 0};

  double ds_du[Quadrature::N_SHAPE], ds_dv[Quadrature::N_SHAPE], ds_dw[Quadrature::N_SHAPE];
  mat_vec_mul8(d_data->B_inv(), db_du, ds_du);
  mat_vec_mul8(d_data->B_inv(), db_dv, ds_dv);
  mat_vec_mul8(d_data->B_inv(), db_dw, ds_dw);

  // Store as 8x3 matrix: for each i in 0..7, store ds_du, ds_dv, ds_dw as columns
  for (int i = 0; i < Quadrature::N_SHAPE; ++i)
  {
    d_data->ds_du_pre(idx)(i, 0) = ds_du[i];
    d_data->ds_du_pre(idx)(i, 1) = ds_dv[i];
    d_data->ds_du_pre(idx)(i, 2) = ds_dw[i];
  }
}

__device__ void b_vec(double u, double v, double w, double *out)
{
  out[0] = 1.0;
  out[1] = u;
  out[2] = v;
  out[3] = w;
  out[4] = u * v;
  out[5] = u * w;
  out[6] = u * u;
  out[7] = u * u * u;
}

__device__ void b_vec_xi(double xi, double eta, double zeta, double L, double W, double H, double *out)
{
  double u = L * xi / 2.0;
  double v = W * eta / 2.0;
  double w = H * zeta / 2.0;
  b_vec(u, v, w, out);
}

// Device function for Jacobian determinant in normalized coordinates
__device__ void calc_det_J_xi(double xi,
                              double eta,
                              double zeta,
                              Eigen::Map<Eigen::MatrixXd> B_inv,
                              Eigen::Map<Eigen::VectorXd> x12_jac,
                              Eigen::Map<Eigen::VectorXd> y12_jac,
                              Eigen::Map<Eigen::VectorXd> z12_jac,
                              double L,
                              double W,
                              double H,
                              double *J_out)
{
  double db_dxi[Quadrature::N_SHAPE] = {
      0.0, L / 2, 0.0, 0.0, (L * W / 4) * eta, (L * H / 4) * zeta, (L * L / 2) * xi, (3 * L * L * L / 8) * xi * xi};
  double db_deta[Quadrature::N_SHAPE] = {0.0, 0.0, W / 2, 0.0, (L * W / 4) * xi, 0.0, 0.0, 0.0};
  double db_dzeta[Quadrature::N_SHAPE] = {0.0, 0.0, 0.0, H / 2, 0.0, (L * H / 4) * xi, 0.0, 0.0};

  double ds_dxi[Quadrature::N_SHAPE], ds_deta[Quadrature::N_SHAPE], ds_dzeta[Quadrature::N_SHAPE];
  mat_vec_mul8(B_inv, db_dxi, ds_dxi);
  mat_vec_mul8(B_inv, db_deta, ds_deta);
  mat_vec_mul8(B_inv, db_dzeta, ds_dzeta);

  // Nodal matrix: 3 × 8
  // J = N_mat_jac @ np.column_stack([ds_dxi, ds_deta, ds_dzeta])
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      J_out[i * 3 + j] = 0.0;

  for (int i = 0; i < Quadrature::N_SHAPE; ++i)
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
}

__global__ void mass_matrix_qp_kernel(GPU_ANCF3243_Data *d_data)
{
  int n_qp_per_elem = Quadrature::N_QP_6 * Quadrature::N_QP_2 * Quadrature::N_QP_2;
  int thread_global = blockIdx.x * blockDim.x + threadIdx.x;
  int elem = thread_global / (Quadrature::N_SHAPE * Quadrature::N_SHAPE);
  int item_local = thread_global % (Quadrature::N_SHAPE * Quadrature::N_SHAPE);
  if (elem >= d_data->get_n_beam())
    return;

  for (int qp_local = 0; qp_local < n_qp_per_elem; qp_local++)
  {
    // Decode qp_local into (ixi, ieta, izeta)
    int ixi = qp_local / (Quadrature::N_QP_2 * Quadrature::N_QP_2);
    int ieta = (qp_local / Quadrature::N_QP_2) % Quadrature::N_QP_2;
    int izeta = qp_local % Quadrature::N_QP_2;

    double xi = d_data->gauss_xi_m()(ixi);
    double eta = d_data->gauss_eta()(ieta);
    double zeta = d_data->gauss_zeta()(izeta);
    double weight = d_data->weight_xi_m()(ixi) * d_data->weight_eta()(ieta) * d_data->weight_zeta()(izeta);

    // Get element's node offset
    int node_offset = d_data->offset_start()(elem);

    // Get local nodal coordinates for this element
    Eigen::Map<Eigen::VectorXd> x_loc = d_data->x12(elem);
    Eigen::Map<Eigen::VectorXd> y_loc = d_data->y12(elem);
    Eigen::Map<Eigen::VectorXd> z_loc = d_data->z12(elem);

    // Compute shape function at this QP
    double b[8];
    b_vec_xi(xi, eta, zeta, d_data->L(), d_data->W(), d_data->H(), b);
    // b_vec_xi(xi, eta, zeta, 2.0, 1.0, 1.0, b);

    // Compute s = B_inv @ b
    double s[8];
    mat_vec_mul8(d_data->B_inv(), b, s);

    // Compute Jacobian determinant
    double J[9];
    calc_det_J_xi(xi, eta, zeta, d_data->B_inv(), x_loc, y_loc, z_loc, d_data->L(), d_data->W(), d_data->H(), J);
    double detJ = det3x3(J);

    // For each local node, output (global_node, value)
    int i_local = item_local / Quadrature::N_SHAPE;        // Local node index (0-7)
    int j_local = item_local % Quadrature::N_SHAPE;        // Local shape function index (0-7)
    int i_global = d_data->offset_start()(elem) + i_local; // Global node index
    int j_global = d_data->offset_start()(elem) + j_local; // Global shape function index

    atomicAdd(d_data->node_values(i_global, j_global), d_data->rho0() * s[i_local] * s[j_local] * weight * detJ);
  }
}

__device__ void compute_deformation_gradient(int elem_idx, int qp_idx, GPU_ANCF3243_Data *d_data)
{
  // Initialize F to zero
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      d_data->F(elem_idx, qp_idx)(i, j) = 0.0;
    }
  }

  // Extract local nodal coordinates (e vectors)
  double e[8][3]; // 8 nodes, each with 3 coordinates
  for (int i = 0; i < 8; i++)
  {
    e[i][0] = d_data->x12(elem_idx)(i); // x coordinate
    e[i][1] = d_data->y12(elem_idx)(i); // y coordinate
    e[i][2] = d_data->z12(elem_idx)(i); // z coordinate
  }

  // Compute F = sum_i e_i ⊗ ∇s_i
  // F is 3x3 matrix stored in row-major order
  for (int i = 0; i < Quadrature::N_SHAPE; i++)
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
    for (int row = 0; row < 3; row++)
    { // e_i components
      for (int col = 0; col < 3; col++)
      { // ∇s_i components
        d_data->F(elem_idx, qp_idx)(row, col) +=
            e[i][row] * grad_s_i[col];
      }
    }
  }
}

__global__ void deformation_gradient_kernel(GPU_ANCF3243_Data *d_data)
{
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx = thread_idx / Quadrature::N_TOTAL_QP;
  int qp_idx = thread_idx % Quadrature::N_TOTAL_QP;

  if (elem_idx >= d_data->get_n_beam() || qp_idx >= Quadrature::N_TOTAL_QP)
    return;

  compute_deformation_gradient(elem_idx, qp_idx, d_data);
}

__device__ void compute_p_from_F(int elem_idx, int qp_idx, GPU_ANCF3243_Data *d_data)
{
  // --- Compute C = F^T * F ---
  double FtF[3][3] = {0.0};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        FtF[i][j] += d_data->F(elem_idx, qp_idx)(k, i) * d_data->F(elem_idx, qp_idx)(k, j);

  // --- trace(F^T F) ---
  double tr_FtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

  // 1. Compute Ft (transpose of F)
  double Ft[3][3] = {0};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
      Ft[i][j] = d_data->F(elem_idx, qp_idx)(j, i); // transpose
    }

  // 2. Compute G = F * Ft
  double G[3][3] = {0}; // G = F * F^T
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
      {
        G[i][j] += d_data->F(elem_idx, qp_idx)(i, k) * Ft[k][j];
      }

  // 3. Compute FFF = G * F = (F * Ft) * F
  double FFF[3][3] = {0};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
      {
        FFF[i][j] += G[i][k] * d_data->F(elem_idx, qp_idx)(k, j);
      }

  // --- Compute P ---
  double factor = d_data->lambda() * (0.5 * tr_FtF - 1.5);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
      d_data->P(elem_idx, qp_idx)(i, j) = factor * d_data->F(elem_idx, qp_idx)(i, j) + d_data->mu() * (FFF[i][j] - d_data->F(elem_idx, qp_idx)(i, j));
    }
}

__global__ void calc_p_kernel(GPU_ANCF3243_Data *d_data)
{
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx = thread_idx / Quadrature::N_TOTAL_QP;
  int qp_idx = thread_idx % Quadrature::N_TOTAL_QP;

  if (elem_idx >= d_data->get_n_beam() || qp_idx >= Quadrature::N_TOTAL_QP)
    return;

  compute_p_from_F(elem_idx, qp_idx, d_data);
}

void GPU_ANCF3243_Data::CalcDeformationGradient()
{
  int threads = 128;
  int blocks = (n_beam * Quadrature::N_TOTAL_QP + threads - 1) / threads;
  deformation_gradient_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}

void GPU_ANCF3243_Data::CalcPFromF()
{
  int threads = 128;
  int blocks = (n_beam * Quadrature::N_TOTAL_QP + threads - 1) / threads;
  calc_p_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}

void GPU_ANCF3243_Data::CalcDsDuPre()
{
  // Launch kernel
  dim3 blocks_pre(Quadrature::N_QP_3, Quadrature::N_QP_2);
  dim3 threads_pre(Quadrature::N_QP_2);
  ds_du_pre_kernel<<<blocks_pre, threads_pre>>>(2.0, 1.0, 1.0, d_data);
  cudaDeviceSynchronize();
}

void GPU_ANCF3243_Data::PrintDsDuPre()
{
  // Allocate host memory for all quadrature points
  const int total_size = Quadrature::N_TOTAL_QP * Quadrature::N_SHAPE * 3;
  double *h_ds_du_pre_raw = new double[total_size];

  // Copy from device to host
  HANDLE_ERROR(cudaMemcpy(h_ds_du_pre_raw, d_ds_du_pre, total_size * sizeof(double), cudaMemcpyDeviceToHost));

  // Print each quadrature point's matrix
  for (int qp = 0; qp < Quadrature::N_TOTAL_QP; ++qp)
  {
    std::cout << "\n=== Quadrature Point " << qp << " ===" << std::endl;

    // Create Eigen::Map for this quadrature point's data
    double *qp_data = h_ds_du_pre_raw + qp * Quadrature::N_SHAPE * 3;
    Eigen::Map<Eigen::MatrixXd> ds_du_matrix(qp_data, Quadrature::N_SHAPE, 3);

    // Print the 8x3 matrix with column headers
    std::cout << "        ds/du       ds/dv       ds/dw" << std::endl;
    for (int i = 0; i < Quadrature::N_SHAPE; ++i)
    {
      std::cout << "Node " << i << ": ";
      for (int j = 0; j < 3; ++j)
      {
        std::cout << std::setw(10) << std::fixed << std::setprecision(6) << ds_du_matrix(i, j) << " ";
      }
      std::cout << std::endl;
    }
  }

  delete[] h_ds_du_pre_raw;
}

void GPU_ANCF3243_Data::CalcMassMatrix()
{
  // Mass terms computation
  const int N_OUT = n_beam * Quadrature::N_SHAPE * Quadrature::N_SHAPE;

  // Launch kernel
  int threads = 128;
  int blocks = (N_OUT + threads - 1) / threads;
  mass_matrix_qp_kernel<<<blocks, threads>>>(d_data);

  cudaDeviceSynchronize();
}

void GPU_ANCF3243_Data::RetrieveMassMatrixToCPU(Eigen::MatrixXd &mass_matrix)
{
  // Allocate host memory for all quadrature points
  const int total_size = n_coef * n_coef;

  mass_matrix.resize(n_coef, n_coef);

  // Copy from device to host
  HANDLE_ERROR(cudaMemcpy(mass_matrix.data(), d_node_values, total_size * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_ANCF3243_Data::RetrieveInternalForceToCPU(Eigen::VectorXd &internal_force)
{
  int expected_size = n_coef * 3;
  internal_force.resize(expected_size);

  HANDLE_ERROR(cudaMemcpy(internal_force.data(), d_f_elem_out, expected_size * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_ANCF3243_Data::RetrieveDeformationGradientToCPU(std::vector<std::vector<Eigen::MatrixXd>> &deformation_gradient)
{
  deformation_gradient.resize(n_beam);
  for (int i = 0; i < n_beam; i++)
  {
    deformation_gradient[i].resize(Quadrature::N_TOTAL_QP);
    for (int j = 0; j < Quadrature::N_TOTAL_QP; j++)
    {
      deformation_gradient[i][j].resize(3, 3);
      HANDLE_ERROR(cudaMemcpy(deformation_gradient[i][j].data(), d_F + i * Quadrature::N_TOTAL_QP * 3 * 3 + j * 3 * 3, 3 * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    }
  }
}

void GPU_ANCF3243_Data::RetrievePFromFToCPU(std::vector<std::vector<Eigen::MatrixXd>> &p_from_F)
{
  p_from_F.resize(n_beam);
  for (int i = 0; i < n_beam; i++)
  {
    p_from_F[i].resize(Quadrature::N_TOTAL_QP);
    for (int j = 0; j < Quadrature::N_TOTAL_QP; j++)
    {
      p_from_F[i][j].resize(3, 3);
      HANDLE_ERROR(cudaMemcpy(p_from_F[i][j].data(), d_P + i * Quadrature::N_TOTAL_QP * 3 * 3 + j * 3 * 3, 3 * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    }
  }
}

void GPU_ANCF3243_Data::RetrieveConstraintDataToCPU(Eigen::VectorXd &constraint)
{
  int expected_size = 12;
  constraint.resize(expected_size);
  HANDLE_ERROR(cudaMemcpy(constraint.data(), d_constraint, expected_size * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_ANCF3243_Data::RetrieveConstraintJacobianToCPU(Eigen::MatrixXd &constraint_jac)
{
  int expected_size = 12 * n_coef * 3;
  constraint_jac.resize(12, n_coef * 3);
  HANDLE_ERROR(cudaMemcpy(constraint_jac.data(), d_constraint_jac, expected_size * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_ANCF3243_Data::RetrievePositionToCPU(Eigen::VectorXd &x12, Eigen::VectorXd &y12, Eigen::VectorXd &z12)
{
  int expected_size = n_coef;
  x12.resize(expected_size);
  y12.resize(expected_size);
  z12.resize(expected_size);
  HANDLE_ERROR(cudaMemcpy(x12.data(), d_x12, expected_size * sizeof(double), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(y12.data(), d_y12, expected_size * sizeof(double), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(z12.data(), d_z12, expected_size * sizeof(double), cudaMemcpyDeviceToHost));
}

__device__ void compute_internal_force(int elem_idx, int node_idx, GPU_ANCF3243_Data *d_data)
{

  double f_i[3] = {0};
  int node_base = d_data->offset_start()(elem_idx);
  double geom = (d_data->L() * d_data->W() * d_data->H()) / 8.0;

  // set 0
  for (int d = 0; d < 3; ++d)
  {
    d_data->f_elem_out(node_base + node_idx)(d) = 0.0;
  }

  for (int qp_idx = 0; qp_idx < Quadrature::N_TOTAL_QP; qp_idx++)
  {
    double grad_s[3];
    grad_s[0] = d_data->ds_du_pre(qp_idx)(node_idx, 0);
    grad_s[1] = d_data->ds_du_pre(qp_idx)(node_idx, 1);
    grad_s[2] = d_data->ds_du_pre(qp_idx)(node_idx, 2);

    double scale = d_data->weight_xi()(qp_idx / (Quadrature::N_QP_2 * Quadrature::N_QP_2)) *
                   d_data->weight_eta()((qp_idx / Quadrature::N_QP_2) % Quadrature::N_QP_2) *
                   d_data->weight_zeta()(qp_idx % Quadrature::N_QP_2);

    for (int r = 0; r < 3; ++r)
    {
      for (int c = 0; c < 3; ++c)
      {
        f_i[r] += (d_data->P(elem_idx, qp_idx)(r, c) * grad_s[c]) * scale * geom;
        // printf("P(%d, %d)(%d, %d) = %.17f\n", elem_idx, qp_idx, r, c, d_data->P(elem_idx, qp_idx)(r, c));
        //  printf("f_i[%d] += P(%d, %d)(%d, %d) * grad_s[%d] * scale = %f\n", r, elem_idx, qp_idx, r, c, c, f_i[r]);
      }
    }
  }

  for (int d = 0; d < 3; ++d)
  {
    atomicAdd(&d_data->f_elem_out(node_base + node_idx)(d), f_i[d]);
  }
}

__global__ void compute_internal_force_kernel(GPU_ANCF3243_Data *d_data)
{
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx = thread_idx / Quadrature::N_SHAPE;
  int node_idx = thread_idx % Quadrature::N_SHAPE;

  if (elem_idx >= d_data->get_n_beam() || node_idx >= Quadrature::N_SHAPE)
    return;

  compute_internal_force(elem_idx, node_idx, d_data);
}

void GPU_ANCF3243_Data::CalcInternalForce()
{
  int threads = 128;
  int blocks = (n_beam * Quadrature::N_SHAPE + threads - 1) / threads;
  compute_internal_force_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}

__device__ void compute_constraint_data(GPU_ANCF3243_Data *d_data)
{
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx == 0)
  {
    d_data->constraint()[0] = d_data->x12()(0) - (-1.0);
    d_data->constraint()[1] = d_data->y12()(0) - (1.0);
    d_data->constraint()[2] = d_data->z12()(0) - (0.0);

    d_data->constraint()[3] = d_data->x12()(1) - (1.0);
    d_data->constraint()[4] = d_data->y12()(1) - (0.0);
    d_data->constraint()[5] = d_data->z12()(1) - (0.0);

    d_data->constraint()[6] = d_data->x12()(2) - (0.0);
    d_data->constraint()[7] = d_data->y12()(2) - (1.0);
    d_data->constraint()[8] = d_data->z12()(2) - (0.0);

    d_data->constraint()[9] = d_data->x12()(3) - (0.0);
    d_data->constraint()[10] = d_data->y12()(3) - (0.0);
    d_data->constraint()[11] = d_data->z12()(3) - (1.0);

    for (int i = 0; i < 12; i++)
    {
      d_data->constraint_jac()(i, i) = 1.0;
    }
  }
}

__global__ void compute_constraint_data_kernel(GPU_ANCF3243_Data *d_data)
{
  compute_constraint_data(d_data);
}

void GPU_ANCF3243_Data::CalcConstraintData()
{
  int threads = 128;
  int blocks = (n_beam * Quadrature::N_SHAPE + threads - 1) / threads;
  compute_constraint_data_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}

__device__ double solver_grad_L(int tid, GPU_ANCF3243_Data *d_data)
{
  double res = 0.0;

  int node_i = tid / 3;
  int dof_i = tid % 3;

  // Mass matrix contribution
  for (int node_j = 0; node_j < d_data->get_n_coef(); node_j++)
  {
    double mass_ij = *(d_data->node_values(node_i, node_j));
    int tid_j = node_j * 3 + dof_i;

    double v_diff = d_data->v_guess()[tid_j] - d_data->v_prev()[tid_j];
    res += mass_ij * v_diff / d_data->solver_time_step();
  }

  // Internal force
  res -= (-d_data->f_elem_out()(tid));

  if (tid == 3 * d_data->get_n_coef() - 10)
  {
    res -= 3100.0;
  }

  // Constraints
  for (int i = 0; i < 12; i++)
  {
    res += d_data->constraint_jac()(i, tid) *
           (d_data->lambda_guess()[i] + d_data->rho0() * d_data->solver_time_step() * d_data->constraint()[i]);
  }

  return res;
}

__global__ void
one_step_nesterov_kernel(GPU_ANCF3243_Data *d_data)
{
  cg::grid_group grid = cg::this_grid();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // assign x12_prev, y12_prev, z12_prev
  if (tid < d_data->get_n_coef())
  {
    d_data->x12_prev()(tid) = d_data->x12()(tid);
    d_data->y12_prev()(tid) = d_data->y12()(tid);
    d_data->z12_prev()(tid) = d_data->z12()(tid);
  }

  for (int outer_iter = 0; outer_iter < d_data->solver_max_outer(); outer_iter++)
  {
    // Initialize variables for each thread
    double v_k = 0.0;
    double v_next = 0.0;
    double v_km1 = 0.0;
    double t = 1.0;
    double v_prev = 0.0;

    if (tid == 0)
    {
      *d_data->prev_norm_g() = 0.0;
      *d_data->norm_g() = 0.0;
      *d_data->flag() = 0;
      printf("resetting\n");
    }

    grid.sync();
    volatile int *flag_ptr = d_data->flag();

    // Initialize for valid threads only
    if (tid < d_data->get_n_coef() * 3)
    {
      v_prev = d_data->v_prev()[tid];
      v_k = d_data->v_guess()[tid];
      v_km1 = d_data->v_guess()[tid]; // zero momentum at first step
      t = 1.0;
    }

    double t_next = 1.0; // Declare t_next here

    for (int inner_iter = 0; inner_iter < d_data->solver_max_inner(); inner_iter++)
    {
      if (*flag_ptr == 0)
      {
        if (tid == 0)
        {
          printf("outer iter: %d, inner iter: %d\n", outer_iter, inner_iter);
        }

        // Step 1: Each thread computes its look-ahead velocity component
        double y = 0.0; // Declare y here
        if (tid < d_data->get_n_coef() * 3)
        {
          t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t * t));
          double beta = (t - 1.0) / t_next;
          y = v_k + beta * (v_k - v_km1);

          // Store look-ahead velocity temporarily
          d_data->v_guess()[tid] = y; // Use v_guess as temp storage for y
        }

        grid.sync();

        // Step 2: Update scratch positions using look-ahead velocities
        if (tid < d_data->get_n_coef())
        {
          d_data->x12()(tid) = d_data->x12_prev()(tid) + d_data->solver_time_step() * d_data->v_guess()(tid * 3 + 0);
          d_data->y12()(tid) = d_data->y12_prev()(tid) + d_data->solver_time_step() * d_data->v_guess()(tid * 3 + 1);
          d_data->z12()(tid) = d_data->z12_prev()(tid) + d_data->solver_time_step() * d_data->v_guess()(tid * 3 + 2);
        }

        grid.sync();

        // print f_elem_out
        if (tid == 0)
        {
          printf("pre f_elem_out");
          for (int i = 0; i < 3 * d_data->get_n_coef(); i++)
          {
            printf("%f ", d_data->f_elem_out()(i));
          }
          printf("\n");
        }

        // Step 3: Compute internal forces at look-ahead positions

        if (tid < d_data->get_n_beam() * Quadrature::N_TOTAL_QP)
        {
          int elem_idx = tid / Quadrature::N_TOTAL_QP;
          int qp_idx = tid % Quadrature::N_TOTAL_QP;
          compute_deformation_gradient(elem_idx, qp_idx, d_data);
        }

        grid.sync();

        if (tid < d_data->get_n_beam() * Quadrature::N_TOTAL_QP)
        {
          int elem_idx = tid / Quadrature::N_TOTAL_QP;
          int qp_idx = tid % Quadrature::N_TOTAL_QP;
          compute_p_from_F(elem_idx, qp_idx, d_data);
        }

        grid.sync();

        if (tid < d_data->get_n_beam() * Quadrature::N_SHAPE)
        {
          int elem_idx = tid / Quadrature::N_SHAPE;
          int node_idx = tid % Quadrature::N_SHAPE;
          compute_internal_force(elem_idx, node_idx, d_data);
        }

        grid.sync();

        if (tid == 0)
        {
          printf("post f_elem_out");
          for (int i = 0; i < 3 * d_data->get_n_coef(); i++)
          {
            printf("%f ", d_data->f_elem_out()(i));
          }
          printf("\n");
        }

        if (tid == 0)
        {
          compute_constraint_data(d_data);
        }

        grid.sync();

        if (tid < d_data->get_n_coef() * 3)
        {
          double g = solver_grad_L(tid, d_data);
          d_data->g()[tid] = g;
        }

        grid.sync();

        if (tid == 0)
        {
          // calculate norm of g
          double norm_g = 0.0;
          for (int i = 0; i < 3 * d_data->get_n_coef(); i++)
          {
            norm_g += d_data->g()(i) * d_data->g()(i);
          }
          *d_data->norm_g() = sqrt(norm_g);
          printf("norm_g: %.17f\n", *d_data->norm_g());

          if (inner_iter > 0 && abs(*d_data->norm_g() - *d_data->prev_norm_g()) < d_data->solver_inner_tol())
          {
            printf("Converged diff: %.17f\n", *d_data->norm_g() - *d_data->prev_norm_g());
            *d_data->flag() = 1;
          }
        }

        grid.sync();

        // Step 4: Compute gradients and update velocities
        if (tid < d_data->get_n_coef() * 3)
        {

          v_next = y - d_data->solver_alpha() * d_data->g()[tid];

          // Update for next iteration
          v_km1 = v_k;
          v_k = v_next;
          t = t_next;

          // Store final velocity
          d_data->v_guess()[tid] = v_next;
        }

        grid.sync();

        if (tid == 0)
        {
          *d_data->prev_norm_g() = *d_data->norm_g();
        }

        grid.sync();
      }
    }

    // After inner loop convergence, update v_prev for next outer iteration
    if (tid < d_data->get_n_coef() * 3)
    {
      d_data->v_prev()[tid] = d_data->v_guess()[tid];
    }

    grid.sync();

    // Update positions: q_new = q_prev + h * v (parallel across threads)
    if (tid < d_data->get_n_coef())
    {
      d_data->x12()(tid) = d_data->x12_prev()(tid) + d_data->v_guess()(tid * 3 + 0) * d_data->solver_time_step();
      d_data->y12()(tid) = d_data->y12_prev()(tid) + d_data->v_guess()(tid * 3 + 1) * d_data->solver_time_step();
      d_data->z12()(tid) = d_data->z12_prev()(tid) + d_data->v_guess()(tid * 3 + 2) * d_data->solver_time_step();
    }

    grid.sync();

    // Only thread 0 handles constraint computation and dual variable updates
    if (tid == 0)
    {
      // Compute constraints at new position
      compute_constraint_data(d_data);

      // Dual variable update: lam += rho * h * c(q_new)
      for (int i = 0; i < 12; i++)
      {
        d_data->lambda_guess()[i] += d_data->rho0() * d_data->solver_time_step() * d_data->constraint()[i];
      }
    }

    grid.sync();
  }

  // finally write data back to x12, y12, z12
  if (tid < d_data->get_n_coef())
  {
    d_data->x12()(tid) = d_data->x12_prev()(tid) + d_data->v_guess()(tid * 3 + 0) * d_data->solver_time_step();
    d_data->y12()(tid) = d_data->y12_prev()(tid) + d_data->v_guess()(tid * 3 + 1) * d_data->solver_time_step();
    d_data->z12()(tid) = d_data->z12_prev()(tid) + d_data->v_guess()(tid * 3 + 2) * d_data->solver_time_step();
  }

  grid.sync();
}

void GPU_ANCF3243_Data::OneStepNesterov()
{
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  int threads = 256;

  cudaDeviceProp props;
  HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

  int maxBlocksPerSm = 0;
  HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSm, one_step_nesterov_kernel, threads, 0));
  int maxCoopBlocks = maxBlocksPerSm * props.multiProcessorCount;

  int N = max(n_coef * 3, n_beam * Quadrature::N_TOTAL_QP);
  int blocksNeeded = (N + threads - 1) / threads;
  int blocks = std::min(blocksNeeded, maxCoopBlocks);

  void *args[] = {&d_data};

  HANDLE_ERROR(cudaEventRecord(start));
  HANDLE_ERROR(cudaLaunchCooperativeKernel((void *)one_step_nesterov_kernel, blocks, threads, args));
  HANDLE_ERROR(cudaEventRecord(stop));

  HANDLE_ERROR(cudaDeviceSynchronize());

  float milliseconds = 0;
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

  std::cout << "OneStepNesterov kernel time: " << milliseconds << " ms" << std::endl;

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
}