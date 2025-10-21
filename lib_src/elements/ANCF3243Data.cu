#include <cooperative_groups.h>

#include <iomanip>

#include "ANCF3243Data.cuh"
#include "ANCF3243DataFunc.cuh"
namespace cg = cooperative_groups;

// Kernel: one thread per quadrature point, computes 8x3 ds_du_pre
__global__ void ds_du_pre_kernel(double L, double W, double H,
                                 GPU_ANCF3243_Data *d_data) {
  int ixi   = blockIdx.x;
  int ieta  = blockIdx.y;
  int izeta = threadIdx.x;
  int idx   = ixi * Quadrature::N_QP_2 * Quadrature::N_QP_2 +
            ieta * Quadrature::N_QP_2 + izeta;

  double xi   = d_data->gauss_xi()(ixi);
  double eta  = d_data->gauss_eta()(ieta);
  double zeta = d_data->gauss_zeta()(izeta);

  double u = L * xi / 2.0;
  double v = W * eta / 2.0;
  double w = H * zeta / 2.0;

  double db_du[Quadrature::N_SHAPE_3243] = {0, 1, 0, 0, v, w, 2 * u, 3 * u * u};
  double db_dv[Quadrature::N_SHAPE_3243] = {0, 0, 1, 0, u, 0, 0, 0};
  double db_dw[Quadrature::N_SHAPE_3243] = {0, 0, 0, 1, 0, u, 0, 0};

  double ds_du[Quadrature::N_SHAPE_3243], ds_dv[Quadrature::N_SHAPE_3243],
      ds_dw[Quadrature::N_SHAPE_3243];
  ancf3243_mat_vec_mul8(d_data->B_inv(), db_du, ds_du);
  ancf3243_mat_vec_mul8(d_data->B_inv(), db_dv, ds_dv);
  ancf3243_mat_vec_mul8(d_data->B_inv(), db_dw, ds_dw);

  // Store as 8x3 matrix: for each i in 0..7, store ds_du, ds_dv, ds_dw as
  // columns
  for (int i = 0; i < Quadrature::N_SHAPE_3243; ++i) {
    d_data->ds_du_pre(idx)(i, 0) = ds_du[i];
    d_data->ds_du_pre(idx)(i, 1) = ds_dv[i];
    d_data->ds_du_pre(idx)(i, 2) = ds_dw[i];
  }
}

__global__ void mass_matrix_qp_kernel(GPU_ANCF3243_Data *d_data) {
  int n_qp_per_elem =
      Quadrature::N_QP_6 * Quadrature::N_QP_2 * Quadrature::N_QP_2;
  int thread_global = blockIdx.x * blockDim.x + threadIdx.x;
  int elem =
      thread_global / (Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243);
  int item_local =
      thread_global % (Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243);

  if (elem >= d_data->gpu_n_beam())
    return;

  for (int qp_local = 0; qp_local < n_qp_per_elem; qp_local++) {
    // Decode qp_local into (ixi, ieta, izeta)
    int ixi   = qp_local / (Quadrature::N_QP_2 * Quadrature::N_QP_2);
    int ieta  = (qp_local / Quadrature::N_QP_2) % Quadrature::N_QP_2;
    int izeta = qp_local % Quadrature::N_QP_2;

    double xi     = d_data->gauss_xi_m()(ixi);
    double eta    = d_data->gauss_eta()(ieta);
    double zeta   = d_data->gauss_zeta()(izeta);
    double weight = d_data->weight_xi_m()(ixi) * d_data->weight_eta()(ieta) *
                    d_data->weight_zeta()(izeta);

    // Get element's node offset
    int node_offset = d_data->offset_start()(elem);

    // Get local nodal coordinates for this element
    Eigen::Map<Eigen::VectorXd> x_loc = d_data->x12(elem);
    Eigen::Map<Eigen::VectorXd> y_loc = d_data->y12(elem);
    Eigen::Map<Eigen::VectorXd> z_loc = d_data->z12(elem);

    // Compute shape function at this QP
    double b[8];
    ancf3243_b_vec_xi(xi, eta, zeta, d_data->L(), d_data->W(), d_data->H(), b);
    // ancf3243_b_vec_xi(xi, eta, zeta, 2.0, 1.0, 1.0, b);

    // Compute s = B_inv @ b
    double s[8];
    ancf3243_mat_vec_mul8(d_data->B_inv(), b, s);

    // Compute Jacobian determinant
    double J[9];
    ancf3243_calc_det_J_xi(xi, eta, zeta, d_data->B_inv(), x_loc, y_loc, z_loc,
                           d_data->L(), d_data->W(), d_data->H(), J);
    double detJ = ancf3243_det3x3(J);

    // For each local node, output (global_node, value)
    int i_local =
        item_local / Quadrature::N_SHAPE_3243;  // Local node index (0-7)
    int j_local = item_local %
                  Quadrature::N_SHAPE_3243;  // Local shape function index (0-7)
    int i_global = d_data->offset_start()(elem) + i_local;  // Global node index
    int j_global =
        d_data->offset_start()(elem) + j_local;  // Global shape function index

    atomicAdd(d_data->node_values(i_global, j_global),
              d_data->rho0() * s[i_local] * s[j_local] * weight * detJ);
  }
}

__global__ void calc_p_kernel(GPU_ANCF3243_Data *d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx   = thread_idx / Quadrature::N_TOTAL_QP_3_2_2;
  int qp_idx     = thread_idx % Quadrature::N_TOTAL_QP_3_2_2;

  if (elem_idx >= d_data->gpu_n_beam() ||
      qp_idx >= Quadrature::N_TOTAL_QP_3_2_2)
    return;

  compute_p(elem_idx, qp_idx, d_data);
}

void GPU_ANCF3243_Data::CalcP() {
  int threads = 128;
  int blocks  = (n_beam * Quadrature::N_TOTAL_QP_3_2_2 + threads - 1) / threads;
  calc_p_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}

void GPU_ANCF3243_Data::CalcDsDuPre() {
  // Launch kernel
  dim3 blocks_pre(Quadrature::N_QP_3, Quadrature::N_QP_2);
  dim3 threads_pre(Quadrature::N_QP_2);
  ds_du_pre_kernel<<<blocks_pre, threads_pre>>>(2.0, 1.0, 1.0, d_data);
  cudaDeviceSynchronize();
}

void GPU_ANCF3243_Data::PrintDsDuPre() {
  // Allocate host memory for all quadrature points
  const int total_size =
      Quadrature::N_TOTAL_QP_3_2_2 * Quadrature::N_SHAPE_3243 * 3;
  double *h_ds_du_pre_raw = new double[total_size];

  // Copy from device to host
  HANDLE_ERROR(cudaMemcpy(h_ds_du_pre_raw, d_ds_du_pre,
                          total_size * sizeof(double), cudaMemcpyDeviceToHost));

  // Print each quadrature point's matrix
  for (int qp = 0; qp < Quadrature::N_TOTAL_QP_3_2_2; ++qp) {
    std::cout << "\n=== Quadrature Point " << qp << " ===" << std::endl;

    // Create Eigen::Map for this quadrature point's data
    double *qp_data = h_ds_du_pre_raw + qp * Quadrature::N_SHAPE_3243 * 3;
    Eigen::Map<Eigen::MatrixXd> ds_du_matrix(qp_data, Quadrature::N_SHAPE_3243,
                                             3);

    // Print the 8x3 matrix with column headers
    std::cout << "        ds/du       ds/dv       ds/dw" << std::endl;
    for (int i = 0; i < Quadrature::N_SHAPE_3243; ++i) {
      std::cout << "Node " << i << ": ";
      for (int j = 0; j < 3; ++j) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(6)
                  << ds_du_matrix(i, j) << " ";
      }
      std::cout << std::endl;
    }
  }

  delete[] h_ds_du_pre_raw;
}

void GPU_ANCF3243_Data::CalcMassMatrix() {
  // Mass terms computation
  const int N_OUT =
      n_beam * Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243;

  // Launch kernel
  int threads = 128;
  int blocks  = (N_OUT + threads - 1) / threads;
  mass_matrix_qp_kernel<<<blocks, threads>>>(d_data);

  cudaDeviceSynchronize();
}

void GPU_ANCF3243_Data::ConvertToCSRMass() {
  int num_rows = n_coef;
  int num_cols = n_coef;
  int ld       = num_cols;

  int *d_csr_offsets_temp;
  int *d_csr_columns_temp;
  double *d_csr_values_temp;
  int *d_nnz_temp;

  // Device memory management
  double *d_dense = d_node_values;
  HANDLE_ERROR(
      cudaMalloc((void **)&d_csr_offsets_temp, (num_rows + 1) * sizeof(int)));

  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matB;
  cusparseDnMatDescr_t matA;
  void *dBuffer     = NULL;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // Create dense matrix A
  CHECK_CUSPARSE(cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense,
                                     CUDA_R_64F, CUSPARSE_ORDER_ROW));
  // Create sparse matrix B in CSR format
  CHECK_CUSPARSE(cusparseCreateCsr(&matB, num_rows, num_cols, 0,
                                   d_csr_offsets_temp, NULL, NULL,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

  // // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
      handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));
  HANDLE_ERROR(cudaMalloc(&dBuffer, bufferSize));

  // execute Sparse to Dense conversion
  CHECK_CUSPARSE(cusparseDenseToSparse_analysis(
      handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));
  // get number of non-zero elements
  int64_t num_rows_tmp, num_cols_tmp, nnz;
  CHECK_CUSPARSE(
      cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp, &nnz));

  // copy over nnz
  HANDLE_ERROR(cudaMalloc((void **)&d_nnz_temp, sizeof(int)));
  HANDLE_ERROR(
      cudaMemcpy(d_nnz_temp, &nnz, sizeof(int), cudaMemcpyHostToDevice));

  int *h_csr_offsets   = new int[num_rows + 1];
  int *h_csr_columns   = new int[nnz];
  double *h_csr_values = new double[nnz];

  // allocate CSR column indices and values
  HANDLE_ERROR(cudaMalloc((void **)&d_csr_columns_temp, nnz * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&d_csr_values_temp, nnz * sizeof(double)));
  // reset offsets, column indices, and values pointers
  CHECK_CUSPARSE(cusparseCsrSetPointers(matB, d_csr_offsets_temp,
                                        d_csr_columns_temp, d_csr_values_temp));
  CHECK_CUSPARSE(cusparseDenseToSparse_convert(
      handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

  HANDLE_ERROR(cudaMemcpy(h_csr_offsets, d_csr_offsets_temp,
                          (num_rows + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(h_csr_columns, d_csr_columns_temp, nnz * sizeof(int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(h_csr_values, d_csr_values_temp, nnz * sizeof(double),
                          cudaMemcpyDeviceToHost));

  // copy all temp arrays to class members
  HANDLE_ERROR(cudaMalloc((void **)&d_csr_columns, nnz * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&d_csr_values, nnz * sizeof(double)));
  HANDLE_ERROR(
      cudaMalloc((void **)&d_csr_offsets, (num_rows + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&d_nnz, sizeof(int)));

  HANDLE_ERROR(cudaMemcpy(d_csr_offsets, d_csr_offsets_temp,
                          (num_rows + 1) * sizeof(int),
                          cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(d_csr_columns, d_csr_columns_temp, nnz * sizeof(int),
                          cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(d_csr_values, d_csr_values_temp, nnz * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(
      cudaMemcpy(d_nnz, d_nnz_temp, sizeof(int), cudaMemcpyDeviceToDevice));

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroyDnMat(matA));
  CHECK_CUSPARSE(cusparseDestroySpMat(matB));
  CHECK_CUSPARSE(cusparseDestroy(handle));
  HANDLE_ERROR(cudaFree(dBuffer));

  delete[] h_csr_offsets;
  delete[] h_csr_columns;
  delete[] h_csr_values;

  // Free temporary allocations
  HANDLE_ERROR(cudaFree(d_csr_offsets_temp));
  HANDLE_ERROR(cudaFree(d_csr_columns_temp));
  HANDLE_ERROR(cudaFree(d_csr_values_temp));
  HANDLE_ERROR(cudaFree(d_nnz_temp));

  // Flash GPU data back to cpu, update pointer then flash back
  GPU_ANCF3243_Data *h_data_flash =
      (GPU_ANCF3243_Data *)malloc(sizeof(GPU_ANCF3243_Data));
  HANDLE_ERROR(cudaMemcpy(h_data_flash, d_data, sizeof(GPU_ANCF3243_Data),
                          cudaMemcpyDeviceToHost));
  h_data_flash->d_csr_offsets = d_csr_offsets;
  h_data_flash->d_csr_columns = d_csr_columns;
  h_data_flash->d_csr_values  = d_csr_values;
  h_data_flash->d_nnz         = d_nnz;
  HANDLE_ERROR(cudaMemcpy(d_data, h_data_flash, sizeof(GPU_ANCF3243_Data),
                          cudaMemcpyHostToDevice));

  free(h_data_flash);

  is_csr_setup = true;
}

void GPU_ANCF3243_Data::RetrieveMassMatrixToCPU(Eigen::MatrixXd &mass_matrix) {
  // Allocate host memory for all quadrature points
  const int total_size = n_coef * n_coef;

  mass_matrix.resize(n_coef, n_coef);

  // Copy from device to host
  HANDLE_ERROR(cudaMemcpy(mass_matrix.data(), d_node_values,
                          total_size * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_ANCF3243_Data::RetrieveInternalForceToCPU(
    Eigen::VectorXd &internal_force) {
  int expected_size = n_coef * 3;
  internal_force.resize(expected_size);

  HANDLE_ERROR(cudaMemcpy(internal_force.data(), d_f_int,
                          expected_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void GPU_ANCF3243_Data::RetrieveDeformationGradientToCPU(
    std::vector<std::vector<Eigen::MatrixXd>> &deformation_gradient) {
  deformation_gradient.resize(n_beam);
  for (int i = 0; i < n_beam; i++) {
    deformation_gradient[i].resize(Quadrature::N_TOTAL_QP_3_2_2);
    for (int j = 0; j < Quadrature::N_TOTAL_QP_3_2_2; j++) {
      deformation_gradient[i][j].resize(3, 3);
      HANDLE_ERROR(
          cudaMemcpy(deformation_gradient[i][j].data(),
                     d_F + i * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 + j * 3 * 3,
                     3 * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    }
  }
}

void GPU_ANCF3243_Data::RetrievePFromFToCPU(
    std::vector<std::vector<Eigen::MatrixXd>> &p_from_F) {
  p_from_F.resize(n_beam);
  for (int i = 0; i < n_beam; i++) {
    p_from_F[i].resize(Quadrature::N_TOTAL_QP_3_2_2);
    for (int j = 0; j < Quadrature::N_TOTAL_QP_3_2_2; j++) {
      p_from_F[i][j].resize(3, 3);
      HANDLE_ERROR(
          cudaMemcpy(p_from_F[i][j].data(),
                     d_P + i * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 + j * 3 * 3,
                     3 * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    }
  }
}

void GPU_ANCF3243_Data::RetrieveConstraintDataToCPU(
    Eigen::VectorXd &constraint) {
  int expected_size = n_constraint;
  constraint.resize(expected_size);
  HANDLE_ERROR(cudaMemcpy(constraint.data(), d_constraint,
                          expected_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void GPU_ANCF3243_Data::RetrieveConstraintJacobianToCPU(
    Eigen::MatrixXd &constraint_jac) {
  int expected_size = 12 * n_coef * 3;
  constraint_jac.resize(12, n_coef * 3);
  HANDLE_ERROR(cudaMemcpy(constraint_jac.data(), d_constraint_jac,
                          expected_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void GPU_ANCF3243_Data::RetrievePositionToCPU(Eigen::VectorXd &x12,
                                              Eigen::VectorXd &y12,
                                              Eigen::VectorXd &z12) {
  int expected_size = n_coef;
  x12.resize(expected_size);
  y12.resize(expected_size);
  z12.resize(expected_size);
  HANDLE_ERROR(cudaMemcpy(x12.data(), d_x12, expected_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(y12.data(), d_y12, expected_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(z12.data(), d_z12, expected_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

__global__ void compute_internal_force_kernel(GPU_ANCF3243_Data *d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx   = thread_idx / Quadrature::N_SHAPE_3243;
  int node_idx   = thread_idx % Quadrature::N_SHAPE_3243;

  if (elem_idx >= d_data->gpu_n_beam() || node_idx >= Quadrature::N_SHAPE_3243)
    return;

  compute_internal_force(elem_idx, node_idx, d_data);
}

void GPU_ANCF3243_Data::CalcInternalForce() {
  int threads = 128;
  int blocks  = (n_beam * Quadrature::N_SHAPE_3243 + threads - 1) / threads;
  compute_internal_force_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}

__global__ void compute_constraint_data_kernel(GPU_ANCF3243_Data *d_data) {
  compute_constraint_data(d_data);
}

void GPU_ANCF3243_Data::CalcConstraintData() {
  int threads = 128;
  int blocks  = (n_beam * Quadrature::N_SHAPE_3243 + threads - 1) / threads;
  compute_constraint_data_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}
