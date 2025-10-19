#include <cooperative_groups.h>

#include <iomanip>

#include "FEAT10Data.cuh"
#include "FEAT10DataFunc.cuh"

namespace cg = cooperative_groups;

__global__ void dn_du_pre_kernel(GPU_FEAT10_Data *d_data) {
  // Get global thread index
  int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate element index and quadrature point index
  int elem_idx = global_thread_idx / Quadrature::N_QP_T10_5;
  int qp_idx   = global_thread_idx % Quadrature::N_QP_T10_5;

  // Bounds check
  if (elem_idx >= d_data->gpu_n_elem() || qp_idx >= Quadrature::N_QP_T10_5) {
    return;
  }

  // Get quadrature point coordinates (xi, eta, zeta)
  double xi   = d_data->tet5pt_x(qp_idx);  // L2 in Python code
  double eta  = d_data->tet5pt_y(qp_idx);  // L3 in Python code
  double zeta = d_data->tet5pt_z(qp_idx);  // L4 in Python code

  // Compute barycentric coordinates
  double L1   = 1.0 - xi - eta - zeta;
  double L2   = xi;
  double L3   = eta;
  double L4   = zeta;
  double L[4] = {L1, L2, L3, L4};

  // Derivatives of barycentric coordinates (dL matrix from Python)
  double dL[4][3] = {
      {-1.0, -1.0, -1.0},  // dL1/dxi, dL1/deta, dL1/dzeta
      {1.0, 0.0, 0.0},     // dL2/dxi, dL2/deta, dL2/dzeta
      {0.0, 1.0, 0.0},     // dL3/dxi, dL3/deta, dL3/dzeta
      {0.0, 0.0, 1.0}      // dL4/dxi, dL4/deta, dL4/dzeta
  };

  // Compute shape function derivatives dN_dxi for all 10 nodes
  double dN_dxi[10][3];

  // Corner nodes (0-3): dN_dxi[i, :] = (4*L[i]-1)*dL[i, :]
  for (int i = 0; i < 4; i++) {
    double factor = 4.0 * L[i] - 1.0;
    for (int j = 0; j < 3; j++) {
      dN_dxi[i][j] = factor * dL[i][j];
    }
  }

  // Edge nodes (4-9): dN_dxi[k, :] = 4*(L[i]*dL[j, :] + L[j]*dL[i, :])
  // Edge connectivity: [(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]
  int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};

  for (int k = 0; k < 6; k++) {
    int i = edges[k][0];
    int j = edges[k][1];

    for (int d = 0; d < 3; d++) {
      dN_dxi[k + 4][d] = 4.0 * (L[i] * dL[j][d] + L[j] * dL[i][d]);
    }
  }

  // Get element node coordinates for this element
  double X_elem[10][3];  // 10 nodes × 3 coordinates
  for (int node = 0; node < 10; node++) {
    int global_node_idx = d_data->element_connectivity()(elem_idx, node);
    X_elem[node][0]     = d_data->x12()(global_node_idx);  // x coordinate
    X_elem[node][1]     = d_data->y12()(global_node_idx);  // y coordinate
    X_elem[node][2]     = d_data->z12()(global_node_idx);  // z coordinate
  }

  // Compute Jacobian matrix J = Σ(X_node ⊗ dN_dxi)
  double J[3][3] = {{0.0}};  // Initialize to zero
  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {    // Node coordinate components
      for (int j = 0; j < 3; j++) {  // Natural coordinate derivatives
        J[i][j] += X_elem[a][i] * dN_dxi[a][j];  // Outer product
      }
    }
  }

  // Compute determinant of J (3x3 matrix)
  double detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

  // Store the determinant in d_detJ_ref
  d_data->detJ_ref(elem_idx, qp_idx) = detJ;

  // Compute J^T (transpose)
  double JT[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      JT[i][j] = J[j][i];
    }
  }

  // Solve JT * grad_N = dN_dxi for each shape function
  double grad_N[10][3];
  for (int a = 0; a < 10; a++) {
    // Solve 3×3 system: JT * grad_N[a] = dN_dxi[a]
    // You'll need a 3×3 linear solver here (LU decomposition, Gaussian
    // elimination, etc.)
    solve_3x3_system(JT, dN_dxi[a], grad_N[a]);
  }

  // Store the PHYSICAL gradients in grad_N_ref
  for (int i = 0; i < 10; i++) {
    d_data->grad_N_ref(elem_idx, qp_idx)(i, 0) = grad_N[i][0];  // ∂N_i/∂x
    d_data->grad_N_ref(elem_idx, qp_idx)(i, 1) = grad_N[i][1];  // ∂N_i/∂y
    d_data->grad_N_ref(elem_idx, qp_idx)(i, 2) = grad_N[i][2];  // ∂N_i/∂z
  }
}

__global__ void mass_matrix_qp_kernel(GPU_FEAT10_Data *d_data) {
  int n_qp_per_elem = Quadrature::N_QP_T10_5;  // 5 quadrature points
  int thread_global = blockIdx.x * blockDim.x + threadIdx.x;

  // Decode: which element and which (i, j) node pair?
  int elem       = thread_global / (10 * 10);  // 10 nodes per element
  int item_local = thread_global % (10 * 10);

  if (elem >= d_data->gpu_n_elem())
    return;

  // Decode item_local into (i_local, j_local) node indices
  int i_local = item_local / 10;  // Local node i (0-9)
  int j_local = item_local % 10;  // Local node j (0-9)

  // Get global node indices
  int i_global = d_data->element_connectivity()(elem, i_local);
  int j_global = d_data->element_connectivity()(elem, j_local);

  // Get material density
  double rho = d_data->rho0();

  // Accumulator for this (i, j) pair across all QPs
  double mass_contribution = 0.0;

  // Loop over all quadrature points
  for (int qp = 0; qp < n_qp_per_elem; qp++) {
    // Get quadrature point coordinates
    double xi   = d_data->tet5pt_x(qp);
    double eta  = d_data->tet5pt_y(qp);
    double zeta = d_data->tet5pt_z(qp);
    double wq   = d_data->tet5pt_weights(qp);
    // printf("xi: %f, eta: %f, zeta: %f, wq: %f\n", xi, eta, zeta, wq);

    // Compute barycentric coordinates
    double L1   = 1.0 - xi - eta - zeta;
    double L2   = xi;
    double L3   = eta;
    double L4   = zeta;
    double L[4] = {L1, L2, L3, L4};

    // Compute shape functions
    double N[10];

    // Corner nodes (0-3)
    for (int k = 0; k < 4; k++) {
      N[k] = L[k] * (2.0 * L[k] - 1.0);
    }

    // Edge nodes (4-9)
    int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
    for (int k = 0; k < 6; k++) {
      int ii   = edges[k][0];
      int jj   = edges[k][1];
      N[k + 4] = 4.0 * L[ii] * L[jj];
    }

    // Get determinant (pre-computed)
    double detJ = d_data->detJ_ref(elem, qp);

    // Accumulate: rho * N[i] * N[j] * detJ * wq
    mass_contribution += rho * N[i_local] * N[j_local] * detJ * wq;
  }

  atomicAdd(d_data->node_values(i_global, j_global), mass_contribution);
}

void GPU_FEAT10_Data::CalcDnDuPre() {
  int total_threads = n_elem * Quadrature::N_QP_T10_5;

  int threads_per_block = 128;  // or another suitable block size
  int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  dn_du_pre_kernel<<<blocks, threads_per_block>>>(d_data);
  cudaDeviceSynchronize();
}

__global__ void calc_p_kernel(GPU_FEAT10_Data *d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx   = thread_idx / Quadrature::N_QP_T10_5;
  int qp_idx     = thread_idx % Quadrature::N_QP_T10_5;

  if (elem_idx >= d_data->gpu_n_elem() || qp_idx >= Quadrature::N_QP_T10_5)
    return;

  feat10_compute_p(elem_idx, qp_idx, d_data);
}

void GPU_FEAT10_Data::CalcP() {
  int threads = 128;
  int blocks  = (n_elem * Quadrature::N_QP_T10_5 + threads - 1) / threads;
  calc_p_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}

__global__ void compute_internal_force_kernel(GPU_FEAT10_Data *d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx =
      thread_idx / Quadrature::N_NODE_T10_10;  // 10 nodes per element
  int node_local =
      thread_idx % Quadrature::N_NODE_T10_10;  // Local node index (0-9)

  if (elem_idx >= d_data->gpu_n_elem() ||
      node_local >= Quadrature::N_NODE_T10_10)
    return;

  feat10_compute_internal_force(elem_idx, node_local, d_data);
}

void GPU_FEAT10_Data::CalcInternalForce() {
  int threads = 128;
  int blocks  = (n_elem * Quadrature::N_NODE_T10_10 + threads - 1) / threads;
  compute_internal_force_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}

void GPU_FEAT10_Data::CalcMassMatrix() {
  // Launch: n_elem × 10 × 10 threads
  int total_threads     = n_elem * 10 * 10;
  int threads_per_block = 128;
  int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  mass_matrix_qp_kernel<<<blocks, threads_per_block>>>(d_data);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void GPU_FEAT10_Data::ConvertToCSRMass() {
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
  GPU_FEAT10_Data *h_data_flash =
      (GPU_FEAT10_Data *)malloc(sizeof(GPU_FEAT10_Data));
  HANDLE_ERROR(cudaMemcpy(h_data_flash, d_data, sizeof(GPU_FEAT10_Data),
                          cudaMemcpyDeviceToHost));
  h_data_flash->d_csr_offsets = d_csr_offsets;
  h_data_flash->d_csr_columns = d_csr_columns;
  h_data_flash->d_csr_values  = d_csr_values;
  h_data_flash->d_nnz         = d_nnz;
  HANDLE_ERROR(cudaMemcpy(d_data, h_data_flash, sizeof(GPU_FEAT10_Data),
                          cudaMemcpyHostToDevice));

  free(h_data_flash);
}

void GPU_FEAT10_Data::RetrieveDetJToCPU(
    std::vector<std::vector<double>> &detJ) {
  detJ.resize(n_elem);
  for (int elem_idx = 0; elem_idx < n_elem; elem_idx++) {
    detJ[elem_idx].resize(Quadrature::N_QP_T10_5);
    HANDLE_ERROR(cudaMemcpy(
        detJ[elem_idx].data(), d_detJ_ref + elem_idx * Quadrature::N_QP_T10_5,
        Quadrature::N_QP_T10_5 * sizeof(double), cudaMemcpyDeviceToHost));
  }
}

void GPU_FEAT10_Data::RetrieveDnDuPreToCPU(
    std::vector<std::vector<Eigen::MatrixXd>> &dn_du_pre) {
  // Resize to [n_elem][N_QP_T10_5]
  dn_du_pre.resize(n_elem);

  for (int elem_idx = 0; elem_idx < n_elem; elem_idx++) {
    dn_du_pre[elem_idx].resize(Quadrature::N_QP_T10_5);

    for (int qp_idx = 0; qp_idx < Quadrature::N_QP_T10_5; qp_idx++) {
      // Each QP matrix: 10 × 3 (10 shape functions × 3 derivatives)
      dn_du_pre[elem_idx][qp_idx].resize(10, 3);

      // Calculate offset for this specific element + QP
      int offset = (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 10 * 3;
      int size   = 10 * 3 * sizeof(double);

      HANDLE_ERROR(cudaMemcpy(dn_du_pre[elem_idx][qp_idx].data(),
                              d_grad_N_ref + offset, size,
                              cudaMemcpyDeviceToHost));
    }
  }
}

void GPU_FEAT10_Data::RetrieveMassMatrixToCPU(Eigen::MatrixXd &mass_matrix) {
  int total_size = n_coef * n_coef;
  mass_matrix.resize(n_coef, n_coef);
  HANDLE_ERROR(cudaMemcpy(mass_matrix.data(), d_node_values,
                          total_size * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_FEAT10_Data::RetrievePFromFToCPU(
    std::vector<std::vector<Eigen::MatrixXd>> &p_from_F) {
  // Resize to [n_elem][N_QP_T10_5]
  p_from_F.resize(n_elem);

  for (int elem_idx = 0; elem_idx < n_elem; elem_idx++) {
    p_from_F[elem_idx].resize(Quadrature::N_QP_T10_5);

    for (int qp_idx = 0; qp_idx < Quadrature::N_QP_T10_5; qp_idx++) {
      // Each P matrix: 3 × 3 (first Piola-Kirchhoff stress tensor)
      p_from_F[elem_idx][qp_idx].resize(3, 3);

      // Calculate offset for this specific element + QP
      // P is stored as [elem][qp](i,j) where i,j are 0,1,2
      int offset = (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 3 * 3;
      int size   = 3 * 3 * sizeof(double);

      HANDLE_ERROR(cudaMemcpy(p_from_F[elem_idx][qp_idx].data(), d_P + offset,
                              size, cudaMemcpyDeviceToHost));
    }
  }
}

void GPU_FEAT10_Data::RetrieveInternalForceToCPU(
    Eigen::VectorXd &internal_force) {
  // Resize to total DOFs (3 * number of nodes)
  int total_dofs = 3 * n_coef;
  internal_force.resize(total_dofs);

  // Copy from device to host
  HANDLE_ERROR(cudaMemcpy(internal_force.data(), d_f_int,
                          total_dofs * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_FEAT10_Data::RetrieveExternalForceToCPU(
    Eigen::VectorXd &external_force) {
  // Resize to total DOFs (3 * number of nodes)
  int total_dofs = 3 * n_coef;
  external_force.resize(total_dofs);

  // Copy from device to host
  HANDLE_ERROR(cudaMemcpy(external_force.data(), d_f_ext,
                          total_dofs * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPU_FEAT10_Data::RetrievePositionToCPU(Eigen::VectorXd &x12,
                                            Eigen::VectorXd &y12,
                                            Eigen::VectorXd &z12) {
  // Resize to total number of nodes
  int total_nodes = n_coef;
  x12.resize(total_nodes);
  y12.resize(total_nodes);
  z12.resize(total_nodes);

  // Copy from device to host
  HANDLE_ERROR(cudaMemcpy(x12.data(), d_h_x12, total_nodes * sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(y12.data(), d_h_y12, total_nodes * sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(z12.data(), d_h_z12, total_nodes * sizeof(double),
                          cudaMemcpyDeviceToHost));
}