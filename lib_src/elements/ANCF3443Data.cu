#include <cooperative_groups.h>

#include <iomanip>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    ANCF3443Data.cu
 * Brief:   Implements GPU data management and element kernels for ANCF 3443
 *          shell elements. Handles allocation, initialization, and evaluation
 *          of mass, stiffness, internal forces, and constraints for shell
 *          kinematics.
 *==============================================================
 *==============================================================*/

#include "ANCF3443Data.cuh"
#include "ANCF3443DataFunc.cuh"
namespace cg = cooperative_groups;

__device__ __forceinline__ int binary_search_column_csr_mass_3443(
    const int *cols, int n_cols, int target) {
  int left = 0;
  int right = n_cols - 1;
  while (left <= right) {
    int mid = left + ((right - left) >> 1);
    int v = cols[mid];
    if (v == target) return mid;
    if (v < target)
      left = mid + 1;
    else
      right = mid - 1;
  }
  return -1;
}

__global__ void build_mass_keys_3443_kernel(GPU_ANCF3443_Data *d_data,
                                           unsigned long long *d_keys) {
  const int total = d_data->gpu_n_beam() * Quadrature::N_SHAPE_3443 *
                    Quadrature::N_SHAPE_3443;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }

  const int elem = tid / (Quadrature::N_SHAPE_3443 * Quadrature::N_SHAPE_3443);
  const int item_local =
      tid % (Quadrature::N_SHAPE_3443 * Quadrature::N_SHAPE_3443);
  const int i_local = item_local / Quadrature::N_SHAPE_3443;
  const int j_local = item_local % Quadrature::N_SHAPE_3443;

  const int i_global =
      d_data->element_connectivity()(elem, i_local / 4) * 4 + (i_local % 4);
  const int j_global =
      d_data->element_connectivity()(elem, j_local / 4) * 4 + (j_local % 4);

  const unsigned long long key =
      (static_cast<unsigned long long>(static_cast<unsigned int>(i_global))
       << 32) |
      static_cast<unsigned long long>(static_cast<unsigned int>(j_global));
  d_keys[tid] = key;
}

__global__ void decode_mass_keys_3443_kernel(const unsigned long long *d_keys,
                                            int nnz, int *d_csr_columns,
                                            int *d_row_counts) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }

  const unsigned long long key = d_keys[tid];
  const int row = static_cast<int>(key >> 32);
  const int col = static_cast<int>(key & 0xffffffffULL);
  d_csr_columns[tid] = col;
  atomicAdd(d_row_counts + row, 1);
}

__global__ void set_last_offset_3443_kernel(int *d_offsets, int n_rows,
                                            int nnz) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_offsets[n_rows] = nnz;
  }
}

__global__ void ds_du_pre_kernel(double L, double W, double H,
                                 GPU_ANCF3443_Data *d_data) {
  int ixi   = blockIdx.x;
  int ieta  = blockIdx.y;
  int izeta = threadIdx.x;
  int idx   = ixi * Quadrature::N_QP_4 * Quadrature::N_QP_3 +
            ieta * Quadrature::N_QP_3 + izeta;

  double xi   = d_data->gauss_xi()(ixi);
  double eta  = d_data->gauss_eta()(ieta);
  double zeta = d_data->gauss_zeta()(izeta);

  double u = L * xi / 2.0;
  double v = W * eta / 2.0;
  double w = H * zeta / 2.0;

  double db_du[Quadrature::N_SHAPE_3443] = {
      0.0,              // d/du 1
      1.0,              // d/du u
      0.0,              // d/du v
      0.0,              // d/du w
      v,                // d/du uv
      w,                // d/du uw
      0.0,              // d/du vw
      v * w,            // d/du uvw
      2.0 * u,          // d/du u^2
      0.0,              // d/du v^2
      2.0 * u * v,      // d/du u^2 v
      v * v,            // d/du u v^2
      3.0 * u * u,      // d/du u^3
      0.0,              // d/du v^3
      3.0 * u * u * v,  // d/du u^3 v
      v * v * v         // d/du u v^3
  };

  double db_dv[Quadrature::N_SHAPE_3443] = {
      0.0,             // d/dv 1
      0.0,             // d/dv u
      1.0,             // d/dv v
      0.0,             // d/dv w
      u,               // d/dv uv
      0.0,             // d/dv uw
      w,               // d/dv vw
      u * w,           // d/dv uvw
      0.0,             // d/dv u^2
      2.0 * v,         // d/dv v^2
      u * u,           // d/dv u^2 v
      2.0 * u * v,     // d/dv u v^2
      0.0,             // d/dv u^3
      3.0 * v * v,     // d/dv v^3
      u * u * u,       // d/dv u^3 v
      3.0 * u * v * v  // d/dv u v^3
  };

  double db_dw[Quadrature::N_SHAPE_3443] = {
      0.0,    // d/dw 1
      0.0,    // d/dw u
      0.0,    // d/dw v
      1.0,    // d/dw w
      0.0,    // d/dw uv
      u,      // d/dw uw
      v,      // d/dw vw
      u * v,  // d/dw uvw
      0.0,    // d/dw u^2
      0.0,    // d/dw v^2
      0.0,    // d/dw u^2 v
      0.0,    // d/dw u v^2
      0.0,    // d/dw u^3
      0.0,    // d/dw v^3
      0.0,    // d/dw u^3 v
      0.0     // d/dw u v^3
  };

  double ds_du[Quadrature::N_SHAPE_3443], ds_dv[Quadrature::N_SHAPE_3443],
      ds_dw[Quadrature::N_SHAPE_3443];
  ancf3443_mat_vec_mul(d_data->B_inv(), db_du, ds_du);
  ancf3443_mat_vec_mul(d_data->B_inv(), db_dv, ds_dv);
  ancf3443_mat_vec_mul(d_data->B_inv(), db_dw, ds_dw);

  // Store as 8x3 matrix: for each i in 0..7, store ds_du, ds_dv, ds_dw as
  // columns
  for (int i = 0; i < Quadrature::N_SHAPE_3443; ++i) {
    d_data->ds_du_pre(idx)(i, 0) = ds_du[i];
    d_data->ds_du_pre(idx)(i, 1) = ds_dv[i];
    d_data->ds_du_pre(idx)(i, 2) = ds_dw[i];
  }
}

__global__ void mass_matrix_qp_kernel(GPU_ANCF3443_Data *d_data) {
  int n_qp_per_elem =
      Quadrature::N_QP_7 * Quadrature::N_QP_7 * Quadrature::N_QP_3;
  int thread_global = blockIdx.x * blockDim.x + threadIdx.x;
  int elem =
      thread_global / (Quadrature::N_SHAPE_3443 * Quadrature::N_SHAPE_3443);
  int item_local =
      thread_global % (Quadrature::N_SHAPE_3443 * Quadrature::N_SHAPE_3443);
  if (elem >= d_data->gpu_n_beam())
    return;

  for (int qp_local = 0; qp_local < n_qp_per_elem; qp_local++) {
    // Decode qp_local into (ixi, ieta, izeta)
    int ixi   = qp_local / (Quadrature::N_QP_7 * Quadrature::N_QP_3);
    int ieta  = (qp_local / Quadrature::N_QP_3) % Quadrature::N_QP_7;
    int izeta = qp_local % Quadrature::N_QP_3;

    double xi     = d_data->gauss_xi_m()(ixi);
    double eta    = d_data->gauss_eta_m()(ieta);
    double zeta   = d_data->gauss_zeta_m()(izeta);
    double weight = d_data->weight_xi_m()(ixi) * d_data->weight_eta_m()(ieta) *
                    d_data->weight_zeta_m()(izeta);

    // Get local nodal coordinates for this element
    double x_local_arr[Quadrature::N_SHAPE_3443],
        y_local_arr[Quadrature::N_SHAPE_3443],
        z_local_arr[Quadrature::N_SHAPE_3443];
    d_data->x12_elem(elem, x_local_arr);
    d_data->y12_elem(elem, y_local_arr);
    d_data->z12_elem(elem, z_local_arr);
    Eigen::Map<Eigen::VectorXd> x_loc(x_local_arr, Quadrature::N_SHAPE_3443);
    Eigen::Map<Eigen::VectorXd> y_loc(y_local_arr, Quadrature::N_SHAPE_3443);
    Eigen::Map<Eigen::VectorXd> z_loc(z_local_arr, Quadrature::N_SHAPE_3443);

    // Compute shape function at this QP
    double b[Quadrature::N_SHAPE_3443];
    ancf3443_b_vec_xi(xi, eta, zeta, d_data->L(), d_data->W(), d_data->H(), b);

    // Compute s = B_inv @ b
    double s[Quadrature::N_SHAPE_3443];
    ancf3443_mat_vec_mul(d_data->B_inv(), b, s);

    // Compute Jacobian determinant
    double J[9];
    ancf3443_calc_det_J_xi(xi, eta, zeta, d_data->B_inv(), x_loc, y_loc, z_loc,
                           d_data->L(), d_data->W(), d_data->H(), J);
    double detJ = ancf3443_det3x3(J);

    // For each local node, output (global_node, value)
    int i_local = item_local / Quadrature::N_SHAPE_3443;
    int j_local = item_local % Quadrature::N_SHAPE_3443;
    int i_global =
        d_data->element_connectivity()(elem, i_local / 4) * 4 + (i_local % 4);
    int j_global =
        d_data->element_connectivity()(elem, j_local / 4) * 4 + (j_local % 4);

    const double mass_contrib =
        d_data->rho0() * s[i_local] * s[j_local] * weight * detJ;
    const int row_start = d_data->csr_offsets()[i_global];
    const int row_end   = d_data->csr_offsets()[i_global + 1];
    const int n_cols    = row_end - row_start;
    const int local_idx = binary_search_column_csr_mass_3443(
        d_data->csr_columns() + row_start, n_cols, j_global);
    if (local_idx >= 0) {
      atomicAdd(d_data->csr_values() + row_start + local_idx, mass_contrib);
    }
  }
}

__global__ void calc_p_kernel(GPU_ANCF3443_Data *d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx   = thread_idx / Quadrature::N_TOTAL_QP_4_4_3;
  int qp_idx     = thread_idx % Quadrature::N_TOTAL_QP_4_4_3;

  if (elem_idx >= d_data->gpu_n_beam() ||
      qp_idx >= Quadrature::N_TOTAL_QP_4_4_3)
    return;

  // Standalone CalcP: no solver velocity provided
  compute_p(elem_idx, qp_idx, d_data, nullptr, 0.0);
}

void GPU_ANCF3443_Data::CalcP() {
  int threads = 128;
  int blocks  = (n_beam * Quadrature::N_TOTAL_QP_4_4_3 + threads - 1) / threads;
  calc_p_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}

void GPU_ANCF3443_Data::CalcDsDuPre() {
  // Launch kernel
  dim3 blocks_pre(Quadrature::N_QP_4, Quadrature::N_QP_4);
  dim3 threads_pre(Quadrature::N_QP_3);
  ds_du_pre_kernel<<<blocks_pre, threads_pre>>>(2.0, 1.0, 1.0, d_data);
  cudaDeviceSynchronize();
}

void GPU_ANCF3443_Data::PrintDsDuPre() {
  // Allocate host memory for all quadrature points
  const int total_size =
      Quadrature::N_TOTAL_QP_4_4_3 * Quadrature::N_SHAPE_3443 * 3;
  double *h_ds_du_pre_raw = new double[total_size];

  // Copy from device to host
  HANDLE_ERROR(cudaMemcpy(h_ds_du_pre_raw, d_ds_du_pre,
                          total_size * sizeof(double), cudaMemcpyDeviceToHost));

  // Print each quadrature point's matrix
  for (int qp = 0; qp < Quadrature::N_TOTAL_QP_4_4_3; ++qp) {
    std::cout << "\n=== Quadrature Point " << qp << " ===" << std::endl;

    // Create Eigen::Map for this quadrature point's data
    double *qp_data = h_ds_du_pre_raw + qp * Quadrature::N_SHAPE_3443 * 3;
    Eigen::Map<Eigen::MatrixXd> ds_du_matrix(qp_data, Quadrature::N_SHAPE_3443,
                                             3);

    // Print the 16x3 matrix with column headers
    std::cout << "        ds/du       ds/dv       ds/dw" << std::endl;
    for (int i = 0; i < Quadrature::N_SHAPE_3443; ++i) {
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

void GPU_ANCF3443_Data::CalcMassMatrix() {
  if (!is_csr_setup) {
    BuildMassCSRPattern();
  }

  int h_nnz = 0;
  HANDLE_ERROR(cudaMemcpy(&h_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_nnz > 0) {
    HANDLE_ERROR(cudaMemset(d_csr_values, 0,
                            static_cast<size_t>(h_nnz) * sizeof(double)));
  }

  // Mass terms computation
  const int N_OUT =
      n_beam * Quadrature::N_SHAPE_3443 * Quadrature::N_SHAPE_3443;

  // Launch kernel
  int threads = 128;
  int blocks  = (N_OUT + threads - 1) / threads;
  mass_matrix_qp_kernel<<<blocks, threads>>>(d_data);

  cudaDeviceSynchronize();
}

void GPU_ANCF3443_Data::BuildMassCSRPattern() {
  if (is_csr_setup) {
    return;
  }

  const int total_keys =
      n_beam * Quadrature::N_SHAPE_3443 * Quadrature::N_SHAPE_3443;
  unsigned long long *d_keys = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_keys,
                          static_cast<size_t>(total_keys) * sizeof(unsigned long long)));

  {
    constexpr int threads = 256;
    const int blocks = (total_keys + threads - 1) / threads;
    build_mass_keys_3443_kernel<<<blocks, threads>>>(d_data, d_keys);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  thrust::device_ptr<unsigned long long> keys_begin(d_keys);
  thrust::device_ptr<unsigned long long> keys_end = keys_begin + total_keys;
  thrust::sort(thrust::device, keys_begin, keys_end);
  thrust::device_ptr<unsigned long long> keys_unique_end =
      thrust::unique(thrust::device, keys_begin, keys_end);

  const int nnz = static_cast<int>(keys_unique_end - keys_begin);

  HANDLE_ERROR(cudaMalloc((void **)&d_csr_offsets,
                          static_cast<size_t>(n_coef + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&d_csr_columns,
                          static_cast<size_t>(nnz) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&d_csr_values,
                          static_cast<size_t>(nnz) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc((void **)&d_nnz, sizeof(int)));

  int *d_row_counts = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_row_counts, static_cast<size_t>(n_coef) * sizeof(int)));
  HANDLE_ERROR(cudaMemset(d_row_counts, 0, static_cast<size_t>(n_coef) * sizeof(int)));

  {
    constexpr int threads = 256;
    const int blocks = (nnz + threads - 1) / threads;
    decode_mass_keys_3443_kernel<<<blocks, threads>>>(d_keys, nnz, d_csr_columns,
                                                      d_row_counts);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  thrust::device_ptr<int> row_counts_begin(d_row_counts);
  thrust::device_ptr<int> offsets_begin(d_csr_offsets);
  thrust::exclusive_scan(thrust::device, row_counts_begin,
                         row_counts_begin + n_coef, offsets_begin);

  {
    set_last_offset_3443_kernel<<<1, 1>>>(d_csr_offsets, n_coef, nnz);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  HANDLE_ERROR(cudaMemcpy(d_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(d_csr_values, 0, static_cast<size_t>(nnz) * sizeof(double)));

  HANDLE_ERROR(cudaFree(d_row_counts));
  HANDLE_ERROR(cudaFree(d_keys));

  is_csr_setup = true;
  HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3443_Data),
                          cudaMemcpyHostToDevice));
}

namespace {
__global__ void build_constraint_j_csr_3443_kernel(int n_constraint,
                                                   const int* fixed_nodes,
                                                   int* j_offsets,
                                                   int* j_columns,
                                                   double* j_values) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_constraint) {
    return;
  }

  j_offsets[tid] = tid;

  int fixed_idx = tid / 3;
  int dof       = tid % 3;
  int node      = fixed_nodes[fixed_idx];
  j_columns[tid] = node * 3 + dof;
  j_values[tid]  = 1.0;
}

__global__ void build_constraint_jt_row_counts_3443_kernel(int n_constraint,
                                                           const int* fixed_nodes,
                                                           int* row_counts) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_constraint) {
    return;
  }

  int fixed_idx = tid / 3;
  int dof       = tid % 3;
  int node      = fixed_nodes[fixed_idx];
  int row       = node * 3 + dof;
  atomicAdd(&row_counts[row], 1);
}

__global__ void build_constraint_jt_fill_3443_kernel(int n_constraint,
                                                     const int* fixed_nodes,
                                                     const int* offsets,
                                                     int* row_positions,
                                                     int* columns,
                                                     double* values) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_constraint) {
    return;
  }

  int fixed_idx = tid / 3;
  int dof       = tid % 3;
  int node      = fixed_nodes[fixed_idx];
  int row       = node * 3 + dof;

  int pos = atomicAdd(&row_positions[row], 1);
  int out = offsets[row] + pos;

  columns[out] = tid;
  values[out]  = 1.0;
}
}  // namespace

// This function converts the TRANSPOSE of the constraint Jacobian matrix to CSR
// format
void GPU_ANCF3443_Data::ConvertToCSR_ConstraintJacT() {
  if (is_cj_csr_setup) {
    return;
  }
  if (!is_constraints_setup || n_constraint == 0) {
    return;
  }

  const int num_rows = n_coef * 3;
  const int nnz      = n_constraint;

  HANDLE_ERROR(cudaMalloc((void**)&d_cj_csr_offsets,
                          static_cast<size_t>(num_rows + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&d_cj_csr_columns,
                          static_cast<size_t>(nnz) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&d_cj_csr_values,
                          static_cast<size_t>(nnz) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc((void**)&d_cj_nnz, sizeof(int)));

  int* d_row_counts    = nullptr;
  int* d_row_positions = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_row_counts, static_cast<size_t>(num_rows) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_row_positions, static_cast<size_t>(num_rows) * sizeof(int)));
  HANDLE_ERROR(cudaMemset(d_row_counts, 0, static_cast<size_t>(num_rows) * sizeof(int)));

  {
    constexpr int threads = 256;
    const int blocks      = (n_constraint + threads - 1) / threads;
    build_constraint_jt_row_counts_3443_kernel<<<blocks, threads>>>(
        n_constraint, d_fixed_nodes, d_row_counts);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  thrust::device_ptr<int> counts_begin(d_row_counts);
  thrust::device_ptr<int> offsets_begin(d_cj_csr_offsets);
  thrust::exclusive_scan(thrust::device, counts_begin,
                         counts_begin + num_rows, offsets_begin);

  set_last_offset_3443_kernel<<<1, 1>>>(d_cj_csr_offsets, num_rows, nnz);
  HANDLE_ERROR(cudaDeviceSynchronize());

  HANDLE_ERROR(cudaMemset(d_row_positions, 0,
                          static_cast<size_t>(num_rows) * sizeof(int)));

  {
    constexpr int threads = 256;
    const int blocks      = (n_constraint + threads - 1) / threads;
    build_constraint_jt_fill_3443_kernel<<<blocks, threads>>>(
        n_constraint, d_fixed_nodes, d_cj_csr_offsets, d_row_positions,
        d_cj_csr_columns, d_cj_csr_values);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  HANDLE_ERROR(cudaFree(d_row_counts));
  HANDLE_ERROR(cudaFree(d_row_positions));

  HANDLE_ERROR(cudaMemcpy(d_cj_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
  is_cj_csr_setup = true;
  HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3443_Data),
                          cudaMemcpyHostToDevice));
}

// This function converts the Constraint Jacobian matrix J to CSR format
// (Rows = Constraints, Cols = DOFs)
void GPU_ANCF3443_Data::ConvertToCSR_ConstraintJac() {
  if (is_j_csr_setup) {
    return;
  }
  if (!is_constraints_setup || n_constraint == 0) {
    return;
  }

  const int nnz = n_constraint;

  HANDLE_ERROR(cudaMalloc((void**)&d_j_csr_offsets,
                          static_cast<size_t>(n_constraint + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&d_j_csr_columns,
                          static_cast<size_t>(nnz) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&d_j_csr_values,
                          static_cast<size_t>(nnz) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc((void**)&d_j_nnz, sizeof(int)));

  {
    constexpr int threads = 256;
    const int blocks      = (n_constraint + threads - 1) / threads;
    build_constraint_j_csr_3443_kernel<<<blocks, threads>>>(
        n_constraint, d_fixed_nodes, d_j_csr_offsets, d_j_csr_columns,
        d_j_csr_values);
    HANDLE_ERROR(cudaDeviceSynchronize());
    set_last_offset_3443_kernel<<<1, 1>>>(d_j_csr_offsets, n_constraint, nnz);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  HANDLE_ERROR(cudaMemcpy(d_j_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
  is_j_csr_setup = true;
  HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3443_Data),
                          cudaMemcpyHostToDevice));
}

void GPU_ANCF3443_Data::RetrieveMassCSRToCPU(std::vector<int> &offsets,
                                            std::vector<int> &columns,
                                            std::vector<double> &values) {
  offsets.assign(static_cast<size_t>(n_coef) + 1, 0);
  columns.clear();
  values.clear();

  if (!is_csr_setup) {
    return;
  }

  int h_nnz = 0;
  HANDLE_ERROR(cudaMemcpy(&h_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));

  columns.resize(static_cast<size_t>(h_nnz));
  values.resize(static_cast<size_t>(h_nnz));

  HANDLE_ERROR(cudaMemcpy(offsets.data(), d_csr_offsets,
                          static_cast<size_t>(n_coef + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(columns.data(), d_csr_columns,
                          static_cast<size_t>(h_nnz) * sizeof(int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(values.data(), d_csr_values,
                          static_cast<size_t>(h_nnz) * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void GPU_ANCF3443_Data::RetrieveInternalForceToCPU(
    Eigen::VectorXd &internal_force) {
  int expected_size = n_coef * 3;
  internal_force.resize(expected_size);

  HANDLE_ERROR(cudaMemcpy(internal_force.data(), d_f_int,
                          expected_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void GPU_ANCF3443_Data::RetrieveDeformationGradientToCPU(
    std::vector<std::vector<Eigen::MatrixXd>> &deformation_gradient) {
  deformation_gradient.resize(n_beam);
  for (int i = 0; i < n_beam; i++) {
    deformation_gradient[i].resize(Quadrature::N_TOTAL_QP_4_4_3);
    for (int j = 0; j < Quadrature::N_TOTAL_QP_4_4_3; j++) {
      deformation_gradient[i][j].resize(3, 3);
      HANDLE_ERROR(
          cudaMemcpy(deformation_gradient[i][j].data(),
                     d_F + i * Quadrature::N_TOTAL_QP_4_4_3 * 3 * 3 + j * 3 * 3,
                     3 * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    }
  }
}

void GPU_ANCF3443_Data::RetrievePFromFToCPU(
    std::vector<std::vector<Eigen::MatrixXd>> &p_from_F) {
  p_from_F.resize(n_beam);
  for (int i = 0; i < n_beam; i++) {
    p_from_F[i].resize(Quadrature::N_TOTAL_QP_4_4_3);
    for (int j = 0; j < Quadrature::N_TOTAL_QP_4_4_3; j++) {
      p_from_F[i][j].resize(3, 3);
      HANDLE_ERROR(
          cudaMemcpy(p_from_F[i][j].data(),
                     d_P + i * Quadrature::N_TOTAL_QP_4_4_3 * 3 * 3 + j * 3 * 3,
                     3 * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    }
  }
}

void GPU_ANCF3443_Data::RetrieveConstraintDataToCPU(
    Eigen::VectorXd &constraint) {
  int expected_size = n_constraint;
  constraint.resize(expected_size);
  HANDLE_ERROR(cudaMemcpy(constraint.data(), d_constraint,
                          expected_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void GPU_ANCF3443_Data::RetrieveConstraintJacobianToCPU(
    Eigen::MatrixXd &constraint_jac) {
  constraint_jac.resize(n_constraint, n_coef * 3);
  constraint_jac.setZero();

  if (!is_constraints_setup || n_constraint == 0) {
    return;
  }

  std::vector<int> h_fixed(static_cast<size_t>(n_constraint / 3), 0);
  HANDLE_ERROR(cudaMemcpy(h_fixed.data(), d_fixed_nodes,
                          static_cast<size_t>(n_constraint / 3) * sizeof(int),
                          cudaMemcpyDeviceToHost));

  for (int i = 0; i < n_constraint / 3; ++i) {
    int node = h_fixed[static_cast<size_t>(i)];
    constraint_jac(i * 3 + 0, node * 3 + 0) = 1.0;
    constraint_jac(i * 3 + 1, node * 3 + 1) = 1.0;
    constraint_jac(i * 3 + 2, node * 3 + 2) = 1.0;
  }
}

void GPU_ANCF3443_Data::RetrievePositionToCPU(Eigen::VectorXd &x12,
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

__global__ void compute_internal_force_kernel(GPU_ANCF3443_Data *d_data) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx   = thread_idx / Quadrature::N_SHAPE_3443;
  int node_idx   = thread_idx % Quadrature::N_SHAPE_3443;

  if (elem_idx >= d_data->gpu_n_beam() || node_idx >= Quadrature::N_SHAPE_3443)
    return;

  compute_internal_force(elem_idx, node_idx, d_data);
}

void GPU_ANCF3443_Data::CalcInternalForce() {
  int threads = 128;
  int blocks  = (n_beam * Quadrature::N_SHAPE_3443 + threads - 1) / threads;
  compute_internal_force_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}

__global__ void compute_constraint_data_kernel(GPU_ANCF3443_Data *d_data) {
  compute_constraint_data(d_data);
}

void GPU_ANCF3443_Data::CalcConstraintData() {
  if (!is_constraints_setup) {
    std::cerr << "constraint is not set up" << std::endl;
    return;
  }

  int threads = 128;
  int blocks  = (n_beam * Quadrature::N_SHAPE_3443 + threads - 1) / threads;
  compute_constraint_data_kernel<<<blocks, threads>>>(d_data);
  cudaDeviceSynchronize();
}
