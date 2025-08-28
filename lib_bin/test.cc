#include "../lib_src/GPUMemoryManager.cuh"
#include "../lib_utils/cpu_utils.h"
#include "../lib_utils/quadrature_utils.h"
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <iostream>

const double E = 7e8;     // Young's modulus
const double nu = 0.33;   // Poisson's ratio
const double rho0 = 2700; // Density

// // Device function to compute Jacobian matrix
// __device__ void calc_det_J(double u,
//                            double v,
//                            double w,
//                            const double* B_inv,
//                            const double* x12_jac,
//                            const double* y12_jac,
//                            const double* z12_jac,
//                            double* J_out  // 3x3 matrix output
// ) {
//     // Basis vector derivatives
//     double db_du[N_SHAPE] = {0, 1, 0, 0, v, w, 2 * u, 3 * u * u};
//     double db_dv[N_SHAPE] = {0, 0, 1, 0, u, 0, 0, 0};
//     double db_dw[N_SHAPE] = {0, 0, 0, 1, 0, u, 0, 0};

//     // Shape function derivatives
//     double ds_du[N_SHAPE], ds_dv[N_SHAPE], ds_dw[N_SHAPE];
//     mat_vec_mul8(B_inv, db_du, ds_du);
//     mat_vec_mul8(B_inv, db_dv, ds_dv);
//     mat_vec_mul8(B_inv, db_dw, ds_dw);

//     // Compute Jacobian matrix (3x3)
//     for (int i = 0; i < 3; i++) {
//         for (int j = 0; j < 3; j++) {
//             J_out[i * 3 + j] = 0.0;
//         }
//     }

//     // First column (ds_du)
//     for (int i = 0; i < N_SHAPE; i++) {
//         J_out[0 * 3 + 0] += x12_jac[i] * ds_du[i];
//         J_out[1 * 3 + 0] += y12_jac[i] * ds_du[i];
//         J_out[2 * 3 + 0] += z12_jac[i] * ds_du[i];
//     }

//     // Second column (ds_dv)
//     for (int i = 0; i < N_SHAPE; i++) {
//         J_out[0 * 3 + 1] += x12_jac[i] * ds_dv[i];
//         J_out[1 * 3 + 1] += y12_jac[i] * ds_dv[i];
//         J_out[2 * 3 + 1] += z12_jac[i] * ds_dv[i];
//     }

//     // Third column (ds_dw)
//     for (int i = 0; i < N_SHAPE; i++) {
//         J_out[0 * 3 + 2] += x12_jac[i] * ds_dw[i];
//         J_out[1 * 3 + 2] += y12_jac[i] * ds_dw[i];
//         J_out[2 * 3 + 2] += z12_jac[i] * ds_dw[i];
//     }
// }

// // Kernel to compute detJ_pre
// __global__ void detJ_pre_kernel(double L,
//                                 double W,
//                                 double H,
//                                 const double* gauss_xi,
//                                 const double* gauss_eta,
//                                 const double* gauss_zeta,
//                                 const double* B_inv,
//                                 const double* x12_jac,
//                                 const double* y12_jac,
//                                 const double* z12_jac,
//                                 double* detJ_pre  // output array
// ) {
//     int ixi = blockIdx.x;
//     int ieta = blockIdx.y;
//     int izeta = threadIdx.x;
//     int idx = ixi * N_QP_2 * N_QP_2 + ieta * N_QP_2 + izeta;

//     double xi = gauss_xi[ixi];
//     double eta = gauss_eta[ieta];
//     double zeta = gauss_zeta[izeta];

//     double u = L * xi / 2.0;
//     double v = W * eta / 2.0;
//     double w = H * zeta / 2.0;

//     // Compute Jacobian matrix
//     double J[9];  // 3x3 matrix
//     calc_det_J(u, v, w, B_inv, x12_jac, y12_jac, z12_jac, J);

//     // Compute and store determinant
//     detJ_pre[idx] = det3x3(J);
// }

// Device function to compute deformation gradient
// __device__ void
// compute_deformation_gradient(int elem_idx, int qp_idx, const double
// *ds_du_pre,
//                              const double *x12, const double *y12,
//                              const double *z12, const int *offset_start,
//                              int total_qp, double *F_out) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   // Initialize F to zero
//   for (int i = 0; i < 9; i++) {
//     F_out[elem_idx * total_qp * 9 + qp_idx * 9 + i] = 0.0;
//   }

//   // Get the node indices for this element
//   int node_base = offset_start[elem_idx];

//   // Extract local nodal coordinates (e vectors)
//   double e[8][3]; // 8 nodes, each with 3 coordinates
//   for (int i = 0; i < 8; i++) {
//     e[i][0] = x12[node_base + i]; // x coordinate
//     e[i][1] = y12[node_base + i]; // y coordinate
//     e[i][2] = z12[node_base + i]; // z coordinate
//   }

//   // Compute F = sum_i e_i ⊗ ∇s_i
//   // F is 3x3 matrix stored in row-major order
//   for (int i = 0; i < 8; i++) { // Loop over nodes
//     // Get gradient of shape function i (∇s_i) - this needs proper indexing
//     // Assuming ds_du_pre is laid out as [qp_total][8][3]
//     // You'll need to provide the correct qp_idx for the current quadrature
//     // point
//     double grad_s_i[3];
//     grad_s_i[0] = ds_du_pre[qp_idx * 8 * 3 + i * 3 + 0]; // ∂s_i/∂u
//     grad_s_i[1] = ds_du_pre[qp_idx * 8 * 3 + i * 3 + 1]; // ∂s_i/∂v
//     grad_s_i[2] = ds_du_pre[qp_idx * 8 * 3 + i * 3 + 2]; // ∂s_i/∂w

//     // Compute outer product: e_i ⊗ ∇s_i and add to F
//     for (int row = 0; row < 3; row++) {   // e_i components
//       for (int col = 0; col < 3; col++) { // ∇s_i components
//         F_out[elem_idx * total_qp * 9 + qp_idx * 9 + row * 3 + col] +=
//             e[i][row] * grad_s_i[col];
//       }
//     }
//   }
// }

// __device__ void
// compute_internal_force_qp(int elem_idx, int qp_idx, const double *ds_du_pre,
//                           const double *d_F, const double *weight_u,
//                           const double *weight_v, const double *weight_w,
//                           double L, double W, double H, double mu,
//                           double lambda, int total_qp, double *f_elem_out) {
//   // --- Load ds/du ---
//   double grad_s[8][3];
//   for (int i = 0; i < 8; ++i) {
//     for (int d = 0; d < 3; ++d) {
//       grad_s[i][d] = ds_du_pre[qp_idx * 8 * 3 + i * 3 + d];
//     }
//   }

//   // --- Load F ---
//   double F[3][3];
//   for (int i = 0; i < 3; ++i)
//     for (int j = 0; j < 3; ++j)
//       F[i][j] = d_F[elem_idx * total_qp * 9 + qp_idx * 9 + i * 3 + j];

//   // --- Compute C = F^T * F ---
//   double FtF[3][3] = {0};
//   for (int i = 0; i < 3; ++i)
//     for (int j = 0; j < 3; ++j)
//       for (int k = 0; k < 3; ++k)
//         FtF[i][j] += F[k][i] * F[k][j];

//   // --- trace(F^T F) ---
//   double tr_FtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

//   // 1. Compute Ft (transpose of F)
//   double Ft[3][3];
//   for (int i = 0; i < 3; ++i)
//     for (int j = 0; j < 3; ++j)
//       Ft[i][j] = F[j][i]; // transpose

//   // 2. Compute G = F * Ft
//   double G[3][3] = {0}; // G = F * F^T
//   for (int i = 0; i < 3; ++i)
//     for (int j = 0; j < 3; ++j)
//       for (int k = 0; k < 3; ++k)
//         G[i][j] += F[i][k] * Ft[k][j];

//   // 3. Compute FFF = G * F = (F * Ft) * F
//   double FFF[3][3] = {0};
//   for (int i = 0; i < 3; ++i)
//     for (int j = 0; j < 3; ++j)
//       for (int k = 0; k < 3; ++k)
//         FFF[i][j] += G[i][k] * F[k][j];

//   // --- Compute P ---
//   double P[3][3];
//   double factor = lambda * (0.5 * tr_FtF - 1.5);
//   for (int i = 0; i < 3; ++i)
//     for (int j = 0; j < 3; ++j) {
//       P[i][j] = factor * F[i][j] + mu * (FFF[i][j] - F[i][j]);
//     }

//   // --- Compute internal force f_elem_out --

//   // --- Load scale ---
//   double scale = weight_u[qp_idx / (N_QP_2 * N_QP_2)] *
//                  weight_v[(qp_idx / N_QP_2) % N_QP_2] *
//                  weight_w[qp_idx % N_QP_2];

//   // --- Compute f_i = P * ∇s_i ---
//   for (int i = 0; i < 8; ++i) {
//     double f_i[3] = {0};
//     for (int r = 0; r < 3; ++r)
//       for (int c = 0; c < 3; ++c) {
//         f_i[r] += P[r][c] * grad_s[i][c];
//         if (f_i[r] != 0.0) {
//           printf("f_i[%d][%d] = %f\n", i, r, f_i[r]);
//         }
//       }

//     for (int d = 0; d < 3; ++d) {
//       atomicAdd(&f_elem_out[elem_idx * 8 * 3 + i * 3 + d], f_i[d] * scale);
//     }
//   }
// }

int main() {
  // initialize GPU data structure
  int n_beam = 2;
  GPU_ANCF3243_Data gpu_3243_data(n_beam);
  gpu_3243_data.Initialize();

  double L = 2.0, W = 1.0, H = 1.0;

  const double E = 7e8;     // Young's modulus
  const double nu = 0.33;   // Poisson's ratio
  const double rho0 = 2700; // Density

  std::cout << "Number of beams: " << gpu_3243_data.get_n_beam() << std::endl;
  std::cout << "Total nodes: " << gpu_3243_data.get_n_coef() << std::endl;

  // Compute B_inv on CPU
  Eigen::MatrixXd h_B_inv(Quadrature::N_SHAPE, Quadrature::N_SHAPE);
  ANCFCPUUtils::B12_matrix(2.0, 1.0, 1.0, h_B_inv, Quadrature::N_SHAPE);

  // Generate nodal coordinates for multiple beams - using Eigen vectors
  Eigen::VectorXd h_x12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_y12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_z12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_x12_jac(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_y12_jac(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_z12_jac(gpu_3243_data.get_n_coef());

  ANCFCPUUtils::generate_beam_coordinates(n_beam, h_x12, h_y12, h_z12);

  // print h_x12
  for (int i = 0; i < gpu_3243_data.get_n_coef(); i++) {
    printf("h_x12(%d) = %f\n", i, h_x12(i));
  }

  // print h_y12
  for (int i = 0; i < gpu_3243_data.get_n_coef(); i++) {
    printf("h_y12(%d) = %f\n", i, h_y12(i));
  }

  // print h_z12
  for (int i = 0; i < gpu_3243_data.get_n_coef(); i++) {
    printf("h_z12(%d) = %f\n", i, h_z12(i));
  }

  h_x12_jac = h_x12;
  h_y12_jac = h_y12;
  h_z12_jac = h_z12;

  // Calculate offsets - using Eigen vectors
  Eigen::VectorXi h_offset_start(gpu_3243_data.get_n_beam());
  Eigen::VectorXi h_offset_end(gpu_3243_data.get_n_beam());
  ANCFCPUUtils::calculate_offsets(gpu_3243_data.get_n_beam(), h_offset_start,
                                  h_offset_end);

  gpu_3243_data.Setup(L, W, H, rho0, nu, E, h_B_inv, Quadrature::gauss_xi_m,
                      Quadrature::gauss_xi, Quadrature::gauss_eta,
                      Quadrature::gauss_zeta, Quadrature::weight_xi_m,
                      Quadrature::weight_xi, Quadrature::weight_eta,
                      Quadrature::weight_zeta, h_x12, h_y12, h_z12,
                      h_offset_start, h_offset_end);

  gpu_3243_data.CalcDsDuPre();
  gpu_3243_data.PrintDsDuPre();
  gpu_3243_data.CalcMassMatrix();

  Eigen::MatrixXd mass_matrix;
  gpu_3243_data.RetrieveMassMatrixToCPU(mass_matrix);

  std::cout << "mass matrix:" << std::endl;
  for (int i = 0; i < mass_matrix.rows(); i++) {
    for (int j = 0; j < mass_matrix.cols(); j++) {
      std::cout << mass_matrix(i, j) << " ";
    }
    std::cout << std::endl;
  }
  gpu_3243_data.CalcDeformationGradient();
  gpu_3243_data.CalcPFromF();
  std::cout << "done calculating p from f" << std::endl;

  //    gpu_3243_data.calc_int_force();

  // // Allocate GPU memory for all nodal coordinates
  // double *d_x12_jac, *d_y12_jac, *d_z12_jac;
  // double *d_x12, *d_y12, *d_z12;
  // cudaMalloc(&d_x12_jac, N_COEF * sizeof(double));
  // cudaMalloc(&d_y12_jac, N_COEF * sizeof(double));
  // cudaMalloc(&d_z12_jac, N_COEF * sizeof(double));
  // cudaMalloc(&d_x12, N_COEF * sizeof(double));
  // cudaMalloc(&d_y12, N_COEF * sizeof(double));
  // cudaMalloc(&d_z12, N_COEF * sizeof(double));

  // // Copy all nodal coordinates to GPU
  // cudaMemcpy(d_x12_jac, h_x12.data(), N_COEF * sizeof(double),
  // cudaMemcpyHostToDevice); cudaMemcpy(d_y12_jac, h_y12.data(), N_COEF *
  // sizeof(double), cudaMemcpyHostToDevice); cudaMemcpy(d_z12_jac,
  // h_z12.data(), N_COEF * sizeof(double), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_x12, h_x12.data(), N_COEF * sizeof(double),
  // cudaMemcpyHostToDevice); cudaMemcpy(d_y12, h_y12.data(), N_COEF *
  // sizeof(double), cudaMemcpyHostToDevice); cudaMemcpy(d_z12, h_z12.data(),
  // N_COEF * sizeof(double), cudaMemcpyHostToDevice);

  // // Allocate GPU memory for offset arrays
  // int *d_offset_start, *d_offset_end;
  // cudaMalloc(&d_offset_start, N_BEAM * sizeof(int));
  // cudaMalloc(&d_offset_end, N_BEAM * sizeof(int));

  // // Copy offset arrays to GPU
  // cudaMemcpy(d_offset_start, h_offset_start.data(), N_BEAM * sizeof(int),
  // cudaMemcpyHostToDevice); cudaMemcpy(d_offset_end, h_offset_end.data(),
  // N_BEAM * sizeof(int), cudaMemcpyHostToDevice);

  // // Copy results back to host
  // int* h_node_indices = new int[N_COEF * N_COEF];
  // double* h_node_values = new double[N_COEF * N_COEF];
  // cudaMemcpy(h_node_indices, d_node_indices, N_COEF * N_COEF * sizeof(int),
  // cudaMemcpyDeviceToHost); cudaMemcpy(h_node_values, d_node_values, N_COEF
  // * N_COEF * sizeof(double), cudaMemcpyDeviceToHost);
  // // Print all results for verification
  // std::cout << "\nMass matrix output:" << std::endl;
  // for (int i = 0; i < N_COEF * N_COEF; i++) {
  //     std::cout << h_node_values[i] << std::endl;
  // }

  // std::cout << "Mass terms kernel executed" << std::endl;

  gpu_3243_data.Destroy();

  return 0;
}
