#include "../lib_src/GPUMemoryManager.cuh"
#include "../lib_utils/cpu_utils.h"
#include "../lib_utils/quadrature_utils.h"
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <iostream>

const double E = 7e8;     // Young's modulus
const double nu = 0.33;   // Poisson's ratio
const double rho0 = 2700; // Density

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

  gpu_3243_data.CalcInternalForce();
  std::cout << "done calculating internal force" << std::endl;

  Eigen::VectorXd internal_force;
  gpu_3243_data.RetrieveInternalForceToCPU(internal_force);
  std::cout << "internal force:" << std::endl;
  for (int i = 0; i < internal_force.size(); i++) {
    std::cout << internal_force(i) << " ";
  }

  std::cout << std::endl;

  gpu_3243_data.Destroy();

  return 0;
}
