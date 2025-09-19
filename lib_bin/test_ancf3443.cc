#include "../../lib_utils/quadrature_utils.h"
#include "../lib_src/elements/ANCF3443Data.cuh"
#include "../lib_src/solvers/SyncedNesterov.cuh"
#include "../lib_utils/cpu_utils.h"
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

const double E = 7e8;     // Young's modulus
const double nu = 0.33;   // Poisson's ratio
const double rho0 = 2700; // Density

int main() {
  // initialize GPU data structure
  int n_beam = 2; // this is working
  GPU_ANCF3443_Data gpu_3443_data(n_beam);
  gpu_3443_data.Initialize();

  double L = 2.0, W = 1.0, H = 1.0;

  const double E = 7e8;     // Young's modulus
  const double nu = 0.33;   // Poisson's ratio
  const double rho0 = 2700; // Density

  std::cout << "Number of beams: " << gpu_3443_data.get_n_beam() << std::endl;
  std::cout << "Total nodes: " << gpu_3443_data.get_n_coef() << std::endl;

  // Compute B_inv on CPU
  Eigen::MatrixXd h_B_inv(Quadrature::N_SHAPE_3443, Quadrature::N_SHAPE_3443);
  ANCFCPUUtils::ANCF3443_B12_matrix(2.0, 1.0, 1.0, h_B_inv,
                                    Quadrature::N_SHAPE_3443);

  std::cout << "B_inv:" << std::endl;
  std::cout << h_B_inv << std::endl;

  // Generate nodal coordinates for multiple beams - using Eigen vectors
  Eigen::VectorXd h_x12(gpu_3443_data.get_n_coef());
  Eigen::VectorXd h_y12(gpu_3443_data.get_n_coef());
  Eigen::VectorXd h_z12(gpu_3443_data.get_n_coef());
  Eigen::MatrixXi element_connectivity(gpu_3443_data.get_n_beam(), 4);
  Eigen::VectorXd h_x12_jac(gpu_3443_data.get_n_coef());
  Eigen::VectorXd h_y12_jac(gpu_3443_data.get_n_coef());
  Eigen::VectorXd h_z12_jac(gpu_3443_data.get_n_coef());

  ANCFCPUUtils::ANCF3443_generate_beam_coordinates(n_beam, h_x12, h_y12, h_z12,
                                                   element_connectivity);

  // print h_x12
  for (int i = 0; i < gpu_3443_data.get_n_coef(); i++) {
    printf("h_x12(%d) = %f\n", i, h_x12(i));
  }

  // print h_y12
  for (int i = 0; i < gpu_3443_data.get_n_coef(); i++) {
    printf("h_y12(%d) = %f\n", i, h_y12(i));
  }

  // print h_z12
  for (int i = 0; i < gpu_3443_data.get_n_coef(); i++) {
    printf("h_z12(%d) = %f\n", i, h_z12(i));
  }

  for (int i = 0; i < gpu_3443_data.get_n_beam(); i++) {
    printf("element_connectivity(%d, :) = %d %d %d %d\n", i,
           element_connectivity(i, 0), element_connectivity(i, 1),
           element_connectivity(i, 2), element_connectivity(i, 3));
  }

  h_x12_jac = h_x12;
  h_y12_jac = h_y12;
  h_z12_jac = h_z12;

  gpu_3443_data.Setup(L, W, H, rho0, nu, E, h_B_inv, Quadrature::gauss_xi_m_7,
                      Quadrature::gauss_eta_m_7, Quadrature::gauss_zeta_m_3,
                      Quadrature::gauss_xi_4, Quadrature::gauss_eta_4,
                      Quadrature::gauss_zeta_3, Quadrature::weight_xi_m_7,
                      Quadrature::weight_eta_m_7, Quadrature::weight_zeta_m_3,
                      Quadrature::weight_xi_4, Quadrature::weight_eta_4,
                      Quadrature::weight_zeta_3, h_x12, h_y12, h_z12,
                      element_connectivity);

  gpu_3443_data.CalcDsDuPre();
  gpu_3443_data.PrintDsDuPre();

  std::cout << "done PrintDsDuPre" << std::endl;

  std::cout << "gpu_3443_data.n_beam" << gpu_3443_data.get_n_beam()
            << std::endl;
  std::cout << "gpu_3443_data.n_coef" << gpu_3443_data.get_n_coef()
            << std::endl;

  gpu_3443_data.CalcMassMatrix();

  std::cout << "done CalcMassMatrix" << std::endl;

  Eigen::MatrixXd mass_matrix;
  gpu_3443_data.RetrieveMassMatrixToCPU(mass_matrix);

  std::cout << "done RetrieveMassMatrixToCPU" << std::endl;

  std::cout << "mass matrix:" << std::endl;
  for (int i = 0; i < mass_matrix.rows(); i++) {
    for (int j = 0; j < mass_matrix.cols(); j++) {
      std::cout << mass_matrix(i, j) << " ";
    }
    std::cout << std::endl;
  }

  // // Set highest precision for cout
  std::cout << std::fixed << std::setprecision(17);

  gpu_3443_data.CalcP();
  std::cout << "done calculating p" << std::endl;

  std::vector<std::vector<Eigen::MatrixXd>> p_from_F;
  gpu_3443_data.RetrievePFromFToCPU(p_from_F);
  std::cout << "p from f:" << std::endl;

  for (int i = 0; i < p_from_F.size(); i++) {
    std::cout << "Element " << i << ":" << std::endl;
    for (int j = 0; j < p_from_F[i].size(); j++) // quadrature points
    {
      std::cout << "  QP " << j << ":" << std::endl;
      std::cout << p_from_F[i][j] << std::endl; // 3x3 matrix
      std::cout << std::endl;                   // Extra space between matrices
    }
  }

  gpu_3443_data.CalcInternalForce();
  std::cout << "done calculating internal force" << std::endl;
  Eigen::VectorXd internal_force;
  gpu_3443_data.RetrieveInternalForceToCPU(internal_force);
  std::cout << "internal force:" << std::endl;
  for (int i = 0; i < internal_force.size(); i++) {
    std::cout << internal_force(i) << " ";
  }

  std::cout << std::endl;

  gpu_3443_data.CalcConstraintData();
  std::cout << "done calculating constraint data" << std::endl;

  Eigen::VectorXd constraint;
  gpu_3443_data.RetrieveConstraintDataToCPU(constraint);
  std::cout << "constraint:" << std::endl;
  for (int i = 0; i < constraint.size(); i++) {
    std::cout << constraint(i) << " ";
  }
  std::cout << std::endl;

  Eigen::MatrixXd constraint_jac;
  gpu_3443_data.RetrieveConstraintJacobianToCPU(constraint_jac);
  std::cout << "constraint jacobian:" << std::endl;
  for (int i = 0; i < constraint_jac.rows(); i++) {
    for (int j = 0; j < constraint_jac.cols(); j++) {
      std::cout << constraint_jac(i, j) << " ";
    }
    std::cout << std::endl;
  }

  // // alpha, solver_rho, inner_tol, outer_tol, max_outer, max_inner,
  // // timestep
  SyncedNesterovParams params = {1.0e-8, 1e14, 1.0e-6, 1.0e-6, 5, 300, 1.0e-3};
  SyncedNesterovSolver solver(&gpu_3443_data, 24);
  solver.Setup();
  solver.SetParameters(&params);
  for (int i = 0; i < 1; i++) {
    solver.Solve();
  }

  Eigen::VectorXd x12, y12, z12;
  gpu_3443_data.RetrievePositionToCPU(x12, y12, z12);

  std::cout << "x12:" << std::endl;
  for (int i = 0; i < x12.size(); i++) {
    std::cout << x12(i) << " ";
  }

  std::cout << std::endl;

  std::cout << "y12:" << std::endl;
  for (int i = 0; i < y12.size(); i++) {
    std::cout << y12(i) << " ";
  }

  std::cout << std::endl;

  std::cout << "z12:" << std::endl;
  for (int i = 0; i < z12.size(); i++) {
    std::cout << z12(i) << " ";
  }

  std::cout << std::endl;

  gpu_3443_data.Destroy();

  return 0;
}
