/**
 * ANCF3243 Beam AdamW Test
 *
 * Author: Json Zhou
 * Email:  zzhou292@wisc.edu
 *
 * This driver sets up a simple 1D ANCF3243 beam, assembles its mass matrix,
 * internal forces, and constraint data on the GPU, and then advances the
 * system using the synchronized AdamW solver. It is primarily used to
 * validate element assembly, constraint handling, and the AdamW time
 * integration loop in isolation from collision.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

#include "../../lib_utils/quadrature_utils.h"
#include "../lib_src/elements/ANCF3243Data.cuh"
#include "../lib_src/solvers/SyncedAdamW.cuh"
#include "../lib_utils/cpu_utils.h"
#include "../lib_utils/mesh_utils.h"

const double E    = 7e8;   // Young's modulus
const double nu   = 0.33;  // Poisson's ratio
const double rho0 = 2700;  // Density

int main() {
  double L = 2.0, W = 1.0, H = 1.0;

  const double E    = 7e8;   // Young's modulus
  const double nu   = 0.33;  // Poisson's ratio
  const double rho0 = 2700;  // Density

  // Create 1D horizontal grid mesh (3 elements, 4 nodes)
  ANCFCPUUtils::GridMeshGenerator grid_gen(3 * L, 0.0, L, true, false);
  grid_gen.generate_mesh();

  int n_nodes    = grid_gen.get_num_nodes();
  int n_elements = grid_gen.get_num_elements();

  std::cout << "Number of nodes: " << n_nodes << std::endl;
  std::cout << "Number of elements: " << n_elements << std::endl;
  std::cout << "Total DOFs: " << 4 * n_nodes << std::endl;

  // initialize GPU data structure
  GPU_ANCF3243_Data gpu_3243_data(n_nodes, n_elements);
  gpu_3243_data.Initialize();

  // Compute B_inv on CPU
  Eigen::MatrixXd h_B_inv(Quadrature::N_SHAPE_3243, Quadrature::N_SHAPE_3243);
  ANCFCPUUtils::ANCF3243_B12_matrix(L, W, H, h_B_inv, Quadrature::N_SHAPE_3243);

  // Generate nodal coordinates using GridMeshGenerator
  Eigen::VectorXd h_x12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_y12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_z12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_x12_jac(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_y12_jac(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_z12_jac(gpu_3243_data.get_n_coef());

  grid_gen.get_coordinates(h_x12, h_y12, h_z12);

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

  // Get element connectivity - using GridMeshGenerator
  Eigen::MatrixXi h_element_connectivity;
  grid_gen.get_element_connectivity(h_element_connectivity);

  // Debug: print element connectivity
  std::cout << "Element connectivity:" << std::endl;
  for (int i = 0; i < h_element_connectivity.rows(); i++) {
    std::cout << "Element " << i << ": [" << h_element_connectivity(i, 0)
              << ", " << h_element_connectivity(i, 1) << "]" << std::endl;
  }

  // ======================================================================

  // set fixed nodal unknowns
  Eigen::VectorXi h_fixed_nodes(4);
  h_fixed_nodes << 0, 1, 2, 3;
  gpu_3243_data.SetNodalFixed(h_fixed_nodes);

  // set external force
  Eigen::VectorXd h_f_ext(gpu_3243_data.get_n_coef() * 3);
  // set external force applied at the end of the beam to be 0,0,3100
  h_f_ext.setZero();
  h_f_ext(3 * gpu_3243_data.get_n_coef() - 10) = 3100.0;
  gpu_3243_data.SetExternalForce(h_f_ext);

  // set up the system
  gpu_3243_data.Setup(L, W, H, h_B_inv, Quadrature::gauss_xi_m_6,
                      Quadrature::gauss_xi_3, Quadrature::gauss_eta_2,
                      Quadrature::gauss_zeta_2, Quadrature::weight_xi_m_6,
                      Quadrature::weight_xi_3, Quadrature::weight_eta_2,
                      Quadrature::weight_zeta_2, h_x12, h_y12, h_z12,
                      h_element_connectivity);

  gpu_3243_data.SetDensity(rho0);
  gpu_3243_data.SetDamping(0.0, 0.0);

  gpu_3243_data.SetSVK(E, nu);

  // ======================================================================

  gpu_3243_data.CalcDsDuPre();
  gpu_3243_data.PrintDsDuPre();
  gpu_3243_data.CalcMassMatrix();

  Eigen::MatrixXd mass_matrix;
  gpu_3243_data.RetrieveMassMatrixToCPU(mass_matrix);

  std::cout << "mass matrix:" << std::endl;
  for (int i = 0; i < mass_matrix.rows(); i++) {
    for (int j = 0; j < mass_matrix.cols(); j++) {
      std::cout << std::setw(10) << std::setprecision(3) << mass_matrix(i, j)
                << " ";
    }
    std::cout << std::endl;
  }

  gpu_3243_data.ConvertToCSRMass();

  std::cout << "done ConvertToCSRMass" << std::endl;

  gpu_3243_data.CalcConstraintData();

  std::cout << "done CalcConstraintData" << std::endl;

  gpu_3243_data.ConvertTOCSRConstraintJac();

  std::cout << "done ConvertTOCSRConstraintJac" << std::endl;

  // // Set highest precision for cout
  std::cout << std::fixed << std::setprecision(17);

  gpu_3243_data.CalcP();
  std::cout << "done calculating p" << std::endl;

  std::vector<std::vector<Eigen::MatrixXd>> p_from_F;
  gpu_3243_data.RetrievePFromFToCPU(p_from_F);
  std::cout << "p from f:" << std::endl;

  for (size_t i = 0; i < p_from_F.size(); i++) {
    std::cout << "Element " << i << ":" << std::endl;
    for (size_t j = 0; j < p_from_F[i].size(); j++)  // quadrature points
    {
      std::cout << "  QP " << j << ":" << std::endl;
      std::cout << p_from_F[i][j] << std::endl;  // 3x3 matrix
      std::cout << std::endl;                    // Extra space between matrices
    }
  }

  gpu_3243_data.CalcInternalForce();
  std::cout << "done calculating internal force" << std::endl;

  Eigen::VectorXd internal_force;
  gpu_3243_data.RetrieveInternalForceToCPU(internal_force);
  std::cout << "internal force:" << std::endl;
  for (int i = 0; i < internal_force.size(); i++) {
    std::cout << internal_force(i) << " ";
  }

  std::cout << std::endl;

  gpu_3243_data.CalcConstraintData();
  std::cout << "done calculating constraint data" << std::endl;

  Eigen::VectorXd constraint;
  gpu_3243_data.RetrieveConstraintDataToCPU(constraint);
  std::cout << "constraint:" << std::endl;
  for (int i = 0; i < constraint.size(); i++) {
    std::cout << constraint(i) << " ";
  }
  std::cout << std::endl;

  Eigen::MatrixXd constraint_jac;
  gpu_3243_data.RetrieveConstraintJacobianToCPU(constraint_jac);
  std::cout << "constraint jacobian:" << std::endl;
  for (int i = 0; i < constraint_jac.rows(); i++) {
    for (int j = 0; j < constraint_jac.cols(); j++) {
      std::cout << constraint_jac(i, j) << " ";
    }
    std::cout << std::endl;
  }

  SyncedAdamWParams params = {2e-4, 0.9,  0.999, 1e-8, 1e-4, 0.998, 1e-1,
                              1e-6, 1e14, 5,     500,  1e-3, 10};

  // for now, n_constraints needs to be explicitly defined
  SyncedAdamWSolver solver(&gpu_3243_data, gpu_3243_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  for (int i = 0; i < 50; i++) {
    solver.Solve();
  }

  Eigen::VectorXd x12, y12, z12;
  gpu_3243_data.RetrievePositionToCPU(x12, y12, z12);

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

  gpu_3243_data.Destroy();

  return 0;
}
