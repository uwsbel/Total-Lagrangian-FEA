#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "../lib_src/elements/ANCF3243Data.cuh"
#include "../lib_src/elements/ANCF3443Data.cuh"
#include "../lib_src/elements/FEAT10Data.cuh"
#include "../lib_utils/cpu_utils.h"
#include "../lib_utils/csv_utils.h"

// Test fixture class for FEAT10 tests
class TestSparseMass : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup code that runs before each test
  }

  void TearDown() override {
    // Cleanup code that runs after each test
  }
};

// ========================================
// Tests for calculate_offsets
// ========================================

TEST_F(TestSparseMass, FEA_T10_SparseMassMatrix) {
  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;
  int n_nodes, n_elems;

  n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
      "data/meshes/T10/resolution/beam_3x2x1_res0.1.node", nodes);
  n_elems = ANCFCPUUtils::FEAT10_read_elements(
      "data/meshes/T10/resolution/beam_3x2x1_res0.1.ele", elements);

  std::cout << "mesh read nodes: " << n_nodes << std::endl;
  std::cout << "mesh read elements: " << n_elems << std::endl;

  // print nodes and elements matrix
  std::cout << "nodes matrix:" << std::endl;
  std::cout << nodes << std::endl;
  std::cout << "elements matrix:" << std::endl;
  std::cout << elements << std::endl;

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);

  gpu_t10_data.Initialize();

  // Extract coordinate vectors from nodes matrix
  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);  // X coordinates
    h_y12(i) = nodes(i, 1);  // Y coordinates
    h_z12(i) = nodes(i, 2);  // Z coordinates
  }

  const double E    = 7e8;   // Young's modulus
  const double nu   = 0.33;  // Poisson's ratio
  const double rho0 = 2700;  // Density

  // Get quadrature data from quadrature_utils.h
  const Eigen::VectorXd& tet5pt_x_host       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y_host       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z_host       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights_host = Quadrature::tet5pt_weights;

  // ======================================

  // Find all nodes with x == 0
  std::vector<int> fixed_node_indices;
  for (int i = 0; i < h_x12.size(); ++i) {
    if (std::abs(h_x12(i)) < 1e-8) {  // tolerance for floating point
      fixed_node_indices.push_back(i);
    }
  }

  // Convert to Eigen::VectorXi
  Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(i) = fixed_node_indices[i];
  }

  // print fixed nodes
  std::cout << "Fixed nodes (z == 0):" << std::endl;
  for (int i = 0; i < h_fixed_nodes.size(); ++i) {
    std::cout << h_fixed_nodes(i) << " ";
  }
  std::cout << std::endl;

  // Set fixed nodes
  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  // ====================================

  // Call Setup with all required parameters
  gpu_t10_data.Setup(rho0, nu, E,          // Material properties
                     tet5pt_x_host,        // Quadrature points
                     tet5pt_y_host,        // Quadrature points
                     tet5pt_z_host,        // Quadrature points
                     tet5pt_weights_host,  // Quadrature weights
                     h_x12, h_y12, h_z12,  // Node coordinates
                     elements);            // Element connectivity

  gpu_t10_data.CalcDnDuPre();

  // 2. Retrieve results
  std::vector<std::vector<Eigen::MatrixXd>> ref_grads;
  gpu_t10_data.RetrieveDnDuPreToCPU(ref_grads);

  for (size_t i = 0; i < ref_grads.size(); i++) {
    for (size_t j = 0; j < ref_grads[i].size(); j++) {
      std::cout << ref_grads[i][j] << std::endl;
    }
  }

  gpu_t10_data.CalcMassMatrix();

  std::cout << "done CalcMassMatrix" << std::endl;

  Eigen::MatrixXd mass_matrix;
  gpu_t10_data.RetrieveMassMatrixToCPU(mass_matrix);

  std::cout << "mass_matrix (size: " << mass_matrix.rows() << " x "
            << mass_matrix.cols() << "):" << std::endl;

  // print mass matrix
  std::cout << "mass_matrix (size: " << mass_matrix.rows() << " x "
            << mass_matrix.cols() << "):" << std::endl;
  for (int i = 0; i < mass_matrix.rows(); ++i) {
    for (int j = 0; j < mass_matrix.cols(); ++j) {
      std::cout << std::setw(12) << std::setprecision(6) << std::fixed
                << mass_matrix(i, j) << " ";
    }
    std::cout << std::endl;

    // start converting to csr sparse format
  }

  gpu_t10_data.ConvertToCSRMass();

  gpu_t10_data.CalcConstraintData();

  gpu_t10_data.ConvertTOCSRConstraintJac();
}

TEST_F(TestSparseMass, ANCF_3443_SparseMassMatrix) {
  // initialize GPU data structure
  int n_beam = 2;  // this is working
  GPU_ANCF3443_Data gpu_3443_data(n_beam);
  gpu_3443_data.Initialize();

  double L = 2.0, W = 1.0, H = 1.0;

  const double E    = 7e8;   // Young's modulus
  const double nu   = 0.33;  // Poisson's ratio
  const double rho0 = 2700;  // Density

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

  // set fixed nodal unknowns
  Eigen::VectorXi h_fixed_nodes(8);
  h_fixed_nodes << 0, 1, 2, 3, 12, 13, 14, 15;
  gpu_3443_data.SetNodalFixed(h_fixed_nodes);

  // set external force
  Eigen::VectorXd h_f_ext(gpu_3443_data.get_n_coef() * 3);
  // set external force applied at the end of the beam to be 0,0,3100
  h_f_ext.setZero();
  h_f_ext(3 * gpu_3443_data.get_n_coef() - 4)  = -125.0;
  h_f_ext(3 * gpu_3443_data.get_n_coef() - 10) = 500.0;
  h_f_ext(3 * gpu_3443_data.get_n_coef() - 16) = 125.0;
  h_f_ext(3 * gpu_3443_data.get_n_coef() - 22) = 500.0;
  gpu_3443_data.SetExternalForce(h_f_ext);

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

  gpu_3443_data.ConvertToCSRMass();

  gpu_3443_data.CalcConstraintData();

  gpu_3443_data.ConvertTOCSRConstraintJac();
}

TEST_F(TestSparseMass, ANCF_3243_SparseMassMatrix) {
  // initialize GPU data structure
  int n_beam = 3;  // this is working
  GPU_ANCF3243_Data gpu_3243_data(n_beam);
  gpu_3243_data.Initialize();

  double L = 2.0, W = 1.0, H = 1.0;

  const double E    = 7e8;   // Young's modulus
  const double nu   = 0.33;  // Poisson's ratio
  const double rho0 = 2700;  // Density

  std::cout << "Number of beams: " << gpu_3243_data.get_n_beam() << std::endl;
  std::cout << "Total nodes: " << gpu_3243_data.get_n_coef() << std::endl;

  // Compute B_inv on CPU
  Eigen::MatrixXd h_B_inv(Quadrature::N_SHAPE_3243, Quadrature::N_SHAPE_3243);
  ANCFCPUUtils::ANCF3243_B12_matrix(2.0, 1.0, 1.0, h_B_inv,
                                    Quadrature::N_SHAPE_3243);

  // Generate nodal coordinates for multiple beams - using Eigen vectors
  Eigen::VectorXd h_x12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_y12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_z12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_x12_jac(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_y12_jac(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_z12_jac(gpu_3243_data.get_n_coef());

  ANCFCPUUtils::ANCF3243_generate_beam_coordinates(n_beam, h_x12, h_y12, h_z12);

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
  ANCFCPUUtils::ANCF3243_calculate_offsets(gpu_3243_data.get_n_beam(),
                                           h_offset_start, h_offset_end);

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
  gpu_3243_data.Setup(L, W, H, rho0, nu, E, h_B_inv, Quadrature::gauss_xi_m_6,
                      Quadrature::gauss_xi_3, Quadrature::gauss_eta_2,
                      Quadrature::gauss_zeta_2, Quadrature::weight_xi_m_6,
                      Quadrature::weight_xi_3, Quadrature::weight_eta_2,
                      Quadrature::weight_zeta_2, h_x12, h_y12, h_z12,
                      h_offset_start, h_offset_end);

  // ======================================================================

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

  gpu_3243_data.ConvertToCSRMass();

  gpu_3243_data.CalcConstraintData();

  gpu_3243_data.ConvertTOCSRConstraintJac();
}
