#include <gtest/gtest.h>

#include <Eigen/Dense>

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

TEST_F(TestSparseMass, test0) {
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
}