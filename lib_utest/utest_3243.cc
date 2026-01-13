/**
 * ANCF3243 Mass Matrix Unit Tests
 *
 * Author: Json Zhou
 * Email:  zzhou292@wisc.edu
 *
 * This test suite verifies GPU ANCF3243 element mass matrix assembly against
 * CSV reference results for 2-beam and 3-beam configurations generated from
 * the CPU utilities. It checks symmetry, positive definiteness, and
 * entry-wise agreement within a small tolerance.
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "../lib_src/elements/ANCF3243Data.cuh"
#include "../lib_utils/cpu_utils.h"
#include "../lib_utils/csv_utils.h"
#include "../lib_utils/mesh_utils.h"

// Test fixture class for ANCF tests
class Test3243 : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup code that runs before each test
  }

  void TearDown() override {
    // Cleanup code that runs after each test
  }
};

TEST(Test3243, MassMatrix2Beams) {
  double L = 2.0, W = 1.0, H = 1.0;

  // Create 1D horizontal grid mesh (2 elements, 3 nodes)
  ANCFCPUUtils::GridMeshGenerator grid_gen(2 * L, 0.0, L, true, false);
  grid_gen.generate_mesh();

  int n_nodes    = grid_gen.get_num_nodes();
  int n_elements = grid_gen.get_num_elements();

  GPU_ANCF3243_Data gpu_3243_data(n_nodes, n_elements);
  gpu_3243_data.Initialize();

  const double E    = 7e8;   // Young's modulus
  const double nu   = 0.33;  // Poisson's ratio
  const double rho0 = 2700;  // Density

  Eigen::VectorXd h_x12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_y12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_z12(gpu_3243_data.get_n_coef());

  grid_gen.get_coordinates(h_x12, h_y12, h_z12);

  Eigen::MatrixXi h_element_connectivity;
  grid_gen.get_element_connectivity(h_element_connectivity);

  gpu_3243_data.Setup(L, W, H, Quadrature::gauss_xi_m_6,
                      Quadrature::gauss_xi_3, Quadrature::gauss_eta_2,
                      Quadrature::gauss_zeta_2, Quadrature::weight_xi_m_6,
                      Quadrature::weight_xi_3, Quadrature::weight_eta_2,
                      Quadrature::weight_zeta_2, h_x12, h_y12, h_z12,
                      h_element_connectivity);

  gpu_3243_data.SetDensity(rho0);
  gpu_3243_data.SetDamping(0.0, 0.0);

  gpu_3243_data.SetSVK(E, nu);

  gpu_3243_data.CalcDsDuPre();
  gpu_3243_data.CalcMassMatrix();

  Eigen::MatrixXd expected_mass_matrix;
  std::string csv_filepath = "data/utest/mass_matrix_2_beam.csv";
  ASSERT_TRUE(CSVUtils::loadMatrixCSV(expected_mass_matrix, csv_filepath));

  std::vector<int> offsets;
  std::vector<int> columns;
  std::vector<double> values;
  gpu_3243_data.RetrieveMassCSRToCPU(offsets, columns, values);

  Eigen::MatrixXd computed_mass_matrix(expected_mass_matrix.rows(),
                                       expected_mass_matrix.cols());
  computed_mass_matrix.setZero();

  for (int i = 0; i < gpu_3243_data.get_n_coef(); ++i) {
    const int start = offsets[static_cast<size_t>(i)];
    const int end   = offsets[static_cast<size_t>(i) + 1];
    for (int p = start; p < end; ++p) {
      const int j = columns[static_cast<size_t>(p)];
      computed_mass_matrix(i, j) = values[static_cast<size_t>(p)];
    }
  }

  ASSERT_EQ(computed_mass_matrix.rows(), expected_mass_matrix.rows());
  ASSERT_EQ(computed_mass_matrix.cols(), expected_mass_matrix.cols());

  double tolerance = 1e-4;

  for (int i = 0; i < computed_mass_matrix.rows(); ++i) {
    for (int j = 0; j < computed_mass_matrix.cols(); ++j) {
      EXPECT_NEAR(computed_mass_matrix(i, j), expected_mass_matrix(i, j),
                  tolerance);
    }
  }

  EXPECT_GT(computed_mass_matrix.determinant(), 0);

  Eigen::MatrixXd diff =
      computed_mass_matrix - computed_mass_matrix.transpose();
  EXPECT_LT(diff.norm(), tolerance);

  gpu_3243_data.Destroy();
}

TEST(Test3243, MassMatrix3Beams) {
  double L = 2.0, W = 1.0, H = 1.0;

  // Create 1D horizontal grid mesh (3 elements, 4 nodes)
  ANCFCPUUtils::GridMeshGenerator grid_gen(3 * L, 0.0, L, true, false);
  grid_gen.generate_mesh();

  int n_nodes    = grid_gen.get_num_nodes();
  int n_elements = grid_gen.get_num_elements();

  GPU_ANCF3243_Data gpu_3243_data(n_nodes, n_elements);
  gpu_3243_data.Initialize();

  const double E    = 7e8;   // Young's modulus
  const double nu   = 0.33;  // Poisson's ratio
  const double rho0 = 2700;  // Density

  // B_inv is derived from (L,W,H) and computed inside Setup().

  Eigen::VectorXd h_x12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_y12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_z12(gpu_3243_data.get_n_coef());

  grid_gen.get_coordinates(h_x12, h_y12, h_z12);

  Eigen::MatrixXi h_element_connectivity;
  grid_gen.get_element_connectivity(h_element_connectivity);

  gpu_3243_data.Setup(L, W, H, Quadrature::gauss_xi_m_6,
                      Quadrature::gauss_xi_3, Quadrature::gauss_eta_2,
                      Quadrature::gauss_zeta_2, Quadrature::weight_xi_m_6,
                      Quadrature::weight_xi_3, Quadrature::weight_eta_2,
                      Quadrature::weight_zeta_2, h_x12, h_y12, h_z12,
                      h_element_connectivity);

  gpu_3243_data.SetDensity(rho0);
  gpu_3243_data.SetDamping(0.0, 0.0);

  gpu_3243_data.SetSVK(E, nu);

  gpu_3243_data.CalcDsDuPre();
  gpu_3243_data.CalcMassMatrix();

  Eigen::MatrixXd expected_mass_matrix;
  std::string csv_filepath = "data/utest/mass_matrix_3_beam.csv";
  ASSERT_TRUE(CSVUtils::loadMatrixCSV(expected_mass_matrix, csv_filepath));

  std::vector<int> offsets;
  std::vector<int> columns;
  std::vector<double> values;
  gpu_3243_data.RetrieveMassCSRToCPU(offsets, columns, values);

  Eigen::MatrixXd computed_mass_matrix(expected_mass_matrix.rows(),
                                       expected_mass_matrix.cols());
  computed_mass_matrix.setZero();

  for (int i = 0; i < gpu_3243_data.get_n_coef(); ++i) {
    const int start = offsets[static_cast<size_t>(i)];
    const int end   = offsets[static_cast<size_t>(i) + 1];
    for (int p = start; p < end; ++p) {
      const int j = columns[static_cast<size_t>(p)];
      computed_mass_matrix(i, j) = values[static_cast<size_t>(p)];
    }
  }

  ASSERT_EQ(computed_mass_matrix.rows(), expected_mass_matrix.rows());
  ASSERT_EQ(computed_mass_matrix.cols(), expected_mass_matrix.cols());

  double tolerance = 1e-4;

  for (int i = 0; i < computed_mass_matrix.rows(); ++i) {
    for (int j = 0; j < computed_mass_matrix.cols(); ++j) {
      EXPECT_NEAR(computed_mass_matrix(i, j), expected_mass_matrix(i, j),
                  tolerance);
    }
  }

  EXPECT_GT(computed_mass_matrix.determinant(), 0);

  Eigen::MatrixXd diff =
      computed_mass_matrix - computed_mass_matrix.transpose();
  EXPECT_LT(diff.norm(), tolerance);

  gpu_3243_data.Destroy();
}
