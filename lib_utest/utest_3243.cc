#include "../lib_src/GPUMemoryManager.cuh"
#include "../lib_utils/cpu_utils.h"
#include "../lib_utils/csv_utils.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

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
  int n_beam = 2;
  GPU_ANCF3243_Data gpu_3243_data(n_beam);
  gpu_3243_data.Initialize();

  double L = 2.0, W = 1.0, H = 1.0;

  const double E = 7e8;     // Young's modulus
  const double nu = 0.33;   // Poisson's ratio
  const double rho0 = 2700; // Density

  Eigen::MatrixXd h_B_inv(Quadrature::N_SHAPE, Quadrature::N_SHAPE);
  ANCFCPUUtils::B12_matrix(2.0, 1.0, 1.0, h_B_inv, Quadrature::N_SHAPE);

  Eigen::VectorXd h_x12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_y12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_z12(gpu_3243_data.get_n_coef());

  ANCFCPUUtils::generate_beam_coordinates(n_beam, h_x12, h_y12, h_z12);

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
  gpu_3243_data.CalcMassMatrix();

  Eigen::MatrixXd computed_mass_matrix;
  gpu_3243_data.RetrieveMassMatrixToCPU(computed_mass_matrix);

  Eigen::MatrixXd expected_mass_matrix;
  std::string csv_filepath = "data/utest/mass_matrix_2_beam.csv";
  ASSERT_TRUE(CSVUtils::loadMatrixCSV(expected_mass_matrix, csv_filepath));

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
  int n_beam = 3;
  GPU_ANCF3243_Data gpu_3243_data(n_beam);
  gpu_3243_data.Initialize();

  double L = 2.0, W = 1.0, H = 1.0;

  const double E = 7e8;     // Young's modulus
  const double nu = 0.33;   // Poisson's ratio
  const double rho0 = 2700; // Density

  Eigen::MatrixXd h_B_inv(Quadrature::N_SHAPE, Quadrature::N_SHAPE);
  ANCFCPUUtils::B12_matrix(2.0, 1.0, 1.0, h_B_inv, Quadrature::N_SHAPE);

  Eigen::VectorXd h_x12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_y12(gpu_3243_data.get_n_coef());
  Eigen::VectorXd h_z12(gpu_3243_data.get_n_coef());

  ANCFCPUUtils::generate_beam_coordinates(n_beam, h_x12, h_y12, h_z12);

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
  gpu_3243_data.CalcMassMatrix();

  Eigen::MatrixXd computed_mass_matrix;
  gpu_3243_data.RetrieveMassMatrixToCPU(computed_mass_matrix);

  Eigen::MatrixXd expected_mass_matrix;
  std::string csv_filepath = "data/utest/mass_matrix_3_beam.csv";
  ASSERT_TRUE(CSVUtils::loadMatrixCSV(expected_mass_matrix, csv_filepath));

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