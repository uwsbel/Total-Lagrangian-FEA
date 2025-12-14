/**
 * ANCF CPU Utility Unit Tests
 *
 * Author: Json Zhou
 * Email:  zzhou292@wisc.edu
 *
 * This file contains unit tests for CPU-side helper routines, including the
 * ANCF3243 coefficient offset calculation and the ANCF3443 shell beam
 * coordinate/element generators. It checks that offsets follow the expected
 * pattern and that generated node coordinates and connectivities match
 * hand-derived reference values.
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "../lib_utils/cpu_utils.h"

// Test fixture class for ANCF tests
class ANCFUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// ========================================
// Tests for calculate_offsets
// ========================================

TEST_F(ANCFUtilsTest, CalculateOffsets_SingleBeam) {
  const int n_beam = 1;

  Eigen::VectorXi offset_start(n_beam), offset_end(n_beam);

  ANCFCPUUtils::ANCF3243_calculate_offsets(n_beam, offset_start, offset_end);

  // Check vector sizes
  EXPECT_EQ(offset_start.size(), n_beam);
  EXPECT_EQ(offset_end.size(), n_beam);

  // Check offset values for single beam
  // offset_start(0) = 0 * 4 = 0
  // offset_end(0) = 0 + 7 = 7
  EXPECT_EQ(offset_start(0), 0);
  EXPECT_EQ(offset_end(0), 7);
}

TEST_F(ANCFUtilsTest, CalculateOffsets_TwoBeams) {
  const int n_beam = 2;

  Eigen::VectorXi offset_start(n_beam), offset_end(n_beam);

  ANCFCPUUtils::ANCF3243_calculate_offsets(n_beam, offset_start, offset_end);

  // Check offset values for two beams
  // Beam 0: offset_start(0) = 0 * 4 = 0, offset_end(0) = 0 + 7 = 7
  // Beam 1: offset_start(1) = 1 * 4 = 4, offset_end(1) = 4 + 7 = 11
  EXPECT_EQ(offset_start(0), 0);
  EXPECT_EQ(offset_end(0), 7);
  EXPECT_EQ(offset_start(1), 4);
  EXPECT_EQ(offset_end(1), 11);
}

TEST_F(ANCFUtilsTest, CalculateOffsets_MultipleBeams) {
  const int n_beam = 5;

  Eigen::VectorXi offset_start(n_beam), offset_end(n_beam);

  ANCFCPUUtils::ANCF3243_calculate_offsets(n_beam, offset_start, offset_end);

  // Check vector sizes
  EXPECT_EQ(offset_start.size(), n_beam);
  EXPECT_EQ(offset_end.size(), n_beam);

  // Check pattern: offset_start(i) = i * 4, offset_end(i) = offset_start(i) + 7
  for (int i = 0; i < n_beam; ++i) {
    EXPECT_EQ(offset_start(i), i * 4);
    EXPECT_EQ(offset_end(i), offset_start(i) + 7);
  }

  // Check specific values
  EXPECT_EQ(offset_start(0), 0);
  EXPECT_EQ(offset_end(0), 7);
  EXPECT_EQ(offset_start(1), 4);
  EXPECT_EQ(offset_end(1), 11);
  EXPECT_EQ(offset_start(2), 8);
  EXPECT_EQ(offset_end(2), 15);
  EXPECT_EQ(offset_start(3), 12);
  EXPECT_EQ(offset_end(3), 19);
  EXPECT_EQ(offset_start(4), 16);
  EXPECT_EQ(offset_end(4), 23);
}

TEST_F(ANCFUtilsTest, CalculateOffsets_VerifySpan) {
  const int n_beam = 3;

  Eigen::VectorXi offset_start(n_beam), offset_end(n_beam);

  ANCFCPUUtils::ANCF3243_calculate_offsets(n_beam, offset_start, offset_end);

  // Verify that each beam spans exactly 8 elements (0-7 inclusive)
  for (int i = 0; i < n_beam; ++i) {
    int span = offset_end(i) - offset_start(i) + 1;
    EXPECT_EQ(span, 8) << "Beam " << i << " should span 8 elements";
  }
}

// ========================================
// Tests for 3443 generate_shell_elements
// ========================================

TEST_F(ANCFUtilsTest, GenerateShellElements3443_2) {
  const int n_beam = 2;  // For your 24 DOF, 6-node, 2-element example

  Eigen::MatrixXi element_connectivity;
  Eigen::VectorXd x12, y12, z12;

  ANCFCPUUtils::ANCF3443_generate_beam_coordinates(n_beam, x12, y12, z12,
                                                   element_connectivity);

  // Expected arrays
  double x12_arr[24] = {
      0.0, 1.0, 0.0, 0.0,  // Node 0
      2.0, 1.0, 0.0, 0.0,  // Node 1
      2.0, 1.0, 0.0, 0.0,  // Node 2
      0.0, 1.0, 0.0, 0.0,  // Node 3
      4.0, 1.0, 0.0, 0.0,  // Node 4
      4.0, 1.0, 0.0, 0.0   // Node 5
  };
  double y12_arr[24] = {
      0.0, 0.0, 1.0, 0.0,  // Node 0
      0.0, 0.0, 1.0, 0.0,  // Node 1
      1.0, 0.0, 1.0, 0.0,  // Node 2
      1.0, 0.0, 1.0, 0.0,  // Node 3
      0.0, 0.0, 1.0, 0.0,  // Node 4
      1.0, 0.0, 1.0, 0.0   // Node 5
  };
  double z12_arr[24] = {
      0.0, 0.0, 0.0, 1.0,  // Node 0
      0.0, 0.0, 0.0, 1.0,  // Node 1
      0.0, 0.0, 0.0, 1.0,  // Node 2
      0.0, 0.0, 0.0, 1.0,  // Node 3
      0.0, 0.0, 0.0, 1.0,  // Node 4
      0.0, 0.0, 0.0, 1.0   // Node 5
  };

  Eigen::VectorXd x12_expected = Eigen::Map<Eigen::VectorXd>(x12_arr, 24);
  Eigen::VectorXd y12_expected = Eigen::Map<Eigen::VectorXd>(y12_arr, 24);
  Eigen::VectorXd z12_expected = Eigen::Map<Eigen::VectorXd>(z12_arr, 24);

  // Check coordinates
  EXPECT_TRUE(x12.isApprox(x12_expected));
  EXPECT_TRUE(y12.isApprox(y12_expected));
  EXPECT_TRUE(z12.isApprox(z12_expected));

  // Check connectivity
  Eigen::MatrixXi connectivity_expected(2, 4);
  connectivity_expected << 0, 1, 2, 3, 1, 4, 5, 2;
  EXPECT_EQ(element_connectivity.rows(), 2);
  EXPECT_EQ(element_connectivity.cols(), 4);
  EXPECT_TRUE(
      (element_connectivity.array() == connectivity_expected.array()).all());
}

TEST_F(ANCFUtilsTest, GenerateShellElements3443_3) {
  const int n_beam = 3;  // For your 24 DOF, 6-node, 2-element example

  Eigen::MatrixXi element_connectivity;
  Eigen::VectorXd x12, y12, z12;

  ANCFCPUUtils::ANCF3443_generate_beam_coordinates(n_beam, x12, y12, z12,
                                                   element_connectivity);

  // Expected arrays
  double x12_arr[32] = {
      0.0, 1.0, 0.0, 0.0,  // Node 0
      2.0, 1.0, 0.0, 0.0,  // Node 1
      2.0, 1.0, 0.0, 0.0,  // Node 2
      0.0, 1.0, 0.0, 0.0,  // Node 3
      4.0, 1.0, 0.0, 0.0,  // Node 4
      4.0, 1.0, 0.0, 0.0,  // Node 5
      6.0, 1.0, 0.0, 0.0,  // Node 6

      6.0, 1.0, 0.0, 0.0  // Node 7
  };
  double y12_arr[32] = {0.0, 0.0, 1.0, 0.0,  // Node 0
                        0.0, 0.0, 1.0, 0.0,  // Node 1
                        1.0, 0.0, 1.0, 0.0,  // Node 2
                        1.0, 0.0, 1.0, 0.0,  // Node 3
                        0.0, 0.0, 1.0, 0.0,  // Node 4
                        1.0, 0.0, 1.0, 0.0,  // Node 5
                        0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};

  double z12_arr[32] = {
      0.0, 0.0, 0.0, 1.0,  // Node 0
      0.0, 0.0, 0.0, 1.0,  // Node 1
      0.0, 0.0, 0.0, 1.0,  // Node 2
      0.0, 0.0, 0.0, 1.0,  // Node 3
      0.0, 0.0, 0.0, 1.0,  // Node 4
      0.0, 0.0, 0.0, 1.0,  // Node 5
      0.0, 0.0, 0.0, 1.0,  // Node 6
      0.0, 0.0, 0.0, 1.0   // Node 7
  };

  Eigen::VectorXd x12_expected = Eigen::Map<Eigen::VectorXd>(x12_arr, 32);
  Eigen::VectorXd y12_expected = Eigen::Map<Eigen::VectorXd>(y12_arr, 32);
  Eigen::VectorXd z12_expected = Eigen::Map<Eigen::VectorXd>(z12_arr, 32);

  // Check coordinates
  EXPECT_TRUE(x12.isApprox(x12_expected));
  EXPECT_TRUE(y12.isApprox(y12_expected));
  EXPECT_TRUE(z12.isApprox(z12_expected));

  // Check connectivity
  Eigen::MatrixXi connectivity_expected(3, 4);
  connectivity_expected << 0, 1, 2, 3, 1, 4, 5, 2, 4, 6, 7, 5;
  EXPECT_EQ(element_connectivity.rows(), 3);
  EXPECT_EQ(element_connectivity.cols(), 4);
  EXPECT_TRUE(
      (element_connectivity.array() == connectivity_expected.array()).all());
}