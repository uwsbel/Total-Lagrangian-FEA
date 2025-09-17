#include "../lib_utils/cpu_utils.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

// Test fixture class for ANCF tests
class ANCFUtilsTest : public ::testing::Test
{
protected:
  void SetUp() override {}

  void TearDown() override {}
};

// ========================================
// Tests for calculate_offsets
// ========================================

TEST_F(ANCFUtilsTest, CalculateOffsets_SingleBeam)
{
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

TEST_F(ANCFUtilsTest, CalculateOffsets_TwoBeams)
{
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

TEST_F(ANCFUtilsTest, CalculateOffsets_MultipleBeams)
{
  const int n_beam = 5;

  Eigen::VectorXi offset_start(n_beam), offset_end(n_beam);

  ANCFCPUUtils::ANCF3243_calculate_offsets(n_beam, offset_start, offset_end);

  // Check vector sizes
  EXPECT_EQ(offset_start.size(), n_beam);
  EXPECT_EQ(offset_end.size(), n_beam);

  // Check pattern: offset_start(i) = i * 4, offset_end(i) = offset_start(i) + 7
  for (int i = 0; i < n_beam; ++i)
  {
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

TEST_F(ANCFUtilsTest, CalculateOffsets_VerifySpan)
{
  const int n_beam = 3;

  Eigen::VectorXi offset_start(n_beam), offset_end(n_beam);

  ANCFCPUUtils::ANCF3243_calculate_offsets(n_beam, offset_start, offset_end);

  // Verify that each beam spans exactly 8 elements (0-7 inclusive)
  for (int i = 0; i < n_beam; ++i)
  {
    int span = offset_end(i) - offset_start(i) + 1;
    EXPECT_EQ(span, 8) << "Beam " << i << " should span 8 elements";
  }
}
