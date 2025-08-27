#include "../lib_src/GPUMemoryManager.cuh"
#include "../lib_utils/cpu_utils.h"
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

// Basic math test (just to verify gtest is working)
TEST(BasicTest, OnePlusOneEqualsTwo) { EXPECT_EQ(1 + 1, 2); }
