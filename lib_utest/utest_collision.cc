#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "../lib_src/collision/Broadphase.cuh"
#include "../lib_utils/cpu_utils.h"
#include "../lib_utils/mesh_utils.h"

// Test collision class
class TestCollision : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup code that runs before each test
  }

  void TearDown() override {
    // Cleanup code that runs after each test
  }
};

TEST_F(TestCollision, BroadphaseInitialization) {
  // Initialize the Broadphase collision detection object
  Broadphase broadphase;

  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;

  int n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
      "data/meshes/T10/bunny_ascii_26.1.node", nodes);
  int n_elems = ANCFCPUUtils::FEAT10_read_elements(
      "data/meshes/T10/bunny_ascii_26.1.ele", elements);

  ASSERT_GT(n_nodes, 0);
  ASSERT_GT(n_elems, 0);

  // Initialize broadphase with mesh data
  broadphase.Initialize(nodes, elements);
  broadphase.CreateAABB();
  broadphase.BuildNeighborMap();  // NEW: Build neighbor connectivity
  broadphase.SortAABBs(0);
  broadphase.DetectCollisions();  // Now filters neighbors!
  broadphase.PrintCollisionPairs();
}