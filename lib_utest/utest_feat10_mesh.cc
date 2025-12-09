/**
 * FEAT10 Mesh I/O Unit Tests
 *
 * Author: Json Zhou
 * Email:  zzhou292@wisc.edu
 *
 * These tests exercise FEAT10 mesh reading utilities by loading beam and
 * bunny tetrahedral meshes from disk and printing basic statistics and
 * matrices. They provide a quick regression check that FEAT10 node/element
 * files are parsed correctly before running larger simulations.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

#include "../../lib_utils/quadrature_utils.h"
#include "../lib_src/elements/FEAT10Data.cuh"
#include "../lib_src/solvers/SyncedAdamW.cuh"
#include "../lib_utils/cpu_utils.h"

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

TEST(Test3243, T10Beam) {
  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;

  int n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
      "data/meshes/T10/beam_3x2x1.1.node", nodes);
  int n_elems = ANCFCPUUtils::FEAT10_read_elements(
      "data/meshes/T10/beam_3x2x1.1.ele", elements);

  std::cout << "beam mesh read nodes: " << n_nodes << std::endl;
  std::cout << "beam mesh read elements: " << n_elems << std::endl;

  // print nodes and elements matrix
  std::cout << "nodes matrix:" << std::endl;
  std::cout << nodes << std::endl;
  std::cout << "elements matrix:" << std::endl;
  std::cout << elements << std::endl;
}

TEST(Test3243, T10Bunny) {
  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;

  int n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
      "data/meshes/T10/bunny_ascii_26.1.node", nodes);
  int n_elems = ANCFCPUUtils::FEAT10_read_elements(
      "data/meshes/T10/bunny_ascii_26.1.ele", elements);

  std::cout << "bunny mesh read nodes: " << n_nodes << std::endl;
  std::cout << "bunny mesh read elements: " << n_elems << std::endl;

  // print nodes and elements matrix
  std::cout << "nodes matrix:" << std::endl;
  std::cout << nodes << std::endl;
  std::cout << "elements matrix:" << std::endl;
  std::cout << elements << std::endl;
}
