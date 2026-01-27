/*
 * Unit test for HRZ lumped mass computation.
 * Validates GPU implementation against Python reference.
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "lib_src/elements/FEAT10Data.cuh"
#include "lib_utils/cpu_utils.h"

// Helper function to read CSV file with reference mass values
std::vector<double> read_csv_values(const std::string& filename) {
  std::vector<double> values;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filename << std::endl;
    return values;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty()) {
      values.push_back(std::stod(line));
    }
  }
  return values;
}

class HRZMassTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Bazel runfiles paths (relative to workspace root)
    std::string runfiles_dir = "";

    // Check if running under bazel
    const char* test_srcdir = std::getenv("TEST_SRCDIR");
    const char* test_workspace = std::getenv("TEST_WORKSPACE");
    if (test_srcdir && test_workspace) {
      runfiles_dir = std::string(test_srcdir) + "/" + std::string(test_workspace) + "/";
    }

    // Paths to mesh files
    node_file_ = runfiles_dir + "data/meshes/T10/beam_3x2x1.1.node";
    ele_file_ = runfiles_dir + "data/meshes/T10/beam_3x2x1.1.ele";
    reference_file_ = runfiles_dir + "data/utest/hrz_mass_reference.csv";

    // Material parameters (must match Python reference)
    rho_ = 2700.0;
  }

  std::string node_file_;
  std::string ele_file_;
  std::string reference_file_;
  double rho_;
};

TEST_F(HRZMassTest, CompareWithPythonReference) {
  // Read mesh using shared CPU utilities
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;
  ASSERT_GT(ANCFCPUUtils::FEAT10_read_nodes(node_file_, nodes), 0)
      << "Failed to read nodes from " << node_file_;
  ASSERT_GT(ANCFCPUUtils::FEAT10_read_elements(ele_file_, elements), 0)
      << "Failed to read elements from " << ele_file_;

  int n_nodes = nodes.rows();
  int n_elem = elements.rows();

  std::cout << "Mesh: " << n_nodes << " nodes, " << n_elem << " elements"
            << std::endl;

  // Read reference values from CSV (must exist)
  std::vector<double> reference_mass = read_csv_values(reference_file_);
  ASSERT_FALSE(reference_mass.empty())
      << "Reference mass CSV missing or empty: " << reference_file_;
  ASSERT_EQ(static_cast<size_t>(n_nodes), reference_mass.size())
      << "Reference mass entries (" << reference_mass.size()
      << ") do not match node count (" << n_nodes << ")";

  int n_compare = std::min(static_cast<int>(reference_mass.size()), n_nodes);

  // Setup quadrature points (5-point Keast rule)
  Eigen::VectorXd tet5pt_x(5), tet5pt_y(5), tet5pt_z(5), tet5pt_weights(5);

  // Quadrature points in (xi, eta, zeta) coordinates
  // Point 0: centroid
  tet5pt_x(0) = 0.25;
  tet5pt_y(0) = 0.25;
  tet5pt_z(0) = 0.25;
  // Points 1-4: near vertices
  double a = 0.5, b = 1.0 / 6.0;
  tet5pt_x(1) = a;  tet5pt_y(1) = b;  tet5pt_z(1) = b;
  tet5pt_x(2) = b;  tet5pt_y(2) = a;  tet5pt_z(2) = b;
  tet5pt_x(3) = b;  tet5pt_y(3) = b;  tet5pt_z(3) = a;
  tet5pt_x(4) = b;  tet5pt_y(4) = b;  tet5pt_z(4) = b;

  // Weights
  tet5pt_weights(0) = -4.0 / 5.0 * (1.0 / 6.0);
  tet5pt_weights(1) = 9.0 / 20.0 * (1.0 / 6.0);
  tet5pt_weights(2) = 9.0 / 20.0 * (1.0 / 6.0);
  tet5pt_weights(3) = 9.0 / 20.0 * (1.0 / 6.0);
  tet5pt_weights(4) = 9.0 / 20.0 * (1.0 / 6.0);

  // Extract position vectors
  Eigen::VectorXd h_x12 = nodes.col(0);
  Eigen::VectorXd h_y12 = nodes.col(1);
  Eigen::VectorXd h_z12 = nodes.col(2);

  // Create and setup GPU element data
  GPU_FEAT10_Data element(n_elem, n_nodes);
  element.Initialize();
  element.Setup(tet5pt_x, tet5pt_y, tet5pt_z, tet5pt_weights, h_x12, h_y12,
                h_z12, elements);

  // Set density
  element.SetDensity(rho_);

  // Compute reference gradients (needed for detJ)
  element.CalcDnDuPre();

  // Compute HRZ lumped mass on GPU
  element.CalcLumpedMassHRZ();

  // Retrieve GPU results
  Eigen::VectorXd gpu_mass;
  element.RetrieveLumpedMassToCPU(gpu_mass);

  // Compare results
  double total_mass_gpu = gpu_mass.sum();

  // Expected total mass: rho * volume = 2700 * 6 = 16200 kg
  double expected_total_mass = 16200.0;

  std::cout << "Total mass (GPU): " << total_mass_gpu << " kg" << std::endl;
  std::cout << "Total mass (Expected): " << expected_total_mass << " kg" << std::endl;

  // Check total mass
  double rel_error_total = std::abs(total_mass_gpu - expected_total_mass) / expected_total_mass;
  std::cout << "Relative error in total mass: " << rel_error_total << std::endl;
  EXPECT_LT(rel_error_total, 1e-10) << "Total mass mismatch";

  // Check individual nodal masses against reference (first n_compare nodes)
  double max_rel_error = 0.0;
  int max_error_node = -1;
  for (int i = 0; i < n_compare; i++) {
    double rel_error = std::abs(gpu_mass(i) - reference_mass[i]) / reference_mass[i];
    if (rel_error > max_rel_error) {
      max_rel_error = rel_error;
      max_error_node = i;
    }
    EXPECT_LT(rel_error, 1e-10)
        << "Mass mismatch at node " << i << ": GPU=" << gpu_mass(i)
        << ", Ref=" << reference_mass[i];
  }

  std::cout << "Max relative error (first " << n_compare << " nodes): "
            << max_rel_error << " at node " << max_error_node << std::endl;

  // Check all masses are positive
  for (int i = 0; i < n_nodes; i++) {
    EXPECT_GT(gpu_mass(i), 0.0) << "Non-positive mass at node " << i;
  }

  // Print first few values for visual comparison
  std::cout << "\nFirst 10 nodal masses (GPU vs Reference):" << std::endl;
  for (int i = 0; i < std::min(10, n_compare); i++) {
    std::cout << "  Node " << i << ": GPU=" << gpu_mass(i)
              << ", Ref=" << reference_mass[i]
              << ", Diff=" << (gpu_mass(i) - reference_mass[i]) << std::endl;
  }

  // Cleanup
  element.Destroy();
}

TEST_F(HRZMassTest, TotalMassConservation) {
  // Read mesh
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;
  ASSERT_GT(ANCFCPUUtils::FEAT10_read_nodes(node_file_, nodes), 0)
      << "Failed to read nodes from " << node_file_;
  ASSERT_GT(ANCFCPUUtils::FEAT10_read_elements(ele_file_, elements), 0)
      << "Failed to read elements from " << ele_file_;

  int n_nodes = nodes.rows();
  int n_elem = elements.rows();

  // Setup quadrature
  Eigen::VectorXd tet5pt_x(5), tet5pt_y(5), tet5pt_z(5), tet5pt_weights(5);
  tet5pt_x(0) = 0.25;  tet5pt_y(0) = 0.25;  tet5pt_z(0) = 0.25;
  double a = 0.5, b = 1.0 / 6.0;
  tet5pt_x(1) = a;  tet5pt_y(1) = b;  tet5pt_z(1) = b;
  tet5pt_x(2) = b;  tet5pt_y(2) = a;  tet5pt_z(2) = b;
  tet5pt_x(3) = b;  tet5pt_y(3) = b;  tet5pt_z(3) = a;
  tet5pt_x(4) = b;  tet5pt_y(4) = b;  tet5pt_z(4) = b;
  tet5pt_weights(0) = -4.0 / 5.0 * (1.0 / 6.0);
  for (int i = 1; i < 5; i++) {
    tet5pt_weights(i) = 9.0 / 20.0 * (1.0 / 6.0);
  }

  Eigen::VectorXd h_x12 = nodes.col(0);
  Eigen::VectorXd h_y12 = nodes.col(1);
  Eigen::VectorXd h_z12 = nodes.col(2);

  GPU_FEAT10_Data element(n_elem, n_nodes);
  element.Initialize();
  element.Setup(tet5pt_x, tet5pt_y, tet5pt_z, tet5pt_weights, h_x12, h_y12,
                h_z12, elements);
  element.SetDensity(rho_);
  element.CalcDnDuPre();
  element.CalcLumpedMassHRZ();

  Eigen::VectorXd gpu_mass;
  element.RetrieveLumpedMassToCPU(gpu_mass);

  // Expected total mass: rho * volume
  // For a 3x2x1 beam: volume = 6 m³, mass = 2700 * 6 = 16200 kg
  double expected_total_mass = rho_ * 6.0;
  double actual_total_mass = gpu_mass.sum();

  std::cout << "Expected total mass: " << expected_total_mass << " kg" << std::endl;
  std::cout << "Actual total mass: " << actual_total_mass << " kg" << std::endl;

  double rel_error = std::abs(actual_total_mass - expected_total_mass) / expected_total_mass;
  std::cout << "Relative error: " << rel_error << std::endl;

  EXPECT_LT(rel_error, 1e-10) << "Total mass not conserved";

  element.Destroy();
}
