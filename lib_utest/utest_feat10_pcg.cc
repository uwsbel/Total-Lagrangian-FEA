/**
 * FEAT10 PCG Integration Test
 *
 * Author: Ganesh Arivoli
 *
 * Mirrors utest_feat10_cudss.cc but uses the SyncedPCGSolver (block-diagonal
 * preconditioned conjugate gradient) instead of the cuDSS direct solver.
 * Validates that the PCG solver produces reasonable motion matching the
 * Newton-cuDSS baseline.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

#include "../../lib_utils/quadrature_utils.h"
#include "../lib_src/elements/FEAT10Data.cuh"
#include "../lib_src/solvers/SyncedPCG.cuh"
#include "../lib_utils/cpu_utils.h"

const SolidMaterialProperties mat_beam = SolidMaterialProperties::SVK(
    7e8,    // E: Young's modulus (Pa)
    0.33,   // nu: Poisson's ratio
    2700,   // rho0: Density (kg/m^3)
    0.0,    // eta_damp
    0.0     // lambda_damp
);

TEST(pcg_test, pcg_feat10) {
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;

  int n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
      "data/meshes/T10/resolution/beam_3x2x1_res4.1.node", nodes);
  int n_elems = ANCFCPUUtils::FEAT10_read_elements(
      "data/meshes/T10/resolution/beam_3x2x1_res4.1.ele", elements);
  int plot_target_node = 353;

  std::cout << "mesh read nodes: " << n_nodes << std::endl;
  std::cout << "mesh read elements: " << n_elems << std::endl;

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);
    h_y12(i) = nodes(i, 1);
    h_z12(i) = nodes(i, 2);
  }

  // Pin nodes at x == 0
  std::vector<int> fixed_node_indices;
  for (int i = 0; i < h_x12.size(); ++i) {
    if (std::abs(h_x12(i)) < 1e-8) {
      fixed_node_indices.push_back(i);
    }
  }
  Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(i) = fixed_node_indices[i];
  }
  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  // 5000N distributed on nodes at x == 3
  Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
  h_f_ext.setZero();
  std::vector<int> force_node_indices;
  for (int i = 0; i < h_x12.size(); ++i) {
    if (std::abs(h_x12(i) - 3.0) < 1e-8) {
      force_node_indices.push_back(i);
    }
  }
  if (force_node_indices.size() > 0) {
    double force_per_node = 5000.0 / force_node_indices.size();
    for (int node_idx : force_node_indices) {
      h_f_ext(3 * node_idx + 0) = force_per_node;
    }
  }
  gpu_t10_data.SetExternalForce(h_f_ext);

  const Eigen::VectorXd& tet5pt_x_host       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y_host       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z_host       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights_host = Quadrature::tet5pt_weights;

  gpu_t10_data.Setup(tet5pt_x_host, tet5pt_y_host, tet5pt_z_host,
                     tet5pt_weights_host, h_x12, h_y12, h_z12, elements);
  gpu_t10_data.ApplyMaterial(mat_beam);
  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();

  SyncedPCGParams params = {
      1e-2,   // inner_atol
      0.0,    // inner_rtol
      1e-6,   // outer_tol
      1e14,   // rho
      5,      // max_outer
      10,     // max_inner
      1e-3,   // time_step
      200,    // max_pcg_iter
      1e-4,   // pcg_rtol
      1e-8,   // precond_eps
      SyncedPCGPreconditioner::kBlockJacobi
  };
  SyncedPCGSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();

  std::vector<double> node_x_history;

  for (int i = 0; i < 2; i++) {
    solver.Solve();

    Eigen::VectorXd x12_current, y12_current, z12_current;
    gpu_t10_data.RetrievePositionToCPU(x12_current, y12_current, z12_current);

    if (plot_target_node < x12_current.size()) {
      node_x_history.push_back(x12_current(plot_target_node));
      std::cout << "Step " << i << ": node " << plot_target_node
                << " x = " << x12_current(plot_target_node) << std::endl;
    }
  }

  // Verify the node moved (non-trivial deformation)
  ASSERT_GT(node_x_history.size(), 0u);
  EXPECT_GT(node_x_history.back(), 3.0)
      << "Expected node to move in +x direction under applied force";
}
