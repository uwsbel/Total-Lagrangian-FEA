/**
 * FEAT10 Beam Resolution Study (VBD)
 *
 * Author: Json Zhou
 * Email:  zzhou292@wisc.edu
 *
 * This driver mirrors the FEAT10 Newton resolution study but advances the
 * cantilever beam using the synchronized VBD (Vertex Block Descent) solver.
 * It varies mesh resolution, applies distributed end loads, and records the
 * displacement history of a target node to CSV.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../../lib_utils/quadrature_utils.h"
#include "../lib_src/elements/FEAT10Data.cuh"
#include "../lib_src/solvers/SyncedVBD.cuh"
#include "../lib_utils/cpu_utils.h"

const double E    = 7e8;   // Young's modulus
const double nu   = 0.33;  // Poisson's ratio
const double rho0 = 2700;  // Density

enum MESH_RESOLUTION { RES_0, RES_2, RES_4, RES_8, RES_16, RES_32 };

enum MATERIAL_MODEL { MAT_SVK, MAT_MOONEY_RIVLIN };

int main() {
  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;
  int plot_target_node;
  int n_nodes, n_elems;

  MESH_RESOLUTION resolution = RES_8;  // Use moderate resolution for testing

  MATERIAL_MODEL material = MAT_SVK;

  if (resolution == RES_0) {
    n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
        "data/meshes/T10/resolution/beam_3x2x1_res0.1.node", nodes);
    n_elems = ANCFCPUUtils::FEAT10_read_elements(
        "data/meshes/T10/resolution/beam_3x2x1_res0.1.ele", elements);
    plot_target_node = 23;
  } else if (resolution == RES_2) {
    n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
        "data/meshes/T10/resolution/beam_3x2x1_res2.1.node", nodes);
    n_elems = ANCFCPUUtils::FEAT10_read_elements(
        "data/meshes/T10/resolution/beam_3x2x1_res2.1.ele", elements);
    plot_target_node = 89;
  } else if (resolution == RES_4) {
    n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
        "data/meshes/T10/resolution/beam_3x2x1_res4.1.node", nodes);
    n_elems = ANCFCPUUtils::FEAT10_read_elements(
        "data/meshes/T10/resolution/beam_3x2x1_res4.1.ele", elements);
    plot_target_node = 353;
  } else if (resolution == RES_8) {
    n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
        "data/meshes/T10/resolution/beam_3x2x1_res8.1.node", nodes);
    n_elems = ANCFCPUUtils::FEAT10_read_elements(
        "data/meshes/T10/resolution/beam_3x2x1_res8.1.ele", elements);
    plot_target_node = 1408;
  } else if (resolution == RES_16) {
    n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
        "data/meshes/T10/resolution/beam_3x2x1_res16.1.node", nodes);
    n_elems = ANCFCPUUtils::FEAT10_read_elements(
        "data/meshes/T10/resolution/beam_3x2x1_res16.1.ele", elements);
    plot_target_node = 5630;
  } else if (resolution == RES_32) {
    n_nodes = ANCFCPUUtils::FEAT10_read_nodes(
        "data/meshes/T10/resolution/beam_3x2x1_res32.1.node", nodes);
    n_elems = ANCFCPUUtils::FEAT10_read_elements(
        "data/meshes/T10/resolution/beam_3x2x1_res32.1.ele", elements);
    plot_target_node = 22529;
  }

  std::cout << "mesh read nodes: " << n_nodes << std::endl;
  std::cout << "mesh read elements: " << n_elems << std::endl;

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);

  std::cout << "gpu_t10_data created" << std::endl;

  gpu_t10_data.Initialize();

  std::cout << "gpu_t10_data initialized" << std::endl;

  // Extract coordinate vectors from nodes matrix
  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);  // X coordinates
    h_y12(i) = nodes(i, 1);  // Y coordinates
    h_z12(i) = nodes(i, 2);  // Z coordinates
  }

  // ==========================================================================

  // Find all nodes with x == 0
  std::vector<int> fixed_node_indices;
  for (int i = 0; i < h_x12.size(); ++i) {
    if (std::abs(h_x12(i)) < 1e-8) {  // tolerance for floating point
      fixed_node_indices.push_back(i);
    }
  }

  // Convert to Eigen::VectorXi
  Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(i) = fixed_node_indices[i];
  }

  // print fixed nodes
  std::cout << "Fixed nodes (x == 0): " << h_fixed_nodes.size() << " nodes"
            << std::endl;

  // Set fixed nodes
  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  // set external force
  // set 5000N force in x direction for all nodes with x = 3
  Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
  h_f_ext.setZero();

  // Find all nodes with x == 3
  std::vector<int> force_node_indices;
  for (int i = 0; i < h_x12.size(); ++i) {
    if (std::abs(h_x12(i) - 3.0) < 1e-8) {  // tolerance for floating point
      force_node_indices.push_back(i);
    }
  }

  // Distribute 5000N equally across these nodes in x direction
  if (force_node_indices.size() > 0) {
    double force_per_node = 5000.0 / force_node_indices.size();
    for (int node_idx : force_node_indices) {
      h_f_ext(3 * node_idx + 0) = force_per_node;  // x direction
    }
  }

  gpu_t10_data.SetExternalForce(h_f_ext);

  // Get quadrature data from quadrature_utils.h
  const Eigen::VectorXd& tet5pt_x_host       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y_host       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z_host       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights_host = Quadrature::tet5pt_weights;

  // Call Setup with all required parameters
  gpu_t10_data.Setup(tet5pt_x_host, tet5pt_y_host, tet5pt_z_host,
                     tet5pt_weights_host, h_x12, h_y12, h_z12, elements);

  gpu_t10_data.SetDensity(rho0);
  gpu_t10_data.SetDamping(0.0, 0.0);

  if (material == MAT_SVK) {
    gpu_t10_data.SetSVK(E, nu);
    std::cout << "Material: SVK" << std::endl;
  } else {
    const double mu    = E / (2.0 * (1.0 + nu));
    const double K     = E / (3.0 * (1.0 - 2.0 * nu));
    const double kappa = 1.5 * K;
    const double mu10  = 0.30 * mu;
    const double mu01  = 0.20 * mu;
    gpu_t10_data.SetMooneyRivlin(mu10, mu01, kappa);
    std::cout << "Material: Mooney-Rivlin" << std::endl;
  }

  // =========================================================================

  gpu_t10_data.CalcDnDuPre();
  std::cout << "gpu_t10_data dndu pre complete" << std::endl;

  gpu_t10_data.CalcMassMatrix();
  std::cout << "done CalcMassMatrix" << std::endl;

  gpu_t10_data.CalcConstraintData();
  std::cout << "done CalcConstraintData" << std::endl;

  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  std::cout << "done ConvertToCSR_ConstraintJacT" << std::endl;

  gpu_t10_data.BuildConstraintJacobianCSR();
  std::cout << "done BuildConstraintJacobianCSR" << std::endl;

  // calculate p
  gpu_t10_data.CalcP();
  std::cout << "done CalcP" << std::endl;

  // calculate internal force
  gpu_t10_data.CalcInternalForce();
  std::cout << "done CalcInternalForce" << std::endl;

  // Create VBD solver with parameters
  // VBD parameters: inner_tol, inner_rtol, outer_tol, rho, max_outer, max_inner,
  //                 time_step, omega, hess_eps, convergence_check_interval
  SyncedVBDParams params = {
      1e-4,   // inner_tol
      1e-4,   // inner_rtol
      1e-6,   // outer_tol
      1e12,   // rho (ALM penalty)
      5,      // max_outer
      500,     // max_inner (VBD sweeps per outer iteration)
      1e-3,   // time_step
      1.5,    // omega (relaxation factor)
      1e-12,  // hess_eps (regularization)
      50       // convergence_check_interval
  };

  SyncedVBDSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);

  // Initialize coloring (done once before simulation)
  solver.InitializeColoring();
  solver.InitializeMassDiagBlocks();

  // Vector to store x position of target node at each step
  std::vector<double> node_x_history;

  std::cout << "\n=== Starting VBD simulation ===" << std::endl;

  for (int i = 0; i < 50; i++) {
    solver.Solve();

    // Retrieve current positions
    Eigen::VectorXd x12_current, y12_current, z12_current;
    gpu_t10_data.RetrievePositionToCPU(x12_current, y12_current, z12_current);

    if (plot_target_node < x12_current.size()) {
      node_x_history.push_back(x12_current(plot_target_node));
      std::cout << "Step " << i << ": node " << plot_target_node
                << " x = " << x12_current(plot_target_node) << std::endl;
    }
  }

  // Write to CSV file
  std::ofstream csv_file("node_x_history_vbd.csv");
  csv_file << std::fixed << std::setprecision(17);
  csv_file << "step,x_position\n";
  for (size_t i = 0; i < node_x_history.size(); i++) {
    csv_file << i << "," << node_x_history[i] << "\n";
  }
  csv_file.close();
  std::cout << "Wrote node " << plot_target_node
            << " x-position history to node_x_history_vbd.csv" << std::endl;

  // Set highest precision for cout
  std::cout << std::fixed << std::setprecision(17);

  Eigen::VectorXd x12, y12, z12;
  gpu_t10_data.RetrievePositionToCPU(x12, y12, z12);

  std::cout << "\nFinal positions (first 10 nodes):" << std::endl;
  for (int i = 0; i < std::min(10, n_nodes); i++) {
    std::cout << "Node " << i << ": (" << x12(i) << ", " << y12(i) << ", "
              << z12(i) << ")" << std::endl;
  }

  gpu_t10_data.Destroy();

  std::cout << "\ngpu_t10_data destroyed" << std::endl;

  return 0;
}
