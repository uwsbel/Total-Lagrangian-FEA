/**
 * FEAT10 Beam Resolution Study (AdamW)
 *
 * Author: Json Zhou
 * Email:  zzhou292@wisc.edu
 *
 * This driver runs a cantilever FEAT10 beam at multiple mesh resolutions
 * (RES_0, RES_2, RES_4, RES_8, RES_16) using the synchronized AdamW solver.
 * It applies distributed end loads, tracks the displacement of a selected
 * node over time, and writes its motion to CSV for mesh-convergence
 * analysis.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../../lib_utils/quadrature_utils.h"
#include "../lib_src/elements/FEAT10Data.cuh"
#include "../lib_src/solvers/SyncedAdamW.cuh"
#include "../lib_utils/cpu_utils.h"

const double E    = 7e8;   // Young's modulus
const double nu   = 0.33;  // Poisson's ratio
const double rho0 = 2700;  // Density

enum MESH_RESOLUTION { RES_0, RES_2, RES_4, RES_8, RES_16 };

int main() {
  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;
  int plot_target_node;
  int n_nodes, n_elems;

  MESH_RESOLUTION resolution = RES_0;

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
  }

  std::cout << "mesh read nodes: " << n_nodes << std::endl;
  std::cout << "mesh read elements: " << n_elems << std::endl;

  // print nodes and elements matrix
  std::cout << "nodes matrix:" << std::endl;
  std::cout << nodes << std::endl;
  std::cout << "elements matrix:" << std::endl;
  std::cout << elements << std::endl;

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
  std::cout << "Fixed nodes (z == 0):" << std::endl;
  for (int i = 0; i < h_fixed_nodes.size(); ++i) {
    std::cout << h_fixed_nodes(i) << " ";
  }
  std::cout << std::endl;

  // Set fixed nodes
  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  // set external force
  // set 5000N force in x direction for all nodes with x = 3(count all number of
  // nodes and equally distribute)
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
  gpu_t10_data.Setup(rho0, nu, E, 0.0, 0.0,  // Material properties + damping
                     tet5pt_x_host,          // Quadrature points
                     tet5pt_y_host,          // Quadrature points
                     tet5pt_z_host,          // Quadrature points
                     tet5pt_weights_host,    // Quadrature weights
                     h_x12, h_y12, h_z12,    // Node coordinates
                     elements);              // Element connectivity

  // =========================================================================

  gpu_t10_data.CalcDnDuPre();

  std::cout << "gpu_t10_data dndu pre complete" << std::endl;

  // 2. Retrieve results
  std::vector<std::vector<Eigen::MatrixXd>> ref_grads;
  gpu_t10_data.RetrieveDnDuPreToCPU(ref_grads);

  std::cout << "ref_grads:" << std::endl;
  for (size_t i = 0; i < ref_grads.size(); i++) {
    for (size_t j = 0; j < ref_grads[i].size(); j++) {
      std::cout << ref_grads[i][j] << std::endl;
    }
  }
  std::cout << "done retrieving ref_grads" << std::endl;

  std::vector<std::vector<double>> detJ;
  gpu_t10_data.RetrieveDetJToCPU(detJ);

  std::cout << "detJ:" << std::endl;
  for (size_t i = 0; i < detJ.size(); i++) {
    for (size_t j = 0; j < detJ[i].size(); j++) {
      std::cout << detJ[i][j] << std::endl;
    }
  }
  std::cout << "done retrieving detJ" << std::endl;

  gpu_t10_data.CalcMassMatrix();

  std::cout << "done CalcMassMatrix" << std::endl;

  Eigen::MatrixXd mass_matrix;
  gpu_t10_data.RetrieveMassMatrixToCPU(mass_matrix);

  std::cout << "mass_matrix (size: " << mass_matrix.rows() << " x "
            << mass_matrix.cols() << "):" << std::endl;

  gpu_t10_data.ConvertToCSRMass();

  std::cout << "done ConvertToCSRMass" << std::endl;

  gpu_t10_data.CalcConstraintData();

  std::cout << "done CalcConstraintData" << std::endl;

  gpu_t10_data.ConvertTOCSRConstraintJac();

  std::cout << "done ConvertTOCSRConstraintJac" << std::endl;

  // // Use Eigen's IOFormat for cleaner output
  Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
  std::cout << mass_matrix.format(CleanFmt) << std::endl;

  std::cout << "\ndone retrieving mass_matrix" << std::endl;

  // calculate p
  gpu_t10_data.CalcP();

  std::cout << "done CalcP" << std::endl;

  // retrieve p
  std::vector<std::vector<Eigen::MatrixXd>> p_from_F;
  gpu_t10_data.RetrievePFromFToCPU(p_from_F);

  std::cout << "P matrices (First Piola-Kirchhoff stress):" << std::endl;
  for (size_t elem = 0; elem < p_from_F.size(); elem++) {
    std::cout << "Element " << elem << ":" << std::endl;
    for (size_t qp = 0; qp < p_from_F[elem].size(); qp++) {
      std::cout << "  Quadrature Point " << qp << ":" << std::endl;
      std::cout << p_from_F[elem][qp] << std::endl;
    }
  }
  std::cout << "done retrieving P matrices" << std::endl;

  // calculate internal force
  gpu_t10_data.CalcInternalForce();
  std::cout << "done CalcInternalForce" << std::endl;

  // retrieve internal force
  Eigen::VectorXd f_int;
  gpu_t10_data.RetrieveInternalForceToCPU(f_int);
  std::cout << "Internal force vector (size: " << f_int.size()
            << "):" << std::endl;
  std::cout << f_int.transpose() << std::endl;
  std::cout << "done retrieving internal force vector" << std::endl;

  SyncedAdamWParams params = {2e-4, 0.9,  0.999, 1e-8, 1e-4, 0.995, 1e-1,
                              1e-6, 1e14, 5,     500,  1e-3, 20};
  SyncedAdamWSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);

  // Vector to store x position of node 353 at each step
  std::vector<double> node_x_history;

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
  std::ofstream csv_file("node_x_history.csv");
  csv_file << std::fixed << std::setprecision(17);
  csv_file << "step,x_position\n";
  for (size_t i = 0; i < node_x_history.size(); i++) {
    csv_file << i << "," << node_x_history[i] << "\n";
  }
  csv_file.close();
  std::cout << "Wrote node " << plot_target_node
            << " x-position history to node_x_history.csv" << std::endl;

  // // Set highest precision for cout
  std::cout << std::fixed << std::setprecision(17);

  Eigen::VectorXd x12, y12, z12;
  gpu_t10_data.RetrievePositionToCPU(x12, y12, z12);

  std::cout << "x12:" << std::endl;
  for (int i = 0; i < x12.size(); i++) {
    std::cout << x12(i) << " ";
  }

  std::cout << std::endl;

  std::cout << "y12:" << std::endl;
  for (int i = 0; i < y12.size(); i++) {
    std::cout << y12(i) << " ";
  }

  std::cout << std::endl;

  std::cout << "z12:" << std::endl;
  for (int i = 0; i < z12.size(); i++) {
    std::cout << z12(i) << " ";
  }

  std::cout << std::endl;

  gpu_t10_data.Destroy();

  std::cout << "gpu_t10_data destroyed" << std::endl;

  return 0;
}
