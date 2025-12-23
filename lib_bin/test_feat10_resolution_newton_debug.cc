/**
 * FEAT10 Beam Resolution Study (Newton) - Debug Version
 *
 * This version removes excessive print statements before the solver execution
 * to provide a cleaner output for debugging solver convergence and performance.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../../lib_utils/quadrature_utils.h"
#include "../lib_src/elements/FEAT10Data.cuh"
#include "../lib_src/solvers/SyncedNewton.cuh"
#include "../lib_utils/cpu_utils.h"

const double E    = 7e8;   // Young's modulus
const double nu   = 0.33;  // Poisson's ratio
const double rho0 = 2700;  // Density

enum MESH_RESOLUTION { RES_0, RES_2, RES_4, RES_8, RES_16 };

enum MATERIAL_MODEL { MAT_SVK, MAT_MOONEY_RIVLIN };

int main(int argc, char** argv) {
  // Parse command line arguments
  int res_arg = 8;     // Default resolution
  int steps_arg = 50;  // Default steps (increased from 2)
  
  if (argc > 1) res_arg = std::stoi(argv[1]);
  if (argc > 2) steps_arg = std::stoi(argv[2]);

  MESH_RESOLUTION resolution;
  switch(res_arg) {
    case 0: resolution = RES_0; break;
    case 2: resolution = RES_2; break;
    case 4: resolution = RES_4; break;
    case 8: resolution = RES_8; break;
    case 16: resolution = RES_16; break;
    default: 
      std::cerr << "Invalid resolution. Using RES_8." << std::endl;
      resolution = RES_8;
      res_arg = 8;
  }

  std::cout << "Running with Resolution: " << res_arg << ", Steps: " << steps_arg << std::endl;

  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;
  int plot_target_node;
  int n_nodes, n_elems;

  // MESH_RESOLUTION resolution = RES_8; // Removed hardcoded
  
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
  }

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  // Extract coordinate vectors from nodes matrix
  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);  // X coordinates
    h_y12(i) = nodes(i, 1);  // Y coordinates
    h_z12(i) = nodes(i, 2);  // Z coordinates
  }

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

  // Set fixed nodes
  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  // set external force
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
  } else {
    const double mu    = E / (2.0 * (1.0 + nu));
    const double K     = E / (3.0 * (1.0 - 2.0 * nu));
    const double kappa = 1.5 * K;
    const double mu10  = 0.30 * mu;
    const double mu01  = 0.20 * mu;
    gpu_t10_data.SetMooneyRivlin(mu10, mu01, kappa);
  }

  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();
  gpu_t10_data.CalcP();
  gpu_t10_data.CalcInternalForce();

  SyncedNewtonParams params = {1e-4, 1e-4, 1e-4, 1e14, 5, 10, 1e-3};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);

  // Vector to store x position of target node at each step
  std::vector<double> node_x_history;

  solver.AnalyzeHessianSparsity();
  solver.SetFixedSparsityPattern(true);  // Enable analysis reuse for fixed structure

  for (int i = 0; i < steps_arg; i++) {
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
  std::string filename = "node_x_history_res" + std::to_string(res_arg) + ".csv";
  std::ofstream csv_file(filename);
  csv_file << std::fixed << std::setprecision(17);
  csv_file << "step,x_position\n";
  for (size_t i = 0; i < node_x_history.size(); i++) {
    csv_file << i << "," << node_x_history[i] << "\n";
  }
  csv_file.close();
  std::cout << "Wrote node " << plot_target_node
            << " x-position history to " << filename << std::endl;

  // Set highest precision for cout
  std::cout << std::fixed << std::setprecision(17);

  Eigen::VectorXd x12, y12, z12;
  gpu_t10_data.RetrievePositionToCPU(x12, y12, z12);

//   std::cout << "x12:" << std::endl;
//   for (int i = 0; i < x12.size(); i++) {
//     std::cout << x12(i) << " ";
//   }
//   std::cout << std::endl;

//   std::cout << "y12:" << std::endl;
//   for (int i = 0; i < y12.size(); i++) {
//     std::cout << y12(i) << " ";
//   }
//   std::cout << std::endl;

//   std::cout << "z12:" << std::endl;
//   for (int i = 0; i < z12.size(); i++) {
//     std::cout << z12(i) << " ";
//   }
//   std::cout << std::endl;

  gpu_t10_data.Destroy();

  return 0;
}

