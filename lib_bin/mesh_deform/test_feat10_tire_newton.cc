/**
 * FEAT10 Tire Newton Mesh Deformation Demo
 *
 * Mirrors the structure of the FEAT10 teapot Newton demo, but loads the FEAT10
 * tire mesh and applies boundary conditions based on mesh height:
 * - Fix nodes in the bottom 10% of the mesh (by +z height).
 * - Apply a constant upward (+z) force distributed to nodes in the top 10%.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../lib_src/solvers/FEAT10ConstraintManager.h"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "lib_utils/cli_utils.h"

// Material properties (using SolidMaterialProperties)
const SolidMaterialProperties mat_tire =
    SolidMaterialProperties::SVK(1.0e7,   // E: Young's modulus (Pa)
                                 0.35,    // nu: Poisson's ratio
                                 1000.0,  // rho0: Density (kg/m³)
                                 5.0e3,   // eta_damp (Pa·s)
                                 5.0e3    // lambda_damp (Pa·s)
    );

int main(int argc, char** argv) {
  ANCFCPUUtils::Cli cli(argv[0]);
  cli.SetDescription(
      "FEAT10 tire Newton mesh deformation demo with selectable mesh "
      "resolution.");
  cli.AddInt("res", 0,
             "tire mesh resolution in data/meshes/T10/tire_scaling "
             "(allowed: 0, 2, 4, 8, 16)");

  std::string cli_err;
  if (!cli.Parse(argc, argv, &cli_err) || cli.HelpRequested()) {
    if (!cli_err.empty()) {
      std::cerr << cli_err << "\n\n";
    }
    cli.PrintUsage(std::cout);
    return cli.HelpRequested() ? 0 : 1;
  }

  const int res                            = cli.GetInt("res");
  constexpr std::array<int, 5> kAllowedRes = {0, 2, 4, 8, 16};
  bool valid_res                           = false;
  for (const int allowed_res : kAllowedRes) {
    if (res == allowed_res) {
      valid_res = true;
      break;
    }
  }
  if (!valid_res) {
    std::cerr << "Invalid --res=" << res << " (allowed: 0, 2, 4, 8, 16)"
              << std::endl;
    return 1;
  }

  const std::string mesh_prefix =
      "data/meshes/T10/tire_scaling/tire_res" + std::to_string(res) + ".1";

  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;

  const int n_nodes =
      ANCFCPUUtils::FEAT10_read_nodes(mesh_prefix + ".node", nodes);
  const int n_elems =
      ANCFCPUUtils::FEAT10_read_elements(mesh_prefix + ".ele", elements);

  std::cout << "mesh: " << mesh_prefix << std::endl;
  std::cout << "mesh read nodes: " << n_nodes << std::endl;
  std::cout << "mesh read elements: " << n_elems << std::endl;

  if (n_nodes <= 0 || n_elems <= 0) {
    std::cerr << "Failed to read tire mesh for --res=" << res << "."
              << std::endl;
    return 1;
  }

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  // Extract coordinate vectors from nodes matrix
  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);
    h_y12(i) = nodes(i, 1);
    h_z12(i) = nodes(i, 2);
  }

  const double z_min = h_z12.minCoeff();
  const double z_max = h_z12.maxCoeff();
  const double z_rng = z_max - z_min;
  const double z_fix = z_min + 0.1 * z_rng;
  const double z_top = z_min + 0.9 * z_rng;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "z_min=" << z_min << " z_max=" << z_max << " z_fix<=" << z_fix
            << " z_top>=" << z_top << std::endl;

  // Fixed nodes: bottom 10% by height
  std::vector<int> fixed_node_indices;
  fixed_node_indices.reserve(static_cast<size_t>(n_nodes));
  for (int i = 0; i < h_z12.size(); ++i) {
    if (h_z12(i) <= z_fix) {
      fixed_node_indices.push_back(i);
    }
  }

  Eigen::VectorXi h_fixed_nodes(static_cast<int>(fixed_node_indices.size()));
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(static_cast<int>(i)) = fixed_node_indices[i];
  }
  std::cout << "Fixed nodes: " << h_fixed_nodes.size() << std::endl;
  FEAT10ConstraintManager constraint_manager(&gpu_t10_data);
  constraint_manager.AddNodesToWorldCD(h_fixed_nodes);
  constraint_manager.Finalize();

  // External force: total upward force distributed over top 10% by height
  Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
  h_f_ext.setZero();

  std::vector<int> force_node_indices;
  force_node_indices.reserve(static_cast<size_t>(n_nodes));
  for (int i = 0; i < h_z12.size(); ++i) {
    if (h_z12(i) >= z_top) {
      force_node_indices.push_back(i);
    }
  }
  std::cout << "Force nodes: " << force_node_indices.size() << std::endl;

  constexpr double total_force_z = 3000.0;  // N
  if (!force_node_indices.empty()) {
    const double force_per_node = total_force_z / force_node_indices.size();
    for (int idx : force_node_indices) {
      h_f_ext(3 * idx + 2) = -force_per_node;
    }
  }
  gpu_t10_data.SetExternalForce(h_f_ext);
  Eigen::VectorXd h_f_ext_released = h_f_ext;
  h_f_ext_released.setZero();

  // Setup + material
  const Eigen::VectorXd& tet5pt_x_host       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y_host       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z_host       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights_host = Quadrature::tet5pt_weights;

  gpu_t10_data.Setup(tet5pt_x_host, tet5pt_y_host, tet5pt_z_host,
                     tet5pt_weights_host, h_x12, h_y12, h_z12, elements);
  gpu_t10_data.ApplyMaterial(mat_tire);

  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();
  gpu_t10_data.CalcP();
  gpu_t10_data.CalcInternalForce();

  // Newton solve
  SyncedNewtonParams params = {1e-4, 1e-4, 1e-4, 1e14, 5, 10, 5e-4, false};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();
  solver.SetFixedSparsityPattern(true);

  int output_interval = 20;
  int output_frame    = 0;
  for (int step = 0; step < 1200; ++step) {
    if (step == 400) {
      std::cout << "Releasing external force at step " << step << std::endl;
      gpu_t10_data.SetExternalForce(h_f_ext_released);
    }
    solver.Solve();
    if (step % output_interval == 0) {
      gpu_t10_data.ComputeVonMises();
      gpu_t10_data.WriteOutputVTU("output/tire_newton_res" +
                                  std::to_string(res) + "_step_" +
                                  std::to_string(output_frame) + ".vtu");
      ++output_frame;
    }
  }

  gpu_t10_data.Destroy();
  return 0;
}
