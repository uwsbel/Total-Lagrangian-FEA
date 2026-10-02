/**
 * FEAT10 Beam, Newton Reference for the Explicit Demo
 *
 * Runs the implicit Newton solver (cuDSS) on the same problem as
 * test_feat10_explicit, so the explicit solvers have an implicit reference:
 * 3x2x1 m beam, x = 0 face fixed, 10 kN in +z spread over the x = 3 face for
 * the first half of the run and then released, Mooney-Rivlin material matching
 * the FEAT10Opt compiled constants (E = 3e8, nu = 0.4, rho = 920), no damping.
 *
 * Examples:
 *   ./bazel-bin/lib_bin/beam_sag/test_feat10_explicit_newton_ref --res=8
 *   ./bazel-bin/lib_bin/beam_sag/test_feat10_explicit_newton_ref --res=8 \
 *       --steps=1000 --dt=1e-3 --csv=newton_res8.csv
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEAT10ConstraintManager.h"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "lib_utils/cli_utils.h"

namespace {

// Same material as test_feat10_explicit (and the FEAT10Opt kernel constants).
constexpr double kMR_mu10  = 32142857.142857143;  // Pa (0.30 * mu)
constexpr double kMR_mu01  = 21428571.428571429;  // Pa (0.20 * mu)
constexpr double kMR_kappa = 7.5e8;               // Pa (1.5 * bulk modulus)
constexpr double kMR_rho   = 920.0;               // kg/m^3

const SolidMaterialProperties mat_beam_mr =
    SolidMaterialProperties::MooneyRivlin(kMR_mu10, kMR_mu01, kMR_kappa,
                                          kMR_rho, 0.0, 0.0);

constexpr double kTotalForce = 10000.0;  // N, +z on the x = 3 face

// Tracked node per resolution, same as test_feat10_explicit.
const std::map<int, int> kTargetNode = {{0, 23},     {2, 89},    {4, 353},
                                        {8, 1408},   {16, 5630}, {32, 22529}};

std::string JoinPath(const std::string& a, const std::string& b) {
  if (a.empty())
    return b;
  if (a.back() == '/')
    return a + b;
  return a + "/" + b;
}

}  // namespace

int main(int argc, char** argv) {
  ANCFCPUUtils::Cli cli(argv[0]);
  cli.SetDescription(
      "Newton reference for the FEAT10 explicit beam demo (same problem).");
  cli.AddInt("res", 0, "beam mesh resolution: 0, 2, 4, 8, 16 or 32");
  cli.AddInt("steps", 1000, "number of time steps (load released at steps/2)");
  cli.AddDouble("dt", 1e-3, "time step (s)");
  cli.AddOptionalString("csv", "",
                        "write the tracked node's position every step");

  std::string cli_err;
  if (!cli.Parse(argc, argv, &cli_err) || cli.HelpRequested()) {
    if (!cli_err.empty()) {
      std::cerr << cli_err << "\n\n";
    }
    cli.PrintUsage(std::cout);
    return cli.HelpRequested() ? 0 : 1;
  }

  const int res      = cli.GetInt("res");
  const int steps    = cli.GetInt("steps");
  const double dt    = cli.GetDouble("dt");
  const auto target  = kTargetNode.find(res);
  if (target == kTargetNode.end() || steps <= 0 || !(dt > 0.0)) {
    std::cerr << "Invalid arguments (res must be 0, 2, 4, 8, 16 or 32; steps "
                 "and dt positive)"
              << std::endl;
    return 1;
  }
  const int plot_target_node = target->second;

  std::string csv_path;
  if (cli.IsSet("csv")) {
    csv_path = cli.GetString("csv");
    if (csv_path.empty()) {
      csv_path = "node_history_feat10_explicit_newton_res" +
                 std::to_string(res) + ".csv";
    }
  }

  std::string workspace_dir = ".";
  if (const char* d = std::getenv("BUILD_WORKSPACE_DIRECTORY")) {
    workspace_dir = d;
  }
  const std::string mesh_prefix = JoinPath(
      workspace_dir, "data/meshes/T10/resolution/beam_3x2x1_res" +
                         std::to_string(res) + ".1");

  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;
  const int n_nodes =
      ANCFCPUUtils::FEAT10_read_nodes(mesh_prefix + ".node", nodes);
  const int n_elems =
      ANCFCPUUtils::FEAT10_read_elements(mesh_prefix + ".ele", elements);
  if (n_nodes <= plot_target_node) {
    std::cerr << "Mesh too small for node " << plot_target_node << std::endl;
    return 1;
  }

  std::cout << "mesh read nodes: " << n_nodes << std::endl;
  std::cout << "mesh read elements: " << n_elems << std::endl;
  std::cout << "res=" << res << " steps=" << steps << " dt=" << dt
            << std::endl;

  GPU_FEAT10_Data data(n_elems, n_nodes);
  data.Initialize();

  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);
    h_y12(i) = nodes(i, 1);
    h_z12(i) = nodes(i, 2);
  }

  // Fixed nodes: x == 0
  std::vector<int> fixed_node_indices;
  for (int i = 0; i < n_nodes; ++i) {
    if (std::abs(h_x12(i)) < 1e-8) {
      fixed_node_indices.push_back(i);
    }
  }
  Eigen::VectorXi h_fixed_nodes(static_cast<int>(fixed_node_indices.size()));
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(static_cast<int>(i)) = fixed_node_indices[i];
  }
  FEAT10ConstraintManager constraint_manager(&data);
  constraint_manager.AddNodesToWorldCD(h_fixed_nodes);
  constraint_manager.Finalize();

  // External force: 10 kN in +z spread over the x == 3 face (first half only)
  Eigen::VectorXd h_f_ext(data.get_n_coef() * 3);
  h_f_ext.setZero();
  std::vector<int> force_node_indices;
  for (int i = 0; i < n_nodes; ++i) {
    if (std::abs(h_x12(i) - 3.0) < 1e-8) {
      force_node_indices.push_back(i);
    }
  }
  for (int node_idx : force_node_indices) {
    h_f_ext(3 * node_idx + 2) = kTotalForce / force_node_indices.size();
  }
  data.SetExternalForce(h_f_ext);

  data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y, Quadrature::tet5pt_z,
             Quadrature::tet5pt_weights, h_x12, h_y12, h_z12, elements);
  data.ApplyMaterial(mat_beam_mr);
  std::cout << "Beam material: Mooney-Rivlin mu10=" << mat_beam_mr.mu10
            << " mu01=" << mat_beam_mr.mu01 << " kappa=" << mat_beam_mr.kappa
            << " rho0=" << mat_beam_mr.rho0 << std::endl;

  data.CalcDnDuPre();
  data.CalcMassMatrix();
  data.CalcConstraintData();
  data.ConvertToCSR_ConstraintJacT();
  data.BuildConstraintJacobianCSR();
  data.CalcP();
  data.CalcInternalForce();

  std::ofstream csv_file;
  if (!csv_path.empty()) {
    csv_file.open(csv_path);
    csv_file << std::fixed << std::setprecision(17);
    csv_file << "step,x_position,y_position,z_position\n";
    std::cout << "Writing CSV: " << csv_path << std::endl;
  }

  SyncedNewtonParams params = {1e-4, 1e-4, 1e-4, 1e14, 5, 10, dt, false};
  if (res == 32) {
    params = {1e-3, 1e-3, 1e-3, 1e14, 5, 10, dt, false};
  }
  SyncedNewtonSolver solver(&data, data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();
  solver.SetFixedSparsityPattern(true);

  Eigen::VectorXd x12, y12, z12;
  for (int step = 0; step < steps; ++step) {
    if (step == steps / 2) {
      h_f_ext.setZero();
      data.SetExternalForce(h_f_ext);
    }
    solver.Solve();

    data.RetrievePositionToCPU(x12, y12, z12);
    const double x = x12(plot_target_node);
    const double y = y12(plot_target_node);
    const double z = z12(plot_target_node);
    std::cout << "Step " << step << ": node " << plot_target_node << " = ("
              << std::setprecision(17) << x << ", " << y << ", " << z << ")"
              << std::endl;
    if (csv_file.is_open()) {
      csv_file << step << "," << x << "," << y << "," << z << "\n";
    }
  }

  data.Destroy();
  return 0;
}
