/**
 * FEAT10 Bunny Newton Test
 *
 * Author: Json Zhou
 * Email:  zzhou292@wisc.edu
 *
 * This simulation loads a FEAT10 bunny mesh, clamps nodes near the base,
 * applies strong downward loads on nodes near the ears, and advances the
 * configuration with the synchronized Newton solver. It is used to stress
 * test FEAT10 internal force assembly, constraint handling, Newton
 * convergence, and VTK output under large deformations.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/quadrature_utils.h"

const double E    = 3.0e8;  // Pa  (~0.3 GPa, between 0.7 GPa and 0.13 GPa)
const double nu   = 0.40;   // polymers tend to be higher than metals
const double rho0 = 920.0;  // kg/m^3, typical polyethylene density

enum MATERIAL_MODEL { MAT_SVK, MAT_MOONEY_RIVLIN };

namespace {

struct Options {
  double dt = 1e-4;                    // Time step (Newton typically needs larger dt)
  int steps = 8000;                    // Number of steps
  bool write_csv = false;              // CSV output flag
  std::string csv_path;                // CSV output path
  int vtk_interval = 10;               // VTK output interval (default: 10, 0 = disabled)
  std::string vtk_dir = "output";      // VTK output directory
  MATERIAL_MODEL material = MAT_MOONEY_RIVLIN;  // Default: Mooney-Rivlin
};

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

bool ParseInt(const std::string& s, int& out) {
  try {
    size_t idx = 0;
    int v = std::stoi(s, &idx);
    if (idx != s.size())
      return false;
    out = v;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseDouble(const std::string& s, double& out) {
  try {
    size_t idx = 0;
    double v = std::stod(s, &idx);
    if (idx != s.size())
      return false;
    out = v;
    return true;
  } catch (...) {
    return false;
  }
}

void PrintUsage(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [--dt=DT] [--steps=N] [--mat=MAT] [--csv[=PATH]]"
            << " [--vtk=N] [--vtk-dir=PATH] [--help]\n"
            << "  --dt=DT       Time step size (default: 1e-4)\n"
            << "  --steps=N     Number of time steps (default: 8000)\n"
            << "  --mat=MAT     svk | mr (default: mr)\n"
            << "  --csv[=PATH]  Write CSV output (optional path)\n"
            << "  --vtk=N       VTK output interval, 0 to disable (default: 10)\n"
            << "  --vtk-dir=P   VTK output directory (default: output)\n"
            << "  --help        Display this help message\n";
}

bool ParseMaterial(const std::string& s, MATERIAL_MODEL& out) {
  if (s == "svk") {
    out = MAT_SVK;
    return true;
  }
  if (s == "mr") {
    out = MAT_MOONEY_RIVLIN;
    return true;
  }
  return false;
}

bool ParseArgs(int argc, char** argv, Options& opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    }
    if (StartsWith(arg, "--dt=")) {
      const std::string v = arg.substr(std::string("--dt=").size());
      if (!ParseDouble(v, opt.dt) || !(opt.dt > 0.0)) {
        std::cerr << "Invalid --dt: " << v << "\n";
        return false;
      }
      continue;
    }
    if (StartsWith(arg, "--steps=")) {
      const std::string v = arg.substr(std::string("--steps=").size());
      if (!ParseInt(v, opt.steps) || opt.steps <= 0) {
        std::cerr << "Invalid --steps: " << v << "\n";
        return false;
      }
      continue;
    }
    if (StartsWith(arg, "--mat=")) {
      const std::string v = arg.substr(std::string("--mat=").size());
      if (!ParseMaterial(v, opt.material)) {
        std::cerr << "Unknown material: " << v << "\n";
        return false;
      }
      continue;
    }
    if (arg == "--csv") {
      opt.write_csv = true;
      continue;
    }
    if (StartsWith(arg, "--csv=")) {
      opt.write_csv = true;
      opt.csv_path = arg.substr(std::string("--csv=").size());
      continue;
    }
    if (StartsWith(arg, "--vtk=")) {
      const std::string v = arg.substr(std::string("--vtk=").size());
      if (!ParseInt(v, opt.vtk_interval) || opt.vtk_interval < 0) {
        std::cerr << "Invalid --vtk: " << v << "\n";
        return false;
      }
      continue;
    }
    if (StartsWith(arg, "--vtk-dir=")) {
      opt.vtk_dir = arg.substr(std::string("--vtk-dir=").size());
      continue;
    }
    std::cerr << "Unknown argument: " << arg << "\n";
    return false;
  }
  return true;
}

std::string JoinPath(const std::string& a, const std::string& b) {
  if (a.empty())
    return b;
  if (a.back() == '/')
    return a + b;
  return a + "/" + b;
}

std::string DefaultOutputDir() {
  if (const char* d = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR")) {
    return d;
  }
  return ".";
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!ParseArgs(argc, argv, opt)) {
    return 1;
  }

  // Workspace directory for mesh paths (Bazel runfiles support)
  std::string workspace_dir = ".";
  if (const char* d = std::getenv("BUILD_WORKSPACE_DIRECTORY")) {
    workspace_dir = d;
  }
  auto mesh_path = [&](const std::string& rel) {
    return JoinPath(workspace_dir, rel);
  };

  // Read mesh data
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;

  const std::string node_file = mesh_path("data/meshes/T10/bunny_ascii_26.1.node");
  const std::string elem_file = mesh_path("data/meshes/T10/bunny_ascii_26.1.ele");

  int n_nodes = ANCFCPUUtils::FEAT10_read_nodes(node_file.c_str(), nodes);
  int n_elems = ANCFCPUUtils::FEAT10_read_elements(elem_file.c_str(), elements);

  std::cout << "mesh read nodes: " << n_nodes << std::endl;
  std::cout << "mesh read elements: " << n_elems << std::endl;

  // print nodes and elements matrix
  std::cout << "nodes matrix:" << std::endl;
  std::cout << nodes << std::endl;
  std::cout << "elements matrix:" << std::endl;
  std::cout << elements << std::endl;

  // Material model (configured via --mat, default: mr)
  MATERIAL_MODEL material = opt.material;

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

  // Find all nodes with z < -4
  std::vector<int> fixed_node_indices;
  for (int i = 0; i < h_z12.size(); ++i) {
    if (h_z12(i) < -4.0) {  // Fix nodes with z coordinate less than -4
      fixed_node_indices.push_back(i);
    }
  }

  // Convert to Eigen::VectorXi
  Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(i) = fixed_node_indices[i];
  }

  // print fixed nodes
  std::cout << "Fixed nodes (z < -4.0):" << std::endl;
  for (int i = 0; i < h_fixed_nodes.size(); ++i) {
    std::cout << h_fixed_nodes(i) << " ";
  }
  std::cout << std::endl;

  // Track node: highest initial z (bunny ear tip)
  int track_node = 0;
  double max_z = h_z12(0);
  for (int i = 1; i < n_nodes; ++i) {
    if (h_z12(i) > max_z) {
      max_z = h_z12(i);
      track_node = i;
    }
  }
  std::cout << "Tracking node " << track_node << " (initial z = " << max_z << ")" << std::endl;

  // Set fixed nodes
  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  // set external force: -1000N in z direction for all nodes above z=4
  Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
  h_f_ext.setZero();

  for (int i = 0; i < h_z12.size(); ++i) {
    if (h_z12(i) > 4.0) {
      h_f_ext(3 * i + 2) = -35000.0;  // z direction
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

  gpu_t10_data.CalcConstraintData();

  std::cout << "done CalcConstraintData" << std::endl;

  gpu_t10_data.ConvertToCSR_ConstraintJacT();

  std::cout << "done ConvertToCSR_ConstraintJacT" << std::endl;

  gpu_t10_data.BuildConstraintJacobianCSR();

  std::cout << "done BuildConstraintJacobianCSR" << std::endl;

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

  // Open CSV file if requested
  std::ofstream csv_file;
  std::string csv_out_path;
  if (opt.write_csv) {
    csv_out_path = opt.csv_path;
    if (csv_out_path.empty()) {
      csv_out_path = JoinPath(DefaultOutputDir(), "bunny_newton.csv");
    }
    csv_file.open(csv_out_path);
    csv_file << std::fixed << std::setprecision(17);
    csv_file << "step,x_position,y_position,z_position,solve_time_ms,iterations\n";
  }

  std::vector<double> step_times_ms;
  step_times_ms.reserve(opt.steps);

  SyncedNewtonParams params = {opt.dt, 1e-6, 1e-4, 1e14, 5, 10, 1e-3};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);

  solver.AnalyzeHessianSparsity();
  solver.SetFixedSparsityPattern(
      true);  // Enable analysis reuse for fixed structure

  int vtk_frame = 0;
  // Release force at 1s: force_release_step = 1.0 / dt
  const int force_release_step = static_cast<int>(1.0 / opt.dt);

  std::cout << "Starting simulation: " << opt.steps << " steps, dt=" << opt.dt << std::endl;
  if (opt.vtk_interval > 0) {
    std::cout << "VTK output interval: " << opt.vtk_interval << " (dir: " << opt.vtk_dir << ")" << std::endl;
  }
  std::cout << "Force will be removed at step " << force_release_step << " (t=" << (force_release_step * opt.dt) << " s)" << std::endl;

  for (int step = 0; step < opt.steps; step++) {
    // Reset external force to zero after force_release_step
    if (step == force_release_step) {
      Eigen::VectorXd h_zero(gpu_t10_data.get_n_coef() * 3);
      h_zero.setZero();
      gpu_t10_data.SetExternalForce(h_zero);
      std::cout << "External force reset to zero at step " << step << std::endl;
    }

    auto solve_start = std::chrono::high_resolution_clock::now();
    solver.Solve();
    auto solve_end = std::chrono::high_resolution_clock::now();
    double solve_time_ms = std::chrono::duration<double, std::milli>(solve_end - solve_start).count();
    step_times_ms.push_back(solve_time_ms);

    // VTK output
    if (opt.vtk_interval > 0 && step % opt.vtk_interval == 0) {
      gpu_t10_data.WriteOutputVTK(JoinPath(opt.vtk_dir, "bunny_newton_step_" +
                                  std::to_string(vtk_frame) + ".vtk"));
      vtk_frame++;
    }

    // CSV output and periodic console updates
    if (opt.write_csv || step % 500 == 0 || step == opt.steps - 1) {
      Eigen::VectorXd x12, y12, z12;
      gpu_t10_data.RetrievePositionToCPU(x12, y12, z12);

      if (step % 500 == 0 || step == opt.steps - 1) {
        std::cout << "Step " << step << "/" << opt.steps << ": node " << track_node
                  << " pos = (" << std::setprecision(6) << x12(track_node) << ", "
                  << y12(track_node) << ", " << z12(track_node) << ")" << std::endl;
      }

      if (opt.write_csv) {
        // Get iteration count from solver (assuming it's available, otherwise use 0)
        int iterations = 0;  // Newton solver doesn't expose this easily, so use 0 for now
        csv_file << step << "," << x12(track_node) << "," << y12(track_node) << ","
                 << z12(track_node) << "," << solve_time_ms << "," << iterations << "\n";
      }
    }
  }

  // Timing summary
  if (!step_times_ms.empty()) {
    double total_time_ms = 0.0;
    for (double t : step_times_ms) {
      total_time_ms += t;
    }
    const double avg_time_ms = total_time_ms / static_cast<double>(opt.steps);
    std::cout << "\nTiming summary:" << std::endl;
    std::cout << "  Total simulation time: " << total_time_ms << " ms" << std::endl;
    std::cout << "  Average step time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (1000.0 / avg_time_ms) << " steps/sec" << std::endl;
  }

  if (opt.write_csv) {
    csv_file.close();
    std::cout << "Wrote CSV: " << csv_out_path << std::endl;
  }

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
