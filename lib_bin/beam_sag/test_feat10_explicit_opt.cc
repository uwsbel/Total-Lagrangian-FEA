/**
 * FEAT10Opt Explicit Test
 *
 * This driver tests the FEAT10Opt element with SyncedExplicitOpt solver.
 * It loads the T10 beam mesh, clamps nodes at x == 0, and advances the
 * system with the SyncedExplicitOptSolver using symplectic Euler.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../lib_src/elements/FEAT10DataOpt.cuh"
#include "../../lib_src/solvers/SyncedExplicitOpt.cuh"
#include "../../lib_utils/cpu_utils.h"

namespace {

// Mooney-Rivlin parameters (match non-opt MR test)
constexpr double kMR_mu10  = 80000.0;   // Pa
constexpr double kMR_mu01  = 20000.0;   // Pa
constexpr double kMR_kappa = 1e6;       // Pa
constexpr double kMR_rho   = 1100.0;    // kg/m^3

struct Options {
  double dt      = 1e-5;
  int steps      = 5000;
  bool write_csv = false;
  std::string csv_path;
};

void PrintUsage(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [--dt=DT] [--steps=N] [--csv[=PATH]] [--help]\n";
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

bool ParseInt(const std::string& s, int& out) {
  try {
    size_t idx = 0;
    int v      = std::stoi(s, &idx);
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
    double v   = std::stod(s, &idx);
    if (idx != s.size())
      return false;
    out = v;
    return true;
  } catch (...) {
    return false;
  }
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
    if (arg == "--csv") {
      opt.write_csv = true;
      continue;
    }
    if (StartsWith(arg, "--csv=")) {
      opt.write_csv = true;
      opt.csv_path  = arg.substr(std::string("--csv=").size());
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

  int device_count          = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&device_count);
  if (dev_err != cudaSuccess || device_count <= 0) {
    std::cerr << "No CUDA device visible (cudaGetDeviceCount returned "
              << device_count << ")" << std::endl;
    return 1;
  }
  HANDLE_ERROR(cudaSetDevice(0));

  cudaDeviceProp props;
  HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));
  std::cout << "CUDA device 0: " << props.name << " (cc " << props.major << "."
            << props.minor << ")" << std::endl;

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
  const std::string node_file = mesh_path("data/meshes/T10/beam_3x2x1.1.node");
  const std::string elem_file = mesh_path("data/meshes/T10/beam_3x2x1.1.ele");

  int n_nodes = ANCFCPUUtils::FEAT10_read_nodes(node_file.c_str(), nodes);
  int n_elems = ANCFCPUUtils::FEAT10_read_elements(elem_file.c_str(), elements);

  std::cout << "Mesh loaded: " << n_nodes << " nodes, " << n_elems
            << " elements" << std::endl;

  // Target node for tracking (matches resolution test res=0)
  const int plot_target_node = 23;
  if (n_nodes <= plot_target_node) {
    std::cerr << "Mesh too small for node " << plot_target_node << " tracking."
              << std::endl;
    return 1;
  }

  // Convert nodes matrix to proper format for FEAT10Opt
  Eigen::MatrixXd positions(n_nodes, 3);
  for (int i = 0; i < n_nodes; i++) {
    positions(i, 0) = nodes(i, 0);
    positions(i, 1) = nodes(i, 1);
    positions(i, 2) = nodes(i, 2);
  }

  // Initialize FEAT10Opt element
  GPU_FEAT10Opt_Data gpu_feat10opt;
  gpu_feat10opt.Initialize(n_elems, n_nodes);
  gpu_feat10opt.Setup(positions, elements);

  // Set material parameters (Mooney-Rivlin) to match non-opt driver
  gpu_feat10opt.SetMooneyRivlin(kMR_mu10, kMR_mu01, kMR_kappa);
  gpu_feat10opt.SetDensity(kMR_rho);

  std::cout << "Material (MR matched): mu10=" << kMR_mu10
            << ", mu01=" << kMR_mu01 << ", kappa=" << kMR_kappa
            << ", rho=" << kMR_rho << std::endl;

  // Compute precomputation (inverse Jacobians)
  gpu_feat10opt.ComputePrecomputation();

  // Compute lumped mass
  gpu_feat10opt.ComputeLumpedMassHRZ();

  // Print some mass info
  Eigen::VectorXf inv_mass;
  gpu_feat10opt.RetrieveInvLumpedMassToCPU(inv_mass);
  float total_mass = 0.0f;
  for (int i = 0; i < n_nodes; i++) {
    if (inv_mass(i) > 0) {
      total_mass += 1.0f / inv_mass(i);
    }
  }
  std::cout << "Total lumped mass: " << total_mass << " kg" << std::endl;

  // Fixed nodes: x == 0
  std::vector<int> fixed_nodes;
  fixed_nodes.reserve(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    if (std::abs(positions(i, 0)) < 1e-8) {
      fixed_nodes.push_back(i);
    }
  }
  std::cout << "Fixed nodes (x=0): " << fixed_nodes.size() << std::endl;

  // External force: distribute 10 kN in +Z across nodes with x == 3
  Eigen::VectorXf f_ext(n_nodes * 3);
  f_ext.setZero();
  std::vector<int> force_nodes;
  force_nodes.reserve(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    if (std::abs(positions(i, 0) - 3.0) < 1e-8) {
      force_nodes.push_back(i);
    }
  }
  if (!force_nodes.empty()) {
    const float force_per_node = 10000.0f / static_cast<float>(force_nodes.size());
    for (int idx : force_nodes) {
      f_ext(3 * idx + 2) = force_per_node;  // +Z direction
    }
  }
  std::cout << "Load nodes (x=3): " << force_nodes.size()
            << " total load = 10000 N in +Z" << std::endl;

  // Create SyncedExplicitOpt solver
  SyncedExplicitOptSolver solver(&gpu_feat10opt);
  solver.SetTimeStep(opt.dt);
  solver.SetFixedNodes(fixed_nodes);
  if (!force_nodes.empty()) {
    solver.SetExternalForce(f_ext);
  }

  // Storage for tracking
  std::vector<double> target_node_x;
  std::vector<double> target_node_y;
  std::vector<double> target_node_z;
  std::vector<float> step_times_ms;
  target_node_x.reserve(opt.steps);
  target_node_y.reserve(opt.steps);
  target_node_z.reserve(opt.steps);
  step_times_ms.reserve(opt.steps);

  std::cout << "Running SyncedExplicitOpt solver: dt=" << opt.dt
            << ", steps=" << opt.steps << std::endl;

  // Debug: at step 0 (initial config), compute F and P for element 0 to check F == I
  {
    gpu_feat10opt.ClearInternalForce();
    gpu_feat10opt.ComputeInternalForce(true, true);
    std::vector<std::vector<Eigen::Matrix3f>> F_per_elem;
    gpu_feat10opt.RetrieveDeformationGradientToCPU(F_per_elem);
    if (!F_per_elem.empty() && !F_per_elem[0].empty()) {
      std::cout << "Step 0 (initial) element 0 QP 0 F =\n"
                << F_per_elem[0][0] << std::endl;
    }
  }

  for (int step = 0; step < opt.steps; ++step) {
    // Remove the applied load halfway through the simulation to mirror
    // the non-optimized explicit test driver.
    if (step == opt.steps / 2) {
      f_ext.setZero();
      solver.SetExternalForce(f_ext);
    }

    solver.Solve();

    float kernel_time = solver.GetLastStepTimeMs();
    step_times_ms.push_back(kernel_time);

    // Get target node position from GPU
    Eigen::VectorXd pos_x, pos_y, pos_z;
    gpu_feat10opt.RetrievePositionToCPU(pos_x, pos_y, pos_z);

    target_node_x.push_back(pos_x(plot_target_node));
    target_node_y.push_back(pos_y(plot_target_node));
    target_node_z.push_back(pos_z(plot_target_node));

    if (step % 500 == 0 || step == opt.steps - 1) {
      std::cout << "Step " << step << "/" << opt.steps << ": node "
                << plot_target_node << " pos = (" << std::setprecision(6)
                << pos_x(plot_target_node) << ", " << pos_y(plot_target_node)
                << ", " << pos_z(plot_target_node)
                << "), kernel time = " << kernel_time << " ms" << std::endl;
    }
  }

  // Print timing statistics
  float total_time = 0.0f;
  for (float t : step_times_ms) {
    total_time += t;
  }
  float avg_time = total_time / opt.steps;
  std::cout << "\nTiming summary:" << std::endl;
  std::cout << "  Total simulation time: " << total_time << " ms" << std::endl;
  std::cout << "  Average step time: " << avg_time << " ms" << std::endl;
  std::cout << "  Throughput: " << (1000.0f / avg_time) << " steps/sec"
            << std::endl;

  // Write CSV if requested
  if (opt.write_csv) {
    std::string out_path = opt.csv_path;
    if (out_path.empty()) {
      out_path = JoinPath(DefaultOutputDir(), "feat10_opt.csv");
    }
    std::ofstream csv_file(out_path);
    csv_file << std::fixed << std::setprecision(17);
    csv_file << "step,x_position,y_position,z_position,kernel_time_ms\n";
    for (int i = 0; i < static_cast<int>(target_node_x.size()); ++i) {
      csv_file << i << "," << target_node_x[i] << "," << target_node_y[i] << ","
               << target_node_z[i] << "," << step_times_ms[i] << "\n";
    }
    std::cout << "Wrote CSV: " << out_path << std::endl;
  }

  // Clean up
  gpu_feat10opt.Destroy();

  std::cout << "\nTest completed successfully!" << std::endl;
  return 0;
}
