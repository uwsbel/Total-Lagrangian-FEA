/**
 * FEAT10 Beam Resolution Study (Unified)
 *
 * Combines Newton / VBD / (non-coop) AdamW into a single binary with a solver
 * selection flag.
 *
 * Note: pass `--solver=adamw` to use `SyncedAdamWNocoopSolver`.
 *
 * Examples:
 *   ./bazel-bin/test_feat10_resolution --solver=adamw --res=8 --steps=50 --dt=1e-3 --csv
 *   ./bazel-bin/test_feat10_resolution --solver=vbd   --res=8 --steps=50 --dt=1e-3 --omega=1.8 --csv=out.csv
 *   ./bazel-bin/test_feat10_resolution --solver=newton --res=8 --steps=50 --dt=1e-3 --csv
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/SyncedAdamWNocoop.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_src/solvers/SyncedVBD.cuh"
#include "../../lib_utils/cpu_utils.h"

namespace {

constexpr double kE = 7e8;
constexpr double kNu = 0.33;
constexpr double kRho0 = 2700;

enum class SolverKind { kNewton, kVbd, kAdamW };

struct Options {
  SolverKind solver = SolverKind::kAdamW;
  int steps = 50;
  double dt = 1e-3;
  double omega = std::numeric_limits<double>::quiet_NaN();  // VBD only
  int res = 8;                                              // 0/2/4/8/16/32
  bool write_csv = false;
  std::string csv_path;
  int material = MATERIAL_MODEL_SVK;  // Material model for FEAT10
};

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [--solver=SOLVER] [--res=R] [--steps=N] [--dt=DT]\n"
      << "                 [--omega=W] [--material=MAT] [--csv[=PATH]] "
          "[--help]\n\n"
      << "  --solver=SOLVER   newton | vbd | adamw (default: adamw)\n"
      << "                   (adamw uses SyncedAdamWNocoopSolver)\n"
      << "  --res=R            0 | 2 | 4 | 8 | 16 | 32 (default: 8)\n"
      << "  --steps=N          number of Solve() calls (default: 50)\n"
      << "  --dt=DT            time step (default: 1e-3)\n"
      << "  --omega=W          VBD relaxation factor (default: 1.8)\n"
      << "  --material=MAT     svk | mr (default: svk)\n"
      << "                   Material model for FEAT10 elements\n"
      << "  --csv[=PATH]       write target-node x history CSV\n";
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

bool ParseInt(const std::string& s, int& out) {
  try {
    size_t idx = 0;
    int v = std::stoi(s, &idx);
    if (idx != s.size()) return false;
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
    if (idx != s.size()) return false;
    out = v;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseSolver(const std::string& s, SolverKind& out) {
  if (s == "newton") {
    out = SolverKind::kNewton;
    return true;
  }
  if (s == "vbd") {
    out = SolverKind::kVbd;
    return true;
  }
  if (s == "adamw") {
    out = SolverKind::kAdamW;
    return true;
  }
  return false;
}

bool ParseMaterial(const std::string& s, int& out) {
  if (s == "svk") {
    out = MATERIAL_MODEL_SVK;
    return true;
  }
  if (s == "mr" || s == "mooney-rivlin") {
    out = MATERIAL_MODEL_MOONEY_RIVLIN;
    return true;
  }
  return false;
}

std::string SolverName(SolverKind solver) {
  switch (solver) {
    case SolverKind::kNewton:
      return "newton";
    case SolverKind::kVbd:
      return "vbd";
    case SolverKind::kAdamW:
      return "adamw";
  }
  return "unknown";
}

bool ParseArgs(int argc, char** argv, Options& opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    }
    if (StartsWith(arg, "--solver=")) {
      const std::string v = arg.substr(std::string("--solver=").size());
      if (!ParseSolver(v, opt.solver)) {
        std::cerr << "Unknown solver: " << v << "\n";
        return false;
      }
      continue;
    }
    if (StartsWith(arg, "--res=")) {
      const std::string v = arg.substr(std::string("--res=").size());
      int r = 0;
      if (!ParseInt(v, r) ||
          !(r == 0 || r == 2 || r == 4 || r == 8 || r == 16 || r == 32)) {
        std::cerr << "Invalid --res: " << v << "\n";
        return false;
      }
      opt.res = r;
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
    if (StartsWith(arg, "--dt=")) {
      const std::string v = arg.substr(std::string("--dt=").size());
      if (!ParseDouble(v, opt.dt) || !(opt.dt > 0.0)) {
        std::cerr << "Invalid --dt: " << v << "\n";
        return false;
      }
      continue;
    }
    if (StartsWith(arg, "--omega=")) {
      const std::string v = arg.substr(std::string("--omega=").size());
      if (!ParseDouble(v, opt.omega) || !(opt.omega > 0.0)) {
        std::cerr << "Invalid --omega: " << v << "\n";
        return false;
      }
      continue;
    }
    if (StartsWith(arg, "--material=")) {
      const std::string v = arg.substr(std::string("--material=").size());
      if (!ParseMaterial(v, opt.material)) {
        std::cerr << "Invalid --material: " << v << " (use 'svk' or 'mr')\n";
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
    std::cerr << "Unknown argument: " << arg << "\n";
    return false;
  }
  return true;
}

std::string JoinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
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

  int device_count = 0;
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
  int plot_target_node = 0;
  int n_nodes = 0;
  int n_elems = 0;

  const std::string res_str = std::to_string(opt.res);
  const std::string node_file =
      mesh_path("data/meshes/T10/resolution/beam_3x2x1_res" + res_str + ".1.node");
  const std::string elem_file =
      mesh_path("data/meshes/T10/resolution/beam_3x2x1_res" + res_str + ".1.ele");

  n_nodes = ANCFCPUUtils::FEAT10_read_nodes(node_file.c_str(), nodes);
  n_elems = ANCFCPUUtils::FEAT10_read_elements(elem_file.c_str(), elements);

  // Keep historical target nodes from the legacy resolution drivers.
  if (opt.res == 0) {
    plot_target_node = 23;
  } else if (opt.res == 2) {
    plot_target_node = 89;
  } else if (opt.res == 4) {
    plot_target_node = 353;
  } else if (opt.res == 8) {
    plot_target_node = 1408;
  } else if (opt.res == 16) {
    plot_target_node = 5630;
  } else if (opt.res == 32) {
    plot_target_node = 22529;
  }

  std::cout << "mesh read nodes: " << n_nodes << std::endl;
  std::cout << "mesh read elements: " << n_elems << std::endl;
  std::cout << "solver=" << SolverName(opt.solver) << " res=" << opt.res
            << " steps=" << opt.steps << " dt=" << opt.dt << std::endl;

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
  fixed_node_indices.reserve(static_cast<size_t>(n_nodes));
  for (int i = 0; i < h_x12.size(); ++i) {
    if (std::abs(h_x12(i)) < 1e-8) {
      fixed_node_indices.push_back(i);
    }
  }

  Eigen::VectorXi h_fixed_nodes(static_cast<int>(fixed_node_indices.size()));
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(static_cast<int>(i)) = fixed_node_indices[i];
  }
  data.SetNodalFixed(h_fixed_nodes);

  // External force: distribute 5000N in +x at x == 3
  Eigen::VectorXd h_f_ext(data.get_n_coef() * 3);
  h_f_ext.setZero();
  std::vector<int> force_node_indices;
  for (int i = 0; i < h_x12.size(); ++i) {
    if (std::abs(h_x12(i) - 3.0) < 1e-8) {
      force_node_indices.push_back(i);
    }
  }
  if (!force_node_indices.empty()) {
    const double force_per_node = 5000.0 / force_node_indices.size();
    for (int node_idx : force_node_indices) {
      h_f_ext(3 * node_idx + 0) = force_per_node;
    }
  }
  data.SetExternalForce(h_f_ext);

  // Setup (reference quadrature from header)
  data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y, Quadrature::tet5pt_z,
             Quadrature::tet5pt_weights, h_x12, h_y12, h_z12, elements);

  data.SetDensity(kRho0);
  data.SetDamping(0.0, 0.0);
  
  // Set material model based on user selection
  if (opt.material == MATERIAL_MODEL_SVK) {
    data.SetSVK(kE, kNu);
    std::cout << "Material: SVK (E=" << kE << ", nu=" << kNu << ")" << std::endl;
  } else if (opt.material == MATERIAL_MODEL_MOONEY_RIVLIN) {
    // Convert SVK parameters (E, nu) to Mooney-Rivlin parameters
    const double mu    = kE / (2.0 * (1.0 + kNu));
    const double K     = kE / (3.0 * (1.0 - 2.0 * kNu));
    const double kappa = 1.5 * K;
    const double mu10  = 0.30 * mu;
    const double mu01  = 0.20 * mu;
    data.SetMooneyRivlin(mu10, mu01, kappa);
    std::cout << "Material: Mooney-Rivlin (mu10=" << mu10 << ", mu01=" << mu01 
              << ", kappa=" << kappa << ")" << std::endl;
  } else {
    std::cerr << "Unknown material model: " << opt.material << std::endl;
    return 1;
  }

  // Common precomputations
  data.CalcDnDuPre();
  data.CalcMassMatrix();
  data.CalcConstraintData();
  data.ConvertToCSR_ConstraintJacT();
  data.BuildConstraintJacobianCSR();

  if (opt.solver != SolverKind::kAdamW) {
    data.CalcP();
    data.CalcInternalForce();
  }

  std::ofstream csv_file;
  if (opt.write_csv) {
    std::string out_path = opt.csv_path;
    if (out_path.empty()) {
      const std::string filename = "node_x_history_feat10_res" + res_str + "_" +
                                   SolverName(opt.solver) + ".csv";
      out_path = JoinPath(DefaultOutputDir(), filename);
    }
    csv_file.open(out_path);
    csv_file << std::fixed << std::setprecision(17);
    csv_file << "step,x_position\n";
    std::cout << "Writing CSV: " << out_path << std::endl;
  }

  auto record_step = [&](int step) {
    Eigen::VectorXd x12_current, y12_current, z12_current;
    data.RetrievePositionToCPU(x12_current, y12_current, z12_current);
    if (plot_target_node < x12_current.size()) {
      const double x = x12_current(plot_target_node);
      std::cout << "Step " << step << ": node " << plot_target_node
                << " x = " << std::setprecision(17) << x << std::endl;
      if (opt.write_csv) {
        csv_file << step << "," << x << "\n";
        csv_file.flush();
      }
    }
  };

  switch (opt.solver) {
    case SolverKind::kNewton: {
      SyncedNewtonParams params = {1e-4, 1e-4, 1e-4, 1e14, 5, 10, opt.dt};
      SyncedNewtonSolver solver(&data, data.get_n_constraint());
      solver.Setup();
      solver.SetParameters(&params);
      solver.AnalyzeHessianSparsity();
      solver.SetFixedSparsityPattern(true);
      for (int step = 0; step < opt.steps; ++step) {
        solver.Solve();
        record_step(step);
      }
      break;
    }
    case SolverKind::kVbd: {
      const double omega = std::isnan(opt.omega) ? 1.8 : opt.omega;
      SyncedVBDParams params = {1e-4,  1e-4,  1e-4,  1e14, 5,   500,  opt.dt,
                                omega, 1e-12, 25,     1};
      SyncedVBDSolver solver(&data, data.get_n_constraint());
      solver.Setup();
      solver.SetParameters(&params);
      solver.InitializeColoring();
      solver.InitializeMassDiagBlocks();
      solver.InitializeFixedMap();
      for (int step = 0; step < opt.steps; ++step) {
        solver.Solve();
        record_step(step);
      }
      break;
    }
    case SolverKind::kAdamW: {
      SyncedAdamWNocoopParams params;
      if (opt.res == 0) {
        params = {2e-4, 0.9, 0.999, 1e-8, 1e-4, 0.995, 1e-1, 1e-6,
                  1e14, 5,   800,  opt.dt, 20,   1e-4, opt.material};
      } else if (opt.res == 2) {
        params = {2e-4, 0.9, 0.999, 1e-8, 1e-4, 0.995, 1e-1, 1e-6,
                  1e14, 5,   800,  opt.dt, 20,   1e-4, opt.material};
      } else if (opt.res == 4) {
        params = {2e-4, 0.9, 0.999, 1e-8, 1e-4, 0.995, 1e-1, 1e-6,
                  1e14, 5,   800,  opt.dt, 20,   1e-4, opt.material};
      } else if (opt.res == 8) {
        params = {2.5e-4, 0.9, 0.999, 1e-8, 1e-4, 0.998, 1e-1, 1e-6,
                  1e14,   5,   800,  opt.dt, 20,   1e-4, opt.material};
      } else if (opt.res == 16) {
        params = {2.5e-4, 0.9, 0.999, 1e-8, 1e-4, 0.998, 1e-1, 1e-6,
                  1e14,   5,   800,  opt.dt, 20,   1e-4, opt.material};
      } else {
        std::cerr << "Unsupported resolution" << std::endl;
        return 1;
      }
      SyncedAdamWNocoopSolver solver(&data, data.get_n_constraint());
      solver.Setup();
      solver.SetParameters(&params);
      for (int step = 0; step < opt.steps; ++step) {
        solver.Solve();
        record_step(step);
      }
      break;
    }
  }

  if (opt.write_csv) {
    csv_file.close();
  }

  data.Destroy();
  return 0;
}
