/**
 * ANCF3443 Shell Unified Solver Test
 *
 * This binary unifies the ANCF3443 shell driver across multiple solvers
 * (Newton / Nesterov / AdamW / VBD). Use `--solver=...` to select the solver.
 *
 * Example:
 *   ./bazel-bin/lib_bin/beam_sag/test_ancf3443 --solver=vbd --res=0 --dt=1e-3 --omega=1.8 --csv
 *
 * Notes:
 * - Steps are hardcoded to 200.
 * - Tip load is released after 100 steps.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/elements/ANCF3443Data.cuh"
#include "../../lib_src/solvers/SyncedAdamWNocoop.cuh"
#include "../../lib_src/solvers/SyncedNesterov.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_src/solvers/SyncedVBD.cuh"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/visualization_utils.h"

namespace {

constexpr double kE    = 7e8;
constexpr double kNu   = 0.33;
constexpr double kRho0 = 2700;

constexpr double kDefaultL       = 2.0;
constexpr double kDefaultW       = 1.0;
constexpr double kDefaultH       = 0.1;
constexpr int kDefaultRes        = 0;
constexpr double kShellXSize     = 4.0;
constexpr double kShellYSize     = 2.0;
constexpr int kVtuEvery          = 20;
constexpr const char* kVtuPrefix = "ancf3443";

enum class SolverKind { kNewton, kNesterov, kAdamW, kVbd };

struct Options {
  SolverKind solver  = SolverKind::kVbd;
  // Structured ANCF3443 shell (XY plane): nx × ny elements over a fixed
  // x_size × y_size domain (hardcoded). nx=ny is derived from `res`.
  int res            = kDefaultRes;
  int steps          = 200;
  double dt          = 5e-4;
  double tip_force_z = std::numeric_limits<double>::quiet_NaN();
  int force_release_step = 100;  // hardcoded (not a CLI option)
  double lrratio     = 0.5;
  double omega       = std::numeric_limits<double>::quiet_NaN();  // VBD only
  bool write_csv     = false;
  std::string csv_path;
  bool write_vtu = false;
};

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [--solver=SOLVER] [--res=R]\n"
      << "                 [--dt=DT]\n"
      << "                 [--tip_force_z=FZ] [--lrratio=R]\n"
      << "                 [--omega=W] [--csv[=PATH]] [--help]\n\n"
      << "  --solver=SOLVER   newton | nesterov | adamw | vbd (default: vbd)\n"
      << "  --res=R           0 | 2 | 4 | 8 | 16 | 32 (default: 0)\n"
       << "                   (0->10x10, 2->20x20, 4->50x50, 8->100x100,\n"
       << "                    16->200x200, 32->400x400)\n"
      << "                   (shell size is fixed: x_size=4, y_size=2)\n"
      << "                   (steps=200, force released after step 100)\n"
      << "  --dt=DT           time step passed to solver params (default: "
         "1e-3)\n"
      << "  --tip_force_z=FZ  total vertical force on free edge (default: "
         "-5000*0.1)\n"
      << "  --lrratio=R       load ratio on -y tip node (default: 0.5)\n"
      << "  --omega=W         VBD relaxation factor (default: 1.8)\n"
      << "  --csv[=PATH]      write tip displacement CSV (default path depends "
         "on solver)\n"
      << "  --vtu             write VTU hex meshes to output/ancf3443/ (every "
         "20 steps)\n";
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

bool ParseSolver(const std::string& s, SolverKind& out) {
  if (s == "newton") {
    out = SolverKind::kNewton;
    return true;
  }
  if (s == "nesterov") {
    out = SolverKind::kNesterov;
    return true;
  }
  if (s == "adamw") {
    out = SolverKind::kAdamW;
    return true;
  }
  if (s == "vbd") {
    out = SolverKind::kVbd;
    return true;
  }
  return false;
}

std::string SolverName(SolverKind solver) {
  switch (solver) {
    case SolverKind::kNewton:
      return "newton";
    case SolverKind::kNesterov:
      return "nesterov";
    case SolverKind::kAdamW:
      return "adamw";
    case SolverKind::kVbd:
      return "vbd";
  }
  return "unknown";
}

int ResolutionFromRes(int res) {
  switch (res) {
    case 0:
      return 10;
    case 2:
      return 20;
    case 4:
      return 50;
    case 8:
      return 100;
    case 16:
      return 150;
    case 32:
      return 200;
  }
  return -1;
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
      int r               = 0;
      if (!ParseInt(v, r) || ResolutionFromRes(r) <= 0) {
        std::cerr << "Invalid --res: " << v << " (expected 0|2|4|8|16|32)\n";
        return false;
      }
      opt.res = r;
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
    if (StartsWith(arg, "--tip_force_z=")) {
      const std::string v = arg.substr(std::string("--tip_force_z=").size());
      if (!ParseDouble(v, opt.tip_force_z)) {
        std::cerr << "Invalid --tip_force_z: " << v << "\n";
        return false;
      }
      continue;
    }
    if (StartsWith(arg, "--lrratio=")) {
      const std::string v = arg.substr(std::string("--lrratio=").size());
      if (!ParseDouble(v, opt.lrratio) || !(opt.lrratio >= 0.0) ||
          !(opt.lrratio <= 1.0)) {
        std::cerr << "Invalid --lrratio (expected [0,1]): " << v << "\n";
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
    if (arg == "--csv") {
      opt.write_csv = true;
      continue;
    }
    if (StartsWith(arg, "--csv=")) {
      opt.write_csv = true;
      opt.csv_path  = arg.substr(std::string("--csv=").size());
      continue;
    }
    if (arg == "--vtu") {
      opt.write_vtu = true;
      continue;
    }
    std::cerr << "Unknown argument: " << arg << "\n";
    return false;
  }
  return true;
}

void WriteTipCsv(const std::string& path,
                 const std::vector<double>& tip_z_history) {
  std::ofstream csv_file(path);
  csv_file << std::fixed << std::setprecision(17);
  csv_file << "step,tip_z\n";
  for (size_t i = 0; i < tip_z_history.size(); ++i) {
    csv_file << i << "," << tip_z_history[i] << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!ParseArgs(argc, argv, opt)) {
    return 1;
  }

  const int derived_res = ResolutionFromRes(opt.res);
  if (derived_res <= 0) {
    std::cerr << "Invalid res configuration.\n";
    return 1;
  }
  const int x_res = derived_res;
  const int y_res = derived_res;

  if (std::isnan(opt.tip_force_z)) {
    opt.tip_force_z = -5000.0 * kDefaultH;
  }

  const std::string vtu_out_dir = "output/ancf3443";
  if (opt.write_vtu) {
    std::filesystem::create_directories(vtu_out_dir);
  }

  const int n_elem  = x_res * y_res;
  const int n_nodes = (x_res + 1) * (y_res + 1);
  GPU_ANCF3443_Data data(n_nodes, n_elem);
  data.Initialize();

  const double L = kShellXSize / static_cast<double>(x_res);
  const double W = kShellYSize / static_cast<double>(y_res);
  const double H = kDefaultH;

  std::cout << "ANCF3443(shell): res=" << opt.res
            << " nx=" << x_res
            << " ny=" << y_res
            << " elements=" << n_elem
            << " nodes=" << n_nodes
            << " coef=" << data.get_n_coef()
            << " solver=" << SolverName(opt.solver)
            << " steps=" << opt.steps
            << " dt=" << opt.dt
            << " L=" << L << " W=" << W << " H=" << H
            << " x_size=" << kShellXSize
            << " y_size=" << kShellYSize
            << " tip_force_z=" << opt.tip_force_z
            << " force_release_step=" << opt.force_release_step
            << std::endl;

  Eigen::VectorXd h_x12(data.get_n_coef());
  Eigen::VectorXd h_y12(data.get_n_coef());
  Eigen::VectorXd h_z12(data.get_n_coef());
  Eigen::MatrixXi element_connectivity(n_elem, 4);
  ANCFCPUUtils::ANCF3443_generate_shell_coordinates(
      kShellXSize, kShellYSize, x_res, y_res,
      h_x12, h_y12, h_z12, element_connectivity);

  const int n_nodes_from_coords = static_cast<int>(h_x12.size()) / 4;
  if (n_nodes_from_coords != n_nodes) {
    std::cerr << "Shell coordinate generator produced n_nodes="
              << n_nodes_from_coords << " but expected " << n_nodes << "\n";
    return 1;
  }
  std::vector<int> fixed_dofs;
  fixed_dofs.reserve(static_cast<size_t>(n_nodes));
  const double x_edge_tol = 1e-10 * std::max(1.0, std::abs(kShellXSize));
  for (int node = 0; node < n_nodes; ++node) {
    const int base = node * 4;
    if (std::abs(h_x12(base + 0) - 0.0) <= x_edge_tol) {
      for (int d = 0; d < 4; ++d) {
        fixed_dofs.push_back(base + d);
      }
    }
  }
  Eigen::VectorXi h_fixed_nodes(static_cast<int>(fixed_dofs.size()));
  for (int i = 0; i < static_cast<int>(fixed_dofs.size()); ++i) {
    h_fixed_nodes(i) = fixed_dofs[static_cast<size_t>(i)];
  }
  data.SetNodalFixed(h_fixed_nodes);

  Eigen::VectorXd h_f_ext_on(data.get_n_coef() * 3);
  h_f_ext_on.setZero();
  std::vector<int> tip_nodes;
  tip_nodes.reserve(static_cast<size_t>(n_nodes));
  double tip_y_min = std::numeric_limits<double>::infinity();
  double tip_y_max = -std::numeric_limits<double>::infinity();
  for (int node = 0; node < n_nodes; ++node) {
    const int base = node * 4;
    if (std::abs(h_x12(base + 0) - kShellXSize) <= x_edge_tol) {
      tip_nodes.push_back(node);
      tip_y_min = std::min(tip_y_min, h_y12(base + 0));
      tip_y_max = std::max(tip_y_max, h_y12(base + 0));
    }
  }
  if (tip_nodes.empty()) {
    std::cerr << "No tip nodes found at x == x_size. Check mesh generation.\n";
    return 1;
  }

  // Distribute total vertical force across all tip-edge nodes. `lrratio`
  // provides a linear bias from -y (ratio) to +y (1-ratio). When lrratio=0.5,
  // this is uniform.
  const double denom_y = (tip_y_max > tip_y_min) ? (tip_y_max - tip_y_min) : 1.0;
  std::vector<double> weights(tip_nodes.size(), 0.0);
  double wsum = 0.0;
  for (size_t i = 0; i < tip_nodes.size(); ++i) {
    const int node = tip_nodes[i];
    const int base = node * 4;
    const double t = (h_y12(base + 0) - tip_y_min) / denom_y;  // in [0,1]
    const double w = (1.0 - t) * opt.lrratio + t * (1.0 - opt.lrratio);
    weights[i] = w;
    wsum += w;
  }
  if (wsum <= 0.0) {
    std::cerr << "Invalid tip-edge load weights (sum <= 0).\n";
    return 1;
  }
  for (size_t i = 0; i < tip_nodes.size(); ++i) {
    const int node = tip_nodes[i];
    const double fz = opt.tip_force_z * (weights[i] / wsum);
    const int coeff = node * 4;  // position coefficient
    h_f_ext_on(coeff * 3 + 2) += fz;
  }
  Eigen::VectorXd h_f_ext_off(data.get_n_coef() * 3);
  h_f_ext_off.setZero();
  data.SetExternalForce(h_f_ext_on);

  auto maybe_release_force = [&](int step) {
    if (opt.force_release_step >= 0 && step == opt.force_release_step) {
      data.SetExternalForce(h_f_ext_off);
      std::cout << "Released tip force at step " << step << std::endl;
    }
  };

  data.Setup(L, W, H, Quadrature::gauss_xi_m_7, Quadrature::gauss_eta_m_7,
             Quadrature::gauss_zeta_m_3, Quadrature::gauss_xi_4,
             Quadrature::gauss_eta_4, Quadrature::gauss_zeta_3,
             Quadrature::weight_xi_m_7, Quadrature::weight_eta_m_7,
             Quadrature::weight_zeta_m_3, Quadrature::weight_xi_4,
             Quadrature::weight_eta_4, Quadrature::weight_zeta_3, h_x12, h_y12,
             h_z12, element_connectivity);

  data.SetDensity(kRho0);
  data.SetDamping(0.0, 0.0);
  data.SetSVK(kE, kNu);

  data.CalcDsDuPre();
  data.CalcMassMatrix();
  data.CalcConstraintData();
  data.ConvertToCSR_ConstraintJacT();
  data.BuildConstraintJacobianCSR();
  data.CalcP();
  data.CalcInternalForce();

  auto want_vtu = [&](int step) {
    return opt.write_vtu && (step % kVtuEvery) == 0;
  };

  auto write_vtu = [&](int step, const Eigen::VectorXd& x12,
                       const Eigen::VectorXd& y12, const Eigen::VectorXd& z12) {
    std::ostringstream oss;
    oss << vtu_out_dir << "/" << kVtuPrefix << "_" << SolverName(opt.solver)
        << "_" << std::setw(6) << std::setfill('0') << step << ".vtu";
    ANCFCPUUtils::VisualizationUtils::ExportANCF3443ToVTU(
        x12, y12, z12, element_connectivity, H, oss.str());
  };

  if (want_vtu(0)) {
    Eigen::VectorXd x12, y12, z12;
    data.RetrievePositionToCPU(x12, y12, z12);
    write_vtu(0, x12, y12, z12);
  }

  std::vector<double> tip_z_history;
  if (opt.write_csv) {
    tip_z_history.reserve(static_cast<size_t>(opt.steps));
  }

  auto tip_z_average = [&](const Eigen::VectorXd& z12) {
    double sum = 0.0;
    for (int node : tip_nodes) {
      sum += z12(node * 4 + 0);
    }
    return sum / static_cast<double>(tip_nodes.size());
  };

  switch (opt.solver) {
    case SolverKind::kNewton: {
      SyncedNewtonParams params = {1e-4, 1e-4, 1e-4, 1e14, 5, 10, opt.dt};
      SyncedNewtonSolver solver(&data, data.get_n_constraint());
      solver.Setup();
      solver.SetParameters(&params);
      for (int step = 0; step < opt.steps; ++step) {
        maybe_release_force(step);
        solver.Solve();
        const int out_step = step + 1;
        const bool do_vtu  = want_vtu(out_step);
        if (opt.write_csv || do_vtu) {
          Eigen::VectorXd x12, y12, z12;
          data.RetrievePositionToCPU(x12, y12, z12);
          if (opt.write_csv) {
            tip_z_history.push_back(tip_z_average(z12));
          }
          if (do_vtu) {
            write_vtu(out_step, x12, y12, z12);
          }
        }
      }
      break;
    }
    case SolverKind::kNesterov: {
      SyncedNesterovParams params = {1.0e-8, 1e14, 1.0e-6, 1.0e-6,
                                     5,      300,  opt.dt};
      SyncedNesterovSolver solver(&data, data.get_n_constraint());
      solver.Setup();
      solver.SetParameters(&params);
      for (int step = 0; step < opt.steps; ++step) {
        maybe_release_force(step);
        solver.Solve();
        const int out_step = step + 1;
        const bool do_vtu  = want_vtu(out_step);
        if (opt.write_csv || do_vtu) {
          Eigen::VectorXd x12, y12, z12;
          data.RetrievePositionToCPU(x12, y12, z12);
          if (opt.write_csv) {
            tip_z_history.push_back(tip_z_average(z12));
          }
          if (do_vtu) {
            write_vtu(out_step, x12, y12, z12);
          }
        }
      }
      break;
    }
    case SolverKind::kAdamW: {
      SyncedAdamWNocoopParams params;
      if (opt.res == 0) {
        params = {3e-1,  0.8,  0.9999, 1e-8, 0.0,   0.997, 1e-4,
                  1e-4,  1e14, 5,      800,   opt.dt, 100,   1e-4};
      } else if (opt.res == 2) {
        params = {2.5e-1, 0.8,  0.999,  1e-8, 0.0,   0.998, 1e-4,
                  1e-4,  1e14, 5,      800,   opt.dt, 100,   1e-4};
      } else if (opt.res == 4) {
        params = {8.0e-2, 0.8,  0.999,  1e-8, 0.0,   0.9986, 1e-4,
                  1e-4,  1e14, 5,      1000,   opt.dt, 100,   1e-4};
      } else if (opt.res == 8) {
        params = {3.0e-2, 0.9,  0.999,  1e-1, 0.0,   0.9986, 1e-3,
                  1e-3,  1e14, 5,      1200,   opt.dt, 100,   1e-3};
      } else if (opt.res == 16) {
        params = {3.0e-2, 0.9,  0.999,  1e-1, 0.0,   0.9986, 1e-3,
                  1e-3,  1e14, 5,      1500,   opt.dt, 100,   1e-3};
      } else if (opt.res == 32) {
        params = {3.0e-2, 0.9,  0.999,  1e-1, 0.0,   0.9986, 1e-3,
                  1e-3,  1e14, 5,      1800,   opt.dt, 100,   1e-3};
      } else {
        std::cerr << "Unsupported resolution for AdamW: " << opt.res
                  << std::endl;
        return 1;
      }
      SyncedAdamWNocoopSolver solver(&data, data.get_n_constraint());
      solver.Setup();
      solver.SetParameters(&params);
      for (int step = 0; step < opt.steps; ++step) {
        maybe_release_force(step);
        solver.Solve();
        const int out_step = step + 1;
        const bool do_vtu  = want_vtu(out_step);
        if (opt.write_csv || do_vtu) {
          Eigen::VectorXd x12, y12, z12;
          data.RetrievePositionToCPU(x12, y12, z12);
          if (opt.write_csv) {
            tip_z_history.push_back(tip_z_average(z12));
          }
          if (do_vtu) {
            write_vtu(out_step, x12, y12, z12);
          }
        }
      }
      break;
    }
    case SolverKind::kVbd: {
      const double omega     = std::isnan(opt.omega) ? 1.8 : opt.omega;
      SyncedVBDParams params = {1e-4,   1e-4,  1e-4,  1e14, 5, 500,
                                opt.dt, omega, 1e-12, 25,   1};
      SyncedVBDSolver solver(&data, data.get_n_constraint());
      solver.Setup();
      solver.SetParameters(&params);
      solver.InitializeColoring();
      solver.InitializeMassDiagBlocks();
      solver.InitializeFixedMap();
      for (int step = 0; step < opt.steps; ++step) {
        maybe_release_force(step);
        solver.Solve();
        const int out_step = step + 1;
        const bool do_vtu  = want_vtu(out_step);
        if (opt.write_csv || do_vtu) {
          Eigen::VectorXd x12, y12, z12;
          data.RetrievePositionToCPU(x12, y12, z12);
          if (opt.write_csv) {
            tip_z_history.push_back(tip_z_average(z12));
          }
          if (do_vtu) {
            write_vtu(out_step, x12, y12, z12);
          }
        }
      }
      break;
    }
  }

  if (opt.write_csv) {
    std::string out_path = opt.csv_path;
    if (out_path.empty()) {
      out_path = "tip_z_history_ancf3443_" + SolverName(opt.solver) + ".csv";
    }
    WriteTipCsv(out_path, tip_z_history);
    std::cout << "Wrote " << out_path << std::endl;
  }

  data.Destroy();
  return 0;
}
