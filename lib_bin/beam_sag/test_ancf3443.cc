/**
 * ANCF3443 Shell Unified Solver Test
 *
 * This binary unifies the ANCF3443 shell driver across multiple solvers
 * (Newton / Nesterov / AdamW / VBD). Use `--solver=...` to select the solver.
 *
 * Example:
 *   ./bazel-bin/test_ancf3443 --solver=vbd --steps=50 --dt=1e-3 --omega=1.8 --csv
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_src/elements/ANCF3443Data.cuh"
#include "../../lib_src/solvers/SyncedAdamWNocoop.cuh"
#include "../../lib_src/solvers/SyncedNesterov.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_src/solvers/SyncedVBD.cuh"
#include "../../lib_utils/cpu_utils.h"

namespace {

constexpr double kE = 7e8;
constexpr double kNu = 0.33;
constexpr double kRho0 = 2700;

enum class SolverKind { kNewton, kNesterov, kAdamW, kVbd };

struct Options {
  SolverKind solver = SolverKind::kVbd;
  int steps = 50;
  double dt = 1e-3;
  double omega = std::numeric_limits<double>::quiet_NaN();  // VBD only
  bool write_csv = false;
  std::string csv_path;
};

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [--solver=SOLVER] [--steps=N] [--dt=DT]\n"
      << "                 [--omega=W] [--csv[=PATH]] [--help]\n\n"
      << "  --solver=SOLVER   newton | nesterov | adamw | vbd (default: vbd)\n"
      << "  --steps=N         number of Solve() calls (default: 50)\n"
      << "  --dt=DT           time step passed to solver params (default: 1e-3)\n"
      << "  --omega=W         VBD relaxation factor (default: 1.8)\n"
      << "  --csv[=PATH]      write tip displacement CSV (default path depends on solver)\n";
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

  const int n_beam = 2;
  GPU_ANCF3443_Data data(n_beam);
  data.Initialize();

  const double L = 2.0, W = 1.0, H = 1.0;

  std::cout << "ANCF3443: beams=" << n_beam << " coef=" << data.get_n_coef()
            << " solver=" << SolverName(opt.solver) << " steps=" << opt.steps
            << " dt=" << opt.dt << std::endl;

  Eigen::MatrixXd h_B_inv(Quadrature::N_SHAPE_3443, Quadrature::N_SHAPE_3443);
  ANCFCPUUtils::ANCF3443_B12_matrix(L, W, H, h_B_inv,
                                    Quadrature::N_SHAPE_3443);

  Eigen::VectorXd h_x12(data.get_n_coef());
  Eigen::VectorXd h_y12(data.get_n_coef());
  Eigen::VectorXd h_z12(data.get_n_coef());
  Eigen::MatrixXi element_connectivity(data.get_n_beam(), 4);
  ANCFCPUUtils::ANCF3443_generate_beam_coordinates(n_beam, h_x12, h_y12, h_z12,
                                                   element_connectivity);

  Eigen::VectorXi h_fixed_nodes(8);
  h_fixed_nodes << 0, 1, 2, 3, 12, 13, 14, 15;
  data.SetNodalFixed(h_fixed_nodes);

  Eigen::VectorXd h_f_ext(data.get_n_coef() * 3);
  h_f_ext.setZero();
  h_f_ext(3 * data.get_n_coef() - 4) = -125.0;
  h_f_ext(3 * data.get_n_coef() - 10) = 500.0;
  h_f_ext(3 * data.get_n_coef() - 16) = 125.0;
  h_f_ext(3 * data.get_n_coef() - 22) = 500.0;
  data.SetExternalForce(h_f_ext);

  data.Setup(L, W, H, h_B_inv, Quadrature::gauss_xi_m_7,
             Quadrature::gauss_eta_m_7, Quadrature::gauss_zeta_m_3,
             Quadrature::gauss_xi_4, Quadrature::gauss_eta_4,
             Quadrature::gauss_zeta_3, Quadrature::weight_xi_m_7,
             Quadrature::weight_eta_m_7, Quadrature::weight_zeta_m_3,
             Quadrature::weight_xi_4, Quadrature::weight_eta_4,
             Quadrature::weight_zeta_3, h_x12, h_y12, h_z12,
             element_connectivity);

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

  const int tip_idx = data.get_n_coef() - 4;
  std::vector<double> tip_z_history;
  if (opt.write_csv) {
    tip_z_history.reserve(static_cast<size_t>(opt.steps));
  }

  switch (opt.solver) {
    case SolverKind::kNewton: {
      SyncedNewtonParams params = {1e-4, 0.0, 1e-6, 1e14, 5, 10, opt.dt};
      SyncedNewtonSolver solver(&data, data.get_n_constraint());
      solver.Setup();
      solver.SetParameters(&params);
      for (int step = 0; step < opt.steps; ++step) {
        solver.Solve();
        if (opt.write_csv) {
          Eigen::VectorXd x12, y12, z12;
          data.RetrievePositionToCPU(x12, y12, z12);
          tip_z_history.push_back(z12(tip_idx));
        }
      }
      break;
    }
    case SolverKind::kNesterov: {
      SyncedNesterovParams params = {1.0e-8, 1e14, 1.0e-6, 1.0e-6, 5, 300,
                                     opt.dt};
      SyncedNesterovSolver solver(&data, data.get_n_constraint());
      solver.Setup();
      solver.SetParameters(&params);
      for (int step = 0; step < opt.steps; ++step) {
        solver.Solve();
        if (opt.write_csv) {
          Eigen::VectorXd x12, y12, z12;
          data.RetrievePositionToCPU(x12, y12, z12);
          tip_z_history.push_back(z12(tip_idx));
        }
      }
      break;
    }
    case SolverKind::kAdamW: {
      SyncedAdamWNocoopParams params = {
          2e-4, 0.9, 0.999, 1e-8, 1e-4, 0.995, 1e-1,
          1e-6, 1e14, 5,    500,  opt.dt, 10,  0.0};
      SyncedAdamWNocoopSolver solver(&data, data.get_n_constraint());
      solver.Setup();
      solver.SetParameters(&params);
      for (int step = 0; step < opt.steps; ++step) {
        solver.Solve();
        if (opt.write_csv) {
          Eigen::VectorXd x12, y12, z12;
          data.RetrievePositionToCPU(x12, y12, z12);
          tip_z_history.push_back(z12(tip_idx));
        }
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
        if (opt.write_csv) {
          Eigen::VectorXd x12, y12, z12;
          data.RetrievePositionToCPU(x12, y12, z12);
          tip_z_history.push_back(z12(tip_idx));
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
