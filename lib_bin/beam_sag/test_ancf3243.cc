/**
 * ANCF3243 Beam Unified Solver Test
 *
 * This binary unifies the ANCF3243 beam driver across multiple solvers
 * (Newton / Nesterov / AdamW / VBD). Use `--solver=...` to select the solver.
 *
 * Example:
 *   ./bazel-bin/test_ancf3243 --solver=vbd --res=0 --steps=200 --dt=1e-3
 * --tip_force_z=5000 --force_release_step=100 --vtu --csv
 */

 //./bazel-bin/lib_bin/beam_sag/test_ancf3243 --solver=adamw --res=16

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/elements/ANCF3243Data.cuh"
#include "../../lib_src/solvers/SyncedAdamWNocoop.cuh"
#include "../../lib_src/solvers/SyncedNesterov.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_src/solvers/SyncedVBD.cuh"
#include "../../lib_utils/mesh_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/visualization_utils.h"

namespace {

constexpr double kE    = 7e8;
constexpr double kNu   = 0.33;
constexpr double kRho0 = 2700;

constexpr double kDefaultL         = 0.2;
constexpr double kDefaultW         = 0.1;
constexpr double kDefaultH         = 0.1;
constexpr double kDefaultTipForceZ = 5000.0;
constexpr int kDefaultRes = 0;
constexpr int kDefaultForceReleaseStep = 100;

constexpr int kVtuEvery          = 20;
constexpr const char* kVtuPrefix = "ancf3243";

enum class SolverKind { kNewton, kNesterov, kAdamW, kVbd };

struct Options {
  SolverKind solver  = SolverKind::kVbd;
  int res            = kDefaultRes;  // res -> num_elements mapping below
  int num_elements_override = -1;    // -1 means use res mapping
  int steps          = 200;
  double dt          = 1e-3;
  double tip_force_z = std::numeric_limits<double>::quiet_NaN();
  int force_release_step = kDefaultForceReleaseStep;  // -1 means never release
  double omega       = std::numeric_limits<double>::quiet_NaN();  // VBD only
  bool write_csv     = false;
  std::string csv_path;
  bool write_vtu = false;
};

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [--solver=SOLVER] [--res=R] [--steps=N] [--dt=DT]\n"
      << "                 [--tip_force_z=FZ]\n"
      << "                 [--force_release_step=N]\n"
      << "                 [--omega=W] [--csv[=PATH]] [--vtu] [--help]\n\n"
      << "  --solver=SOLVER   newton | nesterov | adamw | vbd (default: vbd)\n"
      << "  --res=R           0 | 2 | 4 | 8 | 16 | 32 (default: 0)\n"
      << "                   (0->1000, 2->10000, 4->50000, 8->100000, 16->200000, 32->500000)\n"
      << "  --steps=N         number of Solve() calls (default: 200)\n"
      << "  --dt=DT           time step passed to solver params (default: "
         "1e-3)\n"
      << "  --tip_force_z=FZ  vertical point force at tip node (default: "
         "5000)\n"
      << "  --force_release_step=N  set tip force to 0 starting at step N\n"
      << "                      (default: 100, use -1 to never release)\n"
      << "  --omega=W         VBD relaxation factor (default: 1.0)\n"
      << "  --csv[=PATH]      write tip displacement CSV (default path depends "
         "on solver)\n"
      << "  --vtu             write VTU beam hex meshes to output/ancf3243/ "
         "(every 20 steps)\n";
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

int NumElementsForRes(int res) {
  switch (res) {
    case 0:
      return 1000;
    case 2:
      return 10000;
    case 4:
      return 50000;
    case 8:
      return 100000;
    case 16:
      return 200000;
    case 32:
      return 500000;
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
      if (!ParseInt(v, r) || NumElementsForRes(r) <= 0) {
        std::cerr << "Invalid --res: " << v << " (expected 0|2|4|8|16|32)\n";
        return false;
      }
      opt.res = r;
      continue;
    }
    // Back-compat / escape hatch (not advertised): allow specifying element
    // count directly. If you pass this, it overrides --res.
    if (StartsWith(arg, "--num_elements=") || StartsWith(arg, "--n_elements=") ||
        StartsWith(arg, "--n_beam=")) {
      const std::string v = arg.substr(arg.find('=') + 1);
      if (!ParseInt(v, opt.num_elements_override) || opt.num_elements_override <= 0) {
        std::cerr << "Invalid --num_elements: " << v << "\n";
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
    if (StartsWith(arg, "--tip_force_z=")) {
      const std::string v = arg.substr(std::string("--tip_force_z=").size());
      if (!ParseDouble(v, opt.tip_force_z)) {
        std::cerr << "Invalid --tip_force_z: " << v << "\n";
        return false;
      }
      continue;
    }
    if (StartsWith(arg, "--force_release_step=") ||
        StartsWith(arg, "--release_force_step=")) {
      const std::string v = arg.substr(arg.find('=') + 1);
      if (!ParseInt(v, opt.force_release_step) || opt.force_release_step < -1) {
        std::cerr << "Invalid --force_release_step (expected -1 or >=0): " << v
                  << "\n";
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

int SelectTipNodeByMaxX(const Eigen::VectorXd& x12, int n_nodes) {
  if (n_nodes <= 0) {
    return 0;
  }
  int tip_node = 0;
  double max_x = x12(0);
  for (int node = 1; node < n_nodes; ++node) {
    const double x = x12(4 * node);
    if (x > max_x) {
      max_x   = x;
      tip_node = node;
    }
  }
  return tip_node;
}

bool ValidateBeamConnectivityChain(const Eigen::MatrixXi& conn, int n_nodes,
                                   std::string* error) {
  const int n_elements = static_cast<int>(conn.rows());
  if (conn.cols() != 2) {
    if (error) {
      *error = "connectivity must have 2 columns";
    }
    return false;
  }
  if (n_elements <= 0) {
    if (error) {
      *error = "n_elements must be > 0";
    }
    return false;
  }
  if (n_nodes != n_elements + 1) {
    if (error) {
      std::ostringstream oss;
      oss << "expected n_nodes == n_elements + 1 for a 1D chain (n_nodes="
          << n_nodes << ", n_elements=" << n_elements << ")";
      *error = oss.str();
    }
    return false;
  }
  for (int e = 0; e < n_elements; ++e) {
    const int n0 = conn(e, 0);
    const int n1 = conn(e, 1);
    if (n0 != e || n1 != e + 1) {
      if (error) {
        std::ostringstream oss;
        oss << "unexpected chain connectivity at element " << e << ": (" << n0
            << "," << n1 << "), expected (" << e << "," << (e + 1) << ")";
        *error = oss.str();
      }
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!ParseArgs(argc, argv, opt)) {
    return 1;
  }

  const int num_elements = (opt.num_elements_override > 0)
                               ? opt.num_elements_override
                               : NumElementsForRes(opt.res);
  if (num_elements <= 0) {
    std::cerr << "Invalid res/num_elements configuration.\n";
    return 1;
  }

  const double L           = kDefaultL;  // Element length.
  const double W           = kDefaultW;  // Cross-section width.
  const double H           = kDefaultH;  // Cross-section height.
  const double beam_length = static_cast<double>(num_elements) * L;

  if (std::isnan(opt.tip_force_z)) {
    opt.tip_force_z = kDefaultTipForceZ;
  }

  const std::string vtu_out_dir = "output/ancf3243";
  if (opt.write_vtu) {
    std::filesystem::create_directories(vtu_out_dir);
  }

  ANCFCPUUtils::GridMeshGenerator grid_gen(beam_length, 0.0, L, true, false);
  grid_gen.generate_mesh();

  const int n_nodes    = grid_gen.get_num_nodes();
  const int n_elements = grid_gen.get_num_elements();

  std::cout << "ANCF3243: nodes=" << n_nodes << " elements=" << n_elements
            << " coef=" << (4 * n_nodes) << " solver=" << SolverName(opt.solver)
            << " res=" << opt.res
            << " num_elements=" << num_elements
            << " steps=" << opt.steps
            << " dt=" << opt.dt << " L=" << L << " W=" << W << " H=" << H
            << " tip_force_z=" << opt.tip_force_z
            << " force_release_step=" << opt.force_release_step
            << std::endl;

  GPU_ANCF3243_Data data(n_nodes, n_elements);
  data.Initialize();

  Eigen::VectorXd h_x12(data.get_n_coef());
  Eigen::VectorXd h_y12(data.get_n_coef());
  Eigen::VectorXd h_z12(data.get_n_coef());
  grid_gen.get_coordinates(h_x12, h_y12, h_z12);

  Eigen::MatrixXi h_element_connectivity;
  grid_gen.get_element_connectivity(h_element_connectivity);

  {
    std::string err;
    if (!ValidateBeamConnectivityChain(h_element_connectivity, n_nodes, &err)) {
      std::cerr << "GridMeshGenerator connectivity validation failed: " << err
                << std::endl;
      return 1;
    }
  }

  Eigen::VectorXi h_fixed_nodes(4);
  h_fixed_nodes << 0, 1, 2, 3;
  data.SetNodalFixed(h_fixed_nodes);

  Eigen::VectorXd h_f_ext_on(data.get_n_coef() * 3);
  h_f_ext_on.setZero();
  const int tip_node        = SelectTipNodeByMaxX(h_x12, n_nodes);
  const int tip_coef        = tip_node * 4;  // coef-slot 0 (position) at tip node
  const double tip_x        = h_x12(tip_coef);
  std::cout << "Tip: node=" << tip_node << " coef=" << tip_coef
            << " x=" << tip_x << " (expected ~" << beam_length << ")"
            << std::endl;
  h_f_ext_on(tip_coef * 3 + 2) = opt.tip_force_z;

  Eigen::VectorXd h_f_ext_off(data.get_n_coef() * 3);
  h_f_ext_off.setZero();

  data.SetExternalForce(h_f_ext_on);

  auto maybe_release_force = [&](int step) {
    if (opt.force_release_step >= 0 && step == opt.force_release_step) {
      data.SetExternalForce(h_f_ext_off);
      std::cout << "Released tip force at step " << step << std::endl;
    }
  };

  data.Setup(L, W, H, Quadrature::gauss_xi_m_6, Quadrature::gauss_xi_3,
             Quadrature::gauss_eta_2, Quadrature::gauss_zeta_2,
             Quadrature::weight_xi_m_6, Quadrature::weight_xi_3,
             Quadrature::weight_eta_2, Quadrature::weight_zeta_2, h_x12, h_y12,
             h_z12, h_element_connectivity);

  data.SetDensity(kRho0);

  // Preserve existing per-solver damping defaults:
  // - Newton test uses strong damping to help convergence.
  // - AdamW/Nesterov/VBD tests use zero damping.
  if (opt.solver == SolverKind::kNewton) {
    data.SetDamping(1e5, 1e5);
  } else {
    data.SetDamping(0.0, 0.0);
  }
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
    ANCFCPUUtils::VisualizationUtils::ExportANCF3243ToVTU(
        x12, y12, z12, h_element_connectivity, W, H, oss.str());
  };

  if (want_vtu(0)) {
    Eigen::VectorXd x12, y12, z12;
    data.RetrievePositionToCPU(x12, y12, z12);
    write_vtu(0, x12, y12, z12);
  }

  const int tip_idx = tip_coef;
  std::vector<double> tip_z_history;
  if (opt.write_csv) {
    tip_z_history.reserve(static_cast<size_t>(opt.steps));
  }

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
            tip_z_history.push_back(z12(tip_idx));
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
                                     5,      200,  opt.dt};
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
            tip_z_history.push_back(z12(tip_idx));
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
        params = {3e-1, 0.8,  0.9999, 1e-8, 0.0,    0.995, 1e-4,
                  1e-4, 1e14, 5,     800,  opt.dt, 40,    1e-4};
      } else if (opt.res == 2) {
        params = {3e-1, 0.8,  0.9999, 1e-8, 0.0,    0.995, 1e-4,
                  1e-4, 1e14, 5,     800,  opt.dt, 40,    1e-4};
      } else if (opt.res == 4) {
        params = {3e-1, 0.8,  0.9999, 1e-8, 0.0,    0.995, 1e-4,
                  1e-4, 1e14, 5,     800,  opt.dt, 50,    1e-4};
      } else if (opt.res == 8) {
        params = {3e-1, 0.8,  0.9999, 1e-8, 0.0,    0.995, 1e-4,
                  1e-4, 1e14, 5,     800,  opt.dt, 50,    1e-4};
      } else if (opt.res == 16) {
        params = {3e-1, 0.8,  0.9999, 1e-8, 0.0,    0.995, 1e-4,
                  1e-4, 1e14, 5,     800,  opt.dt, 50,    1e-4};
      } else if (opt.res == 32) {
        params = {3e-1, 0.8,  0.9999, 1e-8, 0.0,    0.995, 1e-3,
                  1e-3, 1e14, 5,     800,  opt.dt, 50,    1e-3};
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
            tip_z_history.push_back(z12(tip_idx));
          }
          if (do_vtu) {
            write_vtu(out_step, x12, y12, z12);
          }
        }
      }
      break;
    }
    case SolverKind::kVbd: {
      const double omega     = std::isnan(opt.omega) ? 1.0 : opt.omega;
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
            tip_z_history.push_back(z12(tip_idx));
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
      out_path = "tip_z_history_ancf3243_" + SolverName(opt.solver) + ".csv";
    }
    WriteTipCsv(out_path, tip_z_history);
    std::cout << "Wrote " << out_path << std::endl;
  }

  data.Destroy();
  return 0;
}
