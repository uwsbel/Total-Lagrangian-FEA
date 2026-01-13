/**
 * ANCF3243 Net Mesh Demo (Newton)
 *
 * Loads a `.ancf3243mesh` net, builds pinned/welded joint constraints from the
 * mesh file, clamps the four net corners, and runs a few Newton steps.
 *
 * Example:
 *   bazel run //lib_bin/mesh_deform:test_ancf3243_net_newton -- --vtu
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/elements/ANCF3243Data.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/mesh_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/visualization_utils.h"

namespace {

constexpr double kE    = 7e8;
constexpr double kNu   = 0.33;
constexpr double kRho0 = 2700;

constexpr double kDefaultW = 0.1;
constexpr double kDefaultH = 0.1;

constexpr int kVtuEvery          = 10;
constexpr const char* kVtuPrefix = "ancf3243_net";
constexpr const char* kDefaultWeldedMeshPath =
    "data/meshes/ANCF3243/net_welded_nx20_ny20_L0.5.ancf3243mesh";
constexpr const char* kDefaultPinnedMeshPath =
    "data/meshes/ANCF3243/net_pinned_nx20_ny20_L0.5.ancf3243mesh";

struct Options {
  std::string joint     = "welded";  // "pinned" or "welded"
  int steps             = 50;
  double dt             = 1e-3;
  double W              = kDefaultW;
  double H              = kDefaultH;
  double center_force_z = -1000.0;
  bool write_vtu        = false;
  bool verbose          = false;
};

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [--joint=KIND] [--steps=N] [--dt=DT] [--W=W] [--H=H]\n"
      << "               [--center_force_z=FZ] [--vtu] [--verbose] [--help]\n\n"
      << "  --joint=KIND        welded | pinned (default: welded)\n"
      << "  --steps=N            number of Solve() calls (default: 50)\n"
      << "  --dt=DT              time step (default: 1e-3)\n"
      << "  --W=W                beam width (default: 0.1)\n"
      << "  --H=H                beam height (default: 0.1)\n"
      << "  --center_force_z=FZ  downward point load at net center (default: "
         "-1000)\n"
      << "  --vtu                write VTU meshes to output/ancf3243_net/ "
         "(every 10 steps)\n";
  std::cout << "  --verbose            print selected center/corner nodes\n";
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
    if (StartsWith(arg, "--joint=")) {
      opt.joint = arg.substr(std::string("--joint=").size());
      if (opt.joint != "welded" && opt.joint != "pinned") {
        std::cerr << "Invalid --joint (expected welded|pinned): " << opt.joint
                  << "\n";
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
    if (StartsWith(arg, "--W=")) {
      const std::string v = arg.substr(std::string("--W=").size());
      if (!ParseDouble(v, opt.W) || !(opt.W > 0.0)) {
        std::cerr << "Invalid --W: " << v << "\n";
        return false;
      }
      continue;
    }
    if (StartsWith(arg, "--H=")) {
      const std::string v = arg.substr(std::string("--H=").size());
      if (!ParseDouble(v, opt.H) || !(opt.H > 0.0)) {
        std::cerr << "Invalid --H: " << v << "\n";
        return false;
      }
      continue;
    }
    if (StartsWith(arg, "--center_force_z=")) {
      const std::string v = arg.substr(std::string("--center_force_z=").size());
      if (!ParseDouble(v, opt.center_force_z)) {
        std::cerr << "Invalid --center_force_z: " << v << "\n";
        return false;
      }
      continue;
    }
    if (arg == "--vtu") {
      opt.write_vtu = true;
      continue;
    }
    if (arg == "--verbose") {
      opt.verbose = true;
      continue;
    }
    std::cerr << "Unknown argument: " << arg << "\n";
    return false;
  }
  return true;
}

struct Bounds2D {
  double xmin = std::numeric_limits<double>::infinity();
  double xmax = -std::numeric_limits<double>::infinity();
  double ymin = std::numeric_limits<double>::infinity();
  double ymax = -std::numeric_limits<double>::infinity();
};

Bounds2D ComputeBoundsXY(const ANCFCPUUtils::ANCF3243Mesh& mesh) {
  Bounds2D b;
  for (int nid = 0; nid < mesh.n_nodes; ++nid) {
    const double x = mesh.x12(4 * nid + 0);
    const double y = mesh.y12(4 * nid + 0);
    b.xmin         = std::min(b.xmin, x);
    b.xmax         = std::max(b.xmax, x);
    b.ymin         = std::min(b.ymin, y);
    b.ymax         = std::max(b.ymax, y);
  }
  return b;
}

struct GridInfo {
  bool valid = false;
  int nx     = 0;
  int ny     = 0;
  double L   = 0.0;
  double ox  = 0.0;
  double oy  = 0.0;
};

GridInfo GetGridInfo(const ANCFCPUUtils::ANCF3243Mesh& mesh) {
  GridInfo info;
  if (!mesh.grid_nx.has_value() || !mesh.grid_ny.has_value() ||
      !mesh.grid_L.has_value() || !mesh.grid_origin.has_value()) {
    return info;
  }
  info.nx    = *mesh.grid_nx;
  info.ny    = *mesh.grid_ny;
  info.L     = *mesh.grid_L;
  info.ox    = (*mesh.grid_origin)(0);
  info.oy    = (*mesh.grid_origin)(1);
  info.valid = (info.nx >= 1 && info.ny >= 1 && info.L > 0.0);
  return info;
}

std::vector<int> FindNodesAtXY(const ANCFCPUUtils::ANCF3243Mesh& mesh, double x,
                               double y, double tol) {
  std::vector<int> nodes;
  for (int nid = 0; nid < mesh.n_nodes; ++nid) {
    const double xn = mesh.x12(4 * nid + 0);
    const double yn = mesh.y12(4 * nid + 0);
    if (std::abs(xn - x) <= tol && std::abs(yn - y) <= tol) {
      nodes.push_back(nid);
    }
  }
  return nodes;
}

void AppendCornerClamps(ANCFCPUUtils::LinearConstraintBuilder& builder,
                        const ANCFCPUUtils::ANCF3243Mesh& mesh, bool verbose) {
  std::vector<std::pair<double, double>> corners;
  const GridInfo grid = GetGridInfo(mesh);
  if (grid.valid) {
    corners = {
        {grid.ox + 0.0 * grid.L, grid.oy + 0.0 * grid.L},
        {grid.ox + grid.nx * grid.L, grid.oy + 0.0 * grid.L},
        {grid.ox + 0.0 * grid.L, grid.oy + grid.ny * grid.L},
        {grid.ox + grid.nx * grid.L, grid.oy + grid.ny * grid.L},
    };
  } else {
    const Bounds2D b = ComputeBoundsXY(mesh);
    corners          = {
        {b.xmin, b.ymin}, {b.xmax, b.ymin}, {b.xmin, b.ymax}, {b.xmax, b.ymax}};
  }

  const Bounds2D b  = ComputeBoundsXY(mesh);
  const double diag = std::hypot(b.xmax - b.xmin, b.ymax - b.ymin);
  const double tol  = std::max(1e-12, 1e-9 * diag);

  for (const auto& [cx, cy] : corners) {
    const std::vector<int> nodes = FindNodesAtXY(mesh, cx, cy, tol);
    if (nodes.empty()) {
      std::cerr << "Warning: could not find any node at corner (" << cx << ", "
                << cy << ")\n";
      continue;
    }
    if (verbose) {
      std::cout << "Corner clamp (" << cx << ", " << cy
                << "): found nodes=" << nodes.size() << " [";
      for (size_t i = 0; i < nodes.size(); ++i) {
        std::cout << nodes[i] << (i + 1 == nodes.size() ? "" : ",");
      }
      std::cout << "]\n";
    }
    if (nodes.size() != 2) {
      std::cerr << "Warning: expected 2 (H/V) nodes at corner (" << cx << ", "
                << cy << "), got " << nodes.size() << "\n";
    }
    for (int nid : nodes) {
      for (int slot = 0; slot < 4; ++slot) {
        const int coef = 4 * nid + slot;
        ANCFCPUUtils::AppendANCF3243FixedCoefficient(builder, coef, mesh.x12,
                                                     mesh.y12, mesh.z12);
      }
    }
  }
}

void ApplyCenterPointLoad(Eigen::VectorXd& f_ext,
                          const ANCFCPUUtils::ANCF3243Mesh& mesh,
                          double force_z, bool verbose) {
  std::vector<std::pair<double, double>> load_points;
  const GridInfo grid = GetGridInfo(mesh);
  if (grid.valid) {
    const int i0 = grid.nx / 2;
    const int j0 = grid.ny / 2;
    const int i1 = (grid.nx % 2 == 0) ? i0 : (i0 + 1);
    const int j1 = (grid.ny % 2 == 0) ? j0 : (j0 + 1);
    for (int ii : {i0, i1}) {
      for (int jj : {j0, j1}) {
        load_points.push_back({grid.ox + ii * grid.L, grid.oy + jj * grid.L});
      }
    }
  } else {
    const Bounds2D b = ComputeBoundsXY(mesh);
    load_points.push_back({0.5 * (b.xmin + b.xmax), 0.5 * (b.ymin + b.ymax)});
  }

  const Bounds2D b  = ComputeBoundsXY(mesh);
  const double diag = std::hypot(b.xmax - b.xmin, b.ymax - b.ymin);
  const double tol  = std::max(1e-12, 1e-9 * diag);

  std::vector<int> loaded_nodes;
  for (const auto& [cx, cy] : load_points) {
    const std::vector<int> nodes = FindNodesAtXY(mesh, cx, cy, tol);
    for (int nid : nodes)
      loaded_nodes.push_back(nid);
  }

  std::sort(loaded_nodes.begin(), loaded_nodes.end());
  loaded_nodes.erase(std::unique(loaded_nodes.begin(), loaded_nodes.end()),
                     loaded_nodes.end());

  if (loaded_nodes.empty()) {
    std::cerr
        << "Warning: no center load node(s) found; skipping center load.\n";
    return;
  }
  if (verbose) {
    std::cout << "Center load points=" << load_points.size()
              << " unique_nodes=" << loaded_nodes.size() << " [";
    for (size_t i = 0; i < loaded_nodes.size(); ++i) {
      std::cout << loaded_nodes[i] << (i + 1 == loaded_nodes.size() ? "" : ",");
    }
    std::cout << "]\n";
  }
  if (loaded_nodes.size() % 2 != 0) {
    std::cerr
        << "Warning: expected an even number of loaded nodes (H/V pairs), got "
        << loaded_nodes.size() << "\n";
  }

  const double per_node = force_z / static_cast<double>(loaded_nodes.size());
  for (int nid : loaded_nodes) {
    const int coef_pos = 4 * nid + 0;
    f_ext(coef_pos * 3 + 2) += per_node;
  }
}

std::string PickDefaultMeshPath(const std::string& joint) {
  namespace fs                             = std::filesystem;
  const std::vector<std::string> preferred = {
      joint == "pinned" ? kDefaultPinnedMeshPath : kDefaultWeldedMeshPath,
      joint == "pinned" ? kDefaultWeldedMeshPath : kDefaultPinnedMeshPath,
  };

  for (const auto& path : preferred) {
    if (fs::exists(path)) {
      return path;
    }
  }

  const fs::path dir("data/meshes/ANCF3243");
  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    return preferred.front();
  }

  std::vector<fs::path> candidates;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file())
      continue;
    const std::string name = entry.path().filename().string();
    if (name.find("net_" + joint + "_") != 0)
      continue;
    if (entry.path().extension().string() != ".ancf3243mesh")
      continue;
    candidates.push_back(entry.path());
  }

  if (candidates.empty()) {
    return preferred.front();
  }
  std::sort(candidates.begin(), candidates.end());
  return candidates.front().string();
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
  }

  Options opt;
  if (!ParseArgs(argc, argv, opt)) {
    return 1;
  }

  const std::string mesh_path = PickDefaultMeshPath(opt.joint);
  if (!std::filesystem::exists(mesh_path)) {
    std::cerr << "Mesh file not found: " << mesh_path << "\n";
    return 2;
  }

  ANCFCPUUtils::ANCF3243Mesh mesh;
  std::string err;
  if (!ANCFCPUUtils::ReadANCF3243MeshFromFile(mesh_path, mesh, &err)) {
    std::cerr << err << "\n";
    return 2;
  }

  const double L = mesh.grid_L.value_or(0.5);
  if (!(L > 0.0)) {
    std::cerr << "Invalid mesh L.\n";
    return 2;
  }

  if (opt.write_vtu) {
    std::filesystem::create_directories("output/ancf3243_net");
  }

  const int n_nodes    = mesh.n_nodes;
  const int n_elements = mesh.n_elements;

  std::cout << "ANCF3243 net: mesh=" << mesh_path << " nodes=" << n_nodes
            << " elements=" << n_elements << " coef=" << (4 * n_nodes)
            << " constraints(from mesh)=" << mesh.constraints.NumRows()
            << " steps=" << opt.steps << " dt=" << opt.dt << " L=" << L
            << " W=" << opt.W << " H=" << opt.H
            << " center_force_z=" << opt.center_force_z << std::endl;

  GPU_ANCF3243_Data data(n_nodes, n_elements);
  data.Initialize();

  Eigen::VectorXd h_f_ext(4 * n_nodes * 3);
  h_f_ext.setZero();
  ApplyCenterPointLoad(h_f_ext, mesh, opt.center_force_z, opt.verbose);
  data.SetExternalForce(h_f_ext);

  data.Setup(L, opt.W, opt.H, Quadrature::gauss_xi_m_6, Quadrature::gauss_xi_3,
             Quadrature::gauss_eta_2, Quadrature::gauss_zeta_2,
             Quadrature::weight_xi_m_6, Quadrature::weight_xi_3,
             Quadrature::weight_eta_2, Quadrature::weight_zeta_2, mesh.x12,
             mesh.y12, mesh.z12, mesh.element_connectivity);

  data.SetDensity(kRho0);
  data.SetDamping(1e5, 1e5);
  data.SetSVK(kE, kNu);

  const int n_coef = 4 * n_nodes;
  const int n_dofs = n_coef * 3;

  std::unique_ptr<ANCFCPUUtils::LinearConstraintBuilder> builder;
  if (mesh.constraints.Empty()) {
    builder = std::make_unique<ANCFCPUUtils::LinearConstraintBuilder>(n_dofs);
  } else {
    builder = std::make_unique<ANCFCPUUtils::LinearConstraintBuilder>(
        n_dofs, mesh.constraints);
  }
  AppendCornerClamps(*builder, mesh, opt.verbose);
  const ANCFCPUUtils::LinearConstraintCSR all_constraints = builder->ToCSR();

  data.SetLinearConstraintsCSR(all_constraints.offsets, all_constraints.columns,
                               all_constraints.values, all_constraints.rhs);

  data.CalcDsDuPre();
  data.CalcMassMatrix();
  data.CalcConstraintData();
  data.CalcP();
  data.CalcInternalForce();

  auto want_vtu = [&](int step) {
    return opt.write_vtu && (step % kVtuEvery) == 0;
  };

  auto write_vtu = [&](int step, const Eigen::VectorXd& x12,
                       const Eigen::VectorXd& y12, const Eigen::VectorXd& z12) {
    std::ostringstream oss;
    oss << "output/ancf3243_net/" << kVtuPrefix << "_" << std::setw(6)
        << std::setfill('0') << step << ".vtu";
    ANCFCPUUtils::VisualizationUtils::ExportANCF3243ToVTU(
        x12, y12, z12, mesh.element_connectivity, opt.W, opt.H, oss.str());
  };

  if (want_vtu(0)) {
    Eigen::VectorXd x12, y12, z12;
    data.RetrievePositionToCPU(x12, y12, z12);
    write_vtu(0, x12, y12, z12);
  }

  SyncedNewtonParams params = {1e-4, 0.0, 1e-6, 1e14, 5, 10, opt.dt};
  SyncedNewtonSolver solver(&data, data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);

  for (int step = 0; step < opt.steps; ++step) {
    solver.Solve();
    const int out_step = step + 1;
    if (want_vtu(out_step)) {
      Eigen::VectorXd x12, y12, z12;
      data.RetrievePositionToCPU(x12, y12, z12);
      write_vtu(out_step, x12, y12, z12);
    }
  }

  data.Destroy();
  return 0;
}
