/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli
 * Email:   arivoli@wisc.edu
 * File:    test_feat10_explicit.cc
 * Brief:   FEAT10 beam explicit test mirroring
 *          f-form-T10-beam-explicit.py with SyncedExplicitSolver.
 *==============================================================
 *==============================================================*/

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/SyncedExplicit.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/quadrature_utils.h"

namespace {

// SVK material properties (aluminum-like, matching FEniCS)
constexpr double kE    = 7.0e8;   // Young's modulus (Pa)
constexpr double kNu   = 0.33;    // Poisson's ratio
constexpr double kRho0 = 2700.0;  // Density (kg/m³)

// Mooney-Rivlin (rubber-like), used when --material=mr
constexpr double kMR_mu10  = 80000.0;  // Pa
constexpr double kMR_mu01  = 20000.0;  // Pa
constexpr double kMR_kappa = 1e6;      // Pa (bulk modulus)
constexpr double kMR_rho   = 1100.0;   // kg/m³

enum class MaterialKind { kSVK, kMR };

struct Options {
  MaterialKind material = MaterialKind::kSVK;
  double dt             = 1e-5;
  int steps             = 5000;
  int res               = 0;  // 0/2/4/8/16/32 beam resolutions
  bool write_csv        = false;
  std::string csv_path;
};

void PrintUsage(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [--mat=MAT] [--res=R] [--dt=DT] [--steps=N] [--csv[=PATH]] [--help]\n"
            << "  --mat=MAT   svk | mr (default: svk)\n"
            << "  --res=R     0 | 2 | 4 | 8 | 16 | 32 (default: 0)\n";
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

bool ParseMaterial(const std::string& s, MaterialKind& out) {
  if (s == "svk") {
    out = MaterialKind::kSVK;
    return true;
  }
  if (s == "mr") {
    out = MaterialKind::kMR;
    return true;
  }
  return false;
}

std::string MaterialName(MaterialKind m) {
  return m == MaterialKind::kSVK ? "svk" : "mr";
}

bool ParseArgs(int argc, char** argv, Options& opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    }
    if (StartsWith(arg, "--mat=")) {
      const std::string v = arg.substr(std::string("--mat=").size());
      if (!ParseMaterial(v, opt.material)) {
        std::cerr << "Unknown material: " << v << "\n";
        return false;
      }
      continue;
    }
    if (StartsWith(arg, "--res=")) {
      const std::string v = arg.substr(std::string("--res=").size());
      int r               = 0;
      if (!ParseInt(v, r) ||
          !(r == 0 || r == 2 || r == 4 || r == 8 || r == 16 || r == 32)) {
        std::cerr << "Invalid --res: " << v << "\n";
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
  int plot_target_node = 0;
  int n_nodes          = 0;
  int n_elems          = 0;

  const std::string res_str =
      std::to_string(opt.res);  // mirror resolution study driver
  const std::string node_file =
      mesh_path("data/meshes/T10/resolution/beam_3x2x1_res" + res_str +
                ".1.node");
  const std::string elem_file =
      mesh_path("data/meshes/T10/resolution/beam_3x2x1_res" + res_str +
                ".1.ele");

  n_nodes = ANCFCPUUtils::FEAT10_read_nodes(node_file.c_str(), nodes);
  n_elems = ANCFCPUUtils::FEAT10_read_elements(elem_file.c_str(), elements);

  // Target node for tracking (mirror resolution driver mapping)
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
  std::cout << "material=" << MaterialName(opt.material) << " res=" << opt.res
            << " steps=" << opt.steps << " dt=" << opt.dt << std::endl;

  if (n_nodes <= plot_target_node) {
    std::cerr << "Mesh too small for node " << plot_target_node << " tracking."
              << std::endl;
    return 1;
  }

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);
    h_y12(i) = nodes(i, 1);
    h_z12(i) = nodes(i, 2);
  }

  const Eigen::VectorXd& tet5pt_x_host       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y_host       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z_host       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights_host = Quadrature::tet5pt_weights;

  gpu_t10_data.Setup(tet5pt_x_host, tet5pt_y_host, tet5pt_z_host,
                     tet5pt_weights_host, h_x12, h_y12, h_z12, elements);
  if (opt.material == MaterialKind::kSVK) {
    gpu_t10_data.SetDensity(kRho0);
    gpu_t10_data.SetSVK(kE, kNu);
  } else {
    gpu_t10_data.SetDensity(kMR_rho);
    gpu_t10_data.SetMooneyRivlin(kMR_mu10, kMR_mu01, kMR_kappa);
  }
  gpu_t10_data.SetDamping(0.0, 0.0);
  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcLumpedMassHRZ();

  // Fixed nodes: x == 0
  std::vector<int> fixed_nodes;
  fixed_nodes.reserve(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    if (std::abs(h_x12(i)) < 1e-8) {
      fixed_nodes.push_back(i);
    }
  }

  // External force: distribute 10000N in +Z at x == 3
  Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
  h_f_ext.setZero();
  std::vector<int> force_node_indices;
  for (int i = 0; i < h_x12.size(); ++i) {
    if (std::abs(h_x12(i) - 3.0) < 1e-8) {
      force_node_indices.push_back(i);
    }
  }
  if (!force_node_indices.empty()) {
    const double force_per_node = 10000.0 / force_node_indices.size();
    for (int node_idx : force_node_indices) {
      h_f_ext(3 * node_idx + 2) = force_per_node;  // Z direction
    }
  }
  gpu_t10_data.SetExternalForce(h_f_ext);

  SyncedExplicitParams params = {opt.dt};
  SyncedExplicitSolver solver(&gpu_t10_data);
  solver.SetParameters(&params);
  solver.SetFixedNodes(fixed_nodes);

  std::vector<double> target_node_x;
  std::vector<double> target_node_y;
  std::vector<double> target_node_z;
  target_node_x.reserve(opt.steps);
  target_node_y.reserve(opt.steps);
  target_node_z.reserve(opt.steps);

  std::cout << "Running explicit solver: material="
            << MaterialName(opt.material) << " dt=" << opt.dt
            << ", steps=" << opt.steps << std::endl;

  for (int step = 0; step < opt.steps; ++step) {
    // Turn off force at halfway point
    if (step == opt.steps / 2) {
      h_f_ext.setZero();
      gpu_t10_data.SetExternalForce(h_f_ext);
    }
    solver.Solve();

    // Print timing for this step
    std::cout << "Step " << step
              << ": SyncedExplicit kernel time: " << solver.GetLastStepTime()
              << " ms" << std::endl;

    double x_target = 0.0, y_target = 0.0, z_target = 0.0;
    HANDLE_ERROR(cudaMemcpy(&x_target,
                            gpu_t10_data.GetX12DevicePtr() + plot_target_node,
                            sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&y_target,
                            gpu_t10_data.GetY12DevicePtr() + plot_target_node,
                            sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&z_target,
                            gpu_t10_data.GetZ12DevicePtr() + plot_target_node,
                            sizeof(double), cudaMemcpyDeviceToHost));
    target_node_x.push_back(x_target);
    target_node_y.push_back(y_target);
    target_node_z.push_back(z_target);

    if (step % 500 == 0) {
      std::cout << "Step " << step << "/" << opt.steps << ": node "
                << plot_target_node << " x = " << std::setprecision(17)
                << x_target << " y = " << y_target << " z = " << z_target
                << std::endl;
    }
  }

  if (opt.write_csv) {
    std::string out_path = opt.csv_path;
    if (out_path.empty()) {
      out_path =
          JoinPath(DefaultOutputDir(), "feat10_gpu_explicit_" +
                                           MaterialName(opt.material) + ".csv");
    }
    std::ofstream csv_file(out_path);
    csv_file << std::fixed << std::setprecision(17);
    csv_file << "step,x_position,y_position,z_position\n";
    for (int i = 0; i < static_cast<int>(target_node_x.size()); ++i) {
      csv_file << i << "," << target_node_x[i] << "," << target_node_y[i] << ","
               << target_node_z[i] << "\n";
    }
    std::cout << "Wrote CSV: " << out_path << std::endl;
  }

  gpu_t10_data.Destroy();
  return 0;
}
