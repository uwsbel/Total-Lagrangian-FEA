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
#include "../../lib_src/elements/FEAT10DataOpt.cuh"
#include "../../lib_src/solvers/SyncedExplicit.cuh"
#include "../../lib_src/solvers/SyncedExplicitOpt.cuh"
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
  bool use_opt          = false;
  MaterialKind material = MaterialKind::kSVK;
  double dt             = 1e-5;
  int steps             = 5000;
  int res               = 0;  // 0/2/4/8/16/32 beam resolutions
  bool write_csv        = false;
  std::string csv_path;
  int csv_interval      = 1;      // write CSV row every N steps
  bool csv_force        = false;  // include internal_force_l2 column
};

void PrintUsage(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [--opt] [--mat=MAT] [--res=R] [--dt=DT] [--steps=N] [--csv[=PATH]]"
               " [--csv-interval=N] [--csv-force] [--help]\n"
            << "  --opt            Use FEAT10Opt + SyncedExplicitOpt (default: standard)\n"
            << "  --mat=MAT        svk | mr (default: svk; ignored with --opt, forces MR)\n"
            << "  --res=R          0 | 2 | 4 | 8 | 16 | 32 (default: 0)\n"
            << "  --dt=DT          Time step size (default: 1e-5)\n"
            << "  --steps=N        Number of steps (default: 5000)\n"
            << "  --csv[=PATH]     Write CSV output (default path if not specified)\n"
            << "  --csv-interval=N Write CSV row every N steps (default: 1)\n"
            << "  --csv-force      Include internal_force_l2 column in CSV\n"
            << "  --help           Show this message\n";
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
    if (arg == "--opt") {
      opt.use_opt = true;
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
    if (StartsWith(arg, "--csv-interval=")) {
      const std::string v = arg.substr(std::string("--csv-interval=").size());
      if (!ParseInt(v, opt.csv_interval) || opt.csv_interval <= 0) {
        std::cerr << "Invalid --csv-interval: " << v << "\n";
        return false;
      }
      continue;
    }
    if (arg == "--csv-force") {
      opt.csv_force = true;
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
  std::cout << "Config: solver=" << (opt.use_opt ? "opt" : "standard")
            << " mat=" << (opt.use_opt ? "mr(forced)" : MaterialName(opt.material))
            << " res=" << opt.res << " dt=" << opt.dt
            << " steps=" << opt.steps << std::endl;

  if (n_nodes <= plot_target_node) {
    std::cerr << "Mesh too small for node " << plot_target_node << " tracking."
              << std::endl;
    return 1;
  }

  // Coordinate vectors (used by both paths)
  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);
    h_y12(i) = nodes(i, 1);
    h_z12(i) = nodes(i, 2);
  }

  // Fixed nodes: x == 0 (common setup)
  std::vector<int> fixed_nodes;
  fixed_nodes.reserve(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    if (std::abs(h_x12(i)) < 1e-8) {
      fixed_nodes.push_back(i);
    }
  }

  // Force nodes: x == 3.0 (common setup)
  std::vector<int> force_node_indices;
  for (int i = 0; i < h_x12.size(); ++i) {
    if (std::abs(h_x12(i) - 3.0) < 1e-8) {
      force_node_indices.push_back(i);
    }
  }
  const double kForceZ = 10000.0;  // 10 kN total in +Z direction

  // Open CSV file if requested (common setup)
  std::ofstream csv_file;
  std::string csv_out_path;
  if (opt.write_csv) {
    csv_out_path = opt.csv_path;
    if (csv_out_path.empty()) {
      csv_out_path = JoinPath(
          DefaultOutputDir(),
          "beam_explicit_" +
              std::string(opt.use_opt ? "opt" : MaterialName(opt.material)) +
              ".csv");
    }
    csv_file.open(csv_out_path);
    csv_file << std::fixed << std::setprecision(17);
    csv_file << "step,x_position,y_position,z_position"
             << (opt.csv_force ? ",internal_force_l2" : "") << "\n";
  }

  // Timing events for per-step Solve() measurement
  cudaEvent_t timing_start, timing_stop;
  HANDLE_ERROR(cudaEventCreate(&timing_start));
  HANDLE_ERROR(cudaEventCreate(&timing_stop));
  std::vector<double> step_times_ms;
  step_times_ms.reserve(opt.steps);

  // ================================================================
  // Optimized path: FEAT10Opt + SyncedExplicitOpt
  // ================================================================
  if (opt.use_opt) {
    Eigen::MatrixXd positions(n_nodes, 3);
    for (int i = 0; i < n_nodes; i++) {
      positions(i, 0) = nodes(i, 0);
      positions(i, 1) = nodes(i, 1);
      positions(i, 2) = nodes(i, 2);
    }

    GPU_FEAT10Opt_Data gpu_feat10opt;
    gpu_feat10opt.Initialize(n_elems, n_nodes);
    gpu_feat10opt.Setup(positions, elements);
    gpu_feat10opt.SetMooneyRivlin(kMR_mu10, kMR_mu01, kMR_kappa);
    gpu_feat10opt.SetDensity(kMR_rho);

    std::cout << "Material (MR): mu10=" << kMR_mu10 << ", mu01=" << kMR_mu01
              << ", kappa=" << kMR_kappa << std::endl;

    gpu_feat10opt.ComputePrecomputation();
    gpu_feat10opt.ComputeLumpedMassHRZ();

    // External force (float for Opt path)
    Eigen::VectorXf f_ext(n_nodes * 3);
    f_ext.setZero();
    if (!force_node_indices.empty()) {
      const float force_per_node =
          static_cast<float>(kForceZ) / static_cast<float>(force_node_indices.size());
      for (int idx : force_node_indices) {
        f_ext(3 * idx + 2) = force_per_node;  // +Z direction
      }
    }

    SyncedExplicitOptSolver solver(&gpu_feat10opt);
    solver.SetTimeStep(opt.dt);
    solver.SetFixedNodes(fixed_nodes);
    solver.SetExternalForce(f_ext);

    for (int step = 0; step < opt.steps; ++step) {
      if (step == opt.steps / 2) {
        f_ext.setZero();
        solver.SetExternalForce(f_ext);
      }

      float step_time_ms = 0.0f;
      HANDLE_ERROR(cudaEventRecord(timing_start));
      solver.Solve();
      HANDLE_ERROR(cudaEventRecord(timing_stop));
      HANDLE_ERROR(cudaEventSynchronize(timing_stop));
      HANDLE_ERROR(cudaEventElapsedTime(&step_time_ms, timing_start, timing_stop));
      step_times_ms.push_back(static_cast<double>(step_time_ms));

      if (((step + 1) % 500) == 0 || step == opt.steps - 1) {
        std::cout << "Progress: step " << (step + 1) << "/" << opt.steps
                  << std::endl;
      }

      const bool is_csv_step = opt.write_csv &&
          (step % opt.csv_interval == 0 || step == opt.steps - 1);
      if (is_csv_step) {
        double tracked_pos[3];
        HANDLE_ERROR(cudaMemcpy(tracked_pos,
            gpu_feat10opt.GetPositionDevicePtr() + 3 * plot_target_node,
            3 * sizeof(double), cudaMemcpyDeviceToHost));
        csv_file << step << "," << tracked_pos[0] << ","
                 << tracked_pos[1] << "," << tracked_pos[2];
        if (opt.csv_force) {
          Eigen::VectorXf f_int;
          gpu_feat10opt.RetrieveInternalForceToCPU(f_int);
          csv_file << "," << static_cast<double>(f_int.norm());
        }
        csv_file << "\n";
      }
    }

    // Final position
    {
      double tracked_pos[3];
      HANDLE_ERROR(cudaMemcpy(tracked_pos,
          gpu_feat10opt.GetPositionDevicePtr() + 3 * plot_target_node,
          3 * sizeof(double), cudaMemcpyDeviceToHost));
      std::cout << "Final position: node " << plot_target_node << " = ("
                << std::setprecision(6) << tracked_pos[0] << ", "
                << tracked_pos[1] << ", " << tracked_pos[2] << ")" << std::endl;
    }

    gpu_feat10opt.Destroy();

  // ================================================================
  // Standard path: FEAT10 + SyncedExplicit
  // ================================================================
  } else {
    GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
    gpu_t10_data.Initialize();

    const Eigen::VectorXd& tet5pt_x = Quadrature::tet5pt_x;
    const Eigen::VectorXd& tet5pt_y = Quadrature::tet5pt_y;
    const Eigen::VectorXd& tet5pt_z = Quadrature::tet5pt_z;
    const Eigen::VectorXd& tet5pt_w = Quadrature::tet5pt_weights;

    gpu_t10_data.Setup(tet5pt_x, tet5pt_y, tet5pt_z, tet5pt_w,
                       h_x12, h_y12, h_z12, elements);

    if (opt.material == MaterialKind::kSVK) {
      gpu_t10_data.SetDensity(kRho0);
      gpu_t10_data.SetSVK(kE, kNu);
      std::cout << "Material: SVK (E=" << kE << ", nu=" << kNu << ")" << std::endl;
    } else {
      gpu_t10_data.SetDensity(kMR_rho);
      gpu_t10_data.SetMooneyRivlin(kMR_mu10, kMR_mu01, kMR_kappa);
      std::cout << "Material (MR): mu10=" << kMR_mu10 << ", mu01=" << kMR_mu01
                << ", kappa=" << kMR_kappa << std::endl;
    }

    gpu_t10_data.SetDamping(0.0, 0.0);
    gpu_t10_data.CalcDnDuPre();
    gpu_t10_data.CalcLumpedMassHRZ();

    // External force (double for standard path)
    Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
    h_f_ext.setZero();
    if (!force_node_indices.empty()) {
      const double force_per_node = kForceZ / force_node_indices.size();
      for (int idx : force_node_indices) {
        h_f_ext(3 * idx + 2) = force_per_node;  // +Z direction
      }
    }
    gpu_t10_data.SetExternalForce(h_f_ext);

    SyncedExplicitParams params = {opt.dt};
    SyncedExplicitSolver solver(&gpu_t10_data);
    solver.SetParameters(&params);
    solver.SetFixedNodes(fixed_nodes);

    for (int step = 0; step < opt.steps; ++step) {
      if (step == opt.steps / 2) {
        h_f_ext.setZero();
        gpu_t10_data.SetExternalForce(h_f_ext);
      }

      float step_time_ms = 0.0f;
      HANDLE_ERROR(cudaEventRecord(timing_start));
      solver.Solve();
      HANDLE_ERROR(cudaEventRecord(timing_stop));
      HANDLE_ERROR(cudaEventSynchronize(timing_stop));
      HANDLE_ERROR(cudaEventElapsedTime(&step_time_ms, timing_start, timing_stop));
      step_times_ms.push_back(static_cast<double>(step_time_ms));

      if (((step + 1) % 500) == 0 || step == opt.steps - 1) {
        std::cout << "Progress: step " << (step + 1) << "/" << opt.steps
                  << std::endl;
      }

      const bool is_csv_step = opt.write_csv &&
          (step % opt.csv_interval == 0 || step == opt.steps - 1);
      if (is_csv_step) {
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
        csv_file << step << "," << x_target << "," << y_target << ","
                 << z_target;
        if (opt.csv_force) {
          Eigen::VectorXd f_int;
          gpu_t10_data.RetrieveInternalForceToCPU(f_int);
          csv_file << "," << f_int.norm();
        }
        csv_file << "\n";
      }
    }

    // Final position
    {
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
      std::cout << "Final position: node " << plot_target_node << " = ("
                << std::setprecision(6) << x_target << ", " << y_target << ", "
                << z_target << ")" << std::endl;
    }

    gpu_t10_data.Destroy();
  }

  // Timing summary
  if (!step_times_ms.empty()) {
    double total_time_ms = 0.0;
    for (double t : step_times_ms) {
      total_time_ms += t;
    }
    const double avg_time_ms = total_time_ms / static_cast<double>(step_times_ms.size());
    std::cout << "\nTiming summary:" << std::endl;
    std::cout << "  Total solve time: " << total_time_ms << " ms"
              << std::endl;
    std::cout << "  Average solve step time: " << avg_time_ms << " ms"
              << std::endl;
    std::cout << "  Throughput: " << (1000.0 / avg_time_ms) << " solve steps/sec"
              << std::endl;
  }
  HANDLE_ERROR(cudaEventDestroy(timing_start));
  HANDLE_ERROR(cudaEventDestroy(timing_stop));

  if (opt.write_csv) {
    csv_file.close();
    std::cout << "Wrote CSV: " << csv_out_path << std::endl;
  }
  return 0;
}
