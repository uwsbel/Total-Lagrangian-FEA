/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli
 * Email:   arivoli@wisc.edu
 * File:    test_feat10_bunny_explicit.cc
 * Brief:   FEAT10 bunny explicit dynamics demo. Supports both
 *          standard (FEAT10 + SyncedExplicit) and optimized
 *          (FEAT10Opt + SyncedExplicitOpt) solver paths via
 *          --opt flag. Clamps base, loads ears, releases halfway.
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

// Material properties
constexpr double kE    = 3.0e8;   // Young's modulus (Pa)
constexpr double kNu   = 0.40;    // Poisson's ratio
constexpr double kRho0 = 920.0;   // Density (kg/m^3)

// Mooney-Rivlin derived from SVK params
constexpr double kMu       = kE / (2.0 * (1.0 + kNu));      // shear modulus
constexpr double kBulk     = kE / (3.0 * (1.0 - 2.0 * kNu)); // bulk modulus
constexpr double kMR_mu10  = 0.30 * kMu;
constexpr double kMR_mu01  = 0.20 * kMu;
constexpr double kMR_kappa = 1.5 * kBulk;

// Boundary conditions
constexpr double kFixedZThreshold = -4.0;   // fix nodes with z < this
constexpr double kForceZThreshold = 4.0;    // apply force to nodes with z > this
constexpr double kForceZ          = -35000.0; // N, downward on ears

enum class MaterialKind { kSVK, kMR };

struct Options {
  bool use_opt          = false;
  MaterialKind material = MaterialKind::kSVK;
  double dt             = 1e-6;
  int steps             = 5000;
  bool write_csv        = false;
  std::string csv_path;
  int vtk_interval      = 0;  // 0 = disabled
  std::string vtk_dir   = "output";  // VTK output directory
};

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

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [--opt] [--mat=MAT] [--dt=DT] [--steps=N] [--csv[=PATH]] "
         "[--vtk=N] [--vtk-dir=PATH] [--help]\n"
      << "  --opt          Use FEAT10Opt + SyncedExplicitOpt (default: standard)\n"
      << "  --mat=MAT      svk | mr (default: svk; ignored with --opt)\n"
      << "  --dt=DT        Time step (default: 1e-6)\n"
      << "  --steps=N      Number of steps (default: 5000)\n"
      << "  --csv[=P]      Write CSV output (optional path)\n"
      << "  --vtk=N        VTK output every N steps (standard path only)\n"
      << "  --vtk-dir=PATH VTK output directory (default: output)\n";
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

  // CUDA device init
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

  const std::string node_file =
      mesh_path("data/meshes/T10/bunny_ascii_26.1.node");
  const std::string elem_file =
      mesh_path("data/meshes/T10/bunny_ascii_26.1.ele");

  const int n_nodes = ANCFCPUUtils::FEAT10_read_nodes(node_file.c_str(), nodes);
  const int n_elems =
      ANCFCPUUtils::FEAT10_read_elements(elem_file.c_str(), elements);

  std::cout << "Mesh loaded: " << n_nodes << " nodes, " << n_elems
            << " elements" << std::endl;

  // Extract coordinate vectors
  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = nodes(i, 0);
    h_y12(i) = nodes(i, 1);
    h_z12(i) = nodes(i, 2);
  }

  // Fixed nodes: z < -4.0 (base clamping, same as Newton demo)
  std::vector<int> fixed_nodes;
  fixed_nodes.reserve(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    if (h_z12(i) < kFixedZThreshold) {
      fixed_nodes.push_back(i);
    }
  }
  std::cout << "Fixed nodes (z < " << kFixedZThreshold
            << "): " << fixed_nodes.size() << std::endl;

  // Force nodes: z > 4.0 (ear loading, same as Newton demo)
  std::vector<int> force_node_indices;
  for (int i = 0; i < n_nodes; ++i) {
    if (h_z12(i) > kForceZThreshold) {
      force_node_indices.push_back(i);
    }
  }
  std::cout << "Force nodes (z > " << kForceZThreshold
            << "): " << force_node_indices.size() << std::endl;

  // Track node: highest initial z (bunny ear tip)
  int track_node = 0;
  double max_z   = h_z12(0);
  for (int i = 1; i < n_nodes; ++i) {
    if (h_z12(i) > max_z) {
      max_z      = h_z12(i);
      track_node = i;
    }
  }
  std::cout << "Tracking node " << track_node << " (initial z = " << max_z
            << ")" << std::endl;

  // Print run configuration
  std::cout << "Config: solver=" << (opt.use_opt ? "opt" : "standard")
            << " mat=" << (opt.use_opt ? "mr(forced)" : MaterialName(opt.material))
            << " dt=" << opt.dt << " steps=" << opt.steps << std::endl;

  if (opt.use_opt && opt.vtk_interval > 0) {
    std::cout << "Warning: VTK output not supported with --opt, ignoring --vtk"
              << std::endl;
    opt.vtk_interval = 0;
  }

  // Open CSV file if requested
  std::ofstream csv_file;
  std::string csv_out_path;
  if (opt.write_csv) {
    csv_out_path = opt.csv_path;
    if (csv_out_path.empty()) {
      csv_out_path = JoinPath(
          DefaultOutputDir(),
          "bunny_explicit_" +
              std::string(opt.use_opt ? "opt" : MaterialName(opt.material)) +
              ".csv");
    }
    csv_file.open(csv_out_path);
    csv_file << std::fixed << std::setprecision(17);
    csv_file << "step,x_position,y_position,z_position,internal_force_l2\n";
  }

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
    gpu_feat10opt.SetDensity(kRho0);

    std::cout << "Material (MR): mu10=" << kMR_mu10 << ", mu01=" << kMR_mu01
              << ", kappa=" << kMR_kappa << std::endl;

    gpu_feat10opt.ComputePrecomputation();
    gpu_feat10opt.ComputeLumpedMassHRZ();

    // External force (float for Opt path)
    Eigen::VectorXf f_ext(n_nodes * 3);
    f_ext.setZero();
    for (int idx : force_node_indices) {
      f_ext(3 * idx + 2) = static_cast<float>(kForceZ);
    }

    SyncedExplicitOptSolver solver(&gpu_feat10opt);
    solver.SetTimeStep(opt.dt);
    solver.SetFixedNodes(fixed_nodes);
    solver.SetExternalForce(f_ext);

    for (int step = 0; step < opt.steps; ++step) {
      // Remove force at halfway point
      if (step == opt.steps / 2) {
        f_ext.setZero();
        solver.SetExternalForce(f_ext);
        std::cout << "External force removed at step " << step << std::endl;
      }

      solver.Solve();

      const double kernel_time = solver.GetLastStepTimeMs();
      step_times_ms.push_back(kernel_time);

      // Periodic console output and CSV
      if (step % 500 == 0 || step == opt.steps - 1 || opt.write_csv) {
        Eigen::VectorXd pos_x, pos_y, pos_z;
        gpu_feat10opt.RetrievePositionToCPU(pos_x, pos_y, pos_z);

        if (step % 500 == 0 || step == opt.steps - 1) {
          std::cout << "Step " << step << "/" << opt.steps << ": node "
                    << track_node << " pos = (" << std::setprecision(6)
                    << pos_x(track_node) << ", " << pos_y(track_node) << ", "
                    << pos_z(track_node) << ")" << std::endl;
        }

        if (opt.write_csv) {
          Eigen::VectorXf f_int;
          gpu_feat10opt.RetrieveInternalForceToCPU(f_int);
          const double l2 = static_cast<double>(f_int.norm());
          csv_file << step << "," << pos_x(track_node) << ","
                   << pos_y(track_node) << "," << pos_z(track_node) << "," << l2
                   << "\n";
        }
      }
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

    gpu_t10_data.Setup(tet5pt_x, tet5pt_y, tet5pt_z, tet5pt_w, h_x12, h_y12,
                       h_z12, elements);
    gpu_t10_data.SetDensity(kRho0);
    gpu_t10_data.SetDamping(0.0, 0.0);

    if (opt.material == MaterialKind::kSVK) {
      gpu_t10_data.SetSVK(kE, kNu);
      std::cout << "Material: SVK (E=" << kE << ", nu=" << kNu << ")"
                << std::endl;
    } else {
      gpu_t10_data.SetMooneyRivlin(kMR_mu10, kMR_mu01, kMR_kappa);
      std::cout << "Material (MR): mu10=" << kMR_mu10 << ", mu01=" << kMR_mu01
                << ", kappa=" << kMR_kappa << std::endl;
    }

    gpu_t10_data.CalcDnDuPre();
    gpu_t10_data.CalcLumpedMassHRZ();

    // External force (double for standard path)
    Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
    h_f_ext.setZero();
    for (int idx : force_node_indices) {
      h_f_ext(3 * idx + 2) = kForceZ;
    }
    gpu_t10_data.SetExternalForce(h_f_ext);

    SyncedExplicitParams params = {opt.dt};
    SyncedExplicitSolver solver(&gpu_t10_data);
    solver.SetParameters(&params);
    solver.SetFixedNodes(fixed_nodes);

    int vtk_frame = 0;

    for (int step = 0; step < opt.steps; ++step) {
      // Remove force at halfway point
      if (step == opt.steps / 2) {
        h_f_ext.setZero();
        gpu_t10_data.SetExternalForce(h_f_ext);
        std::cout << "External force removed at step " << step << std::endl;
      }

      solver.Solve();

      const double kernel_time = solver.GetLastStepTime();
      step_times_ms.push_back(kernel_time);

      // VTK output
      if (opt.vtk_interval > 0 && step % opt.vtk_interval == 0) {
        gpu_t10_data.WriteOutputVTK(JoinPath(opt.vtk_dir, "bunny_explicit_step_" +
                                    std::to_string(vtk_frame) + ".vtk"));
        vtk_frame++;
      }

      // Periodic console output and CSV
      if (step % 500 == 0 || step == opt.steps - 1 || opt.write_csv) {
        double x_track = 0.0, y_track = 0.0, z_track = 0.0;
        HANDLE_ERROR(cudaMemcpy(&x_track,
                                gpu_t10_data.GetX12DevicePtr() + track_node,
                                sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(&y_track,
                                gpu_t10_data.GetY12DevicePtr() + track_node,
                                sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(&z_track,
                                gpu_t10_data.GetZ12DevicePtr() + track_node,
                                sizeof(double), cudaMemcpyDeviceToHost));

        if (step % 500 == 0 || step == opt.steps - 1) {
          std::cout << "Step " << step << "/" << opt.steps << ": node "
                    << track_node << " pos = (" << std::setprecision(6)
                    << x_track << ", " << y_track << ", " << z_track << ")"
                    << std::endl;
        }

        if (opt.write_csv) {
          Eigen::VectorXd f_int;
          gpu_t10_data.RetrieveInternalForceToCPU(f_int);
          const double l2 = f_int.norm();
          csv_file << step << "," << x_track << "," << y_track << ","
                   << z_track << "," << l2 << "\n";
        }
      }
    }

    gpu_t10_data.Destroy();
  }

  // Timing summary
  if (!step_times_ms.empty()) {
    double total_time_ms = 0.0;
    for (double t : step_times_ms) {
      total_time_ms += t;
    }
    const double avg_time_ms =
        total_time_ms / static_cast<double>(opt.steps);
    std::cout << "\nTiming summary:" << std::endl;
    std::cout << "  Total simulation time: " << total_time_ms << " ms"
              << std::endl;
    std::cout << "  Average step time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (1000.0 / avg_time_ms) << " steps/sec"
              << std::endl;
  }

  if (opt.write_csv) {
    csv_file.close();
    std::cout << "Wrote CSV: " << csv_out_path << std::endl;
  }

  return 0;
}
