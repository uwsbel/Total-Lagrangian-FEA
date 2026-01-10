/**
 * Item Drop Onto Floor Simulation
 * Author: Json Zhou (zzhou292@wisc.edu)
 *
 * Drops a deformable item onto a deformable floor slab with its bottom layer
 * vertices fixed. The item falls under gravity and interacts via broadphase +
 * narrowphase contact forces.
 *
 * Solver:
 *   Newton (cuDSS) only.
 */

#include <cuda_runtime.h>

#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/collision/Broadphase.cuh"
#include "../../lib_src/collision/Narrowphase.cuh"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/visualization_utils.h"

// Material properties (for deformable items)
const double E_val = 2e6;    // Young's modulus (Pa)
const double nu    = 0.3;   // Poisson's ratio
const double rho0  = 500.0; // Density (kg/m^3)

// Simulation parameters
const double gravity = -9.81; // Gravity acceleration (m/s^2)
const double dt      = 5e-4;  // Time step (s)
const int    num_steps_default = 4000;

// Contact parameters
const double contact_damping_default  = 0.0;
const double contact_friction_default = 0.6;

using ANCFCPUUtils::VisualizationUtils;

struct BBox {
  Eigen::Vector3d min;
  Eigen::Vector3d max;
  Eigen::Vector3d size() const { return max - min; }
  Eigen::Vector3d center() const { return 0.5 * (min + max); }
};

static BBox ComputeBBox(const Eigen::MatrixXd& nodes,
                        const ANCFCPUUtils::MeshInstance& inst) {
  BBox bb;
  bb.min = Eigen::Vector3d(1e30, 1e30, 1e30);
  bb.max = Eigen::Vector3d(-1e30, -1e30, -1e30);
  for (int i = 0; i < inst.num_nodes; ++i) {
    const int idx = inst.node_offset + i;
    bb.min(0) = std::min(bb.min(0), nodes(idx, 0));
    bb.min(1) = std::min(bb.min(1), nodes(idx, 1));
    bb.min(2) = std::min(bb.min(2), nodes(idx, 2));
    bb.max(0) = std::max(bb.max(0), nodes(idx, 0));
    bb.max(1) = std::max(bb.max(1), nodes(idx, 1));
    bb.max(2) = std::max(bb.max(2), nodes(idx, 2));
  }
  return bb;
}

static void PrintBBox(const std::string& label, const BBox& bb) {
  const Eigen::Vector3d sz = bb.size();
  const Eigen::Vector3d c  = bb.center();
  std::cout << "  " << label << ":\n"
            << "    min = [" << bb.min(0) << ", " << bb.min(1) << ", "
            << bb.min(2) << "]\n"
            << "    max = [" << bb.max(0) << ", " << bb.max(1) << ", "
            << bb.max(2) << "]\n"
            << "    size= [" << sz(0) << ", " << sz(1) << ", " << sz(2) << "]\n"
            << "    ctr = [" << c(0) << ", " << c(1) << ", " << c(2) << "]\n";
}

static bool BBoxOverlaps(const BBox& a, const BBox& b, double eps = 0.0) {
  const bool overlap_x = (a.min(0) <= b.max(0) + eps) && (a.max(0) + eps >= b.min(0));
  const bool overlap_y = (a.min(1) <= b.max(1) + eps) && (a.max(1) + eps >= b.min(1));
  const bool overlap_z = (a.min(2) <= b.max(2) + eps) && (a.max(2) + eps >= b.min(2));
  return overlap_x && overlap_y && overlap_z;
}

namespace {

struct Options {
  double contact_damping  = contact_damping_default;
  double contact_friction = contact_friction_default;
  bool enable_self_collision = false;
  int max_steps = num_steps_default;
  int export_interval = 10;
};

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

bool ParseInt(const std::string& s, int& out) {
  try {
    size_t idx = 0;
    int v      = std::stoi(s, &idx);
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
    double v   = std::stod(s, &idx);
    if (idx != s.size()) return false;
    out = v;
    return true;
  } catch (...) {
    return false;
  }
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage:\n"
      << "  " << argv0
      << " [contact_damping] [contact_friction] [self_collision(0/1)] [max_steps] [export_interval]\n"
      << "  " << argv0 << " [positional args...] [--help]\n";
}

bool ParseArgs(int argc, char** argv, Options& opt) {
  int positional_index = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    }
    if (StartsWith(arg, "--")) {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }

    // Backward-compatible positional args.
    switch (positional_index) {
      case 0: {
        if (!ParseDouble(arg, opt.contact_damping)) return false;
        break;
      }
      case 1: {
        if (!ParseDouble(arg, opt.contact_friction)) return false;
        break;
      }
      case 2: {
        int v = 0;
        if (!ParseInt(arg, v)) return false;
        opt.enable_self_collision = (v != 0);
        break;
      }
      case 3: {
        int v = 0;
        if (!ParseInt(arg, v) || v <= 0) return false;
        opt.max_steps = v;
        break;
      }
      case 4: {
        int v = 0;
        if (!ParseInt(arg, v)) return false;
        opt.export_interval = v;
        break;
      }
      default:
        std::cerr << "Too many positional arguments: " << arg << "\n";
        return false;
    }
    ++positional_index;
  }
  return true;
}

}  // namespace

static void CheckCublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error (" << what << "): status=" << int(status) << "\n";
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  std::cout << "========================================\n";
  std::cout << "Item Drop Onto Floor Simulation\n";
  std::cout << "========================================\n";

  Options opt;
  if (!ParseArgs(argc, argv, opt)) {
    return 1;
  }

  std::cout << "Solver: newton\n";
  std::cout << "Contact damping: " << opt.contact_damping << "\n";
  std::cout << "Contact friction: " << opt.contact_friction << "\n";
  std::cout << "Enable self collision: " << (opt.enable_self_collision ? 1 : 0)
            << "\n";
  std::cout << "Max steps: " << opt.max_steps << "\n";
  std::cout << "Export interval: " << opt.export_interval << "\n";

  std::filesystem::create_directories("output/item_drop");

  // =========================================================================
  // Load meshes using MeshManager
  // =========================================================================
  ANCFCPUUtils::MeshManager mesh_manager;
  const std::string item_mesh_path  = "data/meshes/T10/item_drop/";
  const std::string floor_mesh_path = "data/meshes/T10/bubble_gripper_bunny/";

  const int mesh_floor =
      mesh_manager.LoadMesh(floor_mesh_path + "1_1_01_floor.1.node",
                            floor_mesh_path + "1_1_01_floor.1.ele", "floor");
  const int mesh_dragon1 =
      mesh_manager.LoadMesh(item_mesh_path + "dragon.node",
                            item_mesh_path + "dragon.ele",
                            "dragon_1");
  if (mesh_floor < 0 || mesh_dragon1 < 0) {
    std::cerr << "Failed to load one or more meshes.\n"
              << "  floor path: " << floor_mesh_path << "\n"
              << "  item path:  " << item_mesh_path << "\n";
    return 1;
  }

  const auto& inst_floor   = mesh_manager.GetMeshInstance(mesh_floor);
  const auto& inst_dragon1 = mesh_manager.GetMeshInstance(mesh_dragon1);

  std::cout << "Loaded meshes:\n";
  std::cout << "  Floor:        " << inst_floor.num_nodes << " nodes, "
            << inst_floor.num_elements << " elements\n";
  std::cout << "  Dragon 1:     " << inst_dragon1.num_nodes << " nodes, "
            << inst_dragon1.num_elements << " elements\n";

  // =========================================================================
  // Load pressure fields for collision detection
  // =========================================================================
  bool ok0 = mesh_manager.LoadScalarFieldFromNpz(
      mesh_floor, floor_mesh_path + "1_1_01_floor.1.npz", "p_vertex");
  bool ok1 = mesh_manager.LoadScalarFieldFromNpz(mesh_dragon1,
                                                 item_mesh_path + "dragon.npz",
                                                 "p_vertex");
  if (!ok0 || !ok1) {
    std::cerr << "Failed to load pressure fields from NPZ\n";
    return 1;
  }

  // =========================================================================
  // Scale + place floor and drop items above it
  // =========================================================================
  {
    // The item_drop assets are ~2x2x1 scale (matching the old open-box).
    // Upscale the 1x1 floor slab to better match that footprint.
    const double floor_scale = 2.0;
    mesh_manager.TransformMesh(mesh_floor,
                               ANCFCPUUtils::uniformScale(floor_scale));

    const Eigen::MatrixXd& nodes0 = mesh_manager.GetAllNodes();
    const BBox bb_floor = ComputeBBox(nodes0, inst_floor);
    const BBox bb_d1  = ComputeBBox(nodes0, inst_dragon1);

    std::cout << "Initial bounding boxes:\n";
    PrintBBox("floor", bb_floor);
    PrintBBox("dragon_1", bb_d1);

    auto centerMeshAt = [&](int mesh_id,
                            const ANCFCPUUtils::MeshInstance& inst,
                            const Eigen::Vector3d& target_center) {
      const Eigen::MatrixXd& nodes_before = mesh_manager.GetAllNodes();
      const BBox bb = ComputeBBox(nodes_before, inst);
      const Eigen::Vector3d delta = target_center - bb.center();
      mesh_manager.TransformMesh(mesh_id,
                                 ANCFCPUUtils::translation(delta(0), delta(1),
                                                          delta(2)));
    };

    // Place the floor so its top surface is at z=0 and centered at (0,0).
    const double floor_thickness = bb_floor.size()(2);
    centerMeshAt(mesh_floor, inst_floor,
                 Eigen::Vector3d(0.0, 0.0, -0.5 * floor_thickness));

    const Eigen::MatrixXd& nodes_floor_placed = mesh_manager.GetAllNodes();
    const BBox bb_floor_placed = ComputeBBox(nodes_floor_placed, inst_floor);
    const Eigen::Vector3d floor_center = bb_floor_placed.center();
    const double floor_top_z = bb_floor_placed.max(2);

    const double xy_offset = 0.05;  // small lateral offsets (stay near center)
    const double base_gap  = 0.05;  // initial clearance above the floor

    const double dz_d = 0.5 * bb_d1.size()(2);

    centerMeshAt(mesh_dragon1, inst_dragon1,
                 Eigen::Vector3d(floor_center(0) + xy_offset,
                                 floor_center(1) + xy_offset,
                                 floor_top_z + base_gap + dz_d));

    const Eigen::MatrixXd& nodes1 = mesh_manager.GetAllNodes();
    BBox bb_floor_final  = ComputeBBox(nodes1, inst_floor);
    BBox bb_dragon_final = ComputeBBox(nodes1, inst_dragon1);

    std::cout << "Placed floor and item:\n";
    PrintBBox("floor", bb_floor_final);
    PrintBBox("dragon_1", bb_dragon_final);
  }

  // =========================================================================
  // Build unified arrays
  // =========================================================================
  const Eigen::MatrixXd& initial_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& elements      = mesh_manager.GetAllElements();
  const Eigen::VectorXd& pressure_raw  = mesh_manager.GetAllScalarFields();

  const int n_nodes = mesh_manager.GetTotalNodes();
  const int n_elems = mesh_manager.GetTotalElements();

  std::cout << "Total nodes: " << n_nodes << "\n";
  std::cout << "Total elements: " << n_elems << "\n";
  std::cout << "Pressure field size: " << pressure_raw.size() << "\n";

  Eigen::VectorXd pressure = pressure_raw;

  // =========================================================================
  // Initialize GPU element data
  // =========================================================================
  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = initial_nodes(i, 0);
    h_y12(i) = initial_nodes(i, 1);
    h_z12(i) = initial_nodes(i, 2);
  }

  // Fix floor bottom layer nodes (static support)
  std::vector<int> fixed_node_indices;
  fixed_node_indices.reserve(inst_floor.num_nodes);
  double floor_z_min = 1e30;
  for (int i = 0; i < inst_floor.num_nodes; ++i) {
    const int idx = inst_floor.node_offset + i;
    floor_z_min = std::min(floor_z_min, initial_nodes(idx, 2));
  }
  const double floor_z_fix_threshold = floor_z_min + 1e-6;
  for (int i = 0; i < inst_floor.num_nodes; ++i) {
    const int idx = inst_floor.node_offset + i;
    if (initial_nodes(idx, 2) <= floor_z_fix_threshold) {
      fixed_node_indices.push_back(idx);
    }
  }
  Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(i) = fixed_node_indices[i];
  }
  std::cout << "Fixed " << h_fixed_nodes.size() << " floor bottom nodes\n";
  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  const Eigen::VectorXd& tet5pt_x       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights = Quadrature::tet5pt_weights;

  const double eta_damp    = 1e3;
  const double lambda_damp = 1e3;
  gpu_t10_data.Setup(tet5pt_x, tet5pt_y, tet5pt_z, tet5pt_weights, h_x12, h_y12,
                     h_z12, elements);
  gpu_t10_data.SetDensity(rho0);
  gpu_t10_data.SetDamping(eta_damp, lambda_damp);
  gpu_t10_data.SetSVK(E_val, nu);
  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();

  // Lumped mass per node (row-sum of scalar mass matrix)
  Eigen::VectorXd lumped_mass(n_nodes);
  lumped_mass.setZero();
  {
    std::vector<int> offsets;
    std::vector<int> columns;
    std::vector<double> values;
    gpu_t10_data.RetrieveMassCSRToCPU(offsets, columns, values);
    if (static_cast<int>(offsets.size()) == n_nodes + 1) {
      for (int i = 0; i < n_nodes; ++i) {
        double sum = 0.0;
        for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
          sum += values[k];
        }
        lumped_mass(i) = sum;
      }
    } else {
      std::cerr << "Warning: unexpected mass CSR offsets size " << offsets.size()
                << " (expected " << (n_nodes + 1)
                << "); using unit mass for gravity.\n";
      lumped_mass.setOnes();
    }
  }

  // =========================================================================
  // Solver setup
  // =========================================================================
  SyncedNewtonParams params = {1e-4, 0.0, 1e-6, 1e12, 3, 10, dt};
  auto newton_solver = std::make_unique<SyncedNewtonSolver>(
      &gpu_t10_data, gpu_t10_data.get_n_constraint());
  newton_solver->Setup();
  newton_solver->SetParameters(&params);
  newton_solver->AnalyzeHessianSparsity();
  double* d_vel_guess = newton_solver->GetVelocityGuessDevicePtr();
  if (d_vel_guess == nullptr) {
    std::cerr << "Error: solver velocity guess pointer is null.\n";
    return 1;
  }
  // Contact friction/damping uses this velocity buffer; initialize it so
  // narrowphase doesn't read uninitialized device memory on step 0.
  HANDLE_ERROR(cudaMemset(d_vel_guess, 0, n_nodes * 3 * sizeof(double)));

  // =========================================================================
  // Collision detection setup
  // =========================================================================
  Broadphase broadphase;
  Narrowphase narrowphase;

  Eigen::VectorXi elementMeshIds(n_elems);
  for (int i = 0; i < mesh_manager.GetNumMeshes(); ++i) {
    const auto& instance = mesh_manager.GetMeshInstance(i);
    for (int e = 0; e < instance.num_elements; ++e) {
      elementMeshIds(instance.element_offset + e) = i;
    }
  }

  broadphase.Initialize(mesh_manager);
  broadphase.EnableSelfCollision(opt.enable_self_collision);
  broadphase.BuildNeighborMap();

  narrowphase.Initialize(initial_nodes, elements, pressure, elementMeshIds);
  narrowphase.EnableSelfCollision(opt.enable_self_collision);

  // Shared device node buffer for collision (column-major: [x... y... z...]).
  // Updated from the dynamics state each step via device-to-device copies.
  double* d_nodes_collision = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_nodes_collision, n_nodes * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_nodes_collision, gpu_t10_data.GetX12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(
      cudaMemcpy(d_nodes_collision + n_nodes, gpu_t10_data.GetY12DevicePtr(),
                 n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(d_nodes_collision + 2 * n_nodes,
                          gpu_t10_data.GetZ12DevicePtr(), n_nodes * sizeof(double),
                          cudaMemcpyDeviceToDevice));

  broadphase.BindNodesDevicePtr(d_nodes_collision);
  narrowphase.BindNodesDevicePtr(d_nodes_collision);

  broadphase.CreateAABB();
  broadphase.SortAABBs(0);

  // Precompute gravity on host once and keep it on device.
  Eigen::VectorXd h_f_gravity = Eigen::VectorXd::Zero(n_nodes * 3);
  auto addGravityForInstance = [&](const ANCFCPUUtils::MeshInstance& inst) {
    for (int i = 0; i < inst.num_nodes; ++i) {
      const int idx = inst.node_offset + i;
      h_f_gravity(3 * idx + 2) += lumped_mass(idx) * gravity;
    }
  };
  addGravityForInstance(inst_dragon1);

  double* d_f_gravity = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_f_gravity, n_nodes * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_f_gravity, h_f_gravity.data(),
                          n_nodes * 3 * sizeof(double),
                          cudaMemcpyHostToDevice));

  cublasHandle_t cublas_handle = nullptr;
  CheckCublas(cublasCreate(&cublas_handle), "cublasCreate");

  // =========================================================================
  // Simulation loop
  // =========================================================================
  std::cout << "\nStarting simulation (" << opt.max_steps << " steps)\n\n";

  for (int step = 0; step < opt.max_steps; ++step) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1) Update collision node buffer from the solver state (device->device)
    HANDLE_ERROR(cudaMemcpy(d_nodes_collision, gpu_t10_data.GetX12DevicePtr(),
                            n_nodes * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_nodes_collision + n_nodes, gpu_t10_data.GetY12DevicePtr(),
                   n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(d_nodes_collision + 2 * n_nodes,
                            gpu_t10_data.GetZ12DevicePtr(), n_nodes * sizeof(double),
                            cudaMemcpyDeviceToDevice));

    // 2) Collision detection
    broadphase.CreateAABB();
    broadphase.SortAABBs(0);
    broadphase.DetectCollisions(false);
    const int num_collision_pairs = broadphase.numCollisions;

    narrowphase.SetCollisionPairsDevice(broadphase.GetCollisionPairsDevicePtr(),
                                        num_collision_pairs);
    narrowphase.ComputeContactPatches();

    // 3) External forces: contact + gravity
    narrowphase.ComputeExternalForcesGPUDevice(d_vel_guess, opt.contact_damping,
                                               opt.contact_friction);

    HANDLE_ERROR(cudaMemcpy(gpu_t10_data.GetExternalForceDevicePtr(), d_f_gravity,
                            n_nodes * 3 * sizeof(double),
                            cudaMemcpyDeviceToDevice));

    if (num_collision_pairs > 0) {
      const double alpha = 1.0;
      CheckCublas(cublasDaxpy(
                      cublas_handle, n_nodes * 3, &alpha,
                      narrowphase.GetExternalForcesDevicePtr(), 1,
                      gpu_t10_data.GetExternalForceDevicePtr(), 1),
                  "cublasDaxpy(contact + gravity)");
    }

    // 4) Solve one step
    newton_solver->Solve();

    // 5) Export (host copies only when writing output)
    if (opt.export_interval > 0 && step % opt.export_interval == 0) {
      Eigen::VectorXd x12_current, y12_current, z12_current;
      gpu_t10_data.RetrievePositionToCPU(x12_current, y12_current, z12_current);

      Eigen::MatrixXd current_nodes(n_nodes, 3);
      Eigen::VectorXd displacement(n_nodes * 3);
      for (int i = 0; i < n_nodes; ++i) {
        current_nodes(i, 0)       = x12_current(i);
        current_nodes(i, 1)       = y12_current(i);
        current_nodes(i, 2)       = z12_current(i);
        displacement(3 * i + 0)   = x12_current(i) - initial_nodes(i, 0);
        displacement(3 * i + 1)   = y12_current(i) - initial_nodes(i, 1);
        displacement(3 * i + 2)   = z12_current(i) - initial_nodes(i, 2);
      }

      std::ostringstream filename;
      filename << "output/item_drop/mesh_" << std::setfill('0') << std::setw(4)
               << step << ".vtu";
      VisualizationUtils::ExportMeshWithDisplacement(current_nodes, elements,
                                                     displacement,
                                                     filename.str());

      std::ostringstream patch_filename;
      patch_filename << "output/item_drop/patches_" << std::setfill('0')
                     << std::setw(4) << step << ".vtp";
      narrowphase.RetrieveResults();
      std::vector<ContactPatch> patches = narrowphase.GetValidPatches();
      VisualizationUtils::ExportContactPatchesToVTP(patches,
                                                    patch_filename.str());
    }

    // 6) Progress
    if (step % 20 == 0) {
      auto t1 = std::chrono::high_resolution_clock::now();
      const double step_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      std::cout << "Step " << std::setw(4) << step << ": pairs="
                << std::setw(6) << num_collision_pairs << ", ms="
                << std::fixed << std::setprecision(2) << step_ms << "\n";
    }
  }

  CheckCublas(cublasDestroy(cublas_handle), "cublasDestroy");
  HANDLE_ERROR(cudaFree(d_f_gravity));
  HANDLE_ERROR(cudaFree(d_nodes_collision));

  std::cout << "Done.\n";
  return 0;
}
