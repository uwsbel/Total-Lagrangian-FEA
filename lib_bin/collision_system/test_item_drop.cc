/**
 * Item Drop Into Open Box Simulation
 * Author: Json Zhou (zzhou292@wisc.edu)
 *
 * Drops 2 armadilos and 2 dragons into an open-top box (no lid). The box is
 * fixed; items fall under gravity and interact via broadphase+narrowphase
 * contact forces.
 */

#include <cuda_runtime.h>

#include <cublas_v2.h>

#include <chrono>
#include <Eigen/Dense>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "../../lib_src/collision/Broadphase.cuh"
#include "../../lib_src/collision/Narrowphase.cuh"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/visualization_utils.h"

// Material properties (for deformable items)
const double E_val = 2e6;    // Young's modulus (Pa)
const double nu    = 0.35;   // Poisson's ratio
const double rho0  = 1000.0; // Density (kg/m^3)

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

static void CheckCublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error (" << what << "): status=" << int(status) << "\n";
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  std::cout << "========================================\n";
  std::cout << "Item Drop Into Open Box Simulation\n";
  std::cout << "========================================\n";

  double contact_damping  = contact_damping_default;
  double contact_friction = contact_friction_default;
  bool enable_self_collision = false;
  int max_steps = num_steps_default;
  int export_interval = 10;

  if (argc > 1) contact_damping  = std::atof(argv[1]);
  if (argc > 2) contact_friction = std::atof(argv[2]);
  if (argc > 3) enable_self_collision = (std::atoi(argv[3]) != 0);
  if (argc > 4) {
    int v = std::atoi(argv[4]);
    if (v > 0) max_steps = v;
  }
  if (argc > 5) export_interval = std::atoi(argv[5]);

  std::cout << "Contact damping: " << contact_damping << "\n";
  std::cout << "Contact friction: " << contact_friction << "\n";
  std::cout << "Enable self collision: " << (enable_self_collision ? 1 : 0)
            << "\n";
  std::cout << "Max steps: " << max_steps << "\n";
  std::cout << "Export interval: " << export_interval << "\n";

  std::filesystem::create_directories("output/item_drop");

  // =========================================================================
  // Load meshes using MeshManager
  // =========================================================================
  ANCFCPUUtils::MeshManager mesh_manager;
  const std::string mesh_path = "data/meshes/T10/item_drop/";

  const int mesh_openbox =
      mesh_manager.LoadMesh(mesh_path + "openbox.node", mesh_path + "openbox.ele",
                            "openbox_fixed");
  const int mesh_dragon1 =
      mesh_manager.LoadMesh(mesh_path + "dragon.node", mesh_path + "dragon.ele",
                            "dragon_1");
  const int mesh_arm1 =
      mesh_manager.LoadMesh(mesh_path + "armadilo.node", mesh_path + "armadilo.ele",
                            "armadilo_1");

  if (mesh_openbox < 0 || mesh_dragon1 < 0 || mesh_arm1 < 0) {
    std::cerr << "Failed to load one or more meshes from " << mesh_path << "\n";
    return 1;
  }

  const auto& inst_box     = mesh_manager.GetMeshInstance(mesh_openbox);
  const auto& inst_dragon1 = mesh_manager.GetMeshInstance(mesh_dragon1);
  const auto& inst_arm1    = mesh_manager.GetMeshInstance(mesh_arm1);

  std::cout << "Loaded meshes:\n";
  std::cout << "  Open box:     " << inst_box.num_nodes << " nodes, "
            << inst_box.num_elements << " elements\n";
  std::cout << "  Dragon 1:     " << inst_dragon1.num_nodes << " nodes, "
            << inst_dragon1.num_elements << " elements\n";
  std::cout << "  Armadilo 1:   " << inst_arm1.num_nodes << " nodes, "
            << inst_arm1.num_elements << " elements\n";

  // =========================================================================
  // Load pressure fields for collision detection
  // =========================================================================
  bool ok0 = mesh_manager.LoadScalarFieldFromNpz(mesh_openbox,
                                                mesh_path + "openbox.npz",
                                                "p_vertex");
  bool ok1 = mesh_manager.LoadScalarFieldFromNpz(mesh_dragon1,
                                                mesh_path + "dragon.npz",
                                                "p_vertex");
  bool ok3 = mesh_manager.LoadScalarFieldFromNpz(mesh_arm1,
                                                mesh_path + "armadilo.npz",
                                                "p_vertex");
  if (!ok0 || !ok1 || !ok3) {
    std::cerr << "Failed to load pressure fields from NPZ\n";
    return 1;
  }

  // =========================================================================
  // Compute mesh dimensions and place items above the box opening
  // =========================================================================
  {
    const Eigen::MatrixXd& nodes0 = mesh_manager.GetAllNodes();
    const BBox bb_box = ComputeBBox(nodes0, inst_box);
    const BBox bb_d1  = ComputeBBox(nodes0, inst_dragon1);
    const BBox bb_a1  = ComputeBBox(nodes0, inst_arm1);

    std::cout << "Initial bounding boxes:\n";
    PrintBBox("openbox", bb_box);
    PrintBBox("dragon_1", bb_d1);
    PrintBBox("armadilo_1", bb_a1);

    const Eigen::Vector3d box_center = bb_box.center();
    const double box_top_z           = bb_box.max(2);

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

    const double xy_offset = 0.05;  // small lateral offsets (stay near center)
    const double base_gap  = 0.05;  // initial clearance above box rim

    const double dz_d = 0.5 * bb_d1.size()(2);
    const double dz_a = 0.5 * bb_a1.size()(2);

    centerMeshAt(mesh_dragon1, inst_dragon1,
                 Eigen::Vector3d(box_center(0) + xy_offset,
                                 box_center(1) + xy_offset,
                                 box_top_z + base_gap + dz_d - 0.8));
    centerMeshAt(mesh_arm1, inst_arm1,
                 Eigen::Vector3d(box_center(0) - xy_offset,
                                 box_center(1) - xy_offset,
                                 box_top_z + base_gap + dz_a - 0.2));

    const Eigen::MatrixXd& nodes1 = mesh_manager.GetAllNodes();
    std::cout << "Placed items above box opening:\n";
    PrintBBox("openbox", ComputeBBox(nodes1, inst_box));
    PrintBBox("dragon_1", ComputeBBox(nodes1, inst_dragon1));
    PrintBBox("armadilo_1", ComputeBBox(nodes1, inst_arm1));
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

  // Fix all box nodes (static container)
  std::vector<int> fixed_node_indices;
  fixed_node_indices.reserve(inst_box.num_nodes);
  for (int i = 0; i < inst_box.num_nodes; ++i) {
    fixed_node_indices.push_back(inst_box.node_offset + i);
  }
  Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(i) = fixed_node_indices[i];
  }
  std::cout << "Fixed " << h_fixed_nodes.size() << " box nodes\n";
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
  // Newton solver
  // =========================================================================
  SyncedNewtonParams params = {1e-6, 0.0, 1e-6, 1e10, 3, 5, dt};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();

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
  broadphase.EnableSelfCollision(enable_self_collision);
  broadphase.BuildNeighborMap();

  narrowphase.Initialize(initial_nodes, elements, pressure, elementMeshIds);
  narrowphase.EnableSelfCollision(enable_self_collision);

  // Shared device node buffer for collision (column-major: [x... y... z...]).
  // Updated from the dynamics state each step via device-to-device copies.
  double* d_nodes_collision = nullptr;
  cudaMalloc(&d_nodes_collision, n_nodes * 3 * sizeof(double));
  cudaMemcpy(d_nodes_collision, gpu_t10_data.GetX12DevicePtr(),
             n_nodes * sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_nodes_collision + n_nodes, gpu_t10_data.GetY12DevicePtr(),
             n_nodes * sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_nodes_collision + 2 * n_nodes, gpu_t10_data.GetZ12DevicePtr(),
             n_nodes * sizeof(double), cudaMemcpyDeviceToDevice);

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
  addGravityForInstance(inst_arm1);

  double* d_f_gravity = nullptr;
  cudaMalloc(&d_f_gravity, n_nodes * 3 * sizeof(double));
  cudaMemcpy(d_f_gravity, h_f_gravity.data(), n_nodes * 3 * sizeof(double),
             cudaMemcpyHostToDevice);

  cublasHandle_t cublas_handle = nullptr;
  CheckCublas(cublasCreate(&cublas_handle), "cublasCreate");

  // =========================================================================
  // Simulation loop
  // =========================================================================
  std::cout << "\nStarting simulation (" << max_steps << " steps)\n\n";

  for (int step = 0; step < max_steps; ++step) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1) Update collision node buffer from the solver state (device->device)
    cudaMemcpy(d_nodes_collision, gpu_t10_data.GetX12DevicePtr(),
               n_nodes * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_nodes_collision + n_nodes, gpu_t10_data.GetY12DevicePtr(),
               n_nodes * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_nodes_collision + 2 * n_nodes, gpu_t10_data.GetZ12DevicePtr(),
               n_nodes * sizeof(double), cudaMemcpyDeviceToDevice);

    // 2) Collision detection
    broadphase.CreateAABB();
    broadphase.SortAABBs(0);
    broadphase.DetectCollisions(false);
    const int num_collision_pairs = broadphase.numCollisions;

    narrowphase.SetCollisionPairsDevice(broadphase.GetCollisionPairsDevicePtr(),
                                        num_collision_pairs);
    narrowphase.ComputeContactPatches();

    // 3) External forces: contact + gravity
    narrowphase.ComputeExternalForcesGPUDevice(solver.GetVelocityGuessDevicePtr(),
                                               contact_damping,
                                               contact_friction);

    cudaMemcpy(gpu_t10_data.GetExternalForceDevicePtr(), d_f_gravity,
               n_nodes * 3 * sizeof(double), cudaMemcpyDeviceToDevice);

    if (num_collision_pairs > 0) {
      const double alpha = 1.0;
      CheckCublas(cublasDaxpy(
                      cublas_handle, n_nodes * 3, &alpha,
                      narrowphase.GetExternalForcesDevicePtr(), 1,
                      gpu_t10_data.GetExternalForceDevicePtr(), 1),
                  "cublasDaxpy(contact + gravity)");
    }

    // 4) Solve one step
    solver.Solve();

    // 5) Export (host copies only when writing output)
    if (export_interval > 0 && step % export_interval == 0) {
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
  cudaFree(d_f_gravity);
  cudaFree(d_nodes_collision);

  std::cout << "Done.\n";
  return 0;
}
