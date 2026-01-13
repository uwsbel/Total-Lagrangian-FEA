/**
 * Bubble Gripper Bunny + Floor Simulation
 * Author: Json Zhou (zzhou292@wisc.edu)
 *
 * Two rigid bubble grippers clamp a scaled bunny mesh; a deformable floor slab
 * is placed beneath the bunny with its bottom layer vertices fixed. Contact
 * forces are computed via broadphase + narrowphase collision and fed into the
 * solver.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "../../lib_src/collision/HydroelasticBroadphase.cuh"
#include "../../lib_src/collision/HydroelasticNarrowphase.cuh"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/visualization_utils.h"

// Material properties
const double E_val =
    1e6;  // Young's modulus (Pa) - softer for larger visible deformation
const double nu   = 0.4;     // Poisson's ratio
const double rho0 = 1000.0;  // Density (kg/m^3)

// Simulation parameters
const double gravity = -9.81;  // Gravity acceleration (m/s^2)
const double dt      = 5e-4;   // Time step (s)
const int num_steps  = 3000;  // Number of simulation steps (3x longer duration)
const double grip_speed =
    0.00002;  // Gripper closing speed per step (m, slower again)

// Contact parameters (softer for stability)
const double contact_damping  = 0.0;  // no Drake-style damping amplification
const double contact_friction = 0.7;  // moderate friction

using ANCFCPUUtils::rotationX;
using ANCFCPUUtils::translation;
using ANCFCPUUtils::uniformScale;
using ANCFCPUUtils::VisualizationUtils;

int main(int argc, char** argv) {
  std::cout << "========================================" << std::endl;
  std::cout << "Bubble Gripper Bunny + Floor Simulation" << std::endl;
  std::cout << "========================================" << std::endl;

  bool enable_self_collision = false;
  if (argc > 1) {
    enable_self_collision = (std::atoi(argv[1]) != 0);
  }
  std::cout << "Enable self collision: " << (enable_self_collision ? 1 : 0)
            << std::endl;

  // Create output directory
  std::filesystem::create_directories("output/bubble_gripper_bunny_floor");

  // =========================================================================
  // Load meshes using MeshManager
  // =========================================================================
  ANCFCPUUtils::MeshManager mesh_manager;

  // Base path for bubble gripper meshes
  std::string mesh_path = "data/meshes/T10/bubble_gripper_bunny/";

  // Load bubble gripper 1 (left side, will move right)
  int mesh_gripper1 = mesh_manager.LoadMesh(
      mesh_path + "bubble.1.node", mesh_path + "bubble.1.ele", "gripper_left");
  if (mesh_gripper1 < 0) {
    std::cerr << "Failed to load left gripper mesh" << std::endl;
    return 1;
  }

  // Load bubble gripper 2 (right side, mirrored, will move left)
  int mesh_gripper2 = mesh_manager.LoadMesh(
      mesh_path + "bubble_mirror_xy.1.node",
      mesh_path + "bubble_mirror_xy.1.ele", "gripper_right");
  if (mesh_gripper2 < 0) {
    std::cerr << "Failed to load right gripper mesh" << std::endl;
    return 1;
  }

  // Load scaled bunny (center)
  int mesh_bunny = mesh_manager.LoadMesh(mesh_path + "bunny_3k.1.node",
                                         mesh_path + "bunny_3k.1.ele", "bunny");
  if (mesh_bunny < 0) {
    std::cerr << "Failed to load bunny mesh" << std::endl;
    return 1;
  }

  // The `bunny_3k.1` mesh is in a much larger unit scale than the grippers and
  // floor. Scale it down to meters-ish to avoid huge gravity forces and NaNs.
  const double bunny_scale = 1e-4;
  mesh_manager.TransformMesh(mesh_bunny, uniformScale(bunny_scale));

  // Load floor slab (table)
  int mesh_floor =
      mesh_manager.LoadMesh(mesh_path + "1_1_01_floor.1.node",
                            mesh_path + "1_1_01_floor.1.ele", "floor");
  if (mesh_floor < 0) {
    std::cerr << "Failed to load floor mesh" << std::endl;
    return 1;
  }

  // Get mesh instances
  const auto& inst_gripper1 = mesh_manager.GetMeshInstance(mesh_gripper1);
  const auto& inst_gripper2 = mesh_manager.GetMeshInstance(mesh_gripper2);
  const auto& inst_bunny    = mesh_manager.GetMeshInstance(mesh_bunny);
  const auto& inst_floor    = mesh_manager.GetMeshInstance(mesh_floor);

  std::cout << "Loaded meshes:" << std::endl;
  std::cout << "  Left gripper:  " << inst_gripper1.num_nodes << " nodes, "
            << inst_gripper1.num_elements << " elements" << std::endl;
  std::cout << "  Right gripper: " << inst_gripper2.num_nodes << " nodes, "
            << inst_gripper2.num_elements << " elements" << std::endl;
  std::cout << "  Bunny:         " << inst_bunny.num_nodes << " nodes, "
            << inst_bunny.num_elements << " elements" << std::endl;
  std::cout << "  Floor:         " << inst_floor.num_nodes << " nodes, "
            << inst_floor.num_elements << " elements" << std::endl;

  // =========================================================================
  // Load pressure fields for collision detection
  // =========================================================================
  bool ok1 = mesh_manager.LoadScalarFieldFromNpz(
      mesh_gripper1, mesh_path + "bubble.npz", "p_vertex");
  bool ok2 = mesh_manager.LoadScalarFieldFromNpz(
      mesh_gripper2, mesh_path + "bubble_mirror_xy.npz", "p_vertex");
  bool ok3 = mesh_manager.LoadScalarFieldFromNpz(
      mesh_bunny, mesh_path + "bunny_3k.1.npz", "p_vertex");
  bool ok4 = mesh_manager.LoadScalarFieldFromNpz(
      mesh_floor, mesh_path + "1_1_01_floor.1.npz", "p_vertex");

  if (!ok1 || !ok2 || !ok3 || !ok4) {
    std::cerr << "Failed to load pressure fields from NPZ" << std::endl;
    return 1;
  }

  std::cout << "Loaded pressure fields for collision detection" << std::endl;

  // =========================================================================
  // Compute bunny bounding box and center
  // =========================================================================
  const Eigen::MatrixXd& nodes_before = mesh_manager.GetAllNodes();

  double bunny_min_x = 1e10, bunny_max_x = -1e10;
  double bunny_min_y = 1e10, bunny_max_y = -1e10;
  double bunny_min_z    = 1e10;
  double bunny_center_x = 0, bunny_center_y = 0, bunny_center_z = 0;
  for (int i = 0; i < inst_bunny.num_nodes; ++i) {
    int idx = inst_bunny.node_offset + i;
    bunny_center_x += nodes_before(idx, 0);
    bunny_center_y += nodes_before(idx, 1);
    bunny_center_z += nodes_before(idx, 2);
    bunny_min_x = std::min(bunny_min_x, nodes_before(idx, 0));
    bunny_max_x = std::max(bunny_max_x, nodes_before(idx, 0));
    bunny_min_y = std::min(bunny_min_y, nodes_before(idx, 1));
    bunny_max_y = std::max(bunny_max_y, nodes_before(idx, 1));
    bunny_min_z = std::min(bunny_min_z, nodes_before(idx, 2));
  }
  bunny_center_x /= inst_bunny.num_nodes;
  bunny_center_y /= inst_bunny.num_nodes;
  bunny_center_z /= inst_bunny.num_nodes;

  std::cout << "Bunny center: (" << bunny_center_x << ", " << bunny_center_y
            << ", " << bunny_center_z << ")" << std::endl;
  std::cout << "Bunny x range: [" << bunny_min_x << ", " << bunny_max_x << "]"
            << std::endl;

  // =========================================================================
  // Transform grippers: rotate and position
  // =========================================================================
  // (original bubble geometry: flat base at z=0, dome extends into -z)
  // Here we rotate around X so that the domes face along ±y and position
  // one gripper below the bunny (dome facing +y) and one above (dome facing
  // -y), so the spherical faces look at each other along the y axis.

  double gap        = 0.000;  // Gap between gripper dome and bunny (closer)
  double dome_depth = 0.037;  // Approximate dome depth
  double extra_bottom_offset =
      0.00;  // Move bottom gripper slightly further in -y
  double extra_top_offset = 0.015;  // Move top gripper further toward -y
  double x_shift          = 0.02;   // Shift both grippers toward -x

  // Bottom gripper: rotate +90° around X (dome faces +y)
  // Center so dome tip is at bunny_min_y - gap, then shift slightly further -y
  Eigen::Matrix4d T1 =
      translation(bunny_center_x - x_shift,
                  bunny_min_y - gap - dome_depth - extra_bottom_offset,
                  bunny_center_z) *
      rotationX(M_PI / 2.0);
  mesh_manager.TransformMesh(mesh_gripper1, T1);

  // Top gripper: rotate +90° around X (dome faces -y for mirrored geometry)
  // Center so dome tip is at bunny_max_y + gap, then shift slightly toward -y
  Eigen::Matrix4d T2 =
      translation(bunny_center_x - x_shift,
                  bunny_max_y + gap + dome_depth - extra_top_offset,
                  bunny_center_z) *
      rotationX(M_PI / 2.0);
  mesh_manager.TransformMesh(mesh_gripper2, T2);

  std::cout << "Transformed grippers (rotated and positioned)" << std::endl;

  // =========================================================================
  // Place floor slab beneath the bunny
  // =========================================================================
  {
    const Eigen::MatrixXd& nodes_pre_floor = mesh_manager.GetAllNodes();
    double floor_min_x = 1e10, floor_max_x = -1e10;
    double floor_min_y = 1e10, floor_max_y = -1e10;
    double floor_min_z = 1e10, floor_max_z = -1e10;

    for (int i = 0; i < inst_floor.num_nodes; ++i) {
      int idx     = inst_floor.node_offset + i;
      floor_min_x = std::min(floor_min_x, nodes_pre_floor(idx, 0));
      floor_max_x = std::max(floor_max_x, nodes_pre_floor(idx, 0));
      floor_min_y = std::min(floor_min_y, nodes_pre_floor(idx, 1));
      floor_max_y = std::max(floor_max_y, nodes_pre_floor(idx, 1));
      floor_min_z = std::min(floor_min_z, nodes_pre_floor(idx, 2));
      floor_max_z = std::max(floor_max_z, nodes_pre_floor(idx, 2));
    }

    double floor_center_x = 0.5 * (floor_min_x + floor_max_x);
    double floor_center_y = 0.5 * (floor_min_y + floor_max_y);
    double floor_top_z    = floor_max_z;

    const double floor_gap = 0.003;
    double target_center_x = bunny_center_x;
    double target_center_y = bunny_center_y;
    double target_top_z    = bunny_min_z - floor_gap;

    double dx = target_center_x - floor_center_x;
    double dy = target_center_y - floor_center_y;
    double dz = target_top_z - floor_top_z;

    mesh_manager.TranslateMesh(mesh_floor, dx, dy, dz);
  }

  // =========================================================================
  // Get unified mesh data after transformations
  // =========================================================================
  const Eigen::MatrixXd& initial_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& elements      = mesh_manager.GetAllElements();

  int n_nodes = mesh_manager.GetTotalNodes();
  int n_elems = mesh_manager.GetTotalElements();

  std::cout << "Total nodes: " << n_nodes << std::endl;
  std::cout << "Total elements: " << n_elems << std::endl;

  // Get unified pressure field
  const Eigen::VectorXd& pressure_raw = mesh_manager.GetAllScalarFields();
  std::cout << "Pressure field size: " << pressure_raw.size() << std::endl;

  Eigen::VectorXd pressure = pressure_raw;
  double p_min             = pressure.minCoeff();
  double p_max             = pressure.maxCoeff();
  std::cout << "Pressure range: [" << p_min << ", " << p_max << "]"
            << std::endl;

  // =========================================================================
  // Initialize GPU element data
  // =========================================================================
  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  // Extract coordinate vectors
  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = initial_nodes(i, 0);
    h_y12(i) = initial_nodes(i, 1);
    h_z12(i) = initial_nodes(i, 2);
  }

  // =========================================================================
  // Fix ALL gripper nodes (rigid grippers) and the floor bottom layer nodes
  // =========================================================================
  std::vector<int> fixed_node_indices;
  std::vector<int> floor_bottom_indices;

  // Fix all nodes of gripper 1
  for (int i = 0; i < inst_gripper1.num_nodes; ++i) {
    fixed_node_indices.push_back(inst_gripper1.node_offset + i);
  }

  // Fix all nodes of gripper 2
  for (int i = 0; i < inst_gripper2.num_nodes; ++i) {
    fixed_node_indices.push_back(inst_gripper2.node_offset + i);
  }

  // Fix floor bottom layer: all nodes with z near the floor minimum
  double floor_z_min = 1e10;
  for (int i = 0; i < inst_floor.num_nodes; ++i) {
    int idx     = inst_floor.node_offset + i;
    floor_z_min = std::min(floor_z_min, initial_nodes(idx, 2));
  }
  const double floor_z_fix_threshold = floor_z_min + 1e-6;
  for (int i = 0; i < inst_floor.num_nodes; ++i) {
    int idx  = inst_floor.node_offset + i;
    double z = initial_nodes(idx, 2);
    if (z <= floor_z_fix_threshold) {
      fixed_node_indices.push_back(idx);
      floor_bottom_indices.push_back(idx);
    }
  }

  Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(i) = fixed_node_indices[i];
  }

  std::cout << "Fixed " << h_fixed_nodes.size() << " nodes (all grippers + "
            << floor_bottom_indices.size() << " floor bottom nodes)"
            << std::endl;

  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  // =========================================================================
  // Setup GPU element data
  // =========================================================================
  const Eigen::VectorXd& tet5pt_x       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights = Quadrature::tet5pt_weights;

  // Use soft material for visible deformation
  double eta_damp    = 1e3;  // Viscous damping
  double lambda_damp = 1e3;  // Volume damping
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

  std::cout << "GPU element data initialized" << std::endl;

  // Retrieve lumped mass per node from the scalar mass matrix (row-sum).
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
      std::cerr << "Warning: unexpected mass CSR offsets size "
                << offsets.size() << " (expected " << (n_nodes + 1)
                << "); gravity will be applied with unit mass." << std::endl;
      lumped_mass.setOnes();
    }
  }

  // =========================================================================
  // Initialize Newton solver
  // Use a smaller rho for better numerical robustness in this contact-heavy
  // scenario, while still strongly enforcing constraints.
  // =========================================================================
  SyncedNewtonParams params = {1e-6, 0.0, 1e-6, 1e9, 3, 5, dt};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();

  std::cout << "Newton solver initialized" << std::endl;

  // =========================================================================
  // Initialize collision detection (build-once topology)
  // =========================================================================
  Broadphase broadphase;
  Narrowphase narrowphase;

  // Build element-to-mesh ID mapping for narrowphase
  Eigen::VectorXi elementMeshIds(n_elems);
  for (int i = 0; i < mesh_manager.GetNumMeshes(); ++i) {
    const auto& instance = mesh_manager.GetMeshInstance(i);
    for (int e = 0; e < instance.num_elements; ++e) {
      elementMeshIds(instance.element_offset + e) = i;
    }
  }

  broadphase.Initialize(mesh_manager);
  broadphase.EnableSelfCollision(enable_self_collision);
  broadphase.CreateAABB();
  broadphase.BuildNeighborMap();
  broadphase.SortAABBs(0);

  narrowphase.Initialize(initial_nodes, elements, pressure, elementMeshIds);
  narrowphase.EnableSelfCollision(enable_self_collision);

  std::cout << "Collision detection initialized" << std::endl;

  // Store initial gripper positions for prescribed motion (along y-axis)
  Eigen::VectorXd gripper1_init_y(inst_gripper1.num_nodes);
  Eigen::VectorXd gripper2_init_y(inst_gripper2.num_nodes);
  for (int i = 0; i < inst_gripper1.num_nodes; ++i) {
    gripper1_init_y(i) = initial_nodes(inst_gripper1.node_offset + i, 1);
  }
  for (int i = 0; i < inst_gripper2.num_nodes; ++i) {
    gripper2_init_y(i) = initial_nodes(inst_gripper2.node_offset + i, 1);
  }

  // =========================================================================
  // Simulation loop - bunny drops onto the floor while grippers move as before
  // =========================================================================
  std::cout << "\n========================================" << std::endl;
  std::cout << "Starting simulation (" << num_steps << " steps)" << std::endl;
  std::cout << "========================================\n" << std::endl;

  // Track displacement for visualization
  Eigen::VectorXd displacement(n_nodes * 3);
  displacement.setZero();

  for (int step = 0; step < num_steps; ++step) {
    // ---------------------------------------------------------------------
    // 1. Get current node positions
    // ---------------------------------------------------------------------
    Eigen::VectorXd x12_current, y12_current, z12_current;
    gpu_t10_data.RetrievePositionToCPU(x12_current, y12_current, z12_current);

    // Build current nodes matrix for collision detection
    Eigen::MatrixXd current_nodes(n_nodes, 3);
    for (int i = 0; i < n_nodes; ++i) {
      current_nodes(i, 0) = x12_current(i);
      current_nodes(i, 1) = y12_current(i);
      current_nodes(i, 2) = z12_current(i);
    }

    // ---------------------------------------------------------------------
    // 2. Run collision detection (reuse topology, update positions only)
    // ---------------------------------------------------------------------
    broadphase.UpdateNodes(current_nodes);
    broadphase.CreateAABB();
    broadphase.SortAABBs(0);
    broadphase.DetectCollisions(false);

    int num_collision_pairs = broadphase.numCollisions;

    // Run narrowphase with updated positions and current collision pairs
    narrowphase.UpdateNodes(current_nodes);
    narrowphase.SetCollisionPairsDevice(broadphase.GetCollisionPairsDevicePtr(),
                                        num_collision_pairs);
    narrowphase.ComputeContactPatches();
    narrowphase.RetrieveResults();
    int num_patches = narrowphase.numPatches;

    // ---------------------------------------------------------------------
    // 3. Compute external forces from contact + gravity
    // ---------------------------------------------------------------------
    Eigen::VectorXd h_f_ext(n_nodes * 3);
    h_f_ext.setZero();

    // Add contact forces from collision patches
    Eigen::VectorXd contact_forces = narrowphase.ComputeExternalForcesGPU(
        solver.GetVelocityGuessDevicePtr(), contact_damping, contact_friction);
    if (contact_forces.size() == h_f_ext.size()) {
      h_f_ext += contact_forces;
    }

    // Add gravity to bunny nodes only: f = m * g (z-direction)
    for (int i = 0; i < inst_bunny.num_nodes; ++i) {
      int idx = inst_bunny.node_offset + i;
      h_f_ext(3 * idx + 2) += lumped_mass(idx) * gravity;
    }

    gpu_t10_data.SetExternalForce(h_f_ext);

    // ---------------------------------------------------------------------
    // 4. Update gripper positions (prescribed motion)
    //    - Close until step 1100
    //    - Hold for 200 steps, then reopen back to original positions by final
    //    step
    // ---------------------------------------------------------------------
    double move_amount;
    const int close_steps = 1100;
    const int hold_steps  = 50;  // hold closed before reopening
    if (step <= close_steps) {
      // Closing phase: move inward linearly with step
      move_amount = grip_speed * step;
    } else if (step <= close_steps + hold_steps) {
      // Hold phase: keep at max closure
      move_amount = grip_speed * close_steps;
    } else {
      // Opening phase: move back to original positions by the final step
      int step_end = num_steps - 1;
      double t     = static_cast<double>(step - close_steps - hold_steps) /
                 static_cast<double>(step_end - close_steps - hold_steps);
      move_amount = (1.0 - t) * grip_speed * close_steps;
    }

    for (int i = 0; i < inst_gripper1.num_nodes; ++i) {
      int idx = inst_gripper1.node_offset + i;
      y12_current(idx) =
          gripper1_init_y(i) + move_amount;  // Move +y (from below)
    }
    for (int i = 0; i < inst_gripper2.num_nodes; ++i) {
      int idx = inst_gripper2.node_offset + i;
      y12_current(idx) =
          gripper2_init_y(i) - move_amount;  // Move -y (from above)
    }

    // Clamp floor bottom layer nodes to their initial positions
    for (int k = 0; k < static_cast<int>(floor_bottom_indices.size()); ++k) {
      int idx          = floor_bottom_indices[k];
      x12_current(idx) = initial_nodes(idx, 0);
      y12_current(idx) = initial_nodes(idx, 1);
      z12_current(idx) = initial_nodes(idx, 2);
    }

    // Update GPU with new positions and constraint targets
    gpu_t10_data.UpdatePositions(x12_current, y12_current, z12_current);
    gpu_t10_data.UpdateConstraintTargets(x12_current, y12_current, z12_current);

    // Recompute constraints for new gripper positions
    gpu_t10_data.CalcConstraintData();

    // ---------------------------------------------------------------------
    // 5. Run Newton solver
    // ---------------------------------------------------------------------
    solver.Solve();

    // ---------------------------------------------------------------------
    // 6. Retrieve results and compute displacement
    // ---------------------------------------------------------------------
    gpu_t10_data.RetrievePositionToCPU(x12_current, y12_current, z12_current);

    for (int i = 0; i < n_nodes; ++i) {
      displacement(3 * i)     = x12_current(i) - initial_nodes(i, 0);
      displacement(3 * i + 1) = y12_current(i) - initial_nodes(i, 1);
      displacement(3 * i + 2) = z12_current(i) - initial_nodes(i, 2);
    }

    // Update current_nodes for export
    for (int i = 0; i < n_nodes; ++i) {
      current_nodes(i, 0) = x12_current(i);
      current_nodes(i, 1) = y12_current(i);
      current_nodes(i, 2) = z12_current(i);
    }

    // ---------------------------------------------------------------------
    // 7. Export VTK every 10 steps
    // ---------------------------------------------------------------------
    if (step % 10 == 0) {
      std::ostringstream filename;
      filename << "output/bubble_gripper_bunny_floor/mesh_" << std::setfill('0')
               << std::setw(4) << step << ".vtu";
      VisualizationUtils::ExportMeshWithDisplacement(
          current_nodes, elements, displacement, filename.str());

      // Export contact patches
      std::ostringstream patch_filename;
      patch_filename << "output/bubble_gripper_bunny_floor/patches_"
                     << std::setfill('0') << std::setw(4) << step << ".vtp";
      std::vector<ContactPatch> patches = narrowphase.GetValidPatches();
      VisualizationUtils::ExportContactPatchesToVTP(patches,
                                                    patch_filename.str());
    }

    // ---------------------------------------------------------------------
    // 8. Print progress
    // ---------------------------------------------------------------------
    double max_disp          = displacement.cwiseAbs().maxCoeff();
    double contact_force_mag = contact_forces.norm();

    if (step % 20 == 0) {
      std::cout << "Step " << std::setw(4) << step << ": "
                << "pairs=" << std::setw(5) << num_collision_pairs << ", "
                << "patches=" << std::setw(4) << num_patches << ", "
                << "grip_move=" << std::fixed << std::setprecision(4)
                << move_amount << ", max_disp=" << std::scientific
                << std::setprecision(2) << max_disp
                << ", |f_c|=" << contact_force_mag << std::endl;
    }
  }

  // =========================================================================
  // Cleanup
  // =========================================================================
  gpu_t10_data.Destroy();

  std::cout << "\n========================================" << std::endl;
  std::cout << "Simulation complete!" << std::endl;
  std::cout << "Output files in: output/bubble_gripper_bunny_floor/"
            << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
