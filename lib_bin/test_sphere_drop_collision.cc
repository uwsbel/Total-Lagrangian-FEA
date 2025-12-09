/**
 * Sphere Drop Collision Simulation
 *
 * Author: Json Zhou
 * Email:  zzhou292@wisc.edu
 *
 * This simulation demonstrates a sphere dropped onto another fixed sphere
 * using the Newton solver combined with the collision detection system.
 *
 * Setup:
 * - Two spheres with radius ~0.15
 * - Bottom sphere: fixed in place (all nodes constrained)
 * - Top sphere: initialized slightly above, subject to gravity
 *
 * Each time step:
 * 1. Run collision detection (broadphase + narrowphase)
 * 2. Compute contact forces (f_ext from contact patches)
 * 3. Add gravity to f_ext
 * 4. Run one Newton step
 * 5. Export VTK for visualization
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../lib_src/collision/Broadphase.cuh"
#include "../lib_src/collision/Narrowphase.cuh"
#include "../lib_src/elements/FEAT10Data.cuh"
#include "../lib_src/solvers/SyncedNewton.cuh"
#include "../lib_utils/cpu_utils.h"
#include "../lib_utils/mesh_manager.h"
#include "../lib_utils/quadrature_utils.h"
#include "../lib_utils/visualization_utils.h"

// Material properties
const double E    = 1e7;     // Young's modulus (softer for visible deformation)
const double nu   = 0.3;     // Poisson's ratio
const double rho0 = 1000.0;  // Density (kg/m^3)

// Simulation parameters
const double gravity = -9.81;  // Gravity acceleration (m/s^2)
const double dt      = 5e-4;   // Time step (s) - smaller for stability
const int num_steps  = 6000;   // Number of simulation steps
const double sphere_gap =
    0.02;  // Initial gap between spheres (m) - start closer

using ANCFCPUUtils::VisualizationUtils;

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "Sphere Drop Collision Simulation" << std::endl;
  std::cout << "========================================" << std::endl;

  // Create output directory
  std::filesystem::create_directories("output/sphere_drop");

  // =========================================================================
  // Load meshes using MeshManager
  // =========================================================================
  ANCFCPUUtils::MeshManager mesh_manager;

  // Load bottom sphere (will be fixed)
  int mesh_bottom =
      mesh_manager.LoadMesh("data/meshes/T10/sphere.1.node",
                            "data/meshes/T10/sphere.1.ele", "sphere_bottom");
  if (mesh_bottom < 0) {
    std::cerr << "Failed to load bottom sphere mesh" << std::endl;
    return 1;
  }

  // Load top sphere (will fall)
  int mesh_top =
      mesh_manager.LoadMesh("data/meshes/T10/sphere.1.node",
                            "data/meshes/T10/sphere.1.ele", "sphere_top");
  if (mesh_top < 0) {
    std::cerr << "Failed to load top sphere mesh" << std::endl;
    return 1;
  }

  // Get mesh info
  const auto& instance_bottom = mesh_manager.GetMeshInstance(mesh_bottom);
  const auto& instance_top    = mesh_manager.GetMeshInstance(mesh_top);

  std::cout << "Loaded meshes:" << std::endl;
  std::cout << "  Bottom sphere: " << instance_bottom.num_nodes << " nodes, "
            << instance_bottom.num_elements << " elements" << std::endl;
  std::cout << "  Top sphere: " << instance_top.num_nodes << " nodes, "
            << instance_top.num_elements << " elements" << std::endl;

  // Position the top sphere above the bottom sphere
  // Sphere radius is ~0.15, so translate top sphere up by 2*radius + gap
  double translation_z = 2.0 * 0.15 + sphere_gap;
  mesh_manager.TranslateMesh(mesh_top, 0.0, 0.0, translation_z);

  std::cout << "Translated top sphere by z = " << translation_z << std::endl;

  // Get unified mesh data
  const Eigen::MatrixXd& initial_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& elements      = mesh_manager.GetAllElements();

  int n_nodes = mesh_manager.GetTotalNodes();
  int n_elems = mesh_manager.GetTotalElements();

  std::cout << "Total nodes: " << n_nodes << std::endl;
  std::cout << "Total elements: " << n_elems << std::endl;

  // =========================================================================
  // Load pressure fields for collision detection
  // =========================================================================
  bool ok0 = mesh_manager.LoadScalarFieldFromNpz(
      mesh_bottom, "data/meshes/T10/sphere.1.uncompressed.npz", "p_vertex");
  bool ok1 = mesh_manager.LoadScalarFieldFromNpz(
      mesh_top, "data/meshes/T10/sphere.1.uncompressed.npz", "p_vertex");

  if (!ok0 || !ok1) {
    std::cerr << "Failed to load pressure fields from NPZ" << std::endl;
    return 1;
  }

  const Eigen::VectorXd& pressure = mesh_manager.GetAllScalarFields();
  std::cout << "Loaded pressure field with " << pressure.size() << " values"
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
  // Set fixed nodes (only bottom half of bottom sphere)
  // =========================================================================
  // Compute center of bottom sphere
  double center_z_bottom = 0.0;
  for (int i = 0; i < instance_bottom.num_nodes; ++i) {
    int global_idx = instance_bottom.node_offset + i;
    center_z_bottom += initial_nodes(global_idx, 2);
  }
  center_z_bottom /= instance_bottom.num_nodes;

  // Fix only nodes below the center (bottom half)
  std::vector<int> fixed_node_indices;
  for (int i = 0; i < instance_bottom.num_nodes; ++i) {
    int global_idx = instance_bottom.node_offset + i;
    if (initial_nodes(global_idx, 2) < center_z_bottom) {
      fixed_node_indices.push_back(global_idx);
    }
  }

  Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(i) = fixed_node_indices[i];
  }

  std::cout << "Fixed " << h_fixed_nodes.size()
            << " nodes (bottom half of bottom sphere)" << std::endl;
  std::cout << "Bottom sphere center z: " << center_z_bottom << std::endl;

  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  // =========================================================================
  // Setup GPU element data
  // =========================================================================
  const Eigen::VectorXd& tet5pt_x       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights = Quadrature::tet5pt_weights;

  gpu_t10_data.Setup(rho0, nu, E, 0.0, 0.0,  // Material + damping
                     tet5pt_x, tet5pt_y, tet5pt_z, tet5pt_weights, h_x12, h_y12,
                     h_z12, elements);

  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();
  gpu_t10_data.ConvertToCSRMass();
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertTOCSRConstraintJac();

  std::cout << "GPU element data initialized" << std::endl;

  // =========================================================================
  // Initialize Newton solver
  // =========================================================================
  SyncedNewtonParams params = {1e-4, 1e-8, 1e12, 3, 5, dt};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();

  std::cout << "Newton solver initialized" << std::endl;

  // =========================================================================
  // Initialize collision detection
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

  std::cout << "Collision detection initialized" << std::endl;

  // =========================================================================
  // Simulation loop
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
    // 2. Run collision detection
    // ---------------------------------------------------------------------
    // Destroy previous GPU allocations before re-initializing
    broadphase.Destroy();
    broadphase.Initialize(current_nodes, elements);
    broadphase.CreateAABB();
    broadphase.BuildNeighborMap();
    broadphase.SortAABBs(0);
    broadphase.DetectCollisions();

    int num_collision_pairs = broadphase.numCollisions;

    // Convert to pair vector
    std::vector<std::pair<int, int>> collisionPairs;
    for (const auto& cp : broadphase.h_collisionPairs) {
      collisionPairs.emplace_back(cp.idA, cp.idB);
    }

    // Run narrowphase
    // Destroy previous GPU allocations before re-initializing
    narrowphase.Destroy();
    narrowphase.Initialize(current_nodes, elements, pressure, elementMeshIds);
    narrowphase.SetCollisionPairs(collisionPairs);
    narrowphase.ComputeContactPatches();

    // Get patch count (without transferring all patch data to CPU)
    // We still need RetrieveResults for visualization export
    narrowphase.RetrieveResults();
    int num_patches = narrowphase.numPatches;

    // ---------------------------------------------------------------------
    // 3. Compute external forces (contact + gravity)
    // ---------------------------------------------------------------------
    Eigen::VectorXd h_f_ext(n_nodes * 3);
    h_f_ext.setZero();

    // Add contact forces from collision patches (GPU version)
    Eigen::VectorXd contact_forces = narrowphase.ComputeExternalForcesGPU();
    if (contact_forces.size() == h_f_ext.size()) {
      h_f_ext += contact_forces;
    }

    // Add gravity (only to top sphere nodes - bottom is fixed anyway)
    // F = m * g, distributed per node
    // For simplicity, assume uniform mass distribution
    double total_mass    = rho0 * (4.0 / 3.0 * M_PI * 0.15 * 0.15 * 0.15);
    double mass_per_node = total_mass / instance_top.num_nodes;
    double gravity_force_per_node = mass_per_node * gravity;

    for (int i = 0; i < instance_top.num_nodes; ++i) {
      int global_idx = instance_top.node_offset + i;
      h_f_ext(3 * global_idx + 2) += gravity_force_per_node;  // z-direction
    }

    // Set external force
    gpu_t10_data.SetExternalForce(h_f_ext);

    // ---------------------------------------------------------------------
    // 4. Run one Newton step
    // ---------------------------------------------------------------------
    solver.Solve();

    // ---------------------------------------------------------------------
    // 5. Compute displacement for visualization
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
    // 6. Export VTK every 5 steps
    // ---------------------------------------------------------------------
    if (step % 5 == 0) {
      std::ostringstream filename;
      filename << "output/sphere_drop/mesh_" << std::setfill('0')
               << std::setw(4) << step << ".vtu";
      VisualizationUtils::ExportMeshWithDisplacement(
          current_nodes, elements, displacement, filename.str());

      // Always export contact patches (even if empty) for ParaView time series
      std::ostringstream patch_filename;
      patch_filename << "output/sphere_drop/patches_" << std::setfill('0')
                     << std::setw(4) << step << ".vtp";
      std::vector<ContactPatch> patches = narrowphase.GetValidPatches();
      ANCFCPUUtils::VisualizationUtils::ExportContactPatchesToVTP(
          patches, patch_filename.str());
    }

    // ---------------------------------------------------------------------
    // 7. Print progress
    // ---------------------------------------------------------------------
    // Compute center of mass of top sphere
    double com_z = 0.0;
    for (int i = 0; i < instance_top.num_nodes; ++i) {
      int global_idx = instance_top.node_offset + i;
      com_z += z12_current(global_idx);
    }
    com_z /= instance_top.num_nodes;

    double contact_force_mag = contact_forces.norm();

    if (step % 20 == 0) {
      double gravity_force_total =
          std::abs(gravity_force_per_node * instance_top.num_nodes);
      std::cout << "Step " << std::setw(4) << step << ": "
                << "pairs=" << std::setw(5) << num_collision_pairs << ", "
                << "patches=" << std::setw(4) << num_patches << ", "
                << "top_z=" << std::fixed << std::setprecision(5) << com_z
                << ", |f_g|=" << std::scientific << std::setprecision(2)
                << gravity_force_total << ", |f_c|=" << contact_force_mag
                << std::endl;
    }
  }

  // =========================================================================
  // Cleanup
  // =========================================================================
  gpu_t10_data.Destroy();

  std::cout << "\n========================================" << std::endl;
  std::cout << "Simulation complete!" << std::endl;
  std::cout << "Output files in: output/sphere_drop/" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
