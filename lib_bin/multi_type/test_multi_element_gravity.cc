/**
 * Dragon on ANCF3243 Net Demo
 * Author: Json Zhou (zzhou292@wisc.edu)
 *
 * Drops a T10 dragon mesh onto an ANCF3243 beam net with four corners fixed.
 * Uses DEME collision system for contact between dragon and net.
 * Both bodies are subject to gravity.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/collision/DemeMeshCollisionSystem.h"
#include "../../lib_src/elements/ANCF3243Data.cuh"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/mesh_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/surface_trimesh_extract.h"
#include "../../lib_utils/visualization_utils.h"

// Material properties (using SolidMaterialProperties)
const SolidMaterialProperties mat_dragon = SolidMaterialProperties::SVK(
    1e7,   // E: Young's modulus (Pa)
    0.3,   // nu: Poisson's ratio
    80.0,  // rho0: Density (kg/m³)
    1e4,   // eta_damp
    1e4    // lambda_damp
);

const SolidMaterialProperties mat_net = SolidMaterialProperties::SVK(
    2e8,   // E: Young's modulus (Pa) - softened for stability
    0.33,  // nu: Poisson's ratio
    50.0,  // rho0: Density (kg/m³)
    5e5,   // eta_damp
    5e5    // lambda_damp
);

// Simulation parameters
const double gravity        = -9.81;  // Gravity acceleration (m/s^2)
const double dt             = 1e-4;   // Time step (s)
const int num_steps_default = 100000;
const int export_interval   = 50;

// Contact parameters
const double contact_friction = 0.6;

using ANCFCPUUtils::VisualizationUtils;

static void CheckCublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error (" << what << "): status=" << int(status) << "\n";
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  std::cout << "========================================\n";
  std::cout << "Dragon on ANCF3243 Net Demo\n";
  std::cout << "========================================\n";

  int max_steps = num_steps_default;
  if (argc > 1) {
    max_steps = std::atoi(argv[1]);
    if (max_steps <= 0) max_steps = num_steps_default;
  }
  std::cout << "Running for " << max_steps << " steps\n";

  std::filesystem::create_directories("output/dragon_on_net");

  // =========================================================================
  // Load T10 Dragon mesh
  // =========================================================================
  ANCFCPUUtils::MeshManager mesh_manager;
  const std::string dragon_path = "data/meshes/T10/item_drop/";
  
  const int mesh_dragon = mesh_manager.LoadMesh(
      dragon_path + "dragon.node", dragon_path + "dragon.ele", "dragon");
  
  if (mesh_dragon < 0) {
    std::cerr << "Failed to load dragon mesh from " << dragon_path << "\n";
    return 1;
  }
  
  const auto& inst_dragon = mesh_manager.GetMeshInstance(mesh_dragon);
  std::cout << "Dragon: " << inst_dragon.num_nodes << " nodes, " 
            << inst_dragon.num_elements << " elements\n";

  // =========================================================================
  // Load ANCF3243 Net mesh
  // =========================================================================
  const std::string net_path = "data/meshes/ANCF3243/net_pinned_nx20_ny20_L0.5.ancf3243mesh";
  
  ANCFCPUUtils::ANCF3243Mesh net_mesh;
  std::string err;
  if (!ANCFCPUUtils::ReadANCF3243MeshFromFile(net_path, net_mesh, &err)) {
    std::cerr << "Failed to load net mesh: " << err << "\n";
    return 1;
  }
  
  std::cout << "Net: " << net_mesh.n_nodes << " nodes, " 
            << net_mesh.n_elements << " elements\n";

  // =========================================================================
  // Position meshes: Net at z=0, Dragon above it
  // =========================================================================
  // Scale dragon to fit on net (net is 10x10 units by default)
  mesh_manager.TransformMesh(mesh_dragon, ANCFCPUUtils::uniformScale(4.0));
  
  // Get dragon bounding box and position it above net center
  const Eigen::MatrixXd& dragon_nodes_init = mesh_manager.GetAllNodes();
  double dz_min = 1e30, dz_max = -1e30;
  for (int i = 0; i < inst_dragon.num_nodes; ++i) {
    const int idx = inst_dragon.node_offset + i;
    dz_min = std::min(dz_min, dragon_nodes_init(idx, 2));
    dz_max = std::max(dz_max, dragon_nodes_init(idx, 2));
  }
  
  // Move dragon so its bottom is at z=0.2 (0.2 meter above net)
  double dragon_height = dz_max - dz_min;
  mesh_manager.TransformMesh(mesh_dragon, 
      ANCFCPUUtils::translation(5.0, 5.0, 0.2 - dz_min));

  const Eigen::MatrixXd& dragon_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& dragon_elements = mesh_manager.GetAllElements();

  std::cout << "Dragon positioned at center, height=" << dragon_height << "\n";

  // =========================================================================
  // Setup T10 Dragon GPU data
  // =========================================================================
  GPU_FEAT10_Data gpu_dragon(inst_dragon.num_elements, inst_dragon.num_nodes);
  gpu_dragon.Initialize();

  Eigen::VectorXd h_dragon_x(inst_dragon.num_nodes),
                  h_dragon_y(inst_dragon.num_nodes),
                  h_dragon_z(inst_dragon.num_nodes);
  for (int i = 0; i < inst_dragon.num_nodes; ++i) {
    const int idx = inst_dragon.node_offset + i;
    h_dragon_x(i) = dragon_nodes(idx, 0);
    h_dragon_y(i) = dragon_nodes(idx, 1);
    h_dragon_z(i) = dragon_nodes(idx, 2);
  }

  // Extract elements with local indexing
  Eigen::MatrixXi dragon_elems_local(inst_dragon.num_elements, 10);
  for (int e = 0; e < inst_dragon.num_elements; ++e) {
    int elem_idx = inst_dragon.element_offset + e;
    for (int n = 0; n < 10; ++n) {
      dragon_elems_local(e, n) = dragon_elements(elem_idx, n) - inst_dragon.node_offset;
    }
  }

  gpu_dragon.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                   Quadrature::tet5pt_z, Quadrature::tet5pt_weights,
                   h_dragon_x, h_dragon_y, h_dragon_z, dragon_elems_local);
  gpu_dragon.ApplyMaterial(mat_dragon);
  gpu_dragon.CalcDnDuPre();
  gpu_dragon.CalcMassMatrix();
  // Dragon has no fixed nodes - skip constraint setup

  const int dragon_n_constraints = 0;
  std::cout << "Dragon GPU data initialized, constraints=" 
            << dragon_n_constraints << "\n";

  // =========================================================================
  // Setup ANCF3243 Net GPU data
  // =========================================================================
  const double net_L = net_mesh.grid_L.value_or(0.5);
  const double net_W = 0.1;  // Beam width
  const double net_H = 0.1;  // Beam height

  GPU_ANCF3243_Data gpu_net(net_mesh.n_nodes, net_mesh.n_elements);
  gpu_net.Initialize();

  // Set external force to zero initially
  Eigen::VectorXd h_net_f_ext(4 * net_mesh.n_nodes * 3);
  h_net_f_ext.setZero();
  gpu_net.SetExternalForce(h_net_f_ext);

  gpu_net.Setup(net_L, net_W, net_H, 
                Quadrature::gauss_xi_m_6, Quadrature::gauss_xi_3,
                Quadrature::gauss_eta_2, Quadrature::gauss_zeta_2,
                Quadrature::weight_xi_m_6, Quadrature::weight_xi_3,
                Quadrature::weight_eta_2, Quadrature::weight_zeta_2, 
                net_mesh.x12, net_mesh.y12, net_mesh.z12, 
                net_mesh.element_connectivity);

  gpu_net.SetDensity(mat_net.rho0);
  gpu_net.SetDamping(mat_net.eta_damp, mat_net.lambda_damp);
  gpu_net.SetSVK(mat_net.E, mat_net.nu);

  // Build constraints: mesh constraints + four corner clamps
  const int net_n_coef = 4 * net_mesh.n_nodes;
  const int net_n_dofs = net_n_coef * 3;

  std::unique_ptr<ANCFCPUUtils::LinearConstraintBuilder> builder;
  if (net_mesh.constraints.Empty()) {
    builder = std::make_unique<ANCFCPUUtils::LinearConstraintBuilder>(net_n_dofs);
  } else {
    builder = std::make_unique<ANCFCPUUtils::LinearConstraintBuilder>(
        net_n_dofs, net_mesh.constraints);
  }

  // Find and clamp four corners
  auto FindNodesAtXY = [&](double x, double y, double tol) -> std::vector<int> {
    std::vector<int> nodes;
    for (int nid = 0; nid < net_mesh.n_nodes; ++nid) {
      const double xn = net_mesh.x12(4 * nid);
      const double yn = net_mesh.y12(4 * nid);
      if (std::abs(xn - x) <= tol && std::abs(yn - y) <= tol) {
        nodes.push_back(nid);
      }
    }
    return nodes;
  };

  // Get net bounds
  double xmin = 1e30, xmax = -1e30, ymin = 1e30, ymax = -1e30;
  for (int nid = 0; nid < net_mesh.n_nodes; ++nid) {
    const double x = net_mesh.x12(4 * nid);
    const double y = net_mesh.y12(4 * nid);
    xmin = std::min(xmin, x);
    xmax = std::max(xmax, x);
    ymin = std::min(ymin, y);
    ymax = std::max(ymax, y);
  }

  std::vector<std::pair<double, double>> corners = {
      {xmin, ymin}, {xmax, ymin}, {xmin, ymax}, {xmax, ymax}
  };

  const double tol = 1e-6;
  for (const auto& [cx, cy] : corners) {
    const std::vector<int> nodes = FindNodesAtXY(cx, cy, tol);
    std::cout << "Corner (" << cx << ", " << cy << "): " << nodes.size() << " nodes\n";
    for (int nid : nodes) {
      for (int slot = 0; slot < 4; ++slot) {
        const int coef = 4 * nid + slot;
        ANCFCPUUtils::AppendANCF3243FixedCoefficient(*builder, coef, 
            net_mesh.x12, net_mesh.y12, net_mesh.z12);
      }
    }
  }

  const ANCFCPUUtils::LinearConstraintCSR all_constraints = builder->ToCSR();
  gpu_net.SetLinearConstraintsCSR(all_constraints.offsets, all_constraints.columns,
                                  all_constraints.values, all_constraints.rhs);

  gpu_net.CalcDsDuPre();
  gpu_net.CalcMassMatrix();
  gpu_net.CalcConstraintData();
  gpu_net.CalcP();
  gpu_net.CalcInternalForce();

  std::cout << "Net GPU data initialized, constraints=" 
            << gpu_net.get_n_constraint() << "\n";

  // =========================================================================
  // Setup solvers
  // =========================================================================
  SyncedNewtonParams dragon_params = {1e-4, 0.0, 1e-4, 1e12, 3, 5, dt};
  auto dragon_solver = std::make_unique<SyncedNewtonSolver>(
      &gpu_dragon, dragon_n_constraints);
  dragon_solver->Setup();
  dragon_solver->SetParameters(&dragon_params);
  dragon_solver->AnalyzeHessianSparsity();

  SyncedNewtonParams net_params = {1e-4, 0.0, 1e-4, 1e12, 3, 5, dt};
  SyncedNewtonSolver net_solver(&gpu_net, gpu_net.get_n_constraint());
  net_solver.Setup();
  net_solver.SetParameters(&net_params);

  std::cout << "Solvers initialized.\n";

  // =========================================================================
  // Compute lumped mass for gravity
  // =========================================================================
  Eigen::VectorXd dragon_lumped_mass(inst_dragon.num_nodes);
  dragon_lumped_mass.setZero();
  {
    std::vector<int> offsets, columns;
    std::vector<double> values;
    gpu_dragon.RetrieveMassCSRToCPU(offsets, columns, values);
    if (static_cast<int>(offsets.size()) == inst_dragon.num_nodes + 1) {
      for (int i = 0; i < inst_dragon.num_nodes; ++i) {
        double sum = 0.0;
        for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
          sum += values[k];
        }
        dragon_lumped_mass(i) = sum;
      }
    } else {
      dragon_lumped_mass.setConstant(1.0);
    }
  }

  // Precompute gravity forces
  const int dragon_n_dofs = inst_dragon.num_nodes * 3;
  Eigen::VectorXd h_dragon_gravity = Eigen::VectorXd::Zero(dragon_n_dofs);
  for (int i = 0; i < inst_dragon.num_nodes; ++i) {
    h_dragon_gravity(3 * i + 2) = dragon_lumped_mass(i) * gravity;
  }

  double dragon_mass = dragon_lumped_mass.sum();
  std::cout << "Dragon mass: " << dragon_mass << " kg\n";
  std::cout << "Dragon gravity force (z): " << dragon_mass * gravity << " N\n";

  // Compute net lumped mass for gravity
  // ANCF3243 has 4 coefs per node: [r, dr/ds]
  // Gravity is applied to position coefficients (every 4th starting at 0)
  Eigen::VectorXd net_lumped_mass(net_mesh.n_nodes);
  net_lumped_mass.setZero();
  {
    std::vector<int> offsets, columns;
    std::vector<double> values;
    gpu_net.RetrieveMassCSRToCPU(offsets, columns, values);
    const int n_coef = 4 * net_mesh.n_nodes;
    if (static_cast<int>(offsets.size()) == n_coef + 1) {
      // Sum mass contributions for position coefficients
      for (int nid = 0; nid < net_mesh.n_nodes; ++nid) {
        const int pos_coef = 4 * nid;  // Position coefficient
        double sum = 0.0;
        for (int k = offsets[pos_coef]; k < offsets[pos_coef + 1]; ++k) {
          sum += values[k];
        }
        net_lumped_mass(nid) = sum;
      }
    } else {
      net_lumped_mass.setConstant(1.0);
    }
  }

  // Build net gravity force vector (use net_n_coef, net_n_dofs from above)
  Eigen::VectorXd h_net_gravity = Eigen::VectorXd::Zero(net_n_dofs);
  for (int nid = 0; nid < net_mesh.n_nodes; ++nid) {
    // Apply gravity to position coef (first of each 4)
    const int pos_coef = 4 * nid;
    h_net_gravity(pos_coef * 3 + 2) = net_lumped_mass(nid) * gravity;
  }

  double net_mass = net_lumped_mass.sum();
  std::cout << "Net mass: " << net_mass << " kg\n";
  std::cout << "Net gravity force (z): " << net_mass * gravity << " N\n";

  // Apply gravity to net
  gpu_net.SetExternalForce(h_net_gravity);

  // Allocate device buffers
  double* d_dragon_gravity = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_dragon_gravity, dragon_n_dofs * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_dragon_gravity, h_dragon_gravity.data(),
                          dragon_n_dofs * sizeof(double), cudaMemcpyHostToDevice));

  // Initialize velocity buffer for collision
  double* d_dragon_vel = dragon_solver->GetVelocityGuessDevicePtr();
  HANDLE_ERROR(cudaMemset(d_dragon_vel, 0, dragon_n_dofs * sizeof(double)));

  cublasHandle_t cublas_handle = nullptr;
  CheckCublas(cublasCreate(&cublas_handle), "cublasCreate");

  // =========================================================================
  // Setup DEME collision system
  // =========================================================================
  // Extract dragon surface mesh
  ANCFCPUUtils::SurfaceTriMesh dragon_surface = 
      ANCFCPUUtils::ExtractSurfaceTriMesh(dragon_nodes, dragon_elements, inst_dragon);

  std::cout << "Dragon surface: " << dragon_surface.vertices.size() << " verts, "
            << dragon_surface.triangles.size() << " tris\n";

  // Save dragon surface node IDs before remapping (for force mapping later)
  std::vector<int> dragon_surf_node_ids = dragon_surface.global_node_ids;

  // Extract net surface mesh
  Eigen::VectorXd net_x12_curr, net_y12_curr, net_z12_curr;
  gpu_net.RetrievePositionToCPU(net_x12_curr, net_y12_curr, net_z12_curr);
  ANCFCPUUtils::SurfaceTriMesh net_surface = 
      ANCFCPUUtils::ExtractSurfaceTriMeshFromANCF3243(
          net_x12_curr, net_y12_curr, net_z12_curr,
          net_mesh.element_connectivity, net_W, net_H);

  std::cout << "Net surface: " << net_surface.vertices.size() << " verts, "
            << net_surface.triangles.size() << " tris\n";

  // Save net surface ancf_node_ids before remapping (for force mapping later)
  std::vector<int> net_surf_ancf_node_ids = net_surface.ancf_node_ids;

  // Combined collision node count
  const int dragon_surf_verts = static_cast<int>(dragon_surface.vertices.size());
  const int net_surf_verts = static_cast<int>(net_surface.vertices.size());
  const int total_coll_nodes = dragon_surf_verts + net_surf_verts;

  // Remap dragon surface global_node_ids to sequential indices [0, dragon_surf_verts)
  // This ensures collision buffer indices are within bounds
  for (int i = 0; i < dragon_surf_verts; ++i) {
    dragon_surface.global_node_ids[i] = i;
  }

  // Remap net surface global_node_ids to continue after dragon vertices
  for (int i = 0; i < net_surf_verts; ++i) {
    net_surface.global_node_ids[i] = dragon_surf_verts + i;
  }

  // Create collision bodies
  std::vector<DemeMeshCollisionBody> bodies;
  {
    DemeMeshCollisionBody body;
    body.surface = std::move(dragon_surface);
    body.family = 0;
    body.split_into_patches = true;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface = std::move(net_surface);
    body.family = 1;
    body.split_into_patches = false;
    body.skip_self_contact_forces = true;  // Disable net self-collision forces
    bodies.push_back(std::move(body));
  }

  const double contact_stiffness = 1.0e8;  // Contact stiffness (Young's modulus E)
  auto collision_system = std::make_unique<DemeMeshCollisionSystem>(
      std::move(bodies), contact_friction, contact_stiffness, false);

  // Combined collision node buffer
  double* d_collision_nodes = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_collision_nodes, total_coll_nodes * 3 * sizeof(double)));

  // Host buffer for net surface vertices
  std::vector<double> h_net_surf_xyz(net_surf_verts * 3);

  // Host buffer for combined collision nodes (planar layout: [x0..xN, y0..yN, z0..zN])
  std::vector<double> h_coll_nodes(total_coll_nodes * 3);

  // Helper to update collision node buffer
  auto update_collision_nodes = [&]() {
    // Dragon: use saved node IDs to get positions
    Eigen::VectorXd dx, dy, dz;
    gpu_dragon.RetrievePositionToCPU(dx, dy, dz);
    for (int i = 0; i < dragon_surf_verts; ++i) {
      int node_id = dragon_surf_node_ids[i];
      h_coll_nodes[i] = dx(node_id);                           // x section
      h_coll_nodes[total_coll_nodes + i] = dy(node_id);        // y section
      h_coll_nodes[2 * total_coll_nodes + i] = dz(node_id);    // z section
    }

    // Net: compute surface vertex positions from ANCF coefficients
    Eigen::VectorXd nx, ny, nz;
    gpu_net.RetrievePositionToCPU(nx, ny, nz);
    ANCFCPUUtils::SurfaceTriMesh net_surf_updated = 
        ANCFCPUUtils::ExtractSurfaceTriMeshFromANCF3243(
            nx, ny, nz, net_mesh.element_connectivity, net_W, net_H);
    
    for (int i = 0; i < net_surf_verts; ++i) {
      int idx = dragon_surf_verts + i;  // Offset after dragon verts
      h_coll_nodes[idx] = net_surf_updated.vertices[i].x();                        // x section
      h_coll_nodes[total_coll_nodes + idx] = net_surf_updated.vertices[i].y();     // y section
      h_coll_nodes[2 * total_coll_nodes + idx] = net_surf_updated.vertices[i].z(); // z section
    }
    
    HANDLE_ERROR(cudaMemcpy(d_collision_nodes, h_coll_nodes.data(),
                            total_coll_nodes * 3 * sizeof(double), cudaMemcpyHostToDevice));
  };

  // Initial update
  update_collision_nodes();
  collision_system->BindNodesDevicePtr(d_collision_nodes, total_coll_nodes);

  // Host buffer for collision forces
  std::vector<double> h_coll_forces(total_coll_nodes * 3);

  // Helper to distribute net collision forces to ANCF position DOFs
  auto apply_net_collision_forces = [&](Eigen::VectorXd& net_ext_force) {
    // Accumulate forces from net surface vertices to ANCF nodes
    for (int i = 0; i < net_surf_verts; ++i) {
      const int ancf_node = net_surf_ancf_node_ids[i];
      const int pos_coef = 4 * ancf_node;  // Position coefficient index
      
      // Force at surface vertex i (offset by dragon verts in combined buffer)
      const double fx = h_coll_forces[3 * (dragon_surf_verts + i) + 0];
      const double fy = h_coll_forces[3 * (dragon_surf_verts + i) + 1];
      const double fz = h_coll_forces[3 * (dragon_surf_verts + i) + 2];
      
      // Accumulate to position DOF of ANCF node
      net_ext_force(pos_coef * 3 + 0) += fx;
      net_ext_force(pos_coef * 3 + 1) += fy;
      net_ext_force(pos_coef * 3 + 2) += fz;
    }
  };

  // =========================================================================
  // Simulation loop
  // =========================================================================
  std::cout << "\nStarting simulation (" << max_steps << " steps)\n";
  std::cout << "Time step: " << dt << " s\n\n";

  for (int step = 0; step < max_steps; ++step) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1) Update collision node buffer (both dragon and net)
    update_collision_nodes();

    // 2) Collision detection
    CollisionSystemInput coll_in;
    coll_in.d_nodes_xyz = d_collision_nodes;
    coll_in.n_nodes = total_coll_nodes;
    coll_in.d_vel_xyz = nullptr;  // Velocity not used for force computation
    coll_in.dt = dt;

    CollisionSystemParams coll_params;
    coll_params.damping = 50.0;
    coll_params.friction = contact_friction;

    collision_system->Step(coll_in, coll_params);

    int num_contacts = collision_system->GetNumContacts();

    // 3) Apply external forces to dragon: gravity + contact
    HANDLE_ERROR(cudaMemcpy(gpu_dragon.GetExternalForceDevicePtr(), d_dragon_gravity,
                            dragon_n_dofs * sizeof(double), cudaMemcpyDeviceToDevice));
    
    if (num_contacts > 0) {
      // Get collision forces to host for processing
      HANDLE_ERROR(cudaMemcpy(h_coll_forces.data(), 
                              collision_system->GetExternalForcesDevicePtr(),
                              total_coll_nodes * 3 * sizeof(double), cudaMemcpyDeviceToHost));
      
      // Clamp collision forces to prevent instability
      const double max_force_per_node = 1e6;  // Maximum force per node component
      for (int i = 0; i < total_coll_nodes * 3; ++i) {
        if (std::abs(h_coll_forces[i]) > max_force_per_node) {
          h_coll_forces[i] = std::copysign(max_force_per_node, h_coll_forces[i]);
        }
        if (!std::isfinite(h_coll_forces[i])) {
          h_coll_forces[i] = 0.0;
        }
      }

      // Apply dragon collision forces using saved node ID mapping
      Eigen::VectorXd h_dragon_coll_force = Eigen::VectorXd::Zero(dragon_n_dofs);
      for (int i = 0; i < dragon_surf_verts; ++i) {
        int node_id = dragon_surf_node_ids[i];
        if (node_id >= 0 && node_id < inst_dragon.num_nodes) {
          h_dragon_coll_force(3 * node_id + 0) += h_coll_forces[3 * i + 0];
          h_dragon_coll_force(3 * node_id + 1) += h_coll_forces[3 * i + 1];
          h_dragon_coll_force(3 * node_id + 2) += h_coll_forces[3 * i + 2];
        }
      }
      
      // Add dragon collision forces to external force
      double* d_dragon_coll = nullptr;
      HANDLE_ERROR(cudaMalloc(&d_dragon_coll, dragon_n_dofs * sizeof(double)));
      HANDLE_ERROR(cudaMemcpy(d_dragon_coll, h_dragon_coll_force.data(),
                              dragon_n_dofs * sizeof(double), cudaMemcpyHostToDevice));
      const double alpha = 1.0;
      CheckCublas(cublasDaxpy(cublas_handle, dragon_n_dofs, &alpha,
                              d_dragon_coll, 1,
                              gpu_dragon.GetExternalForceDevicePtr(), 1),
                  "cublasDaxpy dragon");
      HANDLE_ERROR(cudaFree(d_dragon_coll));

      // Apply net collision forces: gravity + contact on position DOFs
      Eigen::VectorXd net_ext_force = h_net_gravity;  // Start with gravity
      apply_net_collision_forces(net_ext_force);
      gpu_net.SetExternalForce(net_ext_force);
    } else {
      // No contacts - just apply gravity to net
      gpu_net.SetExternalForce(h_net_gravity);
    }

    // 4) Solve dragon
    dragon_solver->Solve();

    // 5) Solve net
    net_solver.Solve();

    // 6) Export VTU
    if (step % export_interval == 0) {
      // Dragon VTU
      Eigen::VectorXd dragon_x, dragon_y, dragon_z;
      gpu_dragon.RetrievePositionToCPU(dragon_x, dragon_y, dragon_z);

      Eigen::MatrixXd dragon_current(inst_dragon.num_nodes, 3);
      Eigen::VectorXd dragon_disp(inst_dragon.num_nodes * 3);
      for (int i = 0; i < inst_dragon.num_nodes; ++i) {
        dragon_current(i, 0) = dragon_x(i);
        dragon_current(i, 1) = dragon_y(i);
        dragon_current(i, 2) = dragon_z(i);
        dragon_disp(3 * i + 0) = dragon_x(i) - h_dragon_x(i);
        dragon_disp(3 * i + 1) = dragon_y(i) - h_dragon_y(i);
        dragon_disp(3 * i + 2) = dragon_z(i) - h_dragon_z(i);
      }

      std::ostringstream dragon_file;
      dragon_file << "output/dragon_on_net/dragon_" << std::setfill('0') 
                  << std::setw(4) << step << ".vtu";
      VisualizationUtils::ExportMeshWithDisplacement(
          dragon_current, dragon_elems_local, dragon_disp, dragon_file.str());

      // Net VTU
      Eigen::VectorXd net_x, net_y, net_z;
      gpu_net.RetrievePositionToCPU(net_x, net_y, net_z);

      std::ostringstream net_file;
      net_file << "output/dragon_on_net/net_" << std::setfill('0')
               << std::setw(4) << step << ".vtu";
      VisualizationUtils::ExportANCF3243ToVTU(
          net_x, net_y, net_z, net_mesh.element_connectivity, 
          net_W, net_H, net_file.str());
    }

    // 7) Progress
    if (step % 20 == 0) {
      Eigen::VectorXd dragon_x, dragon_y, dragon_z;
      gpu_dragon.RetrievePositionToCPU(dragon_x, dragon_y, dragon_z);

      auto t1 = std::chrono::high_resolution_clock::now();
      double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

      std::cout << "Step " << std::setw(4) << step
                << ": dragon_z=" << std::fixed << std::setprecision(4) << dragon_z.mean()
                << ", contacts=" << num_contacts
                << ", ms=" << std::setprecision(2) << step_ms << "\n";
    }
  }

  // =========================================================================
  // Cleanup
  // =========================================================================
  CheckCublas(cublasDestroy(cublas_handle), "cublasDestroy");
  HANDLE_ERROR(cudaFree(d_dragon_gravity));
  HANDLE_ERROR(cudaFree(d_collision_nodes));
  gpu_dragon.Destroy();
  gpu_net.Destroy();

  std::cout << "\nDone. Output written to output/dragon_on_net/\n";
  return 0;
}
