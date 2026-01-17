/**
 * Sphere Stacking Simulation
 * Author: Json Zhou (zzhou292@wisc.edu)
 *
 * Three spheres stacking on a fixed plate: two spheres at the bottom with a
 * small gap, one sphere on top. Contact handled via DEME collision system.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>

#include "../../lib_src/collision/DemeMeshCollisionSystem.h"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/cli_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/surface_trimesh_extract.h"
#include "../../lib_utils/visualization_utils.h"

// Material properties for each sphere (using SolidMaterialProperties)
const SolidMaterialProperties mat_plate = SolidMaterialProperties::SVK(
    1e7,     // E: Young's modulus (Pa)
    0.3,     // nu: Poisson's ratio
    1000.0,  // rho0: Density (kg/m³)
    2e4,     // eta_damp
    2e4      // lambda_damp
);

const SolidMaterialProperties mat_sphere1 = SolidMaterialProperties::SVK(
    1e7,     // E: Young's modulus (Pa)
    0.3,     // nu: Poisson's ratio
    1000.0,  // rho0: Density (kg/m³)
    2e4,     // eta_damp
    2e4      // lambda_damp
);

const SolidMaterialProperties mat_sphere2 = SolidMaterialProperties::SVK(
    1e7,     // E: Young's modulus (Pa)
    0.3,     // nu: Poisson's ratio
    1000.0,  // rho0: Density (kg/m³)
    2e4,     // eta_damp
    2e4      // lambda_damp
);

const SolidMaterialProperties mat_sphere3 = SolidMaterialProperties::SVK(
    1e7,     // E: Young's modulus (Pa)
    0.3,     // nu: Poisson's ratio
    300.0,  // rho0: Density (kg/m³)
    2e4,     // eta_damp
    2e4      // lambda_damp
);

// Geometry
const double sphere_radius = 0.15;
const double gap_factor    = 0.3;  // Gap = 0.3 * R between bottom spheres

// Simulation parameters
const double gravity = -9.81;
const double dt      = 5e-4;
const int num_steps  = 5000;

// Contact parameters
// DEME uses coefficient of restitution (CoR) in its frictional Hertzian model.
// Valid range is [0, 1].
const double contact_damping_default  = 0.5;
const double contact_friction_default = 0.01;
const double contact_stiffness        = 2e8;

using ANCFCPUUtils::VisualizationUtils;

static void CheckCublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error (" << what << "): status=" << int(status) << "\n";
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  std::cout << "========================================" << std::endl;
  std::cout << "Sphere Stacking Simulation" << std::endl;
  std::cout << "========================================" << std::endl;

  double contact_damping     = contact_damping_default;
  double contact_friction    = contact_friction_default;
  bool enable_self_collision = false;
  int max_steps              = num_steps;
  int export_interval        = 10;

  const bool has_flag_args =
      (argc > 1 && argv[1] &&
       (std::string_view(argv[1]) == "--help" ||
        std::string_view(argv[1]) == "-h" ||
        std::string_view(argv[1]).rfind("--", 0) == 0));
  if (has_flag_args) {
    ANCFCPUUtils::Cli cli(argv[0] ? argv[0] : "sphere_stacking");
    cli.SetDescription(
        "Sphere stacking demo (DEME mesh-mesh contact coupled to FE).");
    cli.AddDouble("cor", contact_damping_default,
                  "contact restitution (CoR), range [0, 1]");
    cli.AddDouble("mu", contact_friction_default, "contact friction mu");
    cli.AddBool("self_collision", false, "enable self collision");
    cli.AddInt("steps", num_steps, "max simulation steps");
    cli.AddInt("export_interval", 10, "VTK export interval (0 disables)");

    std::string err;
    if (!cli.Parse(argc, argv, &err) || cli.HelpRequested()) {
      if (!err.empty()) std::cerr << "Error: " << err << "\n\n";
      cli.PrintUsage(std::cerr);
      return cli.HelpRequested() ? 0 : 1;
    }

    contact_damping        = cli.GetDouble("cor");
    contact_friction       = cli.GetDouble("mu");
    enable_self_collision  = cli.GetBool("self_collision");
    max_steps              = cli.GetInt("steps");
    export_interval        = cli.GetInt("export_interval");
  } else {
    // Backward-compatible positional args:
    //   argv[1]=CoR, argv[2]=mu, argv[3]=self_collision (0/1),
    //   argv[4]=steps, argv[5]=export_interval
    if (argc > 1) contact_damping = std::atof(argv[1]);
    if (argc > 2) contact_friction = std::atof(argv[2]);
    if (argc > 3) enable_self_collision = (std::atoi(argv[3]) != 0);
    if (argc > 4) {
      int v = std::atoi(argv[4]);
      if (v > 0) max_steps = v;
    }
    if (argc > 5) export_interval = std::atoi(argv[5]);
  }

  std::cout << "Contact restitution (CoR): " << contact_damping << std::endl;
  std::cout << "Contact friction: " << contact_friction << std::endl;
  std::cout << "Enable self collision: " << (enable_self_collision ? 1 : 0) << std::endl;
  std::cout << "Max steps: " << max_steps << std::endl;
  std::cout << "Export interval: " << export_interval << std::endl;

  std::filesystem::create_directories("output/sphere_stacking");

  // =========================================================================
  // Load meshes
  // =========================================================================
  ANCFCPUUtils::MeshManager mesh_manager;

  const std::string sphere_node = "data/meshes/ptest/sphere/sphere_r0p15_low.1.node";
  const std::string sphere_ele  = "data/meshes/ptest/sphere/sphere_r0p15_low.1.ele";
  const std::string plate_node  = "data/meshes/ptest/plate/1.1.001plate.1.node";
  const std::string plate_ele   = "data/meshes/ptest/plate/1.1.001plate.1.ele";

  int mesh_plate = mesh_manager.LoadMesh(plate_node, plate_ele, "plate");
  int mesh_sphere1 = mesh_manager.LoadMesh(sphere_node, sphere_ele, "sphere_left");
  int mesh_sphere2 = mesh_manager.LoadMesh(sphere_node, sphere_ele, "sphere_right");
  int mesh_sphere3 = mesh_manager.LoadMesh(sphere_node, sphere_ele, "sphere_top");

  if (mesh_plate < 0 || mesh_sphere1 < 0 || mesh_sphere2 < 0 || mesh_sphere3 < 0) {
    std::cerr << "Failed to load meshes" << std::endl;
    return 1;
  }

  const auto& inst_plate   = mesh_manager.GetMeshInstance(mesh_plate);
  const auto& inst_sphere1 = mesh_manager.GetMeshInstance(mesh_sphere1);
  const auto& inst_sphere2 = mesh_manager.GetMeshInstance(mesh_sphere2);
  const auto& inst_sphere3 = mesh_manager.GetMeshInstance(mesh_sphere3);

  std::cout << "Loaded meshes:" << std::endl;
  std::cout << "  Plate: " << inst_plate.num_nodes << " nodes, "
            << inst_plate.num_elements << " elements" << std::endl;
  std::cout << "  Sphere1: " << inst_sphere1.num_nodes << " nodes, "
            << inst_sphere1.num_elements << " elements" << std::endl;
  std::cout << "  Sphere2: " << inst_sphere2.num_nodes << " nodes, "
            << inst_sphere2.num_elements << " elements" << std::endl;
  std::cout << "  Sphere3: " << inst_sphere3.num_nodes << " nodes, "
            << inst_sphere3.num_elements << " elements" << std::endl;

  // =========================================================================
  // Position the objects
  // =========================================================================
  const double R   = sphere_radius;
  const double gap = gap_factor * R;  // 0.3 * 0.15 = 0.045

  // Get plate top z (plate spans z from -0.025 to 0.025, top at z=0.025)
  // We need to check the actual plate bounds
  {
    const Eigen::MatrixXd& nodes = mesh_manager.GetAllNodes();
    double plate_z_max = -1e30;
    for (int i = 0; i < inst_plate.num_nodes; ++i) {
      int idx = inst_plate.node_offset + i;
      plate_z_max = std::max(plate_z_max, nodes(idx, 2));
    }
    std::cout << "Plate top z: " << plate_z_max << std::endl;
  }

  // Plate is already centered at origin, top at z ~ 0.025
  // Small clearance above plate for initialization
  const double init_clearance = 0.00002;
  const double plate_top_z    = 0.025;

  // Bottom two spheres: centers at z = plate_top + clearance + R
  // Horizontal positions: separated by (2R + gap) center-to-center
  // Center-to-center distance = 2R + gap = 2*0.15 + 0.045 = 0.345
  // So each sphere is at x = +/- (R + gap/2) = +/- 0.1725
  const double bottom_sphere_z = plate_top_z + init_clearance + R;
  const double x_offset        = R + gap / 2.0;  // 0.15 + 0.0225 = 0.1725

  mesh_manager.TranslateMesh(mesh_sphere1, -x_offset, 0.0, bottom_sphere_z);
  mesh_manager.TranslateMesh(mesh_sphere2, +x_offset, 0.0, bottom_sphere_z);

  // Top sphere: sits on top of the two bottom spheres
  // The top sphere center is at z = bottom_sphere_z + vertical_offset
  // Vertical offset from geometry: sqrt((2R)^2 - (R + gap/2)^2)
  // Actually, the top sphere touches both bottom spheres
  // Distance between bottom sphere centers = 2*x_offset = 2R + gap
  // Top sphere touches both, so triangle formed by 3 sphere centers:
  // - Two bottom centers distance = 2*x_offset
  // - Top-to-bottom-left = 2R (touching)
  // - Top-to-bottom-right = 2R (touching)
  // Using geometry: vertical offset = sqrt((2R)^2 - x_offset^2)
  const double top_vertical_offset = std::sqrt(4.0 * R * R - x_offset * x_offset);
  const double top_sphere_z = bottom_sphere_z + top_vertical_offset - 0.002;

  mesh_manager.TranslateMesh(mesh_sphere3, 0.0, 0.0, top_sphere_z);

  std::cout << "Sphere positions:" << std::endl;
  std::cout << "  Bottom left:  x=" << -x_offset << ", z=" << bottom_sphere_z << std::endl;
  std::cout << "  Bottom right: x=" << +x_offset << ", z=" << bottom_sphere_z << std::endl;
  std::cout << "  Top:          x=0, z=" << top_sphere_z << std::endl;

  // =========================================================================
  const Eigen::MatrixXd& initial_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& elements      = mesh_manager.GetAllElements();

  int n_nodes = mesh_manager.GetTotalNodes();
  int n_elems = mesh_manager.GetTotalElements();

  std::cout << "Total nodes: " << n_nodes << std::endl;
  std::cout << "Total elements: " << n_elems << std::endl;

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

  // =========================================================================
  // Fix all plate nodes
  // =========================================================================
  std::vector<int> fixed_node_indices;
  for (int i = 0; i < inst_plate.num_nodes; ++i) {
    fixed_node_indices.push_back(inst_plate.node_offset + i);
  }

  Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
  for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
    h_fixed_nodes(i) = fixed_node_indices[i];
  }

  std::cout << "Fixed " << h_fixed_nodes.size() << " plate nodes" << std::endl;
  gpu_t10_data.SetNodalFixed(h_fixed_nodes);

  // =========================================================================
  // Setup GPU element data
  // =========================================================================
  const Eigen::VectorXd& tet5pt_x       = Quadrature::tet5pt_x;
  const Eigen::VectorXd& tet5pt_y       = Quadrature::tet5pt_y;
  const Eigen::VectorXd& tet5pt_z       = Quadrature::tet5pt_z;
  const Eigen::VectorXd& tet5pt_weights = Quadrature::tet5pt_weights;

  gpu_t10_data.Setup(tet5pt_x, tet5pt_y, tet5pt_z, tet5pt_weights, h_x12, h_y12,
                     h_z12, elements);

  // Apply material properties to all elements (using sphere1 properties as default)
  // Note: Currently GPU element data uses uniform material properties.
  // The SolidMaterialProperties abstraction allows per-object specification
  // even though GPU storage is shared.
  gpu_t10_data.ApplyMaterial(mat_sphere1);

  // Per-mesh density overrides (affects mass matrix + gravity).
  // This is important because `GPU_FEAT10_Data` otherwise assumes a uniform rho0.
  gpu_t10_data.SetDensityForElementRange(inst_plate.element_offset,
                                        inst_plate.num_elements, mat_plate.rho0);
  gpu_t10_data.SetDensityForElementRange(inst_sphere1.element_offset,
                                        inst_sphere1.num_elements, mat_sphere1.rho0);
  gpu_t10_data.SetDensityForElementRange(inst_sphere2.element_offset,
                                        inst_sphere2.num_elements, mat_sphere2.rho0);
  gpu_t10_data.SetDensityForElementRange(inst_sphere3.element_offset,
                                        inst_sphere3.num_elements, mat_sphere3.rho0);

  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();

  std::cout << "GPU element data initialized" << std::endl;

  // =========================================================================
  // Lumped mass for gravity
  // =========================================================================
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
      lumped_mass.setOnes();
    }
  }

  // =========================================================================
  // Initialize Newton solver
  // =========================================================================
  SyncedNewtonParams params = {1e-4, 0.0, 1e-6, 1e12, 3, 10, dt};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();

  double* d_vel_guess = solver.GetVelocityGuessDevicePtr();
  HANDLE_ERROR(cudaMemset(d_vel_guess, 0, n_nodes * 3 * sizeof(double)));

  std::cout << "Newton solver initialized" << std::endl;

  // =========================================================================
  // Initialize DEME collision system
  // =========================================================================
  ANCFCPUUtils::SurfaceTriMesh plate_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(initial_nodes, elements, inst_plate);
  ANCFCPUUtils::SurfaceTriMesh sphere1_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(initial_nodes, elements, inst_sphere1);
  ANCFCPUUtils::SurfaceTriMesh sphere2_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(initial_nodes, elements, inst_sphere2);
  ANCFCPUUtils::SurfaceTriMesh sphere3_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(initial_nodes, elements, inst_sphere3);

  // Compute sphere masses: mass = rho * volume, volume = (4/3) * pi * R^3
  const double sphere_volume = (4.0 / 3.0) * M_PI * R * R * R;
  const float mass_sphere1 = static_cast<float>(mat_sphere1.rho0 * sphere_volume);
  const float mass_sphere2 = static_cast<float>(mat_sphere2.rho0 * sphere_volume);
  const float mass_sphere3 = static_cast<float>(mat_sphere3.rho0 * sphere_volume);

  std::cout << "Sphere masses: " << mass_sphere1 << ", " << mass_sphere2 
            << ", " << mass_sphere3 << " kg" << std::endl;

  std::vector<DemeMeshCollisionBody> bodies;
  {
    DemeMeshCollisionBody body;
    body.surface                 = std::move(plate_surface);
    body.family                  = 0;
    body.split_into_patches      = false;
    body.skip_self_contact_forces = true;  // plate is fixed
    body.mass                    = 1000.0f;  // plate mass (arbitrary, fixed body)
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(sphere1_surface);
    body.family             = 1;
    body.split_into_patches = true;
    body.mass               = mass_sphere1;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(sphere2_surface);
    body.family             = 2;
    body.split_into_patches = true;
    body.mass               = mass_sphere2;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(sphere3_surface);
    body.family             = 3;
    body.split_into_patches = true;
    body.mass               = mass_sphere3;
    bodies.push_back(std::move(body));
  }

  if (contact_damping < 0.0 || contact_damping > 1.0) {
    std::cerr
        << "[sphere_stacking] Warning: contact restitution (CoR) should be in [0, 1], got "
        << contact_damping << " (will be clamped by DEME).\n";
  }
  auto collision_system = std::make_unique<DemeMeshCollisionSystem>(
      std::move(bodies), contact_friction, contact_stiffness, contact_damping,
      enable_self_collision);

  // Device buffer for collision node positions (column-major: [x... y... z...])
  double* d_nodes_collision = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_nodes_collision, n_nodes * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_nodes_collision, gpu_t10_data.GetX12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(d_nodes_collision + n_nodes,
                          gpu_t10_data.GetY12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(d_nodes_collision + 2 * n_nodes,
                          gpu_t10_data.GetZ12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));

  collision_system->BindNodesDevicePtr(d_nodes_collision, n_nodes);

  // Precompute gravity on device
  Eigen::VectorXd h_f_gravity = Eigen::VectorXd::Zero(n_nodes * 3);
  auto addGravityForInstance = [&](const ANCFCPUUtils::MeshInstance& inst) {
    for (int i = 0; i < inst.num_nodes; ++i) {
      const int idx = inst.node_offset + i;
      h_f_gravity(3 * idx + 2) += lumped_mass(idx) * gravity;
    }
  };
  addGravityForInstance(inst_sphere1);
  addGravityForInstance(inst_sphere2);
  addGravityForInstance(inst_sphere3);

  double* d_f_gravity = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_f_gravity, n_nodes * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_f_gravity, h_f_gravity.data(),
                          n_nodes * 3 * sizeof(double), cudaMemcpyHostToDevice));

  cublasHandle_t cublas_handle = nullptr;
  CheckCublas(cublasCreate(&cublas_handle), "cublasCreate");

  std::cout << "DEME collision system initialized" << std::endl;

  // =========================================================================
  // Simulation loop
  // =========================================================================
  std::cout << "\n========================================" << std::endl;
  std::cout << "Starting simulation (" << max_steps << " steps)" << std::endl;
  std::cout << "========================================\n" << std::endl;

  Eigen::VectorXd displacement(n_nodes * 3);
  displacement.setZero();

  for (int step = 0; step < max_steps; ++step) {
    // Update collision node buffer from solver state (device->device)
    HANDLE_ERROR(cudaMemcpy(d_nodes_collision, gpu_t10_data.GetX12DevicePtr(),
                            n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(d_nodes_collision + n_nodes,
                            gpu_t10_data.GetY12DevicePtr(),
                            n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(d_nodes_collision + 2 * n_nodes,
                            gpu_t10_data.GetZ12DevicePtr(),
                            n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));

    // Collision detection + contact forces
    CollisionSystemInput coll_in;
    coll_in.d_nodes_xyz = d_nodes_collision;
    coll_in.n_nodes     = n_nodes;
    coll_in.d_vel_xyz   = d_vel_guess;
    coll_in.dt          = dt;

    CollisionSystemParams coll_params;
    coll_params.damping  = 0.0;  // DEME uses constructor `restitution` (CoR)
    coll_params.friction = contact_friction;

    collision_system->Step(coll_in, coll_params);
    const int num_contacts = collision_system->GetNumContacts();

    // External forces: gravity + contact
    HANDLE_ERROR(cudaMemcpy(gpu_t10_data.GetExternalForceDevicePtr(),
                            d_f_gravity, n_nodes * 3 * sizeof(double),
                            cudaMemcpyDeviceToDevice));

    if (num_contacts > 0) {
      const double alpha = 1.0;
      CheckCublas(cublasDaxpy(cublas_handle, n_nodes * 3, &alpha,
                              collision_system->GetExternalForcesDevicePtr(), 1,
                              gpu_t10_data.GetExternalForceDevicePtr(), 1),
                  "cublasDaxpy(contact + gravity)");
    }

    // Newton solve
    solver.Solve();

    // Export VTK
    if (export_interval > 0 && step % export_interval == 0) {
      Eigen::VectorXd x12_current, y12_current, z12_current;
      gpu_t10_data.RetrievePositionToCPU(x12_current, y12_current, z12_current);

      Eigen::MatrixXd current_nodes(n_nodes, 3);
      for (int i = 0; i < n_nodes; ++i) {
        current_nodes(i, 0)     = x12_current(i);
        current_nodes(i, 1)     = y12_current(i);
        current_nodes(i, 2)     = z12_current(i);
        displacement(3 * i)     = x12_current(i) - initial_nodes(i, 0);
        displacement(3 * i + 1) = y12_current(i) - initial_nodes(i, 1);
        displacement(3 * i + 2) = z12_current(i) - initial_nodes(i, 2);
      }

      std::ostringstream filename;
      filename << "output/sphere_stacking/mesh_" << std::setfill('0')
               << std::setw(4) << step << ".vtu";
      VisualizationUtils::ExportMeshWithDisplacement(
          current_nodes, elements, displacement, filename.str());
    }

    // Print progress
    if (step % 50 == 0) {
      Eigen::VectorXd x12_current, y12_current, z12_current;
      gpu_t10_data.RetrievePositionToCPU(x12_current, y12_current, z12_current);

      double top_z_avg = 0.0;
      for (int i = 0; i < inst_sphere3.num_nodes; ++i) {
        int idx = inst_sphere3.node_offset + i;
        top_z_avg += z12_current(idx);
      }
      top_z_avg /= inst_sphere3.num_nodes;

      std::cout << "Step " << std::setw(4) << step
                << ": contacts=" << std::setw(5) << num_contacts
                << ", top_z=" << std::fixed << std::setprecision(5) << top_z_avg
                << std::endl;
    }
  }

  // =========================================================================
  // Cleanup
  // =========================================================================
  cublasDestroy(cublas_handle);
  HANDLE_ERROR(cudaFree(d_nodes_collision));
  HANDLE_ERROR(cudaFree(d_f_gravity));
  gpu_t10_data.Destroy();

  std::cout << "\n========================================" << std::endl;
  std::cout << "Simulation complete!" << std::endl;
  std::cout << "Output files in: output/sphere_stacking/" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
