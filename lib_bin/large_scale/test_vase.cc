/**
 * Vase Drop Onto Protective Foam Simulation
 * Author: Json Zhou (zzhou292@wisc.edu)
 *
 * A ceramic bouquet vase is dropped into a protective packaging foam with a
 * matching hollow cavity. The foam is clamped at its bottom face. The vase
 * falls under gravity (-Z) and settles into the foam hollow. Contact is
 * resolved by the DEME mesh-mesh collision system.
 *
 * Two material model options are provided:
 *   MAT_SVK: Saint Venant-Kirchhoff (geometrically nonlinear linear elasticity)
 *   MAT_MR:  Mooney-Rivlin (hyperelastic, suitable for soft foam behavior)
 *
 * Meshes (coordinate system: Z up, gravity = -Z):
 *   Foam: data/meshes/T10/vase/protect_foam_ascii.1.{node,ele}
 *         Bounds: X=[-0.10,0.10], Y=[0,0.29], Z=[-0.10,0.00]
 *         Hollow opens at Z=0 (top face); bottom face at Z=-0.10 is clamped.
 *   Vase: data/meshes/T10/vase/bouquet_vase_ascii.1.{node,ele}
 *         Bounds (as loaded): X=[-0.07,0.07], Y=[0,0.25], Z=[-0.07,0.07]
 *         Translated by (+y=0.02, +z=0.03) to center over the foam hollow.
 *         After translation: Y=[0.02,0.27], Z=[-0.04,0.10]
 *
 * Material notes:
 *   The FE solver uses a single global stiffness (E/nu or mu10/mu01/kappa).
 *   Foam stiffness parameters are applied globally; vase element density is
 *   separately overridden to the ceramic value (2400 kg/m3) so the vase mass
 *   is physically realistic while foam deformation is captured accurately.
 *
 *   SVK ceramic reference:  E ~ 50-100 GPa, nu ~ 0.25, rho ~ 2400 kg/m3
 *   SVK foam reference:     E ~ 50 kPa,     nu ~ 0.35, rho ~ 50 kg/m3
 *   MR foam reference:      mu10 ~ 11 kPa, mu01 ~ 7.4 kPa, kappa ~ 83 kPa
 *
 * Solver: Newton (cuDSS), dt = 5e-4 s
 * Collision: DemeMeshCollisionSystem (DEME mesh-mesh contact)
 * Output: output/vase_drop/mesh_XXXX.vtu  (ParaView compatible)
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/collision/DemeMeshCollisionSystem.h"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/surface_trimesh_extract.h"
#include "prescribed_shake.h"

// ============================================================================
// Material Model Selection
//   MAT_SVK: Saint Venant-Kirchhoff — suitable when strains remain moderate
//   MAT_MR:  Mooney-Rivlin          — better captures large-deformation foam
//            behaviour; mu10 ~ 0.6*G, mu01 ~ 0.4*G (NeoHookean-like split)
// ============================================================================
enum MaterialOption { MAT_SVK, MAT_MR };
static constexpr MaterialOption MATERIAL_OPTION = MAT_SVK;  // <-- change here

// ----------------------------------------------------------------------------
// SVK material parameters
// ----------------------------------------------------------------------------
// Foam (global stiffness applied to all elements):
//   E = 50 kPa — soft packaging polyurethane foam
//   nu = 0.35  — slightly compressible foam
//   rho0 = 50 kg/m3 — typical low-density packaging foam
const SolidMaterialProperties mat_foam_svk = SolidMaterialProperties::SVK(
    1e6,    // E:           50 kPa
    0.35,   // nu
    250.0,  // rho0:        50 kg/m3
    5e3,    // eta_damp:    Kelvin-Voigt shear damping
    5e3     // lambda_damp: Kelvin-Voigt volumetric damping
);

// Vase (ceramic / porcelain). You can lower E if the Newton solve becomes
// stiff.
const SolidMaterialProperties mat_vase_svk =
    SolidMaterialProperties::SVK(5e10,   // E:   ~50 GPa
                                 0.25,   // nu:  ceramic-like
                                 2400.0  // rho: kg/m3
    );

// ----------------------------------------------------------------------------
// Mooney-Rivlin material parameters (equivalent foam stiffness)
//   Derived from E = 50 kPa, nu = 0.35:
//     G     = E / (2*(1+nu))       = 18518 Pa
//     K     = E / (3*(1-2*nu))     = 55556 Pa
//     mu10  = 0.6 * G              = 11111 Pa  (NeoHookean-dominant)
//     mu01  = 0.4 * G              =  7407 Pa
//     kappa = 1.5 * K              = 83333 Pa  (bulk penalty)
// ----------------------------------------------------------------------------
namespace {
constexpr double MR_E  = 5e4;
constexpr double MR_nu = 0.35;
constexpr double MR_G  = MR_E / (2.0 * (1.0 + MR_nu));
constexpr double MR_K  = MR_E / (3.0 * (1.0 - 2.0 * MR_nu));
}  // namespace

const SolidMaterialProperties mat_foam_mr =
    SolidMaterialProperties::MooneyRivlin(0.6 * MR_G,  // mu10: 11111 Pa
                                          0.4 * MR_G,  // mu01:  7407 Pa
                                          1.5 * MR_K,  // kappa: 83333 Pa
                                          50.0         // rho0:  50 kg/m3
    );

static constexpr double rho_vase_mr = 2400.0;  // kg/m3 (ceramic)

// ============================================================================
// Simulation parameters
// ============================================================================
static constexpr double gravity = -9.81;  // m/s^2, applied in -Z
static constexpr double dt      = 1e-4;   // Time step (s)
// Schedule: 1000 static -> 1000 shaking -> 1000 static
static constexpr int static_pre_steps  = 1000;
static constexpr int shake_steps       = 2000;
static constexpr int static_post_steps = 1000;
static constexpr int num_steps =
    static_pre_steps + shake_steps + static_post_steps;  // Total steps
static constexpr int export_interval = 20;  // VTU output every N steps

// Shaking (prescribed motion on fixed foam nodes)
static constexpr int shake_start_step = static_pre_steps;
static constexpr int shake_end_step   = static_pre_steps + shake_steps;
static constexpr double shake_amp_x   = 0.005;  // meters (±)
static constexpr double shake_speed_x = 0.25;   // m/s (piecewise-constant)

// DEME contact parameters
//   mu_s = 0.6: ceramic-on-foam static friction (moderate)
//   mu_k = 0.5: kinetic friction
//   stiffness = 1e6 N/m: soft contact (foam surface compliance)
//   restitution = 0.3: inelastic impact (energy absorbed by foam)
static constexpr double contact_mu_s = 0.6;
static constexpr double contact_mu_k = 0.5;
static constexpr double contact_stiffness =
    8e6;  // reduced: stable with dt=1e-4 and m~1 kg
static constexpr double contact_cor =
    0.1;  // near-zero CoR: highly inelastic (foam absorbs impact)


static void CheckCublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error (" << what << "): status=" << int(status)
              << "\n";
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  std::cout << "========================================\n";
  std::cout << "Vase Drop Onto Protective Foam\n";
  std::cout << "========================================\n";

  // Optional positional args: [mu_s] [mu_k] [self_collision] [steps]
  // [export_interval]
  double mu_s         = contact_mu_s;
  double mu_k         = contact_mu_k;
  bool self_collision = false;
  int max_steps       = num_steps;
  int exp_interval    = export_interval;

  if (argc > 1)
    mu_s = std::atof(argv[1]);
  if (argc > 2)
    mu_k = std::atof(argv[2]);
  if (argc > 3)
    self_collision = (std::atoi(argv[3]) != 0);
  if (argc > 4) {
    int v = std::atoi(argv[4]);
    if (v > 0)
      max_steps = v;
  }
  if (argc > 5)
    exp_interval = std::atoi(argv[5]);

  std::cout << "Material:         "
            << (MATERIAL_OPTION == MAT_SVK ? "SVK" : "Mooney-Rivlin") << "\n";
  std::cout << "dt:               " << dt << " s\n";
  std::cout << "mu_s:             " << mu_s << "\n";
  std::cout << "mu_k:             " << mu_k << "\n";
  std::cout << "self_collision:   " << (self_collision ? "yes" : "no") << "\n";
  std::cout << "max_steps:        " << max_steps << "\n";
  std::cout << "export_interval:  " << exp_interval << "\n\n";

  std::filesystem::create_directories("output/vase_drop");

  // =========================================================================
  // Load meshes: foam (base/container) first, then vase (drop body)
  // =========================================================================
  ANCFCPUUtils::MeshManager mesh_manager;

  const std::string foam_node =
      "data/meshes/T10/vase/protect_foam_ascii.1.node";
  const std::string foam_ele = "data/meshes/T10/vase/protect_foam_ascii.1.ele";
  const std::string vase_node =
      "data/meshes/T10/vase/bouquet_vase_ascii.1.node";
  const std::string vase_ele = "data/meshes/T10/vase/bouquet_vase_ascii.1.ele";

  const int mesh_foam = mesh_manager.LoadMesh(foam_node, foam_ele, "foam");
  const int mesh_vase = mesh_manager.LoadMesh(vase_node, vase_ele, "vase");

  if (mesh_foam < 0 || mesh_vase < 0) {
    std::cerr << "Failed to load meshes.\n"
              << "  foam: " << foam_node << "\n"
              << "  vase: " << vase_node << "\n";
    return 1;
  }

  const auto& inst_foam = mesh_manager.GetMeshInstance(mesh_foam);
  const auto& inst_vase = mesh_manager.GetMeshInstance(mesh_vase);

  std::cout << "Loaded meshes:\n";
  std::cout << "  Foam: " << inst_foam.num_nodes << " nodes, "
            << inst_foam.num_elements << " elements\n";
  std::cout << "  Vase: " << inst_vase.num_nodes << " nodes, "
            << inst_vase.num_elements << " elements\n";

  // Translate vase to align with the foam hollow cavity:
  //   +y = 0.02 m: center vase in Y over the foam hollow
  //   +z = 0.03 m: position vase at the hollow opening in Z
  // After translation: vase Y=[0.02,0.27], Z=[-0.04,0.10]
  // The vase bottom (Z=-0.04) sits just inside the foam opening (Z=0) so
  // gravity gently pulls it down into the hollow.
  mesh_manager.TranslateMesh(mesh_vase, 0.0, 0.025, 0.01);
  std::cout << "Translated vase: (+x=0, +y=0.02, +z=0.03) m\n\n";

  // =========================================================================
  // Build unified node/element arrays
  // =========================================================================
  const Eigen::MatrixXd& initial_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& elements      = mesh_manager.GetAllElements();
  const int n_nodes                    = mesh_manager.GetTotalNodes();
  const int n_elems                    = mesh_manager.GetTotalElements();

  std::cout << "Total: " << n_nodes << " nodes, " << n_elems << " elements\n\n";

  // =========================================================================
  // GPU element data initialization
  // =========================================================================
  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    h_x12(i) = initial_nodes(i, 0);
    h_y12(i) = initial_nodes(i, 1);
    h_z12(i) = initial_nodes(i, 2);
  }

  // Clamp foam nodes below Z = -0.05 m (lower half of foam, Z in [-0.10, 0.00])
  static constexpr double fix_z_threshold = -0.05;

  std::vector<int> fixed_indices;
  fixed_indices.reserve(inst_foam.num_nodes / 2);
  for (int i = 0; i < inst_foam.num_nodes; ++i) {
    const int idx = inst_foam.node_offset + i;
    if (initial_nodes(idx, 2) < fix_z_threshold) {
      fixed_indices.push_back(idx);
    }
  }

  Eigen::VectorXi h_fixed(static_cast<int>(fixed_indices.size()));
  for (int i = 0; i < static_cast<int>(fixed_indices.size()); ++i) {
    h_fixed(i) = fixed_indices[i];
  }
  std::cout << "Fixed " << h_fixed.size() << " foam nodes (Z < "
            << fix_z_threshold << " m)\n";
  gpu_t10_data.SetNodalFixed(h_fixed);

  // Demo-specific: build a device list of the fixed (clamped) foam nodes so we
  // can apply prescribed base motion without adding demo logic to FEAT10Data.
  int* d_fixed_node_ids = nullptr;
  const int n_fixed     = static_cast<int>(fixed_indices.size());
  HANDLE_ERROR(cudaMalloc(&d_fixed_node_ids, n_fixed * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_fixed_node_ids, fixed_indices.data(),
                          n_fixed * sizeof(int), cudaMemcpyHostToDevice));

  // =========================================================================
  // FE setup: quadrature, material, mass matrix
  // =========================================================================
  gpu_t10_data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                     Quadrature::tet5pt_z, Quadrature::tet5pt_weights, h_x12,
                     h_y12, h_z12, elements);

  // Assign per-mesh materials (same model across all meshes).
  // This keeps the original behavior (uniform stiffness, per-mesh density),
  // but now the API supports per-mesh stiffness too (e.g., different E/nu).
  if (MATERIAL_OPTION == MAT_SVK) {
    mesh_manager.SetMeshMaterial(mesh_foam, mat_foam_svk);
    mesh_manager.SetMeshMaterial(mesh_vase, mat_vase_svk);
    gpu_t10_data.ApplyMaterialsFromMeshManager(mesh_manager);

    std::cout << "Material model: SVK\n";
    std::cout << "  Foam: E=" << mat_foam_svk.E << " Pa, nu=" << mat_foam_svk.nu
              << ", rho=" << mat_foam_svk.rho0 << " kg/m3\n";
    std::cout << "  Vase: E=" << mat_vase_svk.E << " Pa, nu=" << mat_vase_svk.nu
              << ", rho=" << mat_vase_svk.rho0 << " kg/m3\n";
  } else {
    SolidMaterialProperties mat_vase_mr = mat_foam_mr;
    mat_vase_mr.rho0                    = rho_vase_mr;
    mesh_manager.SetMeshMaterial(mesh_foam, mat_foam_mr);
    mesh_manager.SetMeshMaterial(mesh_vase, mat_vase_mr);
    gpu_t10_data.ApplyMaterialsFromMeshManager(mesh_manager);

    std::cout << "Material model: Mooney-Rivlin\n";
    std::cout << "  Foam: mu10=" << mat_foam_mr.mu10
              << " Pa, mu01=" << mat_foam_mr.mu01
              << " Pa, kappa=" << mat_foam_mr.kappa
              << " Pa, rho=" << mat_foam_mr.rho0 << " kg/m3\n";
    std::cout << "  Vase: mu10=" << mat_vase_mr.mu10
              << " Pa, mu01=" << mat_vase_mr.mu01
              << " Pa, kappa=" << mat_vase_mr.kappa
              << " Pa, rho=" << mat_vase_mr.rho0 << " kg/m3\n";
  }
  std::cout << "\n";

  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();

  // =========================================================================
  // Lumped mass matrix (for gravity force computation)
  // =========================================================================
  Eigen::VectorXd lumped_mass(n_nodes);
  lumped_mass.setZero();
  {
    std::vector<int> offs, cols;
    std::vector<double> vals;
    gpu_t10_data.RetrieveMassCSRToCPU(offs, cols, vals);
    if (static_cast<int>(offs.size()) == n_nodes + 1) {
      for (int i = 0; i < n_nodes; ++i) {
        for (int k = offs[i]; k < offs[i + 1]; ++k) {
          lumped_mass(i) += vals[k];
        }
      }
    } else {
      std::cerr << "Warning: unexpected mass CSR size; using unit mass.\n";
      lumped_mass.setOnes();
    }
  }

  // =========================================================================
  // Newton solver
  // =========================================================================
  SyncedNewtonParams params = {1e-3, 0.0, 1e-5, 1e12, 3, 10, dt, false};
  SyncedNewtonSolver newton(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  newton.Setup();
  newton.SetParameters(&params);
  newton.AnalyzeHessianSparsity();

  double* d_vel_guess = newton.GetVelocityGuessDevicePtr();
  HANDLE_ERROR(cudaMemset(d_vel_guess, 0, n_nodes * 3 * sizeof(double)));

  std::cout << "Newton solver initialized.\n\n";

  // =========================================================================
  // DEME mesh-mesh collision system
  // =========================================================================
  // Extract surface triangle meshes from the tet10 volumetric meshes
  ANCFCPUUtils::SurfaceTriMesh foam_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(initial_nodes, elements, inst_foam);
  ANCFCPUUtils::SurfaceTriMesh vase_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(initial_nodes, elements, inst_vase);

  std::vector<DemeMeshCollisionBody> bodies;

  // Foam body (family 0): stationary container; no patch splitting needed
  {
    DemeMeshCollisionBody b;
    b.surface                  = std::move(foam_surface);
    b.family                   = 0;
    b.split_into_patches       = false;
    b.skip_self_contact_forces = false;
    bodies.push_back(std::move(b));
  }

  // Vase body (family 1): falling ceramic object; patch splitting improves
  // contact resolution on curved surfaces
  {
    DemeMeshCollisionBody b;
    b.surface            = std::move(vase_surface);
    b.family             = 1;
    b.split_into_patches = true;
    b.patch_angle_deg    = -1.0f;  // use DEME default patch angle
    bodies.push_back(std::move(b));
  }

  // Construct DEME system (7-arg: mu_s, mu_k, stiffness, CoR,
  //                                enable_self_collision, dt)
  auto collision_system = std::make_unique<DemeMeshCollisionSystem>(
      std::move(bodies), mu_s, mu_k, contact_stiffness, contact_cor,
      self_collision, dt);

  // Shared device node buffer for collision (column-major: [x... y... z...])
  double* d_nodes_col = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_nodes_col, n_nodes * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_nodes_col, gpu_t10_data.GetX12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(d_nodes_col + n_nodes, gpu_t10_data.GetY12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(d_nodes_col + 2 * n_nodes,
                          gpu_t10_data.GetZ12DevicePtr(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
  collision_system->BindNodesDevicePtr(d_nodes_col, n_nodes);

  // =========================================================================
  // Gravity: applied to all free nodes in -Z direction
  //   Foam nodes have low mass (rho=50) so foam gravity is negligible but kept
  //   for physical completeness. Fixed nodes are zeroed out.
  // =========================================================================
  Eigen::VectorXd h_f_gravity = Eigen::VectorXd::Zero(n_nodes * 3);
  for (int i = 0; i < n_nodes; ++i) {
    h_f_gravity(3 * i + 2) += lumped_mass(i) * gravity;  // -Z component
  }
  // Zero gravity on clamped foam bottom nodes
  for (int idx : fixed_indices) {
    h_f_gravity(3 * idx + 2) = 0.0;
  }

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
  std::cout << "Starting simulation (" << max_steps << " steps, dt=" << dt
            << " s)\n\n";

  double shake_dx_prev = 0.0;

  for (int step = 0; step < max_steps; ++step) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Prescribed shaking: move fixed foam nodes in ±X with |v|=shake_speed_x.
    // Implemented as a triangular wave so velocity magnitude is constant except
    // at direction reversals.
    if (step >= shake_start_step && step < shake_end_step) {
      const double t_shake = (step - shake_start_step) * dt;
      const double t1      = shake_amp_x / shake_speed_x;  // 0 -> +A
      const double period =
          4.0 * shake_amp_x / shake_speed_x;  // 0 -> +A -> -A -> 0
      const double phase = std::fmod(t_shake, period);

      double dx = 0.0;
      if (phase < t1) {
        dx = shake_speed_x * phase;
      } else if (phase < 3.0 * t1) {
        dx = shake_amp_x - shake_speed_x * (phase - t1);
      } else {
        dx = -shake_amp_x + shake_speed_x * (phase - 3.0 * t1);
      }

      const double delta_dx = dx - shake_dx_prev;
      PrescribedShake::OffsetNodesAndTargets(
          gpu_t10_data.GetX12DevicePtr(), gpu_t10_data.GetX12JacDevicePtr(),
          d_fixed_node_ids, n_fixed, delta_dx);
      shake_dx_prev = dx;
    }

    // 1) Sync node positions to collision buffer (device -> device)
    HANDLE_ERROR(cudaMemcpy(d_nodes_col, gpu_t10_data.GetX12DevicePtr(),
                            n_nodes * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_nodes_col + n_nodes, gpu_t10_data.GetY12DevicePtr(),
                   n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_nodes_col + 2 * n_nodes, gpu_t10_data.GetZ12DevicePtr(),
                   n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));

    // 2) Run DEME collision detection and compute contact forces
    CollisionSystemInput coll_in;
    coll_in.d_nodes_xyz = d_nodes_col;
    coll_in.n_nodes     = n_nodes;
    coll_in.d_vel_xyz   = d_vel_guess;
    coll_in.dt          = dt;

    CollisionSystemParams coll_params;
    coll_params.damping  = contact_cor;
    coll_params.friction = mu_k;

    collision_system->Step(coll_in, coll_params);
    const int num_contacts = collision_system->GetNumContacts();

    // 3) External forces = gravity + contact forces
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

    // 4) Newton step
    newton.Solve();

    // 5) Export VTU
    if (exp_interval > 0 && step % exp_interval == 0) {
      std::ostringstream fn;
      fn << "output/vase_drop/mesh_" << std::setfill('0') << std::setw(4)
         << step << ".vtu";
      gpu_t10_data.WriteOutputVTU(fn.str());
    }

    // 6) Progress report every 20 steps
    if (step % 20 == 0) {
      auto t1 = std::chrono::high_resolution_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      std::cout << "Step " << std::setw(4) << step
                << "  contacts=" << std::setw(5) << num_contacts
                << "  ms=" << std::fixed << std::setprecision(2) << ms << "\n";
    }
  }

  // =========================================================================
  // Cleanup
  // =========================================================================
  CheckCublas(cublasDestroy(cublas_handle), "cublasDestroy");
  HANDLE_ERROR(cudaFree(d_f_gravity));
  HANDLE_ERROR(cudaFree(d_nodes_col));
  HANDLE_ERROR(cudaFree(d_fixed_node_ids));
  gpu_t10_data.Destroy();

  std::cout << "\n========================================\n";
  std::cout << "Simulation complete.\n";
  std::cout << "Output: output/vase_drop/mesh_XXXX.vtu\n";
  std::cout << "========================================\n";
  return 0;
}
