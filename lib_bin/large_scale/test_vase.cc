/**
 * Vase Drop Onto Protective Foam Simulation
 * Author: Json Zhou (zzhou292@wisc.edu)
 *
 * A ceramic bouquet vase is dropped into a protective foam insert with a
 * matching hollow cavity. The foam and vase are solved as separate FEAT10
 * blocks and coupled through the multi-element Newton solver plus DEME mesh
 * contact.
 *
 * This split is required for mixed constitutive models:
 *   - Vase: Saint Venant-Kirchhoff (SVK)
 *   - Foam: Mooney-Rivlin
 *
 * The single FEAT10 object supports per-mesh material parameters, but not a
 * mix of constitutive models inside one object.
 *
 * Output:
 *   output/vase_drop/foam_XXXX.vtu
 *   output/vase_drop/vase_XXXX.vtu
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "../../lib_src/collision/DemeMeshCollisionSystem.h"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEMultiElementProblem.h"
#include "../../lib_src/solvers/MultiElementNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/surface_trimesh_extract.h"
#include "prescribed_shake.h"

namespace {

enum class FoamMaterialType {
  kNeoprene50A     = 0,
  kPolyurethane50A = 1,
  kEva80           = 2,
  kEva95           = 3,
  kNeoprene60A     = 4,
};

struct FoamMaterialPreset {
  const char* cli_name;
  const char* label;
  SolidMaterialProperties material;
};

static constexpr double kFoamEtaDamp          = 5e3;
static constexpr double kFoamLambdaDamp       = 5e3;

// Vase (ceramic / porcelain). Keep SVK to avoid forcing a hyperelastic fit on
// a stiff, small-strain body.
const SolidMaterialProperties kVaseMaterial =
    SolidMaterialProperties::SVK(5e10,   // E: ~50 GPa
                                 0.25,   // nu
                                 2400.0  // rho0: kg/m^3
    );

double MrBulkFromD1(double d1) {
  return 2.0 / d1;
}

double MrBulkFromNu(double c10, double c01, double nu) {
  const double shear = 2.0 * (c10 + c01);
  return (2.0 * shear * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu));
}

SolidMaterialProperties MakeFoamMaterial(double c10, double c01, double kappa,
                                         double rho0) {
  return SolidMaterialProperties::MooneyRivlin(c10, c01, kappa, rho0,
                                               kFoamEtaDamp, kFoamLambdaDamp);
}

FoamMaterialPreset GetFoamMaterialPreset(FoamMaterialType type) {
  switch (type) {
    case FoamMaterialType::kNeoprene50A:
      return {"neoprene50a", "Neoprene / chloroprene rubber (50A)",
              MakeFoamMaterial(0.302e6, 0.076e6,
                               MrBulkFromNu(0.302e6, 0.076e6, 0.49), 1350.0)};
    case FoamMaterialType::kPolyurethane50A:
      return {"polyurethane50a", "Polyurethane elastomer (50A)",
              MakeFoamMaterial(0.302e6, 0.076e6,
                               MrBulkFromNu(0.302e6, 0.076e6, 0.499), 2000.0)};
    case FoamMaterialType::kEva80:
      return {"eva80", "EVA foam 80 kg/m^3",
              MakeFoamMaterial(0.417e6, 0.104e6, MrBulkFromD1(0.736e-6), 80.0)};
    case FoamMaterialType::kEva95:
      return {"eva95", "EVA foam 95 kg/m^3",
              MakeFoamMaterial(0.641e6, 0.160e6, MrBulkFromD1(0.478e-6), 95.0)};
    case FoamMaterialType::kNeoprene60A:
      return {"neoprene60a", "Neoprene / chloroprene rubber (60A)",
              MakeFoamMaterial(0.382e6, 0.096e6,
                               MrBulkFromNu(0.382e6, 0.096e6, 0.49), 1400.0)};
  }

  return GetFoamMaterialPreset(FoamMaterialType::kEva95);
}

bool ParseFoamMaterialType(std::string arg, FoamMaterialType* type_out) {
  std::transform(arg.begin(), arg.end(), arg.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (arg == "0" || arg == "neoprene50a" || arg == "neoprene_50a" ||
      arg == "neoprene-50a" || arg == "neoprene" || arg == "chloroprene") {
    *type_out = FoamMaterialType::kNeoprene50A;
    return true;
  }
  if (arg == "1" || arg == "polyurethane50a" || arg == "polyurethane_50a" ||
      arg == "polyurethane-50a" || arg == "polyurethane" || arg == "pu") {
    *type_out = FoamMaterialType::kPolyurethane50A;
    return true;
  }
  if (arg == "2" || arg == "eva80" || arg == "eva_80" || arg == "eva-80") {
    *type_out = FoamMaterialType::kEva80;
    return true;
  }
  if (arg == "3" || arg == "eva95" || arg == "eva_95" || arg == "eva-95") {
    *type_out = FoamMaterialType::kEva95;
    return true;
  }
  if (arg == "4" || arg == "neoprene60a" || arg == "neoprene_60a" ||
      arg == "neoprene-60a" || arg == "chloroprene60a" ||
      arg == "chloroprene_60a" || arg == "chloroprene-60a") {
    *type_out = FoamMaterialType::kNeoprene60A;
    return true;
  }
  return false;
}

void PrintUsage(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [mu_s] [mu_k] [self_collision] [steps] [export_interval]"
               " --mat <neoprene50a|neoprene60a|polyurethane50a|eva80|eva95>\n"
            << "       " << argv0
            << " --mat=<neoprene50a|neoprene60a|polyurethane50a|eva80|eva95>\n"
            << "\n"
            << "Examples:\n"
            << "  " << argv0 << " --mat polyurethane50a\n"
            << "  " << argv0 << " 0.6 0.5 0 4000 20 --mat eva80\n";
}

Eigen::MatrixXd ExtractLocalNodes(const Eigen::MatrixXd& global_nodes,
                                  const ANCFCPUUtils::MeshInstance& inst) {
  return global_nodes.middleRows(inst.node_offset, inst.num_nodes);
}

Eigen::MatrixXi ExtractLocalElements(const Eigen::MatrixXi& global_elements,
                                     const ANCFCPUUtils::MeshInstance& inst) {
  Eigen::MatrixXi local =
      global_elements.middleRows(inst.element_offset, inst.num_elements);
  local.array() -= inst.node_offset;
  return local;
}

Eigen::VectorXd ExtractAxis(const Eigen::MatrixXd& nodes, int axis) {
  Eigen::VectorXd v(nodes.rows());
  for (int i = 0; i < nodes.rows(); ++i) {
    v(i) = nodes(i, axis);
  }
  return v;
}

std::vector<double> LumpedMassFromFeat10(GPU_FEAT10_Data& data, int n_nodes) {
  std::vector<int> offsets;
  std::vector<int> columns;
  std::vector<double> values;
  data.RetrieveMassCSRToCPU(offsets, columns, values);

  std::vector<double> lump(static_cast<size_t>(n_nodes), 0.0);
  if (static_cast<int>(offsets.size()) != n_nodes + 1) {
    std::cerr
        << "Warning: unexpected FEAT10 mass CSR size; using unit masses.\n";
    std::fill(lump.begin(), lump.end(), 1.0);
    return lump;
  }

  for (int i = 0; i < n_nodes; ++i) {
    for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
      lump[static_cast<size_t>(i)] += values[static_cast<size_t>(k)];
    }
  }
  return lump;
}

void CheckCublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error (" << what << "): status=" << int(status)
              << "\n";
    std::exit(1);
  }
}

}  // namespace

// ============================================================================
// Simulation parameters
// ============================================================================
static constexpr double gravity        = -9.81;
static constexpr double dt             = 1e-4;
static constexpr int static_pre_steps  = 1000;
static constexpr int shake_steps       = 4000;
static constexpr int static_post_steps = 1000;
static constexpr int num_steps =
    static_pre_steps + shake_steps + static_post_steps;
static constexpr int export_interval = 10;

// Shaking (prescribed motion on fixed foam nodes)
static constexpr int shake_start_step = static_pre_steps;
static constexpr int shake_end_step   = static_pre_steps + shake_steps;
static constexpr double shake_amp_x   = 0.005;
static constexpr double shake_speed_x = 0.2;

// DEME contact parameters
static constexpr double contact_mu_s                = 0.6;
static constexpr double contact_mu_k                = 0.5;
static constexpr double contact_stiffness           = 8e6;
static constexpr double contact_cor                 = 0.1;
static constexpr double contact_force_clamp_default = 5e4;
static constexpr int contact_force_knn_default      = 8;

int main(int argc, char** argv) {
  std::cout << "========================================\n";
  std::cout << "Vase Drop Onto Protective Foam\n";
  std::cout << "========================================\n";

  // Optional positional args:
  //   [mu_s] [mu_k] [self_collision] [steps] [export_interval]
  //
  // Optional named arg:
  //   --mat <foam_material>
  //   --mat=<foam_material>
  double mu_s                = contact_mu_s;
  double mu_k                = contact_mu_k;
  bool self_collision        = false;
  int max_steps              = num_steps;
  int exp_interval           = export_interval;
  FoamMaterialType foam_type = FoamMaterialType::kEva95;

  std::vector<std::string> positional_args;
  positional_args.reserve(static_cast<size_t>(argc > 1 ? argc - 1 : 0));

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
    if (arg == "--mat") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value after --mat.\n";
        PrintUsage(argv[0]);
        return 1;
      }
      if (!ParseFoamMaterialType(argv[++i], &foam_type)) {
        std::cerr << "Unknown foam material '" << argv[i]
                  << "'. Expected one of: neoprene50a, neoprene60a, "
                     "polyurethane50a, eva80, eva95.\n";
        return 1;
      }
      continue;
    }
    if (arg.rfind("--mat=", 0) == 0) {
      const std::string value = arg.substr(6);
      if (!ParseFoamMaterialType(value, &foam_type)) {
        std::cerr << "Unknown foam material '" << value
                  << "'. Expected one of: neoprene50a, neoprene60a, "
                     "polyurethane50a, eva80, eva95.\n";
        return 1;
      }
      continue;
    }
    if (!arg.empty() && arg[0] == '-') {
      std::cerr << "Unknown option '" << arg << "'.\n";
      PrintUsage(argv[0]);
      return 1;
    }
    positional_args.push_back(arg);
  }

  if (positional_args.size() > 0)
    mu_s = std::atof(positional_args[0].c_str());
  if (positional_args.size() > 1)
    mu_k = std::atof(positional_args[1].c_str());
  if (positional_args.size() > 2)
    self_collision = (std::atoi(positional_args[2].c_str()) != 0);
  if (positional_args.size() > 3) {
    const int v = std::atoi(positional_args[3].c_str());
    if (v > 0)
      max_steps = v;
  }
  if (positional_args.size() > 4)
    exp_interval = std::atoi(positional_args[4].c_str());

  if (positional_args.size() > 5) {
    std::cerr << "Unexpected extra positional argument '" << positional_args[5]
              << "'. Foam material must be selected with --mat.\n";
    PrintUsage(argv[0]);
    return 1;
  }

  const FoamMaterialPreset foam_preset = GetFoamMaterialPreset(foam_type);
  setenv("DEME_FORCE_CLAMP",
         std::to_string(contact_force_clamp_default).c_str(), 1);
  setenv("DEME_FORCE_DISTRIB_K",
         std::to_string(contact_force_knn_default).c_str(), 1);
  setenv("DEME_CONTACT_E", std::to_string(contact_stiffness).c_str(), 1);
  setenv("DEME_CONTACT_COR", std::to_string(contact_cor).c_str(), 1);

  std::cout << "Vase material:    SVK\n";
  std::cout << "Foam material:    " << foam_preset.label
            << " (Mooney-Rivlin)\n";
  std::cout << "dt:               " << dt << " s\n";
  std::cout << "mu_s:             " << mu_s << "\n";
  std::cout << "mu_k:             " << mu_k << "\n";
  std::cout << "self_collision:   " << (self_collision ? "yes" : "no") << "\n";
  std::cout << "max_steps:        " << max_steps << "\n";
  std::cout << "export_interval:  " << exp_interval << "\n";
  std::cout << "contact_E:        " << contact_stiffness << " Pa\n";
  std::cout << "contact_CoR:      " << contact_cor << "\n";
  std::cout << "contact_clamp:    " << contact_force_clamp_default << " N\n";
  std::cout << "contact_knn:      " << contact_force_knn_default << "\n";
  std::cout << "foam mu10:        " << foam_preset.material.mu10 << " Pa\n";
  std::cout << "foam mu01:        " << foam_preset.material.mu01 << " Pa\n";
  std::cout << "foam kappa:       " << foam_preset.material.kappa << " Pa\n";
  std::cout << "foam rho0:        " << foam_preset.material.rho0
            << " kg/m^3\n\n";

  std::filesystem::create_directories("output/vase_drop");

  // =========================================================================
  // Load meshes: foam first, then vase
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

  mesh_manager.TranslateMesh(mesh_vase, 0.0, 0.025, 0.01);

  const auto& inst_foam               = mesh_manager.GetMeshInstance(mesh_foam);
  const auto& inst_vase               = mesh_manager.GetMeshInstance(mesh_vase);
  const Eigen::MatrixXd& all_nodes    = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& all_elements = mesh_manager.GetAllElements();

  std::cout << "Loaded meshes:\n";
  std::cout << "  Foam: " << inst_foam.num_nodes << " nodes, "
            << inst_foam.num_elements << " elements\n";
  std::cout << "  Vase: " << inst_vase.num_nodes << " nodes, "
            << inst_vase.num_elements << " elements\n";
  std::cout << "  Vase translation: (+x=0, +y=0.025, +z=0.01) m\n\n";

  const Eigen::MatrixXd foam_nodes = ExtractLocalNodes(all_nodes, inst_foam);
  const Eigen::MatrixXi foam_elements =
      ExtractLocalElements(all_elements, inst_foam);
  const Eigen::MatrixXd vase_nodes = ExtractLocalNodes(all_nodes, inst_vase);
  const Eigen::MatrixXi vase_elements =
      ExtractLocalElements(all_elements, inst_vase);

  auto foam_data = std::make_unique<GPU_FEAT10_Data>(inst_foam.num_elements,
                                                     inst_foam.num_nodes);
  auto vase_data = std::make_unique<GPU_FEAT10_Data>(inst_vase.num_elements,
                                                     inst_vase.num_nodes);
  foam_data->Initialize();
  vase_data->Initialize();

  const Eigen::VectorXd foam_x = ExtractAxis(foam_nodes, 0);
  const Eigen::VectorXd foam_y = ExtractAxis(foam_nodes, 1);
  const Eigen::VectorXd foam_z = ExtractAxis(foam_nodes, 2);
  const Eigen::VectorXd vase_x = ExtractAxis(vase_nodes, 0);
  const Eigen::VectorXd vase_y = ExtractAxis(vase_nodes, 1);
  const Eigen::VectorXd vase_z = ExtractAxis(vase_nodes, 2);

  // Clamp foam nodes below Z = -0.05 m.
  static constexpr double fix_z_threshold = -0.09;
  std::vector<int> fixed_indices;
  fixed_indices.reserve(inst_foam.num_nodes / 2);
  for (int i = 0; i < inst_foam.num_nodes; ++i) {
    if (foam_nodes(i, 2) < fix_z_threshold) {
      fixed_indices.push_back(i);
    }
  }

  Eigen::VectorXi h_fixed(static_cast<int>(fixed_indices.size()));
  for (int i = 0; i < static_cast<int>(fixed_indices.size()); ++i) {
    h_fixed(i) = fixed_indices[static_cast<size_t>(i)];
  }
  foam_data->SetNodalFixed(h_fixed);
  std::cout << "Fixed " << h_fixed.size() << " foam nodes (Z < "
            << fix_z_threshold << " m)\n";

  int* d_fixed_node_ids = nullptr;
  const int n_fixed     = static_cast<int>(fixed_indices.size());
  if (n_fixed > 0) {
    HANDLE_ERROR(cudaMalloc(&d_fixed_node_ids, n_fixed * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_fixed_node_ids, fixed_indices.data(),
                            n_fixed * sizeof(int), cudaMemcpyHostToDevice));
  }

  // =========================================================================
  // FE setup
  // =========================================================================
  foam_data->Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                   Quadrature::tet5pt_z, Quadrature::tet5pt_weights, foam_x,
                   foam_y, foam_z, foam_elements);
  vase_data->Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                   Quadrature::tet5pt_z, Quadrature::tet5pt_weights, vase_x,
                   vase_y, vase_z, vase_elements);

  foam_data->ApplyMaterial(foam_preset.material);
  vase_data->ApplyMaterial(kVaseMaterial);

  foam_data->CalcDnDuPre();
  foam_data->CalcMassMatrix();
  foam_data->CalcConstraintData();
  foam_data->ConvertToCSR_ConstraintJacT();
  foam_data->BuildConstraintJacobianCSR();

  vase_data->CalcDnDuPre();
  vase_data->CalcMassMatrix();
  vase_data->CalcConstraintData();
  vase_data->ConvertToCSR_ConstraintJacT();
  vase_data->BuildConstraintJacobianCSR();

  const std::vector<double> foam_lumped_mass =
      LumpedMassFromFeat10(*foam_data, inst_foam.num_nodes);
  const std::vector<double> vase_lumped_mass =
      LumpedMassFromFeat10(*vase_data, inst_vase.num_nodes);
  const double foam_total_mass =
      std::accumulate(foam_lumped_mass.begin(), foam_lumped_mass.end(), 0.0);
  const double vase_total_mass =
      std::accumulate(vase_lumped_mass.begin(), vase_lumped_mass.end(), 0.0);

  std::cout << "Foam block: mass=" << foam_total_mass << " kg\n";
  std::cout << "Vase block: mass=" << vase_total_mass << " kg\n\n";

  // =========================================================================
  // Multi-block FE solve setup
  // =========================================================================
  FEMultiElementProblem problem;
  const int block_foam = problem.AddElementBlock(foam_data.get(), TYPE_T10);
  const int block_vase = problem.AddElementBlock(vase_data.get(), TYPE_T10);
  problem.Finalize();
  problem.SyncPositionsFromElements();
  problem.UpdateCollisionNodeBuffer();

  MultiElementNewtonSolver solver(&problem);
  MultiElementNewtonParams params;
  params.inner_atol         = 1e-3;
  params.inner_rtol         = 1e-4;
  params.outer_tol          = 1e-6;
  params.enable_line_search = true;
  params.rho                = 1e12;
  params.max_outer          = 5;
  params.max_inner          = 10;
  params.time_step          = dt;
  solver.SetParameters(&params);
  solver.Setup();

  FEStateBuffer& state = problem.GetStateBuffer();

  // =========================================================================
  // DEME mesh-mesh collision system
  // =========================================================================
  ANCFCPUUtils::SurfaceTriMesh foam_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elements, inst_foam);
  ANCFCPUUtils::SurfaceTriMesh vase_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elements, inst_vase);

  std::vector<DemeMeshCollisionBody> bodies;
  {
    DemeMeshCollisionBody body;
    body.surface                  = std::move(foam_surface);
    body.family                   = 0;
    body.split_into_patches       = true;
    body.skip_self_contact_forces = false;
    body.mass                     = static_cast<float>(foam_total_mass);
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(vase_surface);
    body.family             = 1;
    body.split_into_patches = true;
    body.patch_angle_deg    = -1.0f;
    body.mass               = static_cast<float>(vase_total_mass);
    bodies.push_back(std::move(body));
  }

  auto collision_system = std::make_unique<DemeMeshCollisionSystem>(
      std::move(bodies), mu_s, mu_k, contact_stiffness, contact_cor,
      self_collision, dt);
  collision_system->BindNodesDevicePtr(state.d_nodes_collision,
                                       state.total_coef);

  // =========================================================================
  // Gravity
  // =========================================================================
  const int total_dofs        = problem.GetTotalDofs();
  Eigen::VectorXd h_f_gravity = Eigen::VectorXd::Zero(total_dofs);

  const int foam_coef_off = state.blocks[block_foam].coef_offset;
  const int vase_coef_off = state.blocks[block_vase].coef_offset;

  for (int node = 0; node < inst_foam.num_nodes; ++node) {
    if (foam_nodes(node, 2) < fix_z_threshold) {
      continue;
    }
    const int coef = foam_coef_off + node;
    h_f_gravity(3 * coef + 2) +=
        foam_lumped_mass[static_cast<size_t>(node)] * gravity;
  }
  for (int node = 0; node < inst_vase.num_nodes; ++node) {
    const int coef = vase_coef_off + node;
    h_f_gravity(3 * coef + 2) +=
        vase_lumped_mass[static_cast<size_t>(node)] * gravity;
  }

  double* d_f_gravity = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_f_gravity, total_dofs * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_f_gravity, h_f_gravity.data(),
                          total_dofs * sizeof(double), cudaMemcpyHostToDevice));

  cublasHandle_t cublas_handle = nullptr;
  CheckCublas(cublasCreate(&cublas_handle), "cublasCreate");

  // =========================================================================
  // Simulation loop
  // =========================================================================
  std::cout << "Starting simulation (" << max_steps << " steps, dt=" << dt
            << " s)\n\n";

  double shake_dx_prev = 0.0;

  for (int step = 0; step < max_steps; ++step) {
    const auto t0 = std::chrono::high_resolution_clock::now();

    if (n_fixed > 0 && step >= shake_start_step && step < shake_end_step) {
      const double t_shake = (step - shake_start_step) * dt;
      const double t1      = shake_amp_x / shake_speed_x;
      const double period  = 4.0 * shake_amp_x / shake_speed_x;
      const double phase   = std::fmod(t_shake, period);

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
          foam_data->GetX12DevicePtr(), foam_data->GetX12JacDevicePtr(),
          d_fixed_node_ids, n_fixed, delta_dx);
      shake_dx_prev = dx;

      // Propagate prescribed-motion changes to the unified state buffer before
      // collision and solve.
      problem.SyncPositionsFromElements();
    }

    problem.UpdateCollisionNodeBuffer();

    CollisionSystemInput coll_in;
    coll_in.d_nodes_xyz = state.d_nodes_collision;
    coll_in.n_nodes     = state.total_coef;
    coll_in.d_vel_xyz   = nullptr;
    coll_in.dt          = dt;

    // DemeMeshCollisionSystem currently takes its actual contact material from
    // construction-time parameters / env overrides, not from per-step params.
    CollisionSystemParams coll_params;
    collision_system->Step(coll_in, coll_params);
    const int num_contacts = collision_system->GetNumContacts();

    double* d_f_ext = solver.GetExternalForceDevicePtr();
    HANDLE_ERROR(cudaMemcpy(d_f_ext, d_f_gravity, total_dofs * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    if (num_contacts > 0) {
      const double alpha = 1.0;
      CheckCublas(cublasDaxpy(cublas_handle, total_dofs, &alpha,
                              collision_system->GetExternalForcesDevicePtr(), 1,
                              d_f_ext, 1),
                  "cublasDaxpy(contact + gravity)");
    }

    solver.Solve();

    if (exp_interval > 0 && step % exp_interval == 0) {
      std::ostringstream foam_fn;
      foam_fn << "output/vase_drop/foam_" << std::setfill('0') << std::setw(4)
              << step << ".vtu";
      foam_data->WriteOutputVTU(foam_fn.str());

      std::ostringstream vase_fn;
      vase_fn << "output/vase_drop/vase_" << std::setfill('0') << std::setw(4)
              << step << ".vtu";
      vase_data->WriteOutputVTU(vase_fn.str());
    }

    if (step % 20 == 0) {
      const auto t1 = std::chrono::high_resolution_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      std::cout << "Step " << std::setw(4) << step
                << "  contacts=" << std::setw(5) << num_contacts
                << "  ms=" << std::fixed << std::setprecision(2) << ms << "\n";
    }
  }

  CheckCublas(cublasDestroy(cublas_handle), "cublasDestroy");
  HANDLE_ERROR(cudaFree(d_f_gravity));
  if (d_fixed_node_ids != nullptr) {
    HANDLE_ERROR(cudaFree(d_fixed_node_ids));
  }

  foam_data->Destroy();
  vase_data->Destroy();

  std::cout << "\n========================================\n";
  std::cout << "Simulation complete.\n";
  std::cout << "Output: output/vase_drop/foam_XXXX.vtu\n";
  std::cout << "        output/vase_drop/vase_XXXX.vtu\n";
  std::cout << "========================================\n";
  return 0;
}
