/**
 * Multi-Item Drop Onto ANCF3443 Beam
 *
 * Scenario:
 *   - Thirteen FEAT10 (T10) deformable items represented in one unified FE
 *     system
 *   - One deformable ANCF3443 beam with the x=0 edge fixed
 *   - A fixed FEAT10 openbox acts as a surrounding collision obstacle
 *   - Contact handled by DEME mesh-mesh collision, coupled via external forces
 *   - Time integration via MultiElementNewtonSolver
 *
 * Output:
 *   output/multiitem_drop_ltest/beam_XXXXXX.vtu
 *   output/multiitem_drop_ltest/teapot_XXXXXX.vtu
 *   output/multiitem_drop_ltest/tire_XXXXXX.vtu
 *   output/multiitem_drop_ltest/openbox_XXXXXX.vtu
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../lib_src/collision/DemeMeshCollisionSystem.h"
#include "../../lib_src/elements/ANCF3443Data.cuh"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEAT10ConstraintManager.h"
#include "../../lib_src/solvers/FEMultiElementProblem.h"
#include "../../lib_src/solvers/MultiElementNewton.cuh"
#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../../lib_utils/rigid_mass_properties.h"
#include "../../lib_utils/surface_trimesh_extract.h"
#include "../../lib_utils/visualization_utils.h"
#include "multiitem_collision_kernels.h"

namespace {

struct BBox {
  Eigen::Vector3d mn =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d mx =
      Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());

  Eigen::Vector3d size() const {
    return mx - mn;
  }
  Eigen::Vector3d center() const {
    return 0.5 * (mn + mx);
  }
};

BBox ComputeBBox(const Eigen::MatrixXd& nodes,
                 const ANCFCPUUtils::MeshInstance& inst) {
  BBox bb;
  for (int i = 0; i < inst.num_nodes; ++i) {
    const int idx = inst.node_offset + i;
    bb.mn(0)      = std::min(bb.mn(0), nodes(idx, 0));
    bb.mn(1)      = std::min(bb.mn(1), nodes(idx, 1));
    bb.mn(2)      = std::min(bb.mn(2), nodes(idx, 2));
    bb.mx(0)      = std::max(bb.mx(0), nodes(idx, 0));
    bb.mx(1)      = std::max(bb.mx(1), nodes(idx, 1));
    bb.mx(2)      = std::max(bb.mx(2), nodes(idx, 2));
  }
  return bb;
}

void PrintBBox(const std::string& name, const BBox& bb) {
  std::cout << "  " << std::left << std::setw(8) << name << " mn=[" << bb.mn(0)
            << ", " << bb.mn(1) << ", " << bb.mn(2) << "] mx=[" << bb.mx(0)
            << ", " << bb.mx(1) << ", " << bb.mx(2) << "] size=["
            << bb.size()(0) << ", " << bb.size()(1) << ", " << bb.size()(2)
            << "]\n";
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
    double sum = 0.0;
    for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
      sum += values[static_cast<size_t>(k)];
    }
    lump[static_cast<size_t>(i)] = sum;
  }
  return lump;
}

std::vector<double> LumpedMassFromAncf3443(GPU_ANCF3443_Data& data,
                                           int n_coef) {
  std::vector<int> offsets;
  std::vector<int> columns;
  std::vector<double> values;
  data.RetrieveMassCSRToCPU(offsets, columns, values);
  std::vector<double> lump(static_cast<size_t>(n_coef), 0.0);
  if (static_cast<int>(offsets.size()) != n_coef + 1) {
    std::cerr
        << "Warning: unexpected ANCF3443 mass CSR size; using unit masses.\n";
    std::fill(lump.begin(), lump.end(), 1.0);
    return lump;
  }
  for (int i = 0; i < n_coef; ++i) {
    double sum = 0.0;
    for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
      sum += values[static_cast<size_t>(k)];
    }
    lump[static_cast<size_t>(i)] = sum;
  }
  return lump;
}

BBox ComputeBBoxAncf3443(const Eigen::VectorXd& x12, const Eigen::VectorXd& y12,
                         const Eigen::VectorXd& z12, int n_nodes) {
  BBox bb;
  for (int node = 0; node < n_nodes; ++node) {
    const int base = node * 4;
    bb.mn(0)       = std::min(bb.mn(0), x12(base + 0));
    bb.mn(1)       = std::min(bb.mn(1), y12(base + 0));
    bb.mn(2)       = std::min(bb.mn(2), z12(base + 0));
    bb.mx(0)       = std::max(bb.mx(0), x12(base + 0));
    bb.mx(1)       = std::max(bb.mx(1), y12(base + 0));
    bb.mx(2)       = std::max(bb.mx(2), z12(base + 0));
  }
  return bb;
}

ANCFCPUUtils::SurfaceTriMesh BuildClosedShellCollisionSurface(
    const Eigen::VectorXd& x12, const Eigen::VectorXd& y12,
    const Eigen::VectorXd& z12, const Eigen::MatrixXi& conn, int n_nodes,
    int n_elems, double thickness) {
  ANCFCPUUtils::SurfaceTriMesh surface;
  surface.global_node_ids.resize(static_cast<size_t>(2 * n_nodes));
  surface.vertices.resize(static_cast<size_t>(2 * n_nodes));
  surface.ancf_node_ids.resize(static_cast<size_t>(2 * n_nodes));
  for (int node = 0; node < n_nodes; ++node) {
    const int base = node * 4;
    const Eigen::Vector3d p(x12(base + 0), y12(base + 0), z12(base + 0));
    const int bot                                     = node;
    const int top                                     = n_nodes + node;
    surface.global_node_ids[static_cast<size_t>(bot)] = bot;
    surface.global_node_ids[static_cast<size_t>(top)] = top;
    surface.ancf_node_ids[static_cast<size_t>(bot)]   = node;
    surface.ancf_node_ids[static_cast<size_t>(top)]   = node;
    surface.vertices[static_cast<size_t>(bot)] =
        p - Eigen::Vector3d(0, 0, 0.5 * thickness);
    surface.vertices[static_cast<size_t>(top)] =
        p + Eigen::Vector3d(0, 0, 0.5 * thickness);
  }

  surface.triangles.reserve(static_cast<size_t>(n_elems) * 6);
  for (int e = 0; e < n_elems; ++e) {
    const int n0 = conn(e, 0);
    const int n1 = conn(e, 1);
    const int n2 = conn(e, 2);
    const int n3 = conn(e, 3);
    const int t0 = n_nodes + n0;
    const int t1 = n_nodes + n1;
    const int t2 = n_nodes + n2;
    const int t3 = n_nodes + n3;
    surface.triangles.emplace_back(t0, t1, t2);
    surface.triangles.emplace_back(t0, t2, t3);
    surface.triangles.emplace_back(n0, n2, n1);
    surface.triangles.emplace_back(n0, n3, n2);
  }

  struct EdgeInfo {
    int count = 0;
    int a_dir = -1;
    int b_dir = -1;
  };
  struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const noexcept {
      return (static_cast<size_t>(p.first) << 32) ^
             static_cast<size_t>(p.second);
    }
  };

  std::unordered_map<std::pair<int, int>, EdgeInfo, PairHash> edge_counts;
  edge_counts.reserve(static_cast<size_t>(n_elems) * 4);
  auto add_edge = [&](int a, int b) {
    const std::pair<int, int> key =
        (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
    EdgeInfo& info = edge_counts[key];
    info.count += 1;
    if (info.count == 1) {
      info.a_dir = a;
      info.b_dir = b;
    }
  };
  for (int e = 0; e < n_elems; ++e) {
    const int n0 = conn(e, 0);
    const int n1 = conn(e, 1);
    const int n2 = conn(e, 2);
    const int n3 = conn(e, 3);
    add_edge(n0, n1);
    add_edge(n1, n2);
    add_edge(n2, n3);
    add_edge(n3, n0);
  }
  for (const auto& kv : edge_counts) {
    const EdgeInfo& info = kv.second;
    if (info.count != 1) {
      continue;
    }
    const int a     = info.a_dir;
    const int b     = info.b_dir;
    const int a_bot = a;
    const int b_bot = b;
    const int a_top = n_nodes + a;
    const int b_top = n_nodes + b;
    surface.triangles.emplace_back(a_bot, b_bot, b_top);
    surface.triangles.emplace_back(a_bot, b_top, a_top);
  }

  return surface;
}

}  // namespace

int main(int argc, char** argv) {
  int device_count          = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&device_count);
  if (dev_err != cudaSuccess || device_count <= 0) {
    std::cerr << "No CUDA device visible (cudaGetDeviceCount: "
              << cudaGetErrorString(dev_err) << ", count=" << device_count
              << ")\n";
    return 1;
  }
  HANDLE_ERROR(cudaSetDevice(0));

  std::cout << "========================================\n";
  std::cout << "Multi-Item Drop LTest (unified T10 system) "
               "(9 tires + openbox) onto "
               "ANCF3443 beam\n";
  std::cout << "========================================\n";

  bool split_items_into_patches = false;
  bool write_csv                = false;
  std::string csv_path          = "output/multiitem_drop_ltest/diagnostics.csv";
  double softness               = 1.0;
  for (int ai = 1; ai < argc; ++ai) {
    const std::string arg(argv[ai]);
    if (arg == "--split_patches") {
      split_items_into_patches = true;
    } else if (arg == "--csv") {
      write_csv = true;
      if (ai + 1 < argc) {
        const std::string next_arg(argv[ai + 1]);
        if (next_arg.rfind("--", 0) != 0) {
          csv_path = next_arg;
          ++ai;
        }
      }
    } else if (arg.rfind("--csv=", 0) == 0) {
      write_csv = true;
      csv_path  = arg.substr(6);
    } else if (arg == "--softness") {
      if (ai + 1 >= argc) {
        std::cerr << "Missing value after --softness\n";
        return 1;
      }
      softness = std::stod(argv[++ai]);
    } else if (arg.rfind("--softness=", 0) == 0) {
      softness = std::stod(arg.substr(11));
    }
  }
  if (!(softness > 0.0)) {
    std::cerr << "--softness must be > 0\n";
    return 1;
  }

  // -------------------------------------------------------------------------
  // Parameters
  // -------------------------------------------------------------------------
  constexpr double gravity = -9.81;
  constexpr double dt      = 1e-4;
  constexpr int steps      = 50000;
  constexpr int vtu_every  = 50;

  constexpr double beam_x_margin = 0.15;
  constexpr double beam_x        = 0.40;
  constexpr double beam_y        = 0.4;
  constexpr double beam_h        = 0.02;
  constexpr int beam_nx          = 30;
  constexpr int beam_ny          = 12;

  constexpr double tire_scale = 0.12;

  SolidMaterialProperties mat_teapot =
      SolidMaterialProperties::SVK(2.0e6, 0.35, 500.0, 1.0e4, 1.0e4);
  SolidMaterialProperties mat_tire =
      SolidMaterialProperties::SVK(5.0e6, 0.4, 900.0, 2.0e4, 2.0e4);
#if 0  // Debug-disabled items: bunnies and armadilo.
  SolidMaterialProperties mat_bunny =
      SolidMaterialProperties::SVK(3.0e6, 0.45, 300.0, 1.0e4, 1.0e4);
  SolidMaterialProperties mat_armadilo =
      SolidMaterialProperties::SVK(2.0e6, 0.35, 200.0, 1.0e4, 1.0e4);
#endif
  if (softness != 1.0) {
    mat_teapot.E /= softness;
    mat_tire.E /= softness;
#if 0  // Debug-disabled items: bunnies and armadilo.
    mat_bunny.E /= softness;
    mat_armadilo.E /= softness;
#endif
  }
  const double beam_E       = 8.0e6 / softness;
  constexpr double beam_nu  = 0.33;
  constexpr double beam_rho = 1200.0;

  constexpr double mu_s         = 0.6;
  constexpr double mu_k         = 0.5;
  constexpr double contact_E    = 1e7;
  constexpr double contact_cor  = 0.5;
  constexpr bool self_collision = false;

  if (std::getenv("DEME_FORCE_CLAMP") == nullptr) {
    setenv("DEME_FORCE_CLAMP", "50000", 1);
  }
  if (std::getenv("DEME_FORCE_DISTRIB_K") == nullptr) {
    setenv("DEME_FORCE_DISTRIB_K", "8", 1);
  }

  std::filesystem::create_directories("output/multiitem_drop_ltest");
  std::cout << "Item softness factor: " << softness
            << " (item Young's modulus scaled by 1/" << softness << ")\n";

  std::ofstream csv_file;
  if (write_csv) {
    const std::filesystem::path csv_fs_path(csv_path);
    if (!csv_fs_path.parent_path().empty()) {
      std::filesystem::create_directories(csv_fs_path.parent_path());
    }
    csv_file.open(csv_path);
    if (!csv_file) {
      std::cerr << "Failed to open CSV output path: " << csv_path << "\n";
      return 1;
    }
    csv_file << std::setprecision(17);
    csv_file << "step,time,num_contacts,collision_step_ms,solver_wall_ms,"
                "solver_block_ms_sum,solver_converged_blocks,"
                "solver_total_outer_iterations,solver_total_inner_iterations,"
                "solver_residual_norm_sum,solver_constraint_norm_sum,"
                "solver_max_residual_norm,solver_max_constraint_norm,"
                "line_search_calls,line_search_successes,"
                "line_search_backtracks_total,line_search_failures,"
                "line_search_alpha_min,line_search_alpha_avg_accepted\n";
    std::cout << "CSV logging enabled: " << csv_path << "\n";
  }

  // -------------------------------------------------------------------------
  // Load + place FEAT10 meshes using MeshManager
  // -------------------------------------------------------------------------
  ANCFCPUUtils::MeshManager mesh_manager;
  const int mesh_teapot = mesh_manager.LoadMesh(
      "data/meshes/T10/tire.node", "data/meshes/T10/tire.ele", "teapot");
  const int mesh_tire = mesh_manager.LoadMesh(
      "data/meshes/T10/tire.node", "data/meshes/T10/tire.ele", "tire");
  const int mesh_tire2 = mesh_manager.LoadMesh(
      "data/meshes/T10/tire.node", "data/meshes/T10/tire.ele", "tire2");
  const int mesh_tire3 = mesh_manager.LoadMesh(
      "data/meshes/T10/tire.node", "data/meshes/T10/tire.ele", "tire3");
  const int mesh_tire4 = mesh_manager.LoadMesh(
      "data/meshes/T10/tire.node", "data/meshes/T10/tire.ele", "tire4");
  const int mesh_tire5 = mesh_manager.LoadMesh(
      "data/meshes/T10/tire.node", "data/meshes/T10/tire.ele", "tire5");
  const int mesh_tire6 = mesh_manager.LoadMesh(
      "data/meshes/T10/tire.node", "data/meshes/T10/tire.ele", "tire6");
  const int mesh_tire7 = mesh_manager.LoadMesh(
      "data/meshes/T10/tire.node", "data/meshes/T10/tire.ele", "tire7");
#if 0  // Debug-disabled items: bunnies.
  const int mesh_bunny = mesh_manager.LoadMesh(
      "data/meshes/T10/bubble_gripper_bunny/bunny_26_scaled_0p01.1.node",
      "data/meshes/T10/bubble_gripper_bunny/bunny_26_scaled_0p01.1.ele",
      "bunny");
  const int mesh_bunny2 = mesh_manager.LoadMesh(
      "data/meshes/T10/bubble_gripper_bunny/bunny_26_scaled_0p01.1.node",
      "data/meshes/T10/bubble_gripper_bunny/bunny_26_scaled_0p01.1.ele",
      "bunny2");
  const int mesh_bunny3 = mesh_manager.LoadMesh(
      "data/meshes/T10/bubble_gripper_bunny/bunny_26_scaled_0p01.1.node",
      "data/meshes/T10/bubble_gripper_bunny/bunny_26_scaled_0p01.1.ele",
      "bunny3");
#endif
  const int mesh_openbox =
      mesh_manager.LoadMesh("data/meshes/T10/item_drop/openbox.node",
                            "data/meshes/T10/item_drop/openbox.ele", "openbox");
#if 0  // Debug-disabled item: armadilo.
  const int mesh_armadilo = mesh_manager.LoadMesh(
      "data/meshes/T10/item_drop/armadilo.node",
      "data/meshes/T10/item_drop/armadilo.ele", "armadilo");
#endif
  const int mesh_tire8 = mesh_manager.LoadMesh(
      "data/meshes/T10/tire.node", "data/meshes/T10/tire.ele", "tire8");
  if (mesh_teapot < 0 || mesh_tire < 0 || mesh_tire2 < 0 || mesh_tire3 < 0 ||
      mesh_tire4 < 0 || mesh_tire5 < 0 || mesh_tire6 < 0 || mesh_tire7 < 0 ||
      mesh_tire8 < 0 || mesh_openbox < 0) {
    std::cerr << "Failed to load "
                 "active tire meshes/openbox.\n";
    return 1;
  }

  const auto& inst_teapot = mesh_manager.GetMeshInstance(mesh_teapot);
  const auto& inst_tire   = mesh_manager.GetMeshInstance(mesh_tire);
  const auto& inst_tire2  = mesh_manager.GetMeshInstance(mesh_tire2);
  const auto& inst_tire3  = mesh_manager.GetMeshInstance(mesh_tire3);
  const auto& inst_tire4  = mesh_manager.GetMeshInstance(mesh_tire4);
  const auto& inst_tire5  = mesh_manager.GetMeshInstance(mesh_tire5);
  const auto& inst_tire6  = mesh_manager.GetMeshInstance(mesh_tire6);
  const auto& inst_tire7  = mesh_manager.GetMeshInstance(mesh_tire7);
  const auto& inst_tire8  = mesh_manager.GetMeshInstance(mesh_tire8);
#if 0  // Debug-disabled items: bunnies.
  const auto& inst_bunny    = mesh_manager.GetMeshInstance(mesh_bunny);
  const auto& inst_bunny2   = mesh_manager.GetMeshInstance(mesh_bunny2);
  const auto& inst_bunny3   = mesh_manager.GetMeshInstance(mesh_bunny3);
#endif
  const auto& inst_openbox = mesh_manager.GetMeshInstance(mesh_openbox);
#if 0  // Debug-disabled item: armadilo.
  const auto& inst_armadilo = mesh_manager.GetMeshInstance(mesh_armadilo);
#endif

  mesh_manager.TransformMesh(mesh_teapot,
                             ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_tire, ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_tire2,
                             ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_tire3,
                             ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_tire4,
                             ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_tire5,
                             ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_tire6,
                             ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_tire7,
                             ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_tire8,
                             ANCFCPUUtils::uniformScale(tire_scale));
#if 0  // Debug-disabled items: bunnies.
#endif
  mesh_manager.TransformMesh(mesh_openbox, ANCFCPUUtils::uniformScale(0.4));
#if 0  // Debug-disabled item: armadilo.
  mesh_manager.TransformMesh(mesh_armadilo, ANCFCPUUtils::uniformScale(0.3));
  mesh_manager.TransformMesh(
      mesh_armadilo,
      ANCFCPUUtils::rotationY((160.0 / 180.0) * std::acos(-1.0)));
  {
    Eigen::Matrix4d armadilo_Rz = Eigen::Matrix4d::Identity();
    const double angle          = std::acos(-1.0);
    const double c              = std::cos(angle);
    const double s              = std::sin(angle);
    armadilo_Rz(0, 0)           = c;
    armadilo_Rz(0, 1)           = -s;
    armadilo_Rz(1, 0)           = s;
    armadilo_Rz(1, 1)           = c;
    mesh_manager.TransformMesh(mesh_armadilo, armadilo_Rz);
  }
#endif

  Eigen::Matrix4d tire_Rz = Eigen::Matrix4d::Identity();
  {
    const double angle = 0.5 * std::acos(-1.0);
    const double c     = std::cos(angle);
    const double s     = std::sin(angle);
    tire_Rz(0, 0)      = c;
    tire_Rz(0, 1)      = -s;
    tire_Rz(1, 0)      = s;
    tire_Rz(1, 1)      = c;
  }
  mesh_manager.TransformMesh(mesh_tire, tire_Rz);
  mesh_manager.TransformMesh(mesh_tire2, tire_Rz);
  mesh_manager.TransformMesh(mesh_tire3, tire_Rz);
  mesh_manager.TransformMesh(mesh_tire4, tire_Rz);
  mesh_manager.TransformMesh(mesh_tire5, tire_Rz);
  mesh_manager.TransformMesh(mesh_tire6, tire_Rz);
  mesh_manager.TransformMesh(mesh_tire7, tire_Rz);
  mesh_manager.TransformMesh(mesh_tire8, tire_Rz);
#if 0  // Debug-disabled items: bunnies.
#endif
  mesh_manager.TransformMesh(mesh_teapot, tire_Rz);
#if 0  // Debug-disabled items: bunnies.
  mesh_manager.TransformMesh(
      mesh_bunny2, ANCFCPUUtils::rotationX((35.0 / 180.0) * std::acos(-1.0)));
  mesh_manager.TransformMesh(
      mesh_bunny3, ANCFCPUUtils::rotationX((25.0 / 180.0) * std::acos(-1.0)));
#endif

  auto place_mesh = [&](int mesh_id, const ANCFCPUUtils::MeshInstance& inst,
                        const Eigen::Vector2d& center_xy, double bottom_z) {
    const Eigen::MatrixXd& nodes0 = mesh_manager.GetAllNodes();
    const BBox bb0                = ComputeBBox(nodes0, inst);
    const Eigen::Vector3d delta(center_xy(0) - bb0.center()(0),
                                center_xy(1) - bb0.center()(1),
                                bottom_z - bb0.mn(2));
    mesh_manager.TransformMesh(
        mesh_id, ANCFCPUUtils::translation(delta(0), delta(1), delta(2)));
  };

  auto place_mesh_center = [&](int mesh_id,
                               const ANCFCPUUtils::MeshInstance& inst,
                               const Eigen::Vector3d& center_xyz) {
    const Eigen::MatrixXd& nodes0 = mesh_manager.GetAllNodes();
    const BBox bb0                = ComputeBBox(nodes0, inst);
    const Eigen::Vector3d delta   = center_xyz - bb0.center();
    mesh_manager.TransformMesh(
        mesh_id, ANCFCPUUtils::translation(delta(0), delta(1), delta(2)));
  };

  const double teapot_bottom_z = -0.16;
  const double tire_bottom_z   = -0.16;
  place_mesh(mesh_teapot, inst_teapot, Eigen::Vector2d(0.50, 0.15),
             teapot_bottom_z);
  place_mesh(mesh_tire, inst_tire, Eigen::Vector2d(0.50, 0.00), tire_bottom_z);
  place_mesh(mesh_tire2, inst_tire2, Eigen::Vector2d(0.50, -0.15),
             tire_bottom_z);
  const double tire_right_bottom_z = tire_bottom_z - 0.12;
  place_mesh(mesh_tire3, inst_tire3, Eigen::Vector2d(0.73, 0.16),
             tire_right_bottom_z);
  place_mesh(mesh_tire4, inst_tire4, Eigen::Vector2d(0.75, 0.00),
             tire_right_bottom_z);
  place_mesh(mesh_tire5, inst_tire5, Eigen::Vector2d(0.77, -0.16),
             tire_right_bottom_z);
  const double tire_lower_bottom_z = -0.52;
  place_mesh(mesh_tire6, inst_tire6, Eigen::Vector2d(0.73, 0.18),
             tire_lower_bottom_z);
  place_mesh(mesh_tire7, inst_tire7, Eigen::Vector2d(0.75, 0.02),
             tire_lower_bottom_z);
  place_mesh(mesh_tire8, inst_tire8, Eigen::Vector2d(0.77, -0.18),
             tire_lower_bottom_z);
#if 0  // Debug-disabled items: bunnies and armadilo.
  {
    const Eigen::MatrixXd& nodes_now = mesh_manager.GetAllNodes();
    const BBox bb_tire2_now          = ComputeBBox(nodes_now, inst_tire2);
    const double armadilo_bottom_z   = bb_tire2_now.mx(2) - 0.06;
    place_mesh(mesh_armadilo, inst_armadilo,
               Eigen::Vector2d(bb_tire2_now.center()(0),
                               bb_tire2_now.center()(1) - 0.06),
               armadilo_bottom_z);
  }

  {
    const Eigen::MatrixXd& nodes_now = mesh_manager.GetAllNodes();
    const BBox bb_tire_now           = ComputeBBox(nodes_now, inst_tire);
    const double bunny_bottom_z      = bb_tire_now.mx(2);
    place_mesh(
        mesh_bunny, inst_bunny,
        Eigen::Vector2d(bb_tire_now.center()(0), bb_tire_now.center()(1)),
        bunny_bottom_z);
  }
  {
    const Eigen::MatrixXd& nodes_now = mesh_manager.GetAllNodes();
    const BBox bb_tire3_now          = ComputeBBox(nodes_now, inst_tire3);
    const double bunny2_bottom_z     = bb_tire3_now.mx(2);
    place_mesh(
        mesh_bunny2, inst_bunny2,
        Eigen::Vector2d(bb_tire3_now.center()(0), bb_tire3_now.center()(1)),
        bunny2_bottom_z);
  }
  {
    const Eigen::MatrixXd& nodes_now = mesh_manager.GetAllNodes();
    const BBox bb_tire4_now          = ComputeBBox(nodes_now, inst_tire4);
    const double bunny3_bottom_z     = bb_tire4_now.mx(2);
    place_mesh(
        mesh_bunny3, inst_bunny3,
        Eigen::Vector2d(bb_tire4_now.center()(0), bb_tire4_now.center()(1)),
        bunny3_bottom_z);
  }
#endif

  place_mesh_center(mesh_openbox, inst_openbox,
                    Eigen::Vector3d(0.6, 0.0, -0.44));

  const int beam_elems = beam_nx * beam_ny;
  const int beam_nodes = (beam_nx + 1) * (beam_ny + 1);
  Eigen::VectorXd beam_x12(4 * beam_nodes);
  Eigen::VectorXd beam_y12(4 * beam_nodes);
  Eigen::VectorXd beam_z12(4 * beam_nodes);
  Eigen::MatrixXi beam_conn(beam_elems, 4);
  ANCFCPUUtils::ANCF3443_generate_shell_coordinates(beam_x, beam_y, beam_nx,
                                                    beam_ny, beam_x12, beam_y12,
                                                    beam_z12, beam_conn);
  for (int node = 0; node < beam_nodes; ++node) {
    const int base = node * 4;
    beam_x12(base + 0) += beam_x_margin;
    beam_y12(base + 0) -= 0.5 * beam_y;
    beam_z12(base + 0) -= 0.18;
  }

  const Eigen::MatrixXd& all_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& all_elems = mesh_manager.GetAllElements();
  const BBox bb_teapot             = ComputeBBox(all_nodes, inst_teapot);
  const BBox bb_tire               = ComputeBBox(all_nodes, inst_tire);
  const BBox bb_tire2              = ComputeBBox(all_nodes, inst_tire2);
  const BBox bb_tire3              = ComputeBBox(all_nodes, inst_tire3);
  const BBox bb_tire4              = ComputeBBox(all_nodes, inst_tire4);
  const BBox bb_tire5              = ComputeBBox(all_nodes, inst_tire5);
  const BBox bb_tire6              = ComputeBBox(all_nodes, inst_tire6);
  const BBox bb_tire7              = ComputeBBox(all_nodes, inst_tire7);
#if 0  // Debug-disabled items: bunnies and armadilo.
  const BBox bb_armadilo           = ComputeBBox(all_nodes, inst_armadilo);
  const BBox bb_bunny              = ComputeBBox(all_nodes, inst_bunny);
  const BBox bb_bunny2             = ComputeBBox(all_nodes, inst_bunny2);
  const BBox bb_bunny3             = ComputeBBox(all_nodes, inst_bunny3);
#endif
  const BBox bb_tire8   = ComputeBBox(all_nodes, inst_tire8);
  const BBox bb_openbox = ComputeBBox(all_nodes, inst_openbox);
  const BBox bb_beam =
      ComputeBBoxAncf3443(beam_x12, beam_y12, beam_z12, beam_nodes);

  std::cout << "Placed FEAT10 items:\n";
  PrintBBox("teapot", bb_teapot);
  PrintBBox("tire", bb_tire);
  PrintBBox("tire2", bb_tire2);
  PrintBBox("tire3", bb_tire3);
  PrintBBox("tire4", bb_tire4);
  PrintBBox("tire5", bb_tire5);
  PrintBBox("tire6", bb_tire6);
  PrintBBox("tire7", bb_tire7);
#if 0  // Debug-disabled items: bunnies and armadilo.
  PrintBBox("armadilo", bb_armadilo);
  PrintBBox("bunny", bb_bunny);
  PrintBBox("bunny2", bb_bunny2);
  PrintBBox("bunny3", bb_bunny3);
#endif
  PrintBBox("tire8", bb_tire8);
  PrintBBox("openbox", bb_openbox);
  PrintBBox("beam", bb_beam);

  std::cout << "Collision: split_items_into_patches="
            << (split_items_into_patches ? "true" : "false") << "\n";

  // -------------------------------------------------------------------------
  // Build one unified FEAT10 block while keeping per-mesh export slices.
  // -------------------------------------------------------------------------
  auto extract_feat10_slice = [&](const ANCFCPUUtils::MeshInstance& inst,
                                  Eigen::VectorXd& x12, Eigen::VectorXd& y12,
                                  Eigen::VectorXd& z12,
                                  Eigen::MatrixXi& elems_local) {
    x12.resize(inst.num_nodes);
    y12.resize(inst.num_nodes);
    z12.resize(inst.num_nodes);
    for (int i = 0; i < inst.num_nodes; ++i) {
      const int idx = inst.node_offset + i;
      x12(i)        = all_nodes(idx, 0);
      y12(i)        = all_nodes(idx, 1);
      z12(i)        = all_nodes(idx, 2);
    }

    elems_local.resize(inst.num_elements, 10);
    for (int e = 0; e < inst.num_elements; ++e) {
      const int elem_idx = inst.element_offset + e;
      for (int n = 0; n < 10; ++n) {
        elems_local(e, n) = all_elems(elem_idx, n) - inst.node_offset;
      }
    }
  };

  Eigen::VectorXd teapot_x0, teapot_y0, teapot_z0;
  Eigen::MatrixXi teapot_elems_local;
  extract_feat10_slice(inst_teapot, teapot_x0, teapot_y0, teapot_z0,
                       teapot_elems_local);

  Eigen::VectorXd tire_x0, tire_y0, tire_z0;
  Eigen::MatrixXi tire_elems_local;
  extract_feat10_slice(inst_tire, tire_x0, tire_y0, tire_z0, tire_elems_local);

  Eigen::VectorXd tire2_x0, tire2_y0, tire2_z0;
  Eigen::MatrixXi tire2_elems_local;
  extract_feat10_slice(inst_tire2, tire2_x0, tire2_y0, tire2_z0,
                       tire2_elems_local);

  Eigen::VectorXd tire3_x0, tire3_y0, tire3_z0;
  Eigen::MatrixXi tire3_elems_local;
  extract_feat10_slice(inst_tire3, tire3_x0, tire3_y0, tire3_z0,
                       tire3_elems_local);

  Eigen::VectorXd tire4_x0, tire4_y0, tire4_z0;
  Eigen::MatrixXi tire4_elems_local;
  extract_feat10_slice(inst_tire4, tire4_x0, tire4_y0, tire4_z0,
                       tire4_elems_local);

  Eigen::VectorXd tire5_x0, tire5_y0, tire5_z0;
  Eigen::MatrixXi tire5_elems_local;
  extract_feat10_slice(inst_tire5, tire5_x0, tire5_y0, tire5_z0,
                       tire5_elems_local);

  Eigen::VectorXd tire6_x0, tire6_y0, tire6_z0;
  Eigen::MatrixXi tire6_elems_local;
  extract_feat10_slice(inst_tire6, tire6_x0, tire6_y0, tire6_z0,
                       tire6_elems_local);

  Eigen::VectorXd tire7_x0, tire7_y0, tire7_z0;
  Eigen::MatrixXi tire7_elems_local;
  extract_feat10_slice(inst_tire7, tire7_x0, tire7_y0, tire7_z0,
                       tire7_elems_local);
  Eigen::VectorXd tire8_x0, tire8_y0, tire8_z0;
  Eigen::MatrixXi tire8_elems_local;
  extract_feat10_slice(inst_tire8, tire8_x0, tire8_y0, tire8_z0,
                       tire8_elems_local);
#if 0  // Debug-disabled items: bunnies.
  Eigen::VectorXd bunny_x0, bunny_y0, bunny_z0;
  Eigen::MatrixXi bunny_elems_local;
  extract_feat10_slice(inst_bunny, bunny_x0, bunny_y0, bunny_z0,
                       bunny_elems_local);

  Eigen::VectorXd bunny2_x0, bunny2_y0, bunny2_z0;
  Eigen::MatrixXi bunny2_elems_local;
  extract_feat10_slice(inst_bunny2, bunny2_x0, bunny2_y0, bunny2_z0,
                       bunny2_elems_local);

  Eigen::VectorXd bunny3_x0, bunny3_y0, bunny3_z0;
  Eigen::MatrixXi bunny3_elems_local;
  extract_feat10_slice(inst_bunny3, bunny3_x0, bunny3_y0, bunny3_z0,
                       bunny3_elems_local);
#endif

  Eigen::VectorXd openbox_x0, openbox_y0, openbox_z0;
  Eigen::MatrixXi openbox_elems_local;
  extract_feat10_slice(inst_openbox, openbox_x0, openbox_y0, openbox_z0,
                       openbox_elems_local);

#if 0  // Debug-disabled item: armadilo.
  Eigen::VectorXd armadilo_x0, armadilo_y0, armadilo_z0;
  Eigen::MatrixXi armadilo_elems_local;
  extract_feat10_slice(inst_armadilo, armadilo_x0, armadilo_y0, armadilo_z0,
                       armadilo_elems_local);
#endif

  const int t10_elems = static_cast<int>(all_elems.rows());
  const int t10_nodes = static_cast<int>(all_nodes.rows());
  auto t10_data       = std::make_unique<GPU_FEAT10_Data>(t10_elems, t10_nodes);
  t10_data->Initialize();

  Eigen::VectorXd t10_x0           = all_nodes.col(0);
  Eigen::VectorXd t10_y0           = all_nodes.col(1);
  Eigen::VectorXd t10_z0           = all_nodes.col(2);
  Eigen::MatrixXi t10_elems_global = all_elems;

  t10_data->Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                  Quadrature::tet5pt_z, Quadrature::tet5pt_weights, t10_x0,
                  t10_y0, t10_z0, t10_elems_global);

  std::vector<int> t10_elem_starts = {
      inst_teapot.element_offset, inst_tire.element_offset,
      inst_tire2.element_offset, inst_tire3.element_offset,
      inst_tire4.element_offset, inst_tire5.element_offset,
      inst_tire6.element_offset, inst_tire7.element_offset,
      inst_tire8.element_offset,
      /* inst_bunny.element_offset, */
      /* inst_bunny2.element_offset, */
      /* inst_bunny3.element_offset, */
      inst_openbox.element_offset
      /* inst_armadilo.element_offset */};
  std::vector<int> t10_elem_counts = {
      inst_teapot.num_elements, inst_tire.num_elements, inst_tire2.num_elements,
      inst_tire3.num_elements, inst_tire4.num_elements, inst_tire5.num_elements,
      inst_tire6.num_elements, inst_tire7.num_elements, inst_tire8.num_elements,
      /* inst_bunny.num_elements, */
      /* inst_bunny2.num_elements, */
      /* inst_bunny3.num_elements, */
      inst_openbox.num_elements
      /* inst_armadilo.num_elements */};
  std::vector<SolidMaterialProperties> t10_materials = {
      mat_tire, mat_tire, mat_tire, mat_tire, mat_tire, mat_tire, mat_tire,
      mat_tire, mat_tire,
      /* mat_bunny, */
      /* mat_bunny, */
      /* mat_bunny, */
      mat_teapot
      /* mat_armadilo */};
  t10_data->ApplyMaterialsByElementRanges(t10_elem_starts, t10_elem_counts,
                                          t10_materials);
  t10_data->CalcDnDuPre();
  t10_data->CalcMassMatrix();
  {
    Eigen::VectorXi fixed_nodes(inst_openbox.num_nodes);
    for (int i = 0; i < inst_openbox.num_nodes; ++i) {
      fixed_nodes(i) = inst_openbox.node_offset + i;
    }
    FEAT10ConstraintManager t10_constraints(t10_data.get());
    t10_constraints.AddNodesToWorldCD(fixed_nodes);
    t10_constraints.Finalize();
  }
  t10_data->CalcConstraintData();
  t10_data->ConvertToCSR_ConstraintJacT();
  t10_data->BuildConstraintJacobianCSR();

  auto beam_data = std::make_unique<GPU_ANCF3443_Data>(beam_nodes, beam_elems);
  beam_data->Initialize();

  std::vector<int> fixed_coefs;
  fixed_coefs.reserve(static_cast<size_t>(4 * (beam_ny + 1)));
  constexpr double x_edge_tol = 1e-12;
  for (int node = 0; node < beam_nodes; ++node) {
    const int base = node * 4;
    if (std::abs(beam_x12(base + 0) - beam_x_margin) <= x_edge_tol) {
      for (int d = 0; d < 4; ++d) {
        fixed_coefs.push_back(base + d);
      }
    }
  }
  Eigen::VectorXi h_fixed(static_cast<int>(fixed_coefs.size()));
  for (int i = 0; i < static_cast<int>(fixed_coefs.size()); ++i) {
    h_fixed(i) = fixed_coefs[static_cast<size_t>(i)];
  }
  beam_data->SetNodalFixed(h_fixed);

  beam_data->Setup(beam_x / static_cast<double>(beam_nx),
                   beam_y / static_cast<double>(beam_ny), beam_h,
                   Quadrature::gauss_xi_m_7, Quadrature::gauss_eta_m_7,
                   Quadrature::gauss_zeta_m_3, Quadrature::gauss_xi_4,
                   Quadrature::gauss_eta_4, Quadrature::gauss_zeta_3,
                   Quadrature::weight_xi_m_7, Quadrature::weight_eta_m_7,
                   Quadrature::weight_zeta_m_3, Quadrature::weight_xi_4,
                   Quadrature::weight_eta_4, Quadrature::weight_zeta_3,
                   beam_x12, beam_y12, beam_z12, beam_conn);
  beam_data->SetDensity(beam_rho);
  beam_data->SetDamping(2e4, 2e4);
  beam_data->SetSVK(beam_E, beam_nu);
  beam_data->CalcDsDuPre();
  beam_data->CalcMassMatrix();
  beam_data->CalcConstraintData();
  beam_data->ConvertToCSR_ConstraintJacT();
  beam_data->BuildConstraintJacobianCSR();

  std::cout << "Beam: nx=" << beam_nx << " ny=" << beam_ny
            << " nodes=" << beam_nodes << " elems=" << beam_elems
            << " constrained_coefs=" << h_fixed.size() << "\n";

  // -------------------------------------------------------------------------
  // Multi-element co-simulation setup
  // -------------------------------------------------------------------------
  FEMultiElementProblem problem;
  const int block_beam = problem.AddElementBlock(beam_data.get(), TYPE_3443);
  const int block_t10  = problem.AddElementBlock(t10_data.get(), TYPE_T10);
  problem.Finalize();
  problem.SyncPositionsFromElements();
  problem.UpdateCollisionNodeBuffer();

  MultiElementNewtonSolver solver(&problem);
  MultiElementNewtonParams params;
  params.inner_atol         = 1e-4;
  params.inner_rtol         = 1e-6;
  params.outer_tol          = 1e-5;
  params.enable_line_search = true;
  params.rho                = 1e14;
  params.max_outer          = 5;
  params.max_inner          = 20;
  params.time_step          = dt;
  solver.SetParameters(&params);
  solver.Setup();

  // -------------------------------------------------------------------------
  // Collision surfaces + coupling maps
  // -------------------------------------------------------------------------
  ANCFCPUUtils::SurfaceTriMesh teapot_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_teapot);
  ANCFCPUUtils::SurfaceTriMesh tire_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_tire);
  ANCFCPUUtils::SurfaceTriMesh tire2_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_tire2);
  ANCFCPUUtils::SurfaceTriMesh tire3_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_tire3);
  ANCFCPUUtils::SurfaceTriMesh tire4_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_tire4);
  ANCFCPUUtils::SurfaceTriMesh tire5_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_tire5);
  ANCFCPUUtils::SurfaceTriMesh tire6_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_tire6);
  ANCFCPUUtils::SurfaceTriMesh tire7_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_tire7);
  ANCFCPUUtils::SurfaceTriMesh tire8_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_tire8);
#if 0  // Debug-disabled items: bunnies.
  ANCFCPUUtils::SurfaceTriMesh bunny_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_bunny);
  ANCFCPUUtils::SurfaceTriMesh bunny2_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_bunny2);
  ANCFCPUUtils::SurfaceTriMesh bunny3_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_bunny3);
#endif
  ANCFCPUUtils::SurfaceTriMesh openbox_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_openbox);
#if 0  // Debug-disabled item: armadilo.
  ANCFCPUUtils::SurfaceTriMesh armadilo_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_armadilo);
#endif
  ANCFCPUUtils::SurfaceTriMesh beam_surface = BuildClosedShellCollisionSurface(
      beam_x12, beam_y12, beam_z12, beam_conn, beam_nodes, beam_elems, beam_h);

  const std::vector<int> teapot_surf_node_ids = teapot_surface.global_node_ids;
  const std::vector<int> tire_surf_node_ids   = tire_surface.global_node_ids;
  const std::vector<int> tire2_surf_node_ids  = tire2_surface.global_node_ids;
  const std::vector<int> tire3_surf_node_ids  = tire3_surface.global_node_ids;
  const std::vector<int> tire4_surf_node_ids  = tire4_surface.global_node_ids;
  const std::vector<int> tire5_surf_node_ids  = tire5_surface.global_node_ids;
  const std::vector<int> tire6_surf_node_ids  = tire6_surface.global_node_ids;
  const std::vector<int> tire7_surf_node_ids  = tire7_surface.global_node_ids;
  const std::vector<int> tire8_surf_node_ids  = tire8_surface.global_node_ids;
#if 0  // Debug-disabled items: bunnies.
  const std::vector<int> bunny_surf_node_ids  = bunny_surface.global_node_ids;
  const std::vector<int> bunny2_surf_node_ids = bunny2_surface.global_node_ids;
  const std::vector<int> bunny3_surf_node_ids = bunny3_surface.global_node_ids;
#endif
  const std::vector<int> openbox_surf_node_ids =
      openbox_surface.global_node_ids;
#if 0  // Debug-disabled item: armadilo.
  const std::vector<int> armadilo_surf_node_ids =
      armadilo_surface.global_node_ids;
#endif
  const std::vector<int> beam_surf_ancf_node_ids = beam_surface.ancf_node_ids;

  const auto t10_lump      = LumpedMassFromFeat10(*t10_data, t10_nodes);
  auto make_t10_mass_props = [&](const ANCFCPUUtils::MeshInstance& inst) {
    return CollisionMassProperties::ComputeFromLumpedNodes(
        all_nodes, inst.node_offset, inst.num_nodes, t10_lump,
        inst.node_offset);
  };
  const auto teapot_mass_props = make_t10_mass_props(inst_teapot);
  const auto tire_mass_props   = make_t10_mass_props(inst_tire);
  const auto tire2_mass_props  = make_t10_mass_props(inst_tire2);
  const auto tire3_mass_props  = make_t10_mass_props(inst_tire3);
  const auto tire4_mass_props  = make_t10_mass_props(inst_tire4);
  const auto tire5_mass_props  = make_t10_mass_props(inst_tire5);
  const auto tire6_mass_props  = make_t10_mass_props(inst_tire6);
  const auto tire7_mass_props  = make_t10_mass_props(inst_tire7);
  const auto tire8_mass_props  = make_t10_mass_props(inst_tire8);
#if 0  // Debug-disabled items: bunnies.
  const auto bunny_mass_props    = make_t10_mass_props(inst_bunny);
  const auto bunny2_mass_props   = make_t10_mass_props(inst_bunny2);
  const auto bunny3_mass_props   = make_t10_mass_props(inst_bunny3);
#endif
  const auto openbox_mass_props = make_t10_mass_props(inst_openbox);
#if 0  // Debug-disabled item: armadilo.
  const auto armadilo_mass_props = make_t10_mass_props(inst_armadilo);
#endif
  const FEStateBuffer& state = problem.GetStateBuffer();
  const int beam_coef_off =
      state.blocks[static_cast<size_t>(block_beam)].coef_offset;
  const int t10_coef_off =
      state.blocks[static_cast<size_t>(block_t10)].coef_offset;

  const int teapot_surf_verts =
      static_cast<int>(teapot_surface.vertices.size());
  const int tire_surf_verts  = static_cast<int>(tire_surface.vertices.size());
  const int tire2_surf_verts = static_cast<int>(tire2_surface.vertices.size());
  const int tire3_surf_verts = static_cast<int>(tire3_surface.vertices.size());
  const int tire4_surf_verts = static_cast<int>(tire4_surface.vertices.size());
  const int tire5_surf_verts = static_cast<int>(tire5_surface.vertices.size());
  const int tire6_surf_verts = static_cast<int>(tire6_surface.vertices.size());
  const int tire7_surf_verts = static_cast<int>(tire7_surface.vertices.size());
  const int tire8_surf_verts = static_cast<int>(tire8_surface.vertices.size());
#if 0  // Debug-disabled items: bunnies.
  const int bunny_surf_verts = static_cast<int>(bunny_surface.vertices.size());
  const int bunny2_surf_verts =
      static_cast<int>(bunny2_surface.vertices.size());
  const int bunny3_surf_verts =
      static_cast<int>(bunny3_surface.vertices.size());
#endif
  const int openbox_surf_verts =
      static_cast<int>(openbox_surface.vertices.size());
#if 0  // Debug-disabled item: armadilo.
  const int armadilo_surf_verts =
      static_cast<int>(armadilo_surface.vertices.size());
#endif
  const int beam_surf_verts = static_cast<int>(beam_surface.vertices.size());
  const int n_coll_nodes =
      teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
      tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
      tire6_surf_verts + tire7_surf_verts + tire8_surf_verts +
      openbox_surf_verts + beam_surf_verts;

  for (int i = 0; i < teapot_surf_verts; ++i) {
    teapot_surface.global_node_ids[static_cast<size_t>(i)] = i;
  }
  for (int i = 0; i < tire_surf_verts; ++i) {
    tire_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + i;
  }
  for (int i = 0; i < tire2_surf_verts; ++i) {
    tire2_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + i;
  }
  for (int i = 0; i < tire3_surf_verts; ++i) {
    tire3_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts + i;
  }
  for (int i = 0; i < tire4_surf_verts; ++i) {
    tire4_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + i;
  }
  for (int i = 0; i < tire5_surf_verts; ++i) {
    tire5_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + i;
  }
  for (int i = 0; i < tire6_surf_verts; ++i) {
    tire6_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts + i;
  }
  for (int i = 0; i < tire7_surf_verts; ++i) {
    tire7_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
        tire6_surf_verts + i;
  }
  for (int i = 0; i < tire8_surf_verts; ++i) {
    tire8_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
        tire6_surf_verts + tire7_surf_verts + i;
  }
#if 0  // Debug-disabled items: bunnies.
  for (int i = 0; i < bunny_surf_verts; ++i) {
    bunny_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
        tire6_surf_verts + tire7_surf_verts + tire8_surf_verts + i;
  }
  for (int i = 0; i < bunny2_surf_verts; ++i) {
    bunny2_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
        tire6_surf_verts + tire7_surf_verts + tire8_surf_verts +
        bunny_surf_verts + i;
  }
  for (int i = 0; i < bunny3_surf_verts; ++i) {
    bunny3_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
        tire6_surf_verts + tire7_surf_verts + tire8_surf_verts +
        bunny_surf_verts +
        bunny2_surf_verts + i;
  }
#endif
  for (int i = 0; i < openbox_surf_verts; ++i) {
    openbox_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
        tire6_surf_verts + tire7_surf_verts + tire8_surf_verts + i;
  }
#if 0  // Debug-disabled item: armadilo.
  for (int i = 0; i < armadilo_surf_verts; ++i) {
    armadilo_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + openbox_surf_verts + i;
  }
#endif
  for (int i = 0; i < beam_surf_verts; ++i) {
    beam_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
        tire6_surf_verts + tire7_surf_verts + tire8_surf_verts +
        openbox_surf_verts + i;
  }

  std::vector<DemeMeshCollisionBody> bodies;
  auto add_body = [&](ANCFCPUUtils::SurfaceTriMesh&& surface,
                      unsigned int family, bool split_into_patches,
                      const CollisionMassProperties::RigidMassProperties& props,
                      bool skip_self_contact_forces = false) {
    (void)props;
    DemeMeshCollisionBody body;
    body.surface                  = std::move(surface);
    body.family                   = family;
    body.split_into_patches       = split_into_patches;
    body.skip_self_contact_forces = skip_self_contact_forces;
    bodies.push_back(std::move(body));
  };
  auto add_body_default_props =
      [&](ANCFCPUUtils::SurfaceTriMesh&& surface, unsigned int family,
          bool split_into_patches, bool skip_self_contact_forces = false) {
        DemeMeshCollisionBody body;
        body.surface                  = std::move(surface);
        body.family                   = family;
        body.split_into_patches       = split_into_patches;
        body.skip_self_contact_forces = skip_self_contact_forces;
        bodies.push_back(std::move(body));
      };
  add_body(std::move(teapot_surface), 0, split_items_into_patches,
           teapot_mass_props);
  add_body(std::move(tire_surface), 1, split_items_into_patches,
           tire_mass_props);
  add_body(std::move(tire2_surface), 2, split_items_into_patches,
           tire2_mass_props);
  add_body(std::move(tire3_surface), 3, split_items_into_patches,
           tire3_mass_props);
  add_body(std::move(tire4_surface), 4, split_items_into_patches,
           tire4_mass_props);
  add_body(std::move(tire5_surface), 11, split_items_into_patches,
           tire5_mass_props);
  add_body(std::move(tire6_surface), 12, split_items_into_patches,
           tire6_mass_props);
  add_body(std::move(tire7_surface), 13, split_items_into_patches,
           tire7_mass_props);
  add_body(std::move(tire8_surface), 14, split_items_into_patches,
           tire8_mass_props);
#if 0  // Debug-disabled items: bunnies.
  add_body(std::move(bunny_surface), 5, split_items_into_patches,
           bunny_mass_props);
  add_body(std::move(bunny2_surface), 6, split_items_into_patches,
           bunny2_mass_props);
  add_body(std::move(bunny3_surface), 7, split_items_into_patches,
           bunny3_mass_props);
#endif
  add_body(std::move(openbox_surface), 8, split_items_into_patches,
           openbox_mass_props, true);
#if 0  // Debug-disabled item: armadilo.
  add_body(std::move(armadilo_surface), 9, split_items_into_patches,
           armadilo_mass_props);
#endif
  // For debugging, let DEME use its default owner-frame properties for the
  // beam instead of the FE-derived rigid mass/MOI estimate.
  add_body_default_props(std::move(beam_surface), 10, split_items_into_patches);

  std::cout << "Creating DEME collision system...\n" << std::flush;
  auto collision_system = std::make_unique<DemeMeshCollisionSystem>(
      std::move(bodies), mu_s, mu_k, contact_E, contact_cor, self_collision,
      dt);
  std::cout << "DEME collision system ready.\n";

  std::vector<int> h_coll_coef(static_cast<size_t>(n_coll_nodes), 0);
  std::vector<double> h_coll_zoff(static_cast<size_t>(n_coll_nodes), 0.0);

  for (int i = 0; i < teapot_surf_verts; ++i) {
    const int global_node = teapot_surf_node_ids[static_cast<size_t>(i)];
    h_coll_coef[static_cast<size_t>(i)] = t10_coef_off + global_node;
  }
  for (int i = 0; i < tire_surf_verts; ++i) {
    const int global_node = tire_surf_node_ids[static_cast<size_t>(i)];
    const int idx         = teapot_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
  for (int i = 0; i < tire2_surf_verts; ++i) {
    const int global_node = tire2_surf_node_ids[static_cast<size_t>(i)];
    const int idx         = teapot_surf_verts + tire_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
  for (int i = 0; i < tire3_surf_verts; ++i) {
    const int global_node = tire3_surf_node_ids[static_cast<size_t>(i)];
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
  for (int i = 0; i < tire4_surf_verts; ++i) {
    const int global_node = tire4_surf_node_ids[static_cast<size_t>(i)];
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
  for (int i = 0; i < tire5_surf_verts; ++i) {
    const int global_node = tire5_surf_node_ids[static_cast<size_t>(i)];
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
  for (int i = 0; i < tire6_surf_verts; ++i) {
    const int global_node = tire6_surf_node_ids[static_cast<size_t>(i)];
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
  for (int i = 0; i < tire7_surf_verts; ++i) {
    const int global_node = tire7_surf_node_ids[static_cast<size_t>(i)];
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
                    tire6_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
  for (int i = 0; i < tire8_surf_verts; ++i) {
    const int global_node = tire8_surf_node_ids[static_cast<size_t>(i)];
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
                    tire6_surf_verts + tire7_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
#if 0  // Debug-disabled items: bunnies.
  for (int i = 0; i < bunny_surf_verts; ++i) {
    const int global_node = bunny_surf_node_ids[static_cast<size_t>(i)];
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
                    tire6_surf_verts + tire7_surf_verts + tire8_surf_verts +
                    i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
  for (int i = 0; i < bunny2_surf_verts; ++i) {
    const int global_node = bunny2_surf_node_ids[static_cast<size_t>(i)];
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
                    tire6_surf_verts + tire7_surf_verts + tire8_surf_verts +
                    bunny_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
  for (int i = 0; i < bunny3_surf_verts; ++i) {
    const int global_node = bunny3_surf_node_ids[static_cast<size_t>(i)];
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
                    tire6_surf_verts + tire7_surf_verts + tire8_surf_verts +
                    bunny_surf_verts +
                    bunny2_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
#endif
  for (int i = 0; i < openbox_surf_verts; ++i) {
    const int global_node = openbox_surf_node_ids[static_cast<size_t>(i)];
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
                    tire6_surf_verts + tire7_surf_verts + tire8_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
#if 0  // Debug-disabled item: armadilo.
  for (int i = 0; i < armadilo_surf_verts; ++i) {
    const int global_node = armadilo_surf_node_ids[static_cast<size_t>(i)];
    const int idx =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + openbox_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = t10_coef_off + global_node;
  }
#endif
  for (int i = 0; i < beam_surf_verts; ++i) {
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
                    tire6_surf_verts + tire7_surf_verts + tire8_surf_verts +
                    openbox_surf_verts + i;
    const int ancf_node = beam_surf_ancf_node_ids[static_cast<size_t>(i)];
    h_coll_coef[static_cast<size_t>(idx)] = beam_coef_off + ancf_node * 4;
    h_coll_zoff[static_cast<size_t>(idx)] =
        (i >= beam_nodes ? 0.5 : -0.5) * beam_h;
  }

  int min_coef = std::numeric_limits<int>::max();
  int max_coef = std::numeric_limits<int>::min();
  for (int c : h_coll_coef) {
    min_coef = std::min(min_coef, c);
    max_coef = std::max(max_coef, c);
  }
  if (min_coef < 0 || max_coef >= state.total_coef) {
    std::cerr << "Invalid collision->coef map: min=" << min_coef
              << " max=" << max_coef << " total_coef=" << state.total_coef
              << "\n";
    return 1;
  }

  int* d_coll_coef     = nullptr;
  double* d_coll_zoff  = nullptr;
  double* d_coll_nodes = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_coll_coef, n_coll_nodes * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_coll_zoff, n_coll_nodes * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(
      &d_coll_nodes, static_cast<size_t>(n_coll_nodes) * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_coll_coef, h_coll_coef.data(),
                          n_coll_nodes * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_coll_zoff, h_coll_zoff.data(),
                          n_coll_nodes * sizeof(double),
                          cudaMemcpyHostToDevice));

  GatherCollisionNodesColumnMajor(d_coll_nodes, n_coll_nodes, state.d_x12,
                                  state.d_y12, state.d_z12, d_coll_coef,
                                  d_coll_zoff);
  HANDLE_ERROR(cudaGetLastError());
  collision_system->BindNodesDevicePtr(d_coll_nodes, n_coll_nodes);

  // -------------------------------------------------------------------------
  // Gravity
  // -------------------------------------------------------------------------
  const int total_dofs = problem.GetTotalDofs();
  std::vector<double> h_f_gravity(static_cast<size_t>(total_dofs), 0.0);

  auto add_feat10_gravity = [&](const ANCFCPUUtils::MeshInstance& inst) {
    for (int node = 0; node < inst.num_nodes; ++node) {
      const int global_node = inst.node_offset + node;
      const int coef        = t10_coef_off + global_node;
      h_f_gravity[static_cast<size_t>(3 * coef + 2)] +=
          t10_lump[static_cast<size_t>(global_node)] * gravity;
    }
  };
  add_feat10_gravity(inst_teapot);
  add_feat10_gravity(inst_tire);
  add_feat10_gravity(inst_tire2);
  add_feat10_gravity(inst_tire3);
  add_feat10_gravity(inst_tire4);
  add_feat10_gravity(inst_tire5);
  add_feat10_gravity(inst_tire6);
  add_feat10_gravity(inst_tire7);
  add_feat10_gravity(inst_tire8);
  // add_feat10_gravity(inst_bunny);
  // add_feat10_gravity(inst_bunny2);
  // add_feat10_gravity(inst_bunny3);
  // add_feat10_gravity(inst_armadilo);

  auto add_ancf_gravity = [&](const std::vector<double>& lump, int coef_off) {
    for (int coef = 0; coef < static_cast<int>(lump.size()); ++coef) {
      h_f_gravity[static_cast<size_t>(3 * (coef_off + coef) + 2)] +=
          lump[static_cast<size_t>(coef)] * gravity;
    }
  };
  const auto beam_lump =
      LumpedMassFromAncf3443(*beam_data, beam_data->get_n_coef());
  add_ancf_gravity(beam_lump, beam_coef_off);

  double* d_f_gravity = nullptr;
  HANDLE_ERROR(cudaMalloc(&d_f_gravity, total_dofs * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_f_gravity, h_f_gravity.data(),
                          total_dofs * sizeof(double), cudaMemcpyHostToDevice));

  // -------------------------------------------------------------------------
  // Simulation loop
  // -------------------------------------------------------------------------
  std::cout << "Starting simulation: steps=" << steps << " dt=" << dt
            << " coll_nodes=" << n_coll_nodes << "\n";

  CollisionSystemParams coll_params;
  coll_params.damping   = contact_cor;
  coll_params.friction  = mu_k;
  coll_params.stiffness = contact_E;

  for (int step = 0; step < steps; ++step) {
    GatherCollisionNodesColumnMajor(d_coll_nodes, n_coll_nodes, state.d_x12,
                                    state.d_y12, state.d_z12, d_coll_coef,
                                    d_coll_zoff);
    HANDLE_ERROR(cudaGetLastError());

    CollisionSystemInput coll_in;
    coll_in.d_nodes_xyz     = d_coll_nodes;
    coll_in.n_nodes         = n_coll_nodes;
    coll_in.d_vel_xyz       = state.d_velocity;
    coll_in.dt              = dt;
    const auto collision_t0 = std::chrono::steady_clock::now();
    collision_system->Step(coll_in, coll_params);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());
    const auto collision_t1 = std::chrono::steady_clock::now();
    const double collision_step_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            collision_t1 - collision_t0)
            .count();
    const int num_contacts = collision_system->GetNumContacts();

    double* d_f_ext = solver.GetExternalForceDevicePtr();
    HANDLE_ERROR(cudaMemcpy(d_f_ext, d_f_gravity, total_dofs * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    ScatterCollisionForcesToExternal(
        collision_system->GetExternalForcesDevicePtr(), n_coll_nodes,
        d_coll_coef, d_f_ext);
    HANDLE_ERROR(cudaGetLastError());

    const auto solver_t0 = std::chrono::steady_clock::now();
    solver.Solve();
    const auto solver_t1 = std::chrono::steady_clock::now();
    const double solver_wall_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            solver_t1 - solver_t0)
            .count();

    int solver_converged_blocks       = 0;
    int solver_total_outer_iterations = 0;
    int solver_total_inner_iterations = 0;
    int line_search_calls             = 0;
    int line_search_successes         = 0;
    int line_search_backtracks_total  = 0;
    int line_search_failures          = 0;
    double solver_block_ms_sum        = 0.0;
    double solver_residual_norm_sum   = 0.0;
    double solver_constraint_norm_sum = 0.0;
    double solver_max_residual_norm   = 0.0;
    double solver_max_constraint_norm = 0.0;
    double line_search_alpha_sum      = 0.0;
    double line_search_alpha_min      = 0.0;
    for (int block_idx = 0; block_idx < solver.GetNumBlocks(); ++block_idx) {
      const SyncedNewtonSolveStats& stats =
          solver.GetBlockSolver(block_idx)->GetLastSolveStats();
      solver_converged_blocks += stats.converged ? 1 : 0;
      solver_total_outer_iterations += stats.outer_iterations;
      solver_total_inner_iterations += stats.inner_iterations;
      line_search_calls += stats.line_search_calls;
      line_search_successes += stats.line_search_successes;
      line_search_backtracks_total += stats.line_search_backtracks_total;
      line_search_failures += stats.line_search_failures;
      solver_block_ms_sum += stats.solve_ms;
      solver_residual_norm_sum += stats.final_residual_norm;
      solver_constraint_norm_sum += stats.final_constraint_norm;
      solver_max_residual_norm =
          std::max(solver_max_residual_norm, stats.final_residual_norm);
      solver_max_constraint_norm =
          std::max(solver_max_constraint_norm, stats.final_constraint_norm);
      line_search_alpha_sum += stats.line_search_alpha_sum;
      if (stats.line_search_alpha_min > 0.0 &&
          (line_search_alpha_min == 0.0 ||
           stats.line_search_alpha_min < line_search_alpha_min)) {
        line_search_alpha_min = stats.line_search_alpha_min;
      }
    }
    const double line_search_alpha_avg_accepted =
        (line_search_successes > 0)
            ? (line_search_alpha_sum /
               static_cast<double>(line_search_successes))
            : 0.0;

    if (write_csv) {
      csv_file << step << "," << ((step + 1) * dt) << "," << num_contacts << ","
               << collision_step_ms << "," << solver_wall_ms << ","
               << solver_block_ms_sum << "," << solver_converged_blocks << ","
               << solver_total_outer_iterations << ","
               << solver_total_inner_iterations << ","
               << solver_residual_norm_sum << "," << solver_constraint_norm_sum
               << "," << solver_max_residual_norm << ","
               << solver_max_constraint_norm << "," << line_search_calls << ","
               << line_search_successes << "," << line_search_backtracks_total
               << "," << line_search_failures << "," << line_search_alpha_min
               << "," << line_search_alpha_avg_accepted << "\n";
    }

    if (vtu_every > 0 && (step % vtu_every) == 0) {
      Eigen::VectorXd beam_x, beam_y, beam_z;
      beam_data->RetrievePositionToCPU(beam_x, beam_y, beam_z);
      beam_data->ComputeVonMises();
      Eigen::VectorXd beam_vm;
      beam_data->RetrieveVonMisesToCPU(beam_vm);
      {
        std::ostringstream fn;
        fn << "output/multiitem_drop_ltest/beam_" << std::setw(6)
           << std::setfill('0') << step << ".vtu";
        ANCFCPUUtils::VisualizationUtils::ExportANCF3443ToVTU(
            beam_x, beam_y, beam_z, beam_conn, beam_h, fn.str(), &beam_vm);
      }

      Eigen::VectorXd t10_x, t10_y, t10_z;
      t10_data->RetrievePositionToCPU(t10_x, t10_y, t10_z);
      t10_data->ComputeVonMises();
      Eigen::VectorXd t10_vm;
      t10_data->RetrieveVonMisesToCPU(t10_vm);

      auto export_t10_slice =
          [&](const std::string& prefix, const ANCFCPUUtils::MeshInstance& inst,
              const Eigen::VectorXd& x0, const Eigen::VectorXd& y0,
              const Eigen::VectorXd& z0, const Eigen::MatrixXi& elems) {
            const int n = inst.num_nodes;
            Eigen::MatrixXd cur(n, 3);
            Eigen::VectorXd disp(n * 3);
            for (int i = 0; i < n; ++i) {
              const int global = inst.node_offset + i;
              cur(i, 0)        = t10_x(global);
              cur(i, 1)        = t10_y(global);
              cur(i, 2)        = t10_z(global);
              disp(3 * i + 0)  = t10_x(global) - x0(i);
              disp(3 * i + 1)  = t10_y(global) - y0(i);
              disp(3 * i + 2)  = t10_z(global) - z0(i);
            }
            Eigen::VectorXd vm =
                t10_vm.segment(inst.element_offset, inst.num_elements);
            std::ostringstream fn;
            fn << "output/multiitem_drop_ltest/" << prefix << "_"
               << std::setw(6) << std::setfill('0') << step << ".vtu";
            ANCFCPUUtils::VisualizationUtils::ExportMeshWithDisplacement(
                cur, elems, disp, fn.str(), &vm);
          };

      export_t10_slice("teapot", inst_teapot, teapot_x0, teapot_y0, teapot_z0,
                       teapot_elems_local);
      export_t10_slice("tire", inst_tire, tire_x0, tire_y0, tire_z0,
                       tire_elems_local);
      export_t10_slice("tire2", inst_tire2, tire2_x0, tire2_y0, tire2_z0,
                       tire2_elems_local);
      export_t10_slice("tire3", inst_tire3, tire3_x0, tire3_y0, tire3_z0,
                       tire3_elems_local);
      export_t10_slice("tire4", inst_tire4, tire4_x0, tire4_y0, tire4_z0,
                       tire4_elems_local);
      export_t10_slice("tire5", inst_tire5, tire5_x0, tire5_y0, tire5_z0,
                       tire5_elems_local);
      export_t10_slice("tire6", inst_tire6, tire6_x0, tire6_y0, tire6_z0,
                       tire6_elems_local);
      export_t10_slice("tire7", inst_tire7, tire7_x0, tire7_y0, tire7_z0,
                       tire7_elems_local);
      export_t10_slice("tire8", inst_tire8, tire8_x0, tire8_y0, tire8_z0,
                       tire8_elems_local);
      // export_t10_slice("bunny", inst_bunny, bunny_x0, bunny_y0, bunny_z0,
      //                  bunny_elems_local);
      // export_t10_slice("bunny2", inst_bunny2, bunny2_x0, bunny2_y0,
      //                  bunny2_z0, bunny2_elems_local);
      // export_t10_slice("bunny3", inst_bunny3, bunny3_x0, bunny3_y0,
      //                  bunny3_z0, bunny3_elems_local);
      // export_t10_slice("armadilo", inst_armadilo, armadilo_x0,
      //                  armadilo_y0, armadilo_z0, armadilo_elems_local);
      export_t10_slice("openbox", inst_openbox, openbox_x0, openbox_y0,
                       openbox_z0, openbox_elems_local);
    }

    if (step % 20 == 0) {
      std::cout << "step=" << std::setw(5) << step
                << " contacts=" << std::setw(6) << num_contacts << "\n";
    }
  }

  HANDLE_ERROR(cudaFree(d_f_gravity));
  HANDLE_ERROR(cudaFree(d_coll_nodes));
  HANDLE_ERROR(cudaFree(d_coll_coef));
  HANDLE_ERROR(cudaFree(d_coll_zoff));

  beam_data->Destroy();
  t10_data->Destroy();
  return 0;
}
