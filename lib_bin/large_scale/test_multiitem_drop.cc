/**
 * Multi-Item Drop Onto ANCF3443 Beam
 *
 * Scenario:
 *   - A deformable ANCF3443 shell beam (1.0 x 0.3 x 0.1) with the x=0 edge
 * fixed
 *   - Two FEAT10 (T10) deformable items (teapot + tire) dropped onto the beam
 *   - Contact handled by DEME mesh-mesh collision, coupled via external forces
 *   - Time integration via MultiElementNewtonSolver (block-diagonal co-sim)
 *
 * Output:
 *   output/multiitem_drop/beam_XXXXXX.vtu
 *   output/multiitem_drop/teapot_XXXXXX.vtu
 *   output/multiitem_drop/tire_XXXXXX.vtu
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../lib_src/collision/DemeMeshCollisionSystem.h"
#include "../../lib_src/elements/ANCF3443Data.cuh"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEMultiElementProblem.h"
#include "../../lib_src/solvers/MultiElementNewton.cuh"
#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
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
  std::cout << "Multi-Item Drop (teapot + 5 tires + 3 bunnies + openbox + "
               "armadilo) onto "
               "ANCF3443 beam\n";
  std::cout << "========================================\n";

  bool split_items_into_patches = false;
  for (int ai = 1; ai < argc; ++ai) {
    const std::string arg(argv[ai]);
    if (arg == "--split_patches") {
      split_items_into_patches = true;
    }
  }

  // -------------------------------------------------------------------------
  // Parameters
  // -------------------------------------------------------------------------
  constexpr double gravity = -9.81;
  constexpr double dt      = 1e-4;
  constexpr int steps      = 10000;
  constexpr int vtu_every  = 50;

  // Beam geometry (shell mid-surface); thickness used only by FE.
  constexpr double beam_x = 0.8;
  constexpr double beam_y = 0.4;
  constexpr double beam_h = 0.02;
  constexpr int beam_nx   = 30;  // not too crazy
  constexpr int beam_ny   = 12;

  // Item placement
  constexpr double drop_clearance = 0.06;  // above beam mid-surface (approx)
  constexpr double tire_scale     = 0.12;  // scale down to fit on 0.3m beam

  // Material (demo defaults; tune as needed)
  const SolidMaterialProperties mat_teapot =
      SolidMaterialProperties::SVK(1.0e6, 0.35, 500.0, 2.0e4, 2.0e4);
  const SolidMaterialProperties mat_tire =
      SolidMaterialProperties::SVK(2.0e6, 0.45, 900.0, 2.0e4, 2.0e4);
  const SolidMaterialProperties mat_bunny =
      SolidMaterialProperties::SVK(1.5e6, 0.45, 300.0, 2.0e4, 2.0e4);
  const SolidMaterialProperties mat_armadilo =
      SolidMaterialProperties::SVK(1.0e6, 0.35, 200.0, 2.0e4, 2.0e4);
  constexpr double beam_E   = 8.0e6;
  constexpr double beam_nu  = 0.33;
  constexpr double beam_rho = 1500.0;

  // Contact
  constexpr double mu_s = 0.5;
  constexpr double mu_k = 0.4;
  // DEME uses a Hertzian-style contact model; too-stiff contact can blow up the
  // implicit Newton solve (NaNs) when sharp features first touch. Keep contact
  // stiffness in the same order as the FE materials by default.
  constexpr double contact_E    = 1e7;
  constexpr double contact_cor  = 0.5;
  constexpr bool self_collision = false;

  // Safety defaults for DEME->FE coupling (override via env vars).
  // These prevent a single contact from injecting an unbounded impulse into the
  // implicit solve when concave/sharp meshes (like openbox) first touch.
  if (std::getenv("DEME_FORCE_CLAMP") == nullptr) {
    setenv("DEME_FORCE_CLAMP", "50000", 1);  // N, per-contact vector norm
  }
  if (std::getenv("DEME_FORCE_DISTRIB_K") == nullptr) {
    setenv("DEME_FORCE_DISTRIB_K", "8", 1);  // spread to more vertices
  }

  std::filesystem::create_directories("output/multiitem_drop");

  // -------------------------------------------------------------------------
  // Load + place FEAT10 meshes (teapot + tire) using MeshManager
  // -------------------------------------------------------------------------
  ANCFCPUUtils::MeshManager mesh_manager;
  const int mesh_teapot =
      mesh_manager.LoadMesh("data/meshes/T10/teapot.1.node",
                            "data/meshes/T10/teapot.1.ele", "teapot");
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
  const int mesh_openbox =
      mesh_manager.LoadMesh("data/meshes/T10/item_drop/openbox.node",
                            "data/meshes/T10/item_drop/openbox.ele", "openbox");
  const int mesh_armadilo = mesh_manager.LoadMesh(
      "data/meshes/T10/item_drop/armadilo.node",
      "data/meshes/T10/item_drop/armadilo.ele", "armadilo");
  if (mesh_teapot < 0 || mesh_tire < 0 || mesh_tire2 < 0 || mesh_tire3 < 0 ||
      mesh_tire4 < 0 || mesh_tire5 < 0 || mesh_bunny < 0 || mesh_bunny2 < 0 ||
      mesh_bunny3 < 0 || mesh_openbox < 0 || mesh_armadilo < 0) {
    std::cerr
        << "Failed to load "
           "teapot/tire/tire2/tire3/tire4/tire5/bunny/bunny2/bunny3/openbox/"
           "armadilo meshes.\n";
    return 1;
  }

  const auto& inst_teapot   = mesh_manager.GetMeshInstance(mesh_teapot);
  const auto& inst_tire     = mesh_manager.GetMeshInstance(mesh_tire);
  const auto& inst_tire2    = mesh_manager.GetMeshInstance(mesh_tire2);
  const auto& inst_tire3    = mesh_manager.GetMeshInstance(mesh_tire3);
  const auto& inst_tire4    = mesh_manager.GetMeshInstance(mesh_tire4);
  const auto& inst_tire5    = mesh_manager.GetMeshInstance(mesh_tire5);
  const auto& inst_bunny    = mesh_manager.GetMeshInstance(mesh_bunny);
  const auto& inst_bunny2   = mesh_manager.GetMeshInstance(mesh_bunny2);
  const auto& inst_bunny3   = mesh_manager.GetMeshInstance(mesh_bunny3);
  const auto& inst_openbox  = mesh_manager.GetMeshInstance(mesh_openbox);
  const auto& inst_armadilo = mesh_manager.GetMeshInstance(mesh_armadilo);

  mesh_manager.TransformMesh(mesh_tire, ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_tire2,
                             ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_tire3,
                             ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_tire4,
                             ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_tire5,
                             ANCFCPUUtils::uniformScale(tire_scale));
  mesh_manager.TransformMesh(mesh_openbox, ANCFCPUUtils::uniformScale(0.4));
  // Shrink armadilo by 8x.
  mesh_manager.TransformMesh(mesh_armadilo, ANCFCPUUtils::uniformScale(0.3));
  mesh_manager.TransformMesh(
      mesh_armadilo,
      ANCFCPUUtils::rotationY((160.0 / 180.0) * std::acos(-1.0)));
  // Rotate both tires by 90 degrees about global +Z (same initial orientation).
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
  mesh_manager.TransformMesh(mesh_teapot, tire_Rz);
  mesh_manager.TransformMesh(
      mesh_bunny2, ANCFCPUUtils::rotationX((35.0 / 180.0) * std::acos(-1.0)));
  mesh_manager.TransformMesh(
      mesh_bunny3, ANCFCPUUtils::rotationX((25.0 / 180.0) * std::acos(-1.0)));

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

  // Beam mid-surface at z=0. Drop items above it.
  place_mesh(mesh_teapot, inst_teapot, Eigen::Vector2d(0.72, 0.0),
             drop_clearance - 0.02);
  const Eigen::Vector2d tire_center_xy(0.45, 0.0);
  const double tire_bottom_z = drop_clearance + 0.06 - 0.02;
  place_mesh(mesh_tire, inst_tire, tire_center_xy, tire_bottom_z);
  place_mesh(mesh_tire2, inst_tire2,
             tire_center_xy + Eigen::Vector2d(-0.25, -0.03), tire_bottom_z);
  place_mesh(mesh_tire3, inst_tire3,
             tire_center_xy + Eigen::Vector2d(0.08, 0.15), tire_bottom_z);
  place_mesh(mesh_tire4, inst_tire4,
             tire_center_xy + Eigen::Vector2d(0.07, -0.12), tire_bottom_z);
  place_mesh(mesh_tire5, inst_tire5,
             tire_center_xy + Eigen::Vector2d(0.07 - 0.30, -0.12),
             tire_bottom_z);

  // Place armadilo on top of the second tire, with a +0.02m clearance in Z.
  {
    const Eigen::MatrixXd& nodes_now = mesh_manager.GetAllNodes();
    const BBox bb_tire2_now          = ComputeBBox(nodes_now, inst_tire2);
    const double armadilo_bottom_z   = bb_tire2_now.mx(2) - 0.04;
    place_mesh(
        mesh_armadilo, inst_armadilo,
        Eigen::Vector2d(bb_tire2_now.center()(0), bb_tire2_now.center()(1)),
        armadilo_bottom_z);
  }

  // Place bunny on top of the tire, with a +0.02m clearance in Z.
  {
    const Eigen::MatrixXd& nodes_now = mesh_manager.GetAllNodes();
    const BBox bb_tire_now           = ComputeBBox(nodes_now, inst_tire);
    const double bunny_bottom_z      = bb_tire_now.mx(2) + 0.02;
    place_mesh(
        mesh_bunny, inst_bunny,
        Eigen::Vector2d(bb_tire_now.center()(0), bb_tire_now.center()(1)),
        bunny_bottom_z);
  }

  // Place bunny2 on top of tire3, with a +0.02m clearance in Z.
  {
    const Eigen::MatrixXd& nodes_now = mesh_manager.GetAllNodes();
    const BBox bb_tire3_now          = ComputeBBox(nodes_now, inst_tire3);
    const double bunny2_bottom_z     = bb_tire3_now.mx(2) + 0.02;
    place_mesh(
        mesh_bunny2, inst_bunny2,
        Eigen::Vector2d(bb_tire3_now.center()(0), bb_tire3_now.center()(1)),
        bunny2_bottom_z);
  }

  // Place bunny3 on top of tire4, with a +0.02m clearance in Z.
  {
    const Eigen::MatrixXd& nodes_now = mesh_manager.GetAllNodes();
    const BBox bb_tire4_now          = ComputeBBox(nodes_now, inst_tire4);
    const double bunny3_bottom_z     = bb_tire4_now.mx(2) + 0.02;
    place_mesh(
        mesh_bunny3, inst_bunny3,
        Eigen::Vector2d(bb_tire4_now.center()(0), bb_tire4_now.center()(1)),
        bunny3_bottom_z);
  }

  // Place the openbox (scaled) with its center near x=0.6, z=-0.55.
  place_mesh_center(mesh_openbox, inst_openbox,
                    Eigen::Vector3d(0.6, 0.0, -0.44));

  const Eigen::MatrixXd& all_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& all_elems = mesh_manager.GetAllElements();
  const BBox bb_teapot             = ComputeBBox(all_nodes, inst_teapot);
  const BBox bb_tire               = ComputeBBox(all_nodes, inst_tire);
  const BBox bb_tire2              = ComputeBBox(all_nodes, inst_tire2);
  const BBox bb_tire3              = ComputeBBox(all_nodes, inst_tire3);
  const BBox bb_tire4              = ComputeBBox(all_nodes, inst_tire4);
  const BBox bb_tire5              = ComputeBBox(all_nodes, inst_tire5);
  const BBox bb_armadilo           = ComputeBBox(all_nodes, inst_armadilo);
  const BBox bb_bunny              = ComputeBBox(all_nodes, inst_bunny);
  const BBox bb_bunny2             = ComputeBBox(all_nodes, inst_bunny2);
  const BBox bb_bunny3             = ComputeBBox(all_nodes, inst_bunny3);
  const BBox bb_openbox            = ComputeBBox(all_nodes, inst_openbox);

  std::cout << "Placed FEAT10 items:\n";
  PrintBBox("teapot", bb_teapot);
  PrintBBox("tire", bb_tire);
  PrintBBox("tire2", bb_tire2);
  PrintBBox("tire3", bb_tire3);
  PrintBBox("tire4", bb_tire4);
  PrintBBox("tire5", bb_tire5);
  PrintBBox("armadilo", bb_armadilo);
  PrintBBox("bunny", bb_bunny);
  PrintBBox("bunny2", bb_bunny2);
  PrintBBox("bunny3", bb_bunny3);
  PrintBBox("openbox", bb_openbox);

  std::cout << "Collision: split_items_into_patches="
            << (split_items_into_patches ? "true" : "false") << "\n";

  // -------------------------------------------------------------------------
  // Build FEAT10 element blocks (teapot + tire) from MeshManager slices
  // -------------------------------------------------------------------------
  auto build_feat10_block = [&](const ANCFCPUUtils::MeshInstance& inst,
                                GPU_FEAT10_Data& out, Eigen::VectorXd& x12,
                                Eigen::VectorXd& y12, Eigen::VectorXd& z12,
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

    out.Initialize();
    out.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y, Quadrature::tet5pt_z,
              Quadrature::tet5pt_weights, x12, y12, z12, elems_local);
  };

  auto teapot_data = std::make_unique<GPU_FEAT10_Data>(inst_teapot.num_elements,
                                                       inst_teapot.num_nodes);
  auto tire_data   = std::make_unique<GPU_FEAT10_Data>(inst_tire.num_elements,
                                                       inst_tire.num_nodes);
  auto tire2_data  = std::make_unique<GPU_FEAT10_Data>(inst_tire2.num_elements,
                                                       inst_tire2.num_nodes);
  auto tire3_data  = std::make_unique<GPU_FEAT10_Data>(inst_tire3.num_elements,
                                                       inst_tire3.num_nodes);
  auto tire4_data  = std::make_unique<GPU_FEAT10_Data>(inst_tire4.num_elements,
                                                       inst_tire4.num_nodes);
  auto tire5_data  = std::make_unique<GPU_FEAT10_Data>(inst_tire5.num_elements,
                                                       inst_tire5.num_nodes);
  auto bunny_data  = std::make_unique<GPU_FEAT10_Data>(inst_bunny.num_elements,
                                                       inst_bunny.num_nodes);
  auto bunny2_data = std::make_unique<GPU_FEAT10_Data>(inst_bunny2.num_elements,
                                                       inst_bunny2.num_nodes);
  auto bunny3_data = std::make_unique<GPU_FEAT10_Data>(inst_bunny3.num_elements,
                                                       inst_bunny3.num_nodes);
  auto openbox_data = std::make_unique<GPU_FEAT10_Data>(
      inst_openbox.num_elements, inst_openbox.num_nodes);
  auto armadilo_data = std::make_unique<GPU_FEAT10_Data>(
      inst_armadilo.num_elements, inst_armadilo.num_nodes);

  Eigen::VectorXd teapot_x0, teapot_y0, teapot_z0;
  Eigen::MatrixXi teapot_elems_local;
  build_feat10_block(inst_teapot, *teapot_data, teapot_x0, teapot_y0, teapot_z0,
                     teapot_elems_local);
  teapot_data->ApplyMaterial(mat_teapot);
  teapot_data->CalcDnDuPre();
  teapot_data->CalcMassMatrix();

  Eigen::VectorXd tire_x0, tire_y0, tire_z0;
  Eigen::MatrixXi tire_elems_local;
  build_feat10_block(inst_tire, *tire_data, tire_x0, tire_y0, tire_z0,
                     tire_elems_local);
  tire_data->ApplyMaterial(mat_tire);
  tire_data->CalcDnDuPre();
  tire_data->CalcMassMatrix();

  Eigen::VectorXd tire2_x0, tire2_y0, tire2_z0;
  Eigen::MatrixXi tire2_elems_local;
  build_feat10_block(inst_tire2, *tire2_data, tire2_x0, tire2_y0, tire2_z0,
                     tire2_elems_local);
  tire2_data->ApplyMaterial(mat_tire);
  tire2_data->CalcDnDuPre();
  tire2_data->CalcMassMatrix();

  Eigen::VectorXd tire3_x0, tire3_y0, tire3_z0;
  Eigen::MatrixXi tire3_elems_local;
  build_feat10_block(inst_tire3, *tire3_data, tire3_x0, tire3_y0, tire3_z0,
                     tire3_elems_local);
  tire3_data->ApplyMaterial(mat_tire);
  tire3_data->CalcDnDuPre();
  tire3_data->CalcMassMatrix();

  Eigen::VectorXd tire4_x0, tire4_y0, tire4_z0;
  Eigen::MatrixXi tire4_elems_local;
  build_feat10_block(inst_tire4, *tire4_data, tire4_x0, tire4_y0, tire4_z0,
                     tire4_elems_local);
  tire4_data->ApplyMaterial(mat_tire);
  tire4_data->CalcDnDuPre();
  tire4_data->CalcMassMatrix();

  Eigen::VectorXd tire5_x0, tire5_y0, tire5_z0;
  Eigen::MatrixXi tire5_elems_local;
  build_feat10_block(inst_tire5, *tire5_data, tire5_x0, tire5_y0, tire5_z0,
                     tire5_elems_local);
  tire5_data->ApplyMaterial(mat_tire);
  tire5_data->CalcDnDuPre();
  tire5_data->CalcMassMatrix();

  Eigen::VectorXd bunny_x0, bunny_y0, bunny_z0;
  Eigen::MatrixXi bunny_elems_local;
  build_feat10_block(inst_bunny, *bunny_data, bunny_x0, bunny_y0, bunny_z0,
                     bunny_elems_local);
  bunny_data->ApplyMaterial(mat_bunny);
  bunny_data->CalcDnDuPre();
  bunny_data->CalcMassMatrix();

  Eigen::VectorXd bunny2_x0, bunny2_y0, bunny2_z0;
  Eigen::MatrixXi bunny2_elems_local;
  build_feat10_block(inst_bunny2, *bunny2_data, bunny2_x0, bunny2_y0, bunny2_z0,
                     bunny2_elems_local);
  bunny2_data->ApplyMaterial(mat_bunny);
  bunny2_data->CalcDnDuPre();
  bunny2_data->CalcMassMatrix();

  Eigen::VectorXd bunny3_x0, bunny3_y0, bunny3_z0;
  Eigen::MatrixXi bunny3_elems_local;
  build_feat10_block(inst_bunny3, *bunny3_data, bunny3_x0, bunny3_y0, bunny3_z0,
                     bunny3_elems_local);
  bunny3_data->ApplyMaterial(mat_bunny);
  bunny3_data->CalcDnDuPre();
  bunny3_data->CalcMassMatrix();

  Eigen::VectorXd openbox_x0, openbox_y0, openbox_z0;
  Eigen::MatrixXi openbox_elems_local;
  build_feat10_block(inst_openbox, *openbox_data, openbox_x0, openbox_y0,
                     openbox_z0, openbox_elems_local);
  openbox_data->ApplyMaterial(mat_teapot);
  openbox_data->CalcDnDuPre();
  openbox_data->CalcMassMatrix();
  {
    Eigen::VectorXi fixed_nodes(inst_openbox.num_nodes);
    for (int i = 0; i < inst_openbox.num_nodes; ++i) {
      fixed_nodes(i) = i;
    }
    openbox_data->SetNodalFixed(fixed_nodes);
  }
  openbox_data->CalcConstraintData();
  openbox_data->ConvertToCSR_ConstraintJacT();
  openbox_data->BuildConstraintJacobianCSR();

  Eigen::VectorXd armadilo_x0, armadilo_y0, armadilo_z0;
  Eigen::MatrixXi armadilo_elems_local;
  build_feat10_block(inst_armadilo, *armadilo_data, armadilo_x0, armadilo_y0,
                     armadilo_z0, armadilo_elems_local);
  armadilo_data->ApplyMaterial(mat_armadilo);
  armadilo_data->CalcDnDuPre();
  armadilo_data->CalcMassMatrix();

  // -------------------------------------------------------------------------
  // Build ANCF3443 beam block (structured shell) with x=0 edge fixed
  // -------------------------------------------------------------------------
  const int beam_elems = beam_nx * beam_ny;
  const int beam_nodes = (beam_nx + 1) * (beam_ny + 1);
  auto beam_data = std::make_unique<GPU_ANCF3443_Data>(beam_nodes, beam_elems);
  beam_data->Initialize();

  Eigen::VectorXd beam_x12(beam_data->get_n_coef());
  Eigen::VectorXd beam_y12(beam_data->get_n_coef());
  Eigen::VectorXd beam_z12(beam_data->get_n_coef());
  Eigen::MatrixXi beam_conn(beam_elems, 4);
  ANCFCPUUtils::ANCF3443_generate_shell_coordinates(beam_x, beam_y, beam_nx,
                                                    beam_ny, beam_x12, beam_y12,
                                                    beam_z12, beam_conn);

  // Center beam in Y around 0 by offsetting position coefficients only.
  for (int node = 0; node < beam_nodes; ++node) {
    const int base = node * 4;
    beam_y12(base + 0) -= 0.5 * beam_y;
  }

  // Fix x=0 edge: constrain all 4 coefficients of those nodes.
  std::vector<int> fixed_coefs;
  fixed_coefs.reserve(static_cast<size_t>(beam_nodes));
  const double x_edge_tol = 1e-12;
  for (int node = 0; node < beam_nodes; ++node) {
    const int base = node * 4;
    if (std::abs(beam_x12(base + 0) - 0.0) <= x_edge_tol) {
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

  const double L = beam_x / static_cast<double>(beam_nx);
  const double W = beam_y / static_cast<double>(beam_ny);
  beam_data->Setup(L, W, beam_h, Quadrature::gauss_xi_m_7,
                   Quadrature::gauss_eta_m_7, Quadrature::gauss_zeta_m_3,
                   Quadrature::gauss_xi_4, Quadrature::gauss_eta_4,
                   Quadrature::gauss_zeta_3, Quadrature::weight_xi_m_7,
                   Quadrature::weight_eta_m_7, Quadrature::weight_zeta_m_3,
                   Quadrature::weight_xi_4, Quadrature::weight_eta_4,
                   Quadrature::weight_zeta_3, beam_x12, beam_y12, beam_z12,
                   beam_conn);
  beam_data->SetDensity(beam_rho);
  beam_data->SetDamping(5e3, 5e3);
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
  const int block_beam   = problem.AddElementBlock(beam_data.get(), TYPE_3443);
  const int block_teapot = problem.AddElementBlock(teapot_data.get(), TYPE_T10);
  const int block_tire   = problem.AddElementBlock(tire_data.get(), TYPE_T10);
  const int block_tire2  = problem.AddElementBlock(tire2_data.get(), TYPE_T10);
  const int block_tire3  = problem.AddElementBlock(tire3_data.get(), TYPE_T10);
  const int block_tire4  = problem.AddElementBlock(tire4_data.get(), TYPE_T10);
  const int block_tire5  = problem.AddElementBlock(tire5_data.get(), TYPE_T10);
  const int block_bunny  = problem.AddElementBlock(bunny_data.get(), TYPE_T10);
  const int block_bunny2 = problem.AddElementBlock(bunny2_data.get(), TYPE_T10);
  const int block_bunny3 = problem.AddElementBlock(bunny3_data.get(), TYPE_T10);
  const int block_openbox =
      problem.AddElementBlock(openbox_data.get(), TYPE_T10);
  const int block_armadilo =
      problem.AddElementBlock(armadilo_data.get(), TYPE_T10);
  problem.Finalize();
  problem.SyncPositionsFromElements();
  problem.UpdateCollisionNodeBuffer();

  MultiElementNewtonSolver solver(&problem);
  MultiElementNewtonParams params;
  params.inner_atol = 1e-4;
  params.inner_rtol = 1e-4;
  params.outer_tol  = 1e-5;
  params.rho        = 1e12;
  params.max_outer  = 3;
  params.max_inner  = 10;
  params.time_step  = dt;
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
  ANCFCPUUtils::SurfaceTriMesh bunny_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_bunny);
  ANCFCPUUtils::SurfaceTriMesh bunny2_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_bunny2);
  ANCFCPUUtils::SurfaceTriMesh bunny3_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_bunny3);
  ANCFCPUUtils::SurfaceTriMesh openbox_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_openbox);
  ANCFCPUUtils::SurfaceTriMesh armadilo_surface =
      ANCFCPUUtils::ExtractSurfaceTriMesh(all_nodes, all_elems, inst_armadilo);

  // Save FEAT10 surface node IDs (mesh-manager global IDs) before remapping to
  // collision-buffer indices.
  const std::vector<int> teapot_surf_node_ids = teapot_surface.global_node_ids;
  const std::vector<int> tire_surf_node_ids   = tire_surface.global_node_ids;
  const std::vector<int> tire2_surf_node_ids  = tire2_surface.global_node_ids;
  const std::vector<int> tire3_surf_node_ids  = tire3_surface.global_node_ids;
  const std::vector<int> tire4_surf_node_ids  = tire4_surface.global_node_ids;
  const std::vector<int> tire5_surf_node_ids  = tire5_surface.global_node_ids;
  const std::vector<int> bunny_surf_node_ids  = bunny_surface.global_node_ids;
  const std::vector<int> bunny2_surf_node_ids = bunny2_surface.global_node_ids;
  const std::vector<int> bunny3_surf_node_ids = bunny3_surface.global_node_ids;
  const std::vector<int> openbox_surf_node_ids =
      openbox_surface.global_node_ids;
  const std::vector<int> armadilo_surf_node_ids =
      armadilo_surface.global_node_ids;

  // Beam collision surface: a closed "thick plate" built from the ANCF3443
  // midsurface + thickness. This avoids feeding DEME an open 2D sheet mesh.
  ANCFCPUUtils::SurfaceTriMesh beam_surface;
  beam_surface.global_node_ids.resize(static_cast<size_t>(2 * beam_nodes));
  beam_surface.vertices.resize(static_cast<size_t>(2 * beam_nodes));
  beam_surface.ancf_node_ids.resize(static_cast<size_t>(2 * beam_nodes));
  for (int node = 0; node < beam_nodes; ++node) {
    const int base = node * 4;
    const Eigen::Vector3d p(beam_x12(base + 0), beam_y12(base + 0),
                            beam_z12(base + 0));
    const int bot                                          = node;
    const int top                                          = beam_nodes + node;
    beam_surface.global_node_ids[static_cast<size_t>(bot)] = bot;
    beam_surface.global_node_ids[static_cast<size_t>(top)] = top;
    beam_surface.ancf_node_ids[static_cast<size_t>(bot)]   = node;
    beam_surface.ancf_node_ids[static_cast<size_t>(top)]   = node;
    beam_surface.vertices[static_cast<size_t>(bot)] =
        p - Eigen::Vector3d(0, 0, 0.5 * beam_h);
    beam_surface.vertices[static_cast<size_t>(top)] =
        p + Eigen::Vector3d(0, 0, 0.5 * beam_h);
  }
  beam_surface.triangles.reserve(static_cast<size_t>(beam_elems) * 4);
  for (int e = 0; e < beam_elems; ++e) {
    const int n0 = beam_conn(e, 0);
    const int n1 = beam_conn(e, 1);
    const int n2 = beam_conn(e, 2);
    const int n3 = beam_conn(e, 3);
    const int t0 = beam_nodes + n0;
    const int t1 = beam_nodes + n1;
    const int t2 = beam_nodes + n2;
    const int t3 = beam_nodes + n3;
    // Top (+Z) surface (assumes beam_conn is consistently ordered).
    beam_surface.triangles.emplace_back(t0, t1, t2);
    beam_surface.triangles.emplace_back(t0, t2, t3);
    // Bottom (-Z) surface (flip winding).
    beam_surface.triangles.emplace_back(n0, n2, n1);
    beam_surface.triangles.emplace_back(n0, n3, n2);
  }

  // Boundary side walls: find boundary edges and add quads (2 tris each)
  // connecting bottom/top copies of the edge nodes.
  struct EdgeInfo {
    int count = 0;
    int a_dir = -1;  // oriented edge a->b as it appears in the unique quad
    int b_dir = -1;
  };
  struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const noexcept {
      return (static_cast<size_t>(p.first) << 32) ^
             static_cast<size_t>(p.second);
    }
  };
  std::unordered_map<std::pair<int, int>, EdgeInfo, PairHash> edge_counts;
  edge_counts.reserve(static_cast<size_t>(beam_elems) * 4);
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
  for (int e = 0; e < beam_elems; ++e) {
    const int n0 = beam_conn(e, 0);
    const int n1 = beam_conn(e, 1);
    const int n2 = beam_conn(e, 2);
    const int n3 = beam_conn(e, 3);
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
    const int a = info.a_dir;
    const int b = info.b_dir;
    // Side wall quad (outward normal points to the "right" of edge a->b).
    const int a_bot = a;
    const int b_bot = b;
    const int a_top = beam_nodes + a;
    const int b_top = beam_nodes + b;
    beam_surface.triangles.emplace_back(a_bot, b_bot, b_top);
    beam_surface.triangles.emplace_back(a_bot, b_top, a_top);
  }

  // Save ANCF node IDs for beam surface vertices before moving `beam_surface`
  // into the collision system.
  const std::vector<int> beam_surf_ancf_node_ids = beam_surface.ancf_node_ids;

  const FEStateBuffer& state = problem.GetStateBuffer();
  const int beam_coef_off =
      state.blocks[static_cast<size_t>(block_beam)].coef_offset;
  const int teapot_coef_off =
      state.blocks[static_cast<size_t>(block_teapot)].coef_offset;
  const int tire_coef_off =
      state.blocks[static_cast<size_t>(block_tire)].coef_offset;
  const int tire2_coef_off =
      state.blocks[static_cast<size_t>(block_tire2)].coef_offset;
  const int tire3_coef_off =
      state.blocks[static_cast<size_t>(block_tire3)].coef_offset;
  const int tire4_coef_off =
      state.blocks[static_cast<size_t>(block_tire4)].coef_offset;
  const int tire5_coef_off =
      state.blocks[static_cast<size_t>(block_tire5)].coef_offset;
  const int bunny_coef_off =
      state.blocks[static_cast<size_t>(block_bunny)].coef_offset;
  const int bunny2_coef_off =
      state.blocks[static_cast<size_t>(block_bunny2)].coef_offset;
  const int bunny3_coef_off =
      state.blocks[static_cast<size_t>(block_bunny3)].coef_offset;
  const int openbox_coef_off =
      state.blocks[static_cast<size_t>(block_openbox)].coef_offset;
  const int armadilo_coef_off =
      state.blocks[static_cast<size_t>(block_armadilo)].coef_offset;

  // Map each surface vertex to the unified coefficient index in
  // collision vertex buffer:
  // - teapot/tire vertices correspond to FEAT10 nodes (one coef per node)
  // - beam vertices correspond to ANCF3443 node position coefficients (node*4)
  //   with a per-vertex z-offset (+/- H/2) applied during gather
  const int teapot_surf_verts =
      static_cast<int>(teapot_surface.vertices.size());
  const int tire_surf_verts  = static_cast<int>(tire_surface.vertices.size());
  const int tire2_surf_verts = static_cast<int>(tire2_surface.vertices.size());
  const int tire3_surf_verts = static_cast<int>(tire3_surface.vertices.size());
  const int tire4_surf_verts = static_cast<int>(tire4_surface.vertices.size());
  const int tire5_surf_verts = static_cast<int>(tire5_surface.vertices.size());
  const int bunny_surf_verts = static_cast<int>(bunny_surface.vertices.size());
  const int bunny2_surf_verts =
      static_cast<int>(bunny2_surface.vertices.size());
  const int bunny3_surf_verts =
      static_cast<int>(bunny3_surface.vertices.size());
  const int openbox_surf_verts =
      static_cast<int>(openbox_surface.vertices.size());
  const int armadilo_surf_verts =
      static_cast<int>(armadilo_surface.vertices.size());
  const int beam_surf_verts = static_cast<int>(beam_surface.vertices.size());
  const int n_coll_nodes =
      teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
      tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
      bunny_surf_verts + bunny2_surf_verts + bunny3_surf_verts +
      openbox_surf_verts + armadilo_surf_verts + beam_surf_verts;

  // Remap surface global_node_ids to match collision node buffer indices.
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
  for (int i = 0; i < bunny_surf_verts; ++i) {
    bunny_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts + i;
  }
  for (int i = 0; i < bunny2_surf_verts; ++i) {
    bunny2_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
        bunny_surf_verts + i;
  }
  for (int i = 0; i < bunny3_surf_verts; ++i) {
    bunny3_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
        bunny_surf_verts + bunny2_surf_verts + i;
  }
  for (int i = 0; i < openbox_surf_verts; ++i) {
    openbox_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
        bunny_surf_verts + bunny2_surf_verts + bunny3_surf_verts + i;
  }
  for (int i = 0; i < armadilo_surf_verts; ++i) {
    armadilo_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
        bunny_surf_verts + bunny2_surf_verts + bunny3_surf_verts +
        openbox_surf_verts + i;
  }
  for (int i = 0; i < beam_surf_verts; ++i) {
    beam_surface.global_node_ids[static_cast<size_t>(i)] =
        teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
        tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
        bunny_surf_verts + bunny2_surf_verts + bunny3_surf_verts +
        openbox_surf_verts + armadilo_surf_verts + i;
  }

  // Create collision bodies.
  std::vector<DemeMeshCollisionBody> bodies;
  {
    DemeMeshCollisionBody body;
    body.surface = std::move(teapot_surface);
    body.family  = 0;  // keep unique families to allow disabling self-collision
    body.split_into_patches = split_items_into_patches;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(tire_surface);
    body.family             = 1;
    body.split_into_patches = split_items_into_patches;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(tire2_surface);
    body.family             = 2;
    body.split_into_patches = split_items_into_patches;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(tire3_surface);
    body.family             = 3;
    body.split_into_patches = split_items_into_patches;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(tire4_surface);
    body.family             = 4;
    body.split_into_patches = split_items_into_patches;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(tire5_surface);
    body.family             = 11;
    body.split_into_patches = split_items_into_patches;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(bunny_surface);
    body.family             = 5;
    body.split_into_patches = split_items_into_patches;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(bunny2_surface);
    body.family             = 6;
    body.split_into_patches = split_items_into_patches;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(bunny3_surface);
    body.family             = 7;
    body.split_into_patches = split_items_into_patches;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(openbox_surface);
    body.family             = 8;
    body.split_into_patches = split_items_into_patches;
    // The openbox is fully fixed in the FE problem. Treat it as a static
    // collision obstacle by skipping contact-force coupling for this body.
    body.skip_self_contact_forces = true;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(armadilo_surface);
    body.family             = 9;
    body.split_into_patches = split_items_into_patches;
    bodies.push_back(std::move(body));
  }
  {
    DemeMeshCollisionBody body;
    body.surface            = std::move(beam_surface);
    body.family             = 10;
    body.split_into_patches = false;
    bodies.push_back(std::move(body));
  }

  std::cout << "Creating DEME collision system...\n" << std::flush;
  auto collision_system = std::make_unique<DemeMeshCollisionSystem>(
      std::move(bodies), mu_s, mu_k, contact_E, contact_cor, self_collision,
      dt);
  std::cout << "DEME collision system ready.\n";

  // Map collision vertices to unified FE coefficients (for gather/scatter).
  std::vector<int> h_coll_coef(static_cast<size_t>(n_coll_nodes), 0);
  std::vector<double> h_coll_zoff(static_cast<size_t>(n_coll_nodes), 0.0);

  for (int i = 0; i < teapot_surf_verts; ++i) {
    const int global_node = teapot_surf_node_ids[static_cast<size_t>(i)];
    const int local_node  = global_node - inst_teapot.node_offset;
    h_coll_coef[static_cast<size_t>(i)] = teapot_coef_off + local_node;
  }
  for (int i = 0; i < tire_surf_verts; ++i) {
    const int global_node = tire_surf_node_ids[static_cast<size_t>(i)];
    const int local_node  = global_node - inst_tire.node_offset;
    const int idx         = teapot_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = tire_coef_off + local_node;
  }
  for (int i = 0; i < tire2_surf_verts; ++i) {
    const int global_node = tire2_surf_node_ids[static_cast<size_t>(i)];
    const int local_node  = global_node - inst_tire2.node_offset;
    const int idx         = teapot_surf_verts + tire_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = tire2_coef_off + local_node;
  }
  for (int i = 0; i < tire3_surf_verts; ++i) {
    const int global_node = tire3_surf_node_ids[static_cast<size_t>(i)];
    const int local_node  = global_node - inst_tire3.node_offset;
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = tire3_coef_off + local_node;
  }
  for (int i = 0; i < tire4_surf_verts; ++i) {
    const int global_node = tire4_surf_node_ids[static_cast<size_t>(i)];
    const int local_node  = global_node - inst_tire4.node_offset;
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = tire4_coef_off + local_node;
  }
  for (int i = 0; i < tire5_surf_verts; ++i) {
    const int global_node = tire5_surf_node_ids[static_cast<size_t>(i)];
    const int local_node  = global_node - inst_tire5.node_offset;
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = tire5_coef_off + local_node;
  }
  for (int i = 0; i < bunny_surf_verts; ++i) {
    const int global_node = bunny_surf_node_ids[static_cast<size_t>(i)];
    const int local_node  = global_node - inst_bunny.node_offset;
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = bunny_coef_off + local_node;
  }
  for (int i = 0; i < bunny2_surf_verts; ++i) {
    const int global_node = bunny2_surf_node_ids[static_cast<size_t>(i)];
    const int local_node  = global_node - inst_bunny2.node_offset;
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
                    bunny_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = bunny2_coef_off + local_node;
  }
  for (int i = 0; i < bunny3_surf_verts; ++i) {
    const int global_node = bunny3_surf_node_ids[static_cast<size_t>(i)];
    const int local_node  = global_node - inst_bunny3.node_offset;
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
                    bunny_surf_verts + bunny2_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = bunny3_coef_off + local_node;
  }
  for (int i = 0; i < openbox_surf_verts; ++i) {
    const int global_node = openbox_surf_node_ids[static_cast<size_t>(i)];
    const int local_node  = global_node - inst_openbox.node_offset;
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
                    bunny_surf_verts + bunny2_surf_verts + bunny3_surf_verts +
                    i;
    h_coll_coef[static_cast<size_t>(idx)] = openbox_coef_off + local_node;
  }
  for (int i = 0; i < armadilo_surf_verts; ++i) {
    const int global_node = armadilo_surf_node_ids[static_cast<size_t>(i)];
    const int local_node  = global_node - inst_armadilo.node_offset;
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
                    bunny_surf_verts + bunny2_surf_verts + bunny3_surf_verts +
                    openbox_surf_verts + i;
    h_coll_coef[static_cast<size_t>(idx)] = armadilo_coef_off + local_node;
  }
  for (int i = 0; i < beam_surf_verts; ++i) {
    const int idx = teapot_surf_verts + tire_surf_verts + tire2_surf_verts +
                    tire3_surf_verts + tire4_surf_verts + tire5_surf_verts +
                    bunny_surf_verts + bunny2_surf_verts + bunny3_surf_verts +
                    openbox_surf_verts + armadilo_surf_verts + i;
    const int ancf_node = beam_surf_ancf_node_ids[static_cast<size_t>(i)];
    h_coll_coef[static_cast<size_t>(idx)] = beam_coef_off + ancf_node * 4 + 0;
    // bottom vertices are [0..beam_nodes), top are [beam_nodes..2*beam_nodes)
    const bool is_top                     = (i >= beam_nodes);
    h_coll_zoff[static_cast<size_t>(idx)] = (is_top ? 0.5 : -0.5) * beam_h;
  }

  // Validate coefficient map ranges.
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
  // Gravity (unified external force buffer template)
  // -------------------------------------------------------------------------
  const int total_dofs = problem.GetTotalDofs();
  std::vector<double> h_f_gravity(static_cast<size_t>(total_dofs), 0.0);

  const auto teapot_lump =
      LumpedMassFromFeat10(*teapot_data, inst_teapot.num_nodes);
  const auto tire_lump = LumpedMassFromFeat10(*tire_data, inst_tire.num_nodes);
  const auto tire2_lump =
      LumpedMassFromFeat10(*tire2_data, inst_tire2.num_nodes);
  const auto tire3_lump =
      LumpedMassFromFeat10(*tire3_data, inst_tire3.num_nodes);
  const auto tire4_lump =
      LumpedMassFromFeat10(*tire4_data, inst_tire4.num_nodes);
  const auto tire5_lump =
      LumpedMassFromFeat10(*tire5_data, inst_tire5.num_nodes);
  const auto bunny_lump =
      LumpedMassFromFeat10(*bunny_data, inst_bunny.num_nodes);
  const auto bunny2_lump =
      LumpedMassFromFeat10(*bunny2_data, inst_bunny2.num_nodes);
  const auto bunny3_lump =
      LumpedMassFromFeat10(*bunny3_data, inst_bunny3.num_nodes);
  const auto armadilo_lump =
      LumpedMassFromFeat10(*armadilo_data, inst_armadilo.num_nodes);

  for (int node = 0; node < inst_teapot.num_nodes; ++node) {
    const int coef = teapot_coef_off + node;
    h_f_gravity[static_cast<size_t>(3 * coef + 2)] +=
        teapot_lump[static_cast<size_t>(node)] * gravity;
  }
  for (int node = 0; node < inst_tire.num_nodes; ++node) {
    const int coef = tire_coef_off + node;
    h_f_gravity[static_cast<size_t>(3 * coef + 2)] +=
        tire_lump[static_cast<size_t>(node)] * gravity;
  }
  for (int node = 0; node < inst_tire2.num_nodes; ++node) {
    const int coef = tire2_coef_off + node;
    h_f_gravity[static_cast<size_t>(3 * coef + 2)] +=
        tire2_lump[static_cast<size_t>(node)] * gravity;
  }
  for (int node = 0; node < inst_tire3.num_nodes; ++node) {
    const int coef = tire3_coef_off + node;
    h_f_gravity[static_cast<size_t>(3 * coef + 2)] +=
        tire3_lump[static_cast<size_t>(node)] * gravity;
  }
  for (int node = 0; node < inst_tire4.num_nodes; ++node) {
    const int coef = tire4_coef_off + node;
    h_f_gravity[static_cast<size_t>(3 * coef + 2)] +=
        tire4_lump[static_cast<size_t>(node)] * gravity;
  }
  for (int node = 0; node < inst_tire5.num_nodes; ++node) {
    const int coef = tire5_coef_off + node;
    h_f_gravity[static_cast<size_t>(3 * coef + 2)] +=
        tire5_lump[static_cast<size_t>(node)] * gravity;
  }
  for (int node = 0; node < inst_bunny.num_nodes; ++node) {
    const int coef = bunny_coef_off + node;
    h_f_gravity[static_cast<size_t>(3 * coef + 2)] +=
        bunny_lump[static_cast<size_t>(node)] * gravity;
  }
  for (int node = 0; node < inst_bunny2.num_nodes; ++node) {
    const int coef = bunny2_coef_off + node;
    h_f_gravity[static_cast<size_t>(3 * coef + 2)] +=
        bunny2_lump[static_cast<size_t>(node)] * gravity;
  }
  for (int node = 0; node < inst_bunny3.num_nodes; ++node) {
    const int coef = bunny3_coef_off + node;
    h_f_gravity[static_cast<size_t>(3 * coef + 2)] +=
        bunny3_lump[static_cast<size_t>(node)] * gravity;
  }
  for (int node = 0; node < inst_armadilo.num_nodes; ++node) {
    const int coef = armadilo_coef_off + node;
    h_f_gravity[static_cast<size_t>(3 * coef + 2)] +=
        armadilo_lump[static_cast<size_t>(node)] * gravity;
  }

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
    // 1) Update collision node buffer from the unified FE state.
    GatherCollisionNodesColumnMajor(d_coll_nodes, n_coll_nodes, state.d_x12,
                                    state.d_y12, state.d_z12, d_coll_coef,
                                    d_coll_zoff);
    HANDLE_ERROR(cudaGetLastError());

    // 2) Collision detection + contact forces.
    CollisionSystemInput coll_in;
    coll_in.d_nodes_xyz = d_coll_nodes;
    coll_in.n_nodes     = n_coll_nodes;
    coll_in.d_vel_xyz   = nullptr;  // optional
    coll_in.dt          = dt;
    collision_system->Step(coll_in, coll_params);
    // DEME may use internal streams; synchronize here so any runtime errors
    // surface at the correct time step.
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());
    const int num_contacts = collision_system->GetNumContacts();

    // 3) External forces = gravity + contact (add onto unified buffer).
    double* d_f_ext = solver.GetExternalForceDevicePtr();
    HANDLE_ERROR(cudaMemcpy(d_f_ext, d_f_gravity, total_dofs * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    ScatterCollisionForcesToExternal(
        collision_system->GetExternalForcesDevicePtr(), n_coll_nodes,
        d_coll_coef, d_f_ext);
    HANDLE_ERROR(cudaGetLastError());

    // 4) Co-simulation Newton step.
    solver.Solve();

    // 5) Export.
    if (vtu_every > 0 && (step % vtu_every) == 0) {
      Eigen::VectorXd bx, by, bz;
      beam_data->RetrievePositionToCPU(bx, by, bz);
      {
        std::ostringstream fn;
        fn << "output/multiitem_drop/beam_" << std::setw(6) << std::setfill('0')
           << step << ".vtu";
        ANCFCPUUtils::VisualizationUtils::ExportANCF3443ToVTU(
            bx, by, bz, beam_conn, beam_h, fn.str());
      }

      auto export_t10 =
          [&](const std::string& prefix, GPU_FEAT10_Data& data,
              const Eigen::VectorXd& x0, const Eigen::VectorXd& y0,
              const Eigen::VectorXd& z0, const Eigen::MatrixXi& elems) {
            Eigen::VectorXd x, y, z;
            data.RetrievePositionToCPU(x, y, z);
            const int n = static_cast<int>(x.size());
            Eigen::MatrixXd cur(n, 3);
            Eigen::VectorXd disp(n * 3);
            for (int i = 0; i < n; ++i) {
              cur(i, 0)       = x(i);
              cur(i, 1)       = y(i);
              cur(i, 2)       = z(i);
              disp(3 * i + 0) = x(i) - x0(i);
              disp(3 * i + 1) = y(i) - y0(i);
              disp(3 * i + 2) = z(i) - z0(i);
            }
            std::ostringstream fn;
            fn << "output/multiitem_drop/" << prefix << "_" << std::setw(6)
               << std::setfill('0') << step << ".vtu";
            ANCFCPUUtils::VisualizationUtils::ExportMeshWithDisplacement(
                cur, elems, disp, fn.str());
          };

      export_t10("teapot", *teapot_data, teapot_x0, teapot_y0, teapot_z0,
                 teapot_elems_local);
      export_t10("tire", *tire_data, tire_x0, tire_y0, tire_z0,
                 tire_elems_local);
      export_t10("tire2", *tire2_data, tire2_x0, tire2_y0, tire2_z0,
                 tire2_elems_local);
      export_t10("tire3", *tire3_data, tire3_x0, tire3_y0, tire3_z0,
                 tire3_elems_local);
      export_t10("tire4", *tire4_data, tire4_x0, tire4_y0, tire4_z0,
                 tire4_elems_local);
      export_t10("tire5", *tire5_data, tire5_x0, tire5_y0, tire5_z0,
                 tire5_elems_local);
      export_t10("bunny", *bunny_data, bunny_x0, bunny_y0, bunny_z0,
                 bunny_elems_local);
      export_t10("bunny2", *bunny2_data, bunny2_x0, bunny2_y0, bunny2_z0,
                 bunny2_elems_local);
      export_t10("bunny3", *bunny3_data, bunny3_x0, bunny3_y0, bunny3_z0,
                 bunny3_elems_local);
      export_t10("armadilo", *armadilo_data, armadilo_x0, armadilo_y0,
                 armadilo_z0, armadilo_elems_local);
      export_t10("openbox", *openbox_data, openbox_x0, openbox_y0, openbox_z0,
                 openbox_elems_local);
    }

    if (step % 20 == 0) {
      std::cout << "step=" << std::setw(5) << step
                << " contacts=" << std::setw(6) << num_contacts << "\n";
    }
  }

  // -------------------------------------------------------------------------
  // Cleanup
  // -------------------------------------------------------------------------
  HANDLE_ERROR(cudaFree(d_f_gravity));
  HANDLE_ERROR(cudaFree(d_coll_nodes));
  HANDLE_ERROR(cudaFree(d_coll_coef));
  HANDLE_ERROR(cudaFree(d_coll_zoff));

  teapot_data->Destroy();
  tire_data->Destroy();
  tire2_data->Destroy();
  tire3_data->Destroy();
  tire4_data->Destroy();
  tire5_data->Destroy();
  bunny_data->Destroy();
  bunny2_data->Destroy();
  bunny3_data->Destroy();
  armadilo_data->Destroy();
  openbox_data->Destroy();
  beam_data->Destroy();
  return 0;
}
