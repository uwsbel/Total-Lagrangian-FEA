/**
 * FEAT10 Piston-Pump Cylindrical-Joint Demo (Holistic Solver)
 *
 * Loads the pump cup and piston T10 meshes as two independent FEAT10 blocks,
 * aligns the piston shaft with the cup axis, fixes the outer annulus of the
 * cup bottom face against the world, and connects the cup and piston through a
 * cylindrical joint evaluated by the mixed holistic constraint path:
 *   FEMultiElementProblem + MixedConstraintSystem + HolisticNewtonSolver
 *
 * Output:
 *   output/engineering_joint/piston_pump_cylindrical_holistic_cup_XXXXXX.vtu
 *   output/engineering_joint/piston_pump_cylindrical_holistic_piston_XXXXXX.vtu
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../lib_src/constraints/MixedConstraintSystem.h"
#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEMultiElementProblem.h"
#include "../../lib_src/solvers/HolisticNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"

namespace {

constexpr double kDt             = 1e-4;
constexpr int kNumStepsDefault   = 1200;
constexpr int kExportIntervalDef = 20;
constexpr int kForceReleaseSteps = 200;

constexpr double kCupOuterAnnulusRadiusFraction = 0.70;
constexpr double kBottomFaceTolerance           = 1e-8;
constexpr double kJointOffset                   = 0.002;

constexpr double kHandleAxialForce  = 30.0;
constexpr double kHandleCoupleForce = 10.0;

const SolidMaterialProperties kCupMaterial =
    SolidMaterialProperties::SVK(2.0e7,   // E
                                 0.32,    // nu
                                 1150.0,  // rho0
                                 2.0e3,   // eta_damp
                                 2.0e3    // lambda_damp
    );

const SolidMaterialProperties kPistonMaterial =
    SolidMaterialProperties::SVK(5.0e7,   // E
                                 0.32,    // nu
                                 1750.0,  // rho0
                                 1.5e3,   // eta_damp
                                 1.5e3    // lambda_damp
    );

struct Bounds {
  Eigen::Vector3d min = Eigen::Vector3d::Zero();
  Eigen::Vector3d max = Eigen::Vector3d::Zero();

  Eigen::Vector3d size() const {
    return max - min;
  }
};

Bounds ComputeInstanceBounds(const Eigen::MatrixXd& all_nodes,
                             const ANCFCPUUtils::MeshInstance& instance) {
  Bounds bounds;
  bounds.min = all_nodes.row(instance.node_offset).transpose();
  bounds.max = bounds.min;
  for (int local_node = 0; local_node < instance.num_nodes; ++local_node) {
    const Eigen::Vector3d p =
        all_nodes.row(instance.node_offset + local_node).transpose();
    bounds.min = bounds.min.cwiseMin(p);
    bounds.max = bounds.max.cwiseMax(p);
  }
  return bounds;
}

Eigen::MatrixXd ExtractLocalNodes(const Eigen::MatrixXd& global_nodes,
                                  const ANCFCPUUtils::MeshInstance& instance) {
  return global_nodes.middleRows(instance.node_offset, instance.num_nodes);
}

Eigen::MatrixXi ExtractLocalElements(const Eigen::MatrixXi& global_elements,
                                     const ANCFCPUUtils::MeshInstance& instance) {
  Eigen::MatrixXi local =
      global_elements.middleRows(instance.element_offset, instance.num_elements);
  local.array() -= instance.node_offset;
  return local;
}

Eigen::VectorXd ExtractAxis(const Eigen::MatrixXd& nodes, int axis) {
  Eigen::VectorXd values(nodes.rows());
  for (int i = 0; i < nodes.rows(); ++i) {
    values(i) = nodes(i, axis);
  }
  return values;
}

Eigen::Vector3d EvaluateCurrentPointPosition(
    const MixedConstraintPointBinding& point, const Eigen::VectorXd& x,
    const Eigen::VectorXd& y, const Eigen::VectorXd& z) {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  for (int i = 0; i < point.count; ++i) {
    const int coef = point.coef_indices[i];
    const double weight = point.weights[i];
    position += weight * Eigen::Vector3d(x(coef), y(coef), z(coef));
  }
  return position;
}

void RetrieveUnifiedPositions(const FEStateBuffer& state, Eigen::VectorXd* x,
                              Eigen::VectorXd* y, Eigen::VectorXd* z) {
  x->resize(state.total_coef);
  y->resize(state.total_coef);
  z->resize(state.total_coef);
  HANDLE_ERROR(cudaMemcpy(x->data(), state.d_x12,
                          static_cast<size_t>(state.total_coef) *
                              sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(y->data(), state.d_y12,
                          static_cast<size_t>(state.total_coef) *
                              sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(z->data(), state.d_z12,
                          static_cast<size_t>(state.total_coef) *
                              sizeof(double),
                          cudaMemcpyDeviceToHost));
}

Eigen::Vector3d ComputePistonShaftCenter(const Eigen::MatrixXd& all_nodes,
                                         const ANCFCPUUtils::MeshInstance& inst,
                                         const Bounds& bounds) {
  const double height = bounds.size().z();
  const double z_lo   = bounds.min.z() + 0.16 * height;
  const double z_hi   = bounds.min.z() + 0.82 * height;
  const double r_cut  = 0.025;

  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  int count              = 0;
  for (int local_node = 0; local_node < inst.num_nodes; ++local_node) {
    const Eigen::Vector3d p =
        all_nodes.row(inst.node_offset + local_node).transpose();
    const double r = std::hypot(p.x(), p.y());
    if (p.z() < z_lo || p.z() > z_hi || r > r_cut) {
      continue;
    }
    center += p;
    ++count;
  }

  if (count == 0) {
    throw std::runtime_error(
        "Failed to identify piston shaft center from the pump mesh");
  }
  return center / static_cast<double>(count);
}

int SelectCupAxisAnchorNode(const Eigen::MatrixXd& all_nodes,
                            const ANCFCPUUtils::MeshInstance& inst) {
  std::vector<int> axis_nodes;
  axis_nodes.reserve(8);
  for (int local_node = 0; local_node < inst.num_nodes; ++local_node) {
    const int global_node   = inst.node_offset + local_node;
    const Eigen::Vector3d p = all_nodes.row(global_node).transpose();
    if (std::hypot(p.x(), p.y()) < 0.003) {
      axis_nodes.push_back(global_node);
    }
  }

  if (axis_nodes.empty()) {
    throw std::runtime_error(
        "Failed to identify an axis-aligned anchor in the cup mesh");
  }

  std::sort(axis_nodes.begin(), axis_nodes.end(), [&all_nodes](int a, int b) {
    return all_nodes(a, 2) < all_nodes(b, 2);
  });
  return axis_nodes[axis_nodes.size() / 2];
}

std::vector<int> SelectBottomCupOuterAnnulusNodes(
    const Eigen::MatrixXd& all_nodes, const ANCFCPUUtils::MeshInstance& inst,
    const Bounds& bounds, double annulus_radius_fraction,
    double bottom_face_tolerance) {
  const double z_face = bounds.min.z();
  const double r_max =
      std::max(bounds.max.head<2>().norm(), bounds.min.head<2>().norm());
  const double r_cut = annulus_radius_fraction * r_max;

  std::vector<int> nodes;
  for (int local_node = 0; local_node < inst.num_nodes; ++local_node) {
    const int global_node   = inst.node_offset + local_node;
    const Eigen::Vector3d p = all_nodes.row(global_node).transpose();
    const double r          = std::hypot(p.x(), p.y());
    if (std::abs(p.z() - z_face) <= bottom_face_tolerance && r >= r_cut) {
      nodes.push_back(global_node);
    }
  }
  return nodes;
}

Eigen::Vector3d ComputeTopHandleBarCenter(
    const Eigen::MatrixXd& all_nodes, const ANCFCPUUtils::MeshInstance& inst,
    const Bounds& bounds) {
  const double z_max = bounds.max.z();
  const double z_min = z_max - 0.028;

  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  int center_count       = 0;
  for (int local_node = 0; local_node < inst.num_nodes; ++local_node) {
    const int global_node   = inst.node_offset + local_node;
    const Eigen::Vector3d p = all_nodes.row(global_node).transpose();
    if (p.z() >= z_min) {
      center += p;
      ++center_count;
    }
  }

  if (center_count == 0) {
    throw std::runtime_error("Failed to identify the piston handle-bar region");
  }
  return center / static_cast<double>(center_count);
}

std::vector<int> SelectCenteredHandleNodes(
    const Eigen::MatrixXd& all_nodes, const ANCFCPUUtils::MeshInstance& inst,
    const Bounds& bounds, const Eigen::Vector3d& center) {
  (void)bounds;
  struct Window {
    double x_half_width;
    double y_half_width;
    double z_half_width;
  };

  const std::array<Window, 4> windows = {{
      {0.008, 0.006, 0.006},
      {0.012, 0.008, 0.008},
      {0.016, 0.010, 0.010},
      {0.022, 0.012, 0.012},
  }};

  for (const auto& window : windows) {
    std::vector<int> nodes;
    for (int local_node = 0; local_node < inst.num_nodes; ++local_node) {
      const int global_node   = inst.node_offset + local_node;
      const Eigen::Vector3d p = all_nodes.row(global_node).transpose();
      if (std::abs(p.x() - center.x()) <= window.x_half_width &&
          std::abs(p.y() - center.y()) <= window.y_half_width &&
          std::abs(p.z() - center.z()) <= window.z_half_width) {
        nodes.push_back(global_node);
      }
    }
    if (!nodes.empty()) {
      return nodes;
    }
  }

  throw std::runtime_error(
      "Failed to identify a centered handle-bar load region on the piston mesh");
}

std::vector<int> SelectHandleSideNodes(const Eigen::MatrixXd& all_nodes,
                                       const ANCFCPUUtils::MeshInstance& inst,
                                       const Bounds& bounds,
                                       const Eigen::Vector3d& center,
                                       int side_sign) {
  const double z_max = bounds.max.z();
  const double z_min = z_max - 0.028;
  double x_min       = std::numeric_limits<double>::infinity();
  double x_max       = -std::numeric_limits<double>::infinity();
  for (int local_node = 0; local_node < inst.num_nodes; ++local_node) {
    const int global_node   = inst.node_offset + local_node;
    const Eigen::Vector3d p = all_nodes.row(global_node).transpose();
    if (p.z() < z_min) {
      continue;
    }
    x_min = std::min(x_min, p.x());
    x_max = std::max(x_max, p.x());
  }

  struct Window {
    double x_span;
    double y_half_width;
    double z_half_width;
  };

  const std::array<Window, 4> windows = {{
      {0.012, 0.010, 0.010},
      {0.018, 0.012, 0.012},
      {0.024, 0.014, 0.014},
      {0.032, 0.016, 0.016},
  }};

  for (const auto& window : windows) {
    std::vector<int> nodes;
    for (int local_node = 0; local_node < inst.num_nodes; ++local_node) {
      const int global_node   = inst.node_offset + local_node;
      const Eigen::Vector3d p = all_nodes.row(global_node).transpose();
      const bool x_ok = side_sign < 0 ? (p.x() <= x_min + window.x_span)
                                      : (p.x() >= x_max - window.x_span);
      if (x_ok && std::abs(p.y() - center.y()) <= window.y_half_width &&
          std::abs(p.z() - center.z()) <= window.z_half_width) {
        nodes.push_back(global_node);
      }
    }
    if (!nodes.empty()) {
      return nodes;
    }
  }

  throw std::runtime_error(
      side_sign < 0 ? "Failed to identify the left handle-bar end region"
                    : "Failed to identify the right handle-bar end region");
}

int SelectHandleTipNode(const Eigen::MatrixXd& all_nodes,
                        const std::vector<int>& handle_nodes) {
  if (handle_nodes.empty()) {
    throw std::runtime_error("Handle-node selection returned no nodes");
  }

  return *std::max_element(
      handle_nodes.begin(), handle_nodes.end(),
      [&all_nodes](int a, int b) { return all_nodes(a, 0) < all_nodes(b, 0); });
}

std::vector<int> ToLocalNodeIndices(const std::vector<int>& global_nodes,
                                    int node_offset, int node_count) {
  std::vector<int> local_nodes;
  local_nodes.reserve(global_nodes.size());
  for (int global_node : global_nodes) {
    const int local_node = global_node - node_offset;
    if (local_node < 0 || local_node >= node_count) {
      throw std::out_of_range(
          "Global node does not belong to the requested mesh instance");
    }
    local_nodes.push_back(local_node);
  }
  return local_nodes;
}

std::vector<int> MakeGlobalCoefIndices(const std::vector<int>& local_nodes,
                                       int coef_offset) {
  std::vector<int> global_coef_indices;
  global_coef_indices.reserve(local_nodes.size());
  for (int local_node : local_nodes) {
    global_coef_indices.push_back(coef_offset + local_node);
  }
  return global_coef_indices;
}

std::string MakeOutputPath(const std::string& body_name, int frame) {
  std::ostringstream oss;
  oss << "output/engineering_joint/piston_pump_cylindrical_holistic_"
      << body_name << "_" << std::setw(6) << std::setfill('0') << frame
      << ".vtu";
  return oss.str();
}

void AddDistributedForce(Eigen::VectorXd* h_f_ext,
                         const std::vector<int>& global_coef_indices,
                         const Eigen::Vector3d& total_force) {
  if (global_coef_indices.empty()) {
    return;
  }
  const Eigen::Vector3d force_per_node =
      total_force / static_cast<double>(global_coef_indices.size());
  for (int coef : global_coef_indices) {
    (*h_f_ext)(3 * coef + 0) += force_per_node.x();
    (*h_f_ext)(3 * coef + 1) += force_per_node.y();
    (*h_f_ext)(3 * coef + 2) += force_per_node.z();
  }
}

Eigen::Vector3d ComputeCurrentNodeCentroid(const std::vector<int>& global_coef_indices,
                                           const Eigen::VectorXd& x,
                                           const Eigen::VectorXd& y,
                                           const Eigen::VectorXd& z) {
  if (global_coef_indices.empty()) {
    throw std::runtime_error("Cannot compute centroid of an empty node set");
  }

  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (int coef : global_coef_indices) {
    center += Eigen::Vector3d(x(coef), y(coef), z(coef));
  }
  return center / static_cast<double>(global_coef_indices.size());
}

void AddNodeToWorldCD(MixedConstraintSystem* constraints, int global_coef,
                      const Eigen::Vector3d& world_point) {
  const MixedConstraintPointBinding point =
      constraints->MakeCoefficientBinding(global_coef);
  constraints->AddPointToWorldCDAxis(point, 0, world_point.x());
  constraints->AddPointToWorldCDAxis(point, 1, world_point.y());
  constraints->AddPointToWorldCDAxis(point, 2, world_point.z());
}

}  // namespace

int main(int argc, char** argv) {
  int max_steps       = kNumStepsDefault;
  int export_interval = kExportIntervalDef;
  if (argc > 1) {
    const int parsed_steps = std::atoi(argv[1]);
    if (parsed_steps > 0) {
      max_steps = parsed_steps;
    }
  }
  if (argc > 2) {
    const int parsed_interval = std::atoi(argv[2]);
    if (parsed_interval > 0) {
      export_interval = parsed_interval;
    }
  }

  std::cout << "===============================================\n";
  std::cout << "FEAT10 Piston Pump Cylindrical (Holistic Solve)\n";
  std::cout << "===============================================\n";
  std::cout << "steps=" << max_steps << " export_interval=" << export_interval
            << "\n";

  std::filesystem::create_directories("output/engineering_joint");

  ANCFCPUUtils::MeshManager mesh_manager;
  const std::string mesh_dir = "data/meshes/T10/pump_cynlindrical";
  const int mesh_cup =
      mesh_manager.LoadMesh(mesh_dir + "/outer_cup.1.node",
                            mesh_dir + "/outer_cup.1.ele", "cup", kCupMaterial);
  const int mesh_piston = mesh_manager.LoadMesh(mesh_dir + "/piston.1.node",
                                                mesh_dir + "/piston.1.ele",
                                                "piston", kPistonMaterial);
  if (mesh_cup < 0 || mesh_piston < 0) {
    std::cerr << "Failed to load pump meshes from " << mesh_dir << std::endl;
    return 1;
  }

  const auto& inst_cup    = mesh_manager.GetMeshInstance(mesh_cup);
  const auto& inst_piston = mesh_manager.GetMeshInstance(mesh_piston);

  Bounds piston_bounds_initial =
      ComputeInstanceBounds(mesh_manager.GetAllNodes(), inst_piston);
  const Eigen::Vector3d piston_shaft_center = ComputePistonShaftCenter(
      mesh_manager.GetAllNodes(), inst_piston, piston_bounds_initial);
  mesh_manager.TranslateMesh(mesh_piston, -piston_shaft_center.x(),
                             -piston_shaft_center.y(), 0.0);

  const Eigen::MatrixXd& all_nodes = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& all_elems = mesh_manager.GetAllElements();

  const Eigen::MatrixXd cup_nodes    = ExtractLocalNodes(all_nodes, inst_cup);
  const Eigen::MatrixXd piston_nodes = ExtractLocalNodes(all_nodes, inst_piston);
  const Eigen::MatrixXi cup_elems    = ExtractLocalElements(all_elems, inst_cup);
  const Eigen::MatrixXi piston_elems =
      ExtractLocalElements(all_elems, inst_piston);

  const Bounds cup_bounds    = ComputeInstanceBounds(all_nodes, inst_cup);
  const Bounds piston_bounds = ComputeInstanceBounds(all_nodes, inst_piston);
  const double cup_outer_annulus_r_cut =
      kCupOuterAnnulusRadiusFraction * cup_bounds.max.head<2>().norm();
  const int cup_axis_anchor_node = SelectCupAxisAnchorNode(all_nodes, inst_cup);
  const Eigen::Vector3d cup_axis_anchor =
      all_nodes.row(cup_axis_anchor_node).transpose();
  const double overlap_z_min =
      std::max(piston_bounds.min.z(), cup_bounds.min.z());
  const double overlap_z_max =
      std::min(piston_bounds.max.z(), cup_bounds.max.z());
  if (overlap_z_max <= overlap_z_min) {
    throw std::runtime_error(
        "Cup and piston do not overlap along the cylindrical-joint axis");
  }
  const double piston_joint_z =
      std::clamp(cup_bounds.max.z() - 0.05, overlap_z_min, overlap_z_max);
  const Eigen::Vector3d piston_axis_point(0.0, 0.0, piston_joint_z);

  std::vector<int> fixed_cup_nodes = SelectBottomCupOuterAnnulusNodes(
      all_nodes, inst_cup, cup_bounds, kCupOuterAnnulusRadiusFraction,
      kBottomFaceTolerance);
  fixed_cup_nodes.erase(
      std::remove(fixed_cup_nodes.begin(), fixed_cup_nodes.end(),
                  cup_axis_anchor_node),
      fixed_cup_nodes.end());
  const Eigen::Vector3d handle_bar_center =
      ComputeTopHandleBarCenter(all_nodes, inst_piston, piston_bounds);
  const std::vector<int> center_handle_nodes = SelectCenteredHandleNodes(
      all_nodes, inst_piston, piston_bounds, handle_bar_center);
  const std::vector<int> left_handle_nodes = SelectHandleSideNodes(
      all_nodes, inst_piston, piston_bounds, handle_bar_center, -1);
  const std::vector<int> right_handle_nodes = SelectHandleSideNodes(
      all_nodes, inst_piston, piston_bounds, handle_bar_center, 1);
  const int handle_tip_node =
      SelectHandleTipNode(all_nodes, right_handle_nodes);

  auto cup_data =
      std::make_unique<GPU_FEAT10_Data>(inst_cup.num_elements, inst_cup.num_nodes);
  cup_data->Initialize();
  cup_data->Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                  Quadrature::tet5pt_z, Quadrature::tet5pt_weights,
                  ExtractAxis(cup_nodes, 0), ExtractAxis(cup_nodes, 1),
                  ExtractAxis(cup_nodes, 2), cup_elems);
  cup_data->ApplyMaterial(kCupMaterial);
  cup_data->CalcDnDuPre();
  cup_data->CalcMassMatrix();

  auto piston_data = std::make_unique<GPU_FEAT10_Data>(inst_piston.num_elements,
                                                       inst_piston.num_nodes);
  piston_data->Initialize();
  piston_data->Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                     Quadrature::tet5pt_z, Quadrature::tet5pt_weights,
                     ExtractAxis(piston_nodes, 0), ExtractAxis(piston_nodes, 1),
                     ExtractAxis(piston_nodes, 2), piston_elems);
  piston_data->ApplyMaterial(kPistonMaterial);
  piston_data->CalcDnDuPre();
  piston_data->CalcMassMatrix();

  FEMultiElementProblem problem;
  const int cup_block = problem.AddElementBlock(cup_data.get(), TYPE_T10);
  const int piston_block = problem.AddElementBlock(piston_data.get(), TYPE_T10);
  problem.Finalize();

  FEStateBuffer& state = problem.GetStateBuffer();
  const int cup_coef_offset =
      state.blocks[static_cast<size_t>(cup_block)].coef_offset;
  const int piston_coef_offset =
      state.blocks[static_cast<size_t>(piston_block)].coef_offset;

  const std::vector<int> fixed_cup_nodes_local =
      ToLocalNodeIndices(fixed_cup_nodes, inst_cup.node_offset, inst_cup.num_nodes);
  const std::vector<int> center_handle_nodes_local = ToLocalNodeIndices(
      center_handle_nodes, inst_piston.node_offset, inst_piston.num_nodes);
  const std::vector<int> left_handle_nodes_local = ToLocalNodeIndices(
      left_handle_nodes, inst_piston.node_offset, inst_piston.num_nodes);
  const std::vector<int> right_handle_nodes_local = ToLocalNodeIndices(
      right_handle_nodes, inst_piston.node_offset, inst_piston.num_nodes);
  const int handle_tip_local_node = handle_tip_node - inst_piston.node_offset;
  if (handle_tip_local_node < 0 || handle_tip_local_node >= inst_piston.num_nodes) {
    throw std::out_of_range("Handle tip node does not belong to the piston block");
  }

  const std::vector<int> center_handle_coef_indices =
      MakeGlobalCoefIndices(center_handle_nodes_local, piston_coef_offset);
  const std::vector<int> left_handle_coef_indices =
      MakeGlobalCoefIndices(left_handle_nodes_local, piston_coef_offset);
  const std::vector<int> right_handle_coef_indices =
      MakeGlobalCoefIndices(right_handle_nodes_local, piston_coef_offset);
  const int handle_tip_coef = piston_coef_offset + handle_tip_local_node;

  std::cout << "cup:      " << inst_cup.num_nodes << " nodes, "
            << inst_cup.num_elements << " elements\n";
  std::cout << "piston:   " << inst_piston.num_nodes << " nodes, "
            << inst_piston.num_elements << " elements\n";
  std::cout << "total:    " << mesh_manager.GetTotalNodes() << " nodes, "
            << mesh_manager.GetTotalElements() << " elements\n";
  std::cout << "aligned piston by xy shift = [" << (-piston_shaft_center.x())
            << ", " << (-piston_shaft_center.y()) << "]\n";
  std::cout << "cup bounds z = [" << cup_bounds.min.z() << ", "
            << cup_bounds.max.z() << "]\n";
  std::cout << "piston bounds z = [" << piston_bounds.min.z() << ", "
            << piston_bounds.max.z() << "]\n";
  std::cout << "cup outer annulus r cut = " << cup_outer_annulus_r_cut << "\n";
  std::cout << "joint overlap z = [" << overlap_z_min << ", " << overlap_z_max
            << "] piston_joint_z=" << piston_joint_z << "\n";
  std::cout << "fixed cup nodes: " << fixed_cup_nodes_local.size() << "\n";
  std::cout << "center handle nodes: " << center_handle_nodes_local.size() << "\n";
  std::cout << "left handle nodes:   " << left_handle_nodes_local.size() << "\n";
  std::cout << "right handle nodes:  " << right_handle_nodes_local.size() << "\n";
  const bool anchor_in_fixed_set =
      std::find(fixed_cup_nodes.begin(), fixed_cup_nodes.end(),
                cup_axis_anchor_node) != fixed_cup_nodes.end();
  std::cout << "cup axis anchor node: " << cup_axis_anchor_node << " at ["
            << cup_axis_anchor.transpose() << "]"
            << " anchor_in_fixed_set=" << (anchor_in_fixed_set ? "yes" : "no")
            << "\n";
  std::cout << "handle tip node: " << handle_tip_node << " at ["
            << all_nodes.row(handle_tip_node) << "]\n";
  std::cout << "handle center:  [" << handle_bar_center.transpose() << "]\n";
  std::cout << "axial handle force = " << kHandleAxialForce
            << " couple side force = " << kHandleCoupleForce
            << " release_last_steps = " << kForceReleaseSteps << "\n";

  MixedConstraintSystem constraints(&problem);
  for (int local_node : fixed_cup_nodes_local) {
    AddNodeToWorldCD(&constraints, cup_coef_offset + local_node,
                     cup_nodes.row(local_node).transpose());
  }
  constraints.AddCylindricalJoint(cup_block, piston_block, cup_axis_anchor,
                                  piston_axis_point, Eigen::Vector3d::UnitZ(),
                                  kJointOffset, 1.0, 1.0);
  const MixedConstraintPointBinding piston_axis_reference =
      constraints.LocateReferencePoint(piston_block, piston_axis_point);
  constraints.Finalize();

  HolisticNewtonParams params;
  params.inner_atol = 1e-4;
  params.inner_rtol = 1e-4;
  params.outer_tol = 1e-7;
  params.rho = 1e12;
  params.max_outer = 8;
  params.max_inner = 12;
  params.time_step = kDt;
  params.enable_line_search = false;

  HolisticNewtonSolver solver(&problem, &constraints);
  solver.SetParameters(&params);
  solver.Setup();

  Eigen::VectorXd x_curr, y_curr, z_curr;
  RetrieveUnifiedPositions(state, &x_curr, &y_curr, &z_curr);
  const Eigen::Vector3d piston_axis_initial = EvaluateCurrentPointPosition(
      piston_axis_reference, x_curr, y_curr, z_curr);
  const Eigen::Vector3d handle_tip_initial(x_curr(handle_tip_coef),
                                           y_curr(handle_tip_coef),
                                           z_curr(handle_tip_coef));
  const Eigen::Vector3d left_handle_initial =
      ComputeCurrentNodeCentroid(left_handle_coef_indices, x_curr, y_curr, z_curr);
  const Eigen::Vector3d right_handle_initial = ComputeCurrentNodeCentroid(
      right_handle_coef_indices, x_curr, y_curr, z_curr);
  Eigen::Vector3d initial_bar_direction =
      right_handle_initial - left_handle_initial;
  if (initial_bar_direction.norm() < 1e-12) {
    initial_bar_direction = Eigen::Vector3d::UnitX();
  } else {
    initial_bar_direction.normalize();
  }

  auto update_handle_forces = [&](const Eigen::VectorXd& x_pos,
                                  const Eigen::VectorXd& y_pos,
                                  const Eigen::VectorXd& z_pos,
                                  bool apply_forces) {
    Eigen::VectorXd h_f_ext = Eigen::VectorXd::Zero(problem.GetTotalDofs());
    const Eigen::Vector3d left_center =
        ComputeCurrentNodeCentroid(left_handle_coef_indices, x_pos, y_pos, z_pos);
    const Eigen::Vector3d right_center = ComputeCurrentNodeCentroid(
        right_handle_coef_indices, x_pos, y_pos, z_pos);

    Eigen::Vector3d bar_direction = right_center - left_center;
    if (bar_direction.norm() < 1e-12) {
      bar_direction = initial_bar_direction;
    } else {
      bar_direction.normalize();
    }

    Eigen::Vector3d couple_direction =
        Eigen::Vector3d::UnitZ().cross(bar_direction);
    if (couple_direction.norm() < 1e-12) {
      couple_direction = Eigen::Vector3d::UnitY();
    } else {
      couple_direction.normalize();
    }

    if (apply_forces) {
      const Eigen::Vector3d left_force =
          0.5 * kHandleAxialForce * Eigen::Vector3d::UnitZ() -
          kHandleCoupleForce * couple_direction;
      const Eigen::Vector3d right_force =
          0.5 * kHandleAxialForce * Eigen::Vector3d::UnitZ() +
          kHandleCoupleForce * couple_direction;

      AddDistributedForce(&h_f_ext, left_handle_coef_indices, left_force);
      AddDistributedForce(&h_f_ext, right_handle_coef_indices, right_force);
    }

    HANDLE_ERROR(cudaMemcpy(state.d_f_ext, h_f_ext.data(),
                            static_cast<size_t>(problem.GetTotalDofs()) *
                                sizeof(double),
                            cudaMemcpyHostToDevice));

    return std::make_pair(bar_direction, couple_direction);
  };

  const auto initial_load_dirs =
      update_handle_forces(x_curr, y_curr, z_curr, true);
  std::cout << "constraints: " << constraints.num_constraints() << "\n";
  std::cout << "initial piston axis point: [" << piston_axis_initial.transpose()
            << "]\n";
  std::cout << "initial handle tip:       [" << handle_tip_initial.transpose()
            << "]\n";
  std::cout << "initial handle bar dir:   ["
            << initial_load_dirs.first.transpose() << "]\n";
  std::cout << "initial couple dir:       ["
            << initial_load_dirs.second.transpose() << "]\n";

  cup_data->WriteOutputVTU(MakeOutputPath("cup", 0));
  piston_data->WriteOutputVTU(MakeOutputPath("piston", 0));

  int output_frame = 1;
  for (int step = 1; step <= max_steps; ++step) {
    const bool apply_forces =
        (max_steps <= kForceReleaseSteps) ||
        (step <= max_steps - kForceReleaseSteps);
    const auto load_dirs =
        update_handle_forces(x_curr, y_curr, z_curr, apply_forces);
    solver.Solve();

    RetrieveUnifiedPositions(state, &x_curr, &y_curr, &z_curr);
    const Eigen::Vector3d piston_axis_current = EvaluateCurrentPointPosition(
        piston_axis_reference, x_curr, y_curr, z_curr);
    const Eigen::Vector3d handle_tip_current(x_curr(handle_tip_coef),
                                             y_curr(handle_tip_coef),
                                             z_curr(handle_tip_coef));
    const Eigen::Vector3d left_handle_current = ComputeCurrentNodeCentroid(
        left_handle_coef_indices, x_curr, y_curr, z_curr);
    const Eigen::Vector3d right_handle_current = ComputeCurrentNodeCentroid(
        right_handle_coef_indices, x_curr, y_curr, z_curr);

    Eigen::VectorXd constraint_values(constraints.num_constraints());
    HANDLE_ERROR(cudaMemcpy(constraint_values.data(),
                            constraints.GetConstraintDevicePtr(),
                            static_cast<size_t>(constraints.num_constraints()) *
                                sizeof(double),
                            cudaMemcpyDeviceToHost));

    std::cout << "step " << step << " piston axis: ["
              << piston_axis_current.transpose() << "]"
              << " axial_disp="
              << (piston_axis_current.z() - piston_axis_initial.z())
              << " radial_drift="
              << std::hypot(piston_axis_current.x(), piston_axis_current.y())
              << " constraint_norm=" << constraint_values.norm() << "\n";
    std::cout << "step " << step << " handle tip:  ["
              << handle_tip_current.transpose() << "]"
              << " tip_disp=["
              << (handle_tip_current - handle_tip_initial).transpose() << "]\n";
    std::cout << "step " << step << " handle bar:  left=["
              << left_handle_current.transpose() << "] right=["
              << right_handle_current.transpose() << "]"
              << " bar_dir=[" << load_dirs.first.transpose() << "]"
              << " couple_dir=[" << load_dirs.second.transpose() << "]"
              << " load_active=" << (apply_forces ? "yes" : "no") << "\n";

    if (step % export_interval == 0) {
      cup_data->WriteOutputVTU(MakeOutputPath("cup", output_frame));
      piston_data->WriteOutputVTU(MakeOutputPath("piston", output_frame));
      ++output_frame;
    }
  }

  cup_data->Destroy();
  piston_data->Destroy();

  std::cout << "Done. Output written to output/engineering_joint/\n";
  return 0;
}
