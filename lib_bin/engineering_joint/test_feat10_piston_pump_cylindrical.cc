/**
 * FEAT10 Piston-Pump Cylindrical-Joint Demo
 *
 * Loads the pump cup and piston T10 meshes, aligns the piston shaft with the
 * cup axis, fixes the outer annulus of the cup bottom face, and connects the
 * two parts with a cylindrical joint. A follower axial pull plus a rotating
 * force couple is applied on the piston's handle bar to drive both sliding
 * and rotation about the shared axis.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../lib_src/elements/FEAT10Data.cuh"
#include "../../lib_src/solvers/FEAT10ConstraintManager.h"
#include "../../lib_src/solvers/SyncedNewton.cuh"
#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
#include "double_pendulum_csv_utils.h"

namespace {

constexpr double kDt             = 5e-4;
constexpr int kNumStepsDefault   = 2000;
constexpr int kExportIntervalDef = 10;
constexpr int kForceReleaseSteps = 0;

constexpr double kCupOuterAnnulusRadiusFraction   = 0.70;
constexpr double kBottomFaceTolerance             = 1e-8;
constexpr double kJointOffset                     = 0.006;
constexpr int kRowsPerWorldFixedGuideFrame        = 9;
constexpr double kPistonJointSearchSlabHalfHeight = 0.0035;
constexpr double kPistonJointSearchRadialCut      = 0.025;
constexpr int kPistonJointSearchSamplesPerWindow  = 96;
constexpr double kPistonJointWindowMarginFraction = 0.05;

constexpr double kHandleAxialForce        = 20.0;
constexpr double kHandleCoupleForce       = 5.0;
constexpr double kHandleBendForceYDefault = 0.0;

const SolidMaterialProperties kCupMaterial =
    SolidMaterialProperties::SVK(3.0e6,   // E
                                 0.32,    // nu
                                 1150.0,  // rho0
                                 2.0e4,   // eta_damp
                                 2.0e4    // lambda_damp
    );

const SolidMaterialProperties kPistonMaterial =
    SolidMaterialProperties::SVK(2.0e8,   // E
                                 0.32,    // nu
                                 1750.0,  // rho0
                                 1.5e3,   // eta_damp
                                 1.5e3    // lambda_damp
    );

enum class GuideMode {
  kDeformableCup   = 0,
  kWorldFixedFrame = 1,
};

enum class LoadMode {
  kCombined  = 0,
  kBendYOnly = 1,
};

std::string NormalizeCliName(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  std::replace(value.begin(), value.end(), '-', '_');
  return value;
}

const char* GuideModeName(GuideMode mode) {
  switch (mode) {
    case GuideMode::kDeformableCup:
      return "deformable_cup";
    case GuideMode::kWorldFixedFrame:
      return "world_fixed_frame";
  }
  return "deformable_cup";
}

const char* LoadModeName(LoadMode mode) {
  switch (mode) {
    case LoadMode::kCombined:
      return "combined";
    case LoadMode::kBendYOnly:
      return "bend_y_only";
  }
  return "combined";
}

bool ParseGuideMode(std::string arg, GuideMode* mode_out) {
  arg = NormalizeCliName(std::move(arg));
  if (arg == "deformable_cup" || arg == "deformable") {
    *mode_out = GuideMode::kDeformableCup;
    return true;
  }
  if (arg == "world_fixed_frame" || arg == "world_fixed" || arg == "fixed") {
    *mode_out = GuideMode::kWorldFixedFrame;
    return true;
  }
  return false;
}

bool ParseLoadMode(std::string arg, LoadMode* mode_out) {
  arg = NormalizeCliName(std::move(arg));
  if (arg == "combined") {
    *mode_out = LoadMode::kCombined;
    return true;
  }
  if (arg == "bend_y_only" || arg == "bendyonly" || arg == "bend_only") {
    *mode_out = LoadMode::kBendYOnly;
    return true;
  }
  return false;
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [steps] [export_interval]"
         " [--bend_force_y=<force>] [--ramp_steps=<steps>] [--csv[=<path>]]"
         " [--guide=<deformable_cup|world_fixed_frame>]"
         " [--load=<combined|bend_y_only>]\n"
      << "       " << argv0
      << " [steps] [export_interval]"
         " [--guide_mode=<deformable_cup|world_fixed_frame>]"
         " [--load_mode=<combined|bend_y_only>]\n"
      << "\n"
      << "Examples:\n"
      << "  " << argv0 << " 400 10 --csv\n"
      << "  " << argv0
      << " 400 10 --guide=world_fixed_frame --load=bend_y_only"
         " --bend_force_y=5.0 --ramp_steps=100"
         " --csv=output/engineering_joint/piston_world.csv\n";
}

struct Bounds {
  Eigen::Vector3d min = Eigen::Vector3d::Zero();
  Eigen::Vector3d max = Eigen::Vector3d::Zero();

  Eigen::Vector3d size() const {
    return max - min;
  }
};

FEAT10ConstraintManager::ElementRange MakeElementRange(
    const ANCFCPUUtils::MeshInstance& instance) {
  return FEAT10ConstraintManager::ElementRange{
      instance.element_offset, instance.element_offset + instance.num_elements};
}

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

Eigen::Vector3d EvaluateCurrentPointPosition(
    const FEAT10ConstraintManager::ReferencePoint& point,
    const Eigen::MatrixXi& connectivity, const Eigen::VectorXd& x,
    const Eigen::VectorXd& y, const Eigen::VectorXd& z) {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  for (int local_node = 0; local_node < Quadrature::N_NODE_T10_10;
       ++local_node) {
    const double weight = point.shape(local_node);
    if (weight == 0.0) {
      continue;
    }
    const int node = connectivity(point.element_idx, local_node);
    position += weight * Eigen::Vector3d(x(node), y(node), z(node));
  }
  return position;
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

std::vector<double> BuildOrderedSearchZCandidates(double z_min, double z_max,
                                                  double preferred_z) {
  if (z_max <= z_min) {
    throw std::runtime_error("Invalid piston joint search interval");
  }

  preferred_z = std::clamp(preferred_z, z_min, z_max);
  std::vector<double> candidates;
  candidates.reserve(kPistonJointSearchSamplesPerWindow + 1);
  candidates.push_back(preferred_z);
  for (int i = 0; i < kPistonJointSearchSamplesPerWindow; ++i) {
    const double alpha =
        static_cast<double>(i) /
        static_cast<double>(kPistonJointSearchSamplesPerWindow - 1);
    candidates.push_back((1.0 - alpha) * z_min + alpha * z_max);
  }

  std::sort(candidates.begin(), candidates.end(),
            [preferred_z](double a, double b) {
              const double da = std::abs(a - preferred_z);
              const double db = std::abs(b - preferred_z);
              if (da != db) {
                return da < db;
              }
              return a < b;
            });
  candidates.erase(
      std::unique(candidates.begin(), candidates.end(),
                  [](double a, double b) { return std::abs(a - b) < 1e-12; }),
      candidates.end());
  return candidates;
}

Eigen::Vector3d EstimatePistonSectionCenter(
    const Eigen::MatrixXd& all_nodes, const ANCFCPUUtils::MeshInstance& inst,
    double z_center, double slab_half_height, double radial_cut) {
  Eigen::Vector2d xy_center = Eigen::Vector2d::Zero();
  int count                 = 0;
  for (int local_node = 0; local_node < inst.num_nodes; ++local_node) {
    const Eigen::Vector3d p =
        all_nodes.row(inst.node_offset + local_node).transpose();
    if (std::abs(p.z() - z_center) > slab_half_height) {
      continue;
    }
    if (std::hypot(p.x(), p.y()) > radial_cut) {
      continue;
    }
    xy_center += p.head<2>();
    ++count;
  }

  if (count == 0) {
    throw std::runtime_error(
        "Failed to estimate piston shaft center on the requested z section");
  }
  xy_center /= static_cast<double>(count);
  return Eigen::Vector3d(xy_center.x(), xy_center.y(), z_center);
}

struct CylindricalRowScales {
  std::array<double, 2> parallel    = {{1.0, 1.0}};
  std::array<double, 2> collocation = {{1.0, 1.0}};
};

struct CylindricalJointResiduals {
  std::array<double, 2> parallel_raw       = {{0.0, 0.0}};
  std::array<double, 2> collocation_raw    = {{0.0, 0.0}};
  std::array<double, 2> parallel_scaled    = {{0.0, 0.0}};
  std::array<double, 2> collocation_scaled = {{0.0, 0.0}};
};

double ComputeDotConstraintRowScale(
    const FEAT10ConstraintManager::ReferencePoint& p,
    const FEAT10ConstraintManager::ReferencePoint& q,
    const FEAT10ConstraintManager::ReferencePoint& r,
    const FEAT10ConstraintManager::ReferencePoint& s,
    const Eigen::MatrixXi& connectivity, const Eigen::VectorXd& x_ref,
    const Eigen::VectorXd& y_ref, const Eigen::VectorXd& z_ref) {
  const Eigen::Vector3d p_pos =
      EvaluateCurrentPointPosition(p, connectivity, x_ref, y_ref, z_ref);
  const Eigen::Vector3d q_pos =
      EvaluateCurrentPointPosition(q, connectivity, x_ref, y_ref, z_ref);
  const Eigen::Vector3d r_pos =
      EvaluateCurrentPointPosition(r, connectivity, x_ref, y_ref, z_ref);
  const Eigen::Vector3d s_pos =
      EvaluateCurrentPointPosition(s, connectivity, x_ref, y_ref, z_ref);
  const double a_norm   = (q_pos - p_pos).norm();
  const double d_norm   = (s_pos - r_pos).norm();
  const double row_norm = std::sqrt(a_norm * a_norm + d_norm * d_norm);
  if (row_norm <= 1e-12) {
    return 1.0;
  }
  return 1.0 / row_norm;
}

CylindricalRowScales BuildCylindricalRowScales(
    const FEAT10ConstraintManager::CylindricalJointGeometry& joint_geometry,
    const Eigen::MatrixXi& connectivity, const Eigen::VectorXd& x_ref,
    const Eigen::VectorXd& y_ref, const Eigen::VectorXd& z_ref) {
  CylindricalRowScales scales;
  scales.parallel[0] = ComputeDotConstraintRowScale(
      joint_geometry.p, joint_geometry.q, joint_geometry.r, joint_geometry.s,
      connectivity, x_ref, y_ref, z_ref);
  scales.parallel[1] = ComputeDotConstraintRowScale(
      joint_geometry.p, joint_geometry.q, joint_geometry.r, joint_geometry.u,
      connectivity, x_ref, y_ref, z_ref);
  scales.collocation[0] = ComputeDotConstraintRowScale(
      joint_geometry.p, joint_geometry.v, joint_geometry.p, joint_geometry.r,
      connectivity, x_ref, y_ref, z_ref);
  scales.collocation[1] = ComputeDotConstraintRowScale(
      joint_geometry.p, joint_geometry.w, joint_geometry.p, joint_geometry.r,
      connectivity, x_ref, y_ref, z_ref);
  return scales;
}

CylindricalJointResiduals EvaluateCylindricalJointResiduals(
    const FEAT10ConstraintManager::CylindricalJointGeometry& joint_geometry,
    const CylindricalRowScales& row_scales, const Eigen::MatrixXi& connectivity,
    const Eigen::VectorXd& x, const Eigen::VectorXd& y,
    const Eigen::VectorXd& z) {
  const Eigen::Vector3d p_pos =
      EvaluateCurrentPointPosition(joint_geometry.p, connectivity, x, y, z);
  const Eigen::Vector3d q_pos =
      EvaluateCurrentPointPosition(joint_geometry.q, connectivity, x, y, z);
  const Eigen::Vector3d r_pos =
      EvaluateCurrentPointPosition(joint_geometry.r, connectivity, x, y, z);
  const Eigen::Vector3d s_pos =
      EvaluateCurrentPointPosition(joint_geometry.s, connectivity, x, y, z);
  const Eigen::Vector3d u_pos =
      EvaluateCurrentPointPosition(joint_geometry.u, connectivity, x, y, z);
  const Eigen::Vector3d v_pos =
      EvaluateCurrentPointPosition(joint_geometry.v, connectivity, x, y, z);
  const Eigen::Vector3d w_pos =
      EvaluateCurrentPointPosition(joint_geometry.w, connectivity, x, y, z);

  CylindricalJointResiduals residuals;
  residuals.parallel_raw[0] =
      (q_pos - p_pos).dot(s_pos - r_pos) - joint_geometry.f_par1;
  residuals.parallel_raw[1] =
      (q_pos - p_pos).dot(u_pos - r_pos) - joint_geometry.f_par2;
  residuals.collocation_raw[0] =
      (v_pos - p_pos).dot(r_pos - p_pos) - joint_geometry.f_col1;
  residuals.collocation_raw[1] =
      (w_pos - p_pos).dot(r_pos - p_pos) - joint_geometry.f_col2;
  residuals.parallel_scaled[0] =
      row_scales.parallel[0] * residuals.parallel_raw[0];
  residuals.parallel_scaled[1] =
      row_scales.parallel[1] * residuals.parallel_raw[1];
  residuals.collocation_scaled[0] =
      row_scales.collocation[0] * residuals.collocation_raw[0];
  residuals.collocation_scaled[1] =
      row_scales.collocation[1] * residuals.collocation_raw[1];
  return residuals;
}

FEAT10ConstraintManager::CylindricalJointGeometry
FindPistonCylindricalJointGeometry(
    FEAT10ConstraintManager* constraint_manager,
    const Eigen::MatrixXd& all_nodes,
    const ANCFCPUUtils::MeshInstance& inst_piston,
    const FEAT10ConstraintManager::CylindricalGuideFrame& shared_cup_frame,
    const FEAT10ConstraintManager::ElementRange& piston_range, double z_min,
    double z_max, double preferred_z, double offset, const char* label) {
  std::string last_error;
  const std::vector<double> search_zs =
      BuildOrderedSearchZCandidates(z_min, z_max, preferred_z);
  for (double z : search_zs) {
    std::vector<Eigen::Vector3d> axis_point_candidates;
    axis_point_candidates.emplace_back(0.0, 0.0, z);
    try {
      const Eigen::Vector3d local_center = EstimatePistonSectionCenter(
          all_nodes, inst_piston, z, kPistonJointSearchSlabHalfHeight,
          kPistonJointSearchRadialCut);
      if ((local_center.head<2>() - axis_point_candidates.front().head<2>())
              .norm() > 1e-10) {
        axis_point_candidates.push_back(local_center);
      }
    } catch (const std::runtime_error& error) {
      last_error = error.what();
    }

    for (const Eigen::Vector3d& axis_point_c : axis_point_candidates) {
      try {
        return constraint_manager->BuildCylindricalJointGeometry(
            shared_cup_frame, piston_range, axis_point_c, offset);
      } catch (const std::runtime_error& error) {
        last_error = error.what();
      }
    }
  }

  std::ostringstream oss;
  oss << "Failed to build " << label
      << " piston cylindrical joint in z window [" << z_min << ", " << z_max
      << "] around preferred z=" << preferred_z;
  if (!last_error.empty()) {
    oss << " (" << last_error << ")";
  }
  throw std::runtime_error(oss.str());
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
  // Use the interior centerline node in the solid bottom plug so the
  // cylindrical helper can still place its perpendicular offset points.
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
      "Failed to identify a centered handle-bar load region on the piston "
      "mesh");
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

std::string MakeOutputPath(int frame) {
  std::ostringstream oss;
  oss << "output/engineering_joint/piston_pump_cylindrical_" << std::setw(6)
      << std::setfill('0') << frame << ".vtu";
  return oss.str();
}

void AddDistributedForce(Eigen::VectorXd* h_f_ext,
                         const std::vector<int>& nodes,
                         const Eigen::Vector3d& total_force) {
  if (nodes.empty()) {
    return;
  }
  const Eigen::Vector3d force_per_node =
      total_force / static_cast<double>(nodes.size());
  for (int node : nodes) {
    (*h_f_ext)(3 * node + 0) += force_per_node.x();
    (*h_f_ext)(3 * node + 1) += force_per_node.y();
    (*h_f_ext)(3 * node + 2) += force_per_node.z();
  }
}

Eigen::Vector3d ComputeCurrentNodeCentroid(const std::vector<int>& nodes,
                                           const Eigen::VectorXd& x,
                                           const Eigen::VectorXd& y,
                                           const Eigen::VectorXd& z) {
  if (nodes.empty()) {
    throw std::runtime_error("Cannot compute centroid of an empty node set");
  }

  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (int node : nodes) {
    center += Eigen::Vector3d(x(node), y(node), z(node));
  }
  return center / static_cast<double>(nodes.size());
}

}  // namespace

int main(int argc, char** argv) {
  int max_steps        = kNumStepsDefault;
  int export_interval  = kExportIntervalDef;
  int ramp_steps       = 0;
  double bend_force_y  = kHandleBendForceYDefault;
  GuideMode guide_mode = GuideMode::kWorldFixedFrame;
  LoadMode load_mode   = LoadMode::kCombined;
  bool write_csv       = false;
  std::string csv_path =
      "output/engineering_joint/piston_pump_cylindrical_metrics.csv";
  int positional_arg_index = 0;
  for (int argi = 1; argi < argc; ++argi) {
    const std::string arg(argv[argi]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
    if (arg.rfind("--", 0) != 0) {
      const int parsed_value = std::atoi(arg.c_str());
      if (parsed_value > 0) {
        if (positional_arg_index == 0) {
          max_steps = parsed_value;
        } else if (positional_arg_index == 1) {
          export_interval = parsed_value;
        } else {
          std::cerr << "Unexpected positional argument: " << arg << std::endl;
          return 1;
        }
        ++positional_arg_index;
        continue;
      }
      std::cerr << "Invalid positional argument: " << arg << std::endl;
      return 1;
    }
    if (arg.rfind("--bend_force_y=", 0) == 0) {
      try {
        bend_force_y =
            std::stod(arg.substr(std::string("--bend_force_y=").size()));
      } catch (const std::exception&) {
        std::cerr << "Invalid value for --bend_force_y: " << arg << std::endl;
        return 1;
      }
    } else if (arg.rfind("--ramp_steps=", 0) == 0) {
      try {
        ramp_steps = std::max(
            0, std::stoi(arg.substr(std::string("--ramp_steps=").size())));
      } catch (const std::exception&) {
        std::cerr << "Invalid value for --ramp_steps: " << arg << std::endl;
        return 1;
      }
    } else if (arg == "--guide" || arg == "--guide_mode") {
      if (argi + 1 >= argc) {
        std::cerr << "Missing value after " << arg << std::endl;
        PrintUsage(argv[0]);
        return 1;
      }
      if (!ParseGuideMode(argv[++argi], &guide_mode)) {
        std::cerr << "Invalid value for " << arg << ": " << argv[argi]
                  << std::endl;
        PrintUsage(argv[0]);
        return 1;
      }
    } else if (arg.rfind("--guide=", 0) == 0) {
      if (!ParseGuideMode(arg.substr(std::string("--guide=").size()),
                          &guide_mode)) {
        std::cerr << "Invalid value for --guide: " << arg << std::endl;
        PrintUsage(argv[0]);
        return 1;
      }
    } else if (arg.rfind("--guide_mode=", 0) == 0) {
      if (!ParseGuideMode(arg.substr(std::string("--guide_mode=").size()),
                          &guide_mode)) {
        std::cerr << "Invalid value for --guide_mode: " << arg << std::endl;
        PrintUsage(argv[0]);
        return 1;
      }
    } else if (arg == "--load" || arg == "--load_mode") {
      if (argi + 1 >= argc) {
        std::cerr << "Missing value after " << arg << std::endl;
        PrintUsage(argv[0]);
        return 1;
      }
      if (!ParseLoadMode(argv[++argi], &load_mode)) {
        std::cerr << "Invalid value for " << arg << ": " << argv[argi]
                  << std::endl;
        PrintUsage(argv[0]);
        return 1;
      }
    } else if (arg.rfind("--load=", 0) == 0) {
      if (!ParseLoadMode(arg.substr(std::string("--load=").size()),
                         &load_mode)) {
        std::cerr << "Invalid value for --load: " << arg << std::endl;
        PrintUsage(argv[0]);
        return 1;
      }
    } else if (arg.rfind("--load_mode=", 0) == 0) {
      if (!ParseLoadMode(arg.substr(std::string("--load_mode=").size()),
                         &load_mode)) {
        std::cerr << "Invalid value for --load_mode: " << arg << std::endl;
        PrintUsage(argv[0]);
        return 1;
      }
    } else if (arg == "--csv") {
      write_csv = true;
    } else if (arg.rfind("--csv=", 0) == 0) {
      write_csv = true;
      csv_path  = arg.substr(std::string("--csv=").size());
    } else {
      std::cerr << "Unknown option: " << arg << "\n"
                << "Expected --bend_force_y=<force>, --ramp_steps=<steps>,"
                << " --csv, --csv=<path>, --guide=<...>, or --load=<...>"
                << std::endl;
      PrintUsage(argv[0]);
      return 1;
    }
  }

  std::cout << "========================================\n";
  std::cout << "FEAT10 Piston Pump Cylindrical-Joint Demo\n";
  std::cout << "========================================\n";
  std::cout << "steps=" << max_steps << " export_interval=" << export_interval
            << " bend_force_y=" << bend_force_y << " ramp_steps=" << ramp_steps
            << " guide_mode=" << GuideModeName(guide_mode)
            << " load_mode=" << LoadModeName(load_mode) << "\n";

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
  const int n_nodes                = mesh_manager.GetTotalNodes();
  const int n_elems                = mesh_manager.GetTotalElements();

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
  // Choose the piston attachment point on the shaft while it is still inside
  // the cup's guided overlap region, so the cylindrical guide represents the
  // actual pump interface rather than a point above the cup.
  const double overlap_span             = overlap_z_max - overlap_z_min;
  const double lower_joint_z_target     = overlap_z_min + 0.25 * overlap_span;
  const double upper_joint_z_target     = overlap_z_min + 0.75 * overlap_span;
  const Eigen::Vector3d world_axis_base = cup_axis_anchor;
  const Eigen::Vector3d world_axis_direction = Eigen::Vector3d::UnitZ();

  std::vector<int> fixed_cup_nodes = SelectBottomCupOuterAnnulusNodes(
      all_nodes, inst_cup, cup_bounds, kCupOuterAnnulusRadiusFraction,
      kBottomFaceTolerance);
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

  std::cout << "cup:      " << inst_cup.num_nodes << " nodes, "
            << inst_cup.num_elements << " elements\n";
  std::cout << "piston:   " << inst_piston.num_nodes << " nodes, "
            << inst_piston.num_elements << " elements\n";
  std::cout << "total:    " << n_nodes << " nodes, " << n_elems
            << " elements\n";
  std::cout << "aligned piston by xy shift = [" << (-piston_shaft_center.x())
            << ", " << (-piston_shaft_center.y()) << "]\n";
  std::cout << "cup bounds z = [" << cup_bounds.min.z() << ", "
            << cup_bounds.max.z() << "]\n";
  std::cout << "piston bounds z = [" << piston_bounds.min.z() << ", "
            << piston_bounds.max.z() << "]\n";
  std::cout << "cup outer annulus r cut = " << cup_outer_annulus_r_cut << "\n";
  std::cout << "joint overlap z = [" << overlap_z_min << ", " << overlap_z_max
            << "] lower_joint_z_target=" << lower_joint_z_target
            << " upper_joint_z_target=" << upper_joint_z_target << "\n";
  std::cout << "fixed cup nodes: " << fixed_cup_nodes.size() << "\n";
  std::cout << "center handle nodes: " << center_handle_nodes.size() << "\n";
  std::cout << "left handle nodes:   " << left_handle_nodes.size() << "\n";
  std::cout << "right handle nodes:  " << right_handle_nodes.size() << "\n";
  const bool anchor_in_fixed_set =
      std::find(fixed_cup_nodes.begin(), fixed_cup_nodes.end(),
                cup_axis_anchor_node) != fixed_cup_nodes.end();
  std::cout << "cup axis anchor node: " << cup_axis_anchor_node << " at ["
            << cup_axis_anchor.transpose() << "]"
            << " anchor_in_fixed_set=" << (anchor_in_fixed_set ? "yes" : "no")
            << "\n";
  std::cout << "world guide axis base: [" << world_axis_base.transpose()
            << "] direction=[" << world_axis_direction.transpose() << "]\n";
  std::cout << "handle tip node: " << handle_tip_node << " at ["
            << all_nodes.row(handle_tip_node) << "]\n";
  std::cout << "handle center:  [" << handle_bar_center.transpose() << "]\n";
  std::cout << "axial handle force = " << kHandleAxialForce
            << " couple side force = " << kHandleCoupleForce
            << " additive bend force y = " << bend_force_y
            << " ramp_steps = " << ramp_steps
            << " release_last_steps = " << kForceReleaseSteps << "\n";

  GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
  gpu_t10_data.Initialize();

  Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    h_x12(i) = all_nodes(i, 0);
    h_y12(i) = all_nodes(i, 1);
    h_z12(i) = all_nodes(i, 2);
  }

  gpu_t10_data.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                     Quadrature::tet5pt_z, Quadrature::tet5pt_weights, h_x12,
                     h_y12, h_z12, all_elems);
  gpu_t10_data.ApplyMaterialsFromMeshManager(mesh_manager);
  gpu_t10_data.CalcDnDuPre();
  gpu_t10_data.CalcMassMatrix();

  FEAT10ConstraintManager constraint_manager(&gpu_t10_data);
  const FEAT10ConstraintManager::ElementRange cup_range =
      MakeElementRange(inst_cup);
  const FEAT10ConstraintManager::ElementRange piston_range =
      MakeElementRange(inst_piston);
  const FEAT10ConstraintManager::CylindricalGuideFrame shared_cup_frame =
      constraint_manager.BuildCylindricalGuideFrame(
          cup_range, world_axis_base, world_axis_direction, kJointOffset);
  const auto& cup_axis_reference       = shared_cup_frame.p;
  const auto& cup_axis_probe_reference = shared_cup_frame.q;

  const double lower_window_min =
      overlap_z_min + kPistonJointWindowMarginFraction * overlap_span;
  const double lower_window_max =
      overlap_z_min + 0.5 * overlap_span -
      kPistonJointWindowMarginFraction * overlap_span;
  const double upper_window_min =
      overlap_z_min + 0.5 * overlap_span +
      kPistonJointWindowMarginFraction * overlap_span;
  const double upper_window_max =
      overlap_z_max - kPistonJointWindowMarginFraction * overlap_span;

  const auto lower_cylindrical_joint = FindPistonCylindricalJointGeometry(
      &constraint_manager, all_nodes, inst_piston, shared_cup_frame,
      piston_range, lower_window_min, lower_window_max, lower_joint_z_target,
      kJointOffset, "lower");
  const auto upper_cylindrical_joint = FindPistonCylindricalJointGeometry(
      &constraint_manager, all_nodes, inst_piston, shared_cup_frame,
      piston_range, upper_window_min, upper_window_max, upper_joint_z_target,
      kJointOffset, "upper");
  const auto& lower_piston_axis_reference     = lower_cylindrical_joint.r;
  const auto& upper_piston_axis_reference     = upper_cylindrical_joint.r;
  const CylindricalRowScales lower_row_scales = BuildCylindricalRowScales(
      lower_cylindrical_joint, all_elems, h_x12, h_y12, h_z12);
  const CylindricalRowScales upper_row_scales = BuildCylindricalRowScales(
      upper_cylindrical_joint, all_elems, h_x12, h_y12, h_z12);

  const Eigen::Vector3d lower_joint_reference_position =
      EvaluateCurrentPointPosition(lower_piston_axis_reference, all_elems,
                                   h_x12, h_y12, h_z12);
  const Eigen::Vector3d upper_joint_reference_position =
      EvaluateCurrentPointPosition(upper_piston_axis_reference, all_elems,
                                   h_x12, h_y12, h_z12);
  std::cout << "lower joint reference point: ["
            << lower_joint_reference_position.transpose() << "]\n";
  std::cout << "upper joint reference point: ["
            << upper_joint_reference_position.transpose() << "]\n";

  // Compute compensating dp2_weight for each cylindrical joint.
  //
  // The DP2 collinearity rows use vectors (v-p) and (r-p), where |v-p| ≈
  // offset and |r-p| ≈ Δz (the vertical distance between the guide frame
  // base and the piston attachment).  When Δz >> offset the standard
  // normalization  w = 1/sqrt(|v-p|² + |r-p|²) ≈ 1/Δz  dilutes the
  // Jacobian's sensitivity to off-axis drift of r from O(1) down to
  // O(offset/Δz).  Boosting dp2_weight by sqrt(offset² + Δz²)/(offset·√2)
  // restores the DP2 row scale to the same level as the DP1 parallelism
  // rows, whose two vectors are both ≈ offset.
  const Eigen::Vector3d guide_p_ref = EvaluateCurrentPointPosition(
      shared_cup_frame.p, all_elems, h_x12, h_y12, h_z12);
  const Eigen::Vector3d guide_v_ref = EvaluateCurrentPointPosition(
      shared_cup_frame.v, all_elems, h_x12, h_y12, h_z12);
  const double guide_offset = (guide_v_ref - guide_p_ref).norm();
  const double lower_connector_len =
      (lower_joint_reference_position - guide_p_ref).norm();
  const double upper_connector_len =
      (upper_joint_reference_position - guide_p_ref).norm();
  const double lower_dp2_weight =
      guide_offset > 1e-14
          ? std::max(1.0, std::sqrt(guide_offset * guide_offset +
                                    lower_connector_len * lower_connector_len) /
                              (guide_offset * std::sqrt(2.0)))
          : 1.0;
  const double upper_dp2_weight =
      guide_offset > 1e-14
          ? std::max(1.0, std::sqrt(guide_offset * guide_offset +
                                    upper_connector_len * upper_connector_len) /
                              (guide_offset * std::sqrt(2.0)))
          : 1.0;
  std::cout << "collinearity dp2_weight: lower=" << lower_dp2_weight
            << " upper=" << upper_dp2_weight
            << " (guide_offset=" << guide_offset
            << " lower_connector=" << lower_connector_len
            << " upper_connector=" << upper_connector_len << ")\n";

  for (int node : fixed_cup_nodes) {
    constraint_manager.AddNodeToWorldCD(node);
  }
  if (guide_mode == GuideMode::kWorldFixedFrame) {
    constraint_manager.AddCylindricalGuideFrameToWorld(shared_cup_frame, 1.0);
  }
  constraint_manager.AddCylindricalJoint(lower_cylindrical_joint, 1.0,
                                         lower_dp2_weight);
  constraint_manager.AddCylindricalJoint(upper_cylindrical_joint, 1.0,
                                         upper_dp2_weight);
  constraint_manager.Finalize();

  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.ConvertToCSR_ConstraintJacT();
  gpu_t10_data.BuildConstraintJacobianCSR();
  gpu_t10_data.CalcP();
  gpu_t10_data.CalcInternalForce();

  SyncedNewtonParams params = {1e-5, 1e-5, 1e-10, 1e12, 8, 12, kDt, false};
  SyncedNewtonSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
  solver.Setup();
  solver.SetParameters(&params);
  solver.AnalyzeHessianSparsity();
  solver.SetFixedSparsityPattern(true);

  std::cout << "constraints: " << gpu_t10_data.get_n_constraint() << "\n";
  constexpr int kRowsPerFixedNode                = 3;
  constexpr int kRowsPerCylindricalJoint         = 4;
  constexpr int kParallelRowsPerCylindricalJoint = 2;
  constexpr int kNumCylindricalJoints            = 2;
  const int guide_frame_constraint_rows =
      guide_mode == GuideMode::kWorldFixedFrame ? kRowsPerWorldFixedGuideFrame
                                                : 0;
  const int cylindrical_joint_row_begin =
      static_cast<int>(fixed_cup_nodes.size()) * kRowsPerFixedNode +
      guide_frame_constraint_rows;
  const int cylindrical_joint_row_end =
      cylindrical_joint_row_begin +
      kNumCylindricalJoints * kRowsPerCylindricalJoint;
  const int lower_joint_row_begin = cylindrical_joint_row_begin;
  const int upper_joint_row_begin =
      cylindrical_joint_row_begin + kRowsPerCylindricalJoint;
  const std::array<int, 4> lower_joint_constraint_rows = {
      lower_joint_row_begin + 0, lower_joint_row_begin + 1,
      lower_joint_row_begin + 2, lower_joint_row_begin + 3};
  const std::array<int, 4> upper_joint_constraint_rows = {
      upper_joint_row_begin + 0, upper_joint_row_begin + 1,
      upper_joint_row_begin + 2, upper_joint_row_begin + 3};
  std::vector<int> cylindrical_constraint_rows;
  std::vector<int> parallel_constraint_rows;
  std::vector<int> collocation_constraint_rows;
  for (int joint_idx = 0; joint_idx < kNumCylindricalJoints; ++joint_idx) {
    const int joint_row_begin =
        cylindrical_joint_row_begin + joint_idx * kRowsPerCylindricalJoint;
    for (int row = 0; row < kRowsPerCylindricalJoint; ++row) {
      cylindrical_constraint_rows.push_back(joint_row_begin + row);
    }
    for (int row = 0; row < kParallelRowsPerCylindricalJoint; ++row) {
      parallel_constraint_rows.push_back(joint_row_begin + row);
    }
    for (int row = kParallelRowsPerCylindricalJoint;
         row < kRowsPerCylindricalJoint; ++row) {
      collocation_constraint_rows.push_back(joint_row_begin + row);
    }
  }
  std::cout << "writing initial frame to " << MakeOutputPath(0) << "\n";
  gpu_t10_data.WriteOutputVTU(MakeOutputPath(0));

  engineering_joint::PistonPumpCsvWriter csv_writer;
  Eigen::VectorXd constraint_values;
  Eigen::VectorXd lambda_values =
      Eigen::VectorXd::Zero(gpu_t10_data.get_n_constraint());
  Eigen::VectorXd augmented_dual_values =
      Eigen::VectorXd::Zero(gpu_t10_data.get_n_constraint());
  std::vector<int> constraint_j_offsets;
  std::vector<int> constraint_j_columns;
  std::vector<double> constraint_j_values;
  Eigen::VectorXd x_curr, y_curr, z_curr;
  gpu_t10_data.RetrievePositionToCPU(x_curr, y_curr, z_curr);
  gpu_t10_data.CalcConstraintData();
  gpu_t10_data.BuildConstraintJacobianCSR();
  gpu_t10_data.RetrieveConstraintDataToCPU(constraint_values);
  gpu_t10_data.RetrieveConstraintJacobianCSRToCPU(
      constraint_j_offsets, constraint_j_columns, constraint_j_values);
  const Eigen::Vector3d cup_axis_initial_base = EvaluateCurrentPointPosition(
      cup_axis_reference, all_elems, x_curr, y_curr, z_curr);
  const Eigen::Vector3d cup_axis_initial_probe = EvaluateCurrentPointPosition(
      cup_axis_probe_reference, all_elems, x_curr, y_curr, z_curr);
  Eigen::Vector3d cup_axis_initial_direction =
      engineering_joint::SafeNormalized(cup_axis_initial_probe -
                                        cup_axis_initial_base);
  if (cup_axis_initial_direction.squaredNorm() < 1e-24) {
    cup_axis_initial_direction = Eigen::Vector3d::UnitZ();
  }
  const Eigen::Vector3d lower_piston_axis_initial =
      EvaluateCurrentPointPosition(lower_piston_axis_reference, all_elems,
                                   x_curr, y_curr, z_curr);
  const Eigen::Vector3d upper_piston_axis_initial =
      EvaluateCurrentPointPosition(upper_piston_axis_reference, all_elems,
                                   x_curr, y_curr, z_curr);
  const Eigen::Vector3d piston_axis_initial = engineering_joint::AveragePoint(
      lower_piston_axis_initial, upper_piston_axis_initial);
  const Eigen::Vector3d handle_tip_initial(x_curr(handle_tip_node),
                                           y_curr(handle_tip_node),
                                           z_curr(handle_tip_node));
  const double initial_world_axis_radial_drift =
      engineering_joint::ComputePointLineDistance(
          piston_axis_initial, world_axis_base, world_axis_direction);
  const double initial_lower_axis_radial_drift =
      engineering_joint::ComputePointLineDistance(lower_piston_axis_initial,
                                                  cup_axis_initial_base,
                                                  cup_axis_initial_direction);
  const double initial_upper_axis_radial_drift =
      engineering_joint::ComputePointLineDistance(upper_piston_axis_initial,
                                                  cup_axis_initial_base,
                                                  cup_axis_initial_direction);
  const Eigen::Vector3d left_handle_initial =
      ComputeCurrentNodeCentroid(left_handle_nodes, x_curr, y_curr, z_curr);
  const Eigen::Vector3d right_handle_initial =
      ComputeCurrentNodeCentroid(right_handle_nodes, x_curr, y_curr, z_curr);
  Eigen::Vector3d initial_bar_direction =
      right_handle_initial - left_handle_initial;
  if (initial_bar_direction.norm() < 1e-12) {
    initial_bar_direction = Eigen::Vector3d::UnitX();
  } else {
    initial_bar_direction.normalize();
  }

  auto compute_load_scale = [&](int step, bool apply_forces) {
    if (!apply_forces) {
      return 0.0;
    }
    if (ramp_steps <= 0) {
      return 1.0;
    }
    return std::min(
        1.0, static_cast<double>(step) / static_cast<double>(ramp_steps));
  };

  auto update_handle_forces = [&](const Eigen::VectorXd& x_pos,
                                  const Eigen::VectorXd& y_pos,
                                  const Eigen::VectorXd& z_pos,
                                  double load_scale) {
    Eigen::VectorXd step_f_ext = Eigen::VectorXd::Zero(n_nodes * 3);
    const Eigen::Vector3d left_center =
        ComputeCurrentNodeCentroid(left_handle_nodes, x_pos, y_pos, z_pos);
    const Eigen::Vector3d right_center =
        ComputeCurrentNodeCentroid(right_handle_nodes, x_pos, y_pos, z_pos);

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

    if (load_scale > 0.0) {
      const Eigen::Vector3d bend_force =
          load_scale * bend_force_y * Eigen::Vector3d::UnitY();

      if (load_mode == LoadMode::kCombined) {
        const Eigen::Vector3d left_force =
            load_scale * (0.25 * kHandleAxialForce * Eigen::Vector3d::UnitZ() -
                          0.5 * kHandleCoupleForce * couple_direction);
        const Eigen::Vector3d right_force =
            load_scale * (0.25 * kHandleAxialForce * Eigen::Vector3d::UnitZ() +
                          0.5 * kHandleCoupleForce * couple_direction);
        AddDistributedForce(&step_f_ext, left_handle_nodes, left_force);
        AddDistributedForce(&step_f_ext, right_handle_nodes, right_force);
        AddDistributedForce(&step_f_ext, center_handle_nodes, bend_force);
      } else {
        // bend_y_only: split the Y-force equally across the left and right
        // handle groups so that the X-moment arms cancel by symmetry,
        // avoiding spurious torque about the piston axis.
        const Eigen::Vector3d half_bend = 0.5 * bend_force;
        AddDistributedForce(&step_f_ext, left_handle_nodes, half_bend);
        AddDistributedForce(&step_f_ext, right_handle_nodes, half_bend);
      }
    }
    gpu_t10_data.SetExternalForce(step_f_ext);

    return std::make_pair(bar_direction, couple_direction);
  };

  const auto initial_load_dirs =
      update_handle_forces(x_curr, y_curr, z_curr, 0.0);
  std::cout << "initial piston axis point: [" << piston_axis_initial.transpose()
            << "]\n";
  std::cout << "initial world-axis drift:  " << initial_world_axis_radial_drift
            << "\n";
  std::cout << "initial lower-axis drift:  " << initial_lower_axis_radial_drift
            << "\n";
  std::cout << "initial upper-axis drift:  " << initial_upper_axis_radial_drift
            << "\n";
  std::cout << "initial handle tip:       [" << handle_tip_initial.transpose()
            << "]\n";
  std::cout << "initial handle bar dir:   ["
            << initial_load_dirs.first.transpose() << "]\n";
  std::cout << "initial couple dir:       ["
            << initial_load_dirs.second.transpose() << "]\n";
  if (write_csv) {
    const std::filesystem::path csv_parent =
        std::filesystem::path(csv_path).parent_path();
    if (!csv_parent.empty()) {
      std::filesystem::create_directories(csv_parent);
    }
    csv_writer.Open(csv_path);
    std::cout << "writing csv metrics to " << csv_path << "\n";
  }

  int output_frame = 1;
  for (int step = 1; step <= max_steps; ++step) {
    const bool apply_forces = (max_steps <= kForceReleaseSteps) ||
                              (step <= max_steps - kForceReleaseSteps);
    const double load_scale = compute_load_scale(step, apply_forces);
    const double applied_bend_force_y = load_scale * bend_force_y;
    const auto load_dirs =
        update_handle_forces(x_curr, y_curr, z_curr, load_scale);
    solver.Solve();

    gpu_t10_data.RetrievePositionToCPU(x_curr, y_curr, z_curr);
    gpu_t10_data.CalcConstraintData();
    gpu_t10_data.BuildConstraintJacobianCSR();
    gpu_t10_data.RetrieveConstraintDataToCPU(constraint_values);
    gpu_t10_data.RetrieveConstraintJacobianCSRToCPU(
        constraint_j_offsets, constraint_j_columns, constraint_j_values);
    HANDLE_ERROR(
        cudaMemcpy(lambda_values.data(), solver.GetLambdaGuessDevicePtr(),
                   static_cast<size_t>(lambda_values.size()) * sizeof(double),
                   cudaMemcpyDeviceToHost));
    augmented_dual_values = lambda_values + params.rho * constraint_values;
    const Eigen::Vector3d lower_piston_axis_current =
        EvaluateCurrentPointPosition(lower_piston_axis_reference, all_elems,
                                     x_curr, y_curr, z_curr);
    const Eigen::Vector3d upper_piston_axis_current =
        EvaluateCurrentPointPosition(upper_piston_axis_reference, all_elems,
                                     x_curr, y_curr, z_curr);
    const Eigen::Vector3d piston_axis_current = engineering_joint::AveragePoint(
        lower_piston_axis_current, upper_piston_axis_current);
    const CylindricalJointResiduals lower_exact_residuals =
        EvaluateCylindricalJointResiduals(lower_cylindrical_joint,
                                          lower_row_scales, all_elems, x_curr,
                                          y_curr, z_curr);
    const CylindricalJointResiduals upper_exact_residuals =
        EvaluateCylindricalJointResiduals(upper_cylindrical_joint,
                                          upper_row_scales, all_elems, x_curr,
                                          y_curr, z_curr);
    const Eigen::Vector3d cup_axis_current_base = EvaluateCurrentPointPosition(
        cup_axis_reference, all_elems, x_curr, y_curr, z_curr);
    const Eigen::Vector3d cup_axis_current_probe = EvaluateCurrentPointPosition(
        cup_axis_probe_reference, all_elems, x_curr, y_curr, z_curr);
    Eigen::Vector3d cup_axis_current_direction =
        engineering_joint::SafeNormalized(cup_axis_current_probe -
                                          cup_axis_current_base);
    if (cup_axis_current_direction.squaredNorm() < 1e-24) {
      cup_axis_current_direction = cup_axis_initial_direction;
    }
    const Eigen::Vector3d handle_tip_current(x_curr(handle_tip_node),
                                             y_curr(handle_tip_node),
                                             z_curr(handle_tip_node));
    const Eigen::Vector3d left_handle_current =
        ComputeCurrentNodeCentroid(left_handle_nodes, x_curr, y_curr, z_curr);
    const Eigen::Vector3d right_handle_current =
        ComputeCurrentNodeCentroid(right_handle_nodes, x_curr, y_curr, z_curr);

    const double piston_axis_relative_radial_drift =
        engineering_joint::ComputePointLineDistance(piston_axis_current,
                                                    cup_axis_current_base,
                                                    cup_axis_current_direction);
    const double lower_axis_radial_drift =
        engineering_joint::ComputePointLineDistance(lower_piston_axis_current,
                                                    cup_axis_current_base,
                                                    cup_axis_current_direction);
    const double upper_axis_radial_drift =
        engineering_joint::ComputePointLineDistance(upper_piston_axis_current,
                                                    cup_axis_current_base,
                                                    cup_axis_current_direction);
    const double world_axis_radial_drift =
        engineering_joint::ComputePointLineDistance(
            piston_axis_current, world_axis_base, world_axis_direction);
    Eigen::Array<double, 8, 1> cylindrical_row_reconstruction_error;
    cylindrical_row_reconstruction_error
        << lower_exact_residuals.parallel_scaled[0] -
               constraint_values(lower_joint_constraint_rows[0]),
        lower_exact_residuals.parallel_scaled[1] -
            constraint_values(lower_joint_constraint_rows[1]),
        lower_exact_residuals.collocation_scaled[0] -
            constraint_values(lower_joint_constraint_rows[2]),
        lower_exact_residuals.collocation_scaled[1] -
            constraint_values(lower_joint_constraint_rows[3]),
        upper_exact_residuals.parallel_scaled[0] -
            constraint_values(upper_joint_constraint_rows[0]),
        upper_exact_residuals.parallel_scaled[1] -
            constraint_values(upper_joint_constraint_rows[1]),
        upper_exact_residuals.collocation_scaled[0] -
            constraint_values(upper_joint_constraint_rows[2]),
        upper_exact_residuals.collocation_scaled[1] -
            constraint_values(upper_joint_constraint_rows[3]);
    const double cylindrical_row_reconstruction_l2 =
        cylindrical_row_reconstruction_error.matrix().norm();
    const double cylindrical_row_reconstruction_linf =
        cylindrical_row_reconstruction_error.abs().maxCoeff();
    std::cout << "step " << step << " piston axis: ["
              << piston_axis_current.transpose() << "]"
              << " axial_disp="
              << (piston_axis_current.z() - piston_axis_initial.z())
              << " relative_radial_drift=" << piston_axis_relative_radial_drift
              << " lower_radial_drift=" << lower_axis_radial_drift
              << " upper_radial_drift=" << upper_axis_radial_drift
              << " world_axis_radial_drift=" << world_axis_radial_drift << "\n";
    std::cout << "step " << step << " handle tip:  ["
              << handle_tip_current.transpose() << "]"
              << " travel=" << (handle_tip_current - handle_tip_initial).norm()
              << "\n";
    std::cout << "step " << step << " handle bar:  ["
              << (right_handle_current - left_handle_current).transpose() << "]"
              << " couple_dir=[" << load_dirs.second.transpose() << "]"
              << " load_scale=" << load_scale
              << " applied_bend_force_y=" << applied_bend_force_y
              << " forces_active=" << (apply_forces ? "yes" : "no") << "\n";
    std::cout << "step " << step << " exact cylindrical row mismatch: l2="
              << cylindrical_row_reconstruction_l2
              << " linf=" << cylindrical_row_reconstruction_linf << "\n";

    // --- Recover joint reaction wrench and compare with theory. ---
    const Eigen::VectorXd joint_reaction =
        engineering_joint::ComputeGeneralizedReactionFromCSR(
            n_nodes * 3, constraint_j_offsets, constraint_j_columns,
            constraint_j_values, augmented_dual_values,
            cylindrical_joint_row_begin, cylindrical_joint_row_end);
    const engineering_joint::HingeWrench piston_joint_reaction =
        engineering_joint::ScaleHingeWrench(
            engineering_joint::EstimateHingeWrench(
                joint_reaction, inst_piston.node_offset, inst_piston.num_nodes,
                x_curr, y_curr, z_curr, piston_axis_current),
            kDt);

    // Theoretical reaction for a pure Y-bending load on a rigid piston.
    // The wrench reference point is piston_axis_current (midpoint of the
    // two joint attachments).  The bending force is distributed over
    // center_handle_nodes; its centroid gives the effective load point.
    //   Force equilibrium:  joint_Fy = -applied_Fy
    //   Moment about reference (right-hand rule, F in +Y at lever +Z):
    //     moment_x = -Fy * lever_arm_z
    const Eigen::Vector3d handle_center_current =
        ComputeCurrentNodeCentroid(center_handle_nodes, x_curr, y_curr, z_curr);
    const double lever_arm_z =
        handle_center_current.z() - piston_axis_current.z();
    const double theory_force_y  = -applied_bend_force_y;
    const double theory_moment_x = -applied_bend_force_y * lever_arm_z;
    std::cout << "step " << step << " joint reaction: force=["
              << piston_joint_reaction.force.transpose() << "] moment=["
              << piston_joint_reaction.moment.transpose()
              << "] theory: force_y=" << theory_force_y
              << " moment_x=" << theory_moment_x
              << " lever_arm_z=" << lever_arm_z << " force_y_err="
              << (piston_joint_reaction.force.y() - theory_force_y)
              << " moment_x_err="
              << (piston_joint_reaction.moment.x() - theory_moment_x) << "\n";

    if (write_csv) {
      csv_writer.WriteRow(
          step, step * kDt,
          engineering_joint::ComputeIndexedL2Norm(constraint_values,
                                                  cylindrical_constraint_rows),
          engineering_joint::ComputeIndexedInfinityNorm(
              constraint_values, cylindrical_constraint_rows),
          engineering_joint::ComputeIndexedL2Norm(constraint_values,
                                                  parallel_constraint_rows),
          engineering_joint::ComputeIndexedInfinityNorm(
              constraint_values, parallel_constraint_rows),
          engineering_joint::ComputeIndexedL2Norm(constraint_values,
                                                  collocation_constraint_rows),
          engineering_joint::ComputeIndexedInfinityNorm(
              constraint_values, collocation_constraint_rows),
          piston_joint_reaction, piston_axis_current,
          piston_axis_current.z() - piston_axis_initial.z(),
          piston_axis_relative_radial_drift, lower_axis_radial_drift,
          upper_axis_radial_drift, world_axis_radial_drift,
          applied_bend_force_y, cylindrical_row_reconstruction_l2,
          cylindrical_row_reconstruction_linf,
          lower_exact_residuals.parallel_raw[0],
          lower_exact_residuals.parallel_raw[1],
          lower_exact_residuals.collocation_raw[0],
          lower_exact_residuals.collocation_raw[1],
          upper_exact_residuals.parallel_raw[0],
          upper_exact_residuals.parallel_raw[1],
          upper_exact_residuals.collocation_raw[0],
          upper_exact_residuals.collocation_raw[1], handle_tip_current);
    }

    if (step % export_interval == 0) {
      gpu_t10_data.WriteOutputVTU(MakeOutputPath(output_frame));
      ++output_frame;
    }
  }

  gpu_t10_data.Destroy();
  std::cout << "Done. Output written to output/engineering_joint/\n";
  return 0;
}
