/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * File:    FEAT10ConstraintManager.cu
 * Brief:   Host-side FEAT10 constraint/joint construction utilities.
 *          This file performs point-location in the reference mesh and lowers
 *          high-level joint descriptions into FEAT10's scalar constraint
 *          layout before handing control to GPU_FEAT10_Data.
 *==============================================================
 *==============================================================*/

#include "FEAT10ConstraintManager.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../../lib_utils/cuda_utils.h"

namespace {

// Host-side T10 basis evaluation helpers used during point-location and
// geometry-based joint construction. These mirror the FEAT10 interpolation used
// later on the GPU, but they run on the CPU because setup happens before the
// runtime constraint layout is uploaded.
constexpr int kT10NodeCount      = Quadrature::N_NODE_T10_10;
constexpr int kNewtonMaxIters    = 16;
constexpr double kNewtonTol      = 1e-11;
constexpr double kBarycentricTol = 1e-8;

__global__ void offset_selected_nodes_and_targets_kernel(
    double* d_pos_axis, double* d_target_axis, const int* d_node_ids,
    int n_node_ids, double delta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_node_ids) {
    return;
  }

  const int node = d_node_ids[i];
  d_pos_axis[node] += delta;
  d_target_axis[node] += delta;
}

Eigen::Matrix<double, kT10NodeCount, 1> EvaluateT10ShapeFunctions(
    double xi, double eta, double zeta) {
  const double l1 = 1.0 - xi - eta - zeta;
  const double l2 = xi;
  const double l3 = eta;
  const double l4 = zeta;
  const std::array<double, 4> l = {l1, l2, l3, l4};

  Eigen::Matrix<double, kT10NodeCount, 1> shape;
  shape.setZero();
  for (int i = 0; i < 4; ++i) {
    shape(i) = l[static_cast<size_t>(i)] *
               (2.0 * l[static_cast<size_t>(i)] - 1.0);
  }

  constexpr int edges[6][2] = {{0, 1}, {1, 2}, {0, 2},
                                {0, 3}, {1, 3}, {2, 3}};
  for (int edge = 0; edge < 6; ++edge) {
    const int i       = edges[edge][0];
    const int j       = edges[edge][1];
    shape(edge + 4) = 4.0 * l[static_cast<size_t>(i)] *
                      l[static_cast<size_t>(j)];
  }

  return shape;
}

Eigen::Matrix<double, kT10NodeCount, 3> EvaluateT10ShapeDerivatives(
    double xi, double eta, double zeta) {
  const double l1 = 1.0 - xi - eta - zeta;
  const double l2 = xi;
  const double l3 = eta;
  const double l4 = zeta;
  const std::array<double, 4> l = {l1, l2, l3, l4};

  constexpr double dl[4][3] = {
      {-1.0, -1.0, -1.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
  };

  Eigen::Matrix<double, kT10NodeCount, 3> deriv;
  deriv.setZero();
  for (int i = 0; i < 4; ++i) {
    const double factor = 4.0 * l[static_cast<size_t>(i)] - 1.0;
    for (int axis = 0; axis < 3; ++axis) {
      deriv(i, axis) = factor * dl[i][axis];
    }
  }

  constexpr int edges[6][2] = {{0, 1}, {1, 2}, {0, 2},
                                {0, 3}, {1, 3}, {2, 3}};
  for (int edge = 0; edge < 6; ++edge) {
    const int i = edges[edge][0];
    const int j = edges[edge][1];
    for (int axis = 0; axis < 3; ++axis) {
      deriv(edge + 4, axis) =
          4.0 * (l[static_cast<size_t>(i)] * dl[j][axis] +
                 l[static_cast<size_t>(j)] * dl[i][axis]);
    }
  }

  return deriv;
}

Eigen::Vector3d EvaluateT10ReferencePosition(const Eigen::Matrix<double, 10, 3>& x_elem,
                                             const Eigen::Matrix<double, kT10NodeCount, 1>& shape) {
  return x_elem.transpose() * shape;
}

Eigen::Matrix3d EvaluateT10ReferenceJacobian(
    const Eigen::Matrix<double, 10, 3>& x_elem,
    const Eigen::Matrix<double, kT10NodeCount, 3>& deriv) {
  Eigen::Matrix3d jacobian = Eigen::Matrix3d::Zero();
  for (int a = 0; a < kT10NodeCount; ++a) {
    jacobian += x_elem.row(a).transpose() * deriv.row(a);
  }
  return jacobian;
}

bool IsReferenceCoordinateInsideTet(const Eigen::Vector3d& coord,
                                    double tol = kBarycentricTol) {
  const double l1 = 1.0 - coord[0] - coord[1] - coord[2];
  return coord[0] >= -tol && coord[1] >= -tol && coord[2] >= -tol &&
         l1 >= -tol;
}

bool AabbContainsPoint(const Eigen::Matrix<double, 10, 3>& x_elem,
                       const Eigen::Vector3d& point, double tol) {
  Eigen::Vector3d min_corner = x_elem.colwise().minCoeff();
  Eigen::Vector3d max_corner = x_elem.colwise().maxCoeff();
  min_corner.array() -= tol;
  max_corner.array() += tol;
  return (point.array() >= min_corner.array()).all() &&
         (point.array() <= max_corner.array()).all();
}

Eigen::Vector3d ComputeLinearTetInitialGuess(
    const Eigen::Matrix<double, 10, 3>& x_elem, const Eigen::Vector3d& point) {
  Eigen::Matrix3d basis;
  basis.col(0) = x_elem.row(1).transpose() - x_elem.row(0).transpose();
  basis.col(1) = x_elem.row(2).transpose() - x_elem.row(0).transpose();
  basis.col(2) = x_elem.row(3).transpose() - x_elem.row(0).transpose();
  const Eigen::Vector3d rhs = point - x_elem.row(0).transpose();
  return basis.colPivHouseholderQr().solve(rhs);
}

Eigen::Vector3d BuildPerpendicularAxis1(const Eigen::Vector3d& axis) {
  const Eigen::Vector3d ex = Eigen::Vector3d::UnitX();
  const Eigen::Vector3d ey = Eigen::Vector3d::UnitY();
  const Eigen::Vector3d seed =
      std::abs(ex.dot(axis)) < 0.9 ? ex : ey;
  Eigen::Vector3d p1 = seed - seed.dot(axis) * axis;
  const double norm = p1.norm();
  if (norm < 1e-12) {
    throw std::runtime_error(
        "FEAT10ConstraintManager: failed to construct perpendicular axis");
  }
  return p1 / norm;
}

}  // namespace

FEAT10ConstraintManager::FEAT10ConstraintManager(GPU_FEAT10_Data* data)
    : data_(data) {
  if (data_ == nullptr) {
    throw std::invalid_argument(
        "FEAT10ConstraintManager: data pointer cannot be null");
  }
}

FEAT10ConstraintManager::~FEAT10ConstraintManager() {
  if (d_constrained_nodes_ != nullptr) {
    cudaFree(d_constrained_nodes_);
  }
}

// Material-point queries are phrased in the reference configuration, so the
// manager pulls one CPU-side snapshot of the undeformed node coordinates and
// connectivity. Subsequent geometric queries reuse this cache instead of asking
// GPU_FEAT10_Data to repatriate the mesh every time.
void FEAT10ConstraintManager::EnsureReferenceCache() const {
  if (reference_cache_ready_) {
    return;
  }

  Eigen::VectorXd x_ref, y_ref, z_ref;
  data_->RetrieveReferencePositionToCPU(x_ref, y_ref, z_ref);
  reference_nodes_.resize(data_->get_n_coef(), 3);
  for (int i = 0; i < data_->get_n_coef(); ++i) {
    reference_nodes_(i, 0) = x_ref(i);
    reference_nodes_(i, 1) = y_ref(i);
    reference_nodes_(i, 2) = z_ref(i);
  }

  data_->RetrieveConnectivityToCPU(connectivity_);
  reference_cache_ready_ = true;
}

FEAT10ConstraintManager::ElementRange FEAT10ConstraintManager::FullElementRange()
    const {
  return ElementRange{0, data_->get_n_elem()};
}

void FEAT10ConstraintManager::AddNodeToWorldCD(int node_id) {
  if (finalized_) {
    throw std::logic_error(
        "FEAT10ConstraintManager::AddNodeToWorldCD: manager already finalized");
  }

  const int n_nodes = data_->get_n_coef();
  if (node_id < 0 || node_id >= n_nodes) {
    throw std::out_of_range(
        "FEAT10ConstraintManager::AddNodeToWorldCD: node id out of range");
  }

  node_ids_.push_back(node_id);
}

void FEAT10ConstraintManager::AddNodesToWorldCD(
    const Eigen::VectorXi& node_ids) {
  for (int i = 0; i < node_ids.size(); ++i) {
    AddNodeToWorldCD(node_ids(i));
  }
}

FEAT10ConstraintManager::ReferencePoint
FEAT10ConstraintManager::LocateReferencePoint(
    const Eigen::Vector3d& reference_point) const {
  return LocateReferencePoint(reference_point, FullElementRange());
}

FEAT10ConstraintManager::ReferencePoint
FEAT10ConstraintManager::LocateNodeReferencePoint(
    const Eigen::Vector3d& reference_point) const {
  return LocateNodeReferencePoint(reference_point, FullElementRange());
}

// LocateReferencePoint() is the core geometric query used by the higher-level
// joint builders. It searches the requested body range, runs a Newton solve in
// reference coordinates, and stores the winning T10 interpolation weights as a
// persistent ReferencePoint.
FEAT10ConstraintManager::ReferencePoint
FEAT10ConstraintManager::LocateReferencePoint(
    const Eigen::Vector3d& reference_point, const ElementRange& range) const {
  EnsureReferenceCache();

  if (range.begin < 0 || range.end > data_->get_n_elem() ||
      range.begin >= range.end) {
    throw std::out_of_range(
        "FEAT10ConstraintManager::LocateReferencePoint: invalid element range");
  }

  ReferencePoint best_point;
  double best_residual = std::numeric_limits<double>::infinity();
  const double aabb_tol = 1e-8;

  for (int elem_idx = range.begin; elem_idx < range.end; ++elem_idx) {
    // Rebuild the local reference element geometry on the CPU. This keeps the
    // setup code simple and avoids introducing a separate host-side mesh view.
    Eigen::Matrix<double, 10, 3> x_elem;
    for (int local_node = 0; local_node < kT10NodeCount; ++local_node) {
      x_elem.row(local_node) =
          reference_nodes_.row(connectivity_(elem_idx, local_node));
    }

    if (!AabbContainsPoint(x_elem, reference_point, aabb_tol)) {
      continue;
    }

    Eigen::Vector3d coord = ComputeLinearTetInitialGuess(x_elem, reference_point);
    bool converged        = false;
    for (int iter = 0; iter < kNewtonMaxIters; ++iter) {
      const auto shape = EvaluateT10ShapeFunctions(coord[0], coord[1], coord[2]);
      const auto deriv = EvaluateT10ShapeDerivatives(coord[0], coord[1], coord[2]);
      const Eigen::Vector3d mapped = EvaluateT10ReferencePosition(x_elem, shape);
      const Eigen::Vector3d residual = mapped - reference_point;
      const double residual_norm = residual.norm();
      if (residual_norm < best_residual) {
        best_residual        = residual_norm;
        best_point.element_idx = elem_idx;
        best_point.shape     = shape;
      }
      if (residual_norm < kNewtonTol && IsReferenceCoordinateInsideTet(coord)) {
        converged = true;
        best_point.element_idx = elem_idx;
        best_point.shape       = shape;
        break;
      }

      const Eigen::Matrix3d jacobian = EvaluateT10ReferenceJacobian(x_elem, deriv);
      const Eigen::Vector3d delta =
          jacobian.colPivHouseholderQr().solve(residual);
      coord -= delta;
      if (!delta.allFinite()) {
        break;
      }
    }

    if (converged) {
      return best_point;
    }
  }

  if (best_point.element_idx < 0 || best_residual > 1e-7) {
    throw std::runtime_error(
        "FEAT10ConstraintManager::LocateReferencePoint: failed to locate reference point in FEAT10 mesh");
  }
  return best_point;
}

FEAT10ConstraintManager::ReferencePoint
FEAT10ConstraintManager::BuildNodalReferencePoint(int elem_idx,
                                                  int local_node) const {
  ReferencePoint point;
  point.element_idx = elem_idx;
  point.shape.setZero();
  point.shape(local_node) = 1.0;
  return point;
}

FEAT10ConstraintManager::ReferencePoint
FEAT10ConstraintManager::LocateNodeReferencePoint(
    const Eigen::Vector3d& reference_point, const ElementRange& range) const {
  EnsureReferenceCache();

  if (range.begin < 0 || range.end > data_->get_n_elem() ||
      range.begin >= range.end) {
    throw std::out_of_range(
        "FEAT10ConstraintManager::LocateNodeReferencePoint: invalid element range");
  }

  constexpr double kNodeTol = 1e-10;
  double best_distance = std::numeric_limits<double>::infinity();

  for (int elem_idx = range.begin; elem_idx < range.end; ++elem_idx) {
    for (int local_node = 0; local_node < kT10NodeCount; ++local_node) {
      const int node = connectivity_(elem_idx, local_node);
      const Eigen::Vector3d node_pos = reference_nodes_.row(node).transpose();
      const double distance = (node_pos - reference_point).norm();
      if (distance < best_distance) {
        best_distance = distance;
      }
      if (distance <= kNodeTol) {
        return BuildNodalReferencePoint(elem_idx, local_node);
      }
    }
  }

  throw std::runtime_error(
      "FEAT10ConstraintManager::LocateNodeReferencePoint: hinge point is not an exact FEAT10 node in the requested element range (nearest distance = " +
      std::to_string(best_distance) + ")");
}

void FEAT10ConstraintManager::AddPointToPointCDAxis(const ReferencePoint& p,
                                                    const ReferencePoint& r,
                                                    int axis, double target) {
  if (finalized_) {
    throw std::logic_error(
        "FEAT10ConstraintManager::AddPointToPointCDAxis: manager already finalized");
  }
  if (axis < 0 || axis > 2) {
    throw std::out_of_range(
        "FEAT10ConstraintManager::AddPointToPointCDAxis: axis must be 0, 1, or 2");
  }

  ScalarConstraint constraint;
  constraint.type      = kFEAT10ConstraintPointPointCD;
  constraint.axis      = axis;
  constraint.target    = target;
  constraint.row_scale = 1.0;
  constraint.points[0] = p;
  constraint.points[1] = r;
  scalar_constraints_.push_back(constraint);
}

void FEAT10ConstraintManager::AddPointToWorldCDAxis(const ReferencePoint& p,
                                                    int axis,
                                                    double target) {
  if (finalized_) {
    throw std::logic_error(
        "FEAT10ConstraintManager::AddPointToWorldCDAxis: manager already finalized");
  }
  if (axis < 0 || axis > 2) {
    throw std::out_of_range(
        "FEAT10ConstraintManager::AddPointToWorldCDAxis: axis must be 0, 1, or 2");
  }

  ScalarConstraint constraint;
  constraint.type      = kFEAT10ConstraintPointWorldCD;
  constraint.axis      = axis;
  constraint.target    = target;
  constraint.row_scale = 1.0;
  constraint.points[0] = p;
  scalar_constraints_.push_back(constraint);
}

Eigen::Vector3d FEAT10ConstraintManager::EvaluateReferencePoint(
    const ReferencePoint& point) const {
  EnsureReferenceCache();

  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  for (int local_node = 0; local_node < kT10NodeCount; ++local_node) {
    const double weight = point.shape(local_node);
    if (weight == 0.0) {
      continue;
    }
    const int node = connectivity_(point.element_idx, local_node);
    position += weight * reference_nodes_.row(node).transpose();
  }
  return position;
}

// DP1 rows are scaled by the reference lengths of the two defining fibers so
// the effective row magnitude stays reasonably consistent as the user changes
// the geometric offset used to create Q/S/T points.
void FEAT10ConstraintManager::AddDP1Constraint(const ReferencePoint& p,
                                               const ReferencePoint& q,
                                               const ReferencePoint& r,
                                               const ReferencePoint& s,
                                               double target,
                                               double weight) {
  if (finalized_) {
    throw std::logic_error(
        "FEAT10ConstraintManager::AddDP1Constraint: manager already finalized");
  }
  if (weight <= 0.0) {
    throw std::invalid_argument(
        "FEAT10ConstraintManager::AddDP1Constraint: weight must be positive");
  }

  const Eigen::Vector3d p_ref = EvaluateReferencePoint(p);
  const Eigen::Vector3d q_ref = EvaluateReferencePoint(q);
  const Eigen::Vector3d r_ref = EvaluateReferencePoint(r);
  const Eigen::Vector3d s_ref = EvaluateReferencePoint(s);
  const double a_norm         = (q_ref - p_ref).norm();
  const double d_norm         = (s_ref - r_ref).norm();

  ScalarConstraint constraint;
  constraint.type      = kFEAT10ConstraintDP1;
  constraint.target    = target;
  constraint.row_scale = weight;
  if (a_norm > 1e-12 && d_norm > 1e-12) {
    constraint.row_scale *= 1.0 / (a_norm * d_norm);
  }
  constraint.points[0] = p;
  constraint.points[1] = q;
  constraint.points[2] = r;
  constraint.points[3] = s;
  scalar_constraints_.push_back(constraint);
}

void FEAT10ConstraintManager::AddWorldDP1Constraint(
    const ReferencePoint& p, const ReferencePoint& q,
    const Eigen::Vector3d& world_direction, double target, double weight) {
  if (finalized_) {
    throw std::logic_error(
        "FEAT10ConstraintManager::AddWorldDP1Constraint: manager already finalized");
  }
  if (weight <= 0.0) {
    throw std::invalid_argument(
        "FEAT10ConstraintManager::AddWorldDP1Constraint: weight must be positive");
  }

  const double d_norm = world_direction.norm();
  if (d_norm < 1e-12) {
    throw std::invalid_argument(
        "FEAT10ConstraintManager::AddWorldDP1Constraint: world direction must be non-zero");
  }

  const Eigen::Vector3d p_ref = EvaluateReferencePoint(p);
  const Eigen::Vector3d q_ref = EvaluateReferencePoint(q);
  const double a_norm         = (q_ref - p_ref).norm();

  ScalarConstraint constraint;
  constraint.type            = kFEAT10ConstraintWorldDP1;
  constraint.target          = target;
  constraint.world_direction = world_direction;
  constraint.row_scale       = weight;
  if (a_norm > 1e-12) {
    constraint.row_scale *= 1.0 / (a_norm * d_norm);
  }
  constraint.points[0] = p;
  constraint.points[1] = q;
  scalar_constraints_.push_back(constraint);
}

void FEAT10ConstraintManager::AddSphericalJoint(const ReferencePoint& p,
                                                const ReferencePoint& r) {
  AddPointToPointCDAxis(p, r, 0);
  AddPointToPointCDAxis(p, r, 1);
  AddPointToPointCDAxis(p, r, 2);
}

void FEAT10ConstraintManager::AddRevoluteJoint(const ReferencePoint& p,
                                               const ReferencePoint& q,
                                               const ReferencePoint& r,
                                               const ReferencePoint& s,
                                               const ReferencePoint& t,
                                               double f1, double f2,
                                               double dp1_weight) {
  AddPointToPointCDAxis(p, r, 0);
  AddPointToPointCDAxis(p, r, 1);
  AddPointToPointCDAxis(p, r, 2);
  AddDP1Constraint(p, q, r, s, f1, dp1_weight);
  AddDP1Constraint(p, q, r, t, f2, dp1_weight);
}

double FEAT10ConstraintManager::ComputeElementCharacteristicLength(
    int elem_idx) const {
  EnsureReferenceCache();

  double max_length = 0.0;
  for (int i = 0; i < 4; ++i) {
    const Eigen::Vector3d xi =
        reference_nodes_.row(connectivity_(elem_idx, i)).transpose();
    for (int j = i + 1; j < 4; ++j) {
      const Eigen::Vector3d xj =
          reference_nodes_.row(connectivity_(elem_idx, j)).transpose();
      max_length = std::max(max_length, (xj - xi).norm());
    }
  }
  return max_length;
}

double FEAT10ConstraintManager::DefaultJointOffset(const ReferencePoint& p,
                                                   const ReferencePoint& r)
    const {
  const double lp = ComputeElementCharacteristicLength(p.element_idx);
  const double lr = ComputeElementCharacteristicLength(r.element_idx);
  return 1e-2 * std::min(lp, lr);
}

FEAT10ConstraintManager::ReferencePoint
FEAT10ConstraintManager::LocateWithAdaptiveOffset(
    const Eigen::Vector3d& base_point, const Eigen::Vector3d& direction,
    const ElementRange& range, double initial_offset) const {
  double offset = initial_offset;
  for (int attempt = 0; attempt < 8; ++attempt) {
    try {
      return LocateReferencePoint(base_point + offset * direction, range);
    } catch (const std::runtime_error&) {
      offset *= 0.5;
    }
  }

  throw std::runtime_error(
      "FEAT10ConstraintManager::LocateWithAdaptiveOffset: failed to place offset reference point for joint construction");
}

void FEAT10ConstraintManager::AddSphericalJoint(
    const ElementRange& body_b, const ElementRange& body_c,
    const Eigen::Vector3d& hinge_point) {
  const ReferencePoint p = LocateReferencePoint(hinge_point, body_b);
  const ReferencePoint r = LocateReferencePoint(hinge_point, body_c);

  AddSphericalJoint(p, r);
}

void FEAT10ConstraintManager::AddSphericalJointToWorld(
    const ElementRange& body, const Eigen::Vector3d& hinge_point) {
  const ReferencePoint p = LocateReferencePoint(hinge_point, body);
  const Eigen::Vector3d p_ref = EvaluateReferencePoint(p);

  AddPointToWorldCDAxis(p, 0, p_ref.x());
  AddPointToWorldCDAxis(p, 1, p_ref.y());
  AddPointToWorldCDAxis(p, 2, p_ref.z());
}

void FEAT10ConstraintManager::AddRevoluteJoint(
    const ElementRange& body_b, const ElementRange& body_c,
    const Eigen::Vector3d& hinge_point, const Eigen::Vector3d& hinge_axis,
    double offset, double dp1_weight) {
  const double axis_norm = hinge_axis.norm();
  if (axis_norm < 1e-12) {
    throw std::invalid_argument(
        "FEAT10ConstraintManager::AddRevoluteJoint: hinge axis must be non-zero");
  }

  const Eigen::Vector3d axis = hinge_axis / axis_norm;
  const ReferencePoint p = LocateReferencePoint(hinge_point, body_b);
  const ReferencePoint r = LocateReferencePoint(hinge_point, body_c);

  if (offset <= 0.0) {
    offset = DefaultJointOffset(p, r);
  }

  const ReferencePoint q =
      LocateWithAdaptiveOffset(hinge_point, axis, body_b, offset);

  const Eigen::Vector3d p1 = BuildPerpendicularAxis1(axis);
  const Eigen::Vector3d p2 = axis.cross(p1).normalized();
  const ReferencePoint s = LocateWithAdaptiveOffset(hinge_point, p1, body_c, offset);
  const ReferencePoint t = LocateWithAdaptiveOffset(hinge_point, p2, body_c, offset);

  AddRevoluteJoint(p, q, r, s, t, 0.0, 0.0, dp1_weight);
}

void FEAT10ConstraintManager::AddRevoluteJointToWorld(
    const ElementRange& body, const Eigen::Vector3d& hinge_point,
    const Eigen::Vector3d& hinge_axis, double offset,
    double dp1_weight) {
  const double axis_norm = hinge_axis.norm();
  if (axis_norm < 1e-12) {
    throw std::invalid_argument(
        "FEAT10ConstraintManager::AddRevoluteJointToWorld: hinge axis must be non-zero");
  }

  const Eigen::Vector3d axis = hinge_axis / axis_norm;
  const ReferencePoint p = LocateReferencePoint(hinge_point, body);
  const Eigen::Vector3d p_ref = EvaluateReferencePoint(p);

  if (offset <= 0.0) {
    offset = 1e-2 * ComputeElementCharacteristicLength(p.element_idx);
  }

  const ReferencePoint q =
      LocateWithAdaptiveOffset(hinge_point, axis, body, offset);

  const Eigen::Vector3d p1 = BuildPerpendicularAxis1(axis);
  const Eigen::Vector3d p2 = axis.cross(p1).normalized();

  AddPointToWorldCDAxis(p, 0, p_ref.x());
  AddPointToWorldCDAxis(p, 1, p_ref.y());
  AddPointToWorldCDAxis(p, 2, p_ref.z());
  AddWorldDP1Constraint(p, q, p1, 0.0, dp1_weight);
  AddWorldDP1Constraint(p, q, p2, 0.0, dp1_weight);
}

// This is the lowering step that bridges the friendly API above to the sparse
// scalar constraint representation inside GPU_FEAT10_Data. Each primitive is
// expanded into scalar rows, row/column sparsity, and term descriptors for both
// C_q and C_q^T assembly.
FEAT10GeneralConstraintLayout
FEAT10ConstraintManager::BuildGeneralConstraintLayout() const {
  EnsureReferenceCache();

  std::vector<FEAT10GeneralConstraintScalar> scalars;
  scalars.reserve(constrained_nodes_.size() * 3 + scalar_constraints_.size());

  // Legacy fixed-node requests are represented as one scalar CD row per axis
  // so they can share the same lowering pipeline as the joint primitives.
  for (int i = 0; i < constrained_nodes_.size(); ++i) {
    const int node = constrained_nodes_(i);
    for (int axis = 0; axis < 3; ++axis) {
      FEAT10GeneralConstraintScalar scalar;
      scalar.type    = kFEAT10ConstraintNodeWorldCD;
      scalar.axis    = axis;
      scalar.node_id = node;
      scalar.target  = reference_nodes_(node, axis);
      scalars.push_back(scalar);
    }
  }

  for (const ScalarConstraint& constraint : scalar_constraints_) {
    FEAT10GeneralConstraintScalar scalar;
    scalar.type    = constraint.type;
    scalar.axis    = constraint.axis;
    scalar.node_id = constraint.node_id;
    scalar.target  = constraint.target;
    scalar.row_scale = constraint.row_scale;
    scalar.world_direction = constraint.world_direction;
    for (int i = 0; i < 4; ++i) {
      scalar.points[i] = constraint.points[i];
    }
    scalars.push_back(scalar);
  }

  const int n_constraints = static_cast<int>(scalars.size());
  // row_dofs stores the sparsity of C_q by row, while dof_rows stores the
  // transpose sparsity needed by the JT scatter path.
  const int n_dofs        = data_->get_n_coef() * 3;

  std::vector<std::vector<int>> row_dofs(static_cast<size_t>(n_constraints));
  std::vector<std::vector<int>> dof_rows(static_cast<size_t>(n_dofs));
  std::vector<std::vector<TermDescriptor>> row_terms(
      static_cast<size_t>(n_constraints));

  auto add_unique = [](std::vector<int>* values, int entry) {
    if (std::find(values->begin(), values->end(), entry) == values->end()) {
      values->push_back(entry);
    }
  };

  auto add_term = [&](int row, int dof, int kind, double scale) {
    add_unique(&row_dofs[static_cast<size_t>(row)], dof);
    add_unique(&dof_rows[static_cast<size_t>(dof)], row);
    row_terms[static_cast<size_t>(row)].push_back(TermDescriptor{dof, kind,
                                                                 scale});
  };

  for (int row = 0; row < n_constraints; ++row) {
    const auto& scalar = scalars[static_cast<size_t>(row)];
    if (scalar.type == kFEAT10ConstraintNodeWorldCD) {
      add_term(row, scalar.node_id * 3 + scalar.axis,
               kFEAT10ConstraintTermConstant, scalar.row_scale);
      continue;
    }

    if (scalar.type == kFEAT10ConstraintPointWorldCD) {
      for (int local_node = 0; local_node < kT10NodeCount; ++local_node) {
        const double wp = scalar.points[0].shape(local_node);
        if (wp != 0.0) {
          const int node =
              connectivity_(scalar.points[0].element_idx, local_node);
          add_term(row, node * 3 + scalar.axis,
                   kFEAT10ConstraintTermConstant, scalar.row_scale * wp);
        }
      }
      continue;
    }

    if (scalar.type == kFEAT10ConstraintPointPointCD) {
      for (int local_node = 0; local_node < kT10NodeCount; ++local_node) {
        const double wp = scalar.points[0].shape(local_node);
        if (wp != 0.0) {
          const int node = connectivity_(scalar.points[0].element_idx, local_node);
          add_term(row, node * 3 + scalar.axis,
                   kFEAT10ConstraintTermConstant, -scalar.row_scale * wp);
        }

        const double wr = scalar.points[1].shape(local_node);
        if (wr != 0.0) {
          const int node = connectivity_(scalar.points[1].element_idx, local_node);
          add_term(row, node * 3 + scalar.axis,
                   kFEAT10ConstraintTermConstant, scalar.row_scale * wr);
        }
      }
      continue;
    }

    if (scalar.type == kFEAT10ConstraintWorldDP1) {
      for (int local_node = 0; local_node < kT10NodeCount; ++local_node) {
        const double wp = scalar.points[0].shape(local_node);
        if (wp != 0.0) {
          const int node =
              connectivity_(scalar.points[0].element_idx, local_node);
          for (int axis = 0; axis < 3; ++axis) {
            add_term(row, node * 3 + axis, kFEAT10ConstraintTermUsesD,
                     -scalar.row_scale * wp);
          }
        }

        const double wq = scalar.points[1].shape(local_node);
        if (wq != 0.0) {
          const int node =
              connectivity_(scalar.points[1].element_idx, local_node);
          for (int axis = 0; axis < 3; ++axis) {
            add_term(row, node * 3 + axis, kFEAT10ConstraintTermUsesD,
                     scalar.row_scale * wq);
          }
        }
      }
      continue;
    }

    for (int local_node = 0; local_node < kT10NodeCount; ++local_node) {
      const double wp = scalar.points[0].shape(local_node);
      if (wp != 0.0) {
        const int node = connectivity_(scalar.points[0].element_idx, local_node);
        for (int axis = 0; axis < 3; ++axis) {
          add_term(row, node * 3 + axis, kFEAT10ConstraintTermUsesD,
                   -scalar.row_scale * wp);
        }
      }

      const double wq = scalar.points[1].shape(local_node);
      if (wq != 0.0) {
        const int node = connectivity_(scalar.points[1].element_idx, local_node);
        for (int axis = 0; axis < 3; ++axis) {
          add_term(row, node * 3 + axis, kFEAT10ConstraintTermUsesD,
                   scalar.row_scale * wq);
        }
      }

      const double wr = scalar.points[2].shape(local_node);
      if (wr != 0.0) {
        const int node = connectivity_(scalar.points[2].element_idx, local_node);
        for (int axis = 0; axis < 3; ++axis) {
          add_term(row, node * 3 + axis, kFEAT10ConstraintTermUsesA,
                   -scalar.row_scale * wr);
        }
      }

      const double ws = scalar.points[3].shape(local_node);
      if (ws != 0.0) {
        const int node = connectivity_(scalar.points[3].element_idx, local_node);
        for (int axis = 0; axis < 3; ++axis) {
          add_term(row, node * 3 + axis, kFEAT10ConstraintTermUsesA,
                   scalar.row_scale * ws);
        }
      }
    }
  }

  for (auto& row : row_dofs) {
    std::sort(row.begin(), row.end());
  }
  for (auto& rows : dof_rows) {
    std::sort(rows.begin(), rows.end());
  }

  FEAT10GeneralConstraintLayout layout;
  layout.scalars = scalars;
  layout.j_offsets.resize(static_cast<size_t>(n_constraints) + 1, 0);
  layout.term_offsets.resize(static_cast<size_t>(n_constraints) + 1, 0);
  for (int row = 0; row < n_constraints; ++row) {
    layout.j_offsets[static_cast<size_t>(row + 1)] =
        layout.j_offsets[static_cast<size_t>(row)] +
        static_cast<int>(row_dofs[static_cast<size_t>(row)].size());
    layout.term_offsets[static_cast<size_t>(row + 1)] =
        layout.term_offsets[static_cast<size_t>(row)] +
        static_cast<int>(row_terms[static_cast<size_t>(row)].size());
    layout.j_columns.insert(layout.j_columns.end(),
                            row_dofs[static_cast<size_t>(row)].begin(),
                            row_dofs[static_cast<size_t>(row)].end());
  }

  layout.jt_offsets.resize(static_cast<size_t>(n_dofs) + 1, 0);
  for (int dof = 0; dof < n_dofs; ++dof) {
    layout.jt_offsets[static_cast<size_t>(dof + 1)] =
        layout.jt_offsets[static_cast<size_t>(dof)] +
        static_cast<int>(dof_rows[static_cast<size_t>(dof)].size());
    layout.jt_columns.insert(layout.jt_columns.end(),
                             dof_rows[static_cast<size_t>(dof)].begin(),
                             dof_rows[static_cast<size_t>(dof)].end());
  }

  for (int row = 0; row < n_constraints; ++row) {
    const auto& dofs = row_dofs[static_cast<size_t>(row)];
    const auto j_offset = layout.j_offsets[static_cast<size_t>(row)];
    for (const TermDescriptor& term : row_terms[static_cast<size_t>(row)]) {
      const auto j_it =
          std::lower_bound(dofs.begin(), dofs.end(), term.dof);
      const int j_index = j_offset + static_cast<int>(j_it - dofs.begin());

      const auto& rows = dof_rows[static_cast<size_t>(term.dof)];
      const auto jt_offset = layout.jt_offsets[static_cast<size_t>(term.dof)];
      const auto jt_it = std::lower_bound(rows.begin(), rows.end(), row);
      const int jt_index =
          jt_offset + static_cast<int>(jt_it - rows.begin());

      layout.term_kinds.push_back(term.kind);
      layout.term_scales.push_back(term.scale);
      layout.term_j_indices.push_back(j_index);
      layout.term_jt_indices.push_back(jt_index);
    }
  }

  return layout;
}

// Finalize() chooses the cheapest FEAT10 runtime mode that can represent the
// requested constraints. Pure node-to-world constraints stay on the original
// fixed-node path; any joint primitive triggers the general sparse layout.
void FEAT10ConstraintManager::Finalize() {
  if (finalized_) {
    return;
  }

  const int n_nodes = data_->get_n_coef();
  std::vector<unsigned char> seen(static_cast<size_t>(n_nodes), 0);
  std::vector<int> unique_nodes;
  unique_nodes.reserve(node_ids_.size());
  for (int node_id : node_ids_) {
    if (seen[static_cast<size_t>(node_id)] != 0) {
      continue;
    }
    seen[static_cast<size_t>(node_id)] = 1;
    unique_nodes.push_back(node_id);
  }

  constrained_nodes_.resize(static_cast<int>(unique_nodes.size()));
  for (int i = 0; i < static_cast<int>(unique_nodes.size()); ++i) {
    constrained_nodes_(i) = unique_nodes[static_cast<size_t>(i)];
  }

  if (d_constrained_nodes_ != nullptr) {
    HANDLE_ERROR(cudaFree(d_constrained_nodes_));
    d_constrained_nodes_ = nullptr;
  }

  if (constrained_nodes_.size() > 0) {
    HANDLE_ERROR(cudaMalloc(&d_constrained_nodes_,
                            constrained_nodes_.size() * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_constrained_nodes_, constrained_nodes_.data(),
                            constrained_nodes_.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
  }

  uses_general_constraints_ = !scalar_constraints_.empty();
  if (uses_general_constraints_) {
    data_->SetGeneralConstraints(BuildGeneralConstraintLayout());
  } else if (constrained_nodes_.size() > 0) {
    data_->SetNodalFixed(constrained_nodes_);
  }

  finalized_ = true;
}

void FEAT10ConstraintManager::OffsetConstrainedNodesAndTargets(int axis,
                                                               double delta) {
  if (!finalized_) {
    throw std::logic_error(
        "FEAT10ConstraintManager::OffsetConstrainedNodesAndTargets: manager not finalized");
  }
  if (uses_general_constraints_) {
    throw std::logic_error(
        "FEAT10ConstraintManager::OffsetConstrainedNodesAndTargets: general FEAT10 joint constraints do not support moving target offsets yet");
  }
  if (constrained_nodes_.size() == 0 || delta == 0.0) {
    return;
  }

  double* d_pos_axis    = nullptr;
  double* d_target_axis = nullptr;
  switch (axis) {
    case 0:
      d_pos_axis    = data_->GetX12DevicePtr();
      d_target_axis = data_->GetX12JacDevicePtr();
      break;
    case 1:
      d_pos_axis    = data_->GetY12DevicePtr();
      d_target_axis = data_->GetY12JacDevicePtr();
      break;
    case 2:
      d_pos_axis    = data_->GetZ12DevicePtr();
      d_target_axis = data_->GetZ12JacDevicePtr();
      break;
    default:
      throw std::out_of_range(
          "FEAT10ConstraintManager::OffsetConstrainedNodesAndTargets: axis must be 0, 1, or 2");
  }

  constexpr int threads = 256;
  const int blocks = (constrained_nodes_.size() + threads - 1) / threads;
  offset_selected_nodes_and_targets_kernel<<<blocks, threads>>>(
      d_pos_axis, d_target_axis, d_constrained_nodes_,
      constrained_nodes_.size(), delta);
}
