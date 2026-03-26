/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * File:    MixedConstraintSystem.cu
 * Brief:   Host-side mixed-element constraint builder and device-side
 *          constraint/Jacobian evaluation for holistic ANCF3243 + FEAT10
 *          solves.
 *==============================================================
 *==============================================================*/

#include "MixedConstraintSystem.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../../lib_utils/cpu_utils.h"
#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../elements/ANCF3243Data.cuh"
#include "../elements/FEAT10Data.cuh"

namespace {

constexpr int kT10NodeCount = Quadrature::N_NODE_T10_10;
constexpr int kBeamCoefCount = Quadrature::N_SHAPE_3243;
constexpr int kPointSlots = 4;

Eigen::Matrix<double, kT10NodeCount, 1> EvaluateT10ShapeFunctions(double xi,
                                                                  double eta,
                                                                  double zeta) {
  const double l1 = xi;
  const double l2 = eta;
  const double l3 = zeta;
  const double l4 = 1.0 - xi - eta - zeta;

  Eigen::Matrix<double, kT10NodeCount, 1> shape;
  shape(0) = l1 * (2.0 * l1 - 1.0);
  shape(1) = l2 * (2.0 * l2 - 1.0);
  shape(2) = l3 * (2.0 * l3 - 1.0);
  shape(3) = l4 * (2.0 * l4 - 1.0);
  shape(4) = 4.0 * l1 * l2;
  shape(5) = 4.0 * l2 * l3;
  shape(6) = 4.0 * l1 * l3;
  shape(7) = 4.0 * l1 * l4;
  shape(8) = 4.0 * l2 * l4;
  shape(9) = 4.0 * l3 * l4;
  return shape;
}

Eigen::Matrix<double, kT10NodeCount, 3> EvaluateT10ShapeDerivatives(double xi,
                                                                    double eta,
                                                                    double zeta) {
  const double l1 = xi;
  const double l2 = eta;
  const double l3 = zeta;
  const double l4 = 1.0 - xi - eta - zeta;

  Eigen::Matrix<double, kT10NodeCount, 3> deriv;
  deriv.row(0) << 4.0 * l1 - 1.0, 0.0, 0.0;
  deriv.row(1) << 0.0, 4.0 * l2 - 1.0, 0.0;
  deriv.row(2) << 0.0, 0.0, 4.0 * l3 - 1.0;
  deriv.row(3) << 1.0 - 4.0 * l4, 1.0 - 4.0 * l4, 1.0 - 4.0 * l4;
  deriv.row(4) << 4.0 * l2, 4.0 * l1, 0.0;
  deriv.row(5) << 0.0, 4.0 * l3, 4.0 * l2;
  deriv.row(6) << 4.0 * l3, 0.0, 4.0 * l1;
  deriv.row(7) << 4.0 * (l4 - l1), -4.0 * l1, -4.0 * l1;
  deriv.row(8) << -4.0 * l2, 4.0 * (l4 - l2), -4.0 * l2;
  deriv.row(9) << -4.0 * l3, -4.0 * l3, 4.0 * (l4 - l3);
  return deriv;
}

bool AabbContainsPoint(const Eigen::Matrix<double, 10, 3>& x_elem,
                       const Eigen::Vector3d& x, double tol) {
  const Eigen::Vector3d x_min = x_elem.colwise().minCoeff().transpose();
  const Eigen::Vector3d x_max = x_elem.colwise().maxCoeff().transpose();
  return ((x.array() >= (x_min.array() - tol)).all() &&
          (x.array() <= (x_max.array() + tol)).all());
}

Eigen::Vector3d ComputeLinearTetInitialGuess(
    const Eigen::Matrix<double, 10, 3>& x_elem, const Eigen::Vector3d& x) {
  Eigen::Matrix3d basis;
  basis.col(0) = x_elem.row(0).transpose() - x_elem.row(3).transpose();
  basis.col(1) = x_elem.row(1).transpose() - x_elem.row(3).transpose();
  basis.col(2) = x_elem.row(2).transpose() - x_elem.row(3).transpose();
  const Eigen::Vector3d rhs = x - x_elem.row(3).transpose();
  return basis.colPivHouseholderQr().solve(rhs);
}

bool IsReferenceCoordinateInsideTet(const Eigen::Vector3d& coord,
                                    double tol = 1e-8) {
  return coord[0] >= -tol && coord[1] >= -tol && coord[2] >= -tol &&
         (coord[0] + coord[1] + coord[2]) <= 1.0 + tol;
}

Eigen::Vector3d BuildPerpendicularAxis1(const Eigen::Vector3d& axis) {
  const Eigen::Vector3d trial =
      (std::abs(axis.z()) < 0.9) ? Eigen::Vector3d::UnitZ()
                                 : Eigen::Vector3d::UnitX();
  Eigen::Vector3d p1 = axis.cross(trial);
  const double norm = p1.norm();
  if (norm < 1e-12) {
    throw std::runtime_error(
        "MixedConstraintSystem: failed to construct perpendicular axis");
  }
  return p1 / norm;
}

Eigen::Matrix<double, kBeamCoefCount, 1> EvaluateANCF3243Shape(
    double xi, double eta, double zeta, double length, double width,
    double height, const Eigen::Matrix<double, 8, 8>& B_inv) {
  const double u = 0.5 * length * xi;
  const double v = 0.5 * width * eta;
  const double w = 0.5 * height * zeta;

  Eigen::Matrix<double, kBeamCoefCount, 1> b;
  b << 1.0, u, v, w, u * v, u * w, u * u, u * u * u;
  return B_inv * b;
}

Eigen::Matrix<double, kBeamCoefCount, 3> EvaluateANCF3243ShapeDerivatives(
    double xi, double eta, double zeta, double length, double width,
    double height, const Eigen::Matrix<double, 8, 8>& B_inv) {
  Eigen::Matrix<double, kBeamCoefCount, 1> db_dxi;
  Eigen::Matrix<double, kBeamCoefCount, 1> db_deta;
  Eigen::Matrix<double, kBeamCoefCount, 1> db_dzeta;

  db_dxi << 0.0, length / 2.0, 0.0, 0.0, (length * width / 4.0) * eta,
      (length * height / 4.0) * zeta, (length * length / 2.0) * xi,
      (3.0 * length * length * length / 8.0) * xi * xi;
  db_deta << 0.0, 0.0, width / 2.0, 0.0, (length * width / 4.0) * xi, 0.0,
      0.0, 0.0;
  db_dzeta << 0.0, 0.0, 0.0, height / 2.0, 0.0,
      (length * height / 4.0) * xi, 0.0, 0.0;

  Eigen::Matrix<double, kBeamCoefCount, 3> deriv;
  deriv.col(0) = B_inv * db_dxi;
  deriv.col(1) = B_inv * db_deta;
  deriv.col(2) = B_inv * db_dzeta;
  return deriv;
}

Eigen::Vector3d EvaluateANCF3243Position(
    const Eigen::Matrix<double, 8, 3>& coeffs,
    const Eigen::Matrix<double, kBeamCoefCount, 1>& shape) {
  Eigen::Vector3d x = Eigen::Vector3d::Zero();
  for (int i = 0; i < kBeamCoefCount; ++i) {
    x += shape(i) * coeffs.row(i).transpose();
  }
  return x;
}

Eigen::Matrix3d EvaluateANCF3243Jacobian(
    const Eigen::Matrix<double, 8, 3>& coeffs,
    const Eigen::Matrix<double, kBeamCoefCount, 3>& deriv) {
  Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
  for (int i = 0; i < kBeamCoefCount; ++i) {
    J += coeffs.row(i).transpose() * deriv.row(i);
  }
  return J;
}

bool IsReferenceCoordinateInsideBeam(const Eigen::Vector3d& coord,
                                     double tol = 1e-6) {
  return std::abs(coord[0]) <= 1.0 + tol && std::abs(coord[1]) <= 1.0 + tol &&
         std::abs(coord[2]) <= 1.0 + tol;
}

void FillBindingArrays(const MixedConstraintPointBinding& point, int row,
                       int slot, std::vector<int>* counts,
                       std::vector<int>* coef_indices,
                       std::vector<double>* weights) {
  const int point_offset = (row * kPointSlots + slot);
  (*counts)[static_cast<size_t>(point_offset)] = point.count;
  for (int i = 0; i < MixedConstraintPointBinding::kMaxCoefficients; ++i) {
    const int idx = point_offset * MixedConstraintPointBinding::kMaxCoefficients + i;
    (*coef_indices)[static_cast<size_t>(idx)] = point.coef_indices[i];
    (*weights)[static_cast<size_t>(idx)] = point.weights[i];
  }
}

__device__ __forceinline__ void EvaluateMixedConstraintPoint(
    int row, int slot, const int* point_counts, const int* point_coef_indices,
    const double* point_weights, const double* d_x, const double* d_y,
    const double* d_z, double out[3]) {
  out[0] = 0.0;
  out[1] = 0.0;
  out[2] = 0.0;

  const int point_offset = row * kPointSlots + slot;
  const int count = point_counts[point_offset];
  const int coef_base =
      point_offset * MixedConstraintPointBinding::kMaxCoefficients;
  for (int i = 0; i < count; ++i) {
    const int coef = point_coef_indices[coef_base + i];
    const double weight = point_weights[coef_base + i];
    out[0] += weight * d_x[coef];
    out[1] += weight * d_y[coef];
    out[2] += weight * d_z[coef];
  }
}

__global__ void evaluate_mixed_constraints_kernel(
    int n_constraints, const int* types, const int* axes, const double* targets,
    const double* row_scales, const double* world_directions,
    const int* point_counts, const int* point_coef_indices,
    const double* point_weights, const double* d_x, const double* d_y,
    const double* d_z, double* d_constraint) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_constraints) {
    return;
  }

  double p[3], q[3], r[3], s[3];
  EvaluateMixedConstraintPoint(row, 0, point_counts, point_coef_indices,
                               point_weights, d_x, d_y, d_z, p);

  const int type = types[row];
  const int axis = axes[row];
  const double scale = row_scales[row];
  if (type == kMixedConstraintPointWorldCD) {
    d_constraint[row] = scale * (p[axis] - targets[row]);
    return;
  }

  EvaluateMixedConstraintPoint(row, 1, point_counts, point_coef_indices,
                               point_weights, d_x, d_y, d_z, q);
  if (type == kMixedConstraintPointPointCD) {
    d_constraint[row] = scale * ((q[axis] - p[axis]) - targets[row]);
    return;
  }

  double a[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
  double d[3] = {0.0, 0.0, 0.0};
  if (type == kMixedConstraintWorldDP1) {
    d[0] = world_directions[row * 3 + 0];
    d[1] = world_directions[row * 3 + 1];
    d[2] = world_directions[row * 3 + 2];
  } else {
    EvaluateMixedConstraintPoint(row, 2, point_counts, point_coef_indices,
                                 point_weights, d_x, d_y, d_z, r);
    EvaluateMixedConstraintPoint(row, 3, point_counts, point_coef_indices,
                                 point_weights, d_x, d_y, d_z, s);
    d[0] = s[0] - r[0];
    d[1] = s[1] - r[1];
    d[2] = s[2] - r[2];
  }

  d_constraint[row] =
      scale * (a[0] * d[0] + a[1] * d[1] + a[2] * d[2] - targets[row]);
}

__global__ void build_mixed_constraint_jacobian_values_kernel(
    int n_constraints, const int* types, const double* world_directions,
    const int* term_offsets, const int* term_kinds, const double* term_scales,
    const int* term_j_indices, const int* term_jt_indices, const int* j_columns,
    const int* point_counts, const int* point_coef_indices,
    const double* point_weights, const double* d_x, const double* d_y,
    const double* d_z, double* d_j_values, double* d_jt_values) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_constraints) {
    return;
  }

  double p[3], q[3], r[3], s[3];
  double a[3] = {0.0, 0.0, 0.0};
  double d[3] = {0.0, 0.0, 0.0};

  const int type = types[row];
  if (type == kMixedConstraintWorldDP1 || type == kMixedConstraintDP1 ||
      type == kMixedConstraintDP2) {
    EvaluateMixedConstraintPoint(row, 0, point_counts, point_coef_indices,
                                 point_weights, d_x, d_y, d_z, p);
    EvaluateMixedConstraintPoint(row, 1, point_counts, point_coef_indices,
                                 point_weights, d_x, d_y, d_z, q);
    a[0] = q[0] - p[0];
    a[1] = q[1] - p[1];
    a[2] = q[2] - p[2];

    if (type == kMixedConstraintWorldDP1) {
      d[0] = world_directions[row * 3 + 0];
      d[1] = world_directions[row * 3 + 1];
      d[2] = world_directions[row * 3 + 2];
    } else {
      EvaluateMixedConstraintPoint(row, 2, point_counts, point_coef_indices,
                                   point_weights, d_x, d_y, d_z, r);
      EvaluateMixedConstraintPoint(row, 3, point_counts, point_coef_indices,
                                   point_weights, d_x, d_y, d_z, s);
      d[0] = s[0] - r[0];
      d[1] = s[1] - r[1];
      d[2] = s[2] - r[2];
    }
  }

  const int term_begin = term_offsets[row];
  const int term_end = term_offsets[row + 1];
  for (int term_idx = term_begin; term_idx < term_end; ++term_idx) {
    const int j_index = term_j_indices[term_idx];
    const int jt_index = term_jt_indices[term_idx];
    const int dof_component = j_columns[j_index] % 3;

    double value = term_scales[term_idx];
    const int kind = term_kinds[term_idx];
    if (kind == kMixedConstraintTermUsesA) {
      value *= a[dof_component];
    } else if (kind == kMixedConstraintTermUsesD) {
      value *= d[dof_component];
    }

    d_j_values[j_index] += value;
    d_jt_values[jt_index] += value;
  }
}

}  // namespace

MixedConstraintSystem::MixedConstraintSystem(FEMultiElementProblem* problem)
    : problem_(problem) {
  if (problem_ == nullptr) {
    throw std::invalid_argument(
        "MixedConstraintSystem: problem pointer cannot be null");
  }
  if (!problem_->IsFinalized()) {
    throw std::invalid_argument(
        "MixedConstraintSystem: problem must be finalized before constraints are constructed");
  }
  BuildBlockCaches();
}

MixedConstraintSystem::~MixedConstraintSystem() {
  if (d_constraint_ != nullptr) {
    cudaFree(d_constraint_);
    cudaFree(d_constraint_types_);
    cudaFree(d_constraint_axes_);
    cudaFree(d_constraint_targets_);
    cudaFree(d_constraint_row_scales_);
    cudaFree(d_constraint_world_directions_);
    cudaFree(d_point_counts_);
    cudaFree(d_point_coef_indices_);
    cudaFree(d_point_weights_);
    cudaFree(d_term_offsets_);
    cudaFree(d_term_kinds_);
    cudaFree(d_term_scales_);
    cudaFree(d_term_j_indices_);
    cudaFree(d_term_jt_indices_);
    cudaFree(d_j_csr_offsets_);
    cudaFree(d_j_csr_columns_);
    cudaFree(d_j_csr_values_);
    cudaFree(d_cj_csr_offsets_);
    cudaFree(d_cj_csr_columns_);
    cudaFree(d_cj_csr_values_);
  }
}

void MixedConstraintSystem::BuildBlockCaches() {
  const FEStateBuffer& state = problem_->GetStateBuffer();
  const int n_blocks = problem_->GetNumBlocks();
  block_cache_.resize(static_cast<size_t>(n_blocks));

  reference_x_.setZero(problem_->GetTotalCoef());
  reference_y_.setZero(problem_->GetTotalCoef());
  reference_z_.setZero(problem_->GetTotalCoef());

  for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
    BlockCache& cache = block_cache_[static_cast<size_t>(block_idx)];
    cache.type = problem_->GetElementType(block_idx);
    cache.coef_offset = state.blocks[static_cast<size_t>(block_idx)].coef_offset;
    cache.coef_count = state.blocks[static_cast<size_t>(block_idx)].coef_count;

    ElementBase* element = problem_->GetElementData(block_idx);
    if (cache.type == TYPE_T10) {
      auto* data = static_cast<GPU_FEAT10_Data*>(element);
      data->RetrieveReferencePositionToCPU(cache.x_ref, cache.y_ref, cache.z_ref);
      data->RetrieveConnectivityToCPU(cache.connectivity);
    } else if (cache.type == TYPE_3243) {
      auto* data = static_cast<GPU_ANCF3243_Data*>(element);
      data->RetrievePositionToCPU(cache.x_ref, cache.y_ref, cache.z_ref);
      data->RetrieveConnectivityToCPU(cache.connectivity);
      data->RetrieveElementDimensionsToCPU(cache.beam_length, cache.beam_width,
                                           cache.beam_height);

      Eigen::VectorXd b_inv_flat;
      ANCFCPUUtils::ANCF3243_B12_matrix_flat_per_element(
          cache.beam_length, cache.beam_width, cache.beam_height, b_inv_flat,
          Quadrature::N_SHAPE_3243);
      cache.beam_B_inv.resize(static_cast<size_t>(cache.connectivity.rows()));
      for (int elem_idx = 0; elem_idx < cache.connectivity.rows(); ++elem_idx) {
        Eigen::Matrix<double, 8, 8> B_inv;
        for (int r = 0; r < 8; ++r) {
          for (int c = 0; c < 8; ++c) {
            const int flat_idx = elem_idx * 64 + r * 8 + c;
            B_inv(r, c) = b_inv_flat(flat_idx);
          }
        }
        cache.beam_B_inv[static_cast<size_t>(elem_idx)] = B_inv;
      }
    } else {
      throw std::invalid_argument(
          "MixedConstraintSystem: only TYPE_T10 and TYPE_3243 are supported");
    }

    reference_x_.segment(cache.coef_offset, cache.coef_count) = cache.x_ref;
    reference_y_.segment(cache.coef_offset, cache.coef_count) = cache.y_ref;
    reference_z_.segment(cache.coef_offset, cache.coef_count) = cache.z_ref;
  }
}

MixedConstraintPointBinding MixedConstraintSystem::MakeCoefficientBinding(
    int global_coef) const {
  if (global_coef < 0 || global_coef >= problem_->GetTotalCoef()) {
    throw std::out_of_range(
        "MixedConstraintSystem::MakeCoefficientBinding: coefficient index out of range");
  }
  MixedConstraintPointBinding point;
  point.count = 1;
  point.coef_indices[0] = global_coef;
  point.weights[0] = 1.0;
  return point;
}

MixedConstraintPointBinding MixedConstraintSystem::LocateReferencePoint(
    int block_idx, const Eigen::Vector3d& reference_point) const {
  if (block_idx < 0 || block_idx >= static_cast<int>(block_cache_.size())) {
    throw std::out_of_range(
        "MixedConstraintSystem::LocateReferencePoint: block index out of range");
  }

  const BlockCache& cache = block_cache_[static_cast<size_t>(block_idx)];
  if (cache.type == TYPE_T10) {
    return LocateReferencePointT10(cache, reference_point);
  }
  if (cache.type == TYPE_3243) {
    return LocateReferencePointANCF3243(cache, reference_point);
  }
  throw std::invalid_argument(
      "MixedConstraintSystem::LocateReferencePoint: unsupported element type");
}

void MixedConstraintSystem::AddPointToPointCDAxis(
    const MixedConstraintPointBinding& p, const MixedConstraintPointBinding& r,
    int axis, double target) {
  if (finalized_) {
    throw std::logic_error(
        "MixedConstraintSystem::AddPointToPointCDAxis: system already finalized");
  }
  if (axis < 0 || axis > 2) {
    throw std::out_of_range(
        "MixedConstraintSystem::AddPointToPointCDAxis: axis must be 0, 1, or 2");
  }

  MixedConstraintScalar scalar;
  scalar.type = kMixedConstraintPointPointCD;
  scalar.axis = axis;
  scalar.target = target;
  scalar.points[0] = p;
  scalar.points[1] = r;
  scalar_constraints_.push_back(scalar);
}

void MixedConstraintSystem::AddPointToWorldCDAxis(
    const MixedConstraintPointBinding& p, int axis, double target) {
  if (finalized_) {
    throw std::logic_error(
        "MixedConstraintSystem::AddPointToWorldCDAxis: system already finalized");
  }
  if (axis < 0 || axis > 2) {
    throw std::out_of_range(
        "MixedConstraintSystem::AddPointToWorldCDAxis: axis must be 0, 1, or 2");
  }

  MixedConstraintScalar scalar;
  scalar.type = kMixedConstraintPointWorldCD;
  scalar.axis = axis;
  scalar.target = target;
  scalar.points[0] = p;
  scalar_constraints_.push_back(scalar);
}

void MixedConstraintSystem::AddWorldDP1Constraint(
    const MixedConstraintPointBinding& p, const MixedConstraintPointBinding& q,
    const Eigen::Vector3d& world_direction, double target, double weight) {
  if (finalized_) {
    throw std::logic_error(
        "MixedConstraintSystem::AddWorldDP1Constraint: system already finalized");
  }
  if (weight <= 0.0) {
    throw std::invalid_argument(
        "MixedConstraintSystem::AddWorldDP1Constraint: weight must be positive");
  }

  const double d_norm = world_direction.norm();
  if (d_norm < 1e-12) {
    throw std::invalid_argument(
        "MixedConstraintSystem::AddWorldDP1Constraint: world direction must be non-zero");
  }

  MixedConstraintScalar scalar;
  scalar.type = kMixedConstraintWorldDP1;
  scalar.target = target;
  scalar.world_direction = world_direction;
  scalar.row_scale = weight / d_norm;
  scalar.points[0] = p;
  scalar.points[1] = q;
  scalar_constraints_.push_back(scalar);
}

void MixedConstraintSystem::AddDotProductConstraint(
    int type, const MixedConstraintPointBinding& p,
    const MixedConstraintPointBinding& q, const MixedConstraintPointBinding& r,
    const MixedConstraintPointBinding& s, double target, double weight) {
  if (finalized_) {
    throw std::logic_error(
        "MixedConstraintSystem::AddDotProductConstraint: system already finalized");
  }
  if (weight <= 0.0) {
    throw std::invalid_argument(
        "MixedConstraintSystem::AddDotProductConstraint: weight must be positive");
  }
  if (type != kMixedConstraintDP1 && type != kMixedConstraintDP2) {
    throw std::invalid_argument(
        "MixedConstraintSystem::AddDotProductConstraint: invalid constraint type");
  }

  const Eigen::Vector3d p_ref = EvaluateReferencePoint(p);
  const Eigen::Vector3d q_ref = EvaluateReferencePoint(q);
  const Eigen::Vector3d r_ref = EvaluateReferencePoint(r);
  const Eigen::Vector3d s_ref = EvaluateReferencePoint(s);
  const double a_norm = (q_ref - p_ref).norm();
  const double d_norm = (s_ref - r_ref).norm();
  const double row_norm = std::sqrt(a_norm * a_norm + d_norm * d_norm);

  MixedConstraintScalar scalar;
  scalar.type = type;
  scalar.target = target;
  scalar.row_scale = weight;
  if (row_norm > 1e-12) {
    scalar.row_scale *= 1.0 / row_norm;
  }
  scalar.points[0] = p;
  scalar.points[1] = q;
  scalar.points[2] = r;
  scalar.points[3] = s;
  scalar_constraints_.push_back(scalar);
}

void MixedConstraintSystem::AddDP1Constraint(
    const MixedConstraintPointBinding& p, const MixedConstraintPointBinding& q,
    const MixedConstraintPointBinding& r, const MixedConstraintPointBinding& s,
    double target, double weight) {
  AddDotProductConstraint(kMixedConstraintDP1, p, q, r, s, target, weight);
}

void MixedConstraintSystem::AddDP2Constraint(
    const MixedConstraintPointBinding& p, const MixedConstraintPointBinding& q,
    const MixedConstraintPointBinding& r, const MixedConstraintPointBinding& s,
    double target, double weight) {
  AddDotProductConstraint(kMixedConstraintDP2, p, q, r, s, target, weight);
}

void MixedConstraintSystem::AddSphericalJoint(
    const MixedConstraintPointBinding& p, const MixedConstraintPointBinding& r) {
  AddPointToPointCDAxis(p, r, 0);
  AddPointToPointCDAxis(p, r, 1);
  AddPointToPointCDAxis(p, r, 2);
}

void MixedConstraintSystem::AddRevoluteJoint(
    const MixedConstraintPointBinding& p, const MixedConstraintPointBinding& q,
    const MixedConstraintPointBinding& r, const MixedConstraintPointBinding& s,
    const MixedConstraintPointBinding& t, double f1, double f2,
    double dp1_weight) {
  AddPointToPointCDAxis(p, r, 0);
  AddPointToPointCDAxis(p, r, 1);
  AddPointToPointCDAxis(p, r, 2);
  AddDP1Constraint(p, q, r, s, f1, dp1_weight);
  AddDP1Constraint(p, q, r, t, f2, dp1_weight);
}

void MixedConstraintSystem::AddFixedJoint(
    const MixedConstraintPointBinding& p, const MixedConstraintPointBinding& q,
    const MixedConstraintPointBinding& w, const MixedConstraintPointBinding& r,
    const MixedConstraintPointBinding& s, const MixedConstraintPointBinding& t,
    double f1, double f2, double f3, double dp1_weight) {
  AddPointToPointCDAxis(p, r, 0);
  AddPointToPointCDAxis(p, r, 1);
  AddPointToPointCDAxis(p, r, 2);
  AddDP1Constraint(p, q, r, s, f1, dp1_weight);
  AddDP1Constraint(p, q, r, t, f2, dp1_weight);
  AddDP1Constraint(p, w, r, t, f3, dp1_weight);
}

void MixedConstraintSystem::AddCylindricalJoint(
    const MixedConstraintPointBinding& p, const MixedConstraintPointBinding& q,
    const MixedConstraintPointBinding& r, const MixedConstraintPointBinding& s,
    const MixedConstraintPointBinding& u, const MixedConstraintPointBinding& v,
    const MixedConstraintPointBinding& w, double f_par1, double f_par2,
    double f_col1, double f_col2, double dp1_weight, double dp2_weight) {
  AddDP1Constraint(p, q, r, s, f_par1, dp1_weight);
  AddDP1Constraint(p, q, r, u, f_par2, dp1_weight);
  AddDP2Constraint(p, v, p, r, f_col1, dp2_weight);
  AddDP2Constraint(p, w, p, r, f_col2, dp2_weight);
}

void MixedConstraintSystem::AddSphericalJoint(
    int block_idx_a, int block_idx_b, const Eigen::Vector3d& hinge_point) {
  const MixedConstraintPointBinding p =
      LocateReferencePoint(block_idx_a, hinge_point);
  const MixedConstraintPointBinding r =
      LocateReferencePoint(block_idx_b, hinge_point);
  AddSphericalJoint(p, r);
}

double MixedConstraintSystem::ComputeDefaultOffset(
    const MixedConstraintPointBinding& a,
    const MixedConstraintPointBinding& b) const {
  const Eigen::Vector3d xa = EvaluateReferencePoint(a);
  const Eigen::Vector3d xb = EvaluateReferencePoint(b);
  return 1e-2 * std::max(1e-6, (xb - xa).norm());
}

MixedConstraintPointBinding MixedConstraintSystem::LocateWithAdaptiveOffset(
    int block_idx, const Eigen::Vector3d& base_point,
    const Eigen::Vector3d& direction, double initial_offset) const {
  double offset = initial_offset;
  for (int attempt = 0; attempt < 8; ++attempt) {
    try {
      return LocateReferencePoint(block_idx, base_point + offset * direction);
    } catch (const std::runtime_error&) {
      offset *= 0.5;
    }
  }
  throw std::runtime_error(
      "MixedConstraintSystem::LocateWithAdaptiveOffset: failed to place offset reference point");
}

void MixedConstraintSystem::AddRevoluteJoint(
    int block_idx_a, int block_idx_b, const Eigen::Vector3d& hinge_point,
    const Eigen::Vector3d& hinge_axis, double offset, double dp1_weight) {
  const double axis_norm = hinge_axis.norm();
  if (axis_norm < 1e-12) {
    throw std::invalid_argument(
        "MixedConstraintSystem::AddRevoluteJoint: hinge axis must be non-zero");
  }

  const Eigen::Vector3d axis = hinge_axis / axis_norm;
  const MixedConstraintPointBinding p =
      LocateReferencePoint(block_idx_a, hinge_point);
  const MixedConstraintPointBinding r =
      LocateReferencePoint(block_idx_b, hinge_point);
  if (offset <= 0.0) {
    offset = ComputeDefaultOffset(p, r);
  }

  const MixedConstraintPointBinding q =
      LocateWithAdaptiveOffset(block_idx_a, hinge_point, axis, offset);
  const Eigen::Vector3d p1 = BuildPerpendicularAxis1(axis);
  const Eigen::Vector3d p2 = axis.cross(p1).normalized();
  const MixedConstraintPointBinding s =
      LocateWithAdaptiveOffset(block_idx_b, hinge_point, p1, offset);
  const MixedConstraintPointBinding t =
      LocateWithAdaptiveOffset(block_idx_b, hinge_point, p2, offset);

  const Eigen::Vector3d p_ref = EvaluateReferencePoint(p);
  const Eigen::Vector3d q_ref = EvaluateReferencePoint(q);
  const Eigen::Vector3d r_ref = EvaluateReferencePoint(r);
  const Eigen::Vector3d s_ref = EvaluateReferencePoint(s);
  const Eigen::Vector3d t_ref = EvaluateReferencePoint(t);
  const double f1 = (q_ref - p_ref).dot(s_ref - r_ref);
  const double f2 = (q_ref - p_ref).dot(t_ref - r_ref);
  AddRevoluteJoint(p, q, r, s, t, f1, f2, dp1_weight);
}

void MixedConstraintSystem::AddFixedJoint(
    int block_idx_a, int block_idx_b, const Eigen::Vector3d& joint_point,
    double offset, double dp1_weight) {
  const MixedConstraintPointBinding p =
      LocateReferencePoint(block_idx_a, joint_point);
  const MixedConstraintPointBinding r =
      LocateReferencePoint(block_idx_b, joint_point);
  if (offset <= 0.0) {
    offset = ComputeDefaultOffset(p, r);
  }

  // A weld removes all relative orientation, so any deterministic,
  // non-degenerate reference frame is acceptable for constructing the
  // offset directions used by the DP1 rows.
  const Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d p1 = BuildPerpendicularAxis1(axis);
  const Eigen::Vector3d p2 = axis.cross(p1).normalized();

  const MixedConstraintPointBinding q =
      LocateWithAdaptiveOffset(block_idx_a, joint_point, axis, offset);
  const MixedConstraintPointBinding w =
      LocateWithAdaptiveOffset(block_idx_a, joint_point, p1, offset);
  const MixedConstraintPointBinding s =
      LocateWithAdaptiveOffset(block_idx_b, joint_point, p1, offset);
  const MixedConstraintPointBinding t =
      LocateWithAdaptiveOffset(block_idx_b, joint_point, p2, offset);

  const Eigen::Vector3d p_ref = EvaluateReferencePoint(p);
  const Eigen::Vector3d q_ref = EvaluateReferencePoint(q);
  const Eigen::Vector3d w_ref = EvaluateReferencePoint(w);
  const Eigen::Vector3d r_ref = EvaluateReferencePoint(r);
  const Eigen::Vector3d s_ref = EvaluateReferencePoint(s);
  const Eigen::Vector3d t_ref = EvaluateReferencePoint(t);

  const Eigen::Vector3d a1_ref = q_ref - p_ref;
  const Eigen::Vector3d a2_ref = w_ref - p_ref;
  const Eigen::Vector3d d1_ref = s_ref - r_ref;
  const Eigen::Vector3d d2_ref = t_ref - r_ref;

  const double f1 = a1_ref.dot(d1_ref);
  const double f2 = a1_ref.dot(d2_ref);
  const double f3 = a2_ref.dot(d2_ref);
  AddFixedJoint(p, q, w, r, s, t, f1, f2, f3, dp1_weight);
}

void MixedConstraintSystem::AddCylindricalJoint(
    int block_idx_a, int block_idx_b, const Eigen::Vector3d& axis_point_a,
    const Eigen::Vector3d& axis_point_b, const Eigen::Vector3d& axis_direction,
    double offset, double dp1_weight, double dp2_weight) {
  const double axis_norm = axis_direction.norm();
  if (axis_norm < 1e-12) {
    throw std::invalid_argument(
        "MixedConstraintSystem::AddCylindricalJoint: axis direction must be non-zero");
  }

  const Eigen::Vector3d axis = axis_direction / axis_norm;
  const MixedConstraintPointBinding p =
      LocateReferencePoint(block_idx_a, axis_point_a);
  const MixedConstraintPointBinding r =
      LocateReferencePoint(block_idx_b, axis_point_b);
  if (offset <= 0.0) {
    offset = ComputeDefaultOffset(p, r);
  }

  const MixedConstraintPointBinding q =
      LocateWithAdaptiveOffset(block_idx_a, axis_point_a, axis, offset);
  const Eigen::Vector3d p1 = BuildPerpendicularAxis1(axis);
  const Eigen::Vector3d p2 = axis.cross(p1).normalized();

  const MixedConstraintPointBinding s =
      LocateWithAdaptiveOffset(block_idx_b, axis_point_b, p1, offset);
  const MixedConstraintPointBinding u =
      LocateWithAdaptiveOffset(block_idx_b, axis_point_b, p2, offset);
  const MixedConstraintPointBinding v =
      LocateWithAdaptiveOffset(block_idx_a, axis_point_a, p1, offset);
  const MixedConstraintPointBinding w =
      LocateWithAdaptiveOffset(block_idx_a, axis_point_a, p2, offset);

  const Eigen::Vector3d p_ref = EvaluateReferencePoint(p);
  const Eigen::Vector3d q_ref = EvaluateReferencePoint(q);
  const Eigen::Vector3d r_ref = EvaluateReferencePoint(r);
  const Eigen::Vector3d s_ref = EvaluateReferencePoint(s);
  const Eigen::Vector3d u_ref = EvaluateReferencePoint(u);
  const Eigen::Vector3d v_ref = EvaluateReferencePoint(v);
  const Eigen::Vector3d w_ref = EvaluateReferencePoint(w);

  const Eigen::Vector3d a_ref = q_ref - p_ref;
  const Eigen::Vector3d d1_ref = s_ref - r_ref;
  const Eigen::Vector3d d2_ref = u_ref - r_ref;
  const Eigen::Vector3d p1_ref = v_ref - p_ref;
  const Eigen::Vector3d p2_ref = w_ref - p_ref;
  const Eigen::Vector3d b_ref = r_ref - p_ref;

  const double f_par1 = a_ref.dot(d1_ref);
  const double f_par2 = a_ref.dot(d2_ref);
  const double f_col1 = p1_ref.dot(b_ref);
  const double f_col2 = p2_ref.dot(b_ref);

  AddCylindricalJoint(p, q, r, s, u, v, w, f_par1, f_par2, f_col1, f_col2,
                      dp1_weight, dp2_weight);
}

Eigen::Vector3d MixedConstraintSystem::EvaluateReferencePoint(
    const MixedConstraintPointBinding& point) const {
  Eigen::Vector3d x = Eigen::Vector3d::Zero();
  for (int i = 0; i < point.count; ++i) {
    const int coef = point.coef_indices[i];
    const double weight = point.weights[i];
    x.x() += weight * reference_x_(coef);
    x.y() += weight * reference_y_(coef);
    x.z() += weight * reference_z_(coef);
  }
  return x;
}

MixedConstraintLayout MixedConstraintSystem::BuildLayout() const {
  MixedConstraintLayout layout;
  layout.scalars = scalar_constraints_;

  const int n_constraints = static_cast<int>(layout.scalars.size());
  const int n_dofs = problem_->GetTotalDofs();
  layout.point_counts.resize(static_cast<size_t>(n_constraints) * kPointSlots, 0);
  layout.point_coef_indices.resize(static_cast<size_t>(n_constraints) *
                                       kPointSlots *
                                       MixedConstraintPointBinding::kMaxCoefficients,
                                   -1);
  layout.point_weights.resize(static_cast<size_t>(n_constraints) * kPointSlots *
                                  MixedConstraintPointBinding::kMaxCoefficients,
                              0.0);

  std::vector<std::vector<int>> row_dofs(static_cast<size_t>(n_constraints));
  std::vector<std::vector<int>> dof_rows(static_cast<size_t>(n_dofs));
  std::vector<std::vector<ScalarConstraintTerm>> row_terms(
      static_cast<size_t>(n_constraints));

  auto add_unique = [](std::vector<int>* values, int entry) {
    if (std::find(values->begin(), values->end(), entry) == values->end()) {
      values->push_back(entry);
    }
  };

  auto add_term = [&](int row, int dof, int kind, double scale) {
    add_unique(&row_dofs[static_cast<size_t>(row)], dof);
    add_unique(&dof_rows[static_cast<size_t>(dof)], row);
    row_terms[static_cast<size_t>(row)].push_back({dof, kind, scale});
  };

  for (int row = 0; row < n_constraints; ++row) {
    const MixedConstraintScalar& scalar = layout.scalars[static_cast<size_t>(row)];
    for (int slot = 0; slot < kPointSlots; ++slot) {
      FillBindingArrays(scalar.points[slot], row, slot, &layout.point_counts,
                        &layout.point_coef_indices, &layout.point_weights);
    }

    if (scalar.type == kMixedConstraintPointWorldCD) {
      const MixedConstraintPointBinding& p = scalar.points[0];
      for (int i = 0; i < p.count; ++i) {
        add_term(row, p.coef_indices[i] * 3 + scalar.axis,
                 kMixedConstraintTermConstant, scalar.row_scale * p.weights[i]);
      }
      continue;
    }

    if (scalar.type == kMixedConstraintPointPointCD) {
      const MixedConstraintPointBinding& p = scalar.points[0];
      const MixedConstraintPointBinding& r = scalar.points[1];
      for (int i = 0; i < p.count; ++i) {
        add_term(row, p.coef_indices[i] * 3 + scalar.axis,
                 kMixedConstraintTermConstant,
                 -scalar.row_scale * p.weights[i]);
      }
      for (int i = 0; i < r.count; ++i) {
        add_term(row, r.coef_indices[i] * 3 + scalar.axis,
                 kMixedConstraintTermConstant,
                 scalar.row_scale * r.weights[i]);
      }
      continue;
    }

    if (scalar.type == kMixedConstraintWorldDP1) {
      const MixedConstraintPointBinding& p = scalar.points[0];
      const MixedConstraintPointBinding& q = scalar.points[1];
      for (int i = 0; i < p.count; ++i) {
        const int coef = p.coef_indices[i];
        for (int axis = 0; axis < 3; ++axis) {
          add_term(row, coef * 3 + axis, kMixedConstraintTermUsesD,
                   -scalar.row_scale * p.weights[i]);
        }
      }
      for (int i = 0; i < q.count; ++i) {
        const int coef = q.coef_indices[i];
        for (int axis = 0; axis < 3; ++axis) {
          add_term(row, coef * 3 + axis, kMixedConstraintTermUsesD,
                   scalar.row_scale * q.weights[i]);
        }
      }
      continue;
    }

    for (int slot = 0; slot < 4; ++slot) {
      const MixedConstraintPointBinding& point = scalar.points[slot];
      for (int i = 0; i < point.count; ++i) {
        const int coef = point.coef_indices[i];
        const double sign =
            (slot == 0 || slot == 2) ? -1.0 : 1.0;
        const int kind =
            (slot == 0 || slot == 1) ? kMixedConstraintTermUsesD
                                     : kMixedConstraintTermUsesA;
        for (int axis = 0; axis < 3; ++axis) {
          add_term(row, coef * 3 + axis, kind,
                   sign * scalar.row_scale * point.weights[i]);
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
    const int j_offset = layout.j_offsets[static_cast<size_t>(row)];
    for (const ScalarConstraintTerm& term : row_terms[static_cast<size_t>(row)]) {
      const auto j_it = std::lower_bound(dofs.begin(), dofs.end(), term.dof);
      const int j_index = j_offset + static_cast<int>(j_it - dofs.begin());

      const auto& rows = dof_rows[static_cast<size_t>(term.dof)];
      const int jt_offset = layout.jt_offsets[static_cast<size_t>(term.dof)];
      const auto jt_it = std::lower_bound(rows.begin(), rows.end(), row);
      const int jt_index = jt_offset + static_cast<int>(jt_it - rows.begin());

      layout.term_kinds.push_back(term.kind);
      layout.term_scales.push_back(term.scale);
      layout.term_j_indices.push_back(j_index);
      layout.term_jt_indices.push_back(jt_index);
    }
  }

  return layout;
}

void MixedConstraintSystem::Finalize() {
  if (finalized_) {
    return;
  }

  host_layout_ = BuildLayout();
  n_constraints_ = static_cast<int>(host_layout_.scalars.size());
  if (n_constraints_ == 0) {
    finalized_ = true;
    return;
  }

  std::vector<int> types(static_cast<size_t>(n_constraints_), 0);
  std::vector<int> axes(static_cast<size_t>(n_constraints_), 0);
  std::vector<double> targets(static_cast<size_t>(n_constraints_), 0.0);
  std::vector<double> row_scales(static_cast<size_t>(n_constraints_), 1.0);
  std::vector<double> world_directions(static_cast<size_t>(n_constraints_) * 3,
                                       0.0);
  for (int row = 0; row < n_constraints_; ++row) {
    const MixedConstraintScalar& scalar =
        host_layout_.scalars[static_cast<size_t>(row)];
    types[static_cast<size_t>(row)] = scalar.type;
    axes[static_cast<size_t>(row)] = scalar.axis;
    targets[static_cast<size_t>(row)] = scalar.target;
    row_scales[static_cast<size_t>(row)] = scalar.row_scale;
    world_directions[static_cast<size_t>(row) * 3 + 0] =
        scalar.world_direction.x();
    world_directions[static_cast<size_t>(row) * 3 + 1] =
        scalar.world_direction.y();
    world_directions[static_cast<size_t>(row) * 3 + 2] =
        scalar.world_direction.z();
  }

  const int j_nnz = static_cast<int>(host_layout_.j_columns.size());
  const int jt_nnz = static_cast<int>(host_layout_.jt_columns.size());
  const size_t constraint_bytes =
      static_cast<size_t>(n_constraints_) * sizeof(double);

  HANDLE_ERROR(cudaMalloc(&d_constraint_, constraint_bytes));
  HANDLE_ERROR(cudaMemset(d_constraint_, 0, constraint_bytes));

  HANDLE_ERROR(cudaMalloc(&d_constraint_types_,
                          static_cast<size_t>(n_constraints_) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_constraint_axes_,
                          static_cast<size_t>(n_constraints_) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_constraint_targets_, constraint_bytes));
  HANDLE_ERROR(cudaMalloc(&d_constraint_row_scales_, constraint_bytes));
  HANDLE_ERROR(cudaMalloc(&d_constraint_world_directions_,
                          world_directions.size() * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_point_counts_,
                          host_layout_.point_counts.size() * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_point_coef_indices_,
                          host_layout_.point_coef_indices.size() * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_point_weights_,
                          host_layout_.point_weights.size() * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_term_offsets_,
                          host_layout_.term_offsets.size() * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_term_kinds_,
                          host_layout_.term_kinds.size() * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_term_scales_,
                          host_layout_.term_scales.size() * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_term_j_indices_,
                          host_layout_.term_j_indices.size() * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_term_jt_indices_,
                          host_layout_.term_jt_indices.size() * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_j_csr_offsets_,
                          host_layout_.j_offsets.size() * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_j_csr_columns_,
                          static_cast<size_t>(j_nnz) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_j_csr_values_,
                          static_cast<size_t>(j_nnz) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_cj_csr_offsets_,
                          host_layout_.jt_offsets.size() * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_cj_csr_columns_,
                          static_cast<size_t>(jt_nnz) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_cj_csr_values_,
                          static_cast<size_t>(jt_nnz) * sizeof(double)));

  HANDLE_ERROR(cudaMemcpy(d_constraint_types_, types.data(),
                          static_cast<size_t>(n_constraints_) * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_constraint_axes_, axes.data(),
                          static_cast<size_t>(n_constraints_) * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_constraint_targets_, targets.data(),
                          constraint_bytes, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_constraint_row_scales_, row_scales.data(),
                          constraint_bytes, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_constraint_world_directions_,
                          world_directions.data(),
                          world_directions.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_point_counts_, host_layout_.point_counts.data(),
                          host_layout_.point_counts.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_point_coef_indices_,
                          host_layout_.point_coef_indices.data(),
                          host_layout_.point_coef_indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_point_weights_, host_layout_.point_weights.data(),
                          host_layout_.point_weights.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_term_offsets_, host_layout_.term_offsets.data(),
                          host_layout_.term_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_term_kinds_, host_layout_.term_kinds.data(),
                          host_layout_.term_kinds.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_term_scales_, host_layout_.term_scales.data(),
                          host_layout_.term_scales.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_term_j_indices_, host_layout_.term_j_indices.data(),
                          host_layout_.term_j_indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_term_jt_indices_,
                          host_layout_.term_jt_indices.data(),
                          host_layout_.term_jt_indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_j_csr_offsets_, host_layout_.j_offsets.data(),
                          host_layout_.j_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_j_csr_columns_, host_layout_.j_columns.data(),
                          static_cast<size_t>(j_nnz) * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_cj_csr_offsets_, host_layout_.jt_offsets.data(),
                          host_layout_.jt_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_cj_csr_columns_, host_layout_.jt_columns.data(),
                          static_cast<size_t>(jt_nnz) * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(d_j_csr_values_, 0,
                          static_cast<size_t>(j_nnz) * sizeof(double)));
  HANDLE_ERROR(cudaMemset(d_cj_csr_values_, 0,
                          static_cast<size_t>(jt_nnz) * sizeof(double)));

  finalized_ = true;
}

void MixedConstraintSystem::Evaluate(double* d_x, double* d_y, double* d_z) {
  if (!finalized_ || n_constraints_ == 0) {
    return;
  }

  const int threads = 256;
  const int blocks = (n_constraints_ + threads - 1) / threads;
  HANDLE_ERROR(cudaMemset(
      d_j_csr_values_, 0,
      static_cast<size_t>(host_layout_.j_columns.size()) * sizeof(double)));
  HANDLE_ERROR(cudaMemset(
      d_cj_csr_values_, 0,
      static_cast<size_t>(host_layout_.jt_columns.size()) * sizeof(double)));

  evaluate_mixed_constraints_kernel<<<blocks, threads>>>(
      n_constraints_, d_constraint_types_, d_constraint_axes_,
      d_constraint_targets_, d_constraint_row_scales_,
      d_constraint_world_directions_, d_point_counts_, d_point_coef_indices_,
      d_point_weights_, d_x, d_y, d_z, d_constraint_);
  build_mixed_constraint_jacobian_values_kernel<<<blocks, threads>>>(
      n_constraints_, d_constraint_types_, d_constraint_world_directions_,
      d_term_offsets_, d_term_kinds_, d_term_scales_, d_term_j_indices_,
      d_term_jt_indices_, d_j_csr_columns_, d_point_counts_,
      d_point_coef_indices_, d_point_weights_, d_x, d_y, d_z, d_j_csr_values_,
      d_cj_csr_values_);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

bool MixedConstraintSystem::HasNonlinearDotConstraints() const {
  for (const MixedConstraintScalar& scalar : scalar_constraints_) {
    if (scalar.type == kMixedConstraintDP1 ||
        scalar.type == kMixedConstraintDP2) {
      return true;
    }
  }
  return false;
}

MixedConstraintPointBinding MixedConstraintSystem::LocateReferencePointT10(
    const BlockCache& cache, const Eigen::Vector3d& reference_point) const {
  MixedConstraintPointBinding best_point;
  double best_residual = std::numeric_limits<double>::infinity();
  const double aabb_tol = 1e-8;

  for (int elem_idx = 0; elem_idx < cache.connectivity.rows(); ++elem_idx) {
    Eigen::Matrix<double, 10, 3> x_elem;
    for (int local_node = 0; local_node < kT10NodeCount; ++local_node) {
      const int global_node = cache.connectivity(elem_idx, local_node);
      x_elem(local_node, 0) = cache.x_ref(global_node);
      x_elem(local_node, 1) = cache.y_ref(global_node);
      x_elem(local_node, 2) = cache.z_ref(global_node);
    }

    if (!AabbContainsPoint(x_elem, reference_point, aabb_tol)) {
      continue;
    }

    Eigen::Vector3d coord = ComputeLinearTetInitialGuess(x_elem, reference_point);
    for (int iter = 0; iter < 20; ++iter) {
      const auto shape =
          EvaluateT10ShapeFunctions(coord[0], coord[1], coord[2]);
      const auto deriv =
          EvaluateT10ShapeDerivatives(coord[0], coord[1], coord[2]);
      Eigen::Vector3d mapped = Eigen::Vector3d::Zero();
      Eigen::Matrix3d jacobian = Eigen::Matrix3d::Zero();
      for (int a = 0; a < kT10NodeCount; ++a) {
        mapped += shape(a) * x_elem.row(a).transpose();
        jacobian += x_elem.row(a).transpose() * deriv.row(a);
      }
      const Eigen::Vector3d residual = mapped - reference_point;
      const double residual_norm = residual.norm();
      if (residual_norm < best_residual) {
        best_residual = residual_norm;
        best_point.count = kT10NodeCount;
        for (int a = 0; a < kT10NodeCount; ++a) {
          best_point.coef_indices[a] = cache.coef_offset + cache.connectivity(elem_idx, a);
          best_point.weights[a] = shape(a);
        }
      }
      if (residual_norm < 1e-10 && IsReferenceCoordinateInsideTet(coord)) {
        return best_point;
      }

      const Eigen::Vector3d delta =
          jacobian.colPivHouseholderQr().solve(residual);
      coord -= delta;
      if (!delta.allFinite()) {
        break;
      }
    }
  }

  if (best_residual > 1e-7) {
    throw std::runtime_error(
        "MixedConstraintSystem::LocateReferencePointT10: failed to locate point in FEAT10 block");
  }
  return best_point;
}

MixedConstraintPointBinding MixedConstraintSystem::LocateReferencePointANCF3243(
    const BlockCache& cache, const Eigen::Vector3d& reference_point) const {
  MixedConstraintPointBinding best_point;
  double best_residual = std::numeric_limits<double>::infinity();

  for (int elem_idx = 0; elem_idx < cache.connectivity.rows(); ++elem_idx) {
    Eigen::Matrix<double, 8, 3> coeffs;
    for (int local_coef = 0; local_coef < 8; ++local_coef) {
      const int node_local = (local_coef < 4) ? 0 : 1;
      const int dof_local = local_coef % 4;
      const int node_global = cache.connectivity(elem_idx, node_local);
      const int coef_idx = 4 * node_global + dof_local;
      coeffs(local_coef, 0) = cache.x_ref(coef_idx);
      coeffs(local_coef, 1) = cache.y_ref(coef_idx);
      coeffs(local_coef, 2) = cache.z_ref(coef_idx);
    }

    const double L = cache.beam_length(elem_idx);
    const double W = cache.beam_width(elem_idx);
    const double H = cache.beam_height(elem_idx);
    const auto& B_inv = cache.beam_B_inv[static_cast<size_t>(elem_idx)];

    Eigen::Vector3d coord = Eigen::Vector3d::Zero();
    const Eigen::Vector3d p0(coeffs(0, 0), coeffs(0, 1), coeffs(0, 2));
    const Eigen::Vector3d p1(coeffs(4, 0), coeffs(4, 1), coeffs(4, 2));
    const Eigen::Vector3d axis = p1 - p0;
    const double axis_len_sq = axis.squaredNorm();
    if (axis_len_sq > 1e-16 && L > 1e-16) {
      const double u = (reference_point - p0).dot(axis.normalized());
      coord[0] = std::max(-1.0, std::min(1.0, 2.0 * u / L - 1.0));
    }

    for (int iter = 0; iter < 24; ++iter) {
      const auto shape =
          EvaluateANCF3243Shape(coord[0], coord[1], coord[2], L, W, H, B_inv);
      const auto deriv = EvaluateANCF3243ShapeDerivatives(coord[0], coord[1],
                                                          coord[2], L, W, H,
                                                          B_inv);
      const Eigen::Vector3d mapped = EvaluateANCF3243Position(coeffs, shape);
      const Eigen::Matrix3d jacobian =
          EvaluateANCF3243Jacobian(coeffs, deriv);
      const Eigen::Vector3d residual = mapped - reference_point;
      const double residual_norm = residual.norm();
      if (residual_norm < best_residual) {
        best_residual = residual_norm;
        best_point.count = kBeamCoefCount;
        for (int i = 0; i < kBeamCoefCount; ++i) {
          const int node_local = (i < 4) ? 0 : 1;
          const int dof_local = i % 4;
          const int node_global = cache.connectivity(elem_idx, node_local);
          best_point.coef_indices[i] =
              cache.coef_offset + 4 * node_global + dof_local;
          best_point.weights[i] = shape(i);
        }
      }
      if (residual_norm < 1e-10 && IsReferenceCoordinateInsideBeam(coord)) {
        return best_point;
      }

      const Eigen::Vector3d delta =
          jacobian.colPivHouseholderQr().solve(residual);
      coord -= delta;
      if (!delta.allFinite()) {
        break;
      }
    }
  }

  if (best_residual > 1e-6) {
    throw std::runtime_error(
        "MixedConstraintSystem::LocateReferencePointANCF3243: failed to locate point in ANCF3243 block");
  }
  return best_point;
}
