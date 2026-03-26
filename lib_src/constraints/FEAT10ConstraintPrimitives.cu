/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * File:    FEAT10ConstraintPrimitives.cu
 * Brief:   FEAT10 primitive constraint builders (CD, DP1, DP2, World-DP1).
 *==============================================================
 *==============================================================*/

#include "FEAT10ConstraintManager.h"

#include <cmath>
#include <stdexcept>

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


// FEAT10's bilinear dot-product constraints use the same scalar equation for
// both DP1 and DP2. They differ only in the intended body ownership pattern
// and, later on, in the exact Hessian assembly for repeated-point cases.
void FEAT10ConstraintManager::AddDotProductConstraint(
    int type, const ReferencePoint& p, const ReferencePoint& q,
    const ReferencePoint& r, const ReferencePoint& s, double target,
    double weight) {
  if (finalized_) {
    throw std::logic_error(
        "FEAT10ConstraintManager::AddDotProductConstraint: manager already finalized");
  }
  if (weight <= 0.0) {
    throw std::invalid_argument(
        "FEAT10ConstraintManager::AddDotProductConstraint: weight must be positive");
  }
  if (type != kFEAT10ConstraintDP1 && type != kFEAT10ConstraintDP2) {
    throw std::invalid_argument(
        "FEAT10ConstraintManager::AddDotProductConstraint: invalid dot-product constraint type");
  }

  const Eigen::Vector3d p_ref = EvaluateReferencePoint(p);
  const Eigen::Vector3d q_ref = EvaluateReferencePoint(q);
  const Eigen::Vector3d r_ref = EvaluateReferencePoint(r);
  const Eigen::Vector3d s_ref = EvaluateReferencePoint(s);
  const double a_norm         = (q_ref - p_ref).norm();
  const double d_norm         = (s_ref - r_ref).norm();

  ScalarConstraint constraint;
  constraint.type      = type;
  constraint.target    = target;
  constraint.row_scale = weight;
  const double row_norm = std::sqrt(a_norm * a_norm + d_norm * d_norm);
  if (row_norm > 1e-12) {
    constraint.row_scale *= 1.0 / row_norm;
  }
  constraint.points[0] = p;
  constraint.points[1] = q;
  constraint.points[2] = r;
  constraint.points[3] = s;
  scalar_constraints_.push_back(constraint);
}

void FEAT10ConstraintManager::AddDP1Constraint(const ReferencePoint& p,
                                               const ReferencePoint& q,
                                               const ReferencePoint& r,
                                               const ReferencePoint& s,
                                               double target,
                                               double weight) {
  AddDotProductConstraint(kFEAT10ConstraintDP1, p, q, r, s, target, weight);
}

void FEAT10ConstraintManager::AddDP2Constraint(const ReferencePoint& p,
                                               const ReferencePoint& q,
                                               const ReferencePoint& r,
                                               const ReferencePoint& s,
                                               double target,
                                               double weight) {
  AddDotProductConstraint(kFEAT10ConstraintDP2, p, q, r, s, target, weight);
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

  ScalarConstraint constraint;
  constraint.type            = kFEAT10ConstraintWorldDP1;
  constraint.target          = target;
  constraint.world_direction = world_direction;
  constraint.row_scale       = weight;
  if (d_norm > 1e-12) {
    constraint.row_scale *= 1.0 / d_norm;
  }
  constraint.points[0] = p;
  constraint.points[1] = q;
  scalar_constraints_.push_back(constraint);
}

