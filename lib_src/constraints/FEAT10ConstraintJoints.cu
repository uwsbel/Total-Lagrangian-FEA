/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * File:    FEAT10ConstraintJoints.cu
 * Brief:   FEAT10 engineering-joint builders built from primitive constraints.
 *==============================================================
 *==============================================================*/

#include "FEAT10ConstraintManager.h"

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

void FEAT10ConstraintManager::AddFixedJoint(
    const ReferencePoint& p, const ReferencePoint& q, const ReferencePoint& w,
    const ReferencePoint& r, const ReferencePoint& s, const ReferencePoint& t,
    double f1, double f2, double f3, double dp1_weight) {
  AddPointToPointCDAxis(p, r, 0);
  AddPointToPointCDAxis(p, r, 1);
  AddPointToPointCDAxis(p, r, 2);

  // Appendix A.8 uses the three orientation-lock rows
  //   a1 dot d1 = f1,
  //   a1 dot d2 = f2,
  //   a2 dot d2 = f3,
  // where a1 = Q - P, a2 = W - P, d1 = S - R, and d2 = T - R. The last row is
  // intentionally paired with d2 rather than d1 so the reference construction
  // operates at perpendicularity instead of a cosine extremum.
  AddDP1Constraint(p, q, r, s, f1, dp1_weight);
  AddDP1Constraint(p, q, r, t, f2, dp1_weight);
  AddDP1Constraint(p, w, r, t, f3, dp1_weight);
}

void FEAT10ConstraintManager::AddCylindricalJoint(
    const ReferencePoint& p, const ReferencePoint& q, const ReferencePoint& r,
    const ReferencePoint& s, const ReferencePoint& u,
    const ReferencePoint& v, const ReferencePoint& w, double f_par1,
    double f_par2, double f_col1, double f_col2, double dp1_weight,
    double dp2_weight) {
  AddDP1Constraint(p, q, r, s, f_par1, dp1_weight);
  AddDP1Constraint(p, q, r, u, f_par2, dp1_weight);
  AddDP2Constraint(p, v, p, r, f_col1, dp2_weight);
  AddDP2Constraint(p, w, p, r, f_col2, dp2_weight);
}

