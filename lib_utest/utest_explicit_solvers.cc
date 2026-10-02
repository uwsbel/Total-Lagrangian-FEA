// Explicit solvers: SyncedExplicitSolver (standard FEAT10 element) and
// SyncedExplicitOptSolver (FEAT10Opt element), one symplectic-Euler step per
// Solve(): v += dt * (f_ext - f_int) / m, fixed nodes zeroed, x += dt * v.
//
// Kernels under test: explicit_compute_p, explicit_clear_internal_force,
// explicit_compute_internal_force, explicit_velocity_update,
// explicit_apply_fixed_node_bc, explicit_position_update (SyncedExplicit.cu);
// syncedExplicitOpt_velocityUpdate, syncedExplicitOpt_applyFixedBC,
// syncedExplicitOpt_positionUpdate plus the FEAT10Opt force kernel
// (SyncedExplicitOpt.cu).
//
// The integrator has closed-form answers for a body at rest and for a rigid
// translation under f_i = m_i a, so the checks are exact or bounded by
// round-off; the derivation of each bound is next to it.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <vector>

#include "lib_src/elements/FEAT10Data.cuh"
#include "lib_src/elements/FEAT10DataOpt.cuh"
#include "lib_src/solvers/SyncedExplicit.cuh"
#include "lib_src/solvers/SyncedExplicitOpt.cuh"
#include "lib_utils/cpu_utils.h"
#include "lib_utils/quadrature_utils.h"

namespace {

const double kMu10 = GPU_FEAT10Opt_Data::kMu10;
const double kMu01 = GPU_FEAT10Opt_Data::kMu01;
const double kBulkK = GPU_FEAT10Opt_Data::kBulkK;
const double kRho = 2700.0;

const double kEpsDouble = 2.2e-16;
const double kEpsFloat = 1.2e-7;

// The standard element's force at rest is double round-off: ~30 roundings of
// eps on the bulk term kappa, times an element area of at most 1 m^2 on the
// res0 beam (3 x 2 x 1 m, elements under 1 m). Measured 5e-7 N.
const double kRestForceBoundStd = 30 * kEpsDouble * kBulkK * 1.0;

void ReadBeamRes0(Eigen::MatrixXd& X, Eigen::MatrixXi& E) {
  ANCFCPUUtils::FEAT10_read_nodes(
      "data/meshes/T10/resolution/beam_3x2x1_res0.1.node", X);
  ANCFCPUUtils::FEAT10_read_elements(
      "data/meshes/T10/resolution/beam_3x2x1_res0.1.ele", E);
}

std::vector<int> NodesOnFaceX0(const Eigen::MatrixXd& X) {
  std::vector<int> nodes;
  for (int i = 0; i < X.rows(); i++)
    if (std::abs(X(i, 0)) < 1e-8) nodes.push_back(i);
  return nodes;
}

Eigen::MatrixXd Stack(const Eigen::VectorXd& x, const Eigen::VectorXd& y,
                      const Eigen::VectorXd& z) {
  Eigen::MatrixXd p(x.size(), 3);
  p << x, y, z;
  return p;
}

// Standard path: positions after n_steps; f_ext_per_mass is the external
// force per unit nodal mass (a uniform acceleration), applied to every node.
// Returns the smallest lumped mass through m_min.
Eigen::MatrixXd RunStandard(const Eigen::MatrixXd& X, const Eigen::MatrixXi& E,
                            double accel, const std::vector<int>& fixed,
                            double dt, int n_steps, double* m_min = nullptr) {
  GPU_FEAT10_Data element(E.rows(), X.rows());
  element.Initialize();
  element.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                Quadrature::tet5pt_z, Quadrature::tet5pt_weights, X.col(0),
                X.col(1), X.col(2), E);
  element.SetMooneyRivlin(kMu10, kMu01, kBulkK);
  element.SetDensity(kRho);
  element.SetDamping(0.0, 0.0);
  element.CalcDnDuPre();
  element.CalcLumpedMassHRZ();

  Eigen::VectorXd mass;
  element.RetrieveLumpedMassToCPU(mass);
  if (m_min) *m_min = mass.minCoeff();
  Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(3 * X.rows());
  for (int i = 0; i < X.rows(); i++) f_ext(3 * i + 2) = mass(i) * accel;
  element.SetExternalForce(f_ext);

  SyncedExplicitSolver solver(&element);
  SyncedExplicitParams params = {dt};
  solver.SetParameters(&params);
  solver.SetFixedNodes(fixed);
  for (int s = 0; s < n_steps; s++) solver.Solve();

  Eigen::VectorXd x, y, z;
  element.RetrievePositionToCPU(x, y, z);
  element.Destroy();
  return Stack(x, y, z);
}

// Opt path, same contract. The external force is a/inv_m in float, which is
// what the solver multiplies by inv_m again.
Eigen::MatrixXd RunOpt(const Eigen::MatrixXd& X, const Eigen::MatrixXi& E,
                       double accel, const std::vector<int>& fixed, double dt,
                       int n_steps) {
  GPU_FEAT10Opt_Data element;
  element.Initialize(E.rows(), X.rows());
  element.Setup(X, E);
  element.SetMooneyRivlin(kMu10, kMu01, kBulkK);
  element.SetDensity(kRho);
  element.ComputePrecomputation();
  element.ComputeLumpedMassHRZ();

  Eigen::VectorXf inv_mass;
  element.RetrieveInvLumpedMassToCPU(inv_mass);
  Eigen::VectorXf f_ext = Eigen::VectorXf::Zero(3 * X.rows());
  for (int i = 0; i < X.rows(); i++)
    f_ext(3 * i + 2) = static_cast<float>(accel) / inv_mass(i);

  SyncedExplicitOptSolver solver(&element);
  solver.SetTimeStep(dt);
  solver.SetFixedNodes(fixed);
  solver.SetExternalForce(f_ext);
  for (int s = 0; s < n_steps; s++) solver.Solve();

  Eigen::VectorXd x, y, z;
  element.RetrievePositionToCPU(x, y, z);
  element.Destroy();
  return Stack(x, y, z);
}

double MaxAbs(const Eigen::MatrixXd& m) { return m.cwiseAbs().maxCoeff(); }

void Report(const char* what, double measured, double tol) {
  std::printf("  %-34s measured %.2e  tolerance %.0e\n", what, measured, tol);
}

}  // namespace

// No loads: the body must not move. Opt's rest force is exactly zero, so the
// positions are bitwise unchanged. The standard element's rest force is
// round-off, so the displacement is bounded by 1/2 (f/m_min) T^2 plus the
// round-off of the position updates themselves, N eps |X|.
TEST(ExplicitSolvers, RestStaysAtRest) {
  Eigen::MatrixXd X;
  Eigen::MatrixXi E;
  ReadBeamRes0(X, E);
  const double dt = 1e-5;
  const int n = 100;
  const double T = n * dt;

  double m_min = 0;
  double moved_std = MaxAbs(RunStandard(X, E, 0.0, {}, dt, n, &m_min) - X);
  double tol_std = 0.5 * kRestForceBoundStd / m_min * T * T +
                   n * kEpsDouble * MaxAbs(X);
  Report("standard, max displacement (m)", moved_std, tol_std);
  EXPECT_LE(moved_std, tol_std);

  double moved_opt = MaxAbs(RunOpt(X, E, 0.0, {}, dt, n) - X);
  Report("opt, max displacement (m)", moved_opt, 0.0);
  EXPECT_EQ(moved_opt, 0.0);
}

// f_i = m_i a on every node, nothing fixed: rigid translation with no internal
// force, and symplectic Euler has the exact discrete solution
//   x_N = X + a dt^2 N (N + 1) / 2.
// Standard: the acceleration is exact in double; the only error is the rest
// force noise through 1/2 (f/m_min) T^2 and the position round-off N eps |X|.
// Opt: the velocity update multiplies a float inverse mass by a float force,
// so each node's acceleration carries up to 2 eps_float; internal forces only
// hold the nodes together and cannot move the centre of mass. The positions
// themselves are double, N eps |X|. dt must be below the explicit stability
// limit of this material (the demos use 1e-5 at res0).
TEST(ExplicitSolvers, FreeBody_ConstantForce) {
  Eigen::MatrixXd X;
  Eigen::MatrixXi E;
  ReadBeamRes0(X, E);
  const double a = 1.0, dt = 1e-5;
  const int n = 1000;
  const double T = n * dt;
  const double u = a * dt * dt * n * (n + 1) / 2.0;  // exact displacement
  Eigen::MatrixXd x_exact = X;
  x_exact.col(2).array() += u;

  double m_min = 0;
  double err_std = MaxAbs(RunStandard(X, E, a, {}, dt, n, &m_min) - x_exact);
  double tol_std = 0.5 * kRestForceBoundStd / m_min * T * T +
                   n * kEpsDouble * MaxAbs(X);
  Report("standard, |x - closed form| (m)", err_std, tol_std);
  EXPECT_LE(err_std, tol_std);

  double err_opt = MaxAbs(RunOpt(X, E, a, {}, dt, n) - x_exact);
  double tol_opt = 2 * kEpsFloat * u + n * kEpsDouble * MaxAbs(X);
  Report("opt, |x - closed form| (m)", err_opt, tol_opt);
  EXPECT_LE(err_opt, tol_opt);
}

// Same load with the x = 0 face fixed: fixed nodes have their velocity zeroed
// before every position update, so they stay exactly where they are; the rest
// of the body moves.
TEST(ExplicitSolvers, FixedNodes_StayFixed) {
  Eigen::MatrixXd X;
  Eigen::MatrixXi E;
  ReadBeamRes0(X, E);
  const std::vector<int> fixed = NodesOnFaceX0(X);
  ASSERT_GT(fixed.size(), 0u);
  const double a = 1.0, dt = 1e-5;
  const int n = 100;

  for (int path = 0; path < 2; path++) {
    Eigen::MatrixXd x = path == 0 ? RunStandard(X, E, a, fixed, dt, n)
                                  : RunOpt(X, E, a, fixed, dt, n);
    double moved_fixed = 0.0;
    for (int i : fixed) moved_fixed = std::max(moved_fixed, MaxAbs(x.row(i) - X.row(i)));
    double moved_any = MaxAbs(x - X);
    Report(path == 0 ? "standard, fixed nodes moved (m)" : "opt, fixed nodes moved (m)",
           moved_fixed, 0.0);
    EXPECT_EQ(moved_fixed, 0.0);
    EXPECT_GT(moved_any, 0.0);
  }
}
