// FEAT10Opt internal force against the standard FEAT10 element.
//
// Kernels under test (FEAT10DataOpt.cu / FEAT10KernelOpt.cuh):
//   computeInverseJacobian_kernel  - per-QP inverse Jacobian, run by
//                                    ComputePrecomputation()
//   internalF_MooneyRivlin_4QP     - fused H, P and nodal force, run by
//                                    ComputeInternalForce(); undamped path
//                                    (no velocities are passed)
// The reference is the standard element (FEAT10Data.cu: dn_du_pre_kernel,
// calc_p_kernel, compute_internal_force_kernel) in double precision with the
// same Mooney-Rivlin material. All deformations here are affine, so both
// quadrature rules integrate the element exactly and the only difference left
// is the Opt kernel's float arithmetic.
//
// Tolerances are derived from that arithmetic (see kRelTol, kFAbsTol below),
// not from the measured values. Each check prints what it measured next to
// its tolerance so drift is visible in the test log.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "lib_src/elements/FEAT10Data.cuh"
#include "lib_src/elements/FEAT10DataOpt.cuh"
#include "lib_utils/cpu_utils.h"
#include "lib_utils/quadrature_utils.h"

namespace {

const double kMu10 = GPU_FEAT10Opt_Data::kMu10;
const double kMu01 = GPU_FEAT10Opt_Data::kMu01;
const double kBulkK = GPU_FEAT10Opt_Data::kBulkK;

// Relative tolerance on force and stress, derived as follows. The kernel
// builds J - 1 from H with ~10 float roundings (10 node terms plus the trace),
// each up to eps = 1.2e-7 of |H|, so the bulk term carries a spurious
// pressure of up to kappa * 10 eps * |H|. The stress it is measured against
// is at least the deviatoric 2 mu |H| with mu = mu10 + mu01; the worst case
// is isochoric shear, where the bulk stress should be exactly zero. Bound:
//   10 eps * kappa / (2 mu) = 10 * 1.2e-7 * 7.5e8 / 1.07e8 = 8e-6,
// rounded up to 1e-5. Stretch-dominated cases sit kappa/mu ~ 7x below it.
const double kRelTol = 1e-5;
// Absolute tolerance on F: H alone, 10 roundings of eps on entries < 0.2.
const double kFAbsTol = 1e-6;

// Unit tetrahedron with mid-edge nodes in T10 order.
Eigen::MatrixXd UnitT10Nodes() {
  Eigen::MatrixXd X(10, 3);
  X << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, .5, 0, 0, .5, .5, 0, 0, .5, 0, 0,
      0, .5, .5, 0, .5, 0, .5, .5;
  return X;
}

Eigen::MatrixXi OneElement() {
  Eigen::MatrixXi E(1, 10);
  E << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  return E;
}

void ReadBeamRes0(Eigen::MatrixXd& X, Eigen::MatrixXi& E) {
  ANCFCPUUtils::FEAT10_read_nodes(
      "data/meshes/T10/resolution/beam_3x2x1_res0.1.node", X);
  ANCFCPUUtils::FEAT10_read_elements(
      "data/meshes/T10/resolution/beam_3x2x1_res0.1.ele", E);
}

// Standard element force for current positions x; P at QP 0 of element 0.
Eigen::VectorXd StandardForce(const Eigen::MatrixXd& X,
                              const Eigen::MatrixXi& E,
                              const Eigen::MatrixXd& x,
                              Eigen::Matrix3d* P0 = nullptr) {
  GPU_FEAT10_Data element(E.rows(), X.rows());
  element.Initialize();
  element.Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y,
                Quadrature::tet5pt_z, Quadrature::tet5pt_weights, X.col(0),
                X.col(1), X.col(2), E);
  element.SetMooneyRivlin(kMu10, kMu01, kBulkK);
  element.CalcDnDuPre();
  element.UpdatePositions(x.col(0), x.col(1), x.col(2));
  element.CalcP();
  element.CalcInternalForce();
  Eigen::VectorXd f;
  element.RetrieveInternalForceToCPU(f);
  if (P0) {
    std::vector<std::vector<Eigen::MatrixXd>> P;
    element.RetrievePFromFToCPU(P);
    *P0 = P[0][0];
  }
  element.Destroy();
  return f;
}

// Opt element force for current positions x; F and P at QP 0 of element 0.
Eigen::VectorXd OptForce(const Eigen::MatrixXd& X, const Eigen::MatrixXi& E,
                         const Eigen::MatrixXd& x,
                         Eigen::Matrix3d* F0 = nullptr,
                         Eigen::Matrix3d* P0 = nullptr) {
  GPU_FEAT10Opt_Data element;
  element.Initialize(E.rows(), X.rows());
  element.Setup(X, E);
  element.SetMooneyRivlin(kMu10, kMu01, kBulkK);
  element.ComputePrecomputation();
  element.UpdatePositions(x);
  element.ClearInternalForce();
  element.ComputeInternalForce(nullptr, F0 != nullptr, P0 != nullptr);
  Eigen::VectorXf f;
  element.RetrieveInternalForceToCPU(f);
  if (F0) {
    std::vector<std::vector<Eigen::Matrix3f>> F;
    element.RetrieveDeformationGradientToCPU(F);
    *F0 = F[0][0].cast<double>();
  }
  if (P0) {
    std::vector<std::vector<Eigen::Matrix3f>> P;
    element.RetrievePiolaToCPU(P);
    *P0 = P[0][0].cast<double>();
  }
  element.Destroy();
  return f.cast<double>();
}

double MaxAbs(const Eigen::MatrixXd& m) { return m.cwiseAbs().maxCoeff(); }

Eigen::MatrixXd Deform(const Eigen::MatrixXd& X, const Eigen::Matrix3d& F) {
  return X * F.transpose();
}

Eigen::Matrix3d GeneralF() {
  Eigen::Matrix3d F;
  F << 1.10, 0.05, 0.02, 0.03, 1.15, 0.04, 0.01, 0.02, 0.95;
  return F;
}

void Report(const char* what, double measured, double tol) {
  std::printf("  %-28s measured %.2e  tolerance %.0e\n", what, measured, tol);
}

}  // namespace

TEST(FEAT10OptForce, Rest_ZeroForce) {
  Eigen::MatrixXd X;
  Eigen::MatrixXi E;
  ReadBeamRes0(X, E);
  double f_tet = MaxAbs(OptForce(UnitT10Nodes(), OneElement(), UnitT10Nodes()));
  double f_beam = MaxAbs(OptForce(X, E, X));
  Report("rest, tet max|f| (N)", f_tet, 0.0);
  Report("rest, beam max|f| (N)", f_beam, 0.0);
  EXPECT_EQ(f_tet, 0.0);
  EXPECT_EQ(f_beam, 0.0);
}

TEST(FEAT10OptForce, AffineDeformations_MatchStandard) {
  Eigen::MatrixXd X;
  Eigen::MatrixXi E;
  ReadBeamRes0(X, E);
  struct Case {
    const char* name;
    Eigen::Matrix3d F;
  };
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d stretch = I, compress = I, shear = I;
  stretch(0, 0) = 1.1;
  compress(0, 0) = 0.9;
  shear(0, 1) = 0.2;
  for (const Case& c : {Case{"stretch 1.1", stretch}, Case{"compress 0.9", compress},
                        Case{"shear 0.2", shear}, Case{"general", GeneralF()}}) {
    Eigen::MatrixXd x = Deform(X, c.F);
    Eigen::VectorXd f_std = StandardForce(X, E, x);
    Eigen::VectorXd f_opt = OptForce(X, E, x);
    double rel = MaxAbs(f_opt - f_std) / MaxAbs(f_std);
    Report(c.name, rel, kRelTol);
    EXPECT_LE(rel, kRelTol) << c.name;
  }
}

TEST(FEAT10OptForce, RigidMotion) {
  Eigen::MatrixXd X;
  Eigen::MatrixXi E;
  ReadBeamRes0(X, E);

  // Translation: the relative displacements are exactly zero, so is the force.
  Eigen::MatrixXd x = X;
  x.col(0).array() += 1000.0;
  double f_translated = MaxAbs(OptForce(X, E, x));
  Report("translated 1000 m, max|f|", f_translated, 0.0);
  EXPECT_EQ(f_translated, 0.0);

  // Rotation: no stress, so the force is float noise on the stiffness scale,
  // taken here as the force of a 10% stretch.
  Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 6, Eigen::Vector3d::UnitZ())
                          .toRotationMatrix();
  Eigen::Matrix3d stretch = Eigen::Matrix3d::Identity();
  stretch(0, 0) = 1.1;
  double f_scale = MaxAbs(OptForce(X, E, Deform(X, stretch)));
  double rel = MaxAbs(OptForce(X, E, Deform(X, R))) / f_scale;
  Report("rotated 30 deg, rel. noise", rel, kRelTol);
  EXPECT_LE(rel, kRelTol);
}

TEST(FEAT10OptForce, F_and_P_MatchStandard) {
  Eigen::MatrixXd X = UnitT10Nodes();
  Eigen::MatrixXi E = OneElement();
  Eigen::Matrix3d F = GeneralF();
  Eigen::MatrixXd x = Deform(X, F);

  Eigen::Matrix3d P_std;
  StandardForce(X, E, x, &P_std);
  Eigen::Matrix3d F_opt, P_opt;
  OptForce(X, E, x, &F_opt, &P_opt);

  double f_err = MaxAbs(F_opt - F);
  double p_rel = MaxAbs(P_opt - P_std) / MaxAbs(P_std);
  Report("F abs error", f_err, kFAbsTol);
  Report("P rel error", p_rel, kRelTol);
  EXPECT_LE(f_err, kFAbsTol);
  EXPECT_LE(p_rel, kRelTol);
}

TEST(FEAT10OptForce, SetMooneyRivlin_AcceptsCompiledMaterial) {
  // The kernel folds the material in at compile time; the setter must accept
  // exactly those values, or every comparison above runs on two materials.
  GPU_FEAT10Opt_Data element;
  element.Initialize(1, 10);
  element.Setup(UnitT10Nodes(), OneElement());
  EXPECT_NO_THROW(element.SetMooneyRivlin(GPU_FEAT10Opt_Data::kMu10,
                                          GPU_FEAT10Opt_Data::kMu01,
                                          GPU_FEAT10Opt_Data::kBulkK));
  element.Destroy();
}

TEST(FEAT10OptForce, SetMooneyRivlin_RejectsOtherMaterial) {
  GPU_FEAT10Opt_Data element;
  element.Initialize(1, 10);
  element.Setup(UnitT10Nodes(), OneElement());
  EXPECT_THROW(element.SetMooneyRivlin(0.5f * GPU_FEAT10Opt_Data::kMu10,
                                       GPU_FEAT10Opt_Data::kMu01,
                                       GPU_FEAT10Opt_Data::kBulkK),
               std::invalid_argument);
  element.Destroy();
}
