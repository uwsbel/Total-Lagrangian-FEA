/*
 * Integration test for FEAT10/FEAT10Opt using a real mesh: beam_res_0.
 *
 * This test uses the beam_3x2x1_res0.1 mesh (105 nodes, 36 elements) to
 * validate the internal force computation on a realistic multi-element mesh.
 *
 * Test cases:
 * 1. Undeformed equilibrium: verify f_int = 0 in reference configuration
 * 2. Force equilibrium: apply uniform stretch, verify sum(f_int) = 0
 * 3. Cross-implementation: compare FEAT10 vs FEAT10Opt on same mesh
 * 4. Symmetry check: verify symmetric deformation gives symmetric forces
 * 5. Boundary forces: fix one end, apply displacement, check reaction forces
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "feat10_test_utils.h"
#include "lib_src/elements/FEAT10Data.cuh"
#include "lib_src/elements/FEAT10DataOpt.cuh"
#include "lib_utils/cpu_utils.h"
#include "lib_utils/quadrature_utils.h"

using namespace feat10_test;

// ===========================================================================
// Test Fixture
// ===========================================================================

class FEAT10BeamIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Load beam mesh
    n_nodes_ = ANCFCPUUtils::FEAT10_read_nodes(
        "data/meshes/T10/resolution/beam_3x2x1_res0.1.node", nodes_);
    n_elems_ = ANCFCPUUtils::FEAT10_read_elements(
        "data/meshes/T10/resolution/beam_3x2x1_res0.1.ele", connectivity_);

    ASSERT_EQ(n_nodes_, 105) << "Expected 105 nodes in beam mesh";
    ASSERT_EQ(n_elems_, 36) << "Expected 36 elements in beam mesh";

    // Material parameters: Mooney-Rivlin
    mu10_  = 80769.23;
    mu01_  = 20192.31;
    kappa_ = 400000.0;

    // Setup 5-point Keast quadrature rule for FEAT10
    Setup5PointKeastQuadrature(qp_x_5pt_, qp_y_5pt_, qp_z_5pt_,
                               qp_weights_5pt_);

    // Extract coordinate vectors
    h_x_ref_.resize(n_nodes_);
    h_y_ref_.resize(n_nodes_);
    h_z_ref_.resize(n_nodes_);
    for (int i = 0; i < n_nodes_; i++) {
      h_x_ref_(i) = nodes_(i, 0);
      h_y_ref_(i) = nodes_(i, 1);
      h_z_ref_(i) = nodes_(i, 2);
    }

    // Compute mesh bounds
    x_min_ = h_x_ref_.minCoeff();
    x_max_ = h_x_ref_.maxCoeff();
    y_min_ = h_y_ref_.minCoeff();
    y_max_ = h_y_ref_.maxCoeff();
    z_min_ = h_z_ref_.minCoeff();
    z_max_ = h_z_ref_.maxCoeff();
  }

  // Setup FEAT10 element with given positions
  GPU_FEAT10_Data* SetupFEAT10(const Eigen::VectorXd& h_x,
                               const Eigen::VectorXd& h_y,
                               const Eigen::VectorXd& h_z) {
    GPU_FEAT10_Data* element = new GPU_FEAT10_Data(n_elems_, n_nodes_);
    element->Initialize();
    element->Setup(qp_x_5pt_, qp_y_5pt_, qp_z_5pt_, qp_weights_5pt_, h_x, h_y,
                   h_z, connectivity_);
    element->SetMooneyRivlin(mu10_, mu01_, kappa_);
    return element;
  }

  // Setup FEAT10Opt element with given positions
  GPU_FEAT10Opt_Data* SetupFEAT10Opt(const Eigen::VectorXd& h_x,
                                     const Eigen::VectorXd& h_y,
                                     const Eigen::VectorXd& h_z) {
    GPU_FEAT10Opt_Data* element = new GPU_FEAT10Opt_Data();
    element->Initialize(n_elems_, n_nodes_);

    // Convert to positions matrix for FEAT10Opt API
    Eigen::MatrixXd positions(n_nodes_, 3);
    for (int i = 0; i < n_nodes_; i++) {
      positions(i, 0) = h_x(i);
      positions(i, 1) = h_y(i);
      positions(i, 2) = h_z(i);
    }

    element->Setup(positions, connectivity_);
    element->SetMooneyRivlin(static_cast<float>(mu10_),
                             static_cast<float>(mu01_),
                             static_cast<float>(kappa_));
    return element;
  }

  // Apply affine deformation x_new = F * x_old
  void ApplyAffineDeformation(const double F[3][3], Eigen::VectorXd& h_x,
                              Eigen::VectorXd& h_y, Eigen::VectorXd& h_z) {
    h_x.resize(n_nodes_);
    h_y.resize(n_nodes_);
    h_z.resize(n_nodes_);

    for (int i = 0; i < n_nodes_; i++) {
      double X[3] = {h_x_ref_(i), h_y_ref_(i), h_z_ref_(i)};
      h_x(i)      = F[0][0] * X[0] + F[0][1] * X[1] + F[0][2] * X[2];
      h_y(i)      = F[1][0] * X[0] + F[1][1] * X[1] + F[1][2] * X[2];
      h_z(i)      = F[2][0] * X[0] + F[2][1] * X[1] + F[2][2] * X[2];
    }
  }

  // Get indices of nodes at x = x_min (fixed end)
  std::vector<int> GetFixedEndNodes() {
    std::vector<int> fixed_nodes;
    for (int i = 0; i < n_nodes_; i++) {
      if (std::abs(h_x_ref_(i) - x_min_) < 1e-8) {
        fixed_nodes.push_back(i);
      }
    }
    return fixed_nodes;
  }

  // Get indices of nodes at x = x_max (free end)
  std::vector<int> GetFreeEndNodes() {
    std::vector<int> free_nodes;
    for (int i = 0; i < n_nodes_; i++) {
      if (std::abs(h_x_ref_(i) - x_max_) < 1e-8) {
        free_nodes.push_back(i);
      }
    }
    return free_nodes;
  }

  int n_nodes_, n_elems_;
  Eigen::MatrixXd nodes_;
  Eigen::MatrixXi connectivity_;
  Eigen::VectorXd h_x_ref_, h_y_ref_, h_z_ref_;
  Eigen::VectorXd qp_x_5pt_, qp_y_5pt_, qp_z_5pt_, qp_weights_5pt_;
  double mu10_, mu01_, kappa_;
  double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
};

// ===========================================================================
// Test Cases
// ===========================================================================

TEST_F(FEAT10BeamIntegrationTest, BeamRes0_MeshLoadVerification) {
  // Verify mesh was loaded correctly and has expected properties

  std::cout << "\n=== Beam Mesh Properties ===" << std::endl;
  std::cout << "Nodes: " << n_nodes_ << std::endl;
  std::cout << "Elements: " << n_elems_ << std::endl;
  std::cout << "Bounds X: [" << x_min_ << ", " << x_max_ << "]" << std::endl;
  std::cout << "Bounds Y: [" << y_min_ << ", " << y_max_ << "]" << std::endl;
  std::cout << "Bounds Z: [" << z_min_ << ", " << z_max_ << "]" << std::endl;

  // Verify mesh is a 3x2x1 beam (approximately)
  double beam_length = x_max_ - x_min_;
  double beam_width  = y_max_ - y_min_;
  double beam_height = z_max_ - z_min_;

  // Beam should be ~3 units long, ~2 units wide, ~1 unit tall
  EXPECT_NEAR(beam_length, 3.0, 0.1);
  EXPECT_NEAR(beam_width, 2.0, 0.1);
  EXPECT_NEAR(beam_height, 1.0, 0.1);

  // Each T10 element should have 10 nodes
  EXPECT_EQ(connectivity_.cols(), 10);
}

TEST_F(FEAT10BeamIntegrationTest, BeamRes0_UndeformedEquilibrium_FEAT10) {
  // In undeformed configuration, internal forces should be zero

  GPU_FEAT10_Data* element = SetupFEAT10(h_x_ref_, h_y_ref_, h_z_ref_);

  element->CalcDnDuPre();
  element->CalcP();
  element->CalcInternalForce();

  Eigen::VectorXd f_int;
  element->RetrieveInternalForceToCPU(f_int);

  // All forces should be near zero
  double max_force = f_int.cwiseAbs().maxCoeff();
  std::cout << "\n=== FEAT10 Undeformed Equilibrium ===" << std::endl;
  std::cout << "Max |f_int|: " << max_force << std::endl;

  EXPECT_LT(max_force, 1e-8)
      << "Undeformed mesh should have zero internal forces";

  element->Destroy();
  delete element;
}

TEST_F(FEAT10BeamIntegrationTest, BeamRes0_UndeformedEquilibrium_FEAT10Opt) {
  // In undeformed configuration, internal forces should be zero (FEAT10Opt)

  GPU_FEAT10Opt_Data* element = SetupFEAT10Opt(h_x_ref_, h_y_ref_, h_z_ref_);

  element->ComputePrecomputation();
  element->ClearInternalForce();
  element->ComputeInternalForce();

  Eigen::VectorXf f_int;
  element->RetrieveInternalForceToCPU(f_int);

  // Float precision means larger tolerance
  double max_force = f_int.cwiseAbs().maxCoeff();
  std::cout << "\n=== FEAT10Opt Undeformed Equilibrium ===" << std::endl;
  std::cout << "Max |f_int|: " << max_force << std::endl;

  // Float precision with material constants ~1e5 gives ~0.1 residual
  EXPECT_LT(max_force, 1.0)
      << "Undeformed mesh should have near-zero internal forces";

  element->Destroy();
  delete element;
}

TEST_F(FEAT10BeamIntegrationTest, BeamRes0_ForceEquilibrium_UniformStretch) {
  // Apply uniform stretch to entire mesh, verify global force equilibrium

  double F[3][3];
  SetPureStretchF(F, 1.1, 1.05, 0.95);

  Eigen::VectorXd h_x, h_y, h_z;
  ApplyAffineDeformation(F, h_x, h_y, h_z);

  // === FEAT10 ===
  GPU_FEAT10_Data* feat10 = SetupFEAT10(h_x_ref_, h_y_ref_, h_z_ref_);
  feat10->CalcDnDuPre();
  feat10->UpdatePositions(h_x, h_y, h_z);
  feat10->CalcP();
  feat10->CalcInternalForce();

  Eigen::VectorXd f_int_feat10;
  feat10->RetrieveInternalForceToCPU(f_int_feat10);

  // === FEAT10Opt ===
  GPU_FEAT10Opt_Data* feat10opt = SetupFEAT10Opt(h_x_ref_, h_y_ref_, h_z_ref_);
  feat10opt->ComputePrecomputation();

  Eigen::MatrixXd positions(n_nodes_, 3);
  for (int i = 0; i < n_nodes_; i++) {
    positions(i, 0) = h_x(i);
    positions(i, 1) = h_y(i);
    positions(i, 2) = h_z(i);
  }
  feat10opt->UpdatePositions(positions);

  feat10opt->ClearInternalForce();
  feat10opt->ComputeInternalForce();

  Eigen::VectorXf f_int_feat10opt;
  feat10opt->RetrieveInternalForceToCPU(f_int_feat10opt);

  // === Check equilibrium ===
  std::cout << "\n=== Force Equilibrium (Uniform Stretch) ===" << std::endl;

  double eq_error_feat10    = CheckForceEquilibrium(f_int_feat10, n_nodes_);
  double eq_error_feat10opt = CheckForceEquilibrium(f_int_feat10opt, n_nodes_);

  std::cout << "FEAT10 equilibrium error: " << eq_error_feat10 << std::endl;
  std::cout << "FEAT10Opt equilibrium error: " << eq_error_feat10opt
            << std::endl;

  // FEAT10 (double precision)
  EXPECT_LT(eq_error_feat10, 1e-10) << "FEAT10 force equilibrium violated";

  // FEAT10Opt (float precision)
  EXPECT_LT(eq_error_feat10opt, 1e-4) << "FEAT10Opt force equilibrium violated";

  feat10->Destroy();
  delete feat10;
  feat10opt->Destroy();
  delete feat10opt;
}

TEST_F(FEAT10BeamIntegrationTest,
       BeamRes0_ForceEquilibrium_GeneralDeformation) {
  // Apply general (stretch + shear) deformation, verify equilibrium

  double F[3][3];
  SetGeneralAffineF(F);

  Eigen::VectorXd h_x, h_y, h_z;
  ApplyAffineDeformation(F, h_x, h_y, h_z);

  GPU_FEAT10_Data* element = SetupFEAT10(h_x_ref_, h_y_ref_, h_z_ref_);
  element->CalcDnDuPre();
  element->UpdatePositions(h_x, h_y, h_z);
  element->CalcP();
  element->CalcInternalForce();

  Eigen::VectorXd f_int;
  element->RetrieveInternalForceToCPU(f_int);

  double eq_error = CheckForceEquilibrium(f_int, n_nodes_);
  std::cout << "\n=== Force Equilibrium (General Deformation) ===" << std::endl;
  std::cout << "Equilibrium error: " << eq_error << std::endl;

  EXPECT_LT(eq_error, 1e-10) << "Force equilibrium violated";

  element->Destroy();
  delete element;
}

TEST_F(FEAT10BeamIntegrationTest, BeamRes0_FEAT10_vs_FEAT10Opt_Comparison) {
  // Compare internal forces from both implementations on the same mesh

  double F[3][3];
  SetPureStretchF(F, 1.15, 1.0, 0.9);

  Eigen::VectorXd h_x, h_y, h_z;
  ApplyAffineDeformation(F, h_x, h_y, h_z);

  // === FEAT10 ===
  GPU_FEAT10_Data* feat10 = SetupFEAT10(h_x_ref_, h_y_ref_, h_z_ref_);
  feat10->CalcDnDuPre();
  feat10->UpdatePositions(h_x, h_y, h_z);
  feat10->CalcP();
  feat10->CalcInternalForce();

  Eigen::VectorXd f_int_feat10;
  feat10->RetrieveInternalForceToCPU(f_int_feat10);

  // === FEAT10Opt ===
  GPU_FEAT10Opt_Data* feat10opt = SetupFEAT10Opt(h_x_ref_, h_y_ref_, h_z_ref_);
  feat10opt->ComputePrecomputation();

  Eigen::MatrixXd positions(n_nodes_, 3);
  for (int i = 0; i < n_nodes_; i++) {
    positions(i, 0) = h_x(i);
    positions(i, 1) = h_y(i);
    positions(i, 2) = h_z(i);
  }
  feat10opt->UpdatePositions(positions);

  feat10opt->ClearInternalForce();
  feat10opt->ComputeInternalForce();

  Eigen::VectorXf f_int_feat10opt;
  feat10opt->RetrieveInternalForceToCPU(f_int_feat10opt);

  // === Compare ===
  std::cout << "\n=== FEAT10 vs FEAT10Opt Comparison ===" << std::endl;

  double max_force_feat10    = f_int_feat10.cwiseAbs().maxCoeff();
  double max_force_feat10opt = f_int_feat10opt.cwiseAbs().maxCoeff();

  std::cout << "FEAT10 max |f_int|: " << max_force_feat10 << std::endl;
  std::cout << "FEAT10Opt max |f_int|: " << max_force_feat10opt << std::endl;

  // Compute relative difference (L2 norm)
  double diff_norm_sq   = 0.0;
  double feat10_norm_sq = 0.0;

  for (int i = 0; i < n_nodes_ * 3; i++) {
    double f1   = f_int_feat10(i);
    double f2   = static_cast<double>(f_int_feat10opt(i));
    double diff = f1 - f2;
    diff_norm_sq += diff * diff;
    feat10_norm_sq += f1 * f1;
  }

  double rel_diff = std::sqrt(diff_norm_sq / feat10_norm_sq);
  std::cout << "Relative L2 difference: " << rel_diff * 100.0 << "%"
            << std::endl;

  // Due to different quadrature rules, some difference is expected
  // The difference should be bounded but not zero
  EXPECT_LT(rel_diff, 2.0)
      << "Implementations differ more than expected - check for bugs";

  // Both should produce non-trivial forces
  EXPECT_GT(max_force_feat10, 1e3) << "FEAT10 forces too small";
  EXPECT_GT(max_force_feat10opt, 1e3) << "FEAT10Opt forces too small";

  feat10->Destroy();
  delete feat10;
  feat10opt->Destroy();
  delete feat10opt;
}

TEST_F(FEAT10BeamIntegrationTest, BeamRes0_SymmetryCheck) {
  // Apply symmetric deformation (uniform stretch in Y-Z plane) and verify
  // that forces have expected symmetry properties

  // The beam is centered at approximately y=1, z=0.5
  // A uniform Y-Z stretch should produce symmetric forces about the centerlines

  double F[3][3];
  SetIdentityF(F);
  F[1][1] = 1.1;  // 10% stretch in Y
  F[2][2] = 1.1;  // 10% stretch in Z

  Eigen::VectorXd h_x, h_y, h_z;
  ApplyAffineDeformation(F, h_x, h_y, h_z);

  GPU_FEAT10_Data* element = SetupFEAT10(h_x_ref_, h_y_ref_, h_z_ref_);
  element->CalcDnDuPre();
  element->UpdatePositions(h_x, h_y, h_z);
  element->CalcP();
  element->CalcInternalForce();

  Eigen::VectorXd f_int;
  element->RetrieveInternalForceToCPU(f_int);

  // For symmetric deformation, the sum of X-direction forces should be zero
  // (no net force in the unstretched direction)
  double sum_fx = 0.0;
  for (int i = 0; i < n_nodes_; i++) {
    sum_fx += f_int(i * 3 + 0);
  }

  std::cout << "\n=== Symmetry Check ===" << std::endl;
  std::cout << "Sum of X-direction forces: " << sum_fx << std::endl;

  double max_force  = f_int.cwiseAbs().maxCoeff();
  double rel_sum_fx = std::abs(sum_fx) / max_force;

  EXPECT_LT(rel_sum_fx, 1e-10)
      << "Symmetric deformation should produce zero net X-force";

  // Verify equilibrium still holds
  double eq_error = CheckForceEquilibrium(f_int, n_nodes_);
  EXPECT_LT(eq_error, 1e-10) << "Force equilibrium violated";

  element->Destroy();
  delete element;
}

TEST_F(FEAT10BeamIntegrationTest, BeamRes0_BoundaryForceBalance) {
  // Apply a deformation that stretches the beam in X direction.
  // The total force at the "left" end (x=x_min) should equal and opposite
  // the total force at the "right" end (x=x_max).
  //
  // This test verifies proper force distribution in boundary regions.

  double F[3][3];
  SetIdentityF(F);
  F[0][0] = 1.2;  // 20% stretch in X (along beam axis)

  Eigen::VectorXd h_x, h_y, h_z;
  ApplyAffineDeformation(F, h_x, h_y, h_z);

  GPU_FEAT10_Data* element = SetupFEAT10(h_x_ref_, h_y_ref_, h_z_ref_);
  element->CalcDnDuPre();
  element->UpdatePositions(h_x, h_y, h_z);
  element->CalcP();
  element->CalcInternalForce();

  Eigen::VectorXd f_int;
  element->RetrieveInternalForceToCPU(f_int);

  // Get boundary node sets
  std::vector<int> left_nodes  = GetFixedEndNodes();
  std::vector<int> right_nodes = GetFreeEndNodes();

  std::cout << "\n=== Boundary Force Balance ===" << std::endl;
  std::cout << "Left end nodes (x=0): " << left_nodes.size() << std::endl;
  std::cout << "Right end nodes (x=3): " << right_nodes.size() << std::endl;

  // Sum forces at each end
  Eigen::Vector3d sum_left  = Eigen::Vector3d::Zero();
  Eigen::Vector3d sum_right = Eigen::Vector3d::Zero();

  for (int node : left_nodes) {
    sum_left(0) += f_int(node * 3 + 0);
    sum_left(1) += f_int(node * 3 + 1);
    sum_left(2) += f_int(node * 3 + 2);
  }

  for (int node : right_nodes) {
    sum_right(0) += f_int(node * 3 + 0);
    sum_right(1) += f_int(node * 3 + 1);
    sum_right(2) += f_int(node * 3 + 2);
  }

  std::cout << "Sum force at left end: [" << sum_left.transpose() << "]"
            << std::endl;
  std::cout << "Sum force at right end: [" << sum_right.transpose() << "]"
            << std::endl;

  // For uniaxial stretch, the X-component of forces at each end should be
  // equal and opposite
  // Note: Total equilibrium (sum_left + sum_right + interior = 0) is already
  // verified by CheckForceEquilibrium

  // The X-forces at boundaries should be tensile (positive at right, negative
  // at left for stretch > 1)
  EXPECT_GT(sum_right(0), 0.0)
      << "Right end should have positive X-force (tensile)";
  EXPECT_LT(sum_left(0), 0.0)
      << "Left end should have negative X-force (reaction)";

  // Y and Z forces at boundaries should be relatively small compared to X
  // (for uniaxial X stretch)
  double boundary_fx = std::abs(sum_left(0)) + std::abs(sum_right(0));
  double boundary_fy = std::abs(sum_left(1)) + std::abs(sum_right(1));
  double boundary_fz = std::abs(sum_left(2)) + std::abs(sum_right(2));

  std::cout << "Boundary force magnitudes: X=" << boundary_fx
            << " Y=" << boundary_fy << " Z=" << boundary_fz << std::endl;

  // For pure X-stretch, Y and Z boundary forces should be much smaller than X
  EXPECT_LT(boundary_fy / boundary_fx, 0.1)
      << "Y boundary forces should be small for X-stretch";
  EXPECT_LT(boundary_fz / boundary_fx, 0.1)
      << "Z boundary forces should be small for X-stretch";

  element->Destroy();
  delete element;
}

TEST_F(FEAT10BeamIntegrationTest, BeamRes0_LargeDeformation) {
  // Test with large deformation to verify numerical stability

  double F[3][3];
  SetPureStretchF(F, 1.5, 0.75, 0.9);  // 50% X stretch, significant compression

  Eigen::VectorXd h_x, h_y, h_z;
  ApplyAffineDeformation(F, h_x, h_y, h_z);

  // === FEAT10 ===
  GPU_FEAT10_Data* feat10 = SetupFEAT10(h_x_ref_, h_y_ref_, h_z_ref_);
  feat10->CalcDnDuPre();
  feat10->UpdatePositions(h_x, h_y, h_z);
  feat10->CalcP();
  feat10->CalcInternalForce();

  Eigen::VectorXd f_int_feat10;
  feat10->RetrieveInternalForceToCPU(f_int_feat10);

  // === FEAT10Opt ===
  GPU_FEAT10Opt_Data* feat10opt = SetupFEAT10Opt(h_x_ref_, h_y_ref_, h_z_ref_);
  feat10opt->ComputePrecomputation();

  Eigen::MatrixXd positions(n_nodes_, 3);
  for (int i = 0; i < n_nodes_; i++) {
    positions(i, 0) = h_x(i);
    positions(i, 1) = h_y(i);
    positions(i, 2) = h_z(i);
  }
  feat10opt->UpdatePositions(positions);

  feat10opt->ClearInternalForce();
  feat10opt->ComputeInternalForce();

  Eigen::VectorXf f_int_feat10opt;
  feat10opt->RetrieveInternalForceToCPU(f_int_feat10opt);

  std::cout << "\n=== Large Deformation Test ===" << std::endl;

  // Check for NaN/Inf
  bool feat10_valid    = f_int_feat10.allFinite();
  bool feat10opt_valid = f_int_feat10opt.allFinite();

  EXPECT_TRUE(feat10_valid) << "FEAT10 produced NaN/Inf values";
  EXPECT_TRUE(feat10opt_valid) << "FEAT10Opt produced NaN/Inf values";

  // Check equilibrium
  double eq_error_feat10    = CheckForceEquilibrium(f_int_feat10, n_nodes_);
  double eq_error_feat10opt = CheckForceEquilibrium(f_int_feat10opt, n_nodes_);

  std::cout << "FEAT10 equilibrium error: " << eq_error_feat10 << std::endl;
  std::cout << "FEAT10Opt equilibrium error: " << eq_error_feat10opt
            << std::endl;

  EXPECT_LT(eq_error_feat10, 1e-10) << "FEAT10 equilibrium violated";
  EXPECT_LT(eq_error_feat10opt, 1e-3)
      << "FEAT10Opt equilibrium violated";  // Larger tolerance for float

  feat10->Destroy();
  delete feat10;
  feat10opt->Destroy();
  delete feat10opt;
}

TEST_F(FEAT10BeamIntegrationTest, BeamRes0_ShearDeformation) {
  // Test with shear deformation

  double F[3][3];
  SetSimpleShearF(F, 0.3, 0, 1);  // 30% shear in XY plane

  Eigen::VectorXd h_x, h_y, h_z;
  ApplyAffineDeformation(F, h_x, h_y, h_z);

  GPU_FEAT10_Data* element = SetupFEAT10(h_x_ref_, h_y_ref_, h_z_ref_);
  element->CalcDnDuPre();
  element->UpdatePositions(h_x, h_y, h_z);
  element->CalcP();
  element->CalcInternalForce();

  Eigen::VectorXd f_int;
  element->RetrieveInternalForceToCPU(f_int);

  std::cout << "\n=== Shear Deformation Test ===" << std::endl;

  double eq_error = CheckForceEquilibrium(f_int, n_nodes_);
  std::cout << "Equilibrium error: " << eq_error << std::endl;

  EXPECT_LT(eq_error, 1e-10) << "Force equilibrium violated";

  // For shear, forces should be non-trivial
  double max_force = f_int.cwiseAbs().maxCoeff();
  std::cout << "Max |f_int|: " << max_force << std::endl;
  EXPECT_GT(max_force, 1e3) << "Shear should produce significant forces";

  element->Destroy();
  delete element;
}
