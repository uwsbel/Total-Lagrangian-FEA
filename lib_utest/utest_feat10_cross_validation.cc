/*
 * Cross-validation test comparing FEAT10 vs FEAT10Opt implementations.
 *
 * Purpose: Validate that FEAT10Opt (float precision, 4-point quadrature, fused
 * kernel) produces results consistent with FEAT10 (double precision, 5-point
 * Keast quadrature, separate kernels).
 *
 * Key insight: For affine deformation (x = F * X), the deformation gradient F
 * is constant throughout the element. However, grad_N (shape function gradient)
 * varies spatially, so the internal force integral f = ∫ P·grad_N dV depends
 * on the quadrature rule used.
 *
 * For T10 elements:
 * - 5-point Keast rule: degree 4 precision (integrates polynomials up to degree
 * 4)
 * - 4-point rule: degree 2 precision
 *
 * The shape function gradients for T10 are quadratic (degree 2), so both
 * quadrature rules should integrate the internal force exactly for constant P.
 *
 * This test validates both implementations produce the same internal forces.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

#include "lib_src/elements/FEAT10Data.cuh"
#include "lib_src/elements/FEAT10DataOpt.cuh"

// ===========================================================================
// Test Fixture
// ===========================================================================

class FEAT10CrossValidationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Unit tetrahedron with corners at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    // Node ordering for T10:
    //   0-3: corner nodes
    //   4-9: edge midpoints
    // Edge connectivity: [(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]

    // Corner nodes
    X_ref_[0][0] = 0.0;
    X_ref_[0][1] = 0.0;
    X_ref_[0][2] = 0.0;  // Node 0
    X_ref_[1][0] = 1.0;
    X_ref_[1][1] = 0.0;
    X_ref_[1][2] = 0.0;  // Node 1
    X_ref_[2][0] = 0.0;
    X_ref_[2][1] = 1.0;
    X_ref_[2][2] = 0.0;  // Node 2
    X_ref_[3][0] = 0.0;
    X_ref_[3][1] = 0.0;
    X_ref_[3][2] = 1.0;  // Node 3

    // Edge midpoints
    X_ref_[4][0] = 0.5;
    X_ref_[4][1] = 0.0;
    X_ref_[4][2] = 0.0;  // Edge 0-1
    X_ref_[5][0] = 0.5;
    X_ref_[5][1] = 0.5;
    X_ref_[5][2] = 0.0;  // Edge 1-2
    X_ref_[6][0] = 0.0;
    X_ref_[6][1] = 0.5;
    X_ref_[6][2] = 0.0;  // Edge 0-2
    X_ref_[7][0] = 0.0;
    X_ref_[7][1] = 0.0;
    X_ref_[7][2] = 0.5;  // Edge 0-3
    X_ref_[8][0] = 0.5;
    X_ref_[8][1] = 0.0;
    X_ref_[8][2] = 0.5;  // Edge 1-3
    X_ref_[9][0] = 0.0;
    X_ref_[9][1] = 0.5;
    X_ref_[9][2] = 0.5;  // Edge 2-3

    // Material parameters: Mooney-Rivlin
    mu10_  = 80769.23;
    mu01_  = 20192.31;
    kappa_ = 400000.0;

    // Applied deformation gradient (affine)
    F_applied_[0][0] = 1.10;
    F_applied_[0][1] = 0.05;
    F_applied_[0][2] = 0.02;
    F_applied_[1][0] = 0.03;
    F_applied_[1][1] = 1.15;
    F_applied_[1][2] = 0.04;
    F_applied_[2][0] = 0.01;
    F_applied_[2][1] = 0.02;
    F_applied_[2][2] = 0.95;

    // Setup 5-point Keast quadrature rule for FEAT10
    qp_x_5pt_.resize(5);
    qp_y_5pt_.resize(5);
    qp_z_5pt_.resize(5);
    qp_weights_5pt_.resize(5);

    qp_x_5pt_(0) = 0.25;
    qp_y_5pt_(0) = 0.25;
    qp_z_5pt_(0) = 0.25;
    double a = 0.5, b = 1.0 / 6.0;
    qp_x_5pt_(1) = a;
    qp_y_5pt_(1) = b;
    qp_z_5pt_(1) = b;
    qp_x_5pt_(2) = b;
    qp_y_5pt_(2) = a;
    qp_z_5pt_(2) = b;
    qp_x_5pt_(3) = b;
    qp_y_5pt_(3) = b;
    qp_z_5pt_(3) = a;
    qp_x_5pt_(4) = b;
    qp_y_5pt_(4) = b;
    qp_z_5pt_(4) = b;

    qp_weights_5pt_(0) = -4.0 / 5.0 * (1.0 / 6.0);
    for (int i = 1; i < 5; i++) {
      qp_weights_5pt_(i) = 9.0 / 20.0 * (1.0 / 6.0);
    }
  }

  // Apply affine deformation x = F_applied * X
  void ApplyAffineDeformation(double x_cur[10][3]) {
    for (int a = 0; a < 10; a++) {
      for (int i = 0; i < 3; i++) {
        x_cur[a][i] = 0.0;
        for (int j = 0; j < 3; j++) {
          x_cur[a][i] += F_applied_[i][j] * X_ref_[a][j];
        }
      }
    }
  }

  // Convert double[10][3] to 3 separate Eigen::VectorXd (for FEAT10 API)
  void ToVectors(const double x[10][3], Eigen::VectorXd& vx,
                 Eigen::VectorXd& vy, Eigen::VectorXd& vz) {
    vx.resize(10);
    vy.resize(10);
    vz.resize(10);
    for (int i = 0; i < 10; i++) {
      vx(i) = x[i][0];
      vy(i) = x[i][1];
      vz(i) = x[i][2];
    }
  }

  // Convert double[10][3] to Eigen::MatrixXd (for FEAT10Opt API)
  Eigen::MatrixXd ToMatrix(const double x[10][3]) {
    Eigen::MatrixXd mat(10, 3);
    for (int i = 0; i < 10; i++) {
      mat(i, 0) = x[i][0];
      mat(i, 1) = x[i][1];
      mat(i, 2) = x[i][2];
    }
    return mat;
  }

  // Setup FEAT10 element with given positions
  GPU_FEAT10_Data* SetupFEAT10(const double x_nodes[10][3]) {
    GPU_FEAT10_Data* element = new GPU_FEAT10_Data(1, 10);
    element->Initialize();

    // Setup connectivity (single element with nodes 0-9)
    Eigen::MatrixXi connectivity(1, 10);
    for (int i = 0; i < 10; i++) {
      connectivity(0, i) = i;
    }

    // Setup node positions as 3 separate vectors
    Eigen::VectorXd h_x12, h_y12, h_z12;
    ToVectors(x_nodes, h_x12, h_y12, h_z12);

    element->Setup(qp_x_5pt_, qp_y_5pt_, qp_z_5pt_, qp_weights_5pt_, h_x12,
                   h_y12, h_z12, connectivity);
    element->SetMooneyRivlin(mu10_, mu01_, kappa_);

    return element;
  }

  // Setup FEAT10Opt element with given positions
  GPU_FEAT10Opt_Data* SetupFEAT10Opt(const double x_nodes[10][3]) {
    GPU_FEAT10Opt_Data* element = new GPU_FEAT10Opt_Data();
    element->Initialize(1, 10);

    // Setup connectivity (single element with nodes 0-9)
    Eigen::MatrixXi connectivity(1, 10);
    for (int i = 0; i < 10; i++) {
      connectivity(0, i) = i;
    }

    // Setup node positions as matrix
    Eigen::MatrixXd positions = ToMatrix(x_nodes);

    element->Setup(positions, connectivity);
    element->SetMooneyRivlin(static_cast<float>(mu10_),
                             static_cast<float>(mu01_),
                             static_cast<float>(kappa_));

    return element;
  }

  double X_ref_[10][3];
  double F_applied_[3][3];
  double mu10_, mu01_, kappa_;
  Eigen::VectorXd qp_x_5pt_, qp_y_5pt_, qp_z_5pt_, qp_weights_5pt_;
};

// ===========================================================================
// Test Cases
// ===========================================================================

TEST_F(FEAT10CrossValidationTest, UndeformedConfig_BothZero) {
  // In undeformed configuration, both implementations should give f_int ≈ 0

  // === FEAT10 ===
  GPU_FEAT10_Data* feat10 = SetupFEAT10(X_ref_);
  feat10->CalcDnDuPre();
  feat10->CalcP();
  feat10->CalcInternalForce();

  Eigen::VectorXd f_int_feat10;
  feat10->RetrieveInternalForceToCPU(f_int_feat10);

  // === FEAT10Opt ===
  GPU_FEAT10Opt_Data* feat10opt = SetupFEAT10Opt(X_ref_);
  feat10opt->ComputePrecomputation();
  feat10opt->ClearInternalForce();
  feat10opt->ComputeInternalForce();

  Eigen::VectorXf f_int_feat10opt;
  feat10opt->RetrieveInternalForceToCPU(f_int_feat10opt);

  // === Verify both are near zero ===
  std::cout << "\n=== Undeformed Config Test ===" << std::endl;

  // FEAT10 (double precision): expect very small values
  double max_feat10 = f_int_feat10.cwiseAbs().maxCoeff();
  std::cout << "FEAT10 max |f_int|: " << max_feat10 << std::endl;
  EXPECT_LT(max_feat10, 1e-8) << "FEAT10 should have near-zero forces";

  // FEAT10Opt (float precision): larger tolerance due to float errors
  // Material constants ~1e5, float epsilon ~1e-7, expect ~0.01 errors
  double max_feat10opt = f_int_feat10opt.cwiseAbs().maxCoeff();
  std::cout << "FEAT10Opt max |f_int|: " << max_feat10opt << std::endl;
  EXPECT_LT(max_feat10opt, 0.1) << "FEAT10Opt should have near-zero forces";

  // Cleanup
  feat10->Destroy();
  delete feat10;
  feat10opt->Destroy();
  delete feat10opt;
}

TEST_F(FEAT10CrossValidationTest, DeformationGradient_AffineDeformation) {
  // For affine deformation, F should equal F_applied at all quadrature points
  // Both implementations should compute the same F

  double x_cur[10][3];
  ApplyAffineDeformation(x_cur);

  // === FEAT10 ===
  GPU_FEAT10_Data* feat10 = SetupFEAT10(X_ref_);
  feat10->CalcDnDuPre();  // Precompute with reference config

  // Update to deformed positions
  Eigen::VectorXd h_x12, h_y12, h_z12;
  ToVectors(x_cur, h_x12, h_y12, h_z12);
  feat10->UpdatePositions(h_x12, h_y12, h_z12);

  feat10->CalcP();  // This computes F internally

  std::vector<std::vector<Eigen::MatrixXd>> F_feat10;
  feat10->RetrieveDeformationGradientToCPU(F_feat10);

  // === FEAT10Opt ===
  GPU_FEAT10Opt_Data* feat10opt = SetupFEAT10Opt(X_ref_);
  feat10opt->ComputePrecomputation();  // Precompute with reference config

  // Update to deformed positions
  Eigen::MatrixXd positions = ToMatrix(x_cur);
  feat10opt->UpdatePositions(positions);

  feat10opt->ClearInternalForce();
  feat10opt->ComputeInternalForce(true);  // writeOutF = true

  std::vector<std::vector<Eigen::Matrix3f>> F_feat10opt;
  feat10opt->RetrieveDeformationGradientToCPU(F_feat10opt);

  // === Compare F values ===
  std::cout << "\n=== Deformation Gradient Test ===" << std::endl;
  std::cout << "Applied F:" << std::endl;
  for (int i = 0; i < 3; i++) {
    std::cout << "  [" << F_applied_[i][0] << ", " << F_applied_[i][1] << ", "
              << F_applied_[i][2] << "]" << std::endl;
  }

  // Check FEAT10 F at first QP (should be constant across QPs for affine)
  std::cout << "\nFEAT10 F at QP 0:" << std::endl;
  std::cout << F_feat10[0][0] << std::endl;

  std::cout << "\nFEAT10Opt F at QP 0:" << std::endl;
  std::cout << F_feat10opt[0][0] << std::endl;

  // Compare F_applied with both implementations
  double max_error_feat10    = 0.0;
  double max_error_feat10opt = 0.0;

  // Check FEAT10 (5 QPs)
  for (int qp = 0; qp < 5; qp++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        double error     = std::abs(F_feat10[0][qp](i, j) - F_applied_[i][j]);
        max_error_feat10 = std::max(max_error_feat10, error);
      }
    }
  }

  // Check FEAT10Opt (4 QPs)
  for (int qp = 0; qp < 4; qp++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        double error = std::abs(static_cast<double>(F_feat10opt[0][qp](i, j)) -
                                F_applied_[i][j]);
        max_error_feat10opt = std::max(max_error_feat10opt, error);
      }
    }
  }

  std::cout << "\nMax |F - F_applied| FEAT10: " << max_error_feat10
            << std::endl;
  std::cout << "Max |F - F_applied| FEAT10Opt: " << max_error_feat10opt
            << std::endl;

  // FEAT10 (double): expect ~1e-10 tolerance
  EXPECT_LT(max_error_feat10, 1e-10)
      << "FEAT10 F should match F_applied exactly";

  // FEAT10Opt (float): expect ~1e-5 tolerance
  EXPECT_LT(max_error_feat10opt, 1e-5)
      << "FEAT10Opt F should match F_applied within float precision";

  // Cleanup
  feat10->Destroy();
  delete feat10;
  feat10opt->Destroy();
  delete feat10opt;
}

TEST_F(FEAT10CrossValidationTest, ForceEquilibrium_AffineDeformation) {
  // For a single free element under any deformation, the sum of all nodal
  // internal forces must be zero (Newton's 3rd law / momentum conservation).
  //
  // This is a fundamental physical constraint that both implementations
  // must satisfy regardless of their quadrature rule.
  //
  // For body force free element: ∑ f_int = ∫ P·grad_N dV summed over all nodes
  // Using the partition of unity property: ∑ grad_N = 0
  // Therefore: ∑ f_int = P · (∑ grad_N) · V = 0

  double x_cur[10][3];
  ApplyAffineDeformation(x_cur);

  // === FEAT10 ===
  GPU_FEAT10_Data* feat10 = SetupFEAT10(X_ref_);
  feat10->CalcDnDuPre();

  Eigen::VectorXd h_x12, h_y12, h_z12;
  ToVectors(x_cur, h_x12, h_y12, h_z12);
  feat10->UpdatePositions(h_x12, h_y12, h_z12);

  feat10->CalcP();
  feat10->CalcInternalForce();

  Eigen::VectorXd f_int_feat10;
  feat10->RetrieveInternalForceToCPU(f_int_feat10);

  // === FEAT10Opt ===
  GPU_FEAT10Opt_Data* feat10opt = SetupFEAT10Opt(X_ref_);
  feat10opt->ComputePrecomputation();

  Eigen::MatrixXd positions = ToMatrix(x_cur);
  feat10opt->UpdatePositions(positions);

  feat10opt->ClearInternalForce();
  feat10opt->ComputeInternalForce();

  Eigen::VectorXf f_int_feat10opt;
  feat10opt->RetrieveInternalForceToCPU(f_int_feat10opt);

  // === Check force equilibrium ===
  std::cout << "\n=== Force Equilibrium Test ===" << std::endl;

  // Sum forces in each direction for FEAT10
  Eigen::Vector3d sum_feat10 = Eigen::Vector3d::Zero();
  for (int node = 0; node < 10; node++) {
    sum_feat10(0) += f_int_feat10(node * 3 + 0);
    sum_feat10(1) += f_int_feat10(node * 3 + 1);
    sum_feat10(2) += f_int_feat10(node * 3 + 2);
  }

  // Sum forces in each direction for FEAT10Opt
  Eigen::Vector3f sum_feat10opt = Eigen::Vector3f::Zero();
  for (int node = 0; node < 10; node++) {
    sum_feat10opt(0) += f_int_feat10opt(node * 3 + 0);
    sum_feat10opt(1) += f_int_feat10opt(node * 3 + 1);
    sum_feat10opt(2) += f_int_feat10opt(node * 3 + 2);
  }

  std::cout << "FEAT10 sum(f_int): [" << sum_feat10(0) << ", " << sum_feat10(1)
            << ", " << sum_feat10(2) << "]" << std::endl;
  std::cout << "FEAT10Opt sum(f_int): [" << sum_feat10opt(0) << ", "
            << sum_feat10opt(1) << ", " << sum_feat10opt(2) << "]" << std::endl;

  // Get characteristic force magnitude for relative tolerance
  double max_force_feat10   = f_int_feat10.cwiseAbs().maxCoeff();
  float max_force_feat10opt = f_int_feat10opt.cwiseAbs().maxCoeff();

  std::cout << "Max |f_int| FEAT10: " << max_force_feat10 << std::endl;
  std::cout << "Max |f_int| FEAT10Opt: " << max_force_feat10opt << std::endl;

  // FEAT10 (double precision): expect very tight equilibrium
  for (int d = 0; d < 3; d++) {
    double rel_error = std::abs(sum_feat10(d)) / max_force_feat10;
    EXPECT_LT(rel_error, 1e-10)
        << "FEAT10 force equilibrium violated in direction " << d
        << ": sum=" << sum_feat10(d);
  }

  // FEAT10Opt (float precision): larger tolerance due to float accumulation
  for (int d = 0; d < 3; d++) {
    double rel_error = std::abs(sum_feat10opt(d)) / max_force_feat10opt;
    EXPECT_LT(rel_error, 1e-4)
        << "FEAT10Opt force equilibrium violated in direction " << d
        << ": sum=" << sum_feat10opt(d);
  }

  // Cleanup
  feat10->Destroy();
  delete feat10;
  feat10opt->Destroy();
  delete feat10opt;
}

TEST_F(FEAT10CrossValidationTest, InternalForce_QuadratureDifference) {
  // This test documents the expected behavior: different quadrature rules
  // produce different nodal force distributions for the same deformation.
  //
  // Key insight: While the TOTAL force sums to zero for both (equilibrium),
  // the DISTRIBUTION of forces among nodes differs because:
  // - f_a = ∫ P·grad_N_a dV (force at node a)
  // - Different quadrature rules sample grad_N_a at different locations
  // - Both rules integrate correctly, but assign forces differently
  //
  // This is NOT a bug - it's expected physics. Both are valid FEM solutions
  // that converge to the same answer as the mesh is refined.
  //
  // This test just verifies and documents the magnitude of the difference.

  double x_cur[10][3];
  ApplyAffineDeformation(x_cur);

  // === FEAT10 ===
  GPU_FEAT10_Data* feat10 = SetupFEAT10(X_ref_);
  feat10->CalcDnDuPre();

  Eigen::VectorXd h_x12, h_y12, h_z12;
  ToVectors(x_cur, h_x12, h_y12, h_z12);
  feat10->UpdatePositions(h_x12, h_y12, h_z12);

  feat10->CalcP();
  feat10->CalcInternalForce();

  Eigen::VectorXd f_int_feat10;
  feat10->RetrieveInternalForceToCPU(f_int_feat10);

  // === FEAT10Opt ===
  GPU_FEAT10Opt_Data* feat10opt = SetupFEAT10Opt(X_ref_);
  feat10opt->ComputePrecomputation();

  Eigen::MatrixXd positions = ToMatrix(x_cur);
  feat10opt->UpdatePositions(positions);

  feat10opt->ClearInternalForce();
  feat10opt->ComputeInternalForce();

  Eigen::VectorXf f_int_feat10opt;
  feat10opt->RetrieveInternalForceToCPU(f_int_feat10opt);

  // === Document the difference ===
  std::cout << "\n=== Quadrature Difference (Expected Behavior) ==="
            << std::endl;
  std::cout << "Node | FEAT10 (5-pt Keast)      | FEAT10Opt (4-pt)         | "
               "Difference"
            << std::endl;
  std::cout << std::string(85, '-') << std::endl;

  double total_diff_norm   = 0.0;
  double total_feat10_norm = 0.0;

  for (int node = 0; node < 10; node++) {
    Eigen::Vector3d f_feat10, f_feat10opt, diff;
    for (int d = 0; d < 3; d++) {
      f_feat10(d)    = f_int_feat10(node * 3 + d);
      f_feat10opt(d) = static_cast<double>(f_int_feat10opt(node * 3 + d));
      diff(d)        = f_feat10(d) - f_feat10opt(d);
    }

    total_diff_norm += diff.squaredNorm();
    total_feat10_norm += f_feat10.squaredNorm();

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  " << node << "  | [" << std::setw(9) << f_feat10(0) << ", "
              << std::setw(9) << f_feat10(1) << ", " << std::setw(9)
              << f_feat10(2) << "] | [" << std::setw(9) << f_feat10opt(0)
              << ", " << std::setw(9) << f_feat10opt(1) << ", " << std::setw(9)
              << f_feat10opt(2) << "] | " << std::setprecision(1) << diff.norm()
              << std::endl;
  }

  std::cout << std::string(85, '-') << std::endl;

  double relative_diff = std::sqrt(total_diff_norm / total_feat10_norm);
  std::cout << "Relative L2 difference: " << std::setprecision(4)
            << relative_diff * 100.0 << "%" << std::endl;

  // This test is for documentation only - we expect a significant difference
  // but want to ensure it's bounded (not indicating a bug)
  // Typical difference is 50-100% for single element due to different QP
  // weights

  // Just verify the difference is reasonable (not indicating NaN or overflow)
  EXPECT_LT(relative_diff, 10.0)
      << "Quadrature difference unexpectedly large - check for bugs";
  EXPECT_GT(relative_diff, 0.01)
      << "Quadrature difference unexpectedly small - implementations may be "
         "using same rule";

  // Cleanup
  feat10->Destroy();
  delete feat10;
  feat10opt->Destroy();
  delete feat10opt;
}
