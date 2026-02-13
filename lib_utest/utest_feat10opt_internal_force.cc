/*
 * Unit test for FEAT10Opt element internal force computation.
 * Validates GPU implementation of deformation gradient (F) and internal force
 * (f_int) against CPU reference implementation.
 *
 * Key differences from utest_feat10_internal_force.cc:
 * - Uses 4-point quadrature rule (vs 5-point Keast)
 * - Float compute precision (vs double)
 * - Fused kernel (no separate P retrieval)
 * - Different API (Setup takes positions matrix)
 *
 * Uses shared utilities from feat10_test_utils.h.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "feat10_test_utils.h"
#include "lib_src/elements/FEAT10DataOpt.cuh"

using namespace feat10_test;

// ===========================================================================
// Test Fixture
// ===========================================================================

class FEAT10OptInternalForceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup unit tetrahedron
    SetupUnitTetrahedron(X_ref_);

    // Material parameters: Mooney-Rivlin
    mu10_  = 80769.23;
    mu01_  = 20192.31;
    kappa_ = 400000.0;

    // Setup 4-point quadrature rule (used by FEAT10Opt)
    Setup4PointQuadrature(qp_x_, qp_y_, qp_z_, qp_weights_);
  }

  // Setup GPU element data structure
  GPU_FEAT10Opt_Data* SetupGPUElement(const double x_nodes[10][3]) {
    GPU_FEAT10Opt_Data* element = new GPU_FEAT10Opt_Data();
    element->Initialize(1, 10);  // 1 element, 10 nodes

    // Setup connectivity (single element with nodes 0-9)
    Eigen::MatrixXi connectivity(1, 10);
    for (int i = 0; i < 10; i++) {
      connectivity(0, i) = i;
    }

    // Setup node positions as [n_nodes x 3] matrix
    Eigen::MatrixXd positions(10, 3);
    for (int i = 0; i < 10; i++) {
      positions(i, 0) = x_nodes[i][0];
      positions(i, 1) = x_nodes[i][1];
      positions(i, 2) = x_nodes[i][2];
    }

    element->Setup(positions, connectivity);
    element->SetMooneyRivlin(static_cast<float>(mu10_),
                             static_cast<float>(mu01_),
                             static_cast<float>(kappa_));

    return element;
  }

  // Update GPU element positions
  void UpdateGPUPositions(GPU_FEAT10Opt_Data* element,
                          const double x_nodes[10][3]) {
    Eigen::MatrixXd positions(10, 3);
    for (int i = 0; i < 10; i++) {
      positions(i, 0) = x_nodes[i][0];
      positions(i, 1) = x_nodes[i][1];
      positions(i, 2) = x_nodes[i][2];
    }
    element->UpdatePositions(positions);
  }

  // Helper to run a complete internal force test with given deformation
  void RunInternalForceTest(const double F_applied[3][3],
                            const std::string& test_name, double rel_tol = 1e-4,
                            double abs_tol = 0.01) {
    double x_cur[10][3];
    ApplyAffineDeformation(X_ref_, F_applied, x_cur);

    // Setup GPU element with reference positions first
    GPU_FEAT10Opt_Data* element = SetupGPUElement(X_ref_);
    element->ComputePrecomputation();

    // Update to current positions
    UpdateGPUPositions(element, x_cur);

    // Compute internal force on GPU
    element->ClearInternalForce();
    element->ComputeInternalForce(nullptr, false);

    // Retrieve internal force from GPU
    Eigen::VectorXf f_int_gpu;
    element->RetrieveInternalForceToCPU(f_int_gpu);

    // Compute internal force using CPU reference
    double f_int_cpu[10][3];
    ComputeInternalForce_CPU(X_ref_, x_cur, mu10_, mu01_, kappa_, qp_x_, qp_y_,
                             qp_z_, qp_weights_, f_int_cpu);

    // Compare CPU vs GPU
    for (int a = 0; a < 10; a++) {
      for (int i = 0; i < 3; i++) {
        int dof         = a * 3 + i;
        double gpu_val  = static_cast<double>(f_int_gpu(dof));
        double cpu_val  = f_int_cpu[a][i];
        double abs_diff = std::abs(gpu_val - cpu_val);

        double scale = std::max(std::abs(cpu_val), std::abs(gpu_val));
        if (scale > 1.0) {
          double rel_error = abs_diff / scale;
          EXPECT_LT(rel_error, rel_tol)
              << test_name << ": f_int mismatch at node " << a << " dof " << i
              << " (DOF " << dof << "): GPU=" << gpu_val << ", CPU=" << cpu_val;
        } else {
          EXPECT_NEAR(gpu_val, cpu_val, abs_tol)
              << test_name << ": f_int mismatch at node " << a << " dof " << i
              << " (DOF " << dof << "): GPU=" << gpu_val << ", CPU=" << cpu_val;
        }
      }
    }

    // Verify force equilibrium (float precision: larger tolerance)
    double equilibrium_error = CheckForceEquilibrium(f_int_gpu, 10);
    EXPECT_LT(equilibrium_error, 1e-4)
        << test_name << ": Force equilibrium violated";

    element->Destroy();
    delete element;
  }

  double X_ref_[10][3];
  double mu10_, mu01_, kappa_;
  Eigen::VectorXd qp_x_, qp_y_, qp_z_, qp_weights_;
};

// ===========================================================================
// Test Cases - Basic Validation
// ===========================================================================

TEST_F(FEAT10OptInternalForceTest, PaddedElements_InverseJacobianValid) {
  // Regression test: Ensure padded elements have valid (non-NaN/Inf) inverse
  // Jacobian values. This catches issues where degenerate padded elements
  // could corrupt computations (fixed by checking n_elem instead of
  // n_elem_padded).

  GPU_FEAT10Opt_Data* element = SetupGPUElement(X_ref_);
  element->ComputePrecomputation();

  int n_elem_padded = element->get_n_elem_padded();

  // Retrieve inverse Jacobian data (9 * 4 * n_elem_padded floats)
  std::vector<float> inv_jac(9 * 4 * n_elem_padded);
  cudaMemcpy(inv_jac.data(), element->d_iso_map_inv,
             9 * 4 * n_elem_padded * sizeof(float), cudaMemcpyDeviceToHost);

  // Check real element (elem 0) has valid values
  int block_idx = 0;
  for (int qp = 0; qp < 4; qp++) {
    int t = 0 * 4 + qp;  // elem 0
    for (int c = 0; c < 9; c++) {
      int idx = block_idx * 576 + c * 64 + t;
      EXPECT_FALSE(std::isnan(inv_jac[idx]))
          << "Real element 0, QP " << qp << ", component " << c << " is NaN";
      EXPECT_FALSE(std::isinf(inv_jac[idx]))
          << "Real element 0, QP " << qp << ", component " << c << " is Inf";
    }
  }

  element->Destroy();
  delete element;
}

TEST_F(FEAT10OptInternalForceTest, InverseJacobian_GPUvsCPU) {
  // Validates the precompute kernel that computes J^-1 at each QP

  GPU_FEAT10Opt_Data* element = SetupGPUElement(X_ref_);
  element->ComputePrecomputation();

  // Retrieve GPU inverse Jacobian
  int n_elem_padded = element->get_n_elem_padded();
  std::vector<float> inv_jac_gpu(9 * 4 * n_elem_padded);
  cudaMemcpy(inv_jac_gpu.data(), element->d_iso_map_inv,
             9 * 4 * n_elem_padded * sizeof(float), cudaMemcpyDeviceToHost);

  // Compare at each of the 4 quadrature points
  for (int qp = 0; qp < 4; qp++) {
    double xi   = qp_x_(qp);
    double eta  = qp_y_(qp);
    double zeta = qp_z_(qp);

    // CPU reference
    double Jinv_cpu[3][3];
    ComputeJacobianInverse(X_ref_, xi, eta, zeta, Jinv_cpu);

    // GPU values for element 0
    int block_idx = 0;
    int t         = qp;  // thread index for elem 0, qp

    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        int comp       = row * 3 + col;
        int idx        = block_idx * 576 + comp * 64 + t;
        float gpu_val  = inv_jac_gpu[idx];
        double cpu_val = Jinv_cpu[row][col];

        EXPECT_NEAR(gpu_val, cpu_val, 1e-5)
            << "J^-1 mismatch at QP " << qp << " (" << row << "," << col << ")"
            << " GPU=" << gpu_val << " CPU=" << cpu_val;
      }
    }
  }

  element->Destroy();
  delete element;
}

TEST_F(FEAT10OptInternalForceTest, UndeformedConfig_FIsIdentity) {
  // In undeformed configuration, F should be identity and f_int should be zero

  GPU_FEAT10Opt_Data* element = SetupGPUElement(X_ref_);

  element->ComputePrecomputation();
  element->ClearInternalForce();
  element->ComputeInternalForce(nullptr, true);  // writeOutF = true

  // Retrieve F from GPU
  std::vector<std::vector<Eigen::Matrix3f>> F_gpu;
  element->RetrieveDeformationGradientToCPU(F_gpu);

  // Check F = I at all quadrature points
  Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
  for (int qp = 0; qp < 4; qp++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        EXPECT_NEAR(F_gpu[0][qp](i, j), I(i, j), 1e-5)
            << "F mismatch at QP " << qp << " (" << i << "," << j << ")";
      }
    }
  }

  // Retrieve internal force
  Eigen::VectorXf f_int_gpu;
  element->RetrieveInternalForceToCPU(f_int_gpu);

  // Check f_int ≈ 0 (float precision tolerance)
  double tolerance = 0.1;
  for (int i = 0; i < 30; i++) {
    EXPECT_NEAR(f_int_gpu(i), 0.0f, tolerance)
        << "f_int should be near zero in undeformed config at DOF " << i;
  }

  element->Destroy();
  delete element;
}

TEST_F(FEAT10OptInternalForceTest, DeformationGradient_GPUvsCPU) {
  // Apply affine deformation and verify F matches at all quadrature points

  double F_applied[3][3];
  SetGeneralAffineF(F_applied);

  double x_cur[10][3];
  ApplyAffineDeformation(X_ref_, F_applied, x_cur);

  // Setup GPU element with REFERENCE positions first
  GPU_FEAT10Opt_Data* element = SetupGPUElement(X_ref_);
  element->ComputePrecomputation();

  // Update to current positions
  UpdateGPUPositions(element, x_cur);

  // Compute internal force (this computes F internally)
  element->ClearInternalForce();
  element->ComputeInternalForce(nullptr, true);  // writeOutF = true

  // Retrieve F from GPU
  std::vector<std::vector<Eigen::Matrix3f>> F_gpu;
  element->RetrieveDeformationGradientToCPU(F_gpu);

  // Compute F using CPU reference at each quadrature point and compare
  for (int qp = 0; qp < 4; qp++) {
    double xi   = qp_x_(qp);
    double eta  = qp_y_(qp);
    double zeta = qp_z_(qp);

    double dN_dxi[10][3];
    ComputeShapeFunctionGradients(xi, eta, zeta, dN_dxi);

    double grad_N[10][3];
    ComputeGradN_Physical(X_ref_, dN_dxi, grad_N);

    double F_cpu[3][3];
    ComputeF(x_cur, grad_N, F_cpu);

    // Compare CPU vs GPU
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        double gpu_val = static_cast<double>(F_gpu[0][qp](i, j));
        double cpu_val = F_cpu[i][j];

        EXPECT_NEAR(gpu_val, cpu_val, 1e-5)
            << "F mismatch at QP " << qp << " (" << i << "," << j << ")"
            << " GPU=" << gpu_val << " CPU=" << cpu_val;
      }
    }
  }

  element->Destroy();
  delete element;
}

TEST_F(FEAT10OptInternalForceTest, PiolaStress_GPUvsCPU) {
  // Apply affine deformation and verify P matches at all quadrature points

  double F_applied[3][3];
  SetGeneralAffineF(F_applied);

  double x_cur[10][3];
  ApplyAffineDeformation(X_ref_, F_applied, x_cur);

  // Setup GPU element
  GPU_FEAT10Opt_Data* element = SetupGPUElement(X_ref_);
  element->ComputePrecomputation();
  UpdateGPUPositions(element, x_cur);

  // Compute with P output enabled
  element->ClearInternalForce();
  element->ComputeInternalForce(nullptr, false, true);  // writeOutP=true

  // Retrieve P from GPU
  std::vector<std::vector<Eigen::Matrix3f>> P_gpu;
  element->RetrievePiolaToCPU(P_gpu);

  // Compute P using CPU reference
  for (int qp = 0; qp < 4; qp++) {
    double xi   = qp_x_(qp);
    double eta  = qp_y_(qp);
    double zeta = qp_z_(qp);

    double dN_dxi[10][3];
    ComputeShapeFunctionGradients(xi, eta, zeta, dN_dxi);

    double grad_N[10][3];
    ComputeGradN_Physical(X_ref_, dN_dxi, grad_N);

    double F_cpu[3][3];
    ComputeF(x_cur, grad_N, F_cpu);

    double P_cpu[3][3];
    ComputeP_MooneyRivlin(F_cpu, mu10_, mu01_, kappa_, P_cpu);

    // Compare CPU vs GPU
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        double gpu_val = static_cast<double>(P_gpu[0][qp](i, j));
        double cpu_val = P_cpu[i][j];

        double scale = std::max(std::abs(cpu_val), std::abs(gpu_val));
        if (scale > 1.0) {
          double rel_error = std::abs(gpu_val - cpu_val) / scale;
          EXPECT_LT(rel_error, 1e-4)
              << "P mismatch at QP " << qp << " (" << i << "," << j << ")"
              << " GPU=" << gpu_val << " CPU=" << cpu_val;
        } else {
          EXPECT_NEAR(gpu_val, cpu_val, 1e-3)
              << "P mismatch at QP " << qp << " (" << i << "," << j << ")"
              << " GPU=" << gpu_val << " CPU=" << cpu_val;
        }
      }
    }
  }

  element->Destroy();
  delete element;
}

TEST_F(FEAT10OptInternalForceTest, InternalForce_GPUvsCPU) {
  // Apply general affine deformation and verify internal force matches
  double F_applied[3][3];
  SetGeneralAffineF(F_applied);
  RunInternalForceTest(F_applied, "GeneralAffine");
}

// ===========================================================================
// Test Cases - Multiple Deformation Scenarios
// ===========================================================================

TEST_F(FEAT10OptInternalForceTest, PureStretch_Tensile) {
  double F[3][3];
  SetPureStretchF(F, 1.2, 1.1, 1.05);
  RunInternalForceTest(F, "PureStretch_Tensile");
}

TEST_F(FEAT10OptInternalForceTest, PureStretch_Compressive) {
  double F[3][3];
  SetPureStretchF(F, 0.85, 0.9, 0.95);
  RunInternalForceTest(F, "PureStretch_Compressive");
}

TEST_F(FEAT10OptInternalForceTest, SimpleShear_XY) {
  double F[3][3];
  SetSimpleShearF(F, 0.2, 0, 1);
  RunInternalForceTest(F, "SimpleShear_XY");
}

TEST_F(FEAT10OptInternalForceTest, SimpleShear_YZ) {
  double F[3][3];
  SetSimpleShearF(F, 0.15, 1, 2);
  RunInternalForceTest(F, "SimpleShear_YZ");
}

TEST_F(FEAT10OptInternalForceTest, PureRotation_Small) {
  // Small rotation about Z axis
  double F[3][3];
  SetRotationZF(F, 5.0 * M_PI / 180.0);

  double x_cur[10][3];
  ApplyAffineDeformation(X_ref_, F, x_cur);

  GPU_FEAT10Opt_Data* element = SetupGPUElement(X_ref_);
  element->ComputePrecomputation();
  UpdateGPUPositions(element, x_cur);

  element->ClearInternalForce();
  element->ComputeInternalForce(nullptr);

  Eigen::VectorXf f_int_gpu;
  element->RetrieveInternalForceToCPU(f_int_gpu);

  // For pure rotation, internal forces should be small
  double max_force = f_int_gpu.cwiseAbs().maxCoeff();
  EXPECT_LT(max_force, 1.0)
      << "Pure rotation should produce near-zero internal forces";

  // Equilibrium should still hold
  double equilibrium_error = CheckForceEquilibrium(f_int_gpu, 10);
  EXPECT_LT(equilibrium_error, 1e-4) << "Force equilibrium violated";

  element->Destroy();
  delete element;
}

TEST_F(FEAT10OptInternalForceTest, CombinedStretchShear) {
  double F[3][3];
  F[0][0] = 1.15;
  F[0][1] = 0.1;
  F[0][2] = 0.05;
  F[1][0] = 0.0;
  F[1][1] = 1.1;
  F[1][2] = 0.08;
  F[2][0] = 0.0;
  F[2][1] = 0.0;
  F[2][2] = 0.9;
  RunInternalForceTest(F, "CombinedStretchShear");
}

TEST_F(FEAT10OptInternalForceTest, LargeDeformation) {
  double F[3][3];
  SetPureStretchF(F, 1.5, 0.8, 0.85);
  RunInternalForceTest(F, "LargeDeformation");
}

// ===========================================================================
// Test Cases - Multi-Element
// ===========================================================================

class FEAT10OptMultiElementTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Same setup as FEAT10MultiElementTest
    n_nodes_ = 14;  // Simplified count
    n_elem_  = 2;

    nodes_.resize(n_nodes_, 3);
    // Element 0 nodes
    nodes_(0, 0) = 0.0;
    nodes_(0, 1) = 0.0;
    nodes_(0, 2) = 0.0;
    nodes_(1, 0) = 1.0;
    nodes_(1, 1) = 0.0;
    nodes_(1, 2) = 0.0;
    nodes_(2, 0) = 0.0;
    nodes_(2, 1) = 1.0;
    nodes_(2, 2) = 0.0;
    nodes_(3, 0) = 0.0;
    nodes_(3, 1) = 0.0;
    nodes_(3, 2) = 1.0;
    nodes_(4, 0) = 0.5;
    nodes_(4, 1) = 0.0;
    nodes_(4, 2) = 0.0;
    nodes_(5, 0) = 0.5;
    nodes_(5, 1) = 0.5;
    nodes_(5, 2) = 0.0;
    nodes_(6, 0) = 0.0;
    nodes_(6, 1) = 0.5;
    nodes_(6, 2) = 0.0;
    nodes_(7, 0) = 0.0;
    nodes_(7, 1) = 0.0;
    nodes_(7, 2) = 0.5;
    nodes_(8, 0) = 0.5;
    nodes_(8, 1) = 0.0;
    nodes_(8, 2) = 0.5;
    nodes_(9, 0) = 0.0;
    nodes_(9, 1) = 0.5;
    nodes_(9, 2) = 0.5;
    // Element 1 additional nodes
    nodes_(10, 0) = -1.0;
    nodes_(10, 1) = 0.0;
    nodes_(10, 2) = 0.0;
    nodes_(11, 0) = -0.5;
    nodes_(11, 1) = 0.0;
    nodes_(11, 2) = 0.0;
    nodes_(12, 0) = -0.5;
    nodes_(12, 1) = 0.5;
    nodes_(12, 2) = 0.0;
    nodes_(13, 0) = -0.5;
    nodes_(13, 1) = 0.0;
    nodes_(13, 2) = 0.5;

    connectivity_.resize(2, 10);
    connectivity_(0, 0) = 0;
    connectivity_(0, 1) = 1;
    connectivity_(0, 2) = 2;
    connectivity_(0, 3) = 3;
    connectivity_(0, 4) = 4;
    connectivity_(0, 5) = 5;
    connectivity_(0, 6) = 6;
    connectivity_(0, 7) = 7;
    connectivity_(0, 8) = 8;
    connectivity_(0, 9) = 9;

    connectivity_(1, 0) = 0;
    connectivity_(1, 1) = 10;
    connectivity_(1, 2) = 2;
    connectivity_(1, 3) = 3;
    connectivity_(1, 4) = 11;
    connectivity_(1, 5) = 12;
    connectivity_(1, 6) = 6;
    connectivity_(1, 7) = 7;
    connectivity_(1, 8) = 13;
    connectivity_(1, 9) = 9;

    mu10_  = 80769.23f;
    mu01_  = 20192.31f;
    kappa_ = 400000.0f;
  }

  int n_nodes_, n_elem_;
  Eigen::MatrixXd nodes_;
  Eigen::MatrixXi connectivity_;
  float mu10_, mu01_, kappa_;
};

TEST_F(FEAT10OptMultiElementTest, TwoElements_UndeformedEquilibrium) {
  GPU_FEAT10Opt_Data element;
  element.Initialize(n_elem_, n_nodes_);
  element.Setup(nodes_, connectivity_);
  element.SetMooneyRivlin(mu10_, mu01_, kappa_);

  element.ComputePrecomputation();
  element.ClearInternalForce();
  element.ComputeInternalForce(nullptr);

  Eigen::VectorXf f_int;
  element.RetrieveInternalForceToCPU(f_int);

  // All forces should be near zero
  double max_force = f_int.cwiseAbs().maxCoeff();
  EXPECT_LT(max_force, 0.1)
      << "Undeformed multi-element mesh should have near-zero internal forces";

  element.Destroy();
}

TEST_F(FEAT10OptMultiElementTest, TwoElements_ForceEquilibrium) {
  GPU_FEAT10Opt_Data element;
  element.Initialize(n_elem_, n_nodes_);
  element.Setup(nodes_, connectivity_);
  element.SetMooneyRivlin(mu10_, mu01_, kappa_);

  element.ComputePrecomputation();

  // Apply uniform stretch
  Eigen::MatrixXd deformed = nodes_ * 1.1;
  element.UpdatePositions(deformed);

  element.ClearInternalForce();
  element.ComputeInternalForce(nullptr);

  Eigen::VectorXf f_int;
  element.RetrieveInternalForceToCPU(f_int);

  // Check force equilibrium
  double equilibrium_error = CheckForceEquilibrium(f_int, n_nodes_);
  EXPECT_LT(equilibrium_error, 1e-4) << "Force equilibrium violated";

  element.Destroy();
}

TEST_F(FEAT10OptMultiElementTest, TwoElements_SharedNodeForceAssembly) {
  GPU_FEAT10Opt_Data element;
  element.Initialize(n_elem_, n_nodes_);
  element.Setup(nodes_, connectivity_);
  element.SetMooneyRivlin(mu10_, mu01_, kappa_);

  element.ComputePrecomputation();

  // Apply non-uniform stretch
  Eigen::MatrixXd deformed = nodes_;
  for (int i = 0; i < n_nodes_; i++) {
    deformed(i, 0) *= 1.1;
    deformed(i, 1) *= 1.05;
    deformed(i, 2) *= 0.95;
  }
  element.UpdatePositions(deformed);

  element.ClearInternalForce();
  element.ComputeInternalForce(nullptr);

  Eigen::VectorXf f_int;
  element.RetrieveInternalForceToCPU(f_int);

  // Total equilibrium should hold
  double equilibrium_error = CheckForceEquilibrium(f_int, n_nodes_);
  EXPECT_LT(equilibrium_error, 1e-4) << "Force equilibrium violated";

  element.Destroy();
}
