/*
 * Unit test for FEAT10 element internal force computation.
 * Validates GPU implementation of deformation gradient (F), first Piola-Kirchhoff
 * stress (P), and internal force (f_int) against CPU reference implementation.
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

#include "lib_src/elements/FEAT10Data.cuh"
#include "lib_utils/quadrature_utils.h"

namespace {

// ===========================================================================
// CPU Reference Implementation Functions
// ===========================================================================

/**
 * Compute T10 shape functions at given parametric coordinates.
 * @param xi   Parametric coordinate (L2 in barycentric)
 * @param eta  Parametric coordinate (L3 in barycentric)
 * @param zeta Parametric coordinate (L4 in barycentric)
 * @param N    Output array of 10 shape function values
 */
void ComputeShapeFunctions(double xi, double eta, double zeta, double N[10]) {
  // Compute barycentric coordinates
  double L1 = 1.0 - xi - eta - zeta;
  double L2 = xi;
  double L3 = eta;
  double L4 = zeta;
  double L[4] = {L1, L2, L3, L4};

  // Corner nodes (0-3): N_i = L_i * (2*L_i - 1)
  for (int i = 0; i < 4; i++) {
    N[i] = L[i] * (2.0 * L[i] - 1.0);
  }

  // Edge nodes (4-9): N_k = 4 * L_i * L_j
  // Edge connectivity: [(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]
  int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
  for (int k = 0; k < 6; k++) {
    int i = edges[k][0];
    int j = edges[k][1];
    N[k + 4] = 4.0 * L[i] * L[j];
  }
}

/**
 * Compute T10 shape function gradients w.r.t. parametric coordinates.
 * @param xi     Parametric coordinate (L2 in barycentric)
 * @param eta    Parametric coordinate (L3 in barycentric)
 * @param zeta   Parametric coordinate (L4 in barycentric)
 * @param dN_dxi Output array of shape function derivatives (10 x 3)
 */
void ComputeShapeFunctionGradients(double xi, double eta, double zeta,
                                   double dN_dxi[10][3]) {
  // Compute barycentric coordinates
  double L1 = 1.0 - xi - eta - zeta;
  double L2 = xi;
  double L3 = eta;
  double L4 = zeta;
  double L[4] = {L1, L2, L3, L4};

  // Derivatives of barycentric coordinates
  double dL[4][3] = {
      {-1.0, -1.0, -1.0},  // dL1/dxi, dL1/deta, dL1/dzeta
      {1.0, 0.0, 0.0},     // dL2/dxi, dL2/deta, dL2/dzeta
      {0.0, 1.0, 0.0},     // dL3/dxi, dL3/deta, dL3/dzeta
      {0.0, 0.0, 1.0}      // dL4/dxi, dL4/deta, dL4/dzeta
  };

  // Corner nodes (0-3): dN_dxi[i, :] = (4*L[i]-1)*dL[i, :]
  for (int i = 0; i < 4; i++) {
    double factor = 4.0 * L[i] - 1.0;
    for (int j = 0; j < 3; j++) {
      dN_dxi[i][j] = factor * dL[i][j];
    }
  }

  // Edge nodes (4-9): dN_dxi[k, :] = 4*(L[i]*dL[j, :] + L[j]*dL[i, :])
  int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
  for (int k = 0; k < 6; k++) {
    int i = edges[k][0];
    int j = edges[k][1];
    for (int d = 0; d < 3; d++) {
      dN_dxi[k + 4][d] = 4.0 * (L[i] * dL[j][d] + L[j] * dL[i][d]);
    }
  }
}

/**
 * Compute Jacobian matrix J = X^T * dN_dxi.
 * @param X_elem  Node coordinates (10 x 3)
 * @param dN_dxi  Shape function gradients in parametric coords (10 x 3)
 * @param J       Output Jacobian matrix (3 x 3)
 */
void ComputeJacobian(const double X_elem[10][3], const double dN_dxi[10][3],
                     double J[3][3]) {
  // J[i][j] = sum_a(X_elem[a][i] * dN_dxi[a][j])
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      J[i][j] = 0.0;
      for (int a = 0; a < 10; a++) {
        J[i][j] += X_elem[a][i] * dN_dxi[a][j];
      }
    }
  }
}

/**
 * Solve 3x3 linear system A * x = b using Gaussian elimination.
 */
void Solve3x3(const double A[3][3], const double b[3], double x[3]) {
  // Create augmented matrix
  double aug[3][4];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      aug[i][j] = A[i][j];
    }
    aug[i][3] = b[i];
  }

  // Forward elimination with partial pivoting
  for (int k = 0; k < 3; k++) {
    int pivot_row = k;
    double max_val = std::abs(aug[k][k]);
    for (int i = k + 1; i < 3; i++) {
      if (std::abs(aug[i][k]) > max_val) {
        max_val = std::abs(aug[i][k]);
        pivot_row = i;
      }
    }

    if (pivot_row != k) {
      for (int j = 0; j < 4; j++) {
        std::swap(aug[k][j], aug[pivot_row][j]);
      }
    }

    for (int i = k + 1; i < 3; i++) {
      double factor = aug[i][k] / aug[k][k];
      for (int j = k; j < 4; j++) {
        aug[i][j] -= factor * aug[k][j];
      }
    }
  }

  // Back substitution
  x[2] = aug[2][3] / aug[2][2];
  x[1] = (aug[1][3] - aug[1][2] * x[2]) / aug[1][1];
  x[0] = (aug[0][3] - aug[0][2] * x[2] - aug[0][1] * x[1]) / aug[0][0];
}

/**
 * Compute shape function gradients in physical coordinates.
 * grad_N = J^{-T} * dN_dxi (solved as JT * grad_N = dN_dxi)
 * Also returns determinant of Jacobian.
 */
double ComputeGradN_Physical(const double X_elem[10][3],
                             const double dN_dxi[10][3], double grad_N[10][3]) {
  // Compute Jacobian
  double J[3][3];
  ComputeJacobian(X_elem, dN_dxi, J);

  // Compute determinant
  double detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

  // Compute J^T
  double JT[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      JT[i][j] = J[j][i];
    }
  }

  // Solve JT * grad_N[a] = dN_dxi[a] for each shape function
  for (int a = 0; a < 10; a++) {
    double b[3] = {dN_dxi[a][0], dN_dxi[a][1], dN_dxi[a][2]};
    double x[3];
    Solve3x3(JT, b, x);
    grad_N[a][0] = x[0];
    grad_N[a][1] = x[1];
    grad_N[a][2] = x[2];
  }

  return detJ;
}

/**
 * Compute deformation gradient F from current positions and reference gradients.
 * F[i][j] = sum_a(x_nodes[a][i] * grad_N[a][j])
 */
void ComputeF(const double x_nodes[10][3], const double grad_N[10][3],
              double F[3][3]) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      F[i][j] = 0.0;
      for (int a = 0; a < 10; a++) {
        F[i][j] += x_nodes[a][i] * grad_N[a][j];
      }
    }
  }
}

/**
 * Compute determinant of 3x3 matrix.
 */
double Det3x3(const double A[3][3]) {
  return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
         A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
         A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

/**
 * Compute inverse transpose of 3x3 matrix.
 */
void InvT3x3(const double A[3][3], double detA, double invT[3][3]) {
  const double eps = 1e-12;
  double safe_det = (std::abs(detA) < eps) ? ((detA >= 0.0) ? eps : -eps) : detA;
  double inv_det = 1.0 / safe_det;

  invT[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_det;
  invT[0][1] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * inv_det;
  invT[0][2] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_det;

  invT[1][0] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * inv_det;
  invT[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_det;
  invT[1][2] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * inv_det;

  invT[2][0] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_det;
  invT[2][1] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * inv_det;
  invT[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_det;
}

/**
 * Compute first Piola-Kirchhoff stress P for Mooney-Rivlin material.
 * Matches GPU implementation in MooneyRivlin.cuh:45-111.
 */
void ComputeP_MooneyRivlin(const double F[3][3], double mu10, double mu01,
                           double kappa, double P[3][3]) {
  // C = F^T * F
  double C[3][3] = {{0.0}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        C[i][j] += F[k][i] * F[k][j];
      }
    }
  }

  // I1 = tr(C)
  double I1 = C[0][0] + C[1][1] + C[2][2];

  // C^2
  double C2[3][3] = {{0.0}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        C2[i][j] += C[i][k] * C[k][j];
      }
    }
  }

  // I2 = 0.5 * (I1^2 - tr(C^2))
  double trC2 = C2[0][0] + C2[1][1] + C2[2][2];
  double I2 = 0.5 * (I1 * I1 - trC2);

  // J = det(F)
  double J = Det3x3(F);

  // F^{-T}
  double FinvT[3][3];
  InvT3x3(F, J, FinvT);

  // J^{-2/3} and J^{-4/3}
  double J13 = std::cbrt(J);
  double Jm23 = 1.0 / (J13 * J13);
  double Jm43 = Jm23 * Jm23;

  // FC = F * C
  double FC[3][3] = {{0.0}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FC[i][j] += F[i][k] * C[k][j];
      }
    }
  }

  // Stress coefficients
  double t1 = 2.0 * mu10 * Jm23;
  double t2 = 2.0 * mu01 * Jm43;
  double t3 = kappa * (J - 1.0) * J;

  // P = t1*(F - I1/3*FinvT) + t2*(I1*F - FC - 2*I2/3*FinvT) + t3*FinvT
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double term1 = F[i][j] - (I1 / 3.0) * FinvT[i][j];
      double term2 = I1 * F[i][j] - FC[i][j] - (2.0 * I2 / 3.0) * FinvT[i][j];
      double term3 = FinvT[i][j];
      P[i][j] = t1 * term1 + t2 * term2 + t3 * term3;
    }
  }
}

/**
 * Compute internal force for a single node at a single quadrature point.
 * f_contribution[i] = sum_j(P[i][j] * grad_N[j]) * detJ * wq
 */
void ComputeInternalForceContribution(const double P[3][3],
                                      const double grad_N[3], double detJ,
                                      double wq, double f_contrib[3]) {
  double dV = detJ * wq;
  for (int i = 0; i < 3; i++) {
    f_contrib[i] = 0.0;
    for (int j = 0; j < 3; j++) {
      f_contrib[i] += P[i][j] * grad_N[j];
    }
    f_contrib[i] *= dV;
  }
}

/**
 * Compute total internal force for all nodes using CPU reference.
 */
void ComputeInternalForce_CPU(
    const double X_ref[10][3],   // Reference node positions
    const double x_cur[10][3],   // Current node positions
    double mu10, double mu01, double kappa,
    const Eigen::VectorXd& qp_x, const Eigen::VectorXd& qp_y,
    const Eigen::VectorXd& qp_z, const Eigen::VectorXd& qp_weights,
    double f_int[10][3]) {
  // Initialize forces to zero
  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {
      f_int[a][i] = 0.0;
    }
  }

  // Loop over quadrature points
  int n_qp = qp_x.size();
  for (int qp = 0; qp < n_qp; qp++) {
    double xi = qp_x(qp);
    double eta = qp_y(qp);
    double zeta = qp_z(qp);
    double wq = qp_weights(qp);

    // Compute shape function gradients in parametric coordinates
    double dN_dxi[10][3];
    ComputeShapeFunctionGradients(xi, eta, zeta, dN_dxi);

    // Compute physical gradients using REFERENCE configuration
    double grad_N[10][3];
    double detJ = ComputeGradN_Physical(X_ref, dN_dxi, grad_N);

    // Compute deformation gradient F using CURRENT positions
    double F[3][3];
    ComputeF(x_cur, grad_N, F);

    // Compute Piola stress P
    double P[3][3];
    ComputeP_MooneyRivlin(F, mu10, mu01, kappa, P);

    // Accumulate force for each node
    for (int a = 0; a < 10; a++) {
      double f_contrib[3];
      double grad_Na[3] = {grad_N[a][0], grad_N[a][1], grad_N[a][2]};
      ComputeInternalForceContribution(P, grad_Na, detJ, wq, f_contrib);
      for (int i = 0; i < 3; i++) {
        f_int[a][i] += f_contrib[i];
      }
    }
  }
}

}  // namespace

// ===========================================================================
// Test Fixture
// ===========================================================================

class FEAT10InternalForceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Unit tetrahedron with corners at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    // Node ordering for T10:
    //   0-3: corner nodes
    //   4-9: edge midpoints
    // Edge connectivity: [(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]

    // Corner nodes
    X_ref_[0][0] = 0.0; X_ref_[0][1] = 0.0; X_ref_[0][2] = 0.0;  // Node 0
    X_ref_[1][0] = 1.0; X_ref_[1][1] = 0.0; X_ref_[1][2] = 0.0;  // Node 1
    X_ref_[2][0] = 0.0; X_ref_[2][1] = 1.0; X_ref_[2][2] = 0.0;  // Node 2
    X_ref_[3][0] = 0.0; X_ref_[3][1] = 0.0; X_ref_[3][2] = 1.0;  // Node 3

    // Edge midpoints
    X_ref_[4][0] = 0.5; X_ref_[4][1] = 0.0; X_ref_[4][2] = 0.0;  // Edge 0-1
    X_ref_[5][0] = 0.5; X_ref_[5][1] = 0.5; X_ref_[5][2] = 0.0;  // Edge 1-2
    X_ref_[6][0] = 0.0; X_ref_[6][1] = 0.5; X_ref_[6][2] = 0.0;  // Edge 0-2
    X_ref_[7][0] = 0.0; X_ref_[7][1] = 0.0; X_ref_[7][2] = 0.5;  // Edge 0-3
    X_ref_[8][0] = 0.5; X_ref_[8][1] = 0.0; X_ref_[8][2] = 0.5;  // Edge 1-3
    X_ref_[9][0] = 0.0; X_ref_[9][1] = 0.5; X_ref_[9][2] = 0.5;  // Edge 2-3

    // Material parameters: Mooney-Rivlin
    mu10_ = 80769.23;
    mu01_ = 20192.31;
    kappa_ = 400000.0;

    // Applied deformation gradient (affine)
    F_applied_[0][0] = 1.10; F_applied_[0][1] = 0.05; F_applied_[0][2] = 0.02;
    F_applied_[1][0] = 0.03; F_applied_[1][1] = 1.15; F_applied_[1][2] = 0.04;
    F_applied_[2][0] = 0.01; F_applied_[2][1] = 0.02; F_applied_[2][2] = 0.95;

    // Setup quadrature points (5-point Keast rule)
    qp_x_.resize(5);
    qp_y_.resize(5);
    qp_z_.resize(5);
    qp_weights_.resize(5);

    qp_x_(0) = 0.25; qp_y_(0) = 0.25; qp_z_(0) = 0.25;
    double a = 0.5, b = 1.0 / 6.0;
    qp_x_(1) = a;  qp_y_(1) = b;  qp_z_(1) = b;
    qp_x_(2) = b;  qp_y_(2) = a;  qp_z_(2) = b;
    qp_x_(3) = b;  qp_y_(3) = b;  qp_z_(3) = a;
    qp_x_(4) = b;  qp_y_(4) = b;  qp_z_(4) = b;

    qp_weights_(0) = -4.0 / 5.0 * (1.0 / 6.0);
    for (int i = 1; i < 5; i++) {
      qp_weights_(i) = 9.0 / 20.0 * (1.0 / 6.0);
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

  // Setup GPU element data structure
  GPU_FEAT10_Data* SetupGPUElement(const double x_nodes[10][3]) {
    int n_elem = 1;
    int n_nodes = 10;

    GPU_FEAT10_Data* element = new GPU_FEAT10_Data(n_elem, n_nodes);
    element->Initialize();

    // Setup connectivity (single element with nodes 0-9)
    Eigen::MatrixXi connectivity(1, 10);
    for (int i = 0; i < 10; i++) {
      connectivity(0, i) = i;
    }

    // Setup node positions
    Eigen::VectorXd h_x12(10), h_y12(10), h_z12(10);
    for (int i = 0; i < 10; i++) {
      h_x12(i) = x_nodes[i][0];
      h_y12(i) = x_nodes[i][1];
      h_z12(i) = x_nodes[i][2];
    }

    element->Setup(qp_x_, qp_y_, qp_z_, qp_weights_, h_x12, h_y12, h_z12,
                   connectivity);
    element->SetMooneyRivlin(mu10_, mu01_, kappa_);

    return element;
  }

  double X_ref_[10][3];
  double F_applied_[3][3];
  double mu10_, mu01_, kappa_;
  Eigen::VectorXd qp_x_, qp_y_, qp_z_, qp_weights_;
};

// ===========================================================================
// Test Cases
// ===========================================================================

TEST_F(FEAT10InternalForceTest, UndeformedConfig_FIsIdentity) {
  // In undeformed configuration, F should be identity and f_int should be zero

  // Setup GPU element with reference (undeformed) positions
  GPU_FEAT10_Data* element = SetupGPUElement(X_ref_);

  // Compute reference gradients and P on GPU
  element->CalcDnDuPre();
  element->CalcP();

  // Retrieve F from GPU
  std::vector<std::vector<Eigen::MatrixXd>> F_gpu;
  element->RetrieveDeformationGradientToCPU(F_gpu);

  // Check F = I at all quadrature points
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  for (int qp = 0; qp < 5; qp++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        EXPECT_NEAR(F_gpu[0][qp](i, j), I(i, j), 1e-10)
            << "F mismatch at QP " << qp << " (" << i << "," << j << ")";
      }
    }
  }

  // Compute internal force on GPU
  element->CalcInternalForce();

  // Retrieve internal force
  Eigen::VectorXd f_int_gpu;
  element->RetrieveInternalForceToCPU(f_int_gpu);

  // Check f_int = 0 (stress-free reference state)
  for (int i = 0; i < 30; i++) {
    EXPECT_NEAR(f_int_gpu(i), 0.0, 1e-8)
        << "f_int should be zero in undeformed config at DOF " << i;
  }

  element->Destroy();
  delete element;
}

TEST_F(FEAT10InternalForceTest, DeformationGradient_GPUvsCPU) {
  // Apply affine deformation and verify F matches at all quadrature points

  double x_cur[10][3];
  ApplyAffineDeformation(x_cur);

  // Setup GPU element with REFERENCE positions first
  GPU_FEAT10_Data* element = SetupGPUElement(X_ref_);

  // Compute reference gradients using REFERENCE configuration
  element->CalcDnDuPre();

  // Update to current (deformed) positions
  Eigen::VectorXd h_x12(10), h_y12(10), h_z12(10);
  for (int i = 0; i < 10; i++) {
    h_x12(i) = x_cur[i][0];
    h_y12(i) = x_cur[i][1];
    h_z12(i) = x_cur[i][2];
  }
  element->UpdatePositions(h_x12, h_y12, h_z12);

  // Compute P (this also computes F internally)
  element->CalcP();

  // Retrieve F from GPU
  std::vector<std::vector<Eigen::MatrixXd>> F_gpu;
  element->RetrieveDeformationGradientToCPU(F_gpu);

  // Compute F using CPU reference
  for (int qp = 0; qp < 5; qp++) {
    double xi = qp_x_(qp);
    double eta = qp_y_(qp);
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
        EXPECT_NEAR(F_gpu[0][qp](i, j), F_cpu[i][j], 1e-10)
            << "F mismatch at QP " << qp << " (" << i << "," << j << ")";
      }
    }
  }

  element->Destroy();
  delete element;
}

TEST_F(FEAT10InternalForceTest, PiolaStress_GPUvsCPU) {
  // Apply affine deformation and verify P matches at all quadrature points

  double x_cur[10][3];
  ApplyAffineDeformation(x_cur);

  // Setup GPU element - need to use reference config for grad_N
  // The GPU computes grad_N from current positions, so we need a workaround
  // Actually, looking at the GPU code, CalcDnDuPre computes grad_N using current
  // positions as reference. For this test, we need to:
  // 1. Setup with reference positions
  // 2. Compute grad_N (CalcDnDuPre)
  // 3. Update positions to current
  // 4. Compute P (CalcP)

  // Setup GPU element with reference positions first
  GPU_FEAT10_Data* element = SetupGPUElement(X_ref_);
  element->CalcDnDuPre();  // Compute grad_N using reference config

  // Update to current positions
  Eigen::VectorXd h_x12(10), h_y12(10), h_z12(10);
  for (int i = 0; i < 10; i++) {
    h_x12(i) = x_cur[i][0];
    h_y12(i) = x_cur[i][1];
    h_z12(i) = x_cur[i][2];
  }
  element->UpdatePositions(h_x12, h_y12, h_z12);

  // Compute P on GPU
  element->CalcP();

  // Retrieve P from GPU
  std::vector<std::vector<Eigen::MatrixXd>> P_gpu;
  element->RetrievePFromFToCPU(P_gpu);

  // Compute P using CPU reference
  for (int qp = 0; qp < 5; qp++) {
    double xi = qp_x_(qp);
    double eta = qp_y_(qp);
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
        EXPECT_NEAR(P_gpu[0][qp](i, j), P_cpu[i][j], 1e-8)
            << "P mismatch at QP " << qp << " (" << i << "," << j << ")";
      }
    }
  }

  element->Destroy();
  delete element;
}

TEST_F(FEAT10InternalForceTest, InternalForce_GPUvsCPU) {
  // Apply affine deformation and verify internal force matches

  double x_cur[10][3];
  ApplyAffineDeformation(x_cur);

  // Setup GPU element with reference positions first
  GPU_FEAT10_Data* element = SetupGPUElement(X_ref_);
  element->CalcDnDuPre();  // Compute grad_N using reference config

  // Update to current positions
  Eigen::VectorXd h_x12(10), h_y12(10), h_z12(10);
  for (int i = 0; i < 10; i++) {
    h_x12(i) = x_cur[i][0];
    h_y12(i) = x_cur[i][1];
    h_z12(i) = x_cur[i][2];
  }
  element->UpdatePositions(h_x12, h_y12, h_z12);

  // Compute P and internal force on GPU
  element->CalcP();
  element->CalcInternalForce();

  // Retrieve internal force from GPU
  Eigen::VectorXd f_int_gpu;
  element->RetrieveInternalForceToCPU(f_int_gpu);

  // Compute internal force using CPU reference
  double f_int_cpu[10][3];
  ComputeInternalForce_CPU(X_ref_, x_cur, mu10_, mu01_, kappa_,
                           qp_x_, qp_y_, qp_z_, qp_weights_, f_int_cpu);

  // Compare CPU vs GPU
  std::cout << "\nInternal force comparison (GPU vs CPU):" << std::endl;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  int max_error_dof = -1;

  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {
      int dof = a * 3 + i;
      double gpu_val = f_int_gpu(dof);
      double cpu_val = f_int_cpu[a][i];
      double abs_diff = std::abs(gpu_val - cpu_val);

      if (abs_diff > max_abs_error) {
        max_abs_error = abs_diff;
        max_error_dof = dof;
      }

      // For near-zero values, use absolute tolerance
      // For larger values, use relative tolerance
      double scale = std::max(std::abs(cpu_val), std::abs(gpu_val));
      if (scale > 1e-6) {
        double rel_error = abs_diff / scale;
        if (rel_error > max_rel_error) {
          max_rel_error = rel_error;
        }
        EXPECT_LT(rel_error, 1e-8)
            << "f_int mismatch at node " << a << " dof " << i
            << " (DOF " << dof << "): GPU=" << gpu_val << ", CPU=" << cpu_val;
      } else {
        // Near-zero: just check absolute difference
        EXPECT_NEAR(gpu_val, cpu_val, 1e-8)
            << "f_int mismatch at node " << a << " dof " << i
            << " (DOF " << dof << "): GPU=" << gpu_val << ", CPU=" << cpu_val;
      }
    }
  }

  std::cout << "Max absolute error: " << max_abs_error << " at DOF "
            << max_error_dof << std::endl;
  std::cout << "Max relative error (for non-zero values): " << max_rel_error
            << std::endl;

  // Print first few force values for debugging
  std::cout << "\nFirst 5 nodes internal force (GPU vs CPU):" << std::endl;
  for (int a = 0; a < 5; a++) {
    std::cout << "  Node " << a << ": GPU=["
              << f_int_gpu(a*3) << ", " << f_int_gpu(a*3+1) << ", "
              << f_int_gpu(a*3+2) << "], CPU=["
              << f_int_cpu[a][0] << ", " << f_int_cpu[a][1] << ", "
              << f_int_cpu[a][2] << "]" << std::endl;
  }

  element->Destroy();
  delete element;
}
