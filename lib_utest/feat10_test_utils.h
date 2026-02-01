/*
 * Shared CPU reference implementation for FEAT10/FEAT10Opt unit tests.
 *
 * This header provides CPU implementations of:
 * - T10 shape functions and gradients
 * - Jacobian and inverse Jacobian computation
 * - Deformation gradient (F) computation
 * - Mooney-Rivlin stress (P) computation
 * - Internal force computation
 *
 * These functions serve as ground truth for validating GPU implementations.
 */

#ifndef LIB_UTEST_FEAT10_TEST_UTILS_H_
#define LIB_UTEST_FEAT10_TEST_UTILS_H_

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace feat10_test {

// ===========================================================================
// Shape Functions
// ===========================================================================

/**
 * Compute T10 shape functions at given parametric coordinates.
 * @param xi   Parametric coordinate (L2 in barycentric)
 * @param eta  Parametric coordinate (L3 in barycentric)
 * @param zeta Parametric coordinate (L4 in barycentric)
 * @param N    Output array of 10 shape function values
 */
inline void ComputeShapeFunctions(double xi, double eta, double zeta,
                                  double N[10]) {
  // Compute barycentric coordinates
  double L1   = 1.0 - xi - eta - zeta;
  double L2   = xi;
  double L3   = eta;
  double L4   = zeta;
  double L[4] = {L1, L2, L3, L4};

  // Corner nodes (0-3): N_i = L_i * (2*L_i - 1)
  for (int i = 0; i < 4; i++) {
    N[i] = L[i] * (2.0 * L[i] - 1.0);
  }

  // Edge nodes (4-9): N_k = 4 * L_i * L_j
  // Edge connectivity: [(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]
  int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
  for (int k = 0; k < 6; k++) {
    int i    = edges[k][0];
    int j    = edges[k][1];
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
inline void ComputeShapeFunctionGradients(double xi, double eta, double zeta,
                                          double dN_dxi[10][3]) {
  // Compute barycentric coordinates
  double L1   = 1.0 - xi - eta - zeta;
  double L2   = xi;
  double L3   = eta;
  double L4   = zeta;
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

// ===========================================================================
// Jacobian Computation
// ===========================================================================

/**
 * Compute Jacobian matrix J = X^T * dN_dxi.
 * @param X_elem  Node coordinates (10 x 3)
 * @param dN_dxi  Shape function gradients in parametric coords (10 x 3)
 * @param J       Output Jacobian matrix (3 x 3)
 */
inline void ComputeJacobian(const double X_elem[10][3],
                            const double dN_dxi[10][3], double J[3][3]) {
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
inline void Solve3x3(const double A[3][3], const double b[3], double x[3]) {
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
    int pivot_row  = k;
    double max_val = std::abs(aug[k][k]);
    for (int i = k + 1; i < 3; i++) {
      if (std::abs(aug[i][k]) > max_val) {
        max_val   = std::abs(aug[i][k]);
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
inline double ComputeGradN_Physical(const double X_elem[10][3],
                                    const double dN_dxi[10][3],
                                    double grad_N[10][3]) {
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
 * Compute inverse Jacobian at a quadrature point using cofactor method.
 * This matches the GPU kernel's computation for direct comparison.
 * @param X_ref   Reference node coordinates (10 x 3)
 * @param xi      Parametric coordinate (L2 in barycentric)
 * @param eta     Parametric coordinate (L3 in barycentric)
 * @param zeta    Parametric coordinate (L4 in barycentric)
 * @param Jinv    Output inverse Jacobian (3 x 3)
 * @param detJ    Output determinant of Jacobian (optional, can be nullptr)
 */
inline void ComputeJacobianInverse(const double X_ref[10][3], double xi,
                                   double eta, double zeta, double Jinv[3][3],
                                   double* detJ_out = nullptr) {
  // Compute shape function gradients in parametric coordinates
  double dN_dxi[10][3];
  ComputeShapeFunctionGradients(xi, eta, zeta, dN_dxi);

  // Compute Jacobian J = sum_a (X_a * dN_a/dxi)
  double J[3][3];
  ComputeJacobian(X_ref, dN_dxi, J);

  // Compute determinant
  double detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

  double invDetJ = 1.0 / detJ;

  // Compute inverse using cofactor/adjugate method (same as GPU kernel)
  Jinv[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * invDetJ;
  Jinv[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) * invDetJ;
  Jinv[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * invDetJ;
  Jinv[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) * invDetJ;
  Jinv[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * invDetJ;
  Jinv[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) * invDetJ;
  Jinv[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * invDetJ;
  Jinv[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) * invDetJ;
  Jinv[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * invDetJ;

  if (detJ_out) {
    *detJ_out = detJ;
  }
}

// ===========================================================================
// Deformation Gradient
// ===========================================================================

/**
 * Compute deformation gradient F from current positions and reference
 * gradients. F[i][j] = sum_a(x_nodes[a][i] * grad_N[a][j])
 */
inline void ComputeF(const double x_nodes[10][3], const double grad_N[10][3],
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

// ===========================================================================
// Matrix Utilities
// ===========================================================================

/**
 * Compute determinant of 3x3 matrix.
 */
inline double Det3x3(const double A[3][3]) {
  return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
         A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
         A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

/**
 * Compute inverse transpose of 3x3 matrix.
 */
inline void InvT3x3(const double A[3][3], double detA, double invT[3][3]) {
  const double eps = 1e-12;
  double safe_det =
      (std::abs(detA) < eps) ? ((detA >= 0.0) ? eps : -eps) : detA;
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

// ===========================================================================
// Stress Computation
// ===========================================================================

/**
 * Compute first Piola-Kirchhoff stress P for Mooney-Rivlin material.
 * Matches GPU implementation in MooneyRivlin.cuh:45-111.
 */
inline void ComputeP_MooneyRivlin(const double F[3][3], double mu10,
                                  double mu01, double kappa, double P[3][3]) {
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
  double I2   = 0.5 * (I1 * I1 - trC2);

  // J = det(F)
  double J = Det3x3(F);

  // F^{-T}
  double FinvT[3][3];
  InvT3x3(F, J, FinvT);

  // J^{-2/3} and J^{-4/3}
  double J13  = std::cbrt(J);
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
      P[i][j]      = t1 * term1 + t2 * term2 + t3 * term3;
    }
  }
}

// ===========================================================================
// Internal Force Computation
// ===========================================================================

/**
 * Compute internal force for a single node at a single quadrature point.
 * f_contribution[i] = sum_j(P[i][j] * grad_N[j]) * detJ * wq
 */
inline void ComputeInternalForceContribution(const double P[3][3],
                                             const double grad_N[3],
                                             double detJ, double wq,
                                             double f_contrib[3]) {
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
inline void ComputeInternalForce_CPU(
    const double X_ref[10][3],  // Reference node positions
    const double x_cur[10][3],  // Current node positions
    double mu10, double mu01, double kappa, const Eigen::VectorXd& qp_x,
    const Eigen::VectorXd& qp_y, const Eigen::VectorXd& qp_z,
    const Eigen::VectorXd& qp_weights, double f_int[10][3]) {
  // Initialize forces to zero
  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {
      f_int[a][i] = 0.0;
    }
  }

  // Loop over quadrature points
  int n_qp = qp_x.size();
  for (int qp = 0; qp < n_qp; qp++) {
    double xi   = qp_x(qp);
    double eta  = qp_y(qp);
    double zeta = qp_z(qp);
    double wq   = qp_weights(qp);

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

// ===========================================================================
// Deformation Utilities
// ===========================================================================

/**
 * Apply affine deformation x = F * X to all nodes.
 */
inline void ApplyAffineDeformation(const double X_ref[10][3],
                                   const double F_applied[3][3],
                                   double x_cur[10][3]) {
  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {
      x_cur[a][i] = 0.0;
      for (int j = 0; j < 3; j++) {
        x_cur[a][i] += F_applied[i][j] * X_ref[a][j];
      }
    }
  }
}

/**
 * Check force equilibrium: sum of all nodal forces should be zero.
 * Returns the relative equilibrium error (|sum(f)| / max|f|).
 */
inline double CheckForceEquilibrium(const Eigen::VectorXd& f_int, int n_nodes) {
  Eigen::Vector3d sum_f = Eigen::Vector3d::Zero();
  double max_force      = 0.0;

  for (int node = 0; node < n_nodes; node++) {
    for (int d = 0; d < 3; d++) {
      sum_f(d) += f_int(node * 3 + d);
      max_force = std::max(max_force, std::abs(f_int(node * 3 + d)));
    }
  }

  if (max_force < 1e-12) {
    return 0.0;  // All forces are zero
  }
  return sum_f.norm() / max_force;
}

/**
 * Check force equilibrium for float vector.
 */
inline double CheckForceEquilibrium(const Eigen::VectorXf& f_int, int n_nodes) {
  Eigen::Vector3f sum_f = Eigen::Vector3f::Zero();
  float max_force       = 0.0f;

  for (int node = 0; node < n_nodes; node++) {
    for (int d = 0; d < 3; d++) {
      sum_f(d) += f_int(node * 3 + d);
      max_force = std::max(max_force, std::abs(f_int(node * 3 + d)));
    }
  }

  if (max_force < 1e-6f) {
    return 0.0;  // All forces are zero
  }
  return static_cast<double>(sum_f.norm() / max_force);
}

// ===========================================================================
// Standard Quadrature Rules
// ===========================================================================

/**
 * Setup 5-point Keast quadrature rule (used by FEAT10).
 * Degree 4 precision.
 */
inline void Setup5PointKeastQuadrature(Eigen::VectorXd& qp_x,
                                       Eigen::VectorXd& qp_y,
                                       Eigen::VectorXd& qp_z,
                                       Eigen::VectorXd& qp_weights) {
  qp_x.resize(5);
  qp_y.resize(5);
  qp_z.resize(5);
  qp_weights.resize(5);

  qp_x(0) = 0.25;
  qp_y(0) = 0.25;
  qp_z(0) = 0.25;

  double a = 0.5, b = 1.0 / 6.0;
  qp_x(1) = a;
  qp_y(1) = b;
  qp_z(1) = b;
  qp_x(2) = b;
  qp_y(2) = a;
  qp_z(2) = b;
  qp_x(3) = b;
  qp_y(3) = b;
  qp_z(3) = a;
  qp_x(4) = b;
  qp_y(4) = b;
  qp_z(4) = b;

  qp_weights(0) = -4.0 / 5.0 * (1.0 / 6.0);
  for (int i = 1; i < 5; i++) {
    qp_weights(i) = 9.0 / 20.0 * (1.0 / 6.0);
  }
}

/**
 * Setup 4-point quadrature rule (used by FEAT10Opt).
 * Degree 2 precision.
 */
inline void Setup4PointQuadrature(Eigen::VectorXd& qp_x, Eigen::VectorXd& qp_y,
                                  Eigen::VectorXd& qp_z,
                                  Eigen::VectorXd& qp_weights) {
  constexpr double a      = 0.1381966011250105;
  constexpr double b      = 0.5854101966249685;
  constexpr double weight = 1.0 / 24.0;

  qp_x.resize(4);
  qp_y.resize(4);
  qp_z.resize(4);
  qp_weights.resize(4);

  qp_x(0) = a;
  qp_y(0) = a;
  qp_z(0) = a;
  qp_x(1) = b;
  qp_y(1) = a;
  qp_z(1) = a;
  qp_x(2) = b;
  qp_y(2) = b;
  qp_z(2) = a;
  qp_x(3) = b;
  qp_y(3) = b;
  qp_z(3) = b;

  for (int i = 0; i < 4; i++) {
    qp_weights(i) = weight;
  }
}

// ===========================================================================
// Standard Test Geometry
// ===========================================================================

/**
 * Setup unit tetrahedron with T10 nodes.
 * Corners at (0,0,0), (1,0,0), (0,1,0), (0,0,1).
 */
inline void SetupUnitTetrahedron(double X_ref[10][3]) {
  // Corner nodes
  X_ref[0][0] = 0.0;
  X_ref[0][1] = 0.0;
  X_ref[0][2] = 0.0;  // Node 0
  X_ref[1][0] = 1.0;
  X_ref[1][1] = 0.0;
  X_ref[1][2] = 0.0;  // Node 1
  X_ref[2][0] = 0.0;
  X_ref[2][1] = 1.0;
  X_ref[2][2] = 0.0;  // Node 2
  X_ref[3][0] = 0.0;
  X_ref[3][1] = 0.0;
  X_ref[3][2] = 1.0;  // Node 3

  // Edge midpoints
  X_ref[4][0] = 0.5;
  X_ref[4][1] = 0.0;
  X_ref[4][2] = 0.0;  // Edge 0-1
  X_ref[5][0] = 0.5;
  X_ref[5][1] = 0.5;
  X_ref[5][2] = 0.0;  // Edge 1-2
  X_ref[6][0] = 0.0;
  X_ref[6][1] = 0.5;
  X_ref[6][2] = 0.0;  // Edge 0-2
  X_ref[7][0] = 0.0;
  X_ref[7][1] = 0.0;
  X_ref[7][2] = 0.5;  // Edge 0-3
  X_ref[8][0] = 0.5;
  X_ref[8][1] = 0.0;
  X_ref[8][2] = 0.5;  // Edge 1-3
  X_ref[9][0] = 0.0;
  X_ref[9][1] = 0.5;
  X_ref[9][2] = 0.5;  // Edge 2-3
}

// ===========================================================================
// Standard Deformation Gradients for Testing
// ===========================================================================

/**
 * Create identity deformation gradient.
 */
inline void SetIdentityF(double F[3][3]) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      F[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }
}

/**
 * Create general affine deformation (stretch + shear).
 */
inline void SetGeneralAffineF(double F[3][3]) {
  F[0][0] = 1.10;
  F[0][1] = 0.05;
  F[0][2] = 0.02;
  F[1][0] = 0.03;
  F[1][1] = 1.15;
  F[1][2] = 0.04;
  F[2][0] = 0.01;
  F[2][1] = 0.02;
  F[2][2] = 0.95;
}

/**
 * Create pure stretch deformation (diagonal F).
 */
inline void SetPureStretchF(double F[3][3], double sx, double sy, double sz) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      F[i][j] = 0.0;
    }
  }
  F[0][0] = sx;
  F[1][1] = sy;
  F[2][2] = sz;
}

/**
 * Create simple shear deformation.
 * F = I + gamma * (e_i ⊗ e_j)
 */
inline void SetSimpleShearF(double F[3][3], double gamma, int i, int j) {
  SetIdentityF(F);
  F[i][j] = gamma;
}

/**
 * Create rotation matrix about Z axis.
 */
inline void SetRotationZF(double F[3][3], double angle_rad) {
  double c = std::cos(angle_rad);
  double s = std::sin(angle_rad);

  F[0][0] = c;
  F[0][1] = -s;
  F[0][2] = 0.0;
  F[1][0] = s;
  F[1][1] = c;
  F[1][2] = 0.0;
  F[2][0] = 0.0;
  F[2][1] = 0.0;
  F[2][2] = 1.0;
}

}  // namespace feat10_test

#endif  // LIB_UTEST_FEAT10_TEST_UTILS_H_
