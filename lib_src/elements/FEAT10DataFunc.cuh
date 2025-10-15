#pragma once
#include <cuda_runtime.h>

#include <cmath>

#include "FEAT10Data.cuh"

// Solve 3x3 linear system: A * x = b
// A: 3x3 coefficient matrix (row-major)
// b: right-hand side vector
// x: solution vector (output)
__device__ void solve_3x3_system(double A[3][3], double b[3], double x[3]) {
  // Create augmented matrix [A|b] for Gaussian elimination
  double aug[3][4];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      aug[i][j] = A[i][j];
    }
    aug[i][3] = b[i];
  }

  // Forward elimination with partial pivoting
  for (int k = 0; k < 3; k++) {
    // Find pivot (largest element in column k)
    int pivot_row  = k;
    double max_val = fabs(aug[k][k]);
    for (int i = k + 1; i < 3; i++) {
      if (fabs(aug[i][k]) > max_val) {
        max_val   = fabs(aug[i][k]);
        pivot_row = i;
      }
    }

    // Swap rows if needed
    if (pivot_row != k) {
      for (int j = 0; j < 4; j++) {
        double temp       = aug[k][j];
        aug[k][j]         = aug[pivot_row][j];
        aug[pivot_row][j] = temp;
      }
    }

    // Check for singular matrix
    if (fabs(aug[k][k]) < 1e-14) {
      // Handle singular case - set solution to zero
      x[0] = x[1] = x[2] = 0.0;
      return;
    }

    // Eliminate column k in rows below
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

__device__ void feat10_compute_p(int elem_idx, int qp_idx,
                                 GPU_FEAT10_Data* d_data) {
  // Get current nodal positions for this element
  double x_nodes[10][3];  // 10 nodes × 3 coordinates
  for (int node = 0; node < 10; node++) {
    int global_node_idx = d_data->element_connectivity()(elem_idx, node);
    x_nodes[node][0]    = d_data->x12()(global_node_idx);  // x coordinate
    x_nodes[node][1]    = d_data->y12()(global_node_idx);  // y coordinate
    x_nodes[node][2]    = d_data->z12()(global_node_idx);  // z coordinate
  }

  // Get precomputed shape function gradients for this element and QP
  // grad_N[a][i] = ∂N_a/∂x_i (physical coordinates)
  double grad_N[10][3];
  for (int a = 0; a < 10; a++) {
    grad_N[a][0] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 0);  // ∂N_a/∂x
    grad_N[a][1] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 1);  // ∂N_a/∂y
    grad_N[a][2] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 2);  // ∂N_a/∂z
  }

  // Initialize deformation gradient F to zero
  double F[3][3] = {{0.0}};

  // Compute F = sum_a (x_nodes[a] ⊗ grad_N[a])
  // F[i][j] = sum_a (x_nodes[a][i] * grad_N[a][j])
  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {    // Current position components
      for (int j = 0; j < 3; j++) {  // Gradient components
        F[i][j] += x_nodes[a][i] * grad_N[a][j];
      }
    }
  }

  // Compute F^T * F
  double FtF[3][3] = {{0.0}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FtF[i][j] += F[k][i] * F[k][j];  // F^T[i][k] * F[k][j]
      }
    }
  }

  // Compute trace(F^T * F)
  double trFtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

  // Compute F * F^T
  double FFt[3][3] = {{0.0}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FFt[i][j] += F[i][k] * F[j][k];  // F[i][k] * F^T[k][j]
      }
    }
  }

  // Compute F * F^T * F
  double FFtF[3][3] = {{0.0}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        FFtF[i][j] += FFt[i][k] * F[k][j];
      }
    }
  }

  // Get material parameters
  double lambda = d_data->lambda();
  double mu     = d_data->mu();

  // Compute P = λ*(0.5*tr(F^T*F) - 1.5)*F + μ*(F*F^T*F - F)
  double lambda_factor = lambda * (0.5 * trFtF - 1.5);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      d_data->P(elem_idx, qp_idx)(i, j) =
          lambda_factor * F[i][j] + mu * (FFtF[i][j] - F[i][j]);
    }
  }
}
