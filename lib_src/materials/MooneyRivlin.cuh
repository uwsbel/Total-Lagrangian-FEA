#pragma once

#if defined(__CUDACC__)
#include <cmath>

__device__ __forceinline__ double mr_det3x3(const double A[3][3]) {
  return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
         A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
         A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

__device__ __forceinline__ void mr_invT3x3(const double A[3][3], double detA,
                                          double invT_out[3][3]) {
  double inv_det = 1.0 / detA;

  invT_out[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_det;
  invT_out[0][1] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * inv_det;
  invT_out[0][2] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_det;

  invT_out[1][0] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * inv_det;
  invT_out[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_det;
  invT_out[1][2] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * inv_det;

  invT_out[2][0] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_det;
  invT_out[2][1] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * inv_det;
  invT_out[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_det;
}

__device__ __forceinline__ void mr_compute_P(const double F[3][3], double mu10,
                                            double mu01, double kappa,
                                            double P_out[3][3]) {
  double C[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
#pragma unroll
      for (int k = 0; k < 3; k++) {
        C[i][j] += F[k][i] * F[k][j];
      }
    }
  }

  double I1 = C[0][0] + C[1][1] + C[2][2];

  double C2[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
#pragma unroll
      for (int k = 0; k < 3; k++) {
        C2[i][j] += C[i][k] * C[k][j];
      }
    }
  }
  double trC2 = C2[0][0] + C2[1][1] + C2[2][2];
  double I2   = 0.5 * (I1 * I1 - trC2);

  double J = mr_det3x3(F);

  double FinvT[3][3];
  mr_invT3x3(F, J, FinvT);

  double J13  = cbrt(J);
  double Jm23 = 1.0 / (J13 * J13);
  double Jm43 = Jm23 * Jm23;

  double FC[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
#pragma unroll
      for (int k = 0; k < 3; k++) {
        FC[i][j] += F[i][k] * C[k][j];
      }
    }
  }

  double t1 = 2.0 * mu10 * Jm23;
  double t2 = 2.0 * mu01 * Jm43;
  double t3 = kappa * (J - 1.0) * J;

#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
      double term1 = F[i][j] - (I1 / 3.0) * FinvT[i][j];
      double term2 = I1 * F[i][j] - FC[i][j] - (2.0 * I2 / 3.0) * FinvT[i][j];
      double term3 = FinvT[i][j];
      P_out[i][j]  = t1 * term1 + t2 * term2 + t3 * term3;
    }
  }
}
#endif
