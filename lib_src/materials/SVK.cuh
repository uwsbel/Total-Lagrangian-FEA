#pragma once
#if defined(__CUDACC__)
__device__ __forceinline__ void svk_compute_P_from_trFtF_and_FFtF(
    const double F[3][3], double trFtF, const double FFtF[3][3], double lambda,
    double mu, double P_out[3][3]) {
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
      P_out[i][j] = 0.0;
    }
  }

  double lambda_factor = lambda * (0.5 * trFtF - 1.5);
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
      P_out[i][j] = lambda_factor * F[i][j] + mu * (FFtF[i][j] - F[i][j]);
    }
  }
}

__device__ __forceinline__ void svk_compute_tangent_block(
    const double Fh_i[3], const double Fh_j[3], double hij, double trE,
    double Fhj_dot_Fhi, const double FFT[3][3], double lambda, double mu,
    double dV, double Kblock[3][3]) {
#pragma unroll
  for (int d = 0; d < 3; d++) {
#pragma unroll
    for (int e = 0; e < 3; e++) {
      double delta = (d == e) ? 1.0 : 0.0;

      double A_de    = lambda * Fh_i[d] * Fh_j[e];
      double B_de    = lambda * trE * hij * delta;
      double C1_de   = mu * Fhj_dot_Fhi * delta;
      double D_de    = mu * Fh_j[d] * Fh_i[e];
      double Etrm_de = mu * hij * FFT[d][e];
      double Ftrm_de = -mu * hij * delta;

      Kblock[d][e] = (A_de + B_de + C1_de + D_de + Etrm_de + Ftrm_de) * dV;
    }
  }
}

__device__ __forceinline__ void svk_compute_P(const double F[3][3],
                                             double lambda, double mu,
                                             double P_out[3][3]) {
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
      P_out[i][j] = 0.0;
    }
  }

  double FtF[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
#pragma unroll
      for (int k = 0; k < 3; k++) {
        FtF[i][j] += F[k][i] * F[k][j];
      }
    }
  }

  double trFtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

  double FFt[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
#pragma unroll
      for (int k = 0; k < 3; k++) {
        FFt[i][j] += F[i][k] * F[j][k];
      }
    }
  }

  double FFtF[3][3] = {{0.0}};
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
#pragma unroll
      for (int k = 0; k < 3; k++) {
        FFtF[i][j] += FFt[i][k] * F[k][j];
      }
    }
  }

  svk_compute_P_from_trFtF_and_FFtF(F, trFtF, FFtF, lambda, mu, P_out);
}
#endif
