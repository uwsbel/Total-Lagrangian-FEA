#include "cpu_utils.h"

namespace ANCFCPUUtils {

void B12_matrix(double L, double W, double H, Eigen::MatrixXd &B_inv_out,
                int n_shape) {
  // Reference coordinates of points P1 and P2
  double u1 = -L / 2.0;
  double u2 = L / 2.0;
  double v = 0.0;
  double w = 0.0;

  // Create an Eigen matrix
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_shape, n_shape);

  // Row 0: Basis function at u1 (b1)
  B(0, 0) = 1.0;
  B(0, 1) = u1;
  B(0, 2) = v;
  B(0, 3) = w;
  B(0, 4) = u1 * v;
  B(0, 5) = u1 * w;
  B(0, 6) = u1 * u1;
  B(0, 7) = u1 * u1 * u1;

  // Row 1: db_du at u1
  B(1, 1) = 1.0;
  B(1, 4) = v;
  B(1, 5) = w;
  B(1, 6) = 2.0 * u1;
  B(1, 7) = 3.0 * u1 * u1;

  // Row 2: db_dv at u1
  B(2, 2) = 1.0;
  B(2, 4) = u1;

  // Row 3: db_dw at u1
  B(3, 3) = 1.0;
  B(3, 5) = u1;

  // Row 4: Basis function at u2 (b2)
  B(4, 0) = 1.0;
  B(4, 1) = u2;
  B(4, 2) = v;
  B(4, 3) = w;
  B(4, 4) = u2 * v;
  B(4, 5) = u2 * w;
  B(4, 6) = u2 * u2;
  B(4, 7) = u2 * u2 * u2;

  // Row 5: db_du at u2
  B(5, 1) = 1.0;
  B(5, 4) = v;
  B(5, 5) = w;
  B(5, 6) = 2.0 * u2;
  B(5, 7) = 3.0 * u2 * u2;

  // Row 6: db_dv at u2
  B(6, 2) = 1.0;
  B(6, 4) = u2;

  // Row 7: db_dw at u2
  B(7, 3) = 1.0;
  B(7, 5) = u2;

  // Compute inverse of transposed B using Eigen
  B_inv_out = B.transpose().inverse();
}

void generate_beam_coordinates(int n_beam, Eigen::VectorXd &x12,
                               Eigen::VectorXd &y12, Eigen::VectorXd &z12) {
  // Initial beam coordinates (first 8 nodes)
  double x_init[8] = {-1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
  double y_init[8] = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
  double z_init[8] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  // Copy initial beam (first 8 nodes)
  for (int i = 0; i < 8; i++) {
    x12(i) = x_init[i];
    y12(i) = y_init[i];
    z12(i) = z_init[i];
  }

  // Add additional beams (append new blocks for additional beams)
  for (int beam = 2; beam <= n_beam; beam++) {
    double x_offset = 2.0;
    int base_idx = 8 + (beam - 2) * 4; // Starting index for new beam nodes
    int prev_base = base_idx - 4;      // Previous beam's last 4 nodes

    // Copy last 4 nodes from previous section and modify
    for (int i = 0; i < 4; i++) {
      x12(base_idx + i) = x12(prev_base + i);
      y12(base_idx + i) = y12(prev_base + i);
      z12(base_idx + i) = z12(prev_base + i);
    }

    // Only shift the first entry of the new beam by x_offset
    x12(base_idx) += x_offset;
  }
}

void calculate_offsets(int n_beam, Eigen::VectorXi &offset_start,
                       Eigen::VectorXi &offset_end) {
  for (int i = 0; i < n_beam; i++) {
    offset_start(i) = i * 4;
    offset_end(i) = offset_start(i) + 7;
  }
}

} // namespace ANCFCPUUtils