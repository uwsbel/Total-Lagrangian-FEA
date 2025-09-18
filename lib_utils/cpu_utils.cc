#include "cpu_utils.h"

namespace ANCFCPUUtils
{

  void ANCF3243_B12_matrix(double L, double W, double H, Eigen::MatrixXd &B_inv_out,
                           int n_shape)
  {
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

  void ANCF3443_B12_matrix(double L, double W, double H, Eigen::MatrixXd &B_inv_out,
                           int n_shape)
  {
    // Reference coordinates of the 4 corner points
    double u1 = -L / 2.0, v1 = -W / 2.0, w1 = 0.0; // Point P1
    double u2 = L / 2.0, v2 = -W / 2.0, w2 = 0.0;  // Point P2
    double u3 = L / 2.0, v3 = W / 2.0, w3 = 0.0;   // Point P3
    double u4 = -L / 2.0, v4 = W / 2.0, w4 = 0.0;  // Point P4

    // Create 16x16 B matrix
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_shape, n_shape);

    // Point P1: Row 0-3 (function value + derivatives)
    // Row 0: b_vec(u1, v1, w1)
    B(0, 0) = 1.0;
    B(0, 1) = u1;
    B(0, 2) = v1;
    B(0, 3) = w1;
    B(0, 4) = u1 * v1;
    B(0, 5) = u1 * w1;
    B(0, 6) = v1 * w1;
    B(0, 7) = u1 * v1 * w1;
    B(0, 8) = u1 * u1;
    B(0, 9) = v1 * v1;
    B(0, 10) = u1 * u1 * v1;
    B(0, 11) = u1 * v1 * v1;
    B(0, 12) = u1 * u1 * u1;
    B(0, 13) = v1 * v1 * v1;
    B(0, 14) = u1 * u1 * u1 * v1;
    B(0, 15) = u1 * v1 * v1 * v1;

    // Row 1: db_du(u1, v1, w1)
    B(1, 1) = 1.0;
    B(1, 4) = v1;
    B(1, 5) = w1;
    B(1, 7) = v1 * w1;
    B(1, 8) = 2.0 * u1;
    B(1, 10) = 2.0 * u1 * v1;
    B(1, 11) = v1 * v1;
    B(1, 12) = 3.0 * u1 * u1;
    B(1, 14) = 3.0 * u1 * u1 * v1;
    B(1, 15) = v1 * v1 * v1;

    // Row 2: db_dv(u1, v1, w1)
    B(2, 2) = 1.0;
    B(2, 4) = u1;
    B(2, 6) = w1;
    B(2, 7) = u1 * w1;
    B(2, 9) = 2.0 * v1;
    B(2, 10) = u1 * u1;
    B(2, 11) = 2.0 * u1 * v1;
    B(2, 13) = 3.0 * v1 * v1;
    B(2, 14) = u1 * u1 * u1;
    B(2, 15) = 3.0 * u1 * v1 * v1;

    // Row 3: db_dw(u1, v1, w1)
    B(3, 3) = 1.0;
    B(3, 5) = u1;
    B(3, 6) = v1;
    B(3, 7) = u1 * v1;

    // Point P2: Row 4-7
    // Row 4: b_vec(u2, v2, w2)
    B(4, 0) = 1.0;
    B(4, 1) = u2;
    B(4, 2) = v2;
    B(4, 3) = w2;
    B(4, 4) = u2 * v2;
    B(4, 5) = u2 * w2;
    B(4, 6) = v2 * w2;
    B(4, 7) = u2 * v2 * w2;
    B(4, 8) = u2 * u2;
    B(4, 9) = v2 * v2;
    B(4, 10) = u2 * u2 * v2;
    B(4, 11) = u2 * v2 * v2;
    B(4, 12) = u2 * u2 * u2;
    B(4, 13) = v2 * v2 * v2;
    B(4, 14) = u2 * u2 * u2 * v2;
    B(4, 15) = u2 * v2 * v2 * v2;

    // Row 5: db_du(u2, v2, w2)
    B(5, 1) = 1.0;
    B(5, 4) = v2;
    B(5, 5) = w2;
    B(5, 7) = v2 * w2;
    B(5, 8) = 2.0 * u2;
    B(5, 10) = 2.0 * u2 * v2;
    B(5, 11) = v2 * v2;
    B(5, 12) = 3.0 * u2 * u2;
    B(5, 14) = 3.0 * u2 * u2 * v2;
    B(5, 15) = v2 * v2 * v2;

    // Row 6: db_dv(u2, v2, w2)
    B(6, 2) = 1.0;
    B(6, 4) = u2;
    B(6, 6) = w2;
    B(6, 7) = u2 * w2;
    B(6, 9) = 2.0 * v2;
    B(6, 10) = u2 * u2;
    B(6, 11) = 2.0 * u2 * v2;
    B(6, 13) = 3.0 * v2 * v2;
    B(6, 14) = u2 * u2 * u2;
    B(6, 15) = 3.0 * u2 * v2 * v2;

    // Row 7: db_dw(u2, v2, w2)
    B(7, 3) = 1.0;
    B(7, 5) = u2;
    B(7, 6) = v2;
    B(7, 7) = u2 * v2;

    // Point P3: Row 8-11
    // Row 8: b_vec(u3, v3, w3)
    B(8, 0) = 1.0;
    B(8, 1) = u3;
    B(8, 2) = v3;
    B(8, 3) = w3;
    B(8, 4) = u3 * v3;
    B(8, 5) = u3 * w3;
    B(8, 6) = v3 * w3;
    B(8, 7) = u3 * v3 * w3;
    B(8, 8) = u3 * u3;
    B(8, 9) = v3 * v3;
    B(8, 10) = u3 * u3 * v3;
    B(8, 11) = u3 * v3 * v3;
    B(8, 12) = u3 * u3 * u3;
    B(8, 13) = v3 * v3 * v3;
    B(8, 14) = u3 * u3 * u3 * v3;
    B(8, 15) = u3 * v3 * v3 * v3;

    // Row 9: db_du(u3, v3, w3)
    B(9, 1) = 1.0;
    B(9, 4) = v3;
    B(9, 5) = w3;
    B(9, 7) = v3 * w3;
    B(9, 8) = 2.0 * u3;
    B(9, 10) = 2.0 * u3 * v3;
    B(9, 11) = v3 * v3;
    B(9, 12) = 3.0 * u3 * u3;
    B(9, 14) = 3.0 * u3 * u3 * v3;
    B(9, 15) = v3 * v3 * v3;

    // Row 10: db_dv(u3, v3, w3)
    B(10, 2) = 1.0;
    B(10, 4) = u3;
    B(10, 6) = w3;
    B(10, 7) = u3 * w3;
    B(10, 9) = 2.0 * v3;
    B(10, 10) = u3 * u3;
    B(10, 11) = 2.0 * u3 * v3;
    B(10, 13) = 3.0 * v3 * v3;
    B(10, 14) = u3 * u3 * u3;
    B(10, 15) = 3.0 * u3 * v3 * v3;

    // Row 11: db_dw(u3, v3, w3)
    B(11, 3) = 1.0;
    B(11, 5) = u3;
    B(11, 6) = v3;
    B(11, 7) = u3 * v3;

    // Point P4: Row 12-15
    // Row 12: b_vec(u4, v4, w4)
    B(12, 0) = 1.0;
    B(12, 1) = u4;
    B(12, 2) = v4;
    B(12, 3) = w4;
    B(12, 4) = u4 * v4;
    B(12, 5) = u4 * w4;
    B(12, 6) = v4 * w4;
    B(12, 7) = u4 * v4 * w4;
    B(12, 8) = u4 * u4;
    B(12, 9) = v4 * v4;
    B(12, 10) = u4 * u4 * v4;
    B(12, 11) = u4 * v4 * v4;
    B(12, 12) = u4 * u4 * u4;
    B(12, 13) = v4 * v4 * v4;
    B(12, 14) = u4 * u4 * u4 * v4;
    B(12, 15) = u4 * v4 * v4 * v4;

    // Row 13: db_du(u4, v4, w4)
    B(13, 1) = 1.0;
    B(13, 4) = v4;
    B(13, 5) = w4;
    B(13, 7) = v4 * w4;
    B(13, 8) = 2.0 * u4;
    B(13, 10) = 2.0 * u4 * v4;
    B(13, 11) = v4 * v4;
    B(13, 12) = 3.0 * u4 * u4;
    B(13, 14) = 3.0 * u4 * u4 * v4;
    B(13, 15) = v4 * v4 * v4;

    // Row 14: db_dv(u4, v4, w4)
    B(14, 2) = 1.0;
    B(14, 4) = u4;
    B(14, 6) = w4;
    B(14, 7) = u4 * w4;
    B(14, 9) = 2.0 * v4;
    B(14, 10) = u4 * u4;
    B(14, 11) = 2.0 * u4 * v4;
    B(14, 13) = 3.0 * v4 * v4;
    B(14, 14) = u4 * u4 * u4;
    B(14, 15) = 3.0 * u4 * v4 * v4;

    // Row 15: db_dw(u4, v4, w4)
    B(15, 3) = 1.0;
    B(15, 5) = u4;
    B(15, 6) = v4;
    B(15, 7) = u4 * v4;

    // Compute inverse of B matrix
    B_inv_out = B.transpose().inverse();
  }

  void ANCF3243_generate_beam_coordinates(int n_beam, Eigen::VectorXd &x12,
                                          Eigen::VectorXd &y12, Eigen::VectorXd &z12)
  {
    // Initial beam coordinates (first 8 nodes)
    double x_init[8] = {-1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
    double y_init[8] = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    double z_init[8] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    // Copy initial beam (first 8 nodes)
    for (int i = 0; i < 8; i++)
    {
      x12(i) = x_init[i];
      y12(i) = y_init[i];
      z12(i) = z_init[i];
    }

    // Add additional beams (append new blocks for additional beams)
    for (int beam = 2; beam <= n_beam; beam++)
    {
      double x_offset = 2.0;
      int base_idx = 8 + (beam - 2) * 4; // Starting index for new beam nodes
      int prev_base = base_idx - 4;      // Previous beam's last 4 nodes

      // Copy last 4 nodes from previous section and modify
      for (int i = 0; i < 4; i++)
      {
        x12(base_idx + i) = x12(prev_base + i);
        y12(base_idx + i) = y12(prev_base + i);
        z12(base_idx + i) = z12(prev_base + i);
      }

      // Only shift the first entry of the new beam by x_offset
      x12(base_idx) += x_offset;
    }
  }

  void ANCF3443_generate_beam_coordinates(int n_beam, Eigen::VectorXd &x12,
                                          Eigen::VectorXd &y12, Eigen::VectorXd &z12, Eigen::MatrixXi &element_connectivity)
  {

    // Number of nodes: 4 for the first element, 2 new for each additional element
    int n_nodes = 4 + 2 * (n_beam - 1);
    int N_dof = n_nodes * 4;

    // Resize coordinate arrays
    x12.resize(N_dof);
    y12.resize(N_dof);
    z12.resize(N_dof);

    // Example: Fill with zeros or your own geometry logic
    x12.setZero();
    y12.setZero();
    z12.setZero();

    // Set first 16 elements (4 nodes × 4 DOFs)
    x12(0) = 0.0;
    x12(1) = 1.0;
    x12(2) = 0.0;
    x12(3) = 0.0; // Node 0
    x12(4) = 2.0;
    x12(5) = 1.0;
    x12(6) = 0.0;
    x12(7) = 0.0; // Node 1
    x12(8) = 2.0;
    x12(9) = 1.0;
    x12(10) = 0.0;
    x12(11) = 0.0; // Node 2
    x12(12) = 0.0;
    x12(13) = 1.0;
    x12(14) = 0.0;
    x12(15) = 0.0; // Node 3

    y12(0) = 0.0;
    y12(1) = 0.0;
    y12(2) = 1.0;
    y12(3) = 0.0; // Node 0
    y12(4) = 0.0;
    y12(5) = 0.0;
    y12(6) = 1.0;
    y12(7) = 0.0; // Node 1
    y12(8) = 1.0;
    y12(9) = 0.0;
    y12(10) = 1.0;
    y12(11) = 0.0; // Node 2
    y12(12) = 1.0;
    y12(13) = 0.0;
    y12(14) = 1.0;
    y12(15) = 0.0; // Node 3

    z12(0) = 0.0;
    z12(1) = 0.0;
    z12(2) = 0.0;
    z12(3) = 1.0; // Node 0
    z12(4) = 0.0;
    z12(5) = 0.0;
    z12(6) = 0.0;
    z12(7) = 1.0; // Node 1
    z12(8) = 0.0;
    z12(9) = 0.0;
    z12(10) = 0.0;
    z12(11) = 1.0; // Node 2
    z12(12) = 0.0;
    z12(13) = 0.0;
    z12(14) = 0.0;
    z12(15) = 1.0; // Node 3

    for (int i = 1; i < n_beam; i++)
    {
      // each new beam has 4 additional nodes
      x12(16 + (i - 1) * 8) = 2.0 * (i + 1);
      x12(17 + (i - 1) * 8) = 1.0;
      x12(18 + (i - 1) * 8) = 0.0;
      x12(19 + (i - 1) * 8) = 0.0;
      x12(20 + (i - 1) * 8) = 2.0 * (i + 1);
      x12(21 + (i - 1) * 8) = 1.0;
      x12(22 + (i - 1) * 8) = 0.0;
      x12(23 + (i - 1) * 8) = 0.0;

      y12(16 + (i - 1) * 8) = 0.0;
      y12(17 + (i - 1) * 8) = 0.0;
      y12(18 + (i - 1) * 8) = 1.0;
      y12(19 + (i - 1) * 8) = 0.0;
      y12(20 + (i - 1) * 8) = 1.0;
      y12(21 + (i - 1) * 8) = 0.0;
      y12(22 + (i - 1) * 8) = 1.0;
      y12(23 + (i - 1) * 8) = 0.0;

      z12(16 + (i - 1) * 8) = 0.0;
      z12(17 + (i - 1) * 8) = 0.0;
      z12(18 + (i - 1) * 8) = 0.0;
      z12(19 + (i - 1) * 8) = 1.0;
      z12(20 + (i - 1) * 8) = 0.0;
      z12(21 + (i - 1) * 8) = 0.0;
      z12(22 + (i - 1) * 8) = 0.0;
      z12(23 + (i - 1) * 8) = 1.0;
    }

    // build connectivity matrix
    element_connectivity.resize(n_beam, 4);
    element_connectivity(0, 0) = 0;
    element_connectivity(0, 1) = 1;
    element_connectivity(0, 2) = 2;
    element_connectivity(0, 3) = 3;

    for (int i = 1; i < n_beam; i++)
    {
      if (i == 1)
      {
        element_connectivity(i, 0) = 1;
        element_connectivity(i, 1) = 4;
        element_connectivity(i, 2) = 5;
        element_connectivity(i, 3) = 2;
      }
      else
      {
        element_connectivity(i, 0) = 4 + (i - 2) * 2;
        element_connectivity(i, 1) = 4 + (i - 1) * 2;
        element_connectivity(i, 2) = 4 + (i - 1) * 2 + 1;
        element_connectivity(i, 3) = 5 + (i - 2) * 2;
      }
    }
  }

  void ANCF3243_calculate_offsets(int n_beam, Eigen::VectorXi &offset_start,
                                  Eigen::VectorXi &offset_end)
  {
    offset_start.resize(n_beam);
    offset_end.resize(n_beam);
    for (int i = 0; i < n_beam; i++)
    {
      offset_start(i) = i * 4;
      offset_end(i) = offset_start(i) + 7;
    }
  }

} // namespace ANCFCPUUtils