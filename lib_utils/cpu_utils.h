#pragma once

#include <Eigen/Dense>

/**
 * Utility functions for ANCF (Absolute Nodal Coordinate Formulation)
 * calculations
 */
namespace ANCFCPUUtils {

/**
 * Construct the B12 matrix and compute its inverse transpose
 * @param L Length parameter
 * @param W Width parameter
 * @param H Height parameter
 * @param B_inv_out Output matrix (inverse transpose of B)
 */
void B12_matrix(double L, double W, double H, Eigen::MatrixXd &B_inv_out,
                int n_shape);

/**
 * Generate beam coordinates for multiple beams
 * @param n_beam Number of beams
 * @param x12 Output x coordinates
 * @param y12 Output y coordinates
 * @param z12 Output z coordinates
 */
void generate_beam_coordinates(int n_beam, Eigen::VectorXd &x12,
                               Eigen::VectorXd &y12, Eigen::VectorXd &z12);

/**
 * Calculate offset indices for beam elements
 * @param n_beam Number of beams
 * @param offset_start Output start indices
 * @param offset_end Output end indices
 */
void calculate_offsets(int n_beam, Eigen::VectorXi &offset_start,
                       Eigen::VectorXi &offset_end);

} // namespace ANCFCPUUtils