#pragma once

#include <Eigen/Dense>

namespace ANCFCPUUtils
{

    /**
     * Construct the B matrix for ANCF3243 elements and compute its inverse transpose
     * @param L Length parameter of the element
     * @param W Width parameter of the element
     * @param H Height parameter of the element
     * @param B_inv_out Output matrix (inverse transpose of B matrix)
     * @param n_shape Number of shape functions (16 for ANCF3243)
     */
    void ANCF3243_B12_matrix(double L, double W, double H, Eigen::MatrixXd &B_inv_out,
                             int n_shape);

    /**
     * Generate 3D beam element coordinates for multiple ANCF3243 elements
     * @param n_beam Number of 3D elements
     * @param x12 Output x coordinates for all nodes
     * @param y12 Output y coordinates for all nodes
     * @param z12 Output z coordinates for all nodes
     */
    void ANCF3243_generate_beam_coordinates(int n_beam, Eigen::VectorXd &x12,
                                            Eigen::VectorXd &y12, Eigen::VectorXd &z12);

    /**
     * Calculate offset indices for ANCF3243 elements
     * @param n_beam Number of 3D elements
     * @param offset_start Output start indices for each element's nodes
     * @param offset_end Output end indices for each element's nodes
     */
    void ANCF3243_calculate_offsets(int n_beam, Eigen::VectorXi &offset_start,
                                    Eigen::VectorXi &offset_end);

    /**
     * Construct the B matrix for ANCF3443 elements and compute its inverse transpose
     * @param L Length parameter of the element
     * @param W Width parameter of the element
     * @param H Height parameter of the element
     * @param B_inv_out Output matrix (inverse transpose of B matrix)
     * @param n_shape Number of shape functions (16 for ANCF3443)
     */
    void ANCF3443_B12_matrix(double L, double W, double H, Eigen::MatrixXd &B_inv_out,
                             int n_shape);

    /**
     * Generate 3D element coordinates for multiple ANCF3443 elements
     * @param n_beam Number of 3D elements
     * @param x12 Output x coordinates for all nodes
     * @param y12 Output y coordinates for all nodes
     * @param z12 Output z coordinates for all nodes
     */
    void ANCF3443_generate_beam_coordinates(int n_beam, Eigen::VectorXd &x12,
                                            Eigen::VectorXd &y12, Eigen::VectorXd &z12, Eigen::MatrixXi &element_connectivity);

    /**
     * Calculate offset indices for ANCF3443 shell elements
     * @param n_beam Number of 3D elements
     * @param offset_start Output start indices for each element's nodes
     * @param offset_end Output end indices for each element's nodes
     */
    void ANCF3443_calculate_offsets(int n_beam, Eigen::VectorXi &offset_start,
                                    Eigen::VectorXi &offset_end);

} // namespace ANCFCPUUtils