#pragma once

#include <Eigen/Dense>

/**
 * Gauss-Legendre quadrature points and weights for ANCF calculations
 */
namespace Quadrature
{

    // Quadrature sizes as constexpr constants
    constexpr int N_QP_6 = 6; // 6-point quadrature
    constexpr int N_QP_3 = 3; // 3-point quadrature
    constexpr int N_QP_2 = 2; // 2-point quadrature

    // Total number of quadrature points
    constexpr int N_TOTAL_QP = N_QP_3 * N_QP_2 * N_QP_2;

    // Number of elements for shape functions
    constexpr int N_SHAPE = 8;

    // 6-point Gauss-Legendre quadrature (symmetric)
    const Eigen::VectorXd gauss_xi_m =
        (Eigen::VectorXd(N_QP_6) << -0.93246951420315202, -0.66120938646626451,
         -0.23861918608319691, 0.23861918608319691, 0.66120938646626451,
         0.93246951420315202)
            .finished();

    const Eigen::VectorXd weight_xi_m =
        (Eigen::VectorXd(N_QP_6) << 0.17132449237917034, 0.36076157304813861,
         0.46791393457269104, 0.46791393457269104, 0.36076157304813861, 0.17132449237917034)
            .finished();

    // 3-point Gauss-Legendre quadrature
    const Eigen::VectorXd gauss_xi =
        (Eigen::VectorXd(N_QP_3) << -0.77459666924148340, 0.00000000000000000, 0.77459666924148340)
            .finished();

    const Eigen::VectorXd weight_xi =
        (Eigen::VectorXd(N_QP_3) << 0.55555555555555556, 0.88888888888888889,
         0.55555555555555556)
            .finished();

    // 2-point Gauss-Legendre quadrature (for eta and zeta directions)
    const Eigen::VectorXd gauss_eta =
        (Eigen::VectorXd(N_QP_2) << -0.57735026918962576, 0.57735026918962576)
            .finished();

    const Eigen::VectorXd weight_eta =
        (Eigen::VectorXd(N_QP_2) << 1.00000000000000000, 1.00000000000000000).finished();

    const Eigen::VectorXd gauss_zeta =
        (Eigen::VectorXd(N_QP_2) << -0.57735026918962576, 0.57735026918962576)
            .finished();

    const Eigen::VectorXd weight_zeta =
        (Eigen::VectorXd(N_QP_2) << 1.00000000000000000, 1.00000000000000000).finished();

} // namespace Quadrature