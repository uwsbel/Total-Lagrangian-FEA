#pragma once

#include <Eigen/Dense>

/**
 * Gauss-Legendre quadrature points and weights for ANCF calculations
 */
namespace Quadrature {

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
    (Eigen::VectorXd(N_QP_6) << -0.932469514203152, -0.661209386466265,
     -0.238619186083197, 0.238619186083197, 0.661209386466265,
     0.932469514203152)
        .finished();

const Eigen::VectorXd weight_xi_m =
    (Eigen::VectorXd(N_QP_6) << 0.171324492379170, 0.360761573048139,
     0.467913934572691, 0.467913934572691, 0.360761573048139, 0.171324492379170)
        .finished();

// 3-point Gauss-Legendre quadrature
const Eigen::VectorXd gauss_xi =
    (Eigen::VectorXd(N_QP_3) << -0.7745966692414834, 0.0, 0.7745966692414834)
        .finished();

const Eigen::VectorXd weight_xi =
    (Eigen::VectorXd(N_QP_3) << 0.5555555555555556, 0.8888888888888888,
     0.5555555555555556)
        .finished();

// 2-point Gauss-Legendre quadrature (for eta and zeta directions)
const Eigen::VectorXd gauss_eta =
    (Eigen::VectorXd(N_QP_2) << -0.5773502691896257, 0.5773502691896257)
        .finished();

const Eigen::VectorXd weight_eta =
    (Eigen::VectorXd(N_QP_2) << 1.0, 1.0).finished();

const Eigen::VectorXd gauss_zeta =
    (Eigen::VectorXd(N_QP_2) << -0.5773502691896257, 0.5773502691896257)
        .finished();

const Eigen::VectorXd weight_zeta =
    (Eigen::VectorXd(N_QP_2) << 1.0, 1.0).finished();

} // namespace Quadrature