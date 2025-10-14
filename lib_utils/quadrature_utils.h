#pragma once

#include <Eigen/Dense>

/**
 * Gauss-Legendre quadrature points and weights for ANCF calculations
 */
namespace Quadrature
{

    // Quadrature sizes as constexpr constants
    constexpr int N_QP_7 = 7; // 7-point quadrature
    constexpr int N_QP_6 = 6; // 6-point quadrature
    constexpr int N_QP_4 = 4;
    constexpr int N_QP_3 = 3; // 3-point quadrature
    constexpr int N_QP_2 = 2; // 2-point quadrature

    // Total number of quadrature points
    constexpr int N_TOTAL_QP_3_2_2 = N_QP_3 * N_QP_2 * N_QP_2;
    constexpr int N_TOTAL_QP_4_4_3 = N_QP_4 * N_QP_4 * N_QP_3;
    constexpr int N_TOTAL_QP_7_7_3 = N_QP_7 * N_QP_7 * N_QP_3;

    // Number of elements for shape functions
    constexpr int N_SHAPE_3243 = 8;
    constexpr int N_SHAPE_3443 = 16;

    // 6-point Gauss-Legendre quadrature (symmetric)
    const Eigen::VectorXd gauss_xi_m_6 =
        (Eigen::VectorXd(N_QP_6) << -0.93246951420315202, -0.66120938646626451,
         -0.23861918608319691, 0.23861918608319691, 0.66120938646626451,
         0.93246951420315202)
            .finished();

    const Eigen::VectorXd weight_xi_m_6 =
        (Eigen::VectorXd(N_QP_6) << 0.17132449237917034, 0.36076157304813861,
         0.46791393457269104, 0.46791393457269104, 0.36076157304813861, 0.17132449237917034)
            .finished();

    const Eigen::VectorXd gauss_xi_m_7 =
        (Eigen::VectorXd(N_QP_7) << -0.949107912342759, -0.741531185599394,
         -0.405845151377397, 0.0, 0.405845151377397,
         0.741531185599394, 0.949107912342759)
            .finished();

    const Eigen::VectorXd weight_xi_m_7 =
        (Eigen::VectorXd(N_QP_7) << 0.129484966168870, 0.279705391489277,
         0.381830050505119, 0.417959183673469, 0.381830050505119,
         0.279705391489277, 0.129484966168870)
            .finished();

    const Eigen::VectorXd gauss_eta_m_7 =
        (Eigen::VectorXd(N_QP_7) << -0.949107912342759, -0.741531185599394,
         -0.405845151377397, 0.0, 0.405845151377397,
         0.741531185599394, 0.949107912342759)
            .finished();

    const Eigen::VectorXd weight_eta_m_7 =
        (Eigen::VectorXd(N_QP_7) << 0.129484966168870, 0.279705391489277,
         0.381830050505119, 0.417959183673469, 0.381830050505119,
         0.279705391489277, 0.129484966168870)
            .finished();

    // 3-point Gauss-Legendre quadrature for zeta (symmetric)
    const Eigen::VectorXd gauss_zeta_m_3 =
        (Eigen::VectorXd(N_QP_3) << -0.7745966692414834, 0.0, 0.7745966692414834)
            .finished();

    const Eigen::VectorXd weight_zeta_m_3 =
        (Eigen::VectorXd(N_QP_3) << 0.5555555555555556, 0.8888888888888888, 0.5555555555555556)
            .finished();

    // ================================================

    // 3-point Gauss-Legendre quadrature
    const Eigen::VectorXd gauss_xi_3 =
        (Eigen::VectorXd(N_QP_3) << -0.77459666924148340, 0.00000000000000000, 0.77459666924148340)
            .finished();

    const Eigen::VectorXd weight_xi_3 =
        (Eigen::VectorXd(N_QP_3) << 0.55555555555555556, 0.88888888888888889,
         0.55555555555555556)
            .finished();

    const Eigen::VectorXd gauss_xi_4 =
        (Eigen::VectorXd(4) << -0.8611363115940526, -0.3399810435848563,
         0.3399810435848563, 0.8611363115940526)
            .finished();

    const Eigen::VectorXd weight_xi_4 =
        (Eigen::VectorXd(4) << 0.3478548451374538, 0.6521451548625461,
         0.6521451548625461, 0.3478548451374538)
            .finished();

    // 2-point Gauss-Legendre quadrature (for eta and zeta directions)
    const Eigen::VectorXd gauss_eta_2 =
        (Eigen::VectorXd(N_QP_2) << -0.57735026918962576, 0.57735026918962576)
            .finished();

    const Eigen::VectorXd weight_eta_2 =
        (Eigen::VectorXd(N_QP_2) << 1.00000000000000000, 1.00000000000000000).finished();

    const Eigen::VectorXd gauss_eta_4 =
        (Eigen::VectorXd(4) << -0.8611363115940526, -0.3399810435848563,
         0.3399810435848563, 0.8611363115940526)
            .finished();

    const Eigen::VectorXd weight_eta_4 =
        (Eigen::VectorXd(4) << 0.3478548451374538, 0.6521451548625461,
         0.6521451548625461, 0.3478548451374538)
            .finished();

    const Eigen::VectorXd gauss_zeta_2 =
        (Eigen::VectorXd(N_QP_2) << -0.57735026918962576, 0.57735026918962576)
            .finished();

    const Eigen::VectorXd weight_zeta_2 =
        (Eigen::VectorXd(N_QP_2) << 1.00000000000000000, 1.00000000000000000).finished();

    const Eigen::VectorXd gauss_zeta_3 =
        (Eigen::VectorXd(N_QP_3) << -0.77459666924148340, 0.00000000000000000, 0.77459666924148340)
            .finished();

    const Eigen::VectorXd weight_zeta_3 =
        (Eigen::VectorXd(N_QP_3) << 0.55555555555555556, 0.88888888888888889,
         0.55555555555555556)
            .finished();

    // 5-point Keast quadrature for tetrahedral elements
    constexpr int N_QP_TET_5 = 5;

    // Barycentric coordinates for 5-point Keast quadrature (each row: [L1, L2, L3, L4])
    const Eigen::Matrix<double, N_QP_TET_5, 4> tet5pt_bary =
        (Eigen::Matrix<double, N_QP_TET_5, 4>() << 0.25, 0.25, 0.25, 0.25,
         0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0,
         1.0 / 6.0, 0.5, 1.0 / 6.0, 1.0 / 6.0,
         1.0 / 6.0, 1.0 / 6.0, 0.5, 1.0 / 6.0,
         1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5)
            .finished();

    // Weights for 5-point Keast quadrature
    const Eigen::VectorXd tet5pt_weights =
        (Eigen::VectorXd(N_QP_TET_5) << -4.0 / 5.0, 9.0 / 20.0, 9.0 / 20.0, 9.0 / 20.0, 9.0 / 20.0).finished() * (1.0 / 6.0);

    // Cartesian coordinates (extract columns 1:3 from barycentric)
    const Eigen::Matrix<double, N_QP_TET_5, 3> tet5pt_xyz = tet5pt_bary.block(0, 1, N_QP_TET_5, 3);

    // Optionally, you can wrap these in a struct for clarity:
    struct Tet5ptQuadrature
    {
        static constexpr int n_points = N_QP_TET_5;
        static const Eigen::Matrix<double, N_QP_TET_5, 4> &barycentric() { return tet5pt_bary; }
        static const Eigen::Matrix<double, N_QP_TET_5, 3> &xyz() { return tet5pt_xyz; }
        static const Eigen::VectorXd &weights() { return tet5pt_weights; }
    };
} // namespace Quadrature