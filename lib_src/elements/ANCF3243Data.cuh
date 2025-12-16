#include <cuda_runtime.h>
#include <cusparse.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>

/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    ANCF3243Data.cuh
 * Brief:   Declares the GPU_ANCF3243_Data structure and host/GPU interfaces
 *          for ANCF 3243 beam elements. Encapsulates mesh connectivity,
 *          quadrature configuration, CSR mass matrices, internal/external
 *          force vectors, and constraint storage shared with solvers.
 *==============================================================
 *==============================================================*/

#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "ElementBase.h"
#include "../materials/MaterialModel.cuh"

// Definition of GPU_ANCF3243 and data access device functions
#pragma once

//
// define a SAP data strucutre
struct GPU_ANCF3243_Data : public ElementBase {
#if defined(__CUDACC__)

  // Const get functions
  __device__ const Eigen::Map<Eigen::MatrixXd> B_inv() const {
    int row_size = Quadrature::N_SHAPE_3243;
    int col_size = Quadrature::N_SHAPE_3243;
    return Eigen::Map<Eigen::MatrixXd>(d_B_inv, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> ds_du_pre(int qp_idx) const {
    int row_size    = Quadrature::N_SHAPE_3243;
    int col_size    = 3;
    double *qp_data = d_ds_du_pre + qp_idx * Quadrature::N_SHAPE_3243 * 3;
    return Eigen::Map<Eigen::MatrixXd>(qp_data, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> gauss_xi_m() const {
    return Eigen::Map<Eigen::VectorXd>(d_gauss_xi_m, Quadrature::N_QP_6);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> gauss_xi() const {
    return Eigen::Map<Eigen::VectorXd>(d_gauss_xi, Quadrature::N_QP_3);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> gauss_eta() const {
    return Eigen::Map<Eigen::VectorXd>(d_gauss_eta, Quadrature::N_QP_2);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> gauss_zeta() const {
    return Eigen::Map<Eigen::VectorXd>(d_gauss_zeta, Quadrature::N_QP_2);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> weight_xi_m() const {
    return Eigen::Map<Eigen::VectorXd>(d_weight_xi_m, Quadrature::N_QP_6);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> weight_xi() const {
    return Eigen::Map<Eigen::VectorXd>(d_weight_xi, Quadrature::N_QP_3);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> weight_eta() const {
    return Eigen::Map<Eigen::VectorXd>(d_weight_eta, Quadrature::N_QP_2);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> weight_zeta() const {
    return Eigen::Map<Eigen::VectorXd>(d_weight_zeta, Quadrature::N_QP_2);
  }

  __device__ Eigen::Map<Eigen::VectorXd> x12_jac() {
    return Eigen::Map<Eigen::VectorXd>(d_x12_jac, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const x12_jac() const {
    return Eigen::Map<Eigen::VectorXd>(d_x12_jac, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> y12_jac() {
    return Eigen::Map<Eigen::VectorXd>(d_y12_jac, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const y12_jac() const {
    return Eigen::Map<Eigen::VectorXd>(d_y12_jac, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> z12_jac() {
    return Eigen::Map<Eigen::VectorXd>(d_z12_jac, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const z12_jac() const {
    return Eigen::Map<Eigen::VectorXd>(d_z12_jac, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> x12() {
    return Eigen::Map<Eigen::VectorXd>(d_x12, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const x12() const {
    return Eigen::Map<Eigen::VectorXd>(d_x12, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> y12() {
    return Eigen::Map<Eigen::VectorXd>(d_y12, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const y12() const {
    return Eigen::Map<Eigen::VectorXd>(d_y12, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> z12() {
    return Eigen::Map<Eigen::VectorXd>(d_z12, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const z12() const {
    return Eigen::Map<Eigen::VectorXd>(d_z12, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> x12(int elem) {
    return Eigen::Map<Eigen::VectorXd>(
        d_x12 + elem * (Quadrature::N_SHAPE_3243 / 2),
        Quadrature::N_SHAPE_3243);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const x12(int elem) const {
    return Eigen::Map<Eigen::VectorXd>(
        d_x12 + elem * (Quadrature::N_SHAPE_3243 / 2),
        Quadrature::N_SHAPE_3243);
  }

  __device__ Eigen::Map<Eigen::VectorXd> y12(int elem) {
    return Eigen::Map<Eigen::VectorXd>(
        d_y12 + elem * (Quadrature::N_SHAPE_3243 / 2),
        Quadrature::N_SHAPE_3243);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const y12(int elem) const {
    return Eigen::Map<Eigen::VectorXd>(
        d_y12 + elem * (Quadrature::N_SHAPE_3243 / 2),
        Quadrature::N_SHAPE_3243);
  }

  __device__ Eigen::Map<Eigen::VectorXd> z12(int elem) {
    return Eigen::Map<Eigen::VectorXd>(
        d_z12 + elem * (Quadrature::N_SHAPE_3243 / 2),
        Quadrature::N_SHAPE_3243);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const z12(int elem) const {
    return Eigen::Map<Eigen::VectorXd>(
        d_z12 + elem * (Quadrature::N_SHAPE_3243 / 2),
        Quadrature::N_SHAPE_3243);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> F(int elem_idx, int qp_idx) {
    return Eigen::Map<Eigen::MatrixXd>(
        d_F + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> F(int elem_idx,
                                                 int qp_idx) const {
    return Eigen::Map<Eigen::MatrixXd>(
        d_F + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> P(int elem_idx, int qp_idx) {
    return Eigen::Map<Eigen::MatrixXd>(
        d_P + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
  }

  // Time-derivative of deformation gradient (viscous computation)
  __device__ Eigen::Map<Eigen::MatrixXd> Fdot(int elem_idx, int qp_idx) {
    return Eigen::Map<Eigen::MatrixXd>(
        d_Fdot + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> Fdot(int elem_idx,
                                                    int qp_idx) const {
    return Eigen::Map<Eigen::MatrixXd>(
        d_Fdot + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
  }

  // Viscous Piola stress storage
  __device__ Eigen::Map<Eigen::MatrixXd> P_vis(int elem_idx, int qp_idx) {
    return Eigen::Map<Eigen::MatrixXd>(
        d_P_vis + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> P_vis(int elem_idx,
                                                     int qp_idx) const {
    return Eigen::Map<Eigen::MatrixXd>(
        d_P_vis + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> P(int elem_idx,
                                                 int qp_idx) const {
    return Eigen::Map<Eigen::MatrixXd>(
        d_P + (elem_idx * Quadrature::N_TOTAL_QP_3_2_2 + qp_idx) * 9, 3, 3);
  }

  __device__ Eigen::Map<Eigen::VectorXd> f_int(int global_node_idx) {
    return Eigen::Map<Eigen::VectorXd>(d_f_int + global_node_idx * 3, 3);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> f_int(
      int global_node_idx) const {
    return Eigen::Map<Eigen::VectorXd>(d_f_int + global_node_idx * 3, 3);
  }

  __device__ Eigen::Map<Eigen::VectorXd> f_int() {
    return Eigen::Map<Eigen::VectorXd>(d_f_int, n_coef * 3);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> f_int() const {
    return Eigen::Map<Eigen::VectorXd>(d_f_int, n_coef * 3);
  }

  __device__ Eigen::Map<Eigen::VectorXd> f_ext(int global_node_idx) {
    return Eigen::Map<Eigen::VectorXd>(d_f_ext + global_node_idx * 3, 3);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> f_ext(
      int global_node_idx) const {
    return Eigen::Map<Eigen::VectorXd>(d_f_ext + global_node_idx * 3, 3);
  }

  __device__ Eigen::Map<Eigen::VectorXd> f_ext() {
    return Eigen::Map<Eigen::VectorXd>(d_f_ext, n_coef * 3);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> f_ext() const {
    return Eigen::Map<Eigen::VectorXd>(d_f_ext, n_coef * 3);
  }

  __device__ Eigen::Map<Eigen::VectorXd> constraint() {
    return Eigen::Map<Eigen::VectorXd>(d_constraint, n_constraint);
  }

  __device__ const Eigen::Map<Eigen::VectorXd> constraint() const {
    return Eigen::Map<Eigen::VectorXd>(d_constraint, n_constraint);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> constraint_jac() {
    return Eigen::Map<Eigen::MatrixXd>(d_constraint_jac, n_constraint,
                                       n_coef * 3);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> constraint_jac() const {
    return Eigen::Map<Eigen::MatrixXd>(d_constraint_jac, n_constraint,
                                       n_coef * 3);
  }

  __device__ Eigen::Map<Eigen::VectorXi> fixed_nodes() {
    return Eigen::Map<Eigen::VectorXi>(d_fixed_nodes, n_constraint / 3);
  }

  // ================================

  __device__ int element_node(int elem, int local_node_idx) const {
    return d_element_connectivity[elem * 2 + local_node_idx];
  }

  __device__ double L() const {
    return *d_L;
  }

  __device__ double W() const {
    return *d_W;
  }

  __device__ double H() const {
    return *d_H;
  }

  __device__ double rho0() const {
    return *d_rho0;
  }

  __device__ double nu() const {
    return *d_nu;
  }

  __device__ double E() const {
    return *d_E;
  }

  __device__ double lambda() const {
    return *d_lambda;
  }

  __device__ double mu() const {
    return *d_mu;
  }

  __device__ int material_model() const {
    return *d_material_model;
  }

  __device__ double mu10() const {
    return *d_mu10;
  }

  __device__ double mu01() const {
    return *d_mu01;
  }

  __device__ double kappa() const {
    return *d_kappa;
  }

  __device__ double eta_damp() const {
    return *d_eta_damp;
  }

  __device__ double lambda_damp() const {
    return *d_lambda_damp;
  }
  __device__ int gpu_n_beam() const {
    return n_beam;
  }

  __device__ int gpu_n_coef() const {
    return n_coef;
  }

  __device__ int gpu_n_constraint() const {
    return n_constraint;
  }
  //===========================================

  __device__ int *csr_offsets() {
    return d_csr_offsets;
  }

  __device__ int *csr_columns() {
    return d_csr_columns;
  }

  __device__ double *csr_values() {
    return d_csr_values;
  }

  __device__ int *cj_csr_offsets() {
    return d_cj_csr_offsets;
  }

  __device__ int *cj_csr_columns() {
    return d_cj_csr_columns;
  }

  __device__ double *cj_csr_values() {
    return d_cj_csr_values;
  }

  __device__ int nnz() {
    return *d_nnz;
  }

#endif
  __host__ __device__ int get_n_beam() const {
    return n_beam;
  }
  __host__ __device__ int get_n_coef() const {
    return n_coef;
  }
  __host__ __device__ int get_n_constraint() const {
    return n_constraint;
  }

  // Constructor
  GPU_ANCF3243_Data(int num_nodes, int num_elements)
      : n_nodes(num_nodes), n_elements(num_elements) {
    n_beam = num_elements;  // Initialize n_beam with n_elements
    n_coef = 4 * n_nodes;   // Non-overlapping DOFs: 4 DOFs per node
    type   = TYPE_3243;
  }

  void Initialize() {
    HANDLE_ERROR(cudaMalloc(
        &d_B_inv,
        Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_ds_du_pre, Quadrature::N_TOTAL_QP_3_2_2 *
                                              Quadrature::N_SHAPE_3243 * 3 *
                                              sizeof(double)));

    HANDLE_ERROR(
        cudaMalloc(&d_gauss_xi_m, Quadrature::N_QP_6 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_gauss_xi, Quadrature::N_QP_3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_gauss_eta, Quadrature::N_QP_2 * sizeof(double)));
    HANDLE_ERROR(
        cudaMalloc(&d_gauss_zeta, Quadrature::N_QP_2 * sizeof(double)));

    HANDLE_ERROR(
        cudaMalloc(&d_weight_xi_m, Quadrature::N_QP_6 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_weight_xi, Quadrature::N_QP_3 * sizeof(double)));
    HANDLE_ERROR(
        cudaMalloc(&d_weight_eta, Quadrature::N_QP_2 * sizeof(double)));
    HANDLE_ERROR(
        cudaMalloc(&d_weight_zeta, Quadrature::N_QP_2 * sizeof(double)));

    HANDLE_ERROR(cudaMalloc(&d_x12_jac, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_y12_jac, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_z12_jac, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_x12, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_y12, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_z12, n_coef * sizeof(double)));

    HANDLE_ERROR(
        cudaMalloc(&d_element_connectivity, n_elements * 2 * sizeof(int)));

    HANDLE_ERROR(cudaMalloc(
        &d_F, n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(
        &d_P, n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(double)));
    // Kelvin-Voigt damping related buffers
    HANDLE_ERROR(cudaMalloc(&d_Fdot, n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 *
                                         3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_P_vis, n_beam * Quadrature::N_TOTAL_QP_3_2_2 *
                                          3 * 3 * sizeof(double)));
    // damping parameters (single scalar copied to device)
    HANDLE_ERROR(cudaMalloc(&d_eta_damp, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_lambda_damp, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_f_int, n_coef * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_f_ext, n_coef * 3 * sizeof(double)));

    // copy struct to device
    HANDLE_ERROR(cudaMalloc(&d_data, sizeof(GPU_ANCF3243_Data)));

    // beam data
    HANDLE_ERROR(cudaMalloc(&d_H, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_W, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_L, sizeof(double)));

    HANDLE_ERROR(cudaMalloc(&d_rho0, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_nu, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_E, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_lambda, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_mu, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_material_model, sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_mu10, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_mu01, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_kappa, sizeof(double)));
  }

  void Setup(double length, double width, double height,
             const Eigen::MatrixXd &h_B_inv, const Eigen::VectorXd &gauss_xi_m,
             const Eigen::VectorXd &gauss_xi, const Eigen::VectorXd &gauss_eta,
             const Eigen::VectorXd &gauss_zeta,
             const Eigen::VectorXd &weight_xi_m,
             const Eigen::VectorXd &weight_xi,
             const Eigen::VectorXd &weight_eta,
             const Eigen::VectorXd &weight_zeta, const Eigen::VectorXd &h_x12,
             const Eigen::VectorXd &h_y12, const Eigen::VectorXd &h_z12,
             const Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>
                 &h_element_connectivity) {
    if (is_setup) {
      std::cerr << "GPU_ANCF3243_Data is already set up." << std::endl;
      return;
    }

    HANDLE_ERROR(cudaMemcpy(
        d_B_inv, h_B_inv.data(),
        Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243 * sizeof(double),
        cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(d_gauss_xi_m, gauss_xi_m.data(),
                            Quadrature::N_QP_6 * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gauss_xi, gauss_xi.data(),
                            Quadrature::N_QP_3 * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gauss_eta, gauss_eta.data(),
                            Quadrature::N_QP_2 * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_gauss_zeta, gauss_zeta.data(),
                            Quadrature::N_QP_2 * sizeof(double),
                            cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(d_weight_xi_m, weight_xi_m.data(),
                            Quadrature::N_QP_6 * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_weight_xi, weight_xi.data(),
                            Quadrature::N_QP_3 * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_weight_eta, weight_eta.data(),
                            Quadrature::N_QP_2 * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_weight_zeta, weight_zeta.data(),
                            Quadrature::N_QP_2 * sizeof(double),
                            cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(d_x12_jac, h_x12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_y12_jac, h_y12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_z12_jac, h_z12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_x12, h_x12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_y12, h_y12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_z12, h_z12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(d_element_connectivity, h_element_connectivity.data(),
                   n_elements * 2 * sizeof(int), cudaMemcpyHostToDevice));

    cudaMemset(d_f_int, 0, n_coef * 3 * sizeof(double));

    cudaMemset(d_F, 0,
               n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(double));
    cudaMemset(d_P, 0,
               n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(double));
    // initialize damping buffers to zero
    cudaMemset(d_Fdot, 0,
               n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(double));
    cudaMemset(d_P_vis, 0,
               n_beam * Quadrature::N_TOTAL_QP_3_2_2 * 3 * 3 * sizeof(double));

    HANDLE_ERROR(
        cudaMemcpy(d_H, &height, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_W, &width, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_L, &length, sizeof(double), cudaMemcpyHostToDevice));

    double rho0        = 0.0;
    double nu          = 0.0;
    double E           = 0.0;
    double mu = E / (2 * (1 + nu));  // Shear modulus μ
    double lambda =
        (E * nu) / ((1 + nu) * (1 - 2 * nu));  // Lamé’s first parameter λ
    double eta_damp    = 0.0;
    double lambda_damp = 0.0;
    int material_model = MATERIAL_MODEL_SVK;
    double mu10        = 0.0;
    double mu01        = 0.0;
    double kappa       = 0.0;

    HANDLE_ERROR(
        cudaMemcpy(d_rho0, &rho0, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_nu, &nu, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_E, &E, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_mu, &mu, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_lambda, &lambda, sizeof(double), cudaMemcpyHostToDevice));
    // copy damping scalars to device (single double each)
    HANDLE_ERROR(cudaMemcpy(d_eta_damp, &eta_damp, sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lambda_damp, &lambda_damp, sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_material_model, &material_model, sizeof(int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_mu10, &mu10, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_mu01, &mu01, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_kappa, &kappa, sizeof(double), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3243_Data),
                            cudaMemcpyHostToDevice));

    is_setup = true;
  }

  /**
   * Set reference density (used for mass/inertial terms).
   */
  void SetDensity(double rho0) {
    if (!is_setup) {
      std::cerr << "GPU_ANCF3243_Data must be set up before setting density."
                << std::endl;
      return;
    }
    HANDLE_ERROR(
        cudaMemcpy(d_rho0, &rho0, sizeof(double), cudaMemcpyHostToDevice));
  }

  /**
   * Set Kelvin-Voigt damping parameters.
   * eta_damp: shear-like damping coefficient
   * lambda_damp: volumetric-like damping coefficient
   */
  void SetDamping(double eta_damp, double lambda_damp) {
    if (!is_setup) {
      std::cerr << "GPU_ANCF3243_Data must be set up before setting damping."
                << std::endl;
      return;
    }
    HANDLE_ERROR(cudaMemcpy(d_eta_damp, &eta_damp, sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lambda_damp, &lambda_damp, sizeof(double),
                            cudaMemcpyHostToDevice));
  }

  /**
   * Select Saint Venant-Kirchhoff (SVK) material model using current E/nu.
   */
  void SetSVK() {
    if (!is_setup) {
      std::cerr << "GPU_ANCF3243_Data must be set up before setting material."
                << std::endl;
      return;
    }

    int material_model = MATERIAL_MODEL_SVK;
    double mu10        = 0.0;
    double mu01        = 0.0;
    double kappa       = 0.0;
    HANDLE_ERROR(cudaMemcpy(d_material_model, &material_model, sizeof(int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_mu10, &mu10, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_mu01, &mu01, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_kappa, &kappa, sizeof(double), cudaMemcpyHostToDevice));
  }

  void SetSVK(double E, double nu) {
    if (!is_setup) {
      std::cerr << "GPU_ANCF3243_Data must be set up before setting material."
                << std::endl;
      return;
    }

    HANDLE_ERROR(cudaMemcpy(d_nu, &nu, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_E, &E, sizeof(double), cudaMemcpyHostToDevice));

    double mu = E / (2 * (1 + nu));
    double lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
    HANDLE_ERROR(cudaMemcpy(d_mu, &mu, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_lambda, &lambda, sizeof(double), cudaMemcpyHostToDevice));

    SetSVK();
  }

  /**
   * Set compressible Mooney-Rivlin parameters.
   * mu10, mu01: isochoric Mooney-Rivlin coefficients
   * kappa: volumetric penalty (bulk-modulus-like) coefficient
   */
  void SetMooneyRivlin(double mu10, double mu01, double kappa) {
    if (!is_setup) {
      std::cerr << "GPU_ANCF3243_Data must be set up before setting material."
                << std::endl;
      return;
    }

    int material_model = MATERIAL_MODEL_MOONEY_RIVLIN;
    HANDLE_ERROR(cudaMemcpy(d_material_model, &material_model, sizeof(int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_mu10, &mu10, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_mu01, &mu01, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_kappa, &kappa, sizeof(double), cudaMemcpyHostToDevice));
  }

  void SetExternalForce(const Eigen::VectorXd &f_ext) {
    if (f_ext.size() != n_coef * 3) {
      std::cerr << "External force vector size mismatch." << std::endl;
      return;
    }
    cudaMemset(d_f_ext, 0, n_coef * 3 * sizeof(double));
    HANDLE_ERROR(cudaMemcpy(d_f_ext, f_ext.data(), n_coef * 3 * sizeof(double),
                            cudaMemcpyHostToDevice));
  }

  void SetNodalFixed(const Eigen::VectorXi &fixed_nodes) {
    if (is_constraints_setup) {
      std::cerr << "GPU_ANCF3243_Data CONSTRAINT is already set up."
                << std::endl;
      return;
    }

    n_constraint = fixed_nodes.size() * 3;

    HANDLE_ERROR(cudaMalloc(&d_constraint, n_constraint * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_constraint_jac,
                            n_constraint * (n_coef * 3) * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_fixed_nodes, fixed_nodes.size() * sizeof(int)));

    HANDLE_ERROR(cudaMemset(d_constraint, 0, n_constraint * sizeof(double)));
    HANDLE_ERROR(cudaMemset(d_constraint_jac, 0,
                            n_constraint * (n_coef * 3) * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(d_fixed_nodes, fixed_nodes.data(),
                            fixed_nodes.size() * sizeof(int),
                            cudaMemcpyHostToDevice));

    is_constraints_setup = true;
  }

  // Free memory
  void Destroy() {
    HANDLE_ERROR(cudaFree(d_B_inv));
    HANDLE_ERROR(cudaFree(d_ds_du_pre));

    HANDLE_ERROR(cudaFree(d_gauss_xi_m));
    HANDLE_ERROR(cudaFree(d_gauss_xi));
    HANDLE_ERROR(cudaFree(d_gauss_eta));
    HANDLE_ERROR(cudaFree(d_gauss_zeta));
    HANDLE_ERROR(cudaFree(d_weight_xi_m));
    HANDLE_ERROR(cudaFree(d_weight_xi));
    HANDLE_ERROR(cudaFree(d_weight_eta));
    HANDLE_ERROR(cudaFree(d_weight_zeta));

    HANDLE_ERROR(cudaFree(d_x12_jac));
    HANDLE_ERROR(cudaFree(d_y12_jac));
    HANDLE_ERROR(cudaFree(d_z12_jac));
    HANDLE_ERROR(cudaFree(d_x12));
    HANDLE_ERROR(cudaFree(d_y12));
    HANDLE_ERROR(cudaFree(d_z12));

    HANDLE_ERROR(cudaFree(d_element_connectivity));

    if (is_csr_setup) {
      HANDLE_ERROR(cudaFree(d_csr_offsets));
      HANDLE_ERROR(cudaFree(d_csr_columns));
      HANDLE_ERROR(cudaFree(d_csr_values));
      HANDLE_ERROR(cudaFree(d_nnz));
    }

    if (is_cj_csr_setup) {
      HANDLE_ERROR(cudaFree(d_cj_csr_offsets));
      HANDLE_ERROR(cudaFree(d_cj_csr_columns));
      HANDLE_ERROR(cudaFree(d_cj_csr_values));
      HANDLE_ERROR(cudaFree(d_cj_nnz));
    }

    HANDLE_ERROR(cudaFree(d_F));
    HANDLE_ERROR(cudaFree(d_P));
    HANDLE_ERROR(cudaFree(d_f_int));
    HANDLE_ERROR(cudaFree(d_f_ext));

    HANDLE_ERROR(cudaFree(d_H));
    HANDLE_ERROR(cudaFree(d_W));
    HANDLE_ERROR(cudaFree(d_L));

    HANDLE_ERROR(cudaFree(d_rho0));
    HANDLE_ERROR(cudaFree(d_nu));
    HANDLE_ERROR(cudaFree(d_E));
    HANDLE_ERROR(cudaFree(d_lambda));
    HANDLE_ERROR(cudaFree(d_mu));
    HANDLE_ERROR(cudaFree(d_material_model));
    HANDLE_ERROR(cudaFree(d_mu10));
    HANDLE_ERROR(cudaFree(d_mu01));
    HANDLE_ERROR(cudaFree(d_kappa));

    if (is_constraints_setup) {
      HANDLE_ERROR(cudaFree(d_constraint));
      HANDLE_ERROR(cudaFree(d_constraint_jac));
      HANDLE_ERROR(cudaFree(d_fixed_nodes));
    }

    HANDLE_ERROR(cudaFree(d_data));
  }

  void CalcDsDuPre();

  void CalcMassMatrix();

  void BuildMassCSRPattern();

  void ConvertTOCSRConstraintJac();

  void BuildConstraintJacobianTransposeCSR() {
    ConvertTOCSRConstraintJac();
  }

  void CalcP();

  void CalcInternalForce();

  void CalcConstraintData() override;

  void PrintDsDuPre();

  void RetrieveMassCSRToCPU(std::vector<int> &offsets,
                            std::vector<int> &columns,
                            std::vector<double> &values);

  void RetrieveDeformationGradientToCPU(
      std::vector<std::vector<Eigen::MatrixXd>> &deformation_gradient);

  void RetrievePFromFToCPU(std::vector<std::vector<Eigen::MatrixXd>> &p_from_F);

  void RetrieveInternalForceToCPU(Eigen::VectorXd &internal_force);

  void RetrieveConstraintDataToCPU(Eigen::VectorXd &constraint);

  void RetrieveConstraintJacobianToCPU(Eigen::MatrixXd &constraint_jac);

  void RetrievePositionToCPU(Eigen::VectorXd &x12, Eigen::VectorXd &y12,
                             Eigen::VectorXd &z12);

  double *Get_Constraint_Ptr() {
    return d_constraint;
  }

  bool Get_Is_Constraint_Setup() {
    return is_constraints_setup;
  }

  GPU_ANCF3243_Data *d_data;  // Storing GPU copy of SAPGPUData

  int n_nodes;
  int n_elements;
  int n_beam;
  int n_coef;
  int n_constraint;

 private:
  double *d_B_inv;
  double *d_ds_du_pre;
  double *d_gauss_xi_m, *d_gauss_xi, *d_gauss_eta, *d_gauss_zeta;
  double *d_weight_xi_m, *d_weight_xi, *d_weight_eta, *d_weight_zeta;

  double *d_x12_jac, *d_y12_jac, *d_z12_jac;
  double *d_x12, *d_y12, *d_z12;

  int *d_element_connectivity;  // n_elements × 2 array of node IDs
  int *d_csr_offsets, *d_csr_columns;
  double *d_csr_values;
  int *d_nnz;

  double *d_F, *d_P;
  // Kelvin-Voigt damping related device buffers
  double *d_Fdot;         // time derivative of F (per element, per qp, 3x3)
  double *d_P_vis;        // viscous Piola (per element, per qp, 3x3)
  double *d_eta_damp;     // per-element (or global) viscous coefficient
  double *d_lambda_damp;  // per-element (or global) second viscous coeff

  double *d_H, *d_W, *d_L;

  double *d_rho0, *d_nu, *d_E, *d_lambda, *d_mu;
  int *d_material_model;
  double *d_mu10, *d_mu01, *d_kappa;

  double *d_constraint, *d_constraint_jac;
  int *d_fixed_nodes;

  int *d_cj_csr_offsets, *d_cj_csr_columns;
  double *d_cj_csr_values;
  int *d_cj_nnz;

  // force related parameters
  double *d_f_int, *d_f_ext;

  bool is_setup             = false;
  bool is_constraints_setup = false;
  bool is_csr_setup         = false;
  bool is_cj_csr_setup      = false;
};
