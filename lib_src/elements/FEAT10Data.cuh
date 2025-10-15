#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "ElementBase.h"

// Add this include at the top:
#include "../../lib_utils/quadrature_utils.h"

// Definition of GPU_ANCF3443 and data access device functions
#pragma once

#ifndef HANDLE_ERROR_MACRO
#define HANDLE_ERROR_MACRO
static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#endif

//
// define a SAP data strucutre
struct GPU_FEAT10_Data : public ElementBase {
#if defined(__CUDACC__)

  // Helper: gather 16 DOFs for an element using connectivity
  __device__ void gather_element_dofs(const double *global,
                                      Eigen::Map<Eigen::MatrixXi> connectivity,
                                      int elem, double *local) const {
    // Each element has 4 nodes, each node has 4 DOFs
    for (int n = 0; n < 4; ++n) {
      int node = connectivity(elem, n);
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        local[n * 4 + d] = global[node * 4 + d];
      }
    }
  }

  __device__ Eigen::Map<Eigen::MatrixXi> element_connectivity() const {
    return Eigen::Map<Eigen::MatrixXi>(d_element_connectivity, n_elem, 4);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> grad_N_ref(int elem_idx, int qp_idx) {
    return Eigen::Map<Eigen::MatrixXd>(
        d_grad_N_ref + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 10 * 3,
        10, 3);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> grad_N_ref(int elem_idx,
                                                          int qp_idx) const {
    return Eigen::Map<Eigen::MatrixXd>(
        d_grad_N_ref + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 10 * 3,
        10, 3);
  }

  __device__ double &detJ_ref(int elem_idx, int qp_idx) {
    return d_detJ_ref[elem_idx * Quadrature::N_QP_T10_5 + qp_idx];
  }

  __device__ double detJ_ref(int elem_idx, int qp_idx) const {
    return d_detJ_ref[elem_idx * Quadrature::N_QP_T10_5 + qp_idx];
  }

  __device__ double tet5pt_x(int qp_idx) {
    return d_tet5pt_x[qp_idx];
  }

  __device__ double tet5pt_y(int qp_idx) {
    return d_tet5pt_y[qp_idx];
  }

  __device__ double tet5pt_z(int qp_idx) {
    return d_tet5pt_z[qp_idx];
  }

  __device__ double tet5pt_weights(int qp_idx) {
    return d_tet5pt_weights[qp_idx];
  }

  __device__ const double tet5pt_weights(int qp_idx) const {
    return d_tet5pt_weights[qp_idx];
  }

  __device__ Eigen::Map<Eigen::VectorXd> x12() {
    return Eigen::Map<Eigen::VectorXd>(d_h_x12, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const x12() const {
    return Eigen::Map<Eigen::VectorXd>(d_h_x12, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> y12() {
    return Eigen::Map<Eigen::VectorXd>(d_h_y12, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const y12() const {
    return Eigen::Map<Eigen::VectorXd>(d_h_y12, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> z12() {
    return Eigen::Map<Eigen::VectorXd>(d_h_z12, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const z12() const {
    return Eigen::Map<Eigen::VectorXd>(d_h_z12, n_coef);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> F(int elem_idx, int qp_idx) {
    return Eigen::Map<Eigen::MatrixXd>(
        d_F + (elem_idx * Quadrature::N_NODE_T10_10 + qp_idx) * 9, 3, 3);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> F(int elem_idx,
                                                 int qp_idx) const {
    return Eigen::Map<Eigen::MatrixXd>(
        d_F + (elem_idx * Quadrature::N_NODE_T10_10 + qp_idx) * 9, 3, 3);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> P(int elem_idx, int qp_idx) {
    return Eigen::Map<Eigen::MatrixXd>(
        d_P + (elem_idx * Quadrature::N_NODE_T10_10 + qp_idx) * 9, 3, 3);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> P(int elem_idx,
                                                 int qp_idx) const {
    return Eigen::Map<Eigen::MatrixXd>(
        d_P + (elem_idx * Quadrature::N_NODE_T10_10 + qp_idx) * 9, 3, 3);
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

  __device__ int gpu_n_elem() const {
    return n_elem;
  }

  __device__ int gpu_n_coef() const {
    return n_coef;
  }

  __device__ int gpu_n_constraint() const {
    return n_constraint;
  }

  // ======================================================

  __device__ Eigen::Map<Eigen::MatrixXd> node_values() {
    return Eigen::Map<Eigen::MatrixXd>(d_node_values, n_coef, n_coef);
  }

  __device__ double *node_values(int i, int j) {
    return d_node_values + j + i * n_coef;
  }

#endif

  __host__ __device__ int get_n_elem() const {
    return n_elem;
  }
  __host__ __device__ int get_n_coef() const {
    return n_coef;
  }
  __host__ __device__ int get_n_constraint() const {
    return n_constraint;
  }

  // Add this missing virtual function from ElementBase:
  __host__ __device__ int get_n_beam() const override {
    return n_elem;
  }

  // Core computation functions (empty implementations for now)
  void CalcDnDuPre();

  void CalcMassMatrix() override;

  void CalcInternalForce() override;

  void CalcConstraintData() override {}

  void CalcP() override;

  void RetrieveMassMatrixToCPU(Eigen::MatrixXd &mass_matrix) override;

  void RetrieveInternalForceToCPU(Eigen::VectorXd &internal_force) override;

  void RetrieveConstraintDataToCPU(Eigen::VectorXd &constraint) override {}

  void RetrieveConstraintJacobianToCPU(
      Eigen::MatrixXd &constraint_jac) override {}

  void RetrievePositionToCPU(Eigen::VectorXd &x12, Eigen::VectorXd &y12,
                             Eigen::VectorXd &z12) override {}

  void RetrieveDeformationGradientToCPU(
      std::vector<std::vector<Eigen::MatrixXd>> &deformation_gradient)
      override {}

  void RetrievePFromFToCPU(
      std::vector<std::vector<Eigen::MatrixXd>> &p_from_F) override;

  void RetrieveDnDuPreToCPU(
      std::vector<std::vector<Eigen::MatrixXd>> &dn_du_pre);

  void RetrieveDetJToCPU(std::vector<std::vector<double>> &detJ);

  // Constructor
  GPU_FEAT10_Data(int num_elements, int num_nodes)
      : n_elem(num_elements), n_coef(num_nodes) {}

  void Initialize() {
    HANDLE_ERROR(cudaMalloc(&d_h_x12, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_h_y12, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_h_z12, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_element_connectivity,
                            n_elem * Quadrature::N_NODE_T10_10 * sizeof(int)));

    HANDLE_ERROR(cudaMalloc(&d_node_values, n_coef * n_coef * sizeof(double)));

    HANDLE_ERROR(
        cudaMalloc(&d_tet5pt_x, Quadrature::N_QP_T10_5 * sizeof(double)));
    HANDLE_ERROR(
        cudaMalloc(&d_tet5pt_y, Quadrature::N_QP_T10_5 * sizeof(double)));
    HANDLE_ERROR(
        cudaMalloc(&d_tet5pt_z, Quadrature::N_QP_T10_5 * sizeof(double)));
    HANDLE_ERROR(
        cudaMalloc(&d_tet5pt_weights, Quadrature::N_QP_T10_5 * sizeof(double)));

    HANDLE_ERROR(cudaMalloc(&d_grad_N_ref, n_elem * Quadrature::N_QP_T10_5 *
                                               10 * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_detJ_ref,
                            n_elem * Quadrature::N_QP_T10_5 * sizeof(double)));

    HANDLE_ERROR(cudaMalloc(
        &d_F, n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(
        &d_P, n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_f_int, n_coef * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_f_ext, n_coef * 3 * sizeof(double)));

    HANDLE_ERROR(cudaMalloc(&d_rho0, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_nu, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_E, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_lambda, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_mu, sizeof(double)));

    //     // copy struct to device
    HANDLE_ERROR(cudaMalloc(&d_data, sizeof(GPU_FEAT10_Data)));
  }

  void Setup(double rho0, double nu, double E,
             const Eigen::VectorXd &tet5pt_x_host,
             const Eigen::VectorXd &tet5pt_y_host,
             const Eigen::VectorXd &tet5pt_z_host,
             const Eigen::VectorXd &tet5pt_weights_host,
             const Eigen::VectorXd &h_x12, const Eigen::VectorXd &h_y12,
             const Eigen::VectorXd &h_z12,
             const Eigen::MatrixXi &element_connectivity) {
    if (is_setup) {
      std::cerr << "GPU_FEAT10_Data is already set up." << std::endl;
      return;
    }

    HANDLE_ERROR(cudaMemcpy(d_h_x12, h_x12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_h_y12, h_y12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_h_z12, h_z12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(d_element_connectivity, element_connectivity.data(),
                            n_elem * Quadrature::N_NODE_T10_10 * sizeof(int),
                            cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(d_tet5pt_x, tet5pt_x_host.data(),
                            Quadrature::N_QP_T10_5 * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_tet5pt_y, tet5pt_y_host.data(),
                            Quadrature::N_QP_T10_5 * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_tet5pt_z, tet5pt_z_host.data(),
                            Quadrature::N_QP_T10_5 * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_tet5pt_weights, tet5pt_weights_host.data(),
                            Quadrature::N_QP_T10_5 * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemset(d_grad_N_ref, 0,
                   n_elem * Quadrature::N_QP_T10_5 * 10 * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMemset(d_detJ_ref, 0,
                            n_elem * Quadrature::N_QP_T10_5 * sizeof(double)));

    HANDLE_ERROR(
        cudaMemset(d_node_values, 0, n_coef * n_coef * sizeof(double)));

    cudaMemset(d_f_int, 0, n_coef * 3 * sizeof(double));

    cudaMemset(d_F, 0,
               n_elem * Quadrature::N_NODE_T10_10 * 3 * 3 * sizeof(double));
    cudaMemset(d_P, 0,
               n_elem * Quadrature::N_NODE_T10_10 * 3 * 3 * sizeof(double));

    HANDLE_ERROR(
        cudaMemcpy(d_rho0, &rho0, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_nu, &nu, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_E, &E, sizeof(double), cudaMemcpyHostToDevice));

    // Compute material constants
    double mu = E / (2 * (1 + nu));  // Shear modulus μ
    double lambda =
        (E * nu) / ((1 + nu) * (1 - 2 * nu));  // Lamé's first parameter λ

    HANDLE_ERROR(cudaMemcpy(d_mu, &mu, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_lambda, &lambda, sizeof(double), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data),
                            cudaMemcpyHostToDevice));

    is_setup = true;
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
      std::cerr << "GPU_FEAT10_Data CONSTRAINT is already set up." << std::endl;
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
    HANDLE_ERROR(cudaFree(d_h_x12));
    HANDLE_ERROR(cudaFree(d_h_y12));
    HANDLE_ERROR(cudaFree(d_h_z12));
    HANDLE_ERROR(cudaFree(d_element_connectivity));
    HANDLE_ERROR(cudaFree(d_node_values));

    HANDLE_ERROR(cudaFree(d_tet5pt_x));
    HANDLE_ERROR(cudaFree(d_tet5pt_y));
    HANDLE_ERROR(cudaFree(d_tet5pt_z));
    HANDLE_ERROR(cudaFree(d_tet5pt_weights));

    HANDLE_ERROR(cudaFree(d_grad_N_ref));
    HANDLE_ERROR(cudaFree(d_detJ_ref));

    HANDLE_ERROR(cudaFree(d_F));
    HANDLE_ERROR(cudaFree(d_P));
    HANDLE_ERROR(cudaFree(d_f_int));
    HANDLE_ERROR(cudaFree(d_f_ext));

    HANDLE_ERROR(cudaFree(d_rho0));
    HANDLE_ERROR(cudaFree(d_nu));
    HANDLE_ERROR(cudaFree(d_E));
    HANDLE_ERROR(cudaFree(d_lambda));
    HANDLE_ERROR(cudaFree(d_mu));

    HANDLE_ERROR(cudaFree(d_data));

    // // Around line 335-337 in Destroy():
    // if (d_constraint != nullptr) {
    //   HANDLE_ERROR(cudaFree(d_constraint));
    // }
    // if (d_constraint_jac != nullptr) {
    //   HANDLE_ERROR(cudaFree(d_constraint_jac));
    // }
    // if (d_fixed_nodes != nullptr) {
    //   HANDLE_ERROR(cudaFree(d_fixed_nodes));
    // }
  }

  GPU_FEAT10_Data *d_data;  // Storing GPU copy of SAPGPUData

  int n_elem;
  int n_coef;
  int n_constraint;

 private:
  // Node positions (global, or per element)
  double *d_h_x12, *d_h_y12, *d_h_z12;  // (n_coef, 1)

  // Element connectivity
  int *d_element_connectivity;  // (n_elem, 10)

  // Mass Matrix
  double *d_node_values;

  // Quadrature points and weights
  double *d_tet5pt_x, *d_tet5pt_y, *d_tet5pt_z;
  double *d_tet5pt_weights;  // (5,)

  // Precomputed reference gradients
  double *d_grad_N_ref;  // (n_elem, 5, 10, 3)
  double *d_detJ_ref;    // (n_elem, 5)

  // Deformation gradient and Piola stress
  double *d_F;  // (n_elem, n_qp, 3, 3)
  double *d_P;  // (n_elem, n_qp, 3, 3)

  // Material properties
  double *d_E, *d_nu, *d_rho0, *d_lambda, *d_mu;

  // Constraint data
  double *d_constraint, *d_constraint_jac;
  int *d_fixed_nodes;

  // Force vectors
  double *d_f_int, *d_f_ext;  // (n_nodes*3)

  bool is_setup             = false;
  bool is_constraints_setup = false;
};
