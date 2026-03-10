#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <vector>

#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/mesh_manager.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../materials/MaterialModel.cuh"
#include "../materials/SolidMaterialProperties.h"
#include "ElementBase.h"

// Definition of GPU_ANCF3443 and data access device functions
#pragma once

// Compact, per-mesh material storage for FEAT10 objects.
// The constitutive model is still assumed uniform across the object (SVK or MR),
// but parameters can vary per imported mesh (element ranges).
struct FEAT10MeshMaterial {
  double rho0        = 0.0;
  double eta_damp    = 0.0;
  double lambda_damp = 0.0;

  // SVK parameters (precomputed Lamé parameters).
  double lambda = 0.0;
  double mu     = 0.0;

  // Mooney-Rivlin parameters.
  double mu10  = 0.0;
  double mu01  = 0.0;
  double kappa = 0.0;
};

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
    return Eigen::Map<Eigen::MatrixXi>(d_element_connectivity, n_elem,
                                       Quadrature::N_NODE_T10_10);
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

  __device__ Eigen::Map<Eigen::VectorXd> const x12_jac() const {
    return Eigen::Map<Eigen::VectorXd>(d_h_x12_jac, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const y12_jac() const {
    return Eigen::Map<Eigen::VectorXd>(d_h_y12_jac, n_coef);
  }

  __device__ Eigen::Map<Eigen::VectorXd> const z12_jac() const {
    return Eigen::Map<Eigen::VectorXd>(d_h_z12_jac, n_coef);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> F(int elem_idx, int qp_idx) {
    return Eigen::Map<Eigen::MatrixXd>(
        d_F + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> F(int elem_idx,
                                                 int qp_idx) const {
    return Eigen::Map<Eigen::MatrixXd>(
        d_F + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> P(int elem_idx, int qp_idx) {
    return Eigen::Map<Eigen::MatrixXd>(
        d_P + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> P(int elem_idx,
                                                 int qp_idx) const {
    return Eigen::Map<Eigen::MatrixXd>(
        d_P + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
  }

  // Time-derivative of deformation gradient (viscous computation)
  __device__ Eigen::Map<Eigen::MatrixXd> Fdot(int elem_idx, int qp_idx) {
    return Eigen::Map<Eigen::MatrixXd>(
        d_Fdot + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> Fdot(int elem_idx,
                                                    int qp_idx) const {
    return Eigen::Map<Eigen::MatrixXd>(
        d_Fdot + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
  }

  // Viscous Piola stress storage
  __device__ Eigen::Map<Eigen::MatrixXd> P_vis(int elem_idx, int qp_idx) {
    return Eigen::Map<Eigen::MatrixXd>(
        d_P_vis + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> P_vis(int elem_idx,
                                                     int qp_idx) const {
    return Eigen::Map<Eigen::MatrixXd>(
        d_P_vis + (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 9, 3, 3);
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

  __device__ Eigen::Map<Eigen::VectorXi> fixed_nodes() {
    return Eigen::Map<Eigen::VectorXi>(d_fixed_nodes, n_constraint / 3);
  }

  // ================================
  __device__ double rho0() const {
    return *d_rho0;
  }

  // Optional per-element density override (used for multi-body problems where
  // each mesh instance has its own rho0). If not configured, falls back to the
  // global scalar density.
  __device__ double rho0(int elem_idx) const {
    if (d_rho0_elem != nullptr) {
      return static_cast<double>(d_rho0_elem[elem_idx]);
    }
    if (d_mesh_materials != nullptr) {
      const int mesh_id = mesh_id_from_elem(elem_idx);
      if (mesh_id >= 0) {
        return d_mesh_materials[mesh_id].rho0;
      }
    }
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

  __device__ double eta_damp() const {
    return *d_eta_damp;
  }

  __device__ double lambda_damp() const {
    return *d_lambda_damp;
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

  // ---------------------------------------------------------------------------
  // Per-mesh material accessors (fallback to global scalars when not configured)
  // ---------------------------------------------------------------------------

  __device__ __forceinline__ int mesh_id_from_elem(int elem_idx) const {
    if (d_mesh_elem_starts == nullptr || d_mesh_elem_ends == nullptr ||
        n_mesh_materials <= 0) {
      return -1;
    }
    // Binary search over sorted element starts. Returns a mesh id only if the
    // element is inside the corresponding [start, end) interval.
    if (elem_idx < d_mesh_elem_starts[0]) {
      return -1;
    }
    int lo = 0;
    int hi = n_mesh_materials;
    while (lo + 1 < hi) {
      const int mid = (lo + hi) >> 1;
      if (elem_idx < d_mesh_elem_starts[mid]) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    return (elem_idx < d_mesh_elem_ends[lo]) ? lo : -1;
  }

  __device__ __forceinline__ FEAT10MeshMaterial mesh_material(int elem_idx)
      const {
    const int mesh_id = mesh_id_from_elem(elem_idx);
    if (mesh_id >= 0 && d_mesh_materials != nullptr) {
      return d_mesh_materials[mesh_id];
    }
    return mesh_material_fallback();
  }

  __device__ __forceinline__ FEAT10MeshMaterial mesh_material_fallback() const {
    FEAT10MeshMaterial mat;
    mat.rho0        = *d_rho0;
    mat.eta_damp    = *d_eta_damp;
    mat.lambda_damp = *d_lambda_damp;
    mat.lambda      = *d_lambda;
    mat.mu          = *d_mu;
    mat.mu10        = *d_mu10;
    mat.mu01        = *d_mu01;
    mat.kappa       = *d_kappa;
    return mat;
  }

  __device__ __forceinline__ double lambda(int elem_idx) const {
    if (d_mesh_materials == nullptr) {
      return *d_lambda;
    }
    return mesh_material(elem_idx).lambda;
  }

  __device__ __forceinline__ double mu(int elem_idx) const {
    if (d_mesh_materials == nullptr) {
      return *d_mu;
    }
    return mesh_material(elem_idx).mu;
  }

  __device__ __forceinline__ double mu10(int elem_idx) const {
    if (d_mesh_materials == nullptr) {
      return *d_mu10;
    }
    return mesh_material(elem_idx).mu10;
  }

  __device__ __forceinline__ double mu01(int elem_idx) const {
    if (d_mesh_materials == nullptr) {
      return *d_mu01;
    }
    return mesh_material(elem_idx).mu01;
  }

  __device__ __forceinline__ double kappa(int elem_idx) const {
    if (d_mesh_materials == nullptr) {
      return *d_kappa;
    }
    return mesh_material(elem_idx).kappa;
  }

  __device__ __forceinline__ double eta_damp(int elem_idx) const {
    if (d_mesh_materials == nullptr) {
      return *d_eta_damp;
    }
    return mesh_material(elem_idx).eta_damp;
  }

  __device__ __forceinline__ double lambda_damp(int elem_idx) const {
    if (d_mesh_materials == nullptr) {
      return *d_lambda_damp;
    }
    return mesh_material(elem_idx).lambda_damp;
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

  __device__ double &vm_stress(int elem_idx) {
    return d_vm_stress[elem_idx];
  }

  __device__ double vm_stress(int elem_idx) const {
    return d_vm_stress[elem_idx];
  }

  // ======================================================

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

  __device__ int *j_csr_offsets() {
    return d_j_csr_offsets;
  }

  __device__ int *j_csr_columns() {
    return d_j_csr_columns;
  }

  __device__ double *j_csr_values() {
    return d_j_csr_values;
  }

  __device__ int nnz() {
    return *d_nnz;
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

  __host__ __device__ int get_n_beam() const override {
    return n_elem;
  }

  void CalcDnDuPre();

  void CalcMassMatrix() override;

  void BuildMassCSRPattern();

  void ConvertToCSR_ConstraintJacT();

  void BuildConstraintJacobianTransposeCSR() {
    ConvertToCSR_ConstraintJacT();
  }

  void ConvertToCSR_ConstraintJac();

  void BuildConstraintJacobianCSR() {
    ConvertToCSR_ConstraintJac();
  }

  void CalcInternalForce() override;

  void CalcConstraintData() override;

  void CalcP() override;

  void RetrieveMassCSRToCPU(std::vector<int> &offsets,
                            std::vector<int> &columns,
                            std::vector<double> &values);

  void RetrieveInternalForceToCPU(Eigen::VectorXd &internal_force) override;

  void RetrieveExternalForceToCPU(Eigen::VectorXd &external_force);

  void RetrieveConstraintDataToCPU(Eigen::VectorXd &constraint) override {}

  void RetrieveConstraintJacobianToCPU(
      Eigen::MatrixXd &constraint_jac) override {}

  void RetrievePositionToCPU(Eigen::VectorXd &x12, Eigen::VectorXd &y12,
                             Eigen::VectorXd &z12) override;

  void RetrieveDeformationGradientToCPU(
      std::vector<std::vector<Eigen::MatrixXd>> &deformation_gradient)
      override {}

  void RetrievePFromFToCPU(
      std::vector<std::vector<Eigen::MatrixXd>> &p_from_F) override;

  void RetrieveDnDuPreToCPU(
      std::vector<std::vector<Eigen::MatrixXd>> &dn_du_pre);

  void RetrieveDetJToCPU(std::vector<std::vector<double>> &detJ);

  void RetrieveConnectivityToCPU(Eigen::MatrixXi &connectivity);

  void WriteOutputVTK(const std::string &filename);

  void ComputeVonMises();

  void RetrieveVonMisesToCPU(Eigen::VectorXd &vm);

  void RetrieveReferencePositionToCPU(Eigen::VectorXd &x_ref,
                                      Eigen::VectorXd &y_ref,
                                      Eigen::VectorXd &z_ref);

  void WriteOutputVTU(const std::string &filename);

  // Constructor
  GPU_FEAT10_Data(int num_elements, int num_nodes)
      : n_elem(num_elements), n_coef(num_nodes), n_constraint(0) {
    type = TYPE_T10;
  }

  void Initialize() {
    HANDLE_ERROR(cudaMalloc(&d_h_x12, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_h_y12, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_h_z12, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_h_x12_jac, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_h_y12_jac, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_h_z12_jac, n_coef * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_element_connectivity,
                            n_elem * Quadrature::N_NODE_T10_10 * sizeof(int)));

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
    // Viscous-related buffers
    HANDLE_ERROR(cudaMalloc(
        &d_Fdot, n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(
        &d_P_vis, n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_vm_stress, n_elem * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_f_int, n_coef * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_f_ext, n_coef * 3 * sizeof(double)));

    HANDLE_ERROR(cudaMalloc(&d_rho0, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_nu, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_E, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_lambda, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_mu, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_material_model, sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_mu10, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_mu01, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_kappa, sizeof(double)));
    // damping parameters
    HANDLE_ERROR(cudaMalloc(&d_eta_damp, sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&d_lambda_damp, sizeof(double)));

    //     // copy struct to device
    HANDLE_ERROR(cudaMalloc(&d_data, sizeof(GPU_FEAT10_Data)));
  }

  void Setup(const Eigen::VectorXd &tet5pt_x_host,
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
    HANDLE_ERROR(cudaMemcpy(d_h_x12_jac, h_x12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_h_y12_jac, h_y12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_h_z12_jac, h_z12.data(), n_coef * sizeof(double),
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

    cudaMemset(d_f_int, 0, n_coef * 3 * sizeof(double));

    cudaMemset(d_F, 0,
               n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double));
    cudaMemset(d_P, 0,
               n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double));
    // initialize viscous buffers to zero
    cudaMemset(d_Fdot, 0,
               n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double));
    cudaMemset(d_P_vis, 0,
               n_elem * Quadrature::N_QP_T10_5 * 3 * 3 * sizeof(double));
    cudaMemset(d_vm_stress, 0, n_elem * sizeof(double));

    double rho0 = 0.0;
    double nu   = 0.0;
    double E    = 0.0;
    // Compute material constants
    double mu = E / (2 * (1 + nu));  // Shear modulus μ
    double lambda =
        (E * nu) / ((1 + nu) * (1 - 2 * nu));  // Lamé's first parameter λ
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
    // copy damping parameters
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

    HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data),
                            cudaMemcpyHostToDevice));

    is_setup = true;
  }

  /**
   * Set reference density (used for mass/inertial terms).
   */
  void SetDensity(double rho0) {
    if (!is_setup) {
      std::cerr << "GPU_FEAT10_Data must be set up before setting density."
                << std::endl;
      return;
    }
    HANDLE_ERROR(
        cudaMemcpy(d_rho0, &rho0, sizeof(double), cudaMemcpyHostToDevice));
  }

  /**
   * Override density for a contiguous element range [elem_start, elem_start + elem_count).
   *
   * This allocates a per-element density array on first use and initializes it
   * to the current global density, then applies the override on the requested
   * range.
   */
  void SetDensityForElementRange(int elem_start, int elem_count, double rho0) {
    if (!is_setup) {
      std::cerr << "GPU_FEAT10_Data must be set up before setting density."
                << std::endl;
      return;
    }
    if (elem_start < 0 || elem_count <= 0 || elem_start + elem_count > n_elem) {
      std::cerr << "SetDensityForElementRange: invalid element range."
                << std::endl;
      return;
    }

    if (d_rho0_elem == nullptr) {
      // Allocate and initialize to the current global density.
      HANDLE_ERROR(cudaMalloc(&d_rho0_elem,
                              static_cast<size_t>(n_elem) * sizeof(float)));
      double rho0_default = 0.0;
      HANDLE_ERROR(cudaMemcpy(&rho0_default, d_rho0, sizeof(double),
                              cudaMemcpyDeviceToHost));
      std::vector<float> init(static_cast<size_t>(n_elem),
                              static_cast<float>(rho0_default));
      HANDLE_ERROR(cudaMemcpy(d_rho0_elem, init.data(),
                              static_cast<size_t>(n_elem) * sizeof(float),
                              cudaMemcpyHostToDevice));

      // Update the device-side copy of this struct so kernels can see the new pointer.
      HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data),
                              cudaMemcpyHostToDevice));
    }

    std::vector<float> range(static_cast<size_t>(elem_count),
                             static_cast<float>(rho0));
    HANDLE_ERROR(cudaMemcpy(d_rho0_elem + elem_start, range.data(),
                            static_cast<size_t>(elem_count) * sizeof(float),
                            cudaMemcpyHostToDevice));
  }

  /**
   * Set Kelvin-Voigt damping parameters.
   * eta_damp: shear-like damping coefficient
   * lambda_damp: volumetric-like damping coefficient
   */
  void SetDamping(double eta_damp, double lambda_damp) {
    if (!is_setup) {
      std::cerr << "GPU_FEAT10_Data must be set up before setting damping."
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
      std::cerr << "GPU_FEAT10_Data must be set up before setting material."
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

  /**
   * Set Saint Venant-Kirchhoff (SVK) parameters.
   * E: Young's modulus
   * nu: Poisson's ratio
   */
  void SetSVK(double E, double nu) {
    if (!is_setup) {
      std::cerr << "GPU_FEAT10_Data must be set up before setting material."
                << std::endl;
      return;
    }

    HANDLE_ERROR(cudaMemcpy(d_nu, &nu, sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_E, &E, sizeof(double), cudaMemcpyHostToDevice));

    double mu     = E / (2 * (1 + nu));
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
      std::cerr << "GPU_FEAT10_Data must be set up before setting material."
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

  /**
   * Apply all material properties from a SolidMaterialProperties struct.
   * This is the preferred way to set material properties for an object.
   */
  void ApplyMaterial(const SolidMaterialProperties &props) {
    if (!is_setup) {
      std::cerr << "GPU_FEAT10_Data must be set up before applying material."
                << std::endl;
      return;
    }

    // Set density
    SetDensity(props.rho0);

    // Set damping
    SetDamping(props.eta_damp, props.lambda_damp);

    // Set material model and parameters
    if (props.material_model == MATERIAL_MODEL_MOONEY_RIVLIN) {
      SetMooneyRivlin(props.mu10, props.mu01, props.kappa);
    } else {
      SetSVK(props.E, props.nu);
    }
  }

  /**
   * Apply per-mesh material properties given mesh element ranges.
   *
   * This stores a compact material table per mesh (one copy per mesh) plus two
   * integer arrays for element-range lookup. It does NOT allocate per-element
   * material arrays.
   *
   * Important: All materials must use the same constitutive model (SVK or MR).
   */
  void ApplyMaterialsByElementRanges(
      const std::vector<int>& elem_starts, const std::vector<int>& elem_counts,
      const std::vector<SolidMaterialProperties>& materials) {
    if (!is_setup) {
      std::cerr << "GPU_FEAT10_Data must be set up before applying materials."
                << std::endl;
      return;
    }
    if (elem_starts.size() != elem_counts.size() ||
        elem_starts.size() != materials.size() || elem_starts.empty()) {
      std::cerr << "ApplyMaterialsByElementRanges: size mismatch." << std::endl;
      return;
    }

    const int n_mesh = static_cast<int>(materials.size());
    for (int i = 0; i < n_mesh; ++i) {
      if (elem_starts[i] < 0 || elem_counts[i] <= 0 ||
          elem_starts[i] + elem_counts[i] > n_elem) {
        std::cerr << "ApplyMaterialsByElementRanges: invalid element range."
                  << std::endl;
        return;
      }
    }

    const int model0 = materials[0].material_model;
    for (int i = 1; i < n_mesh; ++i) {
      if (materials[i].material_model != model0) {
        std::cerr << "ApplyMaterialsByElementRanges: mixed material models are "
                     "not supported (all must be SVK or all MR)."
                  << std::endl;
        return;
      }
    }

    // Apply first material globally as default/fallback.
    ApplyMaterial(materials[0]);

    // Free previous per-mesh configuration (if any).
    if (d_mesh_materials != nullptr) {
      HANDLE_ERROR(cudaFree(d_mesh_materials));
      d_mesh_materials = nullptr;
    }
    if (d_mesh_elem_starts != nullptr) {
      HANDLE_ERROR(cudaFree(d_mesh_elem_starts));
      d_mesh_elem_starts = nullptr;
    }
    if (d_mesh_elem_ends != nullptr) {
      HANDLE_ERROR(cudaFree(d_mesh_elem_ends));
      d_mesh_elem_ends = nullptr;
    }
    n_mesh_materials = 0;

    // Sort by elem_start to satisfy device binary search.
    std::vector<int> order(n_mesh);
    for (int i = 0; i < n_mesh; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return elem_starts[a] < elem_starts[b]; });

    std::vector<int> starts_sorted(n_mesh);
    std::vector<int> ends_sorted(n_mesh);
    std::vector<FEAT10MeshMaterial> mats_sorted(n_mesh);

    for (int i = 0; i < n_mesh; ++i) {
      const int src = order[i];
      const int s   = elem_starts[src];
      const int c   = elem_counts[src];
      starts_sorted[i] = s;
      ends_sorted[i]   = s + c;

      const auto& props = materials[src];
      FEAT10MeshMaterial mat;
      mat.rho0        = props.rho0;
      mat.eta_damp    = props.eta_damp;
      mat.lambda_damp = props.lambda_damp;
      if (model0 == MATERIAL_MODEL_MOONEY_RIVLIN) {
        mat.mu10  = props.mu10;
        mat.mu01  = props.mu01;
        mat.kappa = props.kappa;
      } else {
        mat.mu     = props.mu();
        mat.lambda = props.lambda();
      }
      mats_sorted[i] = mat;
    }

    for (int i = 0; i + 1 < n_mesh; ++i) {
      if (starts_sorted[i] >= starts_sorted[i + 1] ||
          ends_sorted[i] > starts_sorted[i + 1]) {
        std::cerr << "ApplyMaterialsByElementRanges: overlapping or unsorted "
                     "element ranges are not supported."
                  << std::endl;
        return;
      }
    }

    // Allocate and upload per-mesh material table + element range lookup.
    HANDLE_ERROR(cudaMalloc(&d_mesh_elem_starts,
                            static_cast<size_t>(n_mesh) * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_mesh_elem_ends,
                            static_cast<size_t>(n_mesh) * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_mesh_materials,
                            static_cast<size_t>(n_mesh) *
                                sizeof(FEAT10MeshMaterial)));

    HANDLE_ERROR(cudaMemcpy(d_mesh_elem_starts, starts_sorted.data(),
                            static_cast<size_t>(n_mesh) * sizeof(int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_mesh_elem_ends, ends_sorted.data(),
                            static_cast<size_t>(n_mesh) * sizeof(int),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_mesh_materials, mats_sorted.data(),
                            static_cast<size_t>(n_mesh) *
                                sizeof(FEAT10MeshMaterial),
                            cudaMemcpyHostToDevice));

    n_mesh_materials = n_mesh;
    // Update the device-side copy of this struct so kernels can see the new pointers.
    HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data),
                            cudaMemcpyHostToDevice));
  }

  /**
   * Convenience: apply per-mesh materials directly from a MeshManager.
   *
   * Requires each loaded mesh to have an assigned material via
   * MeshManager::SetMeshMaterial (or LoadMesh overload).
   */
  void ApplyMaterialsFromMeshManager(
      const ANCFCPUUtils::MeshManager& mesh_manager) {
    const int n_mesh = mesh_manager.GetNumMeshes();
    if (n_mesh <= 0) {
      std::cerr << "ApplyMaterialsFromMeshManager: no meshes." << std::endl;
      return;
    }
    std::vector<int> elem_starts;
    std::vector<int> elem_counts;
    std::vector<SolidMaterialProperties> mats;
    elem_starts.reserve(n_mesh);
    elem_counts.reserve(n_mesh);
    mats.reserve(n_mesh);

    for (int i = 0; i < n_mesh; ++i) {
      if (!mesh_manager.MeshHasMaterial(i)) {
        std::cerr << "ApplyMaterialsFromMeshManager: mesh " << i
                  << " has no material assigned." << std::endl;
        return;
      }
      const auto& inst = mesh_manager.GetMeshInstance(i);
      elem_starts.push_back(inst.element_offset);
      elem_counts.push_back(inst.num_elements);
      mats.push_back(mesh_manager.GetMeshMaterial(i));
    }
    ApplyMaterialsByElementRanges(elem_starts, elem_counts, mats);
  }

  void SetExternalForce(const Eigen::VectorXd &h_f_ext) {
    if (h_f_ext.size() != n_coef * 3) {
      std::cerr << "External force vector size mismatch." << std::endl;
      return;
    }

    cudaMemset(d_f_ext, 0, n_coef * 3 * sizeof(double));
    HANDLE_ERROR(cudaMemcpy(d_f_ext, h_f_ext.data(),
                            n_coef * 3 * sizeof(double),
                            cudaMemcpyHostToDevice));
  }

  // Device pointer accessors for unified state buffer synchronization.
  double* GetX12DevicePtr() { return d_h_x12; }
  const double* GetX12DevicePtr() const { return d_h_x12; }
  double* GetY12DevicePtr() { return d_h_y12; }
  const double* GetY12DevicePtr() const { return d_h_y12; }
  double* GetZ12DevicePtr() { return d_h_z12; }
  const double* GetZ12DevicePtr() const { return d_h_z12; }
  double* GetX12JacDevicePtr() { return d_h_x12_jac; }
  const double* GetX12JacDevicePtr() const { return d_h_x12_jac; }
  double* GetY12JacDevicePtr() { return d_h_y12_jac; }
  const double* GetY12JacDevicePtr() const { return d_h_y12_jac; }
  double* GetZ12JacDevicePtr() { return d_h_z12_jac; }
  const double* GetZ12JacDevicePtr() const { return d_h_z12_jac; }
  double* GetExternalForceDevicePtr() { return d_f_ext; }
  const double* GetExternalForceDevicePtr() const { return d_f_ext; }

  /**
   * Update node positions on GPU (for prescribed motion of fixed nodes).
   */
  void UpdatePositions(const Eigen::VectorXd &h_x12,
                       const Eigen::VectorXd &h_y12,
                       const Eigen::VectorXd &h_z12) {
    if (h_x12.size() != n_coef || h_y12.size() != n_coef ||
        h_z12.size() != n_coef) {
      std::cerr << "Position vector size mismatch." << std::endl;
      return;
    }
    HANDLE_ERROR(cudaMemcpy(d_h_x12, h_x12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_h_y12, h_y12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_h_z12, h_z12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
  }

  void UpdateConstraintTargets(const Eigen::VectorXd &h_x12,
                               const Eigen::VectorXd &h_y12,
                               const Eigen::VectorXd &h_z12) {
    if (h_x12.size() != n_coef || h_y12.size() != n_coef ||
        h_z12.size() != n_coef) {
      std::cerr << "Position vector size mismatch." << std::endl;
      return;
    }
    HANDLE_ERROR(cudaMemcpy(d_h_x12_jac, h_x12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_h_y12_jac, h_y12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_h_z12_jac, h_z12.data(), n_coef * sizeof(double),
                            cudaMemcpyHostToDevice));
  }

  void SetNodalFixed(const Eigen::VectorXi &fixed_nodes);

  /**
   * Update fixed nodes for dynamic constraint changes (e.g., moving grippers).
   * This reuses existing constraint buffers if the number of fixed nodes
   * matches, otherwise reallocates. After calling this, you must call
   * CalcConstraintData() and rebuild constraint Jacobians (CSR) if needed.
   */
  void UpdateNodalFixed(const Eigen::VectorXi &fixed_nodes);

  // Free memory
  void Destroy() {
    HANDLE_ERROR(cudaFree(d_h_x12));
    HANDLE_ERROR(cudaFree(d_h_y12));
    HANDLE_ERROR(cudaFree(d_h_z12));

    HANDLE_ERROR(cudaFree(d_h_x12_jac));
    HANDLE_ERROR(cudaFree(d_h_y12_jac));
    HANDLE_ERROR(cudaFree(d_h_z12_jac));

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

    if (is_j_csr_setup) {
      HANDLE_ERROR(cudaFree(d_j_csr_offsets));
      HANDLE_ERROR(cudaFree(d_j_csr_columns));
      HANDLE_ERROR(cudaFree(d_j_csr_values));
      HANDLE_ERROR(cudaFree(d_j_nnz));
    }

    HANDLE_ERROR(cudaFree(d_tet5pt_x));
    HANDLE_ERROR(cudaFree(d_tet5pt_y));
    HANDLE_ERROR(cudaFree(d_tet5pt_z));
    HANDLE_ERROR(cudaFree(d_tet5pt_weights));

    HANDLE_ERROR(cudaFree(d_grad_N_ref));
    HANDLE_ERROR(cudaFree(d_detJ_ref));

    HANDLE_ERROR(cudaFree(d_F));
    HANDLE_ERROR(cudaFree(d_P));
    HANDLE_ERROR(cudaFree(d_Fdot));
    HANDLE_ERROR(cudaFree(d_P_vis));
    HANDLE_ERROR(cudaFree(d_vm_stress));
    HANDLE_ERROR(cudaFree(d_f_int));
    HANDLE_ERROR(cudaFree(d_f_ext));

    HANDLE_ERROR(cudaFree(d_rho0));
    if (d_rho0_elem != nullptr) {
      HANDLE_ERROR(cudaFree(d_rho0_elem));
      d_rho0_elem = nullptr;
    }
    HANDLE_ERROR(cudaFree(d_nu));
    HANDLE_ERROR(cudaFree(d_E));
    HANDLE_ERROR(cudaFree(d_lambda));
    HANDLE_ERROR(cudaFree(d_mu));
    HANDLE_ERROR(cudaFree(d_material_model));
    HANDLE_ERROR(cudaFree(d_mu10));
    HANDLE_ERROR(cudaFree(d_mu01));
    HANDLE_ERROR(cudaFree(d_kappa));
    HANDLE_ERROR(cudaFree(d_eta_damp));
    HANDLE_ERROR(cudaFree(d_lambda_damp));

    if (d_mesh_materials != nullptr) {
      HANDLE_ERROR(cudaFree(d_mesh_materials));
      d_mesh_materials = nullptr;
    }
    if (d_mesh_elem_starts != nullptr) {
      HANDLE_ERROR(cudaFree(d_mesh_elem_starts));
      d_mesh_elem_starts = nullptr;
    }
    if (d_mesh_elem_ends != nullptr) {
      HANDLE_ERROR(cudaFree(d_mesh_elem_ends));
      d_mesh_elem_ends = nullptr;
    }
    n_mesh_materials = 0;

    HANDLE_ERROR(cudaFree(d_data));

    if (is_constraints_setup) {
      HANDLE_ERROR(cudaFree(d_constraint));
      HANDLE_ERROR(cudaFree(d_fixed_nodes));
    }
  }

  double *Get_Constraint_Ptr() {
    return d_constraint;
  }

  bool Get_Is_Constraint_Setup() {
    return is_constraints_setup;
  }

  GPU_FEAT10_Data *d_data;  // Storing GPU copy of SAPGPUData

  int n_elem;
  int n_coef;
  int n_constraint;

 private:
  // Node positions (global, or per element)
  double *d_h_x12, *d_h_y12, *d_h_z12;  // (n_coef, 1)
  double *d_h_x12_jac, *d_h_y12_jac, *d_h_z12_jac;

  // Element connectivity
  int *d_element_connectivity;  // (n_elem, 10)

  // Mass Matrix
  // Mass Matrix in CSR format
  int *d_csr_offsets, *d_csr_columns;
  double *d_csr_values;
  int *d_nnz;

  // Quadrature points and weights
  double *d_tet5pt_x, *d_tet5pt_y, *d_tet5pt_z;
  double *d_tet5pt_weights;  // (5,)

  // Precomputed reference gradients
  double *d_grad_N_ref;  // (n_elem, 5, 10, 3)
  double *d_detJ_ref;    // (n_elem, 5)

  // Deformation gradient and Piola stress
  double *d_F;  // (n_elem, n_qp, 3, 3)
  double *d_P;  // (n_elem, n_qp, 3, 3)
  // Time-derivative of deformation gradient and viscous Piola
  double *d_Fdot;   // (n_elem, n_qp, 3, 3)
  double *d_P_vis;  // (n_elem, n_qp, 3, 3)

  // Per-element von Mises stress (averaged over QPs)
  double *d_vm_stress = nullptr;  // (n_elem)

  // Material properties
  double *d_E, *d_nu, *d_rho0, *d_lambda, *d_mu;
  float* d_rho0_elem = nullptr;
  int *d_material_model;
  double *d_mu10, *d_mu01, *d_kappa;
  // Damping parameters
  double *d_eta_damp, *d_lambda_damp;
  // Optional per-mesh material table (one copy per mesh) and element range lookup.
  FEAT10MeshMaterial* d_mesh_materials = nullptr;
  int* d_mesh_elem_starts              = nullptr;
  int* d_mesh_elem_ends                = nullptr;
  int n_mesh_materials                 = 0;

  // Constraint data
  double *d_constraint;
  int *d_fixed_nodes;
  // Constraint Jacobian J^T in CSR format
  int *d_cj_csr_offsets, *d_cj_csr_columns;
  double *d_cj_csr_values;
  int *d_cj_nnz;

  // Constraint Jacobian J in CSR format
  int *d_j_csr_offsets, *d_j_csr_columns;
  double *d_j_csr_values;
  int *d_j_nnz;

  // Force vectors
  double *d_f_int, *d_f_ext;  // (n_nodes*3)

  bool is_setup             = false;
  bool is_constraints_setup = false;
  bool is_csr_setup         = false;
  bool is_cj_csr_setup      = false;
  bool is_j_csr_setup       = false;
};
