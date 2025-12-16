#pragma once
/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    SyncedNewton.cuh
 * Brief:   Declares the SyncedNewtonSolver class for a fully synchronized
 *          Newton method without line search. Manages GPU buffers for
 *          velocities, residuals, sparse Hessian storage, and persistent
 *          cuBLAS / cuDSS handles, and exposes device accessors used by the
 *          Newton kernels and linear-solve routines.
 *==============================================================
 *==============================================================*/

#include <cublas_v2.h>  // cuBLAS header

#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ElementBase.h"
#include "../elements/FEAT10Data.cuh"
#include "SolverBase.h"

// Plain Newton solver - no line search
// Fully synced, computes full gradient and Hessian per iteration

struct SyncedNewtonParams {
  double inner_atol, inner_rtol, outer_tol, rho;
  int max_outer, max_inner;
  double time_step;
};

class SyncedNewtonSolver : public SolverBase {
 public:
  SyncedNewtonSolver(ElementBase *data, int n_constraints)
      : n_coef_(data->get_n_coef()),
        n_beam_(data->get_n_beam()),
        n_constraints_(n_constraints),
        sparse_hessian_initialized_(false),
        h_nnz_(0),
        d_csr_row_offsets_(nullptr),
        d_csr_col_indices_(nullptr),
        d_csr_values_(nullptr),
        d_col_bitset_(nullptr),
        d_nnz_per_row_(nullptr),
        fixed_sparsity_pattern_(false),
        analysis_done_(false) {
    // Type-based casting to get the correct d_data from derived class
    if (data->type == TYPE_3243) {
      type_            = TYPE_3243;
      auto *typed_data = static_cast<GPU_ANCF3243_Data *>(data);
      d_data_ = typed_data->d_data;  // This accesses the derived class's d_data
      n_total_qp_ = Quadrature::N_TOTAL_QP_3_2_2;
      n_shape_    = Quadrature::N_SHAPE_3243;
    } else if (data->type == TYPE_3443) {
      type_            = TYPE_3443;
      auto *typed_data = static_cast<GPU_ANCF3443_Data *>(data);
      d_data_ = typed_data->d_data;  // This accesses the derived class's d_data
      n_total_qp_ = Quadrature::N_TOTAL_QP_4_4_3;
      n_shape_    = Quadrature::N_SHAPE_3443;
    } else if (data->type == TYPE_T10) {
      type_            = TYPE_T10;
      auto *typed_data = static_cast<GPU_FEAT10_Data *>(data);
      d_data_ = typed_data->d_data;  // This accesses the derived class's d_data
      n_total_qp_ = Quadrature::N_QP_T10_5;
      n_shape_    = Quadrature::N_NODE_T10_10;
    } else {
      d_data_ = nullptr;
      std::cerr << "Unknown element type!" << std::endl;
    }

    if (d_data_ == nullptr) {
      std::cerr << "d_data_ is null in SyncedNewtonSolver constructor"
                << std::endl;
    }

    cudaMalloc(&d_v_guess_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_v_prev_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_lambda_guess_, n_constraints_ * sizeof(double));
    cudaMalloc(&d_g_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_dv_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_norm_g_, sizeof(double));
    cudaMalloc(&d_inner_flag_, sizeof(int));
    cudaMalloc(&d_outer_flag_, sizeof(int));
    cudaMalloc(&d_inner_atol_, sizeof(double));
    cudaMalloc(&d_outer_tol_, sizeof(double));
    cudaMalloc(&d_inner_rtol_, sizeof(double));
    cudaMalloc(&d_max_outer_, sizeof(int));
    cudaMalloc(&d_max_inner_, sizeof(int));
    cudaMalloc(&d_time_step_, sizeof(double));
    cudaMalloc(&d_solver_rho_, sizeof(double));

    cudaMalloc(&d_newton_solver_, sizeof(SyncedNewtonSolver));

    cudaMalloc(&d_x12_prev, n_coef_ * sizeof(double));
    cudaMalloc(&d_y12_prev, n_coef_ * sizeof(double));
    cudaMalloc(&d_z12_prev, n_coef_ * sizeof(double));

    cudaMalloc(&d_delta_v_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_r_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_r_dot_r_, sizeof(double));
    cudaMalloc(&d_alpha_cg_, sizeof(double));
    cudaMalloc(&d_beta_cg_, sizeof(double));

    // If data is a T10 and constraint is setup, copy over the constraint ptr
    if (type_ == TYPE_T10) {
      if (static_cast<GPU_FEAT10_Data *>(data)->Get_Is_Constraint_Setup()) {
        d_constraint_ptr_ =
            static_cast<GPU_FEAT10_Data *>(data)->Get_Constraint_Ptr();
      } else {
        d_constraint_ptr_ = nullptr;
      }
    }

    if (type_ == TYPE_3243) {
      if (static_cast<GPU_ANCF3243_Data *>(data)->Get_Is_Constraint_Setup()) {
        d_constraint_ptr_ =
            static_cast<GPU_ANCF3243_Data *>(data)->Get_Constraint_Ptr();
      } else {
        d_constraint_ptr_ = nullptr;
      }
    }

    if (type_ == TYPE_3443) {
      if (static_cast<GPU_ANCF3443_Data *>(data)->Get_Is_Constraint_Setup()) {
        d_constraint_ptr_ =
            static_cast<GPU_ANCF3443_Data *>(data)->Get_Constraint_Ptr();
      } else {
        d_constraint_ptr_ = nullptr;
      }
    }

    // Create persistent linear algebra handles
    cublasCreate(&cublas_handle_);
    cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE);

    cudaMalloc(&d_norm_temp_, sizeof(double));

    // cuDSS handles created in Setup() after sparsity analysis
    cudss_handle_ = nullptr;
    cudss_config_ = nullptr;
    cudss_data_   = nullptr;
  }

  ~SyncedNewtonSolver() {
    cudaFree(d_v_guess_);
    cudaFree(d_v_prev_);
    cudaFree(d_lambda_guess_);
    cudaFree(d_g_);
    cudaFree(d_dv_);
    cudaFree(d_norm_g_);
    cudaFree(d_inner_flag_);
    cudaFree(d_outer_flag_);
    cudaFree(d_inner_atol_);
    cudaFree(d_outer_tol_);
    cudaFree(d_inner_rtol_);
    cudaFree(d_max_outer_);
    cudaFree(d_max_inner_);
    cudaFree(d_time_step_);
    cudaFree(d_solver_rho_);

    cudaFree(d_newton_solver_);

    cudaFree(d_x12_prev);
    cudaFree(d_y12_prev);
    cudaFree(d_z12_prev);

    cudaFree(d_delta_v_);
    cudaFree(d_r_);
    cudaFree(d_r_dot_r_);
    cudaFree(d_alpha_cg_);
    cudaFree(d_beta_cg_);

    // Free sparse Hessian structures
    if (d_csr_row_offsets_)
      cudaFree(d_csr_row_offsets_);
    if (d_csr_col_indices_)
      cudaFree(d_csr_col_indices_);
    if (d_csr_values_)
      cudaFree(d_csr_values_);
    if (d_col_bitset_)
      cudaFree(d_col_bitset_);
    if (d_nnz_per_row_)
      cudaFree(d_nnz_per_row_);

    // Destroy persistent linear algebra handles
    if (cublas_handle_)
      cublasDestroy(cublas_handle_);
    if (d_norm_temp_)
      cudaFree(d_norm_temp_);

    if (cudss_data_)
      cudssDataDestroy(cudss_handle_, cudss_data_);
    if (cudss_config_)
      cudssConfigDestroy(cudss_config_);
    if (cudss_handle_)
      cudssDestroy(cudss_handle_);
  }

  void SetParameters(void *params) override {
    SyncedNewtonParams *p = static_cast<SyncedNewtonParams *>(params);

    h_inner_atol_ = p->inner_atol;
    h_inner_rtol_ = p->inner_rtol;
    h_outer_tol_ = p->outer_tol;

    h_max_outer_ = p->max_outer;
    h_max_inner_ = p->max_inner;

    cudaMemcpy(d_inner_atol_, &p->inner_atol, sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_inner_rtol_, &p->inner_rtol, sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_outer_tol_, &p->outer_tol, sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_outer_, &p->max_outer, sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_inner_, &p->max_inner, sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_time_step_, &p->time_step, sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_solver_rho_, &p->rho, sizeof(double), cudaMemcpyHostToDevice);
  }

  void Setup() {
    cudaMemset(d_x12_prev, 0, n_coef_ * sizeof(double));
    cudaMemset(d_y12_prev, 0, n_coef_ * sizeof(double));
    cudaMemset(d_z12_prev, 0, n_coef_ * sizeof(double));

    cudaMemset(d_v_guess_, 0, n_coef_ * 3 * sizeof(double));
    cudaMemset(d_v_prev_, 0, n_coef_ * 3 * sizeof(double));
    cudaMemset(d_lambda_guess_, 0, n_constraints_ * sizeof(double));
    cudaMemset(d_g_, 0, n_coef_ * 3 * sizeof(double));
    cudaMemset(d_dv_, 0, n_coef_ * 3 * sizeof(double));
    cudaMemset(d_r_dot_r_, 0, sizeof(double));

    HANDLE_ERROR(cudaMemcpy(d_newton_solver_, this, sizeof(SyncedNewtonSolver),
                            cudaMemcpyHostToDevice));
  }

#if defined(__CUDACC__)
  // Device accessors (define as __device__ in .cuh or .cu as needed)
  __device__ Eigen::Map<Eigen::VectorXd> v_guess() {
    return Eigen::Map<Eigen::VectorXd>(d_v_guess_, n_coef_ * 3);
  }
  __device__ Eigen::Map<Eigen::VectorXd> v_prev() {
    return Eigen::Map<Eigen::VectorXd>(d_v_prev_, n_coef_ * 3);
  }
  __device__ Eigen::Map<Eigen::VectorXd> lambda_guess() {
    return Eigen::Map<Eigen::VectorXd>(d_lambda_guess_, n_constraints_);
  }
  __device__ Eigen::Map<Eigen::VectorXd> g() {
    return Eigen::Map<Eigen::VectorXd>(d_g_, 3 * n_coef_);
  }
  __device__ Eigen::Map<Eigen::VectorXd> dv() {
    return Eigen::Map<Eigen::VectorXd>(d_dv_, 3 * n_coef_);
  }

  __device__ int gpu_n_constraints() {
    return n_constraints_;
  }
  __device__ int gpu_n_total_qp() {
    return n_total_qp_;
  }
  __device__ int gpu_n_shape() {
    return n_shape_;
  }

  __device__ double *norm_g() {
    return d_norm_g_;
  }
  __device__ int *inner_flag() {
    return d_inner_flag_;
  }
  __device__ int *outer_flag() {
    return d_outer_flag_;
  }
  __device__ double *solver_rho() {
    return d_solver_rho_;
  }
  __device__ double solver_inner_atol() const {
    return *d_inner_atol_;
  }
  __device__ double solver_outer_tol() const {
    return *d_outer_tol_;
  }
  __device__ int solver_max_outer() const {
    return *d_max_outer_;
  }
  __device__ int solver_max_inner() const {
    return *d_max_inner_;
  }
  __device__ double solver_time_step() const {
    return *d_time_step_;
  }

  __device__ Eigen::Map<Eigen::VectorXd> x12_prev() {
    return Eigen::Map<Eigen::VectorXd>(d_x12_prev, n_coef_);
  }
  __device__ Eigen::Map<Eigen::VectorXd> y12_prev() {
    return Eigen::Map<Eigen::VectorXd>(d_y12_prev, n_coef_);
  }
  __device__ Eigen::Map<Eigen::VectorXd> z12_prev() {
    return Eigen::Map<Eigen::VectorXd>(d_z12_prev, n_coef_);
  }
  __device__ Eigen::Map<Eigen::VectorXd> delta_v() {
    return Eigen::Map<Eigen::VectorXd>(d_delta_v_, 3 * n_coef_);
  }
  __device__ Eigen::Map<Eigen::VectorXd> r() {
    return Eigen::Map<Eigen::VectorXd>(d_r_, 3 * n_coef_);
  }
  __device__ double *r_dot_r() {
    return d_r_dot_r_;
  }
  __device__ double *alpha_cg() {
    return d_alpha_cg_;
  }
  __device__ double *beta_cg() {
    return d_beta_cg_;
  }
#endif

  __host__ __device__ int get_n_coef() const {
    return n_coef_;
  }
  __host__ __device__ int get_n_beam() const {
    return n_beam_;
  }

  // Host accessor for device velocity guess pointer (layout: [vx0, vy0, vz0, ..])
  double* GetVelocityGuessDevicePtr() const {
    return d_v_guess_;
  }

  void OneStepNewtonCuDSS();

  void Solve() override {
    OneStepNewtonCuDSS();
  }

  void SetFixedSparsityPattern(bool fixed) {
    fixed_sparsity_pattern_ = fixed;
  }

  double compute_l2_norm_cublas(double *d_vec, int n_dofs);
  void AnalyzeHessianSparsity();

 private:
  ElementType type_;
  ElementBase *d_data_;
  SyncedNewtonSolver *d_newton_solver_;
  int n_total_qp_, n_shape_;
  int n_coef_, n_beam_, n_constraints_;

  double *d_x12_prev, *d_y12_prev, *d_z12_prev;

  double *d_v_guess_, *d_v_prev_;
  double *d_lambda_guess_, *d_g_, *d_dv_;
  double *d_norm_g_;
  int *d_inner_flag_, *d_outer_flag_;
  double *d_inner_atol_, *d_inner_rtol_, *d_outer_tol_, *d_time_step_,
      *d_solver_rho_;
  double h_inner_atol_, h_outer_tol_, h_inner_rtol_;
  int h_max_outer_, h_max_inner_;
  int *d_max_inner_, *d_max_outer_;
  double *d_delta_v_, *d_r_;
  double *d_r_dot_r_;                // Scalar for dot product storage
  double *d_alpha_cg_, *d_beta_cg_;  // CG scalars

  double *d_constraint_ptr_;

  // Sparse Hessian members
  bool sparse_hessian_initialized_;
  int h_nnz_;                   // Total number of non-zeros (host copy)
  int *d_csr_row_offsets_;      // Size: n_dofs + 1
  int *d_csr_col_indices_;      // Size: nnz
  double *d_csr_values_;        // Size: nnz
  unsigned int *d_col_bitset_;  // Temporary for sparsity analysis
  int *d_nnz_per_row_;          // Temporary workspace

  // Persistent library handles (reused across iterations)
  cublasHandle_t cublas_handle_;
  cudssHandle_t cudss_handle_;
  cudssConfig_t cudss_config_;
  cudssData_t cudss_data_;
  double *d_norm_temp_;  // Reusable temp for cuBLAS norms

  // Analysis reuse flags
  bool fixed_sparsity_pattern_;  // User-set flag: if true, matrix structure is fixed
  bool analysis_done_;           // Internal state: tracks if analysis has been performed
};
