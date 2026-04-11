#pragma once
/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli
 * File:    SyncedPCG.cuh
 * Brief:   Declares the SyncedPCGSolver class for a fully synchronized
 *          inexact Newton method using Preconditioned Conjugate Gradient
 *          (PCG) with selectable diagonal preconditioners, including
 *          scalar Jacobi and 3x3 block Jacobi. Replaces cuDSS direct
 *          factorization from SyncedNewton with iterative cuSPARSE
 *          SpMV-based PCG.
 *==============================================================
 *==============================================================*/

#include <cublas_v2.h>
#include <cusparse.h>

#include <Eigen/Dense>

#include <iostream>

#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ElementBase.h"
#include "../elements/FEAT10Data.cuh"
#include "SolverBase.h"

enum class SyncedPCGPreconditioner {
  kBlockJacobi,
  kJacobi,
};

struct SyncedPCGParams {
  double inner_atol, inner_rtol, outer_tol, rho;
  int max_outer, max_inner;
  double time_step;
  int max_pcg_iter;
  double pcg_rtol;
  double precond_eps;
  SyncedPCGPreconditioner preconditioner;
};

class SyncedPCGSolver : public SolverBase {
 public:
  SyncedPCGSolver(ElementBase *data, int n_constraints)
      : h_data_(data),
        n_coef_(data->get_n_coef()),
        n_beam_(data->get_n_beam()),
        n_constraints_(n_constraints),
        sparse_hessian_initialized_(false),
        h_nnz_(0),
        d_csr_row_offsets_(nullptr),
        d_csr_col_indices_(nullptr),
        d_csr_values_(nullptr),
        spmv_mat_descr_(nullptr),
        spmv_vec_in_descr_(nullptr),
        spmv_vec_out_descr_(nullptr),
        d_spmv_buffer_(nullptr),
        spmv_buffer_size_(0),
        spmv_initialized_(false) {
    if (data->type == TYPE_3243) {
      type_            = TYPE_3243;
      auto *typed_data = static_cast<GPU_ANCF3243_Data *>(data);
      d_data_          = typed_data->d_data;
      n_total_qp_      = Quadrature::N_TOTAL_QP_3_2_2;
      n_shape_         = Quadrature::N_SHAPE_3243;
      typed_data->CalcDsDuPre();
    } else if (data->type == TYPE_3443) {
      type_            = TYPE_3443;
      auto *typed_data = static_cast<GPU_ANCF3443_Data *>(data);
      d_data_          = typed_data->d_data;
      n_total_qp_      = Quadrature::N_TOTAL_QP_4_4_3;
      n_shape_         = Quadrature::N_SHAPE_3443;
      typed_data->CalcDsDuPre();
    } else if (data->type == TYPE_T10) {
      type_            = TYPE_T10;
      auto *typed_data = static_cast<GPU_FEAT10_Data *>(data);
      d_data_          = typed_data->d_data;
      n_total_qp_      = Quadrature::N_QP_T10_5;
      n_shape_         = Quadrature::N_NODE_T10_10;
    } else {
      d_data_ = nullptr;
      std::cerr << "Unknown element type!" << std::endl;
    }

    if (d_data_ == nullptr) {
      std::cerr << "d_data_ is null in SyncedPCGSolver constructor" << std::endl;
    }

    cudaMalloc(&d_v_guess_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_v_prev_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_lambda_guess_, n_constraints_ * sizeof(double));
    cudaMalloc(&d_g_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_time_step_, sizeof(double));
    cudaMalloc(&d_solver_rho_, sizeof(double));

    cudaMalloc(&d_pcg_solver_, sizeof(SyncedPCGSolver));

    cudaMalloc(&d_x12_prev, n_coef_ * sizeof(double));
    cudaMalloc(&d_y12_prev, n_coef_ * sizeof(double));
    cudaMalloc(&d_z12_prev, n_coef_ * sizeof(double));

    // PCG workspace
    cudaMalloc(&d_delta_v_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_r_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_z_pcg_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_p_pcg_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_Ap_pcg_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_precond_inv_, n_coef_ * 9 * sizeof(double));
    cudaMalloc(&d_rz_, sizeof(double));
    cudaMalloc(&d_pAp_, sizeof(double));
    cudaMalloc(&d_alpha_cg_, sizeof(double));
    cudaMalloc(&d_beta_cg_, sizeof(double));

    // Constraint pointer
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

    cublasCreate(&cublas_handle_);
    cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE);
    cudaMalloc(&d_norm_temp_, sizeof(double));

    cusparseCreate(&cusparse_handle_);
  }

  ~SyncedPCGSolver() {
    cudaFree(d_v_guess_);
    cudaFree(d_v_prev_);
    cudaFree(d_lambda_guess_);
    cudaFree(d_g_);
    cudaFree(d_time_step_);
    cudaFree(d_solver_rho_);

    cudaFree(d_pcg_solver_);

    cudaFree(d_x12_prev);
    cudaFree(d_y12_prev);
    cudaFree(d_z12_prev);

    cudaFree(d_delta_v_);
    cudaFree(d_r_);
    cudaFree(d_z_pcg_);
    cudaFree(d_p_pcg_);
    cudaFree(d_Ap_pcg_);
    cudaFree(d_precond_inv_);
    cudaFree(d_rz_);
    cudaFree(d_pAp_);
    cudaFree(d_alpha_cg_);
    cudaFree(d_beta_cg_);

    if (d_csr_row_offsets_)
      cudaFree(d_csr_row_offsets_);
    if (d_csr_col_indices_)
      cudaFree(d_csr_col_indices_);
    if (d_csr_values_)
      cudaFree(d_csr_values_);
    if (cublas_handle_)
      cublasDestroy(cublas_handle_);
    if (d_norm_temp_)
      cudaFree(d_norm_temp_);

    // cuSPARSE cleanup
    if (d_spmv_buffer_)
      cudaFree(d_spmv_buffer_);
    if (spmv_vec_out_descr_)
      cusparseDestroyDnVec(spmv_vec_out_descr_);
    if (spmv_vec_in_descr_)
      cusparseDestroyDnVec(spmv_vec_in_descr_);
    if (spmv_mat_descr_)
      cusparseDestroySpMat(spmv_mat_descr_);
    if (cusparse_handle_)
      cusparseDestroy(cusparse_handle_);
  }

  void SetParameters(void *params) override {
    SyncedPCGParams *p = static_cast<SyncedPCGParams *>(params);

    h_inner_atol_ = p->inner_atol;
    h_inner_rtol_ = p->inner_rtol;
    h_outer_tol_  = p->outer_tol;

    h_max_outer_ = p->max_outer;
    h_max_inner_ = p->max_inner;

    h_max_pcg_iter_ = p->max_pcg_iter;
    h_pcg_rtol_     = p->pcg_rtol;
    h_precond_eps_  = p->precond_eps;
    h_preconditioner_ = p->preconditioner;

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

    HANDLE_ERROR(cudaMemcpy(d_pcg_solver_, this, sizeof(SyncedPCGSolver),
                            cudaMemcpyHostToDevice));
  }

#if defined(__CUDACC__)
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
  __device__ int gpu_n_constraints() { return n_constraints_; }
  __device__ int gpu_n_total_qp() { return n_total_qp_; }
  __device__ int gpu_n_shape() { return n_shape_; }

  __device__ double *solver_rho() { return d_solver_rho_; }
  __device__ double solver_time_step() const { return *d_time_step_; }

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
#endif

  __host__ __device__ int get_n_coef() const { return n_coef_; }
  __host__ __device__ int get_n_beam() const { return n_beam_; }

  double *GetVelocityGuessDevicePtr() const { return d_v_guess_; }

  void SetInitialVelocity(const Eigen::VectorXd &h_v0) {
    if (h_v0.size() != n_coef_ * 3) {
      std::cerr << "SetInitialVelocity: size mismatch (got " << h_v0.size()
                << ", expected " << (n_coef_ * 3) << ")\n";
      return;
    }
    HANDLE_ERROR(cudaMemcpy(d_v_guess_, h_v0.data(),
                            static_cast<size_t>(n_coef_) * 3 * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_v_prev_, h_v0.data(),
                            static_cast<size_t>(n_coef_) * 3 * sizeof(double),
                            cudaMemcpyHostToDevice));
  }

  void SetInitialVelocityFromDevicePtr(const double *d_v0) {
    if (d_v0 == nullptr) {
      std::cerr << "SetInitialVelocityFromDevicePtr: null input pointer\n";
      return;
    }
    HANDLE_ERROR(cudaMemcpy(d_v_guess_, d_v0,
                            static_cast<size_t>(n_coef_) * 3 * sizeof(double),
                            cudaMemcpyDeviceToDevice));
    HANDLE_ERROR(cudaMemcpy(d_v_prev_, d_v0,
                            static_cast<size_t>(n_coef_) * 3 * sizeof(double),
                            cudaMemcpyDeviceToDevice));
  }

  void OneStepPCG();

  void Solve() override { OneStepPCG(); }

  double compute_l2_norm_cublas(double *d_vec, int n_dofs);
  void AnalyzeHessianSparsity();

 private:
  void SetupCuSPARSE();
  void PCGSolve(int n_dofs);

  ElementBase *h_data_;
  ElementType type_;
  ElementBase *d_data_;
  SyncedPCGSolver *d_pcg_solver_;
  int n_total_qp_, n_shape_;
  int n_coef_, n_beam_, n_constraints_;

  double *d_x12_prev, *d_y12_prev, *d_z12_prev;

  double *d_v_guess_, *d_v_prev_;
  double *d_lambda_guess_, *d_g_;
  double *d_time_step_, *d_solver_rho_;
  double h_inner_atol_, h_outer_tol_, h_inner_rtol_;
  int h_max_outer_, h_max_inner_;
  double *d_delta_v_, *d_r_;
  double *d_alpha_cg_, *d_beta_cg_;

  double *d_constraint_ptr_;

  // Sparse Hessian members
  bool sparse_hessian_initialized_;
  int h_nnz_;
  int *d_csr_row_offsets_;
  int *d_csr_col_indices_;
  double *d_csr_values_;

  // cuBLAS
  cublasHandle_t cublas_handle_;
  double *d_norm_temp_;

  // cuSPARSE for SpMV
  cusparseHandle_t cusparse_handle_;
  cusparseSpMatDescr_t spmv_mat_descr_;
  cusparseDnVecDescr_t spmv_vec_in_descr_;
  cusparseDnVecDescr_t spmv_vec_out_descr_;
  void *d_spmv_buffer_;
  size_t spmv_buffer_size_;
  bool spmv_initialized_;

  // PCG workspace
  double *d_z_pcg_;        // Preconditioned residual (n_coef*3)
  double *d_p_pcg_;        // Search direction (n_coef*3)
  double *d_Ap_pcg_;       // H*p result (n_coef*3)
  double *d_precond_inv_;  // Inverted 3x3 diagonal blocks (n_coef*9)
  double *d_rz_;           // r^T z scalar
  double *d_pAp_;          // p^T (H*p) scalar

  // PCG parameters (host copies)
  int h_max_pcg_iter_;
  double h_pcg_rtol_;
  double h_precond_eps_;
  SyncedPCGPreconditioner h_preconditioner_;
};
