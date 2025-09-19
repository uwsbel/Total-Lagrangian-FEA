#pragma once
#include "../../lib_utils/quadrature_utils.h"
#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ElementBase.h"
#include "SolverBase.h"

// this is a true first order Nesterov method
// fully synced, and each inner iteration will compute the full gradient

struct SyncedNesterovParams {
  double alpha, rho, inner_tol, outer_tol;
  int max_outer, max_inner;
  double time_step;
};

class SyncedNesterovSolver : public SolverBase {
public:
  SyncedNesterovSolver(ElementBase *data, int n_constraints)
      : n_coef_(data->get_n_coef()), n_beam_(data->get_n_beam()),
        n_constraints_(n_constraints) {
    // Type-based casting to get the correct d_data from derived class
    if (data->type == TYPE_3243) {
      auto *typed_data = static_cast<GPU_ANCF3243_Data *>(data);
      d_data_ = typed_data->d_data; // This accesses the derived class's d_data
      n_total_qp_ = Quadrature::N_TOTAL_QP_3_2_2;
      n_shape_ = Quadrature::N_SHAPE_3243;
    } else if (data->type == TYPE_3443) {
      auto *typed_data = static_cast<GPU_ANCF3443_Data *>(data);
      d_data_ = typed_data->d_data; // This accesses the derived class's d_data
      n_total_qp_ = Quadrature::N_TOTAL_QP_4_4_3;
      n_shape_ = Quadrature::N_SHAPE_3443;
    } else {
      d_data_ = nullptr;
      std::cerr << "Unknown element type!" << std::endl;
    }

    if (d_data_ == nullptr) {
      std::cerr << "d_data_ is null in SyncedNesterovSolver constructor"
                << std::endl;
    }

    cudaMalloc(&d_v_guess_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_v_prev_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_v_k_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_v_next_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_lambda_guess_, n_constraints_ * sizeof(double));
    cudaMalloc(&d_g_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_prev_norm_g_, sizeof(double));
    cudaMalloc(&d_norm_g_, sizeof(double));
    cudaMalloc(&d_inner_flag_, sizeof(int));
    cudaMalloc(&d_outer_flag_, sizeof(int));
    cudaMalloc(&d_alpha_, sizeof(double));
    cudaMalloc(&d_inner_tol_, sizeof(double));
    cudaMalloc(&d_outer_tol_, sizeof(double));
    cudaMalloc(&d_max_outer_, sizeof(int));
    cudaMalloc(&d_max_inner_, sizeof(int));
    cudaMalloc(&d_time_step_, sizeof(double));
    cudaMalloc(&d_solver_rho_, sizeof(double));

    cudaMalloc(&d_nesterov_solver_, sizeof(SyncedNesterovSolver));

    cudaMalloc(&d_x12_prev, n_coef_ * sizeof(double));
    cudaMalloc(&d_y12_prev, n_coef_ * sizeof(double));
    cudaMalloc(&d_z12_prev, n_coef_ * sizeof(double));
  }

  ~SyncedNesterovSolver() {
    cudaFree(d_v_guess_);
    cudaFree(d_v_prev_);
    cudaFree(d_v_k_);
    cudaFree(d_v_next_);
    cudaFree(d_lambda_guess_);
    cudaFree(d_g_);
    cudaFree(d_prev_norm_g_);
    cudaFree(d_norm_g_);
    cudaFree(d_inner_flag_);
    cudaFree(d_outer_flag_);
    cudaFree(d_alpha_);
    cudaFree(d_inner_tol_);
    cudaFree(d_outer_tol_);
    cudaFree(d_max_outer_);
    cudaFree(d_max_inner_);
    cudaFree(d_time_step_);
    cudaFree(d_solver_rho_);

    cudaFree(d_nesterov_solver_);

    cudaFree(d_x12_prev);
    cudaFree(d_y12_prev);
    cudaFree(d_z12_prev);
  }

  void SetParameters(void *params) override {
    SyncedNesterovParams *p = static_cast<SyncedNesterovParams *>(params);
    cudaMemcpy(d_alpha_, &p->alpha, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inner_tol_, &p->inner_tol, sizeof(double),
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

    cudaMemset(d_v_guess_, 0, n_coef_ * 3 * sizeof(double));
    cudaMemset(d_v_prev_, 0, n_coef_ * 3 * sizeof(double));
    cudaMemset(d_lambda_guess_, 0, n_constraints_ * sizeof(double));
  }

  void Setup() {
    cudaMemset(d_x12_prev, 0, n_coef_ * sizeof(double));
    cudaMemset(d_y12_prev, 0, n_coef_ * sizeof(double));
    cudaMemset(d_z12_prev, 0, n_coef_ * sizeof(double));

    cudaMemset(d_v_guess_, 0, n_coef_ * 3 * sizeof(double));
    cudaMemset(d_v_prev_, 0, n_coef_ * 3 * sizeof(double));
    cudaMemset(d_v_k_, 0, n_coef_ * 3 * sizeof(double));
    cudaMemset(d_v_next_, 0, n_coef_ * 3 * sizeof(double));
    cudaMemset(d_lambda_guess_, 0, n_constraints_ * sizeof(double));
    cudaMemset(d_g_, 0, n_coef_ * 3 * sizeof(double));

    HANDLE_ERROR(cudaMemcpy(d_nesterov_solver_, this,
                            sizeof(SyncedNesterovSolver),
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
  __device__ Eigen::Map<Eigen::VectorXd> v_k() {
    return Eigen::Map<Eigen::VectorXd>(d_v_k_, n_coef_ * 3);
  }
  __device__ Eigen::Map<Eigen::VectorXd> v_next() {
    return Eigen::Map<Eigen::VectorXd>(d_v_next_, n_coef_ * 3);
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

  __device__ double *prev_norm_g() { return d_prev_norm_g_; }
  __device__ double *norm_g() { return d_norm_g_; }
  __device__ int *inner_flag() { return d_inner_flag_; }
  __device__ int *outer_flag() { return d_outer_flag_; }
  __device__ double *solver_rho() { return d_solver_rho_; }
  __device__ double solver_alpha() const { return *d_alpha_; }
  __device__ double solver_inner_tol() const { return *d_inner_tol_; }
  __device__ double solver_outer_tol() const { return *d_outer_tol_; }
  __device__ int solver_max_outer() const { return *d_max_outer_; }
  __device__ int solver_max_inner() const { return *d_max_inner_; }
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
#endif

  __host__ __device__ int get_n_coef() const { return n_coef_; }
  __host__ __device__ int get_n_beam() const { return n_beam_; }

  void OneStepNesterov();

  void Solve() override { OneStepNesterov(); }

private:
  ElementBase *d_data_;
  SyncedNesterovSolver *d_nesterov_solver_;
  int n_total_qp_, n_shape_;
  int n_coef_, n_beam_, n_constraints_;

  double *d_x12_prev, *d_y12_prev, *d_z12_prev;

  double *d_v_guess_, *d_v_prev_, *d_v_k_, *d_v_next_;
  double *d_lambda_guess_, *d_g_;
  double *d_prev_norm_g_, *d_norm_g_;
  int *d_inner_flag_, *d_outer_flag_;
  double *d_alpha_, *d_inner_tol_, *d_outer_tol_, *d_time_step_, *d_solver_rho_;
  int *d_max_inner_, *d_max_outer_;
};