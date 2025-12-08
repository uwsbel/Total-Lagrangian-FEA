#pragma once
#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../elements/ElementBase.h"
#include "../elements/FEAT10Data.cuh"
#include "SolverBase.h"

// This is a shared-memory optimized AdamW method specifically for T10 elements.
// Each block processes 12 elements with 128 threads total (10 threads per
// element, 8 idle). Uses float for shared memory caching with implicit cast
// back to double.

struct SyncedAdamWshmemParams {
  double lr, beta1, beta2, eps, weight_decay, lr_decay;
  double inner_tol, outer_tol, rho;
  int max_outer, max_inner;
  double time_step;
  int convergence_check_interval;
};

// Constants for shared memory kernel
constexpr int THREADS_PER_BLOCK_SHMEM    = 128;
constexpr int THREADS_PER_ELEM           = 10;  // Number of nodes per T10 element
constexpr int ELEMS_PER_BLOCK            = 12;  // 128 / 10 = 12 (8 threads idle)
constexpr int N_QP_T10                   = 5;   // Quadrature points per element
constexpr int N_NODES_T10                = 10;  // Nodes per element

// Shared memory layout per element:
// - grad_N_ref: 10 nodes * 3 components * 5 QP = 150 floats
// - qp_weights: 5 floats
// - detJ_ref: 5 floats
// - x_nodes: 10 * 3 = 30 floats (current nodal positions)
// - connectivity: 10 ints = 10 * sizeof(int)/sizeof(float) 
// Total per element: ~200 floats

class SyncedAdamWshmemSolver : public SolverBase {
 public:
  SyncedAdamWshmemSolver(ElementBase *data, int n_constraints)
      : n_coef_(data->get_n_coef()),
        n_beam_(data->get_n_beam()),
        n_constraints_(n_constraints) {
    // Only supports T10 elements
    if (data->type != TYPE_T10) {
      std::cerr << "SyncedAdamWshmemSolver only supports T10 elements!"
                << std::endl;
      d_data_ = nullptr;
      return;
    }

    type_            = TYPE_T10;
    auto *typed_data = static_cast<GPU_FEAT10_Data *>(data);
    d_data_          = typed_data->d_data;
    n_total_qp_      = Quadrature::N_QP_T10_5;
    n_shape_         = Quadrature::N_NODE_T10_10;

    cudaMalloc(&d_v_guess_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_v_prev_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_v_k_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_v_next_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_lambda_guess_, n_constraints_ * sizeof(double));
    cudaMalloc(&d_g_, n_coef_ * 3 * sizeof(double));
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
    cudaMalloc(&d_convergence_check_interval_, sizeof(int));

    cudaMalloc(&d_adamw_solver_, sizeof(SyncedAdamWshmemSolver));

    cudaMalloc(&d_lr_, sizeof(double));
    cudaMalloc(&d_beta1_, sizeof(double));
    cudaMalloc(&d_beta2_, sizeof(double));
    cudaMalloc(&d_eps_, sizeof(double));
    cudaMalloc(&d_weight_decay_, sizeof(double));
    cudaMalloc(&d_lr_decay_, sizeof(double));

    cudaMalloc(&d_x12_prev, n_coef_ * sizeof(double));
    cudaMalloc(&d_y12_prev, n_coef_ * sizeof(double));
    cudaMalloc(&d_z12_prev, n_coef_ * sizeof(double));

    cudaMalloc(&d_num_steps_, sizeof(int));
  }

  ~SyncedAdamWshmemSolver() {
    cudaFree(d_v_guess_);
    cudaFree(d_v_prev_);
    cudaFree(d_v_k_);
    cudaFree(d_v_next_);
    cudaFree(d_lambda_guess_);
    cudaFree(d_g_);
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
    cudaFree(d_convergence_check_interval_);

    cudaFree(d_lr_);
    cudaFree(d_beta1_);
    cudaFree(d_beta2_);
    cudaFree(d_eps_);
    cudaFree(d_weight_decay_);
    cudaFree(d_lr_decay_);

    cudaFree(d_adamw_solver_);

    cudaFree(d_x12_prev);
    cudaFree(d_y12_prev);
    cudaFree(d_z12_prev);

    cudaFree(d_num_steps_);
  }

  void SetParameters(void *params) override {
    SyncedAdamWshmemParams *p = static_cast<SyncedAdamWshmemParams *>(params);

    cudaMemcpy(d_lr_, &p->lr, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta1_, &p->beta1, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta2_, &p->beta2, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eps_, &p->eps, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_decay_, &p->weight_decay, sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_lr_decay_, &p->lr_decay, sizeof(double),
               cudaMemcpyHostToDevice);

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
    cudaMemcpy(d_convergence_check_interval_, &p->convergence_check_interval,
               sizeof(int), cudaMemcpyHostToDevice);

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

    HANDLE_ERROR(cudaMemcpy(d_adamw_solver_, this,
                            sizeof(SyncedAdamWshmemSolver),
                            cudaMemcpyHostToDevice));
  }

#if defined(__CUDACC__)
  // Device accessors
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
  __device__ double solver_alpha() const {
    return *d_alpha_;
  }
  __device__ double solver_inner_tol() const {
    return *d_inner_tol_;
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
  __device__ double solver_lr() const {
    return *d_lr_;
  }
  __device__ double solver_beta1() const {
    return *d_beta1_;
  }
  __device__ double solver_beta2() const {
    return *d_beta2_;
  }
  __device__ double solver_eps() const {
    return *d_eps_;
  }
  __device__ double solver_weight_decay() const {
    return *d_weight_decay_;
  }
  __device__ double solver_lr_decay() const {
    return *d_lr_decay_;
  }

  __device__ int solver_convergence_check_interval() const {
    return *d_convergence_check_interval_;
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

  __device__ int solver_num_steps() const {
    return *d_num_steps_;
  }
#endif

  __host__ __device__ int get_n_coef() const {
    return n_coef_;
  }
  __host__ __device__ int get_n_beam() const {
    return n_beam_;
  }

  void MultiStepAdamWshmem(int num_steps = 10);

  void Solve() override {
    MultiStepAdamWshmem(10);  // Default: 10 steps
  }

  void Solve(int num_steps) {
    MultiStepAdamWshmem(num_steps);
  }

 private:
  ElementType type_;
  ElementBase *d_data_;
  SyncedAdamWshmemSolver *d_adamw_solver_;
  int n_total_qp_, n_shape_;
  int n_coef_, n_beam_, n_constraints_;

  double *d_x12_prev, *d_y12_prev, *d_z12_prev;

  double *d_v_guess_, *d_v_prev_, *d_v_k_, *d_v_next_;
  double *d_lambda_guess_, *d_g_;
  double *d_norm_g_;
  int *d_inner_flag_, *d_outer_flag_;

  double *d_lr_, *d_beta1_, *d_beta2_, *d_eps_, *d_weight_decay_, *d_lr_decay_;

  double *d_alpha_, *d_inner_tol_, *d_outer_tol_, *d_time_step_, *d_solver_rho_;

  int *d_convergence_check_interval_;

  int *d_max_inner_, *d_max_outer_;

  int *d_num_steps_;
};
