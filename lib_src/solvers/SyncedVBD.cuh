#pragma once
/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    SyncedVBD.cuh
 * Brief:   Declares the SyncedVBDSolver class for a Vertex Block Descent
 *          (VBD) method. This solver uses graph coloring to parallelize
 *          per-node 3×3 block updates within the ALM outer loop.
 *          The current implementation targets FEAT10 (TYPE_T10).
 *          The inner loop performs colored Gauss-Seidel style updates
 *          without assembling a global Hessian matrix.
 *          CUDA graphs are used for the inner sweeps and post-sweep kernels.
 *==============================================================
 *==============================================================*/

#include <cublas_v2.h>

#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/quadrature_utils.h"
#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ElementBase.h"
#include "../elements/FEAT10Data.cuh"
#include "SolverBase.h"

struct SyncedVBDParams {
  double inner_tol, inner_rtol, outer_tol, rho;
  int max_outer, max_inner;
  double time_step;
  double omega;     // Relaxation factor (default 1.0)
  double hess_eps;  // Regularization for local Hessian (default 1e-12)
  int convergence_check_interval;
  int color_group_size;  // Group multiple colors per refresh (default 1)
};

class SyncedVBDSolver : public SolverBase {
 public:
  SyncedVBDSolver(ElementBase *data, int n_constraints)
      : h_data_(data),
        n_coef_(data->get_n_coef()),
        n_beam_(data->get_n_beam()),
        n_constraints_(n_constraints),
        coloring_initialized_(false),
        n_colors_(0),
        d_colors_(nullptr),
        d_color_offsets_(nullptr),
        d_color_nodes_(nullptr),
        d_incidence_offsets_(nullptr),
        d_incidence_data_(nullptr),
        h_color_offsets_cache_(nullptr),
        h_color_offsets_cache_size_(0),
        h_color_group_size_(1),
        n_color_groups_(0),
        h_color_group_offsets_cache_(nullptr),
        h_color_group_offsets_cache_size_(0),
        h_color_group_colors_cache_(nullptr),
        h_color_group_colors_cache_size_(0),
        fixed_map_initialized_(false),
        d_fixed_map_(nullptr),
        d_mass_diag_blocks_(nullptr),
        graph_threads_(0),
        graph_blocks_coef_(0),
        graph_blocks_p_(0),
        graph_n_colors_(0),
        graph_n_constraints_(0),
        graph_color_group_size_(1),
        graph_n_color_groups_(0),
        graph_capture_stream_(nullptr),
        inner_sweep_graph_exec_(nullptr),
        post_outer_graph_exec_(nullptr) {
    // Type-based casting to get the correct d_data from derived class
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
      std::cerr << "d_data_ is null in SyncedVBDSolver constructor"
                << std::endl;
    }

    // Allocate velocity and state buffers
    cudaMalloc(&d_v_guess_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_v_prev_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_lambda_guess_, n_constraints_ * sizeof(double));
    cudaMalloc(&d_g_, n_coef_ * 3 * sizeof(double));
    cudaMalloc(&d_norm_g_, sizeof(double));
    cudaMalloc(&d_inner_flag_, sizeof(int));
    cudaMalloc(&d_outer_flag_, sizeof(int));
    cudaMalloc(&d_inner_tol_, sizeof(double));
    cudaMalloc(&d_inner_rtol_, sizeof(double));
    cudaMalloc(&d_outer_tol_, sizeof(double));
    cudaMalloc(&d_max_outer_, sizeof(int));
    cudaMalloc(&d_max_inner_, sizeof(int));
    cudaMalloc(&d_time_step_, sizeof(double));
    cudaMalloc(&d_solver_rho_, sizeof(double));
    cudaMalloc(&d_omega_, sizeof(double));
    cudaMalloc(&d_hess_eps_, sizeof(double));
    cudaMalloc(&d_convergence_check_interval_, sizeof(int));

    cudaMalloc(&d_vbd_solver_, sizeof(SyncedVBDSolver));

    cudaMalloc(&d_x12_prev, n_coef_ * sizeof(double));
    cudaMalloc(&d_y12_prev, n_coef_ * sizeof(double));
    cudaMalloc(&d_z12_prev, n_coef_ * sizeof(double));

    // Get constraint pointer if available
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

    // Create cuBLAS handle for norm computations
    cublasCreate(&cublas_handle_);
    cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE);
    cudaMalloc(&d_norm_temp_, sizeof(double));

    // Dedicated stream for CUDA graph capture (do not use default stream).
    HANDLE_ERROR(cudaStreamCreateWithFlags(&graph_capture_stream_,
                                           cudaStreamNonBlocking));
  }

  ~SyncedVBDSolver() {
    DestroyCudaGraphs();
    if (graph_capture_stream_) {
      HANDLE_ERROR(cudaStreamDestroy(graph_capture_stream_));
      graph_capture_stream_ = nullptr;
    }

    cudaFree(d_v_guess_);
    cudaFree(d_v_prev_);
    cudaFree(d_lambda_guess_);
    cudaFree(d_g_);
    cudaFree(d_norm_g_);
    cudaFree(d_inner_flag_);
    cudaFree(d_outer_flag_);
    cudaFree(d_inner_tol_);
    cudaFree(d_inner_rtol_);
    cudaFree(d_outer_tol_);
    cudaFree(d_max_outer_);
    cudaFree(d_max_inner_);
    cudaFree(d_time_step_);
    cudaFree(d_solver_rho_);
    cudaFree(d_omega_);
    cudaFree(d_hess_eps_);
    cudaFree(d_convergence_check_interval_);

    cudaFree(d_vbd_solver_);

    cudaFree(d_x12_prev);
    cudaFree(d_y12_prev);
    cudaFree(d_z12_prev);

    // Free coloring data
    if (d_colors_)
      cudaFree(d_colors_);
    if (d_color_offsets_)
      cudaFree(d_color_offsets_);
    if (d_color_nodes_)
      cudaFree(d_color_nodes_);
    if (d_incidence_offsets_)
      cudaFree(d_incidence_offsets_);
    if (d_incidence_data_)
      cudaFree(d_incidence_data_);
    if (d_mass_diag_blocks_)
      cudaFree(d_mass_diag_blocks_);

    if (h_color_offsets_cache_)
      delete[] h_color_offsets_cache_;
    if (d_fixed_map_)
      cudaFree(d_fixed_map_);

    if (cublas_handle_)
      cublasDestroy(cublas_handle_);
    if (d_norm_temp_)
      cudaFree(d_norm_temp_);

    if (h_color_group_offsets_cache_)
      delete[] h_color_group_offsets_cache_;
    if (h_color_group_colors_cache_)
      delete[] h_color_group_colors_cache_;
  }

  void SetParameters(void *params) override {
    SyncedVBDParams *p = static_cast<SyncedVBDParams *>(params);

    h_inner_tol_           = p->inner_tol;
    h_inner_rtol_          = p->inner_rtol;
    h_outer_tol_           = p->outer_tol;
    h_max_outer_           = p->max_outer;
    h_max_inner_           = p->max_inner;
    h_conv_check_interval_ = p->convergence_check_interval;
    h_color_group_size_ = (p->color_group_size > 0 ? p->color_group_size : 1);

    cudaMemcpy(d_inner_tol_, &p->inner_tol, sizeof(double),
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
    cudaMemcpy(d_omega_, &p->omega, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hess_eps_, &p->hess_eps, sizeof(double),
               cudaMemcpyHostToDevice);
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
    cudaMemset(d_lambda_guess_, 0, n_constraints_ * sizeof(double));
    cudaMemset(d_g_, 0, n_coef_ * 3 * sizeof(double));

    HANDLE_ERROR(cudaMemcpy(d_vbd_solver_, this, sizeof(SyncedVBDSolver),
                            cudaMemcpyHostToDevice));
  }

  // Initialize coloring data structure for VBD parallel updates
  // Must be called before first Solve()
  void InitializeColoring();

  // Initialize diagonal mass blocks for VBD
  void InitializeMassDiagBlocks();

  // Initialize fixed node map (node -> constraint index, or -1)
  void InitializeFixedMap();

#if defined(__CUDACC__)
  // Device accessors
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

  __device__ double solver_inner_tol() const {
    return *d_inner_tol_;
  }
  __device__ double solver_inner_rtol() const {
    return *d_inner_rtol_;
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
  __device__ double solver_omega() const {
    return *d_omega_;
  }
  __device__ double solver_hess_eps() const {
    return *d_hess_eps_;
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

  // Coloring accessors
  __device__ int n_colors() const {
    return n_colors_;
  }
  __device__ int *colors() {
    return d_colors_;
  }
  __device__ int *color_offsets() {
    return d_color_offsets_;
  }
  __device__ int *color_nodes() {
    return d_color_nodes_;
  }
  __device__ int *incidence_offsets() {
    return d_incidence_offsets_;
  }
  __device__ int2 *incidence_data() {
    return d_incidence_data_;
  }
  __device__ double *mass_diag_blocks() {
    return d_mass_diag_blocks_;
  }
  __device__ int *fixed_map() {
    return d_fixed_map_;
  }
#endif

  __host__ __device__ int get_n_coef() const {
    return n_coef_;
  }
  __host__ __device__ int get_n_beam() const {
    return n_beam_;
  }

  // Host accessor for device velocity guess pointer
  double *GetVelocityGuessDevicePtr() const {
    return d_v_guess_;
  }

  void OneStepVBD();

  void Solve() override {
    OneStepVBD();
  }

  double compute_l2_norm_cublas(double *d_vec, int n_dofs);

 private:
  ElementBase *h_data_;
  ElementType type_;
  ElementBase *d_data_;
  SyncedVBDSolver *d_vbd_solver_;
  int n_total_qp_, n_shape_;
  int n_coef_, n_beam_, n_constraints_;

  double *d_x12_prev, *d_y12_prev, *d_z12_prev;

  double *d_v_guess_, *d_v_prev_;
  double *d_lambda_guess_, *d_g_;
  double *d_norm_g_;
  int *d_inner_flag_, *d_outer_flag_;
  double *d_inner_tol_, *d_inner_rtol_, *d_outer_tol_, *d_time_step_,
      *d_solver_rho_;
  double *d_omega_, *d_hess_eps_;
  int *d_convergence_check_interval_;
  double h_inner_tol_, h_outer_tol_, h_inner_rtol_;
  int h_max_outer_, h_max_inner_, h_conv_check_interval_;
  int *d_max_inner_, *d_max_outer_;

  double *d_constraint_ptr_;

  // Coloring data structures
  bool coloring_initialized_;
  int n_colors_;
  int *d_colors_;             // Color assignment per node (n_coef_)
  int *d_color_offsets_;      // Offsets into color_nodes for each color
                              // (n_colors_+1)
  int *d_color_nodes_;        // Flat array of node indices grouped by color
  int *d_incidence_offsets_;  // Offset into incidence_data per node (n_coef_+1)
  int2 *d_incidence_data_;    // (element_idx, local_node_idx) pairs
  int *h_color_offsets_cache_;  // Host cache of d_color_offsets_ (n_colors_+1)
  int h_color_offsets_cache_size_;
  int h_color_group_size_;  // Desired number of colors per group
  int n_color_groups_;      // Number of groups in schedule
  int *
      h_color_group_offsets_cache_;  // Offsets into h_color_group_colors_cache_
  int h_color_group_offsets_cache_size_;
  int *h_color_group_colors_cache_;  // Flat list of color indices (size
                                     // n_colors_)
  int h_color_group_colors_cache_size_;
  bool fixed_map_initialized_;
  int *d_fixed_map_;  // (n_coef_) node->k in fixed_nodes, or -1 if not fixed

  // Diagonal mass blocks for VBD
  double *d_mass_diag_blocks_;  // (n_coef_ × 9) - 3x3 blocks per node

  // cuBLAS handle
  cublasHandle_t cublas_handle_;
  double *d_norm_temp_;

  // CUDA Graphs (host-only)
  void DestroyCudaGraphs();
  // Returns true if the graph was re-captured, which also executes the captured
  // work once on the capture stream.
  bool EnsureInnerSweepGraph(int threads, int blocks_p);
  // Returns true if the graph was re-captured, which also executes the captured
  // work once on the capture stream.
  bool EnsurePostOuterGraph(int threads, int blocks_coef);

  int graph_threads_;
  int graph_blocks_coef_;
  int graph_blocks_p_;
  int graph_n_colors_;
  int graph_n_constraints_;
  int graph_color_group_size_;
  int graph_n_color_groups_;
  cudaStream_t graph_capture_stream_;
  cudaGraphExec_t inner_sweep_graph_exec_;
  cudaGraphExec_t post_outer_graph_exec_;
};
