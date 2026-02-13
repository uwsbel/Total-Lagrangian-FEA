/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli
 * Email:   arivoli@wisc.edu
 * File:    SyncedExplicitOpt.cu
 * Brief:   Implements SyncedExplicitOptSolver with GPU kernels for
 *          symplectic Euler explicit time integration (FEAT10Opt).
 *==============================================================
 *==============================================================*/

#include "SyncedExplicitOpt.cuh"

#include <iostream>
#include <vector>

// CUDA Kernels

// Velocity update: v += dt * M^{-1} * (f_ext - f_int)
__global__ void syncedExplicitOpt_velocityUpdate(
    double* __restrict__ d_vel,
    const float* __restrict__ d_f_int,
    const float* __restrict__ d_f_ext,
    const float* __restrict__ d_inv_mass,
    double dt,
    int n_nodes) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n_nodes) return;

  int base = node * 3;

  // Pre-compute dt * M^{-1} (single float→double conversion)
  double dt_minv = dt * d_inv_mass[node];

  // v_{n+1} = v_n + dt * M^{-1} * (f_ext - f_int)
  // Unrolled for better vectorization; implicit float→double promotion
  d_vel[base + 0] += dt_minv * (d_f_ext[base + 0] - d_f_int[base + 0]);
  d_vel[base + 1] += dt_minv * (d_f_ext[base + 1] - d_f_int[base + 1]);
  d_vel[base + 2] += dt_minv * (d_f_ext[base + 2] - d_f_int[base + 2]);
}

// Position update: x += dt * v
__global__ void syncedExplicitOpt_positionUpdate(
    double* __restrict__ d_pos,
    const double* __restrict__ d_vel,
    double dt,
    int n_nodes) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n_nodes) return;

  d_pos[node * 3 + 0] += dt * d_vel[node * 3 + 0];
  d_pos[node * 3 + 1] += dt * d_vel[node * 3 + 1];
  d_pos[node * 3 + 2] += dt * d_vel[node * 3 + 2];
}

// Zero velocity at fixed nodes.
__global__ void syncedExplicitOpt_applyFixedBC(
    double* __restrict__ d_vel,
    const int* __restrict__ d_fixed_nodes,
    int n_fixed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_fixed) return;

  int node = d_fixed_nodes[i];
  d_vel[node * 3 + 0] = 0.0;
  d_vel[node * 3 + 1] = 0.0;
  d_vel[node * 3 + 2] = 0.0;
}

// SyncedExplicitOptSolver Implementation

SyncedExplicitOptSolver::SyncedExplicitOptSolver(GPU_FEAT10Opt_Data* element)
    : element_(element),
      n_nodes_(element->get_n_nodes()),
      d_vel_(nullptr),
      d_fixed_nodes_(nullptr),
      n_fixed_nodes_(0),
      current_time_(0.0),
      current_step_(0),
      is_initialized_(false),
      block_size_(256),
      node_grid_(0),
      bc_grid_(0),
      use_graphs_(false),
      cuda_graph_(nullptr),
      graph_exec_(nullptr),
      graph_captured_(false),
      stream_(nullptr) {
  // Set default parameters
  params_.dt = 1e-6;
  params_.print_interval = 500;  // Default: print every 500 steps (graph size matches)

  // Compute grid dimensions
  node_grid_ = (n_nodes_ + block_size_ - 1) / block_size_;

  AllocateMemory();
}

SyncedExplicitOptSolver::~SyncedExplicitOptSolver() {
  FreeMemory();
}

void SyncedExplicitOptSolver::AllocateMemory() {
  // Allocate velocity array (double precision, interleaved)
  HANDLE_ERROR(cudaMalloc(&d_vel_, n_nodes_ * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemset(d_vel_, 0, n_nodes_ * 3 * sizeof(double)));

  // Create stream for graph operations
  HANDLE_ERROR(cudaStreamCreate(&stream_));

  is_initialized_ = true;
}

void SyncedExplicitOptSolver::FreeMemory() {
  // Destroy graph resources
  if (graph_exec_) {
    HANDLE_ERROR(cudaGraphExecDestroy(graph_exec_));
    graph_exec_ = nullptr;
  }
  if (cuda_graph_) {
    HANDLE_ERROR(cudaGraphDestroy(cuda_graph_));
    cuda_graph_ = nullptr;
  }
  if (stream_) {
    HANDLE_ERROR(cudaStreamDestroy(stream_));
    stream_ = nullptr;
  }

  if (d_vel_) {
    HANDLE_ERROR(cudaFree(d_vel_));
    d_vel_ = nullptr;
  }
  if (d_fixed_nodes_) {
    HANDLE_ERROR(cudaFree(d_fixed_nodes_));
    d_fixed_nodes_ = nullptr;
  }

  is_initialized_ = false;
}

void SyncedExplicitOptSolver::SetParameters(const SyncedExplicitOptParams& params) {
  params_ = params;
}

void SyncedExplicitOptSolver::SetFixedNodes(const std::vector<int>& fixed_nodes) {
  // Free existing fixed nodes array
  if (d_fixed_nodes_) {
    HANDLE_ERROR(cudaFree(d_fixed_nodes_));
    d_fixed_nodes_ = nullptr;
  }

  n_fixed_nodes_ = static_cast<int>(fixed_nodes.size());

  if (n_fixed_nodes_ > 0) {
    HANDLE_ERROR(cudaMalloc(&d_fixed_nodes_, n_fixed_nodes_ * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_fixed_nodes_, fixed_nodes.data(),
                            n_fixed_nodes_ * sizeof(int),
                            cudaMemcpyHostToDevice));
    // Compute BC grid dimensions
    bc_grid_ = (n_fixed_nodes_ + block_size_ - 1) / block_size_;
  } else {
    bc_grid_ = 0;
  }
}

void SyncedExplicitOptSolver::SetExternalForce(const Eigen::VectorXf& f_ext) {
  element_->SetExternalForce(f_ext);
}

void SyncedExplicitOptSolver::ClearExternalForce() {
  element_->ClearExternalForce();
}

void SyncedExplicitOptSolver::Reset() {
  current_time_ = 0.0;
  current_step_ = 0;
  HANDLE_ERROR(cudaMemset(d_vel_, 0, n_nodes_ * 3 * sizeof(double)));
}

void SyncedExplicitOptSolver::RetrieveVelocityToCPU(Eigen::VectorXd& vel_x,
                                                Eigen::VectorXd& vel_y,
                                                Eigen::VectorXd& vel_z) {
  // Copy interleaved velocity to host
  std::vector<double> vel_host(n_nodes_ * 3);
  HANDLE_ERROR(cudaMemcpy(vel_host.data(), d_vel_,
                          n_nodes_ * 3 * sizeof(double),
                          cudaMemcpyDeviceToHost));

  vel_x.resize(n_nodes_);
  vel_y.resize(n_nodes_);
  vel_z.resize(n_nodes_);

  for (int i = 0; i < n_nodes_; i++) {
    vel_x(i) = vel_host[i * 3 + 0];
    vel_y(i) = vel_host[i * 3 + 1];
    vel_z(i) = vel_host[i * 3 + 2];
  }
}

void SyncedExplicitOptSolver::SetVelocityFromCPU(const Eigen::VectorXd& vel_x,
                                             const Eigen::VectorXd& vel_y,
                                             const Eigen::VectorXd& vel_z) {
  if (vel_x.size() != n_nodes_ || vel_y.size() != n_nodes_ ||
      vel_z.size() != n_nodes_) {
    std::cerr << "SyncedExplicitOptSolver: Velocity size mismatch." << std::endl;
    return;
  }

  // Convert to interleaved format
  std::vector<double> vel_host(n_nodes_ * 3);
  for (int i = 0; i < n_nodes_; i++) {
    vel_host[i * 3 + 0] = vel_x(i);
    vel_host[i * 3 + 1] = vel_y(i);
    vel_host[i * 3 + 2] = vel_z(i);
  }

  HANDLE_ERROR(cudaMemcpy(d_vel_, vel_host.data(),
                          n_nodes_ * 3 * sizeof(double),
                          cudaMemcpyHostToDevice));
}

void SyncedExplicitOptSolver::Solve() {
  // Determine which stream to use (for graph capture compatibility)
  cudaStream_t exec_stream = use_graphs_ ? stream_ : 0;
  
  // Stage 1: Clear internal force
  element_->ClearInternalForce(exec_stream);

  // Stage 2: Compute internal force using fused kernel
  element_->ComputeInternalForce(false, exec_stream);  // No F output needed

  // Stage 3: Update velocity
  syncedExplicitOpt_velocityUpdate<<<node_grid_, block_size_, 0, exec_stream>>>(
      d_vel_,
      element_->GetInternalForceDevicePtr(),
      element_->GetExternalForceDevicePtr(),
      element_->GetInvMassDevicePtr(),
      params_.dt,
      n_nodes_);

  // Stage 4: Apply fixed node boundary conditions
  if (n_fixed_nodes_ > 0) {
    syncedExplicitOpt_applyFixedBC<<<bc_grid_, block_size_, 0, exec_stream>>>(
        d_vel_, d_fixed_nodes_, n_fixed_nodes_);
  }

  // Stage 5: Update positions
  syncedExplicitOpt_positionUpdate<<<node_grid_, block_size_, 0, exec_stream>>>(
      element_->GetPositionDevicePtr(),
      d_vel_,
      params_.dt,
      n_nodes_);

}

// CUDA Graphs Implementation

void SyncedExplicitOptSolver::EnableGraphs(bool enable) {
  use_graphs_ = enable;
  graph_captured_ = false;  // Force recapture if toggled
}

void SyncedExplicitOptSolver::CaptureGraph() {
  if (!use_graphs_) {
    std::cerr << "Warning: Graphs not enabled, skipping capture" << std::endl;
    return;
  }
  
  // Clean up old graph if recapturing
  if (graph_exec_) {
    HANDLE_ERROR(cudaGraphExecDestroy(graph_exec_));
    graph_exec_ = nullptr;
  }
  if (cuda_graph_) {
    HANDLE_ERROR(cudaGraphDestroy(cuda_graph_));
    cuda_graph_ = nullptr;
  }
  
  // Begin graph capture on stream (use Relaxed mode to allow cross-stream dependencies)
  HANDLE_ERROR(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed));
  
  // Capture print_interval consecutive Solve() calls
  for (int i = 0; i < params_.print_interval; ++i) {
    Solve();  // This captures all kernels in Solve()
  }
  
  // End capture
  HANDLE_ERROR(cudaStreamEndCapture(stream_, &cuda_graph_));
  
  // Instantiate the graph for execution
  HANDLE_ERROR(cudaGraphInstantiate(&graph_exec_, cuda_graph_, nullptr, nullptr, 0));
  
  graph_captured_ = true;
  
  std::cout << "CUDA Graph captured: " << params_.print_interval 
            << " steps per graph launch" << std::endl;
}

void SyncedExplicitOptSolver::ExecuteGraph() {
  if (!graph_captured_) {
    std::cerr << "Error: Graph not captured yet, call CaptureGraph() first" << std::endl;
    return;
  }
  
  // Launch the graph (executes print_interval Solve() calls in one shot)
  // Note: This is asynchronous - caller must synchronize stream before accessing results
  HANDLE_ERROR(cudaGraphLaunch(graph_exec_, stream_));
}
