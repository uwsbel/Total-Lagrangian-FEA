/*==============================================================
 *==============================================================
 * Project: TL-FEA
 * File:    SyncedExplicit.cu
 * Brief:   Implements the SyncedExplicitSolver class with GPU kernels
 *          for symplectic Euler time integration.
 *==============================================================
 *==============================================================*/

#include "SyncedExplicit.cuh"

#include <cuda_runtime.h>

#include <iostream>

#include "../elements/FEAT10DataFunc.cuh"

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Symplectic Euler velocity update kernel.
 * v_{n+1} = v_n + dt * M^{-1} * (f_ext - f_int)
 *
 * @param d_data     FEAT10 data (provides f_int/f_ext)
 * @param d_vel      Velocity array [n_nodes * 3] (interleaved: vx0,vy0,vz0,vx1,...)
 * @param d_inv_mass Cached inverse lumped mass [n_nodes]
 * @param dt         Time step
 */
__global__ void explicit_velocity_update(GPU_FEAT10_Data* d_data,
                                         double* __restrict__ d_vel,
                                         const double* __restrict__ d_inv_mass,
                                         double dt) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= d_data->n_coef) return;

  double m_inv = d_inv_mass[node];

  // v_{n+1} = v_n + dt * M^{-1} * (f_ext - f_int)
  // Note: f_int is stored as the internal force (not negated), so we subtract it
  for (int c = 0; c < 3; c++) {
    int idx = node * 3 + c;
    double a = (d_data->f_ext()(idx) - d_data->f_int()(idx)) * m_inv;
    d_vel[idx] += dt * a;
  }
}

__global__ void explicit_compute_inv_mass(double* __restrict__ d_inv_mass,
                                          const double* __restrict__ d_mass_lumped,
                                          int n_nodes) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n_nodes) return;
  d_inv_mass[node] = 1.0 / d_mass_lumped[node];
}

/**
 * Symplectic Euler position update kernel.
 * x_{n+1} = x_n + dt * v_{n+1}
 *
 * @param d_data  FEAT10 data (positions stored in x12/y12/z12)
 * @param d_vel   Velocity array [n_nodes * 3] (interleaved)
 * @param dt      Time step
 */
__global__ void explicit_position_update(GPU_FEAT10_Data* d_data,
                                         const double* __restrict__ d_vel,
                                         double dt) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= d_data->n_coef) return;

  d_data->x12()(node) += dt * d_vel[node * 3 + 0];
  d_data->y12()(node) += dt * d_vel[node * 3 + 1];
  d_data->z12()(node) += dt * d_vel[node * 3 + 2];
}

/**
 * Apply fixed node boundary conditions by zeroing velocity.
 *
 * @param d_vel         Velocity array [n_nodes * 3]
 * @param d_fixed_nodes Array of fixed node indices [n_fixed]
 * @param n_fixed       Number of fixed nodes
 */
__global__ void explicit_apply_fixed_node_bc(double* __restrict__ d_vel,
                                    const int* __restrict__ d_fixed_nodes,
                                    int n_fixed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_fixed) return;

  int node = d_fixed_nodes[i];
  d_vel[node * 3 + 0] = 0.0;
  d_vel[node * 3 + 1] = 0.0;
  d_vel[node * 3 + 2] = 0.0;
}

__global__ void explicit_compute_p(GPU_FEAT10_Data* d_data,
                                   const double* __restrict__ d_vel,
                                   double dt) {
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx = tid / Quadrature::N_QP_T10_5;
  int qp_idx   = tid % Quadrature::N_QP_T10_5;

  if (elem_idx >= d_data->gpu_n_elem() || qp_idx >= Quadrature::N_QP_T10_5)
    return;

  compute_p(elem_idx, qp_idx, d_data, d_vel, dt);
}

__global__ void explicit_clear_internal_force(GPU_FEAT10_Data* d_data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_data->n_coef * 3) {
    clear_internal_force(d_data);
  }
}

__global__ void explicit_compute_internal_force(GPU_FEAT10_Data* d_data) {
  int tid        = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx   = tid / Quadrature::N_NODE_T10_10;
  int node_local = tid % Quadrature::N_NODE_T10_10;

  if (elem_idx >= d_data->gpu_n_elem() ||
      node_local >= Quadrature::N_NODE_T10_10)
    return;

  compute_internal_force(elem_idx, node_local, d_data);
}

// ============================================================================
// SyncedExplicitSolver Implementation
// ============================================================================

SyncedExplicitSolver::SyncedExplicitSolver(GPU_FEAT10_Data* element)
    : element_(element),
      n_nodes_(element->get_n_coef()),
      d_vel_(nullptr),
      d_inv_mass_(nullptr),
      d_fixed_nodes_(nullptr),
      n_fixed_nodes_(0),
      current_time_(0.0),
      current_step_(0),
      is_initialized_(false),
      inv_mass_ready_(false) {
  // Set default parameters
  params_.dt = 1e-6;

  AllocateMemory();
}

SyncedExplicitSolver::~SyncedExplicitSolver() { FreeMemory(); }

void SyncedExplicitSolver::AllocateMemory() {
  // Allocate velocity array (3 components per node, interleaved)
  HANDLE_ERROR(cudaMalloc(&d_vel_, n_nodes_ * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemset(d_vel_, 0, n_nodes_ * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_inv_mass_, n_nodes_ * sizeof(double)));

  is_initialized_ = true;
}

void SyncedExplicitSolver::FreeMemory() {
  if (d_vel_) {
    HANDLE_ERROR(cudaFree(d_vel_));
    d_vel_ = nullptr;
  }
  if (d_inv_mass_) {
    HANDLE_ERROR(cudaFree(d_inv_mass_));
    d_inv_mass_ = nullptr;
  }
  if (d_fixed_nodes_) {
    HANDLE_ERROR(cudaFree(d_fixed_nodes_));
    d_fixed_nodes_ = nullptr;
  }
  is_initialized_ = false;
}

void SyncedExplicitSolver::SetParameters(void* params) {
  SyncedExplicitParams* p = static_cast<SyncedExplicitParams*>(params);
  params_ = *p;
}

void SyncedExplicitSolver::SetFixedNodes(const std::vector<int>& fixed_nodes) {
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
  }
}

void SyncedExplicitSolver::Reset() {
  current_time_ = 0.0;
  current_step_ = 0;
  HANDLE_ERROR(cudaMemset(d_vel_, 0, n_nodes_ * 3 * sizeof(double)));
}

void SyncedExplicitSolver::RetrieveVelocityToCPU(Eigen::VectorXd& vel_x,
                                                  Eigen::VectorXd& vel_y,
                                                  Eigen::VectorXd& vel_z) {
  // Allocate temporary buffer for interleaved velocity
  std::vector<double> vel_host(n_nodes_ * 3);
  HANDLE_ERROR(cudaMemcpy(vel_host.data(), d_vel_, n_nodes_ * 3 * sizeof(double),
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

void SyncedExplicitSolver::Solve() {
  // Step 0: Verify lumped mass is computed
  if (!element_->IsLumpedMassComputed()) {
    std::cerr << "SyncedExplicitSolver::Solve() - Lumped mass not computed. "
              << "Call element->CalcLumpedMassHRZ() first." << std::endl;
    return;
  }

  const int block_size = 256;
  int grid_size = (n_nodes_ + block_size - 1) / block_size;
  const int n_elem = element_->n_elem;
  const int qp_threads = n_elem * Quadrature::N_QP_T10_5;
  const int node_threads = n_elem * Quadrature::N_NODE_T10_10;
  const int qp_grid = (qp_threads + block_size - 1) / block_size;
  const int clear_grid = (n_nodes_ * 3 + block_size - 1) / block_size;
  const int node_grid = (node_threads + block_size - 1) / block_size;

  if (!inv_mass_ready_) {
    explicit_compute_inv_mass<<<grid_size, block_size>>>(
        d_inv_mass_, element_->GetLumpedMassDevicePtr(), n_nodes_);
    inv_mass_ready_ = true;
  }

  // Step 1: Compute internal forces
  explicit_compute_p<<<qp_grid, block_size>>>(element_->d_data, d_vel_,
                                              params_.dt);
  explicit_clear_internal_force<<<clear_grid, block_size>>>(element_->d_data);
  explicit_compute_internal_force<<<node_grid, block_size>>>(element_->d_data);

  // Step 2: Update velocity
  explicit_velocity_update<<<grid_size, block_size>>>(
      element_->d_data,
      d_vel_,
      d_inv_mass_,
      params_.dt);

  // Step 3: Apply boundary conditions
  if (n_fixed_nodes_ > 0) {
    int bc_grid = (n_fixed_nodes_ + block_size - 1) / block_size;
    explicit_apply_fixed_node_bc<<<bc_grid, block_size>>>(d_vel_, d_fixed_nodes_,
                                                  n_fixed_nodes_);
  }

  // Step 4: Update positions
  explicit_position_update<<<grid_size, block_size>>>(
      element_->d_data, d_vel_, params_.dt);

  // Update time tracking
  current_time_ += params_.dt;
  current_step_++;
}
