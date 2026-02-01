/*==============================================================
 *==============================================================
 * Project: TL-FEA
 * File:    SyncedExplicitOpt.cuh
 * Brief:   Declares the SyncedExplicitOptSolver class for GPU-accelerated
 *          symplectic Euler time integration with FEAT10Opt elements.
 *          Uses double-precision velocity with float-precision forces.
 *==============================================================
 *==============================================================*/

#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <vector>

#include "../../lib_utils/cuda_utils.h"
#include "../elements/FEAT10DataOpt.cuh"

/**
 * Parameters for SyncedExplicitOptSolver.
 */
struct SyncedExplicitOptParams {
  double dt;  // Time step size
};

/**
 * Fast explicit dynamics solver for FEAT10Opt elements.
 *
 * Key features:
 * - Double-precision positions and velocities
 * - Float-precision forces (from FEAT10Opt kernel)
 * - Fused internal force computation with FEAT10Opt elements
 * - Fixed node boundary conditions
 * - Timing instrumentation
 *
 * Time integration scheme (Symplectic Euler):
 *   v_{n+1} = v_n + dt * M^{-1} * (f_ext - f_int)
 *   x_{n+1} = x_n + dt * v_{n+1}
 */
class SyncedExplicitOptSolver {
 public:
  /**
   * Constructor.
   * @param element Pointer to FEAT10Opt element data (must be initialized/setup)
   */
  explicit SyncedExplicitOptSolver(GPU_FEAT10Opt_Data* element);

  /**
   * Destructor. Frees all allocated device memory.
   */
  ~SyncedExplicitOptSolver();

  // Prevent copying
  SyncedExplicitOptSolver(const SyncedExplicitOptSolver&) = delete;
  SyncedExplicitOptSolver& operator=(const SyncedExplicitOptSolver&) = delete;

  // ============================================================
  // Main Interface
  // ============================================================

  /**
   * Advance simulation by one time step.
   * This performs:
   * 1. Clear internal force
   * 2. Compute internal force (fused kernel)
   * 3. Update velocity: v += dt * M^{-1} * (f_ext - f_int)
   * 4. Apply fixed node BC
   * 5. Update position: x += dt * v
   */
  void Solve();

  /**
   * Set solver parameters.
   * @param params Pointer to SyncedExplicitOptParams struct
   */
  void SetParameters(const SyncedExplicitOptParams& params);

  /**
   * Set time step directly.
   * @param dt Time step size
   */
  void SetTimeStep(double dt) { params_.dt = dt; }

  // ============================================================
  // Boundary Conditions
  // ============================================================

  /**
   * Set fixed (Dirichlet) nodes.
   * Velocity at these nodes will be zeroed after each update.
   * @param fixed_nodes Vector of node indices to fix
   */
  void SetFixedNodes(const std::vector<int>& fixed_nodes);

  /**
   * Set external force on all nodes.
   * @param f_ext External force vector [3 * n_nodes] (interleaved x,y,z)
   */
  void SetExternalForce(const Eigen::VectorXf& f_ext);

  /**
   * Clear external force to zero.
   */
  void ClearExternalForce();

  // ============================================================
  // State Access
  // ============================================================

  /**
   * Get device pointer to velocity array.
   * Layout: [vx0, vy0, vz0, vx1, vy1, vz1, ...] (double precision)
   */
  double* GetVelocityDevicePtr() { return d_vel_; }
  const double* GetVelocityDevicePtr() const { return d_vel_; }

  /**
   * Get device pointer to external force array.
   * Layout: [fx0, fy0, fz0, fx1, fy1, fz1, ...] (float precision)
   */
  float* GetExternalForceDevicePtr() { return d_f_ext_; }
  const float* GetExternalForceDevicePtr() const { return d_f_ext_; }

  /**
   * Get current simulation time.
   */
  double GetCurrentTime() const { return current_time_; }

  /**
   * Get current step number.
   */
  int GetCurrentStep() const { return current_step_; }

  /**
   * Get number of nodes.
   */
  int GetNumNodes() const { return n_nodes_; }

  // ============================================================
  // Timing
  // ============================================================

  /**
   * Get time (in ms) for the last Solve() call.
   */
  float GetLastStepTimeMs() const { return last_step_time_ms_; }

  // ============================================================
  // Data Transfer
  // ============================================================

  /**
   * Retrieve velocity from GPU to CPU.
   * @param vel_x Output x-velocities [n_nodes]
   * @param vel_y Output y-velocities [n_nodes]
   * @param vel_z Output z-velocities [n_nodes]
   */
  void RetrieveVelocityToCPU(Eigen::VectorXd& vel_x, Eigen::VectorXd& vel_y,
                              Eigen::VectorXd& vel_z);

  /**
   * Set velocity from CPU.
   * @param vel_x Input x-velocities [n_nodes]
   * @param vel_y Input y-velocities [n_nodes]
   * @param vel_z Input z-velocities [n_nodes]
   */
  void SetVelocityFromCPU(const Eigen::VectorXd& vel_x,
                          const Eigen::VectorXd& vel_y,
                          const Eigen::VectorXd& vel_z);

  /**
   * Reset simulation to initial state.
   * Zeroes velocity and resets time/step counters.
   */
  void Reset();

 private:
  // Element data reference
  GPU_FEAT10Opt_Data* element_;
  int n_nodes_;

  // Device memory
  double* d_vel_;         // Velocity [3 * n_nodes] (double, interleaved)
  float* d_f_ext_;        // External force [3 * n_nodes] (float, interleaved)
  int* d_fixed_nodes_;    // Fixed node indices [n_fixed]
  int n_fixed_nodes_;

  // Parameters
  SyncedExplicitOptParams params_;
  double current_time_;
  int current_step_;

  // Internal state
  bool is_initialized_;

  // Timing
  cudaEvent_t timing_start_;
  cudaEvent_t timing_stop_;
  float last_step_time_ms_;

  // Internal methods
  void AllocateMemory();
  void FreeMemory();
};
