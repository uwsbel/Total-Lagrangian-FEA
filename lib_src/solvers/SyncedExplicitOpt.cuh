#pragma once

/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli
 * Email:   arivoli@wisc.edu
 * File:    SyncedExplicitOpt.cuh
 * Brief:   Declares SyncedExplicitOptSolver for GPU explicit
 *          symplectic Euler time integration with FEAT10Opt.
 *==============================================================
 *==============================================================*/

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <vector>

#include "../../lib_utils/cuda_utils.h"
#include "../elements/FEAT10DataOpt.cuh"

// Solver parameters.
struct SyncedExplicitOptParams {
  double dt;  // Time step
};

// Explicit dynamics solver for FEAT10Opt elements using symplectic Euler.
// Positions/velocities are double, forces are float.
class SyncedExplicitOptSolver {
 public:
  // Element data must be initialized/setup.
  explicit SyncedExplicitOptSolver(GPU_FEAT10Opt_Data* element);

  // Frees device memory.
  ~SyncedExplicitOptSolver();

  // Prevent copying
  SyncedExplicitOptSolver(const SyncedExplicitOptSolver&) = delete;
  SyncedExplicitOptSolver& operator=(const SyncedExplicitOptSolver&) = delete;

  // Main interface
  // Advance simulation by one time step.
  void Solve();

  // Set solver parameters.
  void SetParameters(const SyncedExplicitOptParams& params);

  // Set time step.
  void SetTimeStep(double dt) { params_.dt = dt; }

  // Boundary conditions
  // Set fixed (Dirichlet) nodes.
  void SetFixedNodes(const std::vector<int>& fixed_nodes);

  // Set external force on all nodes [3 * n_nodes] (x,y,z interleaved).
  void SetExternalForce(const Eigen::VectorXf& f_ext);

  // Clear external force.
  void ClearExternalForce();

  // State access
  // Device pointer to velocity [vx0, vy0, vz0, ...] (double).
  double* GetVelocityDevicePtr() { return d_vel_; }
  const double* GetVelocityDevicePtr() const { return d_vel_; }

  // Device pointer to external force [fx0, fy0, fz0, ...] (float).
  float* GetExternalForceDevicePtr() { return element_->GetExternalForceDevicePtr(); }
  const float* GetExternalForceDevicePtr() const { return element_->GetExternalForceDevicePtr(); }

  // Current simulation time.
  double GetCurrentTime() const { return current_time_; }

  // Current step index.
  int GetCurrentStep() const { return current_step_; }

  // Number of nodes.
  int GetNumNodes() const { return n_nodes_; }

  // Timing
  // Time in ms for last Solve() call.
  float GetLastStepTimeMs() const { return last_step_time_ms_; }

  // Data transfer
  // Download velocity to CPU.
  void RetrieveVelocityToCPU(Eigen::VectorXd& vel_x, Eigen::VectorXd& vel_y,
                              Eigen::VectorXd& vel_z);

  // Upload velocity from CPU.
  void SetVelocityFromCPU(const Eigen::VectorXd& vel_x,
                          const Eigen::VectorXd& vel_y,
                          const Eigen::VectorXd& vel_z);

  // Reset simulation (zero velocity, reset time/step).
  void Reset();

 private:
  // Element data
  GPU_FEAT10Opt_Data* element_;
  int n_nodes_;

  // Device memory
  double* d_vel_;      // Velocity [3 * n_nodes] (double, interleaved)
  int* d_fixed_nodes_; // Fixed node indices [n_fixed]
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
