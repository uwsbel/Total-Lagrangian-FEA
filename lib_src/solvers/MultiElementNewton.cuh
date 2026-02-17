/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    MultiElementNewton.cuh
 * Brief:   Multi-element Newton solver for co-simulation problems.
 *          Uses block-diagonal approach where each element block has its own
 *          SyncedNewtonSolver instance. Cross-block coupling happens via
 *          external forces (collision) applied through unified state buffers.
 *==============================================================
 *==============================================================*/

#pragma once

#include <memory>
#include <vector>

#include "FEMultiElementProblem.h"
#include "SolverBase.h"
#include "SyncedNewton.cuh"

// Parameters for multi-element Newton solver.
struct MultiElementNewtonParams {
  // Newton iteration parameters (passed to each block solver).
  double inner_atol  = 1e-8;
  double inner_rtol  = 1e-6;
  double outer_tol   = 1e-6;
  double rho         = 1000.0;  // Density for mass matrix scaling
  int max_outer      = 10;
  int max_inner      = 100;
  double time_step   = 1e-4;
};

// Multi-element Newton solver using block-diagonal Jacobi approach.
//
// Each element block (ANCF3243, FEAT10, etc.) has its own SyncedNewtonSolver
// instance that computes local gradients and Hessians. Cross-body coupling
// happens through:
// 1. External forces (collision, gravity) applied to unified state buffer.
// 2. Position/velocity synchronization between unified buffer and block solvers.
//
// Algorithm per time step:
// 1. Sync positions from unified buffer to all block solvers.
// 2. Add collision/external forces to block external force buffers.
// 3. Each block solver performs Newton iteration independently.
// 4. Sync updated positions from blocks back to unified buffer.
//
// This is a Gauss-Seidel-like approach within each time step, suitable for
// contact-dominated coupling where explicit external forces provide coupling.
class MultiElementNewtonSolver : public SolverBase {
 public:
  // Construct solver for a multi-element problem.
  // The problem must be finalized before passing to this constructor.
  explicit MultiElementNewtonSolver(FEMultiElementProblem* problem);

  ~MultiElementNewtonSolver() override;

  // Disallow copy/move for CUDA resource safety.
  MultiElementNewtonSolver(const MultiElementNewtonSolver&)            = delete;
  MultiElementNewtonSolver& operator=(const MultiElementNewtonSolver&) = delete;
  MultiElementNewtonSolver(MultiElementNewtonSolver&&)                 = delete;
  MultiElementNewtonSolver& operator=(MultiElementNewtonSolver&&)      = delete;

  // Set solver parameters. Params must be a MultiElementNewtonParams*.
  void SetParameters(void* params) override;

  // Setup all block solvers (call after SetParameters, before Solve).
  void Setup();

  // Perform one time step using block-diagonal Newton iteration.
  void Solve() override;

  // ---------------------------------------------------------------------------
  // State access for external force application
  // ---------------------------------------------------------------------------

  // Get the unified external force device pointer for collision system.
  // Layout: [fx0, fy0, fz0, fx1, fy1, fz1, ...] with total_coef*3 entries.
  // Write collision forces here before calling Solve().
  double* GetExternalForceDevicePtr();

  // Get total DOFs across all blocks.
  int GetTotalDofs() const;

  // ---------------------------------------------------------------------------
  // Block-level accessors (for debugging/visualization)
  // ---------------------------------------------------------------------------

  // Get the underlying single-element solver for a block.
  SyncedNewtonSolver* GetBlockSolver(int block_idx);

  // Get number of blocks.
  int GetNumBlocks() const;

 private:
  FEMultiElementProblem* problem_;  // Not owned
  std::vector<std::unique_ptr<SyncedNewtonSolver>> block_solvers_;
  MultiElementNewtonParams params_;
  bool setup_done_ = false;

  // Distribute external forces from unified buffer to block buffers.
  void DistributeExternalForces();

  // Collect positions from block solvers back to unified buffer.
  void CollectPositions();
};
