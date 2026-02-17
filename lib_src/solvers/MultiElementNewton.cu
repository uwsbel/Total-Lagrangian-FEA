/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    MultiElementNewton.cu
 * Brief:   Implementation of multi-element Newton solver using block-diagonal
 *          approach with per-block SyncedNewtonSolver instances.
 *==============================================================
 *==============================================================*/

#include "MultiElementNewton.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#include "lib_src/elements/ANCF3243Data.cuh"
#include "lib_src/elements/ANCF3443Data.cuh"
#include "lib_src/elements/FEAT10Data.cuh"
#include "lib_utils/cuda_utils.h"

// -----------------------------------------------------------------------------
// Construction / Destruction
// -----------------------------------------------------------------------------

MultiElementNewtonSolver::MultiElementNewtonSolver(
    FEMultiElementProblem* problem)
    : problem_(problem) {
  if (problem_ == nullptr) {
    throw std::invalid_argument(
        "MultiElementNewtonSolver: problem cannot be null");
  }
  if (!problem_->IsFinalized()) {
    throw std::invalid_argument(
        "MultiElementNewtonSolver: problem must be finalized before "
        "constructing solver");
  }

  // Create a SyncedNewtonSolver for each element block.
  const int n_blocks = problem_->GetNumBlocks();
  block_solvers_.reserve(n_blocks);

  for (int i = 0; i < n_blocks; ++i) {
    ElementBase* element = problem_->GetElementData(i);
    ElementType type     = problem_->GetElementType(i);

    // Get constraint count for this block.
    int n_constraints = 0;
    switch (type) {
      case TYPE_3243: {
        auto* data    = static_cast<GPU_ANCF3243_Data*>(element);
        n_constraints = data->get_n_constraint();
        break;
      }
      case TYPE_3443: {
        auto* data    = static_cast<GPU_ANCF3443_Data*>(element);
        n_constraints = data->get_n_constraint();
        break;
      }
      case TYPE_T10: {
        auto* data    = static_cast<GPU_FEAT10_Data*>(element);
        n_constraints = data->get_n_constraint();
        break;
      }
    }

    auto solver =
        std::make_unique<SyncedNewtonSolver>(element, n_constraints);
    block_solvers_.push_back(std::move(solver));
  }

  std::cout << "MultiElementNewtonSolver: Created " << n_blocks
            << " block solvers\n";
}

MultiElementNewtonSolver::~MultiElementNewtonSolver() {
  // block_solvers_ destructs automatically via unique_ptr.
}

// -----------------------------------------------------------------------------
// Setup and Parameters
// -----------------------------------------------------------------------------

void MultiElementNewtonSolver::SetParameters(void* params) {
  if (params == nullptr) {
    throw std::invalid_argument(
        "MultiElementNewtonSolver: params cannot be null");
  }
  params_ = *static_cast<MultiElementNewtonParams*>(params);

  // Create SyncedNewtonParams for each block solver.
  SyncedNewtonParams block_params;
  block_params.inner_atol = params_.inner_atol;
  block_params.inner_rtol = params_.inner_rtol;
  block_params.outer_tol  = params_.outer_tol;
  block_params.rho        = params_.rho;
  block_params.max_outer  = params_.max_outer;
  block_params.max_inner  = params_.max_inner;
  block_params.time_step  = params_.time_step;

  for (auto& solver : block_solvers_) {
    solver->SetParameters(&block_params);
  }
}

void MultiElementNewtonSolver::Setup() {
  if (setup_done_) {
    return;
  }

  for (auto& solver : block_solvers_) {
    solver->Setup();
    solver->SetFixedSparsityPattern(true);  // Assume fixed sparsity for perf
    solver->AnalyzeHessianSparsity();
  }

  setup_done_ = true;
  std::cout << "MultiElementNewtonSolver: Setup complete for "
            << block_solvers_.size() << " blocks\n";
}

// -----------------------------------------------------------------------------
// Solve
// -----------------------------------------------------------------------------

void MultiElementNewtonSolver::Solve() {
  if (!setup_done_) {
    throw std::runtime_error(
        "MultiElementNewtonSolver: must call Setup() before Solve()");
  }

  // 1. Sync positions from unified buffer to element blocks.
  problem_->SyncPositionsToElements();

  // 2. Distribute external forces from unified buffer to block buffers.
  DistributeExternalForces();

  // 3. Solve each block independently.
  for (auto& solver : block_solvers_) {
    solver->Solve();
  }

  // 4. Sync updated positions from blocks back to unified buffer.
  CollectPositions();

  // 5. Update collision node buffer for next step's collision detection.
  problem_->UpdateCollisionNodeBuffer();
}

// -----------------------------------------------------------------------------
// State Access
// -----------------------------------------------------------------------------

double* MultiElementNewtonSolver::GetExternalForceDevicePtr() {
  return problem_->GetStateBuffer().d_f_ext;
}

int MultiElementNewtonSolver::GetTotalDofs() const {
  return problem_->GetTotalDofs();
}

SyncedNewtonSolver* MultiElementNewtonSolver::GetBlockSolver(int block_idx) {
  if (block_idx < 0 || block_idx >= static_cast<int>(block_solvers_.size())) {
    throw std::out_of_range(
        "MultiElementNewtonSolver: block index out of range");
  }
  return block_solvers_[block_idx].get();
}

int MultiElementNewtonSolver::GetNumBlocks() const {
  return static_cast<int>(block_solvers_.size());
}

// -----------------------------------------------------------------------------
// Private Helpers
// -----------------------------------------------------------------------------

void MultiElementNewtonSolver::DistributeExternalForces() {
  const FEStateBuffer& state = problem_->GetStateBuffer();
  const int n_blocks         = problem_->GetNumBlocks();

  for (int i = 0; i < n_blocks; ++i) {
    ElementBase* element = problem_->GetElementData(i);
    ElementType type     = problem_->GetElementType(i);

    const int dof_offset = state.GetBlockDofOffset(i);
    const int dof_count  = state.GetBlockDofCount(i);

    // Get the element's external force pointer and copy from unified buffer.
    double* d_elem_f_ext = nullptr;
    switch (type) {
      case TYPE_3243: {
        auto* data    = static_cast<GPU_ANCF3243_Data*>(element);
        d_elem_f_ext  = data->GetExternalForceDevicePtr();
        break;
      }
      case TYPE_3443: {
        auto* data    = static_cast<GPU_ANCF3443_Data*>(element);
        d_elem_f_ext  = data->GetExternalForceDevicePtr();
        break;
      }
      case TYPE_T10: {
        auto* data    = static_cast<GPU_FEAT10_Data*>(element);
        d_elem_f_ext  = data->GetExternalForceDevicePtr();
        break;
      }
    }

    if (d_elem_f_ext != nullptr) {
      HANDLE_ERROR(cudaMemcpy(d_elem_f_ext, state.d_f_ext + dof_offset,
                              dof_count * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }
  }
}

void MultiElementNewtonSolver::CollectPositions() {
  // Sync positions from element blocks to unified buffer.
  problem_->SyncPositionsFromElements();
}
