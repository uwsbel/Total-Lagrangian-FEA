/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    FEMultiElementProblem.h
 * Brief:   Container for multi-element FE problems that manages multiple
 *          element blocks (ANCF, FEAT10, etc.) with unified state buffers.
 *==============================================================
 *==============================================================*/

#pragma once

#include <memory>
#include <vector>

#include "FEStateBuffer.h"
#include "lib_src/elements/ElementBase.h"

// Forward declarations to avoid including heavy CUDA headers.
struct GPU_ANCF3243_Data;
struct GPU_ANCF3443_Data;
struct GPU_FEAT10_Data;

// Container class for multi-element FE problems.
//
// This class manages multiple element blocks (each of potentially different
// types) with unified state buffers. It provides a consistent interface for:
//   - Adding element blocks with their type tags
//   - Allocating unified state buffers with proper block offsets
//   - Dispatching physics kernels to the correct element type
//   - Providing collision systems with unified node/force buffers
//
// Usage:
//   FEMultiElementProblem problem;
//   int block0 = problem.AddElementBlock(std::move(ancf_data), TYPE_3243);
//   int block1 = problem.AddElementBlock(std::move(t10_data), TYPE_T10);
//   problem.Finalize();
//   // Now state_.d_x12, etc. span both blocks
//
class FEMultiElementProblem {
 public:
  FEMultiElementProblem();
  ~FEMultiElementProblem();

  // Disallow copy/move for CUDA resource safety.
  FEMultiElementProblem(const FEMultiElementProblem&)            = delete;
  FEMultiElementProblem& operator=(const FEMultiElementProblem&) = delete;
  FEMultiElementProblem(FEMultiElementProblem&&)                 = delete;
  FEMultiElementProblem& operator=(FEMultiElementProblem&&)      = delete;

  // Add an element block. Returns the block index (0-based).
  // The problem takes ownership of the element data.
  // Must call Finalize() after adding all blocks.
  int AddElementBlock(ElementBase* element, ElementType type);

  // Finalize the problem after all blocks have been added.
  // Allocates unified state buffers and copies initial positions.
  void Finalize();

  // Check if the problem has been finalized.
  bool IsFinalized() const { return finalized_; }

  // Get the unified state buffer.
  const FEStateBuffer& GetStateBuffer() const { return state_; }
  FEStateBuffer& GetStateBuffer() { return state_; }

  // Get the number of element blocks.
  int GetNumBlocks() const { return static_cast<int>(blocks_.size()); }

  // Get element data for a block (caller must cast to correct type).
  ElementBase* GetElementData(int block_idx) const;
  ElementType GetElementType(int block_idx) const;

  // Accessors for total counts.
  int GetTotalCoef() const { return state_.total_coef; }
  int GetTotalDofs() const { return state_.GetTotalDofs(); }
  int GetTotalConstraints() const;

  // ---------------------------------------------------------------------------
  // Physics kernel dispatch
  // These call the appropriate element-specific GPU kernels for each block.
  // ---------------------------------------------------------------------------

  // Compute internal forces for all blocks.
  void CalcInternalForce();

  // Compute Piola stress for all blocks.
  void CalcP();

  // Compute constraint data for all blocks.
  void CalcConstraintData();

  // ---------------------------------------------------------------------------
  // State synchronization
  // ---------------------------------------------------------------------------

  // Sync unified buffer positions to element-local buffers.
  // Call after modifying state_.d_x12/d_y12/d_z12.
  void SyncPositionsToElements();

  // Sync element-local positions to unified buffer.
  // Call after elements update their positions internally.
  void SyncPositionsFromElements();

  // Update the collision node buffer from current positions.
  // Copies [d_x12, d_y12, d_z12] -> d_nodes_collision in column-major layout.
  void UpdateCollisionNodeBuffer();

  void SetVelocityFromHostPtr(const double* h_vel_xyz, int n_dofs);
  void SetVelocityBlockFromHostPtr(int block_idx, const double* h_vel_xyz,
                                   int n_dofs);
  void SetVelocityFromDevicePtr(const double* d_vel_xyz, int n_dofs);
  void SetVelocityBlockFromDevicePtr(int block_idx, const double* d_vel_xyz,
                                     int n_dofs);

 private:
  struct Block {
    ElementBase* element;  // Owned pointer
    ElementType type;
    int coef_offset;  // Coefficient offset in unified buffer
    int n_coef;       // Number of coefficients in this block
    int n_constraint; // Number of constraints in this block
  };

  std::vector<Block> blocks_;
  FEStateBuffer state_;
  bool finalized_ = false;

  // Internal helpers for type-specific dispatch.
  void CalcInternalForceBlock(int block_idx);
  void CalcPBlock(int block_idx);
  void CalcConstraintDataBlock(int block_idx);
};
