/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    FEMultiElementProblem.cu
 * Brief:   Implementation of FEMultiElementProblem - container for
 *          multi-element FE problems with unified state buffers.
 *==============================================================
 *==============================================================*/

#include "FEMultiElementProblem.h"

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

FEMultiElementProblem::FEMultiElementProblem() = default;

FEMultiElementProblem::~FEMultiElementProblem() {
  // Free unified state buffers.
  if (state_.d_x12)
    cudaFree(state_.d_x12);
  if (state_.d_y12)
    cudaFree(state_.d_y12);
  if (state_.d_z12)
    cudaFree(state_.d_z12);
  if (state_.d_velocity)
    cudaFree(state_.d_velocity);
  if (state_.d_f_ext)
    cudaFree(state_.d_f_ext);
  if (state_.d_nodes_collision)
    cudaFree(state_.d_nodes_collision);

  // Note: Element data is NOT owned by this class - caller manages lifetime.
  // This allows elements to be used with single-element solvers as well.
}

// -----------------------------------------------------------------------------
// Block Management
// -----------------------------------------------------------------------------

int FEMultiElementProblem::AddElementBlock(ElementBase* element,
                                           ElementType type) {
  if (finalized_) {
    throw std::runtime_error(
        "FEMultiElementProblem: cannot add blocks after Finalize()");
  }
  if (element == nullptr) {
    throw std::invalid_argument("FEMultiElementProblem: element cannot be null");
  }

  Block block;
  block.element     = element;
  block.type        = type;
  block.coef_offset = 0;  // Computed in Finalize()
  block.n_coef      = element->get_n_coef();

  // Get constraint count based on element type.
  switch (type) {
    case TYPE_3243: {
      auto* data         = static_cast<GPU_ANCF3243_Data*>(element);
      block.n_constraint = data->get_n_constraint();
      break;
    }
    case TYPE_3443: {
      auto* data         = static_cast<GPU_ANCF3443_Data*>(element);
      block.n_constraint = data->get_n_constraint();
      break;
    }
    case TYPE_T10: {
      auto* data         = static_cast<GPU_FEAT10_Data*>(element);
      block.n_constraint = data->get_n_constraint();
      break;
    }
    default:
      throw std::invalid_argument("FEMultiElementProblem: unknown element type");
  }

  int block_idx = static_cast<int>(blocks_.size());
  blocks_.push_back(block);
  return block_idx;
}

void FEMultiElementProblem::Finalize() {
  if (finalized_) {
    return;
  }
  if (blocks_.empty()) {
    throw std::runtime_error(
        "FEMultiElementProblem: no element blocks added before Finalize()");
  }

  // Compute coefficient offsets for each block.
  int running_offset = 0;
  for (auto& block : blocks_) {
    block.coef_offset = running_offset;
    running_offset += block.n_coef;
  }
  state_.total_coef = running_offset;

  // Populate block ranges in state buffer.
  state_.blocks.clear();
  state_.blocks.reserve(blocks_.size());
  for (const auto& block : blocks_) {
    FEStateBuffer::BlockRange range;
    range.coef_offset = block.coef_offset;
    range.coef_count  = block.n_coef;
    state_.blocks.push_back(range);
  }

  // Allocate unified GPU buffers.
  const size_t coef_bytes = static_cast<size_t>(state_.total_coef) * sizeof(double);
  const size_t dof_bytes  = coef_bytes * 3;

  HANDLE_ERROR(cudaMalloc(&state_.d_x12, coef_bytes));
  HANDLE_ERROR(cudaMalloc(&state_.d_y12, coef_bytes));
  HANDLE_ERROR(cudaMalloc(&state_.d_z12, coef_bytes));
  HANDLE_ERROR(cudaMalloc(&state_.d_velocity, dof_bytes));
  HANDLE_ERROR(cudaMalloc(&state_.d_f_ext, dof_bytes));
  HANDLE_ERROR(cudaMalloc(&state_.d_nodes_collision, dof_bytes));

  // Initialize buffers to zero.
  HANDLE_ERROR(cudaMemset(state_.d_x12, 0, coef_bytes));
  HANDLE_ERROR(cudaMemset(state_.d_y12, 0, coef_bytes));
  HANDLE_ERROR(cudaMemset(state_.d_z12, 0, coef_bytes));
  HANDLE_ERROR(cudaMemset(state_.d_velocity, 0, dof_bytes));
  HANDLE_ERROR(cudaMemset(state_.d_f_ext, 0, dof_bytes));
  HANDLE_ERROR(cudaMemset(state_.d_nodes_collision, 0, dof_bytes));

  // Copy initial positions from element blocks to unified buffer.
  SyncPositionsFromElements();

  finalized_ = true;

  std::cout << "FEMultiElementProblem: Finalized with " << blocks_.size()
            << " blocks, " << state_.total_coef << " total coefficients, "
            << GetTotalDofs() << " total DOFs\n";
}

// -----------------------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------------------

ElementBase* FEMultiElementProblem::GetElementData(int block_idx) const {
  if (block_idx < 0 || block_idx >= static_cast<int>(blocks_.size())) {
    throw std::out_of_range("FEMultiElementProblem: block index out of range");
  }
  return blocks_[block_idx].element;
}

ElementType FEMultiElementProblem::GetElementType(int block_idx) const {
  if (block_idx < 0 || block_idx >= static_cast<int>(blocks_.size())) {
    throw std::out_of_range("FEMultiElementProblem: block index out of range");
  }
  return blocks_[block_idx].type;
}

int FEMultiElementProblem::GetTotalConstraints() const {
  int total = 0;
  for (const auto& block : blocks_) {
    total += block.n_constraint;
  }
  return total;
}

// -----------------------------------------------------------------------------
// State Synchronization
// -----------------------------------------------------------------------------

void FEMultiElementProblem::SyncPositionsToElements() {
  for (const auto& block : blocks_) {
    const size_t count_bytes = static_cast<size_t>(block.n_coef) * sizeof(double);

    switch (block.type) {
      case TYPE_3243: {
        auto* data = static_cast<GPU_ANCF3243_Data*>(block.element);
        HANDLE_ERROR(cudaMemcpy(data->GetX12DevicePtr(),
                                state_.d_x12 + block.coef_offset, count_bytes,
                                cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(data->GetY12DevicePtr(),
                                state_.d_y12 + block.coef_offset, count_bytes,
                                cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(data->GetZ12DevicePtr(),
                                state_.d_z12 + block.coef_offset, count_bytes,
                                cudaMemcpyDeviceToDevice));
        break;
      }
      case TYPE_3443: {
        auto* data = static_cast<GPU_ANCF3443_Data*>(block.element);
        HANDLE_ERROR(cudaMemcpy(data->GetX12DevicePtr(),
                                state_.d_x12 + block.coef_offset, count_bytes,
                                cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(data->GetY12DevicePtr(),
                                state_.d_y12 + block.coef_offset, count_bytes,
                                cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(data->GetZ12DevicePtr(),
                                state_.d_z12 + block.coef_offset, count_bytes,
                                cudaMemcpyDeviceToDevice));
        break;
      }
      case TYPE_T10: {
        auto* data = static_cast<GPU_FEAT10_Data*>(block.element);
        HANDLE_ERROR(cudaMemcpy(data->GetX12DevicePtr(),
                                state_.d_x12 + block.coef_offset, count_bytes,
                                cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(data->GetY12DevicePtr(),
                                state_.d_y12 + block.coef_offset, count_bytes,
                                cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(data->GetZ12DevicePtr(),
                                state_.d_z12 + block.coef_offset, count_bytes,
                                cudaMemcpyDeviceToDevice));
        break;
      }
    }
  }
}

void FEMultiElementProblem::SyncPositionsFromElements() {
  for (const auto& block : blocks_) {
    const size_t count_bytes = static_cast<size_t>(block.n_coef) * sizeof(double);

    switch (block.type) {
      case TYPE_3243: {
        auto* data = static_cast<GPU_ANCF3243_Data*>(block.element);
        HANDLE_ERROR(cudaMemcpy(state_.d_x12 + block.coef_offset,
                                data->GetX12DevicePtr(), count_bytes,
                                cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(state_.d_y12 + block.coef_offset,
                                data->GetY12DevicePtr(), count_bytes,
                                cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(state_.d_z12 + block.coef_offset,
                                data->GetZ12DevicePtr(), count_bytes,
                                cudaMemcpyDeviceToDevice));
        break;
      }
      case TYPE_3443: {
        auto* data = static_cast<GPU_ANCF3443_Data*>(block.element);
        HANDLE_ERROR(cudaMemcpy(state_.d_x12 + block.coef_offset,
                                data->GetX12DevicePtr(), count_bytes,
                                cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(state_.d_y12 + block.coef_offset,
                                data->GetY12DevicePtr(), count_bytes,
                                cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(state_.d_z12 + block.coef_offset,
                                data->GetZ12DevicePtr(), count_bytes,
                                cudaMemcpyDeviceToDevice));
        break;
      }
      case TYPE_T10: {
        auto* data = static_cast<GPU_FEAT10_Data*>(block.element);
        HANDLE_ERROR(cudaMemcpy(state_.d_x12 + block.coef_offset,
                                data->GetX12DevicePtr(), count_bytes,
                                cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(state_.d_y12 + block.coef_offset,
                                data->GetY12DevicePtr(), count_bytes,
                                cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemcpy(state_.d_z12 + block.coef_offset,
                                data->GetZ12DevicePtr(), count_bytes,
                                cudaMemcpyDeviceToDevice));
        break;
      }
    }
  }
}

void FEMultiElementProblem::UpdateCollisionNodeBuffer() {
  const int n = state_.total_coef;
  // Copy x12, y12, z12 to column-major layout: [x0..xn, y0..yn, z0..zn]
  HANDLE_ERROR(cudaMemcpy(state_.d_nodes_collision, state_.d_x12,
                          n * sizeof(double), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(state_.d_nodes_collision + n, state_.d_y12,
                          n * sizeof(double), cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(state_.d_nodes_collision + 2 * n, state_.d_z12,
                          n * sizeof(double), cudaMemcpyDeviceToDevice));
}

// -----------------------------------------------------------------------------
// Physics Kernel Dispatch
// -----------------------------------------------------------------------------

void FEMultiElementProblem::CalcInternalForce() {
  for (int i = 0; i < static_cast<int>(blocks_.size()); ++i) {
    CalcInternalForceBlock(i);
  }
}

void FEMultiElementProblem::CalcP() {
  for (int i = 0; i < static_cast<int>(blocks_.size()); ++i) {
    CalcPBlock(i);
  }
}

void FEMultiElementProblem::CalcConstraintData() {
  for (int i = 0; i < static_cast<int>(blocks_.size()); ++i) {
    CalcConstraintDataBlock(i);
  }
}

void FEMultiElementProblem::CalcInternalForceBlock(int block_idx) {
  const auto& block = blocks_[block_idx];
  switch (block.type) {
    case TYPE_3243: {
      auto* data = static_cast<GPU_ANCF3243_Data*>(block.element);
      data->CalcInternalForce();
      break;
    }
    case TYPE_3443: {
      auto* data = static_cast<GPU_ANCF3443_Data*>(block.element);
      data->CalcInternalForce();
      break;
    }
    case TYPE_T10: {
      auto* data = static_cast<GPU_FEAT10_Data*>(block.element);
      data->CalcInternalForce();
      break;
    }
  }
}

void FEMultiElementProblem::CalcPBlock(int block_idx) {
  const auto& block = blocks_[block_idx];
  switch (block.type) {
    case TYPE_3243: {
      auto* data = static_cast<GPU_ANCF3243_Data*>(block.element);
      data->CalcP();
      break;
    }
    case TYPE_3443: {
      auto* data = static_cast<GPU_ANCF3443_Data*>(block.element);
      data->CalcP();
      break;
    }
    case TYPE_T10: {
      auto* data = static_cast<GPU_FEAT10_Data*>(block.element);
      data->CalcP();
      break;
    }
  }
}

void FEMultiElementProblem::CalcConstraintDataBlock(int block_idx) {
  const auto& block = blocks_[block_idx];
  switch (block.type) {
    case TYPE_3243: {
      auto* data = static_cast<GPU_ANCF3243_Data*>(block.element);
      data->CalcConstraintData();
      break;
    }
    case TYPE_3443: {
      auto* data = static_cast<GPU_ANCF3443_Data*>(block.element);
      data->CalcConstraintData();
      break;
    }
    case TYPE_T10: {
      auto* data = static_cast<GPU_FEAT10_Data*>(block.element);
      data->CalcConstraintData();
      break;
    }
  }
}
