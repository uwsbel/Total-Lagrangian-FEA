/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  OpenAI Codex
 * File:    HolisticNewton.cu
 * Brief:   Monolithic Newton solver for multi-element problems with a
 *          global mixed-element constraint system.
 *==============================================================
 *==============================================================*/

#include "HolisticNewton.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "lib_src/elements/ANCF3243Data.cuh"
#include "lib_src/elements/ANCF3243DataFunc.cuh"
#include "lib_src/elements/FEAT10Data.cuh"
#include "lib_src/elements/FEAT10DataFunc.cuh"
#include "lib_utils/cuda_utils.h"

namespace {

__device__ __forceinline__ int binary_search_column(const int* cols, int len,
                                                    int target) {
  int left = 0;
  int right = len - 1;
  while (left <= right) {
    const int mid = left + ((right - left) >> 1);
    const int val = cols[mid];
    if (val == target) {
      return mid;
    }
    if (val < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}

__global__ void backup_positions_kernel(int n_coef, const double* d_x,
                                        const double* d_y, const double* d_z,
                                        double* d_x_prev, double* d_y_prev,
                                        double* d_z_prev) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_coef) {
    d_x_prev[tid] = d_x[tid];
    d_y_prev[tid] = d_y[tid];
    d_z_prev[tid] = d_z[tid];
  }
}

__global__ void update_positions_from_velocity_kernel(
    int n_coef, double dt, const double* d_x_prev, const double* d_y_prev,
    const double* d_z_prev, const double* d_v_guess, double* d_x, double* d_y,
    double* d_z) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_coef) {
    d_x[tid] = d_x_prev[tid] + dt * d_v_guess[tid * 3 + 0];
    d_y[tid] = d_y_prev[tid] + dt * d_v_guess[tid * 3 + 1];
    d_z[tid] = d_z_prev[tid] + dt * d_v_guess[tid * 3 + 2];
  }
}

template <typename ElementType>
__global__ void accumulate_block_gradient_kernel(
    ElementType* d_data, int coef_offset, const double* d_v_guess,
    const double* d_v_prev, double dt, double* d_g) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= d_data->n_coef * 3) {
    return;
  }

  const int coef_i = tid / 3;
  const int axis = tid % 3;
  const int global_tid = 3 * (coef_offset + coef_i) + axis;
  const double inv_dt = 1.0 / dt;

  double res = 0.0;
  const int* mass_offsets = d_data->csr_offsets();
  const int* mass_columns = d_data->csr_columns();
  const double* mass_values = d_data->csr_values();
  const int row_start = mass_offsets[coef_i];
  const int row_end = mass_offsets[coef_i + 1];

  for (int idx = row_start; idx < row_end; ++idx) {
    const int coef_j = mass_columns[idx];
    const double mass_ij = mass_values[idx];
    const int global_tid_j = 3 * (coef_offset + coef_j) + axis;
    res += mass_ij * (d_v_guess[global_tid_j] - d_v_prev[global_tid_j]) *
           inv_dt;
  }

  res -= (-d_data->f_int()(tid));
  res -= d_data->f_ext()(tid);
  atomicAdd(&d_g[global_tid], res);
}

__global__ void add_mixed_constraint_gradient_kernel(
    int n_dofs, int n_constraints, double dt, double rho,
    const int* d_cj_offsets, const int* d_cj_columns, const double* d_cj_values,
    const double* d_lambda, const double* d_constraint, double* d_g) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_dofs) {
    return;
  }

  double res = 0.0;
  const int row_start = d_cj_offsets[tid];
  const int row_end = d_cj_offsets[tid + 1];
  for (int idx = row_start; idx < row_end; ++idx) {
    const int constraint_idx = d_cj_columns[idx];
    if (constraint_idx < 0 || constraint_idx >= n_constraints) {
      continue;
    }
    res += dt * d_cj_values[idx] *
           (d_lambda[constraint_idx] + rho * d_constraint[constraint_idx]);
  }
  d_g[tid] += res;
}

__global__ void initialize_newton_rhs_kernel(int n_dofs, const double* d_g,
                                             double* d_delta_v, double* d_r) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_dofs) {
    d_delta_v[tid] = 0.0;
    d_r[tid] = -d_g[tid];
  }
}

__global__ void update_velocity_guess_kernel(int n_dofs, double* d_v_guess,
                                             const double* d_delta_v) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_dofs) {
    d_v_guess[tid] += d_delta_v[tid];
  }
}

__global__ void update_dual_variables_kernel(int n_constraints, double rho,
                                             double* d_lambda,
                                             const double* d_constraint) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_constraints) {
    d_lambda[tid] += rho * d_constraint[tid];
  }
}

template <typename ElementType>
__global__ void clear_internal_force_kernel(ElementType* d_data) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_data->n_coef * 3) {
    clear_internal_force(d_data);
  }
}

template <typename ElementType>
__global__ void compute_p_with_velocity_kernel(ElementType* d_data, int n_elem,
                                               int n_qp,
                                               const double* d_v_guess,
                                               double dt) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_elem * n_qp) {
    return;
  }

  const int elem_idx = tid / n_qp;
  const int qp_idx = tid % n_qp;
  compute_p(elem_idx, qp_idx, d_data, d_v_guess, dt);
}

template <typename ElementType>
__global__ void compute_internal_force_kernel_global(ElementType* d_data,
                                                     int n_elem,
                                                     int n_shape) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_elem * n_shape) {
    return;
  }

  const int elem_idx = tid / n_shape;
  const int node_idx = tid % n_shape;
  compute_internal_force(elem_idx, node_idx, d_data);
}

template <typename ElementType>
__global__ void assemble_sparse_hessian_mass_global(
    ElementType* d_data, int coef_offset, double dt, int* d_csr_row_offsets,
    int* d_csr_col_indices, double* d_csr_values) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= d_data->n_coef) {
    return;
  }

  const double inv_dt = 1.0 / dt;
  const int* mass_offsets = d_data->csr_offsets();
  const int* mass_columns = d_data->csr_columns();
  const double* mass_values = d_data->csr_values();
  const int row_start = mass_offsets[tid];
  const int row_end = mass_offsets[tid + 1];

  for (int idx = row_start; idx < row_end; ++idx) {
    const int coef_j = mass_columns[idx];
    const double contrib = mass_values[idx] * inv_dt;

    for (int axis = 0; axis < 3; ++axis) {
      const int global_row = 3 * (coef_offset + tid) + axis;
      const int global_col = 3 * (coef_offset + coef_j) + axis;
      const int csr_row_begin = d_csr_row_offsets[global_row];
      const int csr_row_len =
          d_csr_row_offsets[global_row + 1] - csr_row_begin;
      const int pos = binary_search_column(
          &d_csr_col_indices[csr_row_begin], csr_row_len, global_col);
      if (pos >= 0) {
        atomicAdd(&d_csr_values[csr_row_begin + pos], contrib);
      }
    }
  }
}

template <typename ElementType>
__global__ void assemble_sparse_hessian_tangent_global(
    ElementType* d_data, int coef_offset, int n_elem, int n_qp, double dt,
    int* d_csr_row_offsets, int* d_csr_col_indices, double* d_csr_values) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int elem_idx = tid / n_qp;
  const int qp_idx = tid % n_qp;
  if (elem_idx >= n_elem) {
    return;
  }

  compute_hessian_assemble_csr_global(d_data, elem_idx, qp_idx, coef_offset,
                                      d_csr_row_offsets, d_csr_col_indices,
                                      d_csr_values, dt);
}

__global__ void assemble_sparse_hessian_mixed_constraints(
    int n_constraints, double factor, const int* d_j_offsets,
    const int* d_j_columns, const double* d_j_values, int* d_csr_row_offsets,
    int* d_csr_col_indices, double* d_csr_values) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_constraints) {
    return;
  }

  const int row_start = d_j_offsets[tid];
  const int row_end = d_j_offsets[tid + 1];
  for (int idx_i = row_start; idx_i < row_end; ++idx_i) {
    const int dof_i = d_j_columns[idx_i];
    const double Ji = d_j_values[idx_i];
    const int csr_row_begin = d_csr_row_offsets[dof_i];
    const int csr_row_len = d_csr_row_offsets[dof_i + 1] - csr_row_begin;

    for (int idx_j = row_start; idx_j < row_end; ++idx_j) {
      const int dof_j = d_j_columns[idx_j];
      const double Jj = d_j_values[idx_j];
      const int pos = binary_search_column(&d_csr_col_indices[csr_row_begin],
                                           csr_row_len, dof_j);
      if (pos >= 0) {
        atomicAdd(&d_csr_values[csr_row_begin + pos], factor * Ji * Jj);
      }
    }
  }
}

__global__ void assemble_sparse_hessian_mixed_exact_dot_constraints(
    int n_constraints, const int* d_constraint_types,
    const double* d_constraint_row_scales, const int* d_point_counts,
    const int* d_point_coef_indices, const double* d_point_weights,
    const double* d_lambda, const double* d_constraint, double h_sq,
    double rho,
    int* d_csr_row_offsets, int* d_csr_col_indices, double* d_csr_values) {
  const int constraint_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (constraint_idx >= n_constraints) {
    return;
  }

  const int constraint_type = d_constraint_types[constraint_idx];
  if (constraint_type != kMixedConstraintDP1 &&
      constraint_type != kMixedConstraintDP2) {
    return;
  }

  const double factor =
      h_sq * (d_lambda[constraint_idx] + rho * d_constraint[constraint_idx]) *
      d_constraint_row_scales[constraint_idx];
  if (fabs(factor) < 1e-30) {
    return;
  }

  constexpr double kRoleCoeff[4][4] = {
      {0.0, 0.0, 1.0, -1.0},
      {0.0, 0.0, -1.0, 1.0},
      {1.0, -1.0, 0.0, 0.0},
      {-1.0, 1.0, 0.0, 0.0},
  };

  for (int axis = 0; axis < 3; ++axis) {
    for (int role_a = 0; role_a < 4; ++role_a) {
      const int point_offset_a = constraint_idx * 4 + role_a;
      const int count_a = d_point_counts[point_offset_a];
      if (count_a == 0) {
        continue;
      }
      const int base_a =
          point_offset_a * MixedConstraintPointBinding::kMaxCoefficients;

      for (int role_b = 0; role_b < 4; ++role_b) {
        const double coeff = kRoleCoeff[role_a][role_b];
        if (coeff == 0.0) {
          continue;
        }

        const int point_offset_b = constraint_idx * 4 + role_b;
        const int count_b = d_point_counts[point_offset_b];
        if (count_b == 0) {
          continue;
        }
        const int base_b =
            point_offset_b * MixedConstraintPointBinding::kMaxCoefficients;

        for (int idx_a = 0; idx_a < count_a; ++idx_a) {
          const int coef_a = d_point_coef_indices[base_a + idx_a];
          const double weight_a = d_point_weights[base_a + idx_a];
          const int dof_a = coef_a * 3 + axis;
          const int csr_row_begin = d_csr_row_offsets[dof_a];
          const int csr_row_len =
              d_csr_row_offsets[dof_a + 1] - csr_row_begin;

          for (int idx_b = 0; idx_b < count_b; ++idx_b) {
            const int coef_b = d_point_coef_indices[base_b + idx_b];
            const double weight_b = d_point_weights[base_b + idx_b];
            const int dof_b = coef_b * 3 + axis;
            const int pos = binary_search_column(
                &d_csr_col_indices[csr_row_begin], csr_row_len, dof_b);
            if (pos >= 0) {
              atomicAdd(&d_csr_values[csr_row_begin + pos],
                        factor * coeff * weight_a * weight_b);
            }
          }
        }
      }
    }
  }
}

}  // namespace

HolisticNewtonSolver::HolisticNewtonSolver(
    FEMultiElementProblem* problem, MixedConstraintSystem* mixed_constraints)
    : problem_(problem), mixed_constraints_(mixed_constraints) {
  if (problem_ == nullptr) {
    throw std::invalid_argument(
        "HolisticNewtonSolver: problem pointer cannot be null");
  }
  if (mixed_constraints_ == nullptr) {
    throw std::invalid_argument(
        "HolisticNewtonSolver: mixed constraint system cannot be null");
  }
  if (!problem_->IsFinalized()) {
    throw std::invalid_argument(
        "HolisticNewtonSolver: problem must be finalized before solver construction");
  }

  n_coef_ = problem_->GetTotalCoef();
  n_dofs_ = problem_->GetTotalDofs();

  HANDLE_ERROR(cudaMalloc(&d_v_guess_, static_cast<size_t>(n_dofs_) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_v_prev_, static_cast<size_t>(n_dofs_) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_g_, static_cast<size_t>(n_dofs_) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_delta_v_, static_cast<size_t>(n_dofs_) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_r_, static_cast<size_t>(n_dofs_) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_v_trial_backup_,
                          static_cast<size_t>(n_dofs_) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_x12_prev_,
                          static_cast<size_t>(n_coef_) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_y12_prev_,
                          static_cast<size_t>(n_coef_) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_z12_prev_,
                          static_cast<size_t>(n_coef_) * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_norm_temp_, sizeof(double)));

  cublasCreate(&cublas_handle_);
  cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE);
}

HolisticNewtonSolver::~HolisticNewtonSolver() {
  if (d_v_guess_ != nullptr) {
    cudaFree(d_v_guess_);
  }
  if (d_v_prev_ != nullptr) {
    cudaFree(d_v_prev_);
  }
  if (d_lambda_guess_ != nullptr) {
    cudaFree(d_lambda_guess_);
  }
  if (d_g_ != nullptr) {
    cudaFree(d_g_);
  }
  if (d_delta_v_ != nullptr) {
    cudaFree(d_delta_v_);
  }
  if (d_r_ != nullptr) {
    cudaFree(d_r_);
  }
  if (d_x12_prev_ != nullptr) {
    cudaFree(d_x12_prev_);
  }
  if (d_y12_prev_ != nullptr) {
    cudaFree(d_y12_prev_);
  }
  if (d_z12_prev_ != nullptr) {
    cudaFree(d_z12_prev_);
  }
  if (d_v_trial_backup_ != nullptr) {
    cudaFree(d_v_trial_backup_);
  }
  if (d_norm_temp_ != nullptr) {
    cudaFree(d_norm_temp_);
  }
  if (d_csr_row_offsets_ != nullptr) {
    cudaFree(d_csr_row_offsets_);
  }
  if (d_csr_col_indices_ != nullptr) {
    cudaFree(d_csr_col_indices_);
  }
  if (d_csr_values_ != nullptr) {
    cudaFree(d_csr_values_);
  }
  if (cublas_handle_ != nullptr) {
    cublasDestroy(cublas_handle_);
  }
  if (cudss_data_ != nullptr) {
    cudssDataDestroy(cudss_handle_, cudss_data_);
  }
  if (cudss_config_ != nullptr) {
    cudssConfigDestroy(cudss_config_);
  }
  if (cudss_handle_ != nullptr) {
    cudssDestroy(cudss_handle_);
  }
}

void HolisticNewtonSolver::SetParameters(void* params) {
  if (params == nullptr) {
    throw std::invalid_argument(
        "HolisticNewtonSolver::SetParameters: params cannot be null");
  }
  params_ = *static_cast<HolisticNewtonParams*>(params);
}

double* HolisticNewtonSolver::GetExternalForceDevicePtr() const {
  return problem_->GetStateBuffer().d_f_ext;
}

void HolisticNewtonSolver::InitializeBlockInfos() {
  block_infos_.clear();
  block_infos_.reserve(problem_->GetNumBlocks());

  const FEStateBuffer& state = problem_->GetStateBuffer();
  for (int block_idx = 0; block_idx < problem_->GetNumBlocks(); ++block_idx) {
    BlockLaunchInfo info;
    info.element = problem_->GetElementData(block_idx);
    info.type = problem_->GetElementType(block_idx);
    info.coef_offset = state.blocks[static_cast<size_t>(block_idx)].coef_offset;
    info.coef_count = state.blocks[static_cast<size_t>(block_idx)].coef_count;

    switch (info.type) {
      case TYPE_3243: {
        auto* data = static_cast<GPU_ANCF3243_Data*>(info.element);
        info.n_elem = data->get_n_beam();
        info.n_qp = Quadrature::N_TOTAL_QP_3_2_2;
        break;
      }
      case TYPE_T10: {
        auto* data = static_cast<GPU_FEAT10_Data*>(info.element);
        info.n_elem = data->get_n_beam();
        info.n_qp = Quadrature::N_QP_T10_5;
        break;
      }
      default:
        throw std::invalid_argument(
            "HolisticNewtonSolver: only TYPE_3243 and TYPE_T10 are supported");
    }

    block_infos_.push_back(info);
  }
}

void HolisticNewtonSolver::ValidateProblemConfiguration() const {
  for (int block_idx = 0; block_idx < problem_->GetNumBlocks(); ++block_idx) {
    const ElementType type = problem_->GetElementType(block_idx);
    switch (type) {
      case TYPE_3243: {
        auto* data =
            static_cast<GPU_ANCF3243_Data*>(problem_->GetElementData(block_idx));
        if (data->get_n_constraint() != 0) {
          throw std::invalid_argument(
              "HolisticNewtonSolver currently requires ANCF3243 block-local constraints to be moved into MixedConstraintSystem");
        }
        break;
      }
      case TYPE_T10: {
        auto* data =
            static_cast<GPU_FEAT10_Data*>(problem_->GetElementData(block_idx));
        if (data->get_n_constraint() != 0) {
          throw std::invalid_argument(
              "HolisticNewtonSolver currently requires FEAT10 block-local constraints to be moved into MixedConstraintSystem");
        }
        break;
      }
      default:
        throw std::invalid_argument(
            "HolisticNewtonSolver: unsupported element type in multi-element problem");
    }
  }
}

void HolisticNewtonSolver::BuildConstraintAwareSparsityPattern(
    std::vector<int>& row_offsets, std::vector<int>& col_indices) {
  std::vector<std::vector<int>> coef_adj(static_cast<size_t>(n_coef_));
  for (int coef = 0; coef < n_coef_; ++coef) {
    coef_adj[static_cast<size_t>(coef)].push_back(coef);
  }

  for (const BlockLaunchInfo& block : block_infos_) {
    std::vector<int> mass_offsets;
    std::vector<int> mass_columns;
    std::vector<double> mass_values;

    if (block.type == TYPE_3243) {
      auto* data = static_cast<GPU_ANCF3243_Data*>(block.element);
      data->RetrieveMassCSRToCPU(mass_offsets, mass_columns, mass_values);
    } else {
      auto* data = static_cast<GPU_FEAT10_Data*>(block.element);
      data->RetrieveMassCSRToCPU(mass_offsets, mass_columns, mass_values);
    }

    for (int local_coef = 0; local_coef < block.coef_count; ++local_coef) {
      const int row_start = mass_offsets[static_cast<size_t>(local_coef)];
      const int row_end = mass_offsets[static_cast<size_t>(local_coef + 1)];
      auto& nbrs =
          coef_adj[static_cast<size_t>(block.coef_offset + local_coef)];
      for (int idx = row_start; idx < row_end; ++idx) {
        nbrs.push_back(block.coef_offset + mass_columns[static_cast<size_t>(idx)]);
      }
    }
  }

  if (n_constraints_ > 0) {
    const MixedConstraintLayout& layout = mixed_constraints_->host_layout();
    std::vector<int> row_coefs;
    row_coefs.reserve(8);
    for (int row = 0; row < n_constraints_; ++row) {
      row_coefs.clear();
      const int row_start = layout.j_offsets[static_cast<size_t>(row)];
      const int row_end = layout.j_offsets[static_cast<size_t>(row + 1)];
      for (int idx = row_start; idx < row_end; ++idx) {
        const int dof = layout.j_columns[static_cast<size_t>(idx)];
        row_coefs.push_back(dof / 3);
      }
      std::sort(row_coefs.begin(), row_coefs.end());
      row_coefs.erase(std::unique(row_coefs.begin(), row_coefs.end()),
                      row_coefs.end());
      for (size_t a = 0; a < row_coefs.size(); ++a) {
        for (size_t b = a; b < row_coefs.size(); ++b) {
          coef_adj[static_cast<size_t>(row_coefs[a])].push_back(row_coefs[b]);
          coef_adj[static_cast<size_t>(row_coefs[b])].push_back(row_coefs[a]);
        }
      }
    }
  }

  for (auto& nbrs : coef_adj) {
    std::sort(nbrs.begin(), nbrs.end());
    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
  }

  row_offsets.assign(static_cast<size_t>(n_dofs_) + 1, 0);
  col_indices.clear();
  col_indices.reserve(static_cast<size_t>(n_dofs_) * 24);

  int running = 0;
  for (int dof_row = 0; dof_row < n_dofs_; ++dof_row) {
    const int coef_i = dof_row / 3;
    row_offsets[static_cast<size_t>(dof_row)] = running;
    for (int coef_j : coef_adj[static_cast<size_t>(coef_i)]) {
      const int base = 3 * coef_j;
      col_indices.push_back(base + 0);
      col_indices.push_back(base + 1);
      col_indices.push_back(base + 2);
      running += 3;
    }
  }
  row_offsets[static_cast<size_t>(n_dofs_)] = running;
}

void HolisticNewtonSolver::AnalyzeHessianSparsity() {
  if (sparse_hessian_initialized_) {
    return;
  }

  std::vector<int> row_offsets;
  std::vector<int> col_indices;
  BuildConstraintAwareSparsityPattern(row_offsets, col_indices);

  h_nnz_ = static_cast<int>(col_indices.size());
  HANDLE_ERROR(cudaMalloc(&d_csr_row_offsets_,
                          row_offsets.size() * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_csr_col_indices_,
                          static_cast<size_t>(h_nnz_) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_csr_values_,
                          static_cast<size_t>(h_nnz_) * sizeof(double)));

  HANDLE_ERROR(cudaMemcpy(d_csr_row_offsets_, row_offsets.data(),
                          row_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_csr_col_indices_, col_indices.data(),
                          static_cast<size_t>(h_nnz_) * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(d_csr_values_, 0,
                          static_cast<size_t>(h_nnz_) * sizeof(double)));

  sparse_hessian_initialized_ = true;
}

void HolisticNewtonSolver::Setup() {
  if (setup_done_) {
    return;
  }

  ValidateProblemConfiguration();
  if (!mixed_constraints_->IsFinalized()) {
    mixed_constraints_->Finalize();
  }
  n_constraints_ = mixed_constraints_->num_constraints();
  use_symmetric_constraint_hessian_ =
      mixed_constraints_->HasNonlinearDotConstraints();
  if (n_constraints_ > 0) {
    HANDLE_ERROR(cudaMalloc(&d_lambda_guess_,
                            static_cast<size_t>(n_constraints_) * sizeof(double)));
    HANDLE_ERROR(cudaMemset(d_lambda_guess_, 0,
                            static_cast<size_t>(n_constraints_) * sizeof(double)));
  }

  InitializeBlockInfos();
  AnalyzeHessianSparsity();

  HANDLE_ERROR(cudaMemcpy(d_v_guess_, problem_->GetStateBuffer().d_velocity,
                          static_cast<size_t>(n_dofs_) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemcpy(d_v_prev_, problem_->GetStateBuffer().d_velocity,
                          static_cast<size_t>(n_dofs_) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  HANDLE_ERROR(cudaMemset(d_g_, 0, static_cast<size_t>(n_dofs_) * sizeof(double)));
  HANDLE_ERROR(cudaMemset(d_delta_v_, 0,
                          static_cast<size_t>(n_dofs_) * sizeof(double)));
  HANDLE_ERROR(cudaMemset(d_r_, 0, static_cast<size_t>(n_dofs_) * sizeof(double)));

  int ir_n_steps = 0;
  cudssAlgType_t reorder = CUDSS_ALG_DEFAULT;
  CUDSS_OK(cudssCreate(&cudss_handle_));
  CUDSS_OK(cudssConfigCreate(&cudss_config_));
  CUDSS_OK(cudssDataCreate(cudss_handle_, &cudss_data_));
  CUDSS_OK(cudssConfigSet(cudss_config_, CUDSS_CONFIG_REORDERING_ALG, &reorder,
                          sizeof(reorder)));
  CUDSS_OK(cudssConfigSet(cudss_config_, CUDSS_CONFIG_IR_N_STEPS,
                          &ir_n_steps, sizeof(ir_n_steps)));

  setup_done_ = true;
}

void HolisticNewtonSolver::DistributeExternalForces() {
  const FEStateBuffer& state = problem_->GetStateBuffer();
  for (int block_idx = 0; block_idx < problem_->GetNumBlocks(); ++block_idx) {
    const int dof_offset = state.GetBlockDofOffset(block_idx);
    const int dof_count = state.GetBlockDofCount(block_idx);

    if (problem_->GetElementType(block_idx) == TYPE_3243) {
      auto* data = static_cast<GPU_ANCF3243_Data*>(problem_->GetElementData(block_idx));
      HANDLE_ERROR(cudaMemcpy(data->GetExternalForceDevicePtr(),
                              state.d_f_ext + dof_offset,
                              static_cast<size_t>(dof_count) * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    } else {
      auto* data = static_cast<GPU_FEAT10_Data*>(problem_->GetElementData(block_idx));
      HANDLE_ERROR(cudaMemcpy(data->GetExternalForceDevicePtr(),
                              state.d_f_ext + dof_offset,
                              static_cast<size_t>(dof_count) * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }
  }
}

void HolisticNewtonSolver::BackupCurrentPositions() {
  const FEStateBuffer& state = problem_->GetStateBuffer();
  const int threads = 256;
  const int blocks = (n_coef_ + threads - 1) / threads;
  backup_positions_kernel<<<blocks, threads>>>(
      n_coef_, state.d_x12, state.d_y12, state.d_z12, d_x12_prev_, d_y12_prev_,
      d_z12_prev_);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void HolisticNewtonSolver::UpdateProblemStateFromVelocity() const {
  FEStateBuffer& state = problem_->GetStateBuffer();
  const int threads = 256;
  const int blocks = (n_coef_ + threads - 1) / threads;
  update_positions_from_velocity_kernel<<<blocks, threads>>>(
      n_coef_, params_.time_step, d_x12_prev_, d_y12_prev_, d_z12_prev_,
      d_v_guess_, state.d_x12, state.d_y12, state.d_z12);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void HolisticNewtonSolver::AssembleGradient() {
  HANDLE_ERROR(cudaMemset(d_g_, 0, static_cast<size_t>(n_dofs_) * sizeof(double)));

  const int threads = 256;
  for (const BlockLaunchInfo& block : block_infos_) {
    const int grad_blocks = (block.coef_count * 3 + threads - 1) / threads;
    if (block.type == TYPE_3243) {
      auto* data = static_cast<GPU_ANCF3243_Data*>(block.element)->d_data;
      accumulate_block_gradient_kernel<<<grad_blocks, threads>>>(
          data, block.coef_offset, d_v_guess_, d_v_prev_, params_.time_step,
          d_g_);
    } else {
      auto* data = static_cast<GPU_FEAT10_Data*>(block.element)->d_data;
      accumulate_block_gradient_kernel<<<grad_blocks, threads>>>(
          data, block.coef_offset, d_v_guess_, d_v_prev_, params_.time_step,
          d_g_);
    }
  }

  if (n_constraints_ > 0) {
    const int grad_blocks = (n_dofs_ + threads - 1) / threads;
    add_mixed_constraint_gradient_kernel<<<grad_blocks, threads>>>(
        n_dofs_, n_constraints_, params_.time_step, params_.rho,
        mixed_constraints_->GetConstraintJacobianTransposeOffsetsDevicePtr(),
        mixed_constraints_->GetConstraintJacobianTransposeColumnsDevicePtr(),
        mixed_constraints_->GetConstraintJacobianTransposeValuesDevicePtr(),
        d_lambda_guess_, mixed_constraints_->GetConstraintDevicePtr(), d_g_);
  }

  HANDLE_ERROR(cudaDeviceSynchronize());
}

void HolisticNewtonSolver::AssembleHessian() {
  HANDLE_ERROR(cudaMemset(d_csr_values_, 0,
                          static_cast<size_t>(h_nnz_) * sizeof(double)));

  const int threads = 256;
  for (const BlockLaunchInfo& block : block_infos_) {
    const int mass_blocks = (block.coef_count + threads - 1) / threads;
    const int tangent_blocks =
        (block.n_elem * block.n_qp + threads - 1) / threads;

    if (block.type == TYPE_3243) {
      auto* data = static_cast<GPU_ANCF3243_Data*>(block.element)->d_data;
      assemble_sparse_hessian_mass_global<<<mass_blocks, threads>>>(
          data, block.coef_offset, params_.time_step, d_csr_row_offsets_,
          d_csr_col_indices_, d_csr_values_);
      assemble_sparse_hessian_tangent_global<<<tangent_blocks, threads>>>(
          data, block.coef_offset, block.n_elem, block.n_qp, params_.time_step,
          d_csr_row_offsets_, d_csr_col_indices_, d_csr_values_);
    } else {
      auto* data = static_cast<GPU_FEAT10_Data*>(block.element)->d_data;
      assemble_sparse_hessian_mass_global<<<mass_blocks, threads>>>(
          data, block.coef_offset, params_.time_step, d_csr_row_offsets_,
          d_csr_col_indices_, d_csr_values_);
      assemble_sparse_hessian_tangent_global<<<tangent_blocks, threads>>>(
          data, block.coef_offset, block.n_elem, block.n_qp, params_.time_step,
          d_csr_row_offsets_, d_csr_col_indices_, d_csr_values_);
    }
  }

  if (n_constraints_ > 0) {
    const int constraint_blocks = (n_constraints_ + threads - 1) / threads;
    assemble_sparse_hessian_mixed_constraints<<<constraint_blocks, threads>>>(
        n_constraints_, params_.time_step * params_.time_step * params_.rho,
        mixed_constraints_->GetConstraintJacobianOffsetsDevicePtr(),
        mixed_constraints_->GetConstraintJacobianColumnsDevicePtr(),
        mixed_constraints_->GetConstraintJacobianValuesDevicePtr(),
        d_csr_row_offsets_, d_csr_col_indices_, d_csr_values_);

    if (use_symmetric_constraint_hessian_) {
      assemble_sparse_hessian_mixed_exact_dot_constraints<<<constraint_blocks,
                                                            threads>>>(
          n_constraints_, mixed_constraints_->GetConstraintTypesDevicePtr(),
          mixed_constraints_->GetConstraintRowScalesDevicePtr(),
          mixed_constraints_->GetPointCountsDevicePtr(),
          mixed_constraints_->GetPointCoefficientIndicesDevicePtr(),
          mixed_constraints_->GetPointWeightsDevicePtr(), d_lambda_guess_,
          mixed_constraints_->GetConstraintDevicePtr(),
          params_.time_step * params_.time_step, params_.rho,
          d_csr_row_offsets_, d_csr_col_indices_, d_csr_values_);
    }
  }

  HANDLE_ERROR(cudaDeviceSynchronize());
}

void HolisticNewtonSolver::EvaluateState() {
  UpdateProblemStateFromVelocity();
  problem_->SyncPositionsToElements();

  const int threads = 256;
  for (const BlockLaunchInfo& block : block_infos_) {
    const int compute_p_blocks =
        (block.n_elem * block.n_qp + threads - 1) / threads;
    const int clear_force_blocks = (block.coef_count * 3 + threads - 1) / threads;
    const int n_shape = (block.type == TYPE_3243) ? Quadrature::N_SHAPE_3243
                                                  : Quadrature::N_NODE_T10_10;
    const int internal_force_blocks =
        (block.n_elem * n_shape + threads - 1) / threads;

    if (block.type == TYPE_3243) {
      auto* data = static_cast<GPU_ANCF3243_Data*>(block.element);
      compute_p_with_velocity_kernel<<<compute_p_blocks, threads>>>(
          data->d_data, block.n_elem, block.n_qp,
          d_v_guess_ + block.coef_offset * 3,
          params_.time_step);
      clear_internal_force_kernel<<<clear_force_blocks, threads>>>(data->d_data);
      compute_internal_force_kernel_global<<<internal_force_blocks, threads>>>(
          data->d_data, block.n_elem, n_shape);
    } else {
      auto* data = static_cast<GPU_FEAT10_Data*>(block.element);
      compute_p_with_velocity_kernel<<<compute_p_blocks, threads>>>(
          data->d_data, block.n_elem, block.n_qp,
          d_v_guess_ + block.coef_offset * 3,
          params_.time_step);
      clear_internal_force_kernel<<<clear_force_blocks, threads>>>(data->d_data);
      compute_internal_force_kernel_global<<<internal_force_blocks, threads>>>(
          data->d_data, block.n_elem, n_shape);
    }
  }

  FEStateBuffer& state = problem_->GetStateBuffer();
  mixed_constraints_->Evaluate(state.d_x12, state.d_y12, state.d_z12);
  AssembleGradient();
}

double HolisticNewtonSolver::ComputeL2Norm(double* d_vec, int n_entries) const {
  cublasDnrm2(cublas_handle_, n_entries, d_vec, 1, d_norm_temp_);
  double h_norm = 0.0;
  HANDLE_ERROR(cudaMemcpy(&h_norm, d_norm_temp_, sizeof(double),
                          cudaMemcpyDeviceToHost));
  return h_norm;
}

void HolisticNewtonSolver::UpdateProblemVelocity() const {
  HANDLE_ERROR(cudaMemcpy(problem_->GetStateBuffer().d_velocity, d_v_guess_,
                          static_cast<size_t>(n_dofs_) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
}

void HolisticNewtonSolver::Solve() {
  if (!setup_done_) {
    throw std::runtime_error(
        "HolisticNewtonSolver::Solve: must call Setup() before Solve()");
  }

  HANDLE_ERROR(cudaMemcpy(d_v_guess_, problem_->GetStateBuffer().d_velocity,
                          static_cast<size_t>(n_dofs_) * sizeof(double),
                          cudaMemcpyDeviceToDevice));

  BackupCurrentPositions();
  DistributeExternalForces();

  cudssMatrix_t dssA;
  cudssMatrix_t dssB;
  cudssMatrix_t dssX;
  const cudssMatrixType_t matrix_type =
      use_symmetric_constraint_hessian_ ? CUDSS_MTYPE_SYMMETRIC
                                        : CUDSS_MTYPE_SPD;
  CUDSS_OK(cudssMatrixCreateCsr(&dssA, n_dofs_, n_dofs_, h_nnz_,
                                d_csr_row_offsets_, nullptr,
                                d_csr_col_indices_, d_csr_values_,
                                CUDA_R_32I, CUDA_R_64F, matrix_type,
                                CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO));
  CUDSS_OK(cudssMatrixCreateDn(&dssB, n_dofs_, 1, n_dofs_, d_r_, CUDA_R_64F,
                               CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_OK(cudssMatrixCreateDn(&dssX, n_dofs_, 1, n_dofs_, d_delta_v_,
                               CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

  if (!analysis_done_) {
    std::cout << "HolisticNewtonSolver: CuDSS analysis..." << std::endl;
    CUDSS_OK(cudssExecute(cudss_handle_, CUDSS_PHASE_ANALYSIS, cudss_config_,
                          cudss_data_, dssA, dssX, dssB));
    analysis_done_ = true;
    factorization_done_ = false;
  }

  const int threads = 256;
  const int vec_blocks = (n_dofs_ + threads - 1) / threads;
  const double armijo_c1 = 1e-4;
  const double armijo_shrink = 0.5;
  const int max_armijo_backtracks = 16;

  auto cublas_ok = [](cublasStatus_t status, const char* label) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << label << " failed with cuBLAS status "
                << static_cast<int>(status) << std::endl;
      return false;
    }
    return true;
  };

  auto copy_device_vector = [&](const double* src, double* dst,
                                const char* label) {
    return cublas_ok(cublasDcopy(cublas_handle_, n_dofs_, src, 1, dst, 1),
                     label);
  };

  auto axpy_host_alpha = [&](double alpha, const double* x, double* y,
                             const char* label) {
    const cublasStatus_t set_host_status =
        cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_HOST);
    const cublasStatus_t axpy_status =
        (set_host_status == CUBLAS_STATUS_SUCCESS)
            ? cublasDaxpy(cublas_handle_, n_dofs_, &alpha, x, 1, y, 1)
            : set_host_status;
    const cublasStatus_t set_device_status =
        cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE);
    return cublas_ok(axpy_status, label) &&
           cublas_ok(set_device_status, "cublasSetPointerMode(device)");
  };

  auto line_search = [&](double norm_g) {
    const double phi0 = 0.5 * norm_g * norm_g;
    if (!copy_device_vector(d_v_guess_, d_v_trial_backup_,
                            "cublasDcopy(holistic line-search backup)")) {
      return false;
    }

    double alpha = 1.0;
    for (int ls_iter = 0; ls_iter < max_armijo_backtracks; ++ls_iter) {
      if (!copy_device_vector(d_v_trial_backup_, d_v_guess_,
                              "cublasDcopy(holistic line-search restore)")) {
        return false;
      }
      if (!axpy_host_alpha(alpha, d_delta_v_, d_v_guess_,
                           "cublasDaxpy(holistic line-search trial step)")) {
        return false;
      }

      EvaluateState();
      const double trial_norm_g = ComputeL2Norm(d_g_, n_dofs_);
      const double phi_trial = 0.5 * trial_norm_g * trial_norm_g;
      const double phi_bound = (1.0 - 2.0 * armijo_c1 * alpha) * phi0;
      if (phi_trial <= phi_bound) {
        if (alpha < 1.0) {
          std::cout << "    Holistic Armijo alpha = " << std::fixed
                    << std::setprecision(3) << alpha << std::endl;
        }
        return true;
      }

      alpha *= armijo_shrink;
    }

    copy_device_vector(d_v_trial_backup_, d_v_guess_,
                       "cublasDcopy(holistic line-search reject restore)");
    EvaluateState();
    return false;
  };

  for (int outer_iter = 0; outer_iter < params_.max_outer; ++outer_iter) {
    std::cout << "Holistic outer iter " << outer_iter << std::endl;

    double norm_g0 = -1.0;
    double last_norm_g = std::numeric_limits<double>::infinity();
    bool inner_converged = false;
    bool line_search_failed = false;

    for (int newton_iter = 0; newton_iter < params_.max_inner; ++newton_iter) {
      EvaluateState();
      const double norm_g = ComputeL2Norm(d_g_, n_dofs_);
      last_norm_g = norm_g;
      std::cout << "  Holistic Newton iter " << newton_iter
                << ": ||g|| = " << std::scientific << norm_g << std::endl;

      if (norm_g0 < 0.0) {
        norm_g0 = norm_g;
      }
      if (norm_g < params_.inner_atol ||
          (params_.inner_rtol > 0.0 && norm_g0 > 0.0 &&
           norm_g <= params_.inner_rtol * norm_g0)) {
        inner_converged = true;
        break;
      }

      initialize_newton_rhs_kernel<<<vec_blocks, threads>>>(n_dofs_, d_g_,
                                                            d_delta_v_, d_r_);
      AssembleHessian();

      const cudssPhase_t factor_phase =
          factorization_done_ ? CUDSS_PHASE_REFACTORIZATION
                              : CUDSS_PHASE_FACTORIZATION;
      CUDSS_OK(cudssExecute(cudss_handle_, factor_phase, cudss_config_,
                            cudss_data_, dssA, dssX, dssB));
      factorization_done_ = true;

      HANDLE_ERROR(cudaMemset(d_delta_v_, 0,
                              static_cast<size_t>(n_dofs_) * sizeof(double)));
      CUDSS_OK(cudssExecute(cudss_handle_, CUDSS_PHASE_SOLVE, cudss_config_,
                            cudss_data_, dssA, dssX, dssB));

      if (params_.enable_line_search) {
        if (!line_search(norm_g)) {
          line_search_failed = true;
          break;
        }
      } else {
        update_velocity_guess_kernel<<<vec_blocks, threads>>>(
            n_dofs_, d_v_guess_, d_delta_v_);
        HANDLE_ERROR(cudaDeviceSynchronize());
      }
    }

    EvaluateState();

    if (n_constraints_ > 0 && inner_converged) {
      const int lambda_blocks = (n_constraints_ + threads - 1) / threads;
      update_dual_variables_kernel<<<lambda_blocks, threads>>>(
          n_constraints_, params_.rho, d_lambda_guess_,
          mixed_constraints_->GetConstraintDevicePtr());
      HANDLE_ERROR(cudaDeviceSynchronize());
    }

    bool constraints_converged = (n_constraints_ == 0);
    if (n_constraints_ > 0) {
      const double norm_c = ComputeL2Norm(
          mixed_constraints_->GetConstraintDevicePtr(), n_constraints_);
      std::cout << "  Holistic outer iter " << outer_iter
                << ": ||c|| = " << std::scientific << norm_c << std::endl;
      constraints_converged = norm_c < params_.outer_tol;
    }

    if (inner_converged && constraints_converged) {
      break;
    }

    if (line_search_failed) {
      std::cout << "  Holistic outer iter " << outer_iter
                << ": stopping because Armijo line search failed" << std::endl;
      break;
    }

    if (!inner_converged) {
      std::cout << "  Holistic outer iter " << outer_iter
                << ": inner Newton did not satisfy tolerance (last ||g|| = "
                << std::scientific << last_norm_g << ")" << std::endl;
    }
  }

  HANDLE_ERROR(cudaMemcpy(d_v_prev_, d_v_guess_,
                          static_cast<size_t>(n_dofs_) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  UpdateProblemVelocity();
  UpdateProblemStateFromVelocity();
  problem_->SyncPositionsToElements();
  problem_->UpdateCollisionNodeBuffer();

  CUDSS_OK(cudssMatrixDestroy(dssA));
  CUDSS_OK(cudssMatrixDestroy(dssB));
  CUDSS_OK(cudssMatrixDestroy(dssX));
  HANDLE_ERROR(cudaDeviceSynchronize());
}
