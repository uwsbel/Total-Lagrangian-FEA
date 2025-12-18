/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    SyncedAdamWNocoop.cu
 * Brief:   Implements the GPU-synchronized AdamW optimizer (non-cooperative
 *          variant) used to advance generalized velocities and positions in
 *          RoboDyna. Defines kernels that evaluate element residuals,
 *          constraint contributions, and apply AdamW updates for ANCF3243,
 *          ANCF3443, and FEAT10 element data without cooperative groups.
 *==============================================================
 *==============================================================*/

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "SyncedAdamWNocoop.cuh"

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataFunc.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataFunc.cuh"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"

double SyncedAdamWNocoopSolver::compute_l2_norm_cublas(double *d_vec,
                                                       int n_dofs) {
  cublasDnrm2(cublas_handle_, n_dofs, d_vec, 1, d_norm_temp_cublas_);

  double h_norm = 0.0;
  HANDLE_ERROR(cudaMemcpy(&h_norm, d_norm_temp_cublas_, sizeof(double),
                          cudaMemcpyDeviceToHost));
  return h_norm;
}

template <typename ElementType>
__device__ double solver_grad_L_nocoop(int tid, ElementType *data,
                                       SyncedAdamWNocoopSolver *d_solver) {
  double res = 0.0;

  const int node_i = tid / 3;
  const int dof_i  = tid % 3;

  const double inv_dt = 1.0 / d_solver->solver_time_step();
  const double dt     = d_solver->solver_time_step();

  const double *__restrict__ v_g    = d_solver->v_guess().data();
  const double *__restrict__ v_p    = d_solver->v_prev().data();
  const int *__restrict__ offsets   = data->csr_offsets();
  const int *__restrict__ columns   = data->csr_columns();
  const double *__restrict__ values = data->csr_values();

  int row_start = offsets[node_i];
  int row_end   = offsets[node_i + 1];

  for (int idx = row_start; idx < row_end; idx++) {
    int node_j     = columns[idx];
    double mass_ij = values[idx];
    int tid_j      = node_j * 3 + dof_i;
    double v_diff  = v_g[tid_j] - v_p[tid_j];
    res += mass_ij * v_diff * inv_dt;
  }

  res -= (-data->f_int()(tid));
  res -= data->f_ext()(tid);

  const int n_constraints = d_solver->gpu_n_constraints();
  if (n_constraints > 0) {
    const double rho = *d_solver->solver_rho();

    const double *__restrict__ lam = d_solver->lambda_guess().data();
    const double *__restrict__ con = data->constraint().data();

    const int *__restrict__ cjT_offsets   = data->cj_csr_offsets();
    const int *__restrict__ cjT_columns   = data->cj_csr_columns();
    const double *__restrict__ cjT_values = data->cj_csr_values();

    const int col_start = cjT_offsets[tid];
    const int col_end   = cjT_offsets[tid + 1];

    for (int idx = col_start; idx < col_end; idx++) {
      const int constraint_idx = cjT_columns[idx];
      const double constraint_jac_val = cjT_values[idx];
      const double constraint_val     = con[constraint_idx];

      res += dt * constraint_jac_val *
             (lam[constraint_idx] + rho * constraint_val);
    }
  }

  return res;
}

template <typename ElementType>
__global__ void adamw_save_prev_pos_kernel(ElementType *d_data,
                                          SyncedAdamWNocoopSolver *d_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_solver->get_n_coef()) {
    d_solver->x12_prev()(tid) = d_data->x12()(tid);
    d_solver->y12_prev()(tid) = d_data->y12()(tid);
    d_solver->z12_prev()(tid) = d_data->z12()(tid);
  }
}

__global__ void adamw_init_flags_kernel(SyncedAdamWNocoopSolver *d_solver) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *d_solver->inner_flag() = 0;
    *d_solver->outer_flag() = 0;
    *d_solver->norm_g()     = 0.0;
  }
}

__global__ void adamw_reset_inner_flag_kernel(SyncedAdamWNocoopSolver *d_solver) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *d_solver->inner_flag() = 0;
    *d_solver->norm_g()     = 0.0;
  }
}

__global__ void adamw_zero_g_kernel(SyncedAdamWNocoopSolver *d_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n   = d_solver->get_n_coef() * 3;
  if (tid < n) {
    d_solver->g()(tid) = 0.0;
  }
}

__global__ void adamw_zero_m_v_kernel(SyncedAdamWNocoopSolver *d_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n   = d_solver->get_n_coef() * 3;
  if (tid < n) {
    d_solver->m()(tid)      = 0.0;
    d_solver->v_adam()(tid) = 0.0;
  }
}

__global__ void adamw_update_velocity_kernel(SyncedAdamWNocoopSolver *d_solver,
                                            double lr_current,
                                            double inv_1mb1t,
                                            double inv_1mb2t) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n   = d_solver->get_n_coef() * 3;
  if (tid >= n) {
    return;
  }

  const double beta1        = d_solver->solver_beta1();
  const double beta2        = d_solver->solver_beta2();
  const double eps          = d_solver->solver_eps();
  const double weight_decay = d_solver->solver_weight_decay();

  double g_tid       = d_solver->g()(tid);
  double v_guess_tid = d_solver->v_guess()(tid);

  double m_t = d_solver->m()(tid);
  double v_t = d_solver->v_adam()(tid);

  m_t = beta1 * m_t + (1.0 - beta1) * g_tid;
  v_t = beta2 * v_t + (1.0 - beta2) * g_tid * g_tid;

  double m_hat = m_t * inv_1mb1t;
  double v_hat = v_t * inv_1mb2t;

  double y = v_guess_tid -
             lr_current *
                 (m_hat / (sqrt(v_hat) + eps) + weight_decay * v_guess_tid);

  d_solver->m()(tid)      = m_t;
  d_solver->v_adam()(tid) = v_t;
  d_solver->v_guess()(tid) = y;
}

template <typename ElementType>
__global__ void adamw_update_positions_from_prev_kernel(
    ElementType *d_data, SyncedAdamWNocoopSolver *d_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_solver->get_n_coef()) {
    double dt = d_solver->solver_time_step();
    d_data->x12()(tid) = d_solver->x12_prev()(tid) +
                         dt * d_solver->v_guess()(tid * 3 + 0);
    d_data->y12()(tid) = d_solver->y12_prev()(tid) +
                         dt * d_solver->v_guess()(tid * 3 + 1);
    d_data->z12()(tid) = d_solver->z12_prev()(tid) +
                         dt * d_solver->v_guess()(tid * 3 + 2);
  }
}

template <typename ElementType>
__global__ void adamw_compute_p_kernel(ElementType *d_data,
                                      SyncedAdamWNocoopSolver *d_solver) {
  int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int total = d_solver->get_n_beam() * d_solver->gpu_n_total_qp();

  for (int idx = tid; idx < total; idx += stride) {
    int elem_idx = idx / d_solver->gpu_n_total_qp();
    int qp_idx   = idx % d_solver->gpu_n_total_qp();
    compute_p(elem_idx, qp_idx, d_data, d_solver->v_guess().data(),
              d_solver->solver_time_step());
  }
}

template <typename ElementType>
__global__ void adamw_clear_internal_force_kernel(ElementType *d_data) {
  clear_internal_force(d_data);
}

template <typename ElementType>
__global__ void adamw_compute_internal_force_kernel(
    ElementType *d_data, SyncedAdamWNocoopSolver *d_solver) {
  int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int total = d_solver->get_n_beam() * d_solver->gpu_n_shape();

  for (int idx = tid; idx < total; idx += stride) {
    int elem_idx = idx / d_solver->gpu_n_shape();
    int node_idx = idx % d_solver->gpu_n_shape();
    compute_internal_force(elem_idx, node_idx, d_data);
  }
}

template <typename ElementType>
__global__ void adamw_compute_constraints_kernel(ElementType *d_data) {
  compute_constraint_data(d_data);
}

template <typename ElementType>
__global__ void adamw_compute_gradient_kernel(ElementType *d_data,
                                             SyncedAdamWNocoopSolver *d_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n   = d_solver->get_n_coef() * 3;
  if (tid < n) {
    double g = solver_grad_L_nocoop(tid, d_data, d_solver);
    d_solver->g()(tid) = g;
  }
}

__global__ void adamw_update_v_prev_kernel(SyncedAdamWNocoopSolver *d_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n   = d_solver->get_n_coef() * 3;
  if (tid < n) {
    d_solver->v_prev()(tid) = d_solver->v_guess()(tid);
  }
}

template <typename ElementType>
__global__ void adamw_dual_update_kernel(ElementType *d_data,
                                        SyncedAdamWNocoopSolver *d_solver) {
  int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int n_constraints = d_solver->gpu_n_constraints();
  double dt         = d_solver->solver_time_step();
  double rho        = *d_solver->solver_rho();

  for (int i = tid; i < n_constraints; i += stride) {
    double constraint_val = d_data->constraint()[i];
    d_solver->lambda_guess()[i] += rho * dt * constraint_val;
  }
}

void SyncedAdamWNocoopSolver::OneStepAdamWNocoop() {
  constexpr int threadsPerBlock = 256;

  const int n_coef = n_coef_;
  const int n_dof  = n_coef_ * 3;

  const int blocks_coef = (n_coef + threadsPerBlock - 1) / threadsPerBlock;
  const int blocks_dof  = (n_dof + threadsPerBlock - 1) / threadsPerBlock;

  int blocks_p =
      (n_beam_ * n_total_qp_ + threadsPerBlock - 1) / threadsPerBlock;
  int blocks_if =
      (n_beam_ * n_shape_ + threadsPerBlock - 1) / threadsPerBlock;
  const int blocks_constraints =
      ((n_constraints_ > 0 ? n_constraints_ : 1) + threadsPerBlock - 1) /
      threadsPerBlock;
  const int blocks_fixed_nodes =
      (((n_constraints_ / 3) > 0 ? (n_constraints_ / 3) : 1) + threadsPerBlock -
       1) /
      threadsPerBlock;

  cudaDeviceProp props;
  HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));
  const int max_grid_stride_blocks = 32 * props.multiProcessorCount;
  blocks_p = std::max(1, std::min(blocks_p, max_grid_stride_blocks));
  blocks_if = std::max(1, std::min(blocks_if, max_grid_stride_blocks));

  double lr0 = 0.0, lr_decay = 1.0;
  double inner_tol = 0.0, outer_tol = 0.0;
  double inner_rtol = 0.0;
  double beta1_h = 0.0, beta2_h = 0.0;
  int max_outer = 0, max_inner = 0, conv_check_interval = 1;

  HANDLE_ERROR(cudaMemcpy(&lr0, d_lr_, sizeof(double), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&lr_decay, d_lr_decay_, sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(&beta1_h, d_beta1_, sizeof(double), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(&beta2_h, d_beta2_, sizeof(double), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&inner_tol, d_inner_tol_, sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&inner_rtol, d_inner_rtol_, sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&outer_tol, d_outer_tol_, sizeof(double),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&max_outer, d_max_outer_, sizeof(int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&max_inner, d_max_inner_, sizeof(int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&conv_check_interval, d_convergence_check_interval_,
                          sizeof(int), cudaMemcpyDeviceToHost));

  if (conv_check_interval <= 0) {
    conv_check_interval = 1;
  }

  int effective_max_outer = max_outer;
  if (n_constraints_ == 0) {
    effective_max_outer = 1;
  }

  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start));

  auto one_step_typed = [&](auto *typed_data) {
    adamw_save_prev_pos_kernel<<<blocks_coef, threadsPerBlock>>>(typed_data,
                                                                 d_adamw_solver_);
    adamw_init_flags_kernel<<<1, 1>>>(d_adamw_solver_);

    int outer_flag_h = 0;
    for (int outer_iter = 0; outer_iter < effective_max_outer; outer_iter++) {
      if (outer_flag_h != 0) {
        break;
      }

      adamw_reset_inner_flag_kernel<<<1, 1>>>(d_adamw_solver_);
      HANDLE_ERROR(cudaMemsetAsync(d_g_, 0, n_dof * sizeof(double)));
      HANDLE_ERROR(cudaMemsetAsync(d_m_, 0, n_dof * sizeof(double)));
      HANDLE_ERROR(cudaMemsetAsync(d_v_adam_, 0, n_dof * sizeof(double)));

      int inner_flag_h = 0;
      double norm_g0   = -1.0;
      int last_inner_iter = -1;
      for (int inner_iter = 0; inner_iter < max_inner; inner_iter++) {
        if (inner_flag_h != 0) {
          break;
        }

        last_inner_iter = inner_iter;

        double lr_current = lr0 * pow(lr_decay, inner_iter + 1);
        double t_current  = static_cast<double>(inner_iter + 2);
        double inv_1mb1t  = 1.0 / (1.0 - pow(beta1_h, t_current));
        double inv_1mb2t  = 1.0 / (1.0 - pow(beta2_h, t_current));

        adamw_update_velocity_kernel<<<blocks_dof, threadsPerBlock>>>(
            d_adamw_solver_, lr_current, inv_1mb1t, inv_1mb2t);

        adamw_update_positions_from_prev_kernel<<<blocks_coef, threadsPerBlock>>>(
            typed_data, d_adamw_solver_);

        adamw_compute_p_kernel<<<blocks_p, threadsPerBlock>>>(typed_data,
                                                              d_adamw_solver_);

        adamw_clear_internal_force_kernel<<<blocks_dof, threadsPerBlock>>>(
            typed_data);

        adamw_compute_internal_force_kernel<<<blocks_if, threadsPerBlock>>>(
            typed_data, d_adamw_solver_);

        if (n_constraints_ > 0) {
          adamw_compute_constraints_kernel<<<blocks_fixed_nodes, threadsPerBlock>>>(
              typed_data);
        }

        adamw_compute_gradient_kernel<<<blocks_dof, threadsPerBlock>>>(
            typed_data, d_adamw_solver_);

        if (inner_iter % conv_check_interval == 0) {
          double norm_g = compute_l2_norm_cublas(d_g_, n_dof);
          double norm_v_curr = compute_l2_norm_cublas(d_v_guess_, n_dof);

          if (norm_g0 < 0.0) {
            norm_g0 = norm_g;
          }

          const double tol_abs  = inner_tol * (1.0 + norm_v_curr);
          const double tol_rel  = (inner_rtol > 0.0 && norm_g0 > 0.0)
                                     ? (inner_rtol * norm_g0)
                                     : 0.0;
          const double g_ratio  = (norm_g0 > 0.0) ? (norm_g / norm_g0) : 0.0;

          const bool abs_ok = (norm_g <= tol_abs);
          const bool rel_ok = (tol_rel > 0.0 && norm_g <= tol_rel);

          std::cout << "outer iter: " << outer_iter
                    << ", inner iter: " << inner_iter << std::endl;
          std::cout << "lr: " << std::scientific << std::setprecision(6)
                    << lr_current << std::fixed << std::setprecision(17)
                    << ", norm_g: " << norm_g
                    << ", norm_v: " << norm_v_curr
                    << ", atol*(1+||v||): " << tol_abs
                    << ", rtol*g0: " << tol_rel
                    << ", g/g0: " << g_ratio
                    << std::endl;

          if (abs_ok || rel_ok) {
            std::cout << "Converged: gnorm=" << std::fixed
                      << std::setprecision(17) << norm_g
                      << " <= max(tol_abs, rtol*g0)="
                      << std::max(tol_abs, tol_rel)
                      << std::endl;
            inner_flag_h = 1;
            HANDLE_ERROR(cudaMemcpy(d_inner_flag_, &inner_flag_h, sizeof(int),
                                    cudaMemcpyHostToDevice));
          }
        }
      }

      if (last_inner_iter >= 0) {
        if (inner_flag_h != 0) {
          std::cout << "Outer iter " << outer_iter
                    << ": inner loop converged in " << (last_inner_iter + 1)
                    << " iterations" << std::endl;
        } else {
          std::cout << "Outer iter " << outer_iter
                    << ": inner loop hit max_inner=" << max_inner
                    << std::endl;
        }
      }

      adamw_update_v_prev_kernel<<<blocks_dof, threadsPerBlock>>>(d_adamw_solver_);

      adamw_update_positions_from_prev_kernel<<<blocks_coef, threadsPerBlock>>>(
          typed_data, d_adamw_solver_);

      if (n_constraints_ > 0) {
        adamw_compute_constraints_kernel<<<blocks_fixed_nodes, threadsPerBlock>>>(
            typed_data);

        adamw_dual_update_kernel<<<blocks_constraints, threadsPerBlock>>>(
            typed_data, d_adamw_solver_);

        if (d_constraint_ptr_ == nullptr) {
          std::cerr << "SyncedAdamWNocoop: constraint pointer is null but "
                       "n_constraints_ > 0. Constraint setup is required for "
                       "outer convergence check."
                    << std::endl;
          std::abort();
        }

        double norm_constraint =
            compute_l2_norm_cublas(d_constraint_ptr_, n_constraints_);
        std::cout << "norm_constraint: " << std::fixed << std::setprecision(17)
                  << norm_constraint << std::endl;

        if (norm_constraint < outer_tol) {
          if (inner_flag_h != 0) {
            std::cout << "Converged constraint: " << std::fixed
                      << std::setprecision(17) << norm_constraint << std::endl;
            outer_flag_h = 1;
            HANDLE_ERROR(cudaMemcpy(d_outer_flag_, &outer_flag_h, sizeof(int),
                                    cudaMemcpyHostToDevice));
          } else {
            std::cout << "Constraint below tol, but inner did not converge"
                      << std::endl;
          }
        }
      }
    }

    adamw_update_positions_from_prev_kernel<<<blocks_coef, threadsPerBlock>>>(
        typed_data, d_adamw_solver_);
  };

  if (type_ == TYPE_T10) {
    one_step_typed(static_cast<GPU_FEAT10_Data *>(d_data_));
  } else if (type_ == TYPE_3243) {
    one_step_typed(static_cast<GPU_ANCF3243_Data *>(d_data_));
  } else if (type_ == TYPE_3443) {
    one_step_typed(static_cast<GPU_ANCF3443_Data *>(d_data_));
  }

  HANDLE_ERROR(cudaEventRecord(stop));

  HANDLE_ERROR(cudaDeviceSynchronize());

  float milliseconds = 0;
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "OneStepAdamWNocoop kernel time: " << milliseconds << " ms"
            << std::endl;

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
}
