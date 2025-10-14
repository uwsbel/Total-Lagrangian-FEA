#include <cooperative_groups.h>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataFunc.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataFunc.cuh"
#include "SyncedAdamW.cuh"
namespace cg = cooperative_groups;

__device__ double solver_grad_L(int tid, ElementBase *d_data,
                                SyncedAdamWSolver *d_solver) {
  double res = 0.0;

  int node_i = tid / 3;
  int dof_i  = tid % 3;

  // Mass matrix contribution
  for (int node_j = 0; node_j < d_solver->get_n_coef(); node_j++) {
    double mass_ij = 0.0;
    if (d_data->type == TYPE_3243) {
      auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
      mass_ij    = data->node_values()(node_i, node_j);
    } else if (d_data->type == TYPE_3443) {
      auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
      mass_ij    = data->node_values()(node_i, node_j);
    }

    int tid_j     = node_j * 3 + dof_i;
    double v_diff = d_solver->v_guess()[tid_j] - d_solver->v_prev()[tid_j];
    res += mass_ij * v_diff / d_solver->solver_time_step();
  }

  // Internal force
  if (d_data->type == TYPE_3243) {
    auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
    res -= (-data->f_int()(tid));
  } else if (d_data->type == TYPE_3443) {
    auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
    res -= (-data->f_int()(tid));
  }

  if (d_data->type == TYPE_3243) {
    auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
    res -= data->f_ext()(tid);
  } else if (d_data->type == TYPE_3443) {
    auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
    res -= data->f_ext()(tid);
  }

  // Constraints
  for (int i = 0; i < d_solver->gpu_n_constraints(); i++) {
    double constraint_jac_val = 0.0;
    double constraint_val     = 0.0;

    if (d_data->type == TYPE_3243) {
      auto *data         = static_cast<GPU_ANCF3243_Data *>(d_data);
      constraint_jac_val = data->constraint_jac()(i, tid);
      constraint_val     = data->constraint()[i];
    } else if (d_data->type == TYPE_3443) {
      auto *data         = static_cast<GPU_ANCF3443_Data *>(d_data);
      constraint_jac_val = data->constraint_jac()(i, tid);
      constraint_val     = data->constraint()[i];
    }
    res += constraint_jac_val *
           (d_solver->lambda_guess()[i] + *d_solver->solver_rho() *
                                              d_solver->solver_time_step() *
                                              constraint_val);
  }

  return res;
}

__global__ void one_step_adamw_kernel(ElementBase *d_data,
                                      SyncedAdamWSolver *d_adamw_solver) {
  cg::grid_group grid = cg::this_grid();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // assign x12_prev, y12_prev, z12_prev

  if (tid < d_adamw_solver->get_n_coef()) {
    if (d_data->type == TYPE_3243) {
      auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
      d_adamw_solver->x12_prev()(tid) = data->x12()(tid);
      d_adamw_solver->y12_prev()(tid) = data->y12()(tid);
      d_adamw_solver->z12_prev()(tid) = data->z12()(tid);
    } else if (d_data->type == TYPE_3443) {
      auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
      d_adamw_solver->x12_prev()(tid) = data->x12()(tid);
      d_adamw_solver->y12_prev()(tid) = data->y12()(tid);
      d_adamw_solver->z12_prev()(tid) = data->z12()(tid);
    }
  }

  grid.sync();

  if (tid == 0) {
    *d_adamw_solver->inner_flag() = 0;
    *d_adamw_solver->outer_flag() = 0;
  }

  grid.sync();

  // up here
  // ====================================================================

  for (int outer_iter = 0; outer_iter < d_adamw_solver->solver_max_outer();
       outer_iter++) {
    if (*d_adamw_solver->outer_flag() == 0) {
      // Initialize variables for each thread
      double t   = 1.0;
      double m_t = 0.0;
      double v_t = 0.0;

      double lr           = d_adamw_solver->solver_lr();
      double beta1        = d_adamw_solver->solver_beta1();
      double beta2        = d_adamw_solver->solver_beta2();
      double eps          = d_adamw_solver->solver_eps();
      double weight_decay = d_adamw_solver->solver_weight_decay();

      if (tid == 0) {
        *d_adamw_solver->prev_norm_g() = 0.0;
        *d_adamw_solver->norm_g()      = 0.0;
      }

      grid.sync();

      // Initialize for valid threads only
      if (tid < d_adamw_solver->get_n_coef() * 3) {
        d_adamw_solver->g()(tid) = 0.0;
        t                        = 1.0;
      }

      for (int inner_iter = 0; inner_iter < d_adamw_solver->solver_max_inner();
           inner_iter++) {
        grid.sync();

        if (*d_adamw_solver->inner_flag() == 0) {
          if (tid == 0) {
            printf("outer iter: %d, inner iter: %d\n", outer_iter, inner_iter);
          }

          // Step 1: Each thread computes its look-ahead velocity component
          double y = 0.0;  // Declare y here
          if (tid < d_adamw_solver->get_n_coef() * 3) {
            lr = lr * 0.998;
            t += 1;

            m_t = beta1 * m_t + (1 - beta1) * d_adamw_solver->g()(tid);
            v_t = beta2 * v_t + (1 - beta2) * d_adamw_solver->g()(tid) *
                                    d_adamw_solver->g()(tid);
            double m_hat = m_t / (1 - pow(beta1, t));
            double v_hat = v_t / (1 - pow(beta2, t));
            y            = d_adamw_solver->v_guess()(tid) -
                lr * (m_hat / (sqrt(v_hat) + eps) +
                      weight_decay * d_adamw_solver->v_guess()(tid));

            // Store look-ahead velocity temporarily
            d_adamw_solver->v_guess()(tid) =
                y;  // Use v_guess as temp storage for y
          }

          grid.sync();

          // Step 2: Update scratch positions using current velocities
          if (tid < d_adamw_solver->get_n_coef()) {
            if (d_data->type == TYPE_3243) {
              auto *data       = static_cast<GPU_ANCF3243_Data *>(d_data);
              data->x12()(tid) = d_adamw_solver->x12_prev()(tid) +
                                 d_adamw_solver->solver_time_step() *
                                     d_adamw_solver->v_guess()(tid * 3 + 0);
              data->y12()(tid) = d_adamw_solver->y12_prev()(tid) +
                                 d_adamw_solver->solver_time_step() *
                                     d_adamw_solver->v_guess()(tid * 3 + 1);
              data->z12()(tid) = d_adamw_solver->z12_prev()(tid) +
                                 d_adamw_solver->solver_time_step() *
                                     d_adamw_solver->v_guess()(tid * 3 + 2);
            } else if (d_data->type == TYPE_3443) {
              auto *data       = static_cast<GPU_ANCF3443_Data *>(d_data);
              data->x12()(tid) = d_adamw_solver->x12_prev()(tid) +
                                 d_adamw_solver->solver_time_step() *
                                     d_adamw_solver->v_guess()(tid * 3 + 0);
              data->y12()(tid) = d_adamw_solver->y12_prev()(tid) +
                                 d_adamw_solver->solver_time_step() *
                                     d_adamw_solver->v_guess()(tid * 3 + 1);
              data->z12()(tid) = d_adamw_solver->z12_prev()(tid) +
                                 d_adamw_solver->solver_time_step() *
                                     d_adamw_solver->v_guess()(tid * 3 + 2);
            }
          }

          grid.sync();

          if (tid <
              d_adamw_solver->get_n_beam() * d_adamw_solver->gpu_n_total_qp()) {
            for (int idx = tid; idx < d_adamw_solver->get_n_beam() *
                                          d_adamw_solver->gpu_n_total_qp();
                 idx += grid.size()) {
              int elem_idx = idx / d_adamw_solver->gpu_n_total_qp();
              int qp_idx   = idx % d_adamw_solver->gpu_n_total_qp();
              if (d_data->type == TYPE_3243) {
                ancf3243_compute_p(elem_idx, qp_idx,
                                   static_cast<GPU_ANCF3243_Data *>(d_data));
              } else if (d_data->type == TYPE_3443) {
                ancf3443_compute_p(elem_idx, qp_idx,
                                   static_cast<GPU_ANCF3443_Data *>(d_data));
              }
            }
          }

          grid.sync();

          if (tid <
              d_adamw_solver->get_n_beam() * d_adamw_solver->gpu_n_shape()) {
            for (int idx = tid; idx < d_adamw_solver->get_n_beam() *
                                          d_adamw_solver->gpu_n_shape();
                 idx += grid.size()) {
              int elem_idx = idx / d_adamw_solver->gpu_n_shape();
              int node_idx = idx % d_adamw_solver->gpu_n_shape();
              if (d_data->type == TYPE_3243) {
                ancf3243_compute_internal_force(
                    elem_idx, node_idx,
                    static_cast<GPU_ANCF3243_Data *>(d_data));
              } else if (d_data->type == TYPE_3443) {
                ancf3443_compute_internal_force(
                    elem_idx, node_idx,
                    static_cast<GPU_ANCF3443_Data *>(d_data));
              }
            }
          }

          grid.sync();

          if (tid < d_adamw_solver->gpu_n_constraints() / 3) {
            if (d_data->type == TYPE_3243)
              ancf3243_compute_constraint_data(
                  static_cast<GPU_ANCF3243_Data *>(d_data));
            else if (d_data->type == TYPE_3443)
              ancf3443_compute_constraint_data(
                  static_cast<GPU_ANCF3443_Data *>(d_data));
          }

          grid.sync();

          if (tid < d_adamw_solver->get_n_coef() * 3) {
            double g = solver_grad_L(tid, d_data, d_adamw_solver);
            d_adamw_solver->g()[tid] = g;
          }

          grid.sync();

          if (tid == 0) {
            // calculate norm of g
            double norm_g = 0.0;
            for (int i = 0; i < 3 * d_adamw_solver->get_n_coef(); i++) {
              norm_g += d_adamw_solver->g()(i) * d_adamw_solver->g()(i);
            }
            *d_adamw_solver->norm_g() = sqrt(norm_g);

            // calculate norm of current velocity
            double norm_v_curr = 0.0;
            for (int i = 0; i < 3 * d_adamw_solver->get_n_coef(); i++) {
              norm_v_curr +=
                  d_adamw_solver->v_guess()(i) * d_adamw_solver->v_guess()(i);
            }
            norm_v_curr = sqrt(norm_v_curr);

            printf("norm_g: %.17f, norm_v_curr: %.17f\n",
                   *d_adamw_solver->norm_g(), norm_v_curr);

            // Use the same convergence criterion as Python AdamW
            if (*d_adamw_solver->norm_g() <=
                d_adamw_solver->solver_inner_tol() * (1.0 + norm_v_curr)) {
              printf("Converged: gnorm=%.17f <= tol*(1+||v||)=%.17f\n",
                     *d_adamw_solver->norm_g(),
                     d_adamw_solver->solver_inner_tol() * (1.0 + norm_v_curr));
              *d_adamw_solver->inner_flag() = 1;
            }
          }

          grid.sync();

          if (tid == 0) {
            *d_adamw_solver->prev_norm_g() = *d_adamw_solver->norm_g();
          }

          grid.sync();
        }
      }

      // After inner loop convergence, update v_prev for next outer iteration
      if (tid < d_adamw_solver->get_n_coef() * 3) {
        d_adamw_solver->v_prev()[tid] = d_adamw_solver->v_guess()[tid];
      }

      grid.sync();

      // Update positions: q_new = q_prev + h * v (parallel across threads)
      if (tid < d_adamw_solver->get_n_coef()) {
        if (d_data->type == TYPE_3243) {
          auto *data       = static_cast<GPU_ANCF3243_Data *>(d_data);
          data->x12()(tid) = d_adamw_solver->x12_prev()(tid) +
                             d_adamw_solver->v_guess()(tid * 3 + 0) *
                                 d_adamw_solver->solver_time_step();
          data->y12()(tid) = d_adamw_solver->y12_prev()(tid) +
                             d_adamw_solver->v_guess()(tid * 3 + 1) *
                                 d_adamw_solver->solver_time_step();
          data->z12()(tid) = d_adamw_solver->z12_prev()(tid) +
                             d_adamw_solver->v_guess()(tid * 3 + 2) *
                                 d_adamw_solver->solver_time_step();
        } else if (d_data->type == TYPE_3443) {
          auto *data       = static_cast<GPU_ANCF3443_Data *>(d_data);
          data->x12()(tid) = d_adamw_solver->x12_prev()(tid) +
                             d_adamw_solver->v_guess()(tid * 3 + 0) *
                                 d_adamw_solver->solver_time_step();
          data->y12()(tid) = d_adamw_solver->y12_prev()(tid) +
                             d_adamw_solver->v_guess()(tid * 3 + 1) *
                                 d_adamw_solver->solver_time_step();
          data->z12()(tid) = d_adamw_solver->z12_prev()(tid) +
                             d_adamw_solver->v_guess()(tid * 3 + 2) *
                                 d_adamw_solver->solver_time_step();
        }
      }

      grid.sync();

      // Only thread 0 handles constraint computation and dual variable updates

      if (tid < d_adamw_solver->gpu_n_constraints() / 3) {
        // Compute constraints at new position
        if (d_data->type == TYPE_3243) {
          ancf3243_compute_constraint_data(
              static_cast<GPU_ANCF3243_Data *>(d_data));
        } else if (d_data->type == TYPE_3443) {
          ancf3443_compute_constraint_data(
              static_cast<GPU_ANCF3443_Data *>(d_data));
        }
      }

      if (tid == 0) {
        // Dual variable update: lam += rho * h * c(q_new)
        for (int i = 0; i < d_adamw_solver->gpu_n_constraints(); i++) {
          double constraint_val = 0.0;
          if (d_data->type == TYPE_3243) {
            auto *data     = static_cast<GPU_ANCF3243_Data *>(d_data);
            constraint_val = data->constraint()[i];
          } else if (d_data->type == TYPE_3443) {
            auto *data     = static_cast<GPU_ANCF3443_Data *>(d_data);
            constraint_val = data->constraint()[i];
          }
          d_adamw_solver->lambda_guess()[i] +=
              *d_adamw_solver->solver_rho() *
              d_adamw_solver->solver_time_step() * constraint_val;
        }

        // Termination on the norm of constraint < outer_tol
        double norm_constraint = 0.0;
        for (int i = 0; i < d_adamw_solver->gpu_n_constraints(); i++) {
          double constraint_val = 0.0;
          if (d_data->type == TYPE_3243) {
            auto *data     = static_cast<GPU_ANCF3243_Data *>(d_data);
            constraint_val = data->constraint()[i];
          } else if (d_data->type == TYPE_3443) {
            auto *data     = static_cast<GPU_ANCF3443_Data *>(d_data);
            constraint_val = data->constraint()[i];
          }
          norm_constraint += constraint_val * constraint_val;
        }
        norm_constraint = sqrt(norm_constraint);
        printf("norm_constraint: %.17f\n", norm_constraint);

        if (norm_constraint < d_adamw_solver->solver_outer_tol()) {
          printf("Converged constraint: %.17f\n", norm_constraint);
          *d_adamw_solver->outer_flag() = 1;
        }
      }

      grid.sync();
    }
    grid.sync();
  }

  // finally write data back to x12, y12, z12
  // explicit integration
  if (tid < d_adamw_solver->get_n_coef()) {
    if (d_data->type == TYPE_3243) {
      auto *data       = static_cast<GPU_ANCF3243_Data *>(d_data);
      data->x12()(tid) = d_adamw_solver->x12_prev()(tid) +
                         d_adamw_solver->v_guess()(tid * 3 + 0) *
                             d_adamw_solver->solver_time_step();
      data->y12()(tid) = d_adamw_solver->y12_prev()(tid) +
                         d_adamw_solver->v_guess()(tid * 3 + 1) *
                             d_adamw_solver->solver_time_step();
      data->z12()(tid) = d_adamw_solver->z12_prev()(tid) +
                         d_adamw_solver->v_guess()(tid * 3 + 2) *
                             d_adamw_solver->solver_time_step();
    } else if (d_data->type == TYPE_3443) {
      auto *data       = static_cast<GPU_ANCF3443_Data *>(d_data);
      data->x12()(tid) = d_adamw_solver->x12_prev()(tid) +
                         d_adamw_solver->v_guess()(tid * 3 + 0) *
                             d_adamw_solver->solver_time_step();
      data->y12()(tid) = d_adamw_solver->y12_prev()(tid) +
                         d_adamw_solver->v_guess()(tid * 3 + 1) *
                             d_adamw_solver->solver_time_step();
      data->z12()(tid) = d_adamw_solver->z12_prev()(tid) +
                         d_adamw_solver->v_guess()(tid * 3 + 2) *
                             d_adamw_solver->solver_time_step();
    }
  }

  grid.sync();
}

void SyncedAdamWSolver::OneStepAdamW() {
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  int threads = 128;

  cudaDeviceProp props;
  HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

  int maxBlocksPerSm = 0;
  HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocksPerSm, one_step_adamw_kernel, threads, 0));
  int maxCoopBlocks = maxBlocksPerSm * props.multiProcessorCount;

  int N            = 3 * n_coef_;
  int blocksNeeded = (N + threads - 1) / threads;
  int blocks       = std::min(blocksNeeded, maxCoopBlocks);

  void *args[] = {&d_data_, &d_adamw_solver_};

  HANDLE_ERROR(cudaEventRecord(start));
  HANDLE_ERROR(cudaLaunchCooperativeKernel((void *)one_step_adamw_kernel,
                                           blocks, threads, args));
  HANDLE_ERROR(cudaEventRecord(stop));

  HANDLE_ERROR(cudaDeviceSynchronize());

  float milliseconds = 0;
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

  std::cout << "OneStepAdamW kernel time: " << milliseconds << " ms"
            << std::endl;

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
}