#include <cooperative_groups.h>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataFunc.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataFunc.cuh"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "SyncedAdamW.cuh"
namespace cg = cooperative_groups;

// Templated solver_grad_L
template <typename ElementType>
__device__ double solver_grad_L(int tid, ElementType *data,
                                SyncedAdamWSolver *d_solver) {
  double res = 0.0;

  const int node_i = tid / 3;
  const int dof_i  = tid % 3;

  const int n_coef    = d_solver->get_n_coef();
  const double inv_dt = 1.0 / d_solver->solver_time_step();

  // Cache pointers once
  const double *__restrict__ v_g    = d_solver->v_guess().data();
  const double *__restrict__ v_p    = d_solver->v_prev().data();
  const int *__restrict__ offsets   = data->csr_offsets();
  const int *__restrict__ columns   = data->csr_columns();
  const double *__restrict__ values = data->csr_values();

  // Mass matrix contribution
  int row_start = offsets[node_i];
  int row_end   = offsets[node_i + 1];

  for (int idx = row_start; idx < row_end; idx++) {
    int node_j     = columns[idx];
    double mass_ij = values[idx];
    int tid_j      = node_j * 3 + dof_i;
    double v_diff  = v_g[tid_j] - v_p[tid_j];
    res += mass_ij * v_diff * inv_dt;
  }

  // Internal force
  res -= (-data->f_int()(tid));

  // External force
  res -= data->f_ext()(tid);

  const int n_constraints = d_solver->gpu_n_constraints();

  if (n_constraints > 0) {
    const double rho_dt =
        *d_solver->solver_rho() * d_solver->solver_time_step();

    const double *__restrict__ lam = d_solver->lambda_guess().data();
    const double *__restrict__ con = data->constraint().data();

    // If you have CSC format (or build it once):
    const int *__restrict__ cjT_offsets   = data->cj_csr_offsets();
    const int *__restrict__ cjT_columns   = data->cj_csr_columns();
    const double *__restrict__ cjT_values = data->cj_csr_values();

    // Get all constraints that affect this DOF (tid)
    const int col_start = cjT_offsets[tid];
    const int col_end   = cjT_offsets[tid + 1];

    for (int idx = col_start; idx < col_end; idx++) {
      const int constraint_idx = cjT_columns[idx];  // Which constraint
      const double constraint_jac_val =
          cjT_values[idx];  // J^T[tid, constraint_idx] = J[constraint_idx, tid]
      const double constraint_val = con[constraint_idx];

      // Add constraint contribution: J^T * (lambda + rho*dt*c)
      res +=
          constraint_jac_val * (lam[constraint_idx] + rho_dt * constraint_val);
    }
  }

  return res;
}

// Templated kernel
template <typename ElementType>
__global__ void one_step_adamw_kernel_impl(ElementType *d_data,
                                           SyncedAdamWSolver *d_adamw_solver) {
  cg::grid_group grid = cg::this_grid();
  int tid             = blockIdx.x * blockDim.x + threadIdx.x;

  // Save previous positions
  if (tid < d_adamw_solver->get_n_coef()) {
    d_adamw_solver->x12_prev()(tid) = d_data->x12()(tid);
    d_adamw_solver->y12_prev()(tid) = d_data->y12()(tid);
    d_adamw_solver->z12_prev()(tid) = d_data->z12()(tid);
  }

  grid.sync();

  if (tid == 0) {
    *d_adamw_solver->inner_flag() = 0;
    *d_adamw_solver->outer_flag() = 0;
  }

  grid.sync();

  for (int outer_iter = 0; outer_iter < d_adamw_solver->solver_max_outer();
       outer_iter++) {
    if (*d_adamw_solver->outer_flag() == 0) {
      // Initialize per-thread variables
      double t   = 1.0;
      double m_t = 0.0;
      double v_t = 0.0;

      double lr           = d_adamw_solver->solver_lr();
      double beta1        = d_adamw_solver->solver_beta1();
      double beta2        = d_adamw_solver->solver_beta2();
      double eps          = d_adamw_solver->solver_eps();
      double weight_decay = d_adamw_solver->solver_weight_decay();
      int conv_check_interval =
          d_adamw_solver->solver_convergence_check_interval();

      if (tid == 0) {
        *d_adamw_solver->norm_g() = 0.0;
      }

      grid.sync();

      if (tid < d_adamw_solver->get_n_coef() * 3) {
        d_adamw_solver->g()(tid) = 0.0;
        t                        = 1.0;
      }

      for (int inner_iter = 0; inner_iter < d_adamw_solver->solver_max_inner();
           inner_iter++) {
        grid.sync();

        if (*d_adamw_solver->inner_flag() == 0) {
          if (tid == 0 && inner_iter % conv_check_interval == 0) {
            printf("outer iter: %d, inner iter: %d\n", outer_iter, inner_iter);
          }

          // Step 1: Compute look-ahead velocity
          double y = 0.0;
          if (tid < d_adamw_solver->get_n_coef() * 3) {
            double g_tid       = d_adamw_solver->g()(tid);
            double v_guess_tid = d_adamw_solver->v_guess()(tid);
            lr                 = lr * 0.995;
            t += 1;

            m_t          = beta1 * m_t + (1 - beta1) * g_tid;
            v_t          = beta2 * v_t + (1 - beta2) * g_tid * g_tid;
            double m_hat = m_t / (1 - pow(beta1, t));
            double v_hat = v_t / (1 - pow(beta2, t));
            y            = v_guess_tid -
                lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * v_guess_tid);

            d_adamw_solver->v_guess()(tid) = y;
          }

          grid.sync();

          // Step 2: Update scratch positions
          if (tid < d_adamw_solver->get_n_coef()) {
            d_data->x12()(tid) = d_adamw_solver->x12_prev()(tid) +
                                 d_adamw_solver->solver_time_step() *
                                     d_adamw_solver->v_guess()(tid * 3 + 0);
            d_data->y12()(tid) = d_adamw_solver->y12_prev()(tid) +
                                 d_adamw_solver->solver_time_step() *
                                     d_adamw_solver->v_guess()(tid * 3 + 1);
            d_data->z12()(tid) = d_adamw_solver->z12_prev()(tid) +
                                 d_adamw_solver->solver_time_step() *
                                     d_adamw_solver->v_guess()(tid * 3 + 2);
          }

          grid.sync();

          // Compute P (stress)
          if (tid <
              d_adamw_solver->get_n_beam() * d_adamw_solver->gpu_n_total_qp()) {
            for (int idx = tid; idx < d_adamw_solver->get_n_beam() *
                                          d_adamw_solver->gpu_n_total_qp();
                 idx += grid.size()) {
              int elem_idx = idx / d_adamw_solver->gpu_n_total_qp();
              int qp_idx   = idx % d_adamw_solver->gpu_n_total_qp();
              compute_p(elem_idx, qp_idx, d_data);
            }
          }

          grid.sync();

          // Clear internal force
          if (tid < d_adamw_solver->get_n_coef() * 3) {
            clear_internal_force(d_data);
          }

          grid.sync();

          // Compute internal force
          if (tid <
              d_adamw_solver->get_n_beam() * d_adamw_solver->gpu_n_shape()) {
            for (int idx = tid; idx < d_adamw_solver->get_n_beam() *
                                          d_adamw_solver->gpu_n_shape();
                 idx += grid.size()) {
              int elem_idx = idx / d_adamw_solver->gpu_n_shape();
              int node_idx = idx % d_adamw_solver->gpu_n_shape();
              compute_internal_force(elem_idx, node_idx, d_data);
            }
          }

          grid.sync();

          // Compute constraints
          if (tid < d_adamw_solver->gpu_n_constraints() / 3) {
            compute_constraint_data(d_data);
          }

          grid.sync();

          // Compute gradient
          if (tid < d_adamw_solver->get_n_coef() * 3) {
            double g = solver_grad_L(tid, d_data, d_adamw_solver);
            d_adamw_solver->g()[tid] = g;
          }

          grid.sync();

          // Check convergence
          if (tid == 0 && inner_iter % conv_check_interval == 0) {
            double norm_g = 0.0;
            for (int i = 0; i < 3 * d_adamw_solver->get_n_coef(); i++) {
              norm_g += d_adamw_solver->g()(i) * d_adamw_solver->g()(i);
            }
            *d_adamw_solver->norm_g() = sqrt(norm_g);

            double norm_v_curr = 0.0;
            for (int i = 0; i < 3 * d_adamw_solver->get_n_coef(); i++) {
              norm_v_curr +=
                  d_adamw_solver->v_guess()(i) * d_adamw_solver->v_guess()(i);
            }
            norm_v_curr = sqrt(norm_v_curr);

            printf("norm_g: %.17f, norm_v_curr: %.17f\n",
                   *d_adamw_solver->norm_g(), norm_v_curr);

            if (*d_adamw_solver->norm_g() <=
                d_adamw_solver->solver_inner_tol() * (1.0 + norm_v_curr)) {
              printf("Converged: gnorm=%.17f <= tol*(1+||v||)=%.17f\n",
                     *d_adamw_solver->norm_g(),
                     d_adamw_solver->solver_inner_tol() * (1.0 + norm_v_curr));
              *d_adamw_solver->inner_flag() = 1;
            }
          }

          grid.sync();
        }
      }

      // Update v_prev
      if (tid < d_adamw_solver->get_n_coef() * 3) {
        d_adamw_solver->v_prev()[tid] = d_adamw_solver->v_guess()[tid];
      }

      grid.sync();

      // Update positions
      if (tid < d_adamw_solver->get_n_coef()) {
        d_data->x12()(tid) = d_adamw_solver->x12_prev()(tid) +
                             d_adamw_solver->v_guess()(tid * 3 + 0) *
                                 d_adamw_solver->solver_time_step();
        d_data->y12()(tid) = d_adamw_solver->y12_prev()(tid) +
                             d_adamw_solver->v_guess()(tid * 3 + 1) *
                                 d_adamw_solver->solver_time_step();
        d_data->z12()(tid) = d_adamw_solver->z12_prev()(tid) +
                             d_adamw_solver->v_guess()(tid * 3 + 2) *
                                 d_adamw_solver->solver_time_step();
      }

      grid.sync();

      // Compute constraints at new position
      if (tid < d_adamw_solver->gpu_n_constraints() / 3) {
        compute_constraint_data(d_data);
      }

      grid.sync();

      int n_constraints = d_adamw_solver->gpu_n_constraints();
      for (int i = tid; i < n_constraints; i += grid.size()) {
        double constraint_val = d_data->constraint()[i];
        d_adamw_solver->lambda_guess()[i] +=
            *d_adamw_solver->solver_rho() * d_adamw_solver->solver_time_step() *
            constraint_val;
      }
      grid.sync();

      unsigned long long t3 = clock64();

      // Dual variable update
      if (tid == 0) {
        double norm_constraint = 0.0;
        for (int i = 0; i < d_adamw_solver->gpu_n_constraints(); i++) {
          double constraint_val = d_data->constraint()[i];
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
  }

  // Final position update
  if (tid < d_adamw_solver->get_n_coef()) {
    d_data->x12()(tid) = d_adamw_solver->x12_prev()(tid) +
                         d_adamw_solver->v_guess()(tid * 3 + 0) *
                             d_adamw_solver->solver_time_step();
    d_data->y12()(tid) = d_adamw_solver->y12_prev()(tid) +
                         d_adamw_solver->v_guess()(tid * 3 + 1) *
                             d_adamw_solver->solver_time_step();
    d_data->z12()(tid) = d_adamw_solver->z12_prev()(tid) +
                         d_adamw_solver->v_guess()(tid * 3 + 2) *
                             d_adamw_solver->solver_time_step();
  }

  grid.sync();
}

// Explicit instantiations
template __global__ void one_step_adamw_kernel_impl<GPU_ANCF3243_Data>(
    GPU_ANCF3243_Data *, SyncedAdamWSolver *);
template __global__ void one_step_adamw_kernel_impl<GPU_ANCF3443_Data>(
    GPU_ANCF3443_Data *, SyncedAdamWSolver *);
template __global__ void one_step_adamw_kernel_impl<GPU_FEAT10_Data>(
    GPU_FEAT10_Data *, SyncedAdamWSolver *);

void SyncedAdamWSolver::OneStepAdamW() {
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  int threads = 128;

  cudaDeviceProp props;
  HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

  int N            = 3 * n_coef_;
  int blocksNeeded = (N + threads - 1) / threads;

  HANDLE_ERROR(cudaEventRecord(start));

  // Launch appropriate templated kernel based on element type
  if (type_ == TYPE_3243) {
    int maxBlocksPerSm = 0;
    HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSm, one_step_adamw_kernel_impl<GPU_ANCF3243_Data>, threads,
        0));
    int blocks =
        std::min(blocksNeeded, maxBlocksPerSm * props.multiProcessorCount);

    void *args[] = {&d_data_, &d_adamw_solver_};
    HANDLE_ERROR(cudaLaunchCooperativeKernel(
        (void *)one_step_adamw_kernel_impl<GPU_ANCF3243_Data>, blocks, threads,
        args));
  } else if (type_ == TYPE_3443) {
    int maxBlocksPerSm = 0;
    HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSm, one_step_adamw_kernel_impl<GPU_ANCF3443_Data>, threads,
        0));
    int blocks =
        std::min(blocksNeeded, maxBlocksPerSm * props.multiProcessorCount);

    void *args[] = {&d_data_, &d_adamw_solver_};
    HANDLE_ERROR(cudaLaunchCooperativeKernel(
        (void *)one_step_adamw_kernel_impl<GPU_ANCF3443_Data>, blocks, threads,
        args));
  } else if (type_ == TYPE_T10) {
    int maxBlocksPerSm = 0;
    HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSm, one_step_adamw_kernel_impl<GPU_FEAT10_Data>, threads,
        0));
    int blocks =
        std::min(blocksNeeded, maxBlocksPerSm * props.multiProcessorCount);

    void *args[] = {&d_data_, &d_adamw_solver_};
    HANDLE_ERROR(cudaLaunchCooperativeKernel(
        (void *)one_step_adamw_kernel_impl<GPU_FEAT10_Data>, blocks, threads,
        args));
  }

  HANDLE_ERROR(cudaEventRecord(stop));
  HANDLE_ERROR(cudaDeviceSynchronize());

  float milliseconds = 0;
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

  std::cout << "OneStepAdamW kernel time: " << milliseconds << " ms"
            << std::endl;

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
}