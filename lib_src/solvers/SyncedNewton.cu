#include <cooperative_groups.h>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataFunc.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataFunc.cuh"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "SyncedNewton.cuh"

namespace cg = cooperative_groups;

template <typename ElementType>
__device__ double solver_grad_L(int tid, ElementType *data,
                                SyncedNewtonSolver *d_solver) {
  double res = 0.0;

  const int node_i = tid / 3;
  const int dof_i  = tid % 3;

  const double inv_dt = 1.0 / d_solver->solver_time_step();
  const double dt     = d_solver->solver_time_step();

  // Cache pointers once
  const double *__restrict__ v_g    = d_solver->v_guess().data();
  const double *__restrict__ v_p    = d_solver->v_prev().data();
  const int *__restrict__ offsets   = data->csr_offsets();
  const int *__restrict__ columns   = data->csr_columns();
  const double *__restrict__ values = data->csr_values();

  // Mass matrix contribution: (M @ (v_loc - v_prev)) / h
  int row_start = offsets[node_i];
  int row_end   = offsets[node_i + 1];

  for (int idx = row_start; idx < row_end; idx++) {
    int node_j     = columns[idx];
    double mass_ij = values[idx];
    int tid_j      = node_j * 3 + dof_i;
    double v_diff  = v_g[tid_j] - v_p[tid_j];
    res += mass_ij * v_diff * inv_dt;
  }

  // Mechanical force contribution: - (-f_int + f_ext) = f_int - f_ext
  res -= (-data->f_int()(tid));  // Add f_int
  res -= data->f_ext()(tid);     // Subtract f_ext

  const int n_constraints = d_solver->gpu_n_constraints();

  if (n_constraints > 0) {
    // Python: h * (J.T @ (lam_mult + rho_bb * cA))
    const double rho = *d_solver->solver_rho();

    const double *__restrict__ lam = d_solver->lambda_guess().data();
    const double *__restrict__ con = data->constraint().data();

    // CSR format stores J^T (transpose of constraint Jacobian)
    const int *__restrict__ cjT_offsets   = data->cj_csr_offsets();
    const int *__restrict__ cjT_columns   = data->cj_csr_columns();
    const double *__restrict__ cjT_values = data->cj_csr_values();

    // Get all constraints that affect this DOF (tid)
    const int col_start = cjT_offsets[tid];
    const int col_end   = cjT_offsets[tid + 1];

    for (int idx = col_start; idx < col_end; idx++) {
      const int constraint_idx        = cjT_columns[idx];
      const double constraint_jac_val = cjT_values[idx];
      const double constraint_val     = con[constraint_idx];

      // Add constraint contribution: h * J^T * (lambda + rho*c)
      res += dt * constraint_jac_val *
             (lam[constraint_idx] + rho * constraint_val);
    }
  }

  return res;
}

// Templated Newton kernel - STARTER CODE (implement Newton method here)
template <typename ElementType>
__global__ void one_step_newton_kernel_impl(
    ElementType *d_data, SyncedNewtonSolver *d_newton_solver) {
  cg::grid_group grid = cg::this_grid();
  int tid             = blockIdx.x * blockDim.x + threadIdx.x;

  // Save previous positions
  if (tid < d_newton_solver->get_n_coef()) {
    d_newton_solver->x12_prev()(tid) = d_data->x12()(tid);
    d_newton_solver->y12_prev()(tid) = d_data->y12()(tid);
    d_newton_solver->z12_prev()(tid) = d_data->z12()(tid);
  }

  grid.sync();

  if (tid == 0) {
    *d_newton_solver->inner_flag() = 0;
    *d_newton_solver->outer_flag() = 0;
  }

  grid.sync();

  for (int outer_iter = 0; outer_iter < d_newton_solver->solver_max_outer();
       outer_iter++) {
    if (*d_newton_solver->outer_flag() == 0) {
      // Initialize per-thread variables

      if (tid == 0) {
        *d_newton_solver->norm_g() = 0.0;
      }

      grid.sync();

      if (tid < d_newton_solver->get_n_coef() * 3) {
        d_newton_solver->g()(tid) = 0.0;
      }

      grid.sync();
      // first run to calculate initial gradient
      // Compute P (stress)
      if (tid <
          d_newton_solver->get_n_beam() * d_newton_solver->gpu_n_total_qp()) {
        for (int idx = tid; idx < d_newton_solver->get_n_beam() *
                                      d_newton_solver->gpu_n_total_qp();
             idx += grid.size()) {
          int elem_idx = idx / d_newton_solver->gpu_n_total_qp();
          int qp_idx   = idx % d_newton_solver->gpu_n_total_qp();
          compute_p(elem_idx, qp_idx, d_data);
        }
      }

      grid.sync();

      // Clear internal force
      if (tid < d_newton_solver->get_n_coef() * 3) {
        clear_internal_force(d_data);
      }

      grid.sync();

      // Compute internal force
      if (tid <
          d_newton_solver->get_n_beam() * d_newton_solver->gpu_n_shape()) {
        for (int idx = tid; idx < d_newton_solver->get_n_beam() *
                                      d_newton_solver->gpu_n_shape();
             idx += grid.size()) {
          int elem_idx = idx / d_newton_solver->gpu_n_shape();
          int node_idx = idx % d_newton_solver->gpu_n_shape();
          compute_internal_force(elem_idx, node_idx, d_data);
        }
      }

      grid.sync();

      // Compute constraints
      if (tid < d_newton_solver->gpu_n_constraints() / 3) {
        compute_constraint_data(d_data);
      }

      grid.sync();

      // Compute gradient
      if (tid < d_newton_solver->get_n_coef() * 3) {
        double g                  = solver_grad_L(tid, d_data, d_newton_solver);
        d_newton_solver->g()[tid] = g;
      }

      grid.sync();

      // initialize delta_v (delta_v = 0), r = -g, p = r
      if (tid < d_newton_solver->get_n_coef() * 3) {
        d_newton_solver->delta_v()[tid] = 0.0;
        d_newton_solver->r()[tid]       = -d_newton_solver->g()[tid];
        d_newton_solver->p()[tid]       = d_newton_solver->r()[tid];
      }

      grid.sync();

      // this is newton inner loop
      // TODO: implement CG based hessian storage free here
      for (int inner_iter = 0; inner_iter < d_newton_solver->solver_max_inner();
           inner_iter++) {
        grid.sync();

        // KEY STEP: Matrix free hessian vector multiplication
        // Hp = H * p

        // compute optimal step size alpha
        // r_dot_r = dot(r,r)
        // p_dot_Hp = dot(p, Hp)
        // alpha = r_dot_r / p_dot_Hp

        // update solution and residual
        // delta_v = delta_v + alpha * p
        // r = r - alpha * Hp

        // check convergence:
        // if (norm(r) < d_newton_solver->solver_inner_tol()) {
        //   break;
        // }

        // compute new search direction
        // r_new_dot_r_new = dot(r,r)
        // beta = r_new_dot_r_new / r_dot_r
        // p = r + beta * p
      }

      // Update v_prev
      if (tid < d_newton_solver->get_n_coef() * 3) {
        d_newton_solver->v_prev()[tid] = d_newton_solver->v_guess()[tid];
      }

      grid.sync();

      // Update positions
      if (tid < d_newton_solver->get_n_coef()) {
        d_data->x12()(tid) = d_newton_solver->x12_prev()(tid) +
                             d_newton_solver->v_guess()(tid * 3 + 0) *
                                 d_newton_solver->solver_time_step();
        d_data->y12()(tid) = d_newton_solver->y12_prev()(tid) +
                             d_newton_solver->v_guess()(tid * 3 + 1) *
                                 d_newton_solver->solver_time_step();
        d_data->z12()(tid) = d_newton_solver->z12_prev()(tid) +
                             d_newton_solver->v_guess()(tid * 3 + 2) *
                                 d_newton_solver->solver_time_step();
      }

      grid.sync();

      // Compute constraints at new position
      if (tid < d_newton_solver->gpu_n_constraints() / 3) {
        compute_constraint_data(d_data);
      }

      grid.sync();

      // dual variable update
      int n_constraints = d_newton_solver->gpu_n_constraints();
      for (int i = tid; i < n_constraints; i += grid.size()) {
        double constraint_val = d_data->constraint()[i];
        d_newton_solver->lambda_guess()[i] +=
            *d_newton_solver->solver_rho() *
            d_newton_solver->solver_time_step() * constraint_val;
      }
      grid.sync();

      if (tid == 0) {
        // check constraint convergence
        double norm_constraint = 0.0;
        for (int i = 0; i < d_newton_solver->gpu_n_constraints(); i++) {
          double constraint_val = d_data->constraint()[i];
          norm_constraint += constraint_val * constraint_val;
        }
        norm_constraint = sqrt(norm_constraint);
        printf("norm_constraint: %.17f\n", norm_constraint);

        if (norm_constraint < d_newton_solver->solver_outer_tol()) {
          printf("Converged constraint: %.17f\n", norm_constraint);
          *d_newton_solver->outer_flag() = 1;
        }
      }

      grid.sync();
    }
  }

  // Final position update
  if (tid < d_newton_solver->get_n_coef()) {
    d_data->x12()(tid) = d_newton_solver->x12_prev()(tid) +
                         d_newton_solver->v_guess()(tid * 3 + 0) *
                             d_newton_solver->solver_time_step();
    d_data->y12()(tid) = d_newton_solver->y12_prev()(tid) +
                         d_newton_solver->v_guess()(tid * 3 + 1) *
                             d_newton_solver->solver_time_step();
    d_data->z12()(tid) = d_newton_solver->z12_prev()(tid) +
                         d_newton_solver->v_guess()(tid * 3 + 2) *
                             d_newton_solver->solver_time_step();
  }

  grid.sync();
}

// Explicit instantiations
template __global__ void one_step_newton_kernel_impl<GPU_ANCF3243_Data>(
    GPU_ANCF3243_Data *, SyncedNewtonSolver *);
template __global__ void one_step_newton_kernel_impl<GPU_ANCF3443_Data>(
    GPU_ANCF3443_Data *, SyncedNewtonSolver *);
template __global__ void one_step_newton_kernel_impl<GPU_FEAT10_Data>(
    GPU_FEAT10_Data *, SyncedNewtonSolver *);

// Wrapper function to call the appropriate kernel based on element type
void SyncedNewtonSolver::OneStepNewton() {
  int numBlocks           = (n_coef_ * 3 + 255) / 256;
  int threadsPerBlock     = 256;
  void *kernelArgs_3243[] = {(void *)&d_data_, (void *)&d_newton_solver_};
  void *kernelArgs_3443[] = {(void *)&d_data_, (void *)&d_newton_solver_};
  void *kernelArgs_T10[]  = {(void *)&d_data_, (void *)&d_newton_solver_};

  if (type_ == TYPE_3243) {
    cudaLaunchCooperativeKernel(
        (void *)one_step_newton_kernel_impl<GPU_ANCF3243_Data>, numBlocks,
        threadsPerBlock, kernelArgs_3243);
  } else if (type_ == TYPE_3443) {
    cudaLaunchCooperativeKernel(
        (void *)one_step_newton_kernel_impl<GPU_ANCF3443_Data>, numBlocks,
        threadsPerBlock, kernelArgs_3443);
  } else if (type_ == TYPE_T10) {
    cudaLaunchCooperativeKernel(
        (void *)one_step_newton_kernel_impl<GPU_FEAT10_Data>, numBlocks,
        threadsPerBlock, kernelArgs_T10);
  }

  HANDLE_ERROR(cudaDeviceSynchronize());
}
