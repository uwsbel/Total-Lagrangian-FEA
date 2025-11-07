#include <cooperative_groups.h>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataFunc.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataFunc.cuh"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "SyncedNewton.cuh"

namespace cg = cooperative_groups;

// Templated solver_grad_L - same as AdamW/Nesterov
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
  res -= data->f_ext()(tid);      // Subtract f_ext

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
      res += dt * constraint_jac_val * (lam[constraint_idx] + rho * constraint_val);
    }
  }

  return res;
}

// Templated Newton kernel - STARTER CODE (implement Newton method here)
template <typename ElementType>
__global__ void one_step_newton_kernel_impl(ElementType *d_data,
                                            SyncedNewtonSolver *d_newton_solver) {
  cg::grid_group grid = cg::this_grid();
  int tid             = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: Implement Newton solver
  // 1. Save previous positions
  // 2. Outer loop for ALM iterations
  // 3. Inner loop for Newton iterations
  //    - Compute gradient
  //    - Solve linear system H * dv = -g
  //    - Update velocity
  // 4. Update dual variables (lambda)
  // 5. Check convergence

  if (tid == 0) {
    printf("Newton solver: starter code - not yet implemented\n");
  }

  grid.sync();
}

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
