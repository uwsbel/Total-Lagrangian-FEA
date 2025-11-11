#include <cooperative_groups.h>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataFunc.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataFunc.cuh"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "SyncedNewton.cuh"

namespace cg = cooperative_groups;

// =====================
// Device function to perform Cholesky factorization
__device__ void CholeskyFactorizationFunc(Eigen::Map<Eigen::MatrixXd> M,
                                          Eigen::Map<Eigen::MatrixXd> L,
                                          int thread_idx, int n) {
  cg::grid_group grid = cg::this_grid();

  // Rank of this thread across the WHOLE grid, 0..grid.size()-1
  int j    = thread_idx;
  int j_up = grid.size() - 1;  // "index of the last thread"

  for (int i = 0; i <= j_up; ++i) {
    if (j < n && i <= j && i == j) {
      double sum = 0.0;
      for (int k = 0; k < i; ++k) {
        sum += L(i, k) * L(i, k);
      }
      L(i, i) = sqrt(M(i, i) - sum);
    }

    grid.sync();

    if (j < n && i <= j && j > i) {
      double sum = 0.0;
      for (int k = 0; k < i; ++k) {
        sum += L(j, k) * L(i, k);
      }
      L(j, i) = (M(j, i) - sum) / L(i, i);
    }

    grid.sync();
  }
}

__device__ void CholeskySolveForwardFunc(
    Eigen::Map<Eigen::MatrixXd> L,
    Eigen::Map<Eigen::VectorXd> b,  // Changed from MatrixXd
    Eigen::Map<Eigen::VectorXd> y,  // Changed from MatrixXd
    int thread_idx, size_t n) {
  cg::grid_group grid = cg::this_grid();

  int j    = thread_idx;
  int j_up = grid.size() - 1;  // "index of the last thread"

  // Forward substitution to solve L * y = b
  double sum = 0.0;
  for (int i = 0; i <= j_up; ++i) {
    if (j < n && i <= j && i == j) {
      y(j) = (b(j) - sum) / L(j, j);  // Changed from y(j, 0) and b(j)
    }
    grid.sync();

    if (j < n && i <= j && i < j) {
      sum += L(j, i) * y(i);  // Changed from y(i, 0)
    }

    grid.sync();
  }
}

// Device function to perform backward substitution: L^T * x = y
__device__ void CholeskySolveBackwardFunc(
    Eigen::Map<Eigen::MatrixXd> L,
    Eigen::Map<Eigen::VectorXd> y,  // Changed from MatrixXd
    Eigen::Map<Eigen::VectorXd> x,  // Changed from MatrixXd
    int thread_idx, size_t n) {
  cg::grid_group grid = cg::this_grid();

  int j      = thread_idx;
  int j_down = 0;  // We iterate from n-1 down to 0

  // Backward substitution to solve L^T * x = y
  double sum = 0.0;
  for (int i = n - 1; i >= j_down; --i) {
    // Thread i computes x(i)
    if (j < n && i >= j && i == j) {
      x(j) = (y(j) - sum) / L(j, j);  // Changed from x(j, 0) and y(j, 0)
    }
    grid.sync();

    // Threads j < i accumulate sum for x(j): sum += L(i, j) * x(i)
    if (j < n && i >= j && i > j) {
      sum += L(i, j) * x(i);  // Changed from x(i, 0)
    }

    grid.sync();
  }
}

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

// Templated Newton kernel - CORRECTED STRUCTURE
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

  // ========== OUTER LOOP: ALM iterations (update multipliers) ==========
  for (int outer_iter = 0; outer_iter < d_newton_solver->solver_max_outer();
       outer_iter++) {
    if (*d_newton_solver->outer_flag() == 0) {
      // ========== MIDDLE LOOP: Newton iterations (solve R(v)=0) ==========
      for (int newton_iter = 0;
           newton_iter < d_newton_solver->solver_max_inner(); newton_iter++) {
        // Reset Newton convergence flag
        if (tid == 0) {
          *d_newton_solver->inner_flag() = 0;
        }
        grid.sync();

        if (*d_newton_solver->inner_flag() == 0) {
          // ===== UPDATE POSITIONS FROM CURRENT v (DO THIS FIRST!) =====
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

          // ===== COMPUTE GRADIENT R(v) at current positions =====
          if (tid == 0 && newton_iter % 1 == 0) {
            printf("  Newton iter %d\n", newton_iter);
          }

          // Compute P (stress) at current configuration
          if (tid < d_newton_solver->get_n_beam() *
                        d_newton_solver->gpu_n_total_qp()) {
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

          // Compute gradient R(v)
          if (tid < d_newton_solver->get_n_coef() * 3) {
            double g = solver_grad_L(tid, d_data, d_newton_solver);
            d_newton_solver->g()[tid] = g;
          }
          grid.sync();

          if (tid == 0) {
            double norm_g_check = 0.0;
            for (int i = 0; i < 3 * d_newton_solver->get_n_coef(); i++) {
              norm_g_check += d_newton_solver->g()[i] * d_newton_solver->g()[i];
            }
            printf("    ||g|| just computed = %.6e\n", sqrt(norm_g_check));
          }

          // Check Newton convergence (gradient norm)
          if (tid == 0) {
            double norm_g = 0.0;
            for (int i = 0; i < 3 * d_newton_solver->get_n_coef(); i++) {
              norm_g += d_newton_solver->g()(i) * d_newton_solver->g()(i);
            }
            norm_g                     = sqrt(norm_g);
            *d_newton_solver->norm_g() = norm_g;

            double norm_v = 0.0;
            for (int i = 0; i < 3 * d_newton_solver->get_n_coef(); i++) {
              norm_v +=
                  d_newton_solver->v_guess()(i) * d_newton_solver->v_guess()(i);
            }
            norm_v = sqrt(norm_v);

            if (newton_iter % 1 == 0) {
              printf("    ||R|| = %.6e\n", norm_g);
            }

            // Newton convergence check: ||R|| < tol*(1+||v||)
            if (norm_g < d_newton_solver->solver_inner_tol() * (1.0 + norm_v)) {
              printf("    Newton converged at iteration %d\n", newton_iter);
              *d_newton_solver->inner_flag() = 1;
            }
          }
          grid.sync();

          // If Newton converged, break out
          if (*d_newton_solver->inner_flag() == 1) {
            break;
          }

          // ===== SOLVE H*delta_v = -R using CG =====
          // Initialize CG: delta_v = 0, r = -g, p = r
          if (tid < d_newton_solver->get_n_coef() * 3) {
            d_newton_solver->delta_v()[tid] = 0.0;
            d_newton_solver->r()[tid]       = -d_newton_solver->g()[tid];
          }
          grid.sync();

          // ===== ASSEMBLE FULL HESSIAN: H = (M/h) + h*Kt + h^2*J^T*rho*J =====

          // Step 1: Clear Hessian matrix
          int n_dofs = 3 * d_newton_solver->get_n_coef();
          for (int idx = tid; idx < n_dofs * n_dofs; idx += grid.size()) {
            d_newton_solver->H().data()[idx] = 0.0;
          }
          grid.sync();

          // Step 2: Assemble tangent stiffness contribution: h * Kt
          if (tid < d_newton_solver->get_n_beam() *
                        d_newton_solver->gpu_n_total_qp()) {
            for (int idx = tid; idx < d_newton_solver->get_n_beam() *
                                          d_newton_solver->gpu_n_total_qp();
                 idx += grid.size()) {
              int elem_idx = idx / d_newton_solver->gpu_n_total_qp();
              int qp_idx   = idx % d_newton_solver->gpu_n_total_qp();
              compute_hessian_assemble(elem_idx, qp_idx, d_data,
                                       d_newton_solver->H(),
                                       d_newton_solver->solver_time_step());
            }
          }
          grid.sync();

          // Step 3: Add mass matrix contribution: M/h (diagonal 3x3 blocks)
          const double inv_h = 1.0 / d_newton_solver->solver_time_step();
          if (tid < d_newton_solver->get_n_coef()) {
            const int node_i                  = tid;
            const int *__restrict__ offsets   = d_data->csr_offsets();
            const int *__restrict__ columns   = d_data->csr_columns();
            const double *__restrict__ values = d_data->csr_values();

            int row_start = offsets[node_i];
            int row_end   = offsets[node_i + 1];

            for (int idx = row_start; idx < row_end; idx++) {
              int node_j     = columns[idx];
              double mass_ij = values[idx];

              // Add (mass_ij / h) * I_3x3 to H
              // For each DOF (x, y, z)
              for (int dof = 0; dof < 3; dof++) {
                int global_row = 3 * node_i + dof;
                int global_col = 3 * node_j + dof;
                atomicAdd(&d_newton_solver->H()(global_row, global_col),
                          mass_ij * inv_h);
              }
            }
          }
          grid.sync();

          // Step 4: Add constraint Hessian contribution: h^2 * J^T * rho * J
          const int n_constraints = d_newton_solver->gpu_n_constraints();
          if (n_constraints > 0) {
            const double h      = d_newton_solver->solver_time_step();
            const double rho    = *d_newton_solver->solver_rho();
            const double factor = h * h * rho;

            // CSR format for J^T (transpose of constraint Jacobian)
            const int *__restrict__ cjT_offsets   = d_data->cj_csr_offsets();
            const int *__restrict__ cjT_columns   = d_data->cj_csr_columns();
            const double *__restrict__ cjT_values = d_data->cj_csr_values();

            // Compute h^2 * rho * J^T * J (rank-1 contributions per constraint)
            // J is sparse with identity structure for fixed nodes
            // For each constraint c_idx, J has one entry per DOF
            // J^T @ (rho * J) gives outer product of constraint Jacobian
            // columns

            // Parallel over constraints
            for (int c_idx = tid; c_idx < n_constraints; c_idx += grid.size()) {
              // Find all DOFs affected by this constraint (non-zeros in
              // J[c_idx, :]) For identity-like constraints, each constraint
              // affects exactly one DOF But we need to compute J^T[:, c_idx] @
              // J[c_idx, :]

              // Find which DOFs have non-zero entries in this constraint row
              // We need to iterate over all DOFs and check cjT (which stores
              // J^T)
              for (int dof_i = 0; dof_i < n_dofs; dof_i++) {
                // Check if J[c_idx, dof_i] != 0
                // In CSR for J^T: cjT stores columns of J^T (rows of J)
                int col_start_i = cjT_offsets[dof_i];
                int col_end_i   = cjT_offsets[dof_i + 1];

                double J_c_i = 0.0;
                for (int idx_i = col_start_i; idx_i < col_end_i; idx_i++) {
                  if (cjT_columns[idx_i] == c_idx) {
                    J_c_i = cjT_values[idx_i];
                    break;
                  }
                }

                if (J_c_i != 0.0) {
                  // Now find all dof_j where J[c_idx, dof_j] != 0
                  for (int dof_j = 0; dof_j < n_dofs; dof_j++) {
                    int col_start_j = cjT_offsets[dof_j];
                    int col_end_j   = cjT_offsets[dof_j + 1];

                    double J_c_j = 0.0;
                    for (int idx_j = col_start_j; idx_j < col_end_j; idx_j++) {
                      if (cjT_columns[idx_j] == c_idx) {
                        J_c_j = cjT_values[idx_j];
                        break;
                      }
                    }

                    if (J_c_j != 0.0) {
                      // Add h^2 * rho * J[c_idx, dof_i] * J[c_idx, dof_j] to
                      // H[dof_i, dof_j]
                      atomicAdd(&d_newton_solver->H()(dof_i, dof_j),
                                factor * J_c_i * J_c_j);
                    }
                  }
                }
              }
            }
          }
          grid.sync();

          // ========== INNER LOOP: Solve Linear System (solve H*dv=-g)
          // ========== Cholesky solve
          CholeskyFactorizationFunc(d_newton_solver->H(), d_newton_solver->L(),
                                    tid, n_dofs);
          grid.sync();
          // Forward substitution: L * y = -g
          CholeskySolveForwardFunc(d_newton_solver->L(), d_newton_solver->r(),
                                   d_newton_solver->y(), tid, n_dofs);
          grid.sync();
          // Backward substitution
          CholeskySolveBackwardFunc(d_newton_solver->L(), d_newton_solver->y(),
                                    d_newton_solver->delta_v(), tid, n_dofs);
          grid.sync();
          // ========== END INNER LOOP ==========

          // Apply Newton step: v += delta_v
          if (tid < d_newton_solver->get_n_coef() * 3) {
            d_newton_solver->v_guess()[tid] += d_newton_solver->delta_v()[tid];
          }
          grid.sync();

          // NOTE: Positions will be updated at start of NEXT Newton iteration
        }
      }
      // ========== END NEWTON LOOP ==========

      // Update v_prev for next timestep
      if (tid < d_newton_solver->get_n_coef() * 3) {
        d_newton_solver->v_prev()[tid] = d_newton_solver->v_guess()[tid];
      }
      grid.sync();

      // Compute constraints at converged position
      if (tid < d_newton_solver->gpu_n_constraints() / 3) {
        compute_constraint_data(d_data);
      }
      grid.sync();

      // ALM dual variable update: lambda += rho*c (NO h!)
      int n_constraints = d_newton_solver->gpu_n_constraints();
      for (int i = tid; i < n_constraints; i += grid.size()) {
        double constraint_val = d_data->constraint()[i];
        d_newton_solver->lambda_guess()[i] +=
            *d_newton_solver->solver_rho() * constraint_val;
      }
      grid.sync();

      // Check ALM convergence
      if (tid == 0) {
        double norm_constraint = 0.0;
        for (int i = 0; i < d_newton_solver->gpu_n_constraints(); i++) {
          double constraint_val = d_data->constraint()[i];
          norm_constraint += constraint_val * constraint_val;
        }
        norm_constraint = sqrt(norm_constraint);
        printf("Outer iter %d: norm_constraint = %.6e\n", outer_iter,
               norm_constraint);

        if (norm_constraint < d_newton_solver->solver_outer_tol()) {
          printf("Outer loop converged at iteration %d\n", outer_iter);
          *d_newton_solver->outer_flag() = 1;
        }
      }
      grid.sync();
    }
  }
  // ========== END ALM LOOP ==========

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
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  int numBlocks           = (n_coef_ * 3 + 255) / 256;
  int threadsPerBlock     = 256;
  void *kernelArgs_3243[] = {(void *)&d_data_, (void *)&d_newton_solver_};
  void *kernelArgs_3443[] = {(void *)&d_data_, (void *)&d_newton_solver_};
  void *kernelArgs_T10[]  = {(void *)&d_data_, (void *)&d_newton_solver_};

  HANDLE_ERROR(cudaEventRecord(start));

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

  HANDLE_ERROR(cudaEventRecord(stop));
  HANDLE_ERROR(cudaDeviceSynchronize());

  float milliseconds = 0;
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

  std::cout << "OneStepNewton kernel time: " << milliseconds << " ms"
            << std::endl;

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
}

// ===============================================
// experimental cusparse and cudss one step newton
// ===============================================

// Templated Newton kernel - CORRECTED STRUCTURE
__global__ void cudss_solve_compute_p(GPU_FEAT10_Data *d_data,
                                      SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < d_newton_solver->get_n_beam() * d_newton_solver->gpu_n_total_qp()) {
    int idx      = tid;
    int elem_idx = idx / d_newton_solver->gpu_n_total_qp();
    int qp_idx   = idx % d_newton_solver->gpu_n_total_qp();
    compute_p(elem_idx, qp_idx, d_data);
  }
}

__global__ void cudss_solve_clear_internal_force(GPU_FEAT10_Data *d_data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < d_data->n_coef * 3) {
    clear_internal_force(d_data);
  }
}

__global__ void cudss_solve_compute_internal_force(
    GPU_FEAT10_Data *d_data, SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < d_newton_solver->get_n_beam() * d_newton_solver->gpu_n_shape()) {
    int idx      = tid;
    int elem_idx = idx / d_newton_solver->gpu_n_shape();
    int node_idx = idx % d_newton_solver->gpu_n_shape();
    compute_internal_force(elem_idx, node_idx, d_data);
  }
}

__global__ void cudss_solve_constraints_eval(
    GPU_FEAT10_Data *d_data, SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < d_newton_solver->gpu_n_constraints() / 3) {
    compute_constraint_data(d_data);
  }
}

__global__ void cudss_solve_compute_grad_l(
    GPU_FEAT10_Data *d_data, SyncedNewtonSolver *d_newton_solver) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < d_newton_solver->get_n_coef() * 3) {
    double g                  = solver_grad_L(tid, d_data, d_newton_solver);
    d_newton_solver->g()[tid] = g;
  }

  if (tid == 0) {
    double norm_g_check = 0.0;
    for (int i = 0; i < 3 * d_newton_solver->get_n_coef(); i++) {
      norm_g_check += d_newton_solver->g()[i] * d_newton_solver->g()[i];
    }
    printf("    ||g|| just computed = %.6e\n", sqrt(norm_g_check));
  }
}

// Wrapper function to call the appropriate kernel based on element type
void SyncedNewtonSolver::OneStepNewtonCuDSS() {
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  int threadsPerBlock                = 256;
  int numBlocks_compute_p            = (n_beam_ * n_total_qp_ + 255) / 256;
  int numBlocks_clear_internal_force = (n_coef_ * 3 + 255) / 256;
  int numBlocks_internal_force       = (n_beam_ * n_shape_ + 255) / 256;
  int numBlocks_grad_l               = (n_coef_ * 3 + 255) / 256;
  int numBlocks_constraints_eval     = (n_constraints_ / 3 + 255) / 256;

  HANDLE_ERROR(cudaEventRecord(start));

  // Normal kernel launch (no cooperative groups)
  cudss_solve_compute_p<<<numBlocks_compute_p, threadsPerBlock>>>(
      static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_);

  cudss_solve_clear_internal_force<<<numBlocks_clear_internal_force,
                                     threadsPerBlock>>>(
      static_cast<GPU_FEAT10_Data *>(d_data_));
  cudss_solve_compute_internal_force<<<numBlocks_internal_force,
                                       threadsPerBlock>>>(
      static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_);

  cudss_solve_constraints_eval<<<numBlocks_constraints_eval, threadsPerBlock>>>(
      static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_);
  cudss_solve_compute_grad_l<<<numBlocks_grad_l, threadsPerBlock>>>(
      static_cast<GPU_FEAT10_Data *>(d_data_), d_newton_solver_);

  // initialize
  // clear hessian
  // hessian assemble tangent
  // hessian assemble mass constrib
  // hessian assemble constraint
  // solve

  HANDLE_ERROR(cudaDeviceSynchronize());
  float milliseconds = 0;
  HANDLE_ERROR(cudaEventRecord(stop));
  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

  std::cout << "OneStepNewtonC kernel time: " << milliseconds << " ms"
            << std::endl;

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
}
