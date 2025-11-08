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
            d_newton_solver->p()[tid]       = d_newton_solver->r()[tid];
          }
          grid.sync();

          // ========== INNER LOOP: CG iterations (solve H*dv=-g) ==========
          for (int cg_iter = 0; cg_iter < 200; cg_iter++) {
            // 0) Hp = 0
            if (tid < d_newton_solver->get_n_coef() * 3) {
              d_newton_solver->Hp()[tid] = 0.0;
            }
            grid.sync();

            // 1) Hp = H * p  (RECOMPUTE every iteration)
            // Tangent: h*Kt*p (geometry is frozen for this Newton step)
            if (tid < d_newton_solver->get_n_beam() *
                          d_newton_solver->gpu_n_total_qp()) {
              for (int idx = tid; idx < d_newton_solver->get_n_beam() *
                                            d_newton_solver->gpu_n_total_qp();
                   idx += grid.size()) {
                int elem_idx = idx / d_newton_solver->gpu_n_total_qp();
                int qp_idx   = idx % d_newton_solver->gpu_n_total_qp();
                compute_hessian_p(elem_idx, qp_idx, d_data,
                                  d_newton_solver->p().data(),
                                  d_newton_solver->Hp().data(),
                                  d_newton_solver->solver_time_step());
              }
            }
            grid.sync();

            // Mass: (M/h)*p
            double h = d_newton_solver->solver_time_step();
            if (tid < d_newton_solver->get_n_coef()) {
              int row_start = d_data->csr_offsets()[tid];
              int row_end   = d_data->csr_offsets()[tid + 1];
              for (int dof = 0; dof < 3; dof++) {
                int row_dof_idx = tid * 3 + dof;
                double mass_p   = 0.0;
                for (int idx = row_start; idx < row_end; idx++) {
                  int col_node    = d_data->csr_columns()[idx];
                  double mass_val = d_data->csr_values()[idx];
                  int col_dof_idx = col_node * 3 + dof;
                  mass_p += mass_val * d_newton_solver->p()[col_dof_idx];
                }
                d_newton_solver->Hp()[row_dof_idx] += mass_p / h;
              }
            }
            grid.sync();

            // Constraint: h^2 J^T rho J p
            if (d_newton_solver->gpu_n_constraints() > 0) {
              double rho = *d_newton_solver->solver_rho();
              if (tid < d_newton_solver->gpu_n_constraints()) {
                int row_start = d_data->cj_csr_offsets()[tid];
                int row_end   = d_data->cj_csr_offsets()[tid + 1];
                double Jp_i   = 0.0;
                for (int idx = row_start; idx < row_end; idx++) {
                  int col        = d_data->cj_csr_columns()[idx];
                  double jac_val = d_data->cj_csr_values()[idx];
                  Jp_i += jac_val * d_newton_solver->p()[col];
                }
                for (int idx = row_start; idx < row_end; idx++) {
                  int col        = d_data->cj_csr_columns()[idx];
                  double jac_val = d_data->cj_csr_values()[idx];
                  atomicAdd(&d_newton_solver->Hp()[col],
                            h * h * jac_val * rho * Jp_i);
                }
              }
            }
            grid.sync();

            // 2) Dot products for alpha
            if (tid == 0) {
              *d_newton_solver->r_dot_r()  = 0.0;
              *d_newton_solver->p_dot_Hp() = 0.0;
            }
            grid.sync();

            if (tid < d_newton_solver->get_n_coef() * 3) {
              double r_val  = d_newton_solver->r()[tid];
              double p_val  = d_newton_solver->p()[tid];
              double Hp_val = d_newton_solver->Hp()[tid];
              atomicAdd(d_newton_solver->r_dot_r(), r_val * r_val);
              atomicAdd(d_newton_solver->p_dot_Hp(), p_val * Hp_val);
            }
            grid.sync();

            if (tid == 0) {
              double denom = *d_newton_solver->p_dot_Hp();
              if (denom <= 0.0) {
                printf("CG breakdown: p^T H p = %e (non-SPD?)\n", denom);
                *d_newton_solver->alpha_cg() = 0.0;
              } else {
                *d_newton_solver->alpha_cg() =
                    (*d_newton_solver->r_dot_r()) / denom;
              }
            }
            grid.sync();

            // 3) Update delta_v, r
            if (tid < d_newton_solver->get_n_coef() * 3) {
              double alpha = *d_newton_solver->alpha_cg();
              d_newton_solver->delta_v()[tid] +=
                  alpha * d_newton_solver->p()[tid];
              d_newton_solver->r()[tid] -= alpha * d_newton_solver->Hp()[tid];
            }
            grid.sync();

            // 4) Convergence (relative)
            if (tid == 0)
              *d_newton_solver->norm_r() = 0.0;
            grid.sync();

            if (tid < d_newton_solver->get_n_coef() * 3) {
              double r_val = d_newton_solver->r()[tid];
              atomicAdd(d_newton_solver->norm_r(), r_val * r_val);
            }
            grid.sync();

            bool stop = false;
            if (tid == 0) {
              double r2 = *d_newton_solver->norm_r();
              
              // Store r0 on first CG iteration using r_new_dot_r_new buffer
              if (cg_iter == 0) {
                *d_newton_solver->r_new_dot_r_new() = r2;
              }
              double r0 = *d_newton_solver->r_new_dot_r_new();  // Read stored r0
              
              if (cg_iter % 5 == 0) {
                printf("      CG iter %d: ||r|| = %.6e (rel=%.3e)\n", cg_iter,
                       sqrt(r2), sqrt(r2 / (r0 + 1e-30)));
              }
              stop = (r2 / (r0 + 1e-30) < 1e-12) || (sqrt(r2) < 1e-12);
            }
            grid.sync();
            if (stop)
              break;

            // 5) beta and 6) p update
            if (tid == 0)
              *d_newton_solver->r_new_dot_r_new() = 0.0;
            grid.sync();

            if (tid < d_newton_solver->get_n_coef() * 3) {
              double r_val = d_newton_solver->r()[tid];
              atomicAdd(d_newton_solver->r_new_dot_r_new(), r_val * r_val);
            }
            grid.sync();

            if (tid == 0) {
              double r_new                = *d_newton_solver->r_new_dot_r_new();
              double r_old                = *d_newton_solver->r_dot_r();
              *d_newton_solver->beta_cg() = r_new / (r_old + 1e-30);
            }
            grid.sync();

            if (tid < d_newton_solver->get_n_coef() * 3) {
              double beta = *d_newton_solver->beta_cg();
              d_newton_solver->p()[tid] =
                  d_newton_solver->r()[tid] + beta * d_newton_solver->p()[tid];
            }
            grid.sync();
          }
          // ========== END CG LOOP ==========

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
