#include "SyncedNesterov.cuh"
#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataKernels.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataKernels.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ double solver_grad_L(int tid, ElementBase *d_data, SyncedNesterovSolver *solver)
{
    double res = 0.0;

    int node_i = tid / 3;
    int dof_i = tid % 3;

    // Mass matrix contribution
    for (int node_j = 0; node_j < d_data->get_n_coef(); node_j++)
    {
        double mass_ij = 0.0;
        if (d_data->type == TYPE_3243)
        {
            auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
            mass_ij = data->node_values()(node_i, node_j);
        }
        else if (d_data->type == TYPE_3443)
        {
            auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
            mass_ij = data->node_values()(node_i, node_j);
        }

        int tid_j = node_j * 3 + dof_i;
        double v_diff = solver->v_guess()[tid_j] - solver->v_prev()[tid_j];
        res += mass_ij * v_diff / solver->solver_time_step();
    }

    // Internal force
    if (d_data->type == TYPE_3243)
    {
        auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
        res -= (-data->f_elem_out()(tid));
    }
    else if (d_data->type == TYPE_3443)
    {
        auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
        res -= (-data->f_elem_out()(tid));
    }

    if (tid == 3 * d_data->get_n_coef() - 10)
    {
        res -= 10000.0;
    }

    // Constraints
    for (int i = 0; i < 12; i++)
    {
        double constraint_jac_val = 0.0;
        double constraint_val = 0.0;

        if (d_data->type == TYPE_3243)
        {
            auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
            constraint_jac_val = data->constraint_jac()(i, tid);
            constraint_val = data->constraint()[i];
        }
        else if (d_data->type == TYPE_3443)
        {
            auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
            constraint_jac_val = data->constraint_jac()(i, tid);
            constraint_val = data->constraint()[i];
        }

        res += constraint_jac_val * (solver->lambda_guess()[i] + *solver->solver_rho() * solver->solver_time_step() * constraint_val);
    }

    return res;
}

__global__ void
one_step_nesterov_kernel(ElementBase *d_data, SyncedNesterovSolver *d_nesterov_solver)
{
    cg::grid_group grid = cg::this_grid();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // assign x12_prev, y12_prev, z12_prev
    if (tid < d_data->get_n_coef())
    {
        if (d_data->type == TYPE_3243)
        {
            auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
            d_nesterov_solver->x12_prev()(tid) = data->x12()(tid);
            d_nesterov_solver->y12_prev()(tid) = data->y12()(tid);
            d_nesterov_solver->z12_prev()(tid) = data->z12()(tid);
        }
        else if (d_data->type == TYPE_3443)
        {
            auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
            d_nesterov_solver->x12_prev()(tid) = data->x12()(tid);
            d_nesterov_solver->y12_prev()(tid) = data->y12()(tid);
            d_nesterov_solver->z12_prev()(tid) = data->z12()(tid);
        }
    }

    grid.sync();

    if (tid == 0)
    {
        *d_nesterov_solver->inner_flag() = 0;
        *d_nesterov_solver->outer_flag() = 0;
    }

    grid.sync();

    for (int outer_iter = 0; outer_iter < d_nesterov_solver->solver_max_outer(); outer_iter++)
    {
        if (*d_nesterov_solver->outer_flag() == 0)
        {
            // Initialize variables for each thread
            double v_k = 0.0;
            double v_next = 0.0;
            double v_km1 = 0.0;
            double t = 1.0;
            double v_prev = 0.0;

            if (tid == 0)
            {
                *d_nesterov_solver->prev_norm_g() = 0.0;
                *d_nesterov_solver->norm_g() = 0.0;
                printf("resetting\n");
            }

            grid.sync();

            // Initialize for valid threads only
            if (tid < d_data->get_n_coef() * 3)
            {
                v_prev = d_nesterov_solver->v_prev()(tid);
                v_k = d_nesterov_solver->v_guess()(tid);
                v_km1 = d_nesterov_solver->v_guess()(tid); // zero momentum at first step
                t = 1.0;
            }

            double t_next = 1.0; // Declare t_next here

            for (int inner_iter = 0; inner_iter < d_nesterov_solver->solver_max_inner(); inner_iter++)
            {

                grid.sync();

                if (*d_nesterov_solver->inner_flag() == 0)
                {
                    if (tid == 0)
                    {
                        printf("outer iter: %d, inner iter: %d\n", outer_iter, inner_iter);
                    }

                    // Step 1: Each thread computes its look-ahead velocity component
                    double y = 0.0; // Declare y here
                    if (tid < d_data->get_n_coef() * 3)
                    {
                        t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t * t));
                        double beta = (t - 1.0) / t_next;
                        y = v_k + beta * (v_k - v_km1); // Nesterov Look Ahead

                        // Store look-ahead velocity temporarily
                        d_nesterov_solver->v_guess()(tid) = y; // Use v_guess as temp storage for y
                    }

                    grid.sync();

                    // Step 2: Update scratch positions using look-ahead velocities
                    if (tid < d_data->get_n_coef())
                    {
                        if (d_data->type == TYPE_3243)
                        {
                            auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
                            data->x12()(tid) = d_nesterov_solver->x12_prev()(tid) + d_nesterov_solver->solver_time_step() * d_nesterov_solver->v_guess()(tid * 3 + 0);
                            data->y12()(tid) = d_nesterov_solver->y12_prev()(tid) + d_nesterov_solver->solver_time_step() * d_nesterov_solver->v_guess()(tid * 3 + 1);
                            data->z12()(tid) = d_nesterov_solver->z12_prev()(tid) + d_nesterov_solver->solver_time_step() * d_nesterov_solver->v_guess()(tid * 3 + 2);
                        }
                        else if (d_data->type == TYPE_3443)
                        {
                            auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
                            data->x12()(tid) = d_nesterov_solver->x12_prev()(tid) + d_nesterov_solver->solver_time_step() * d_nesterov_solver->v_guess()(tid * 3 + 0);
                            data->y12()(tid) = d_nesterov_solver->y12_prev()(tid) + d_nesterov_solver->solver_time_step() * d_nesterov_solver->v_guess()(tid * 3 + 1);
                            data->z12()(tid) = d_nesterov_solver->z12_prev()(tid) + d_nesterov_solver->solver_time_step() * d_nesterov_solver->v_guess()(tid * 3 + 2);
                        }
                    }

                    grid.sync();

                    // print f_elem_out
                    // if (tid == 0)
                    // {
                    //     printf("pre f_elem_out");
                    //     for (int i = 0; i < 3 * d_data->get_n_coef(); i++)
                    //     {
                    //         printf("%f ", d_data->f_elem_out()(i));
                    //     }
                    //     printf("\n");
                    // }

                    // Step 3: Compute internal forces at look-ahead positions

                    // if (tid < d_data->get_n_beam() * Quadrature::N_TOTAL_QP)
                    // {
                    //     int elem_idx = tid / Quadrature::N_TOTAL_QP;
                    //     int qp_idx = tid % Quadrature::N_TOTAL_QP;
                    //     compute_deformation_gradient(elem_idx, qp_idx, d_data);
                    // }

                    // grid.sync();

                    if (tid < d_data->get_n_beam() * Quadrature::N_TOTAL_QP)
                    {
                        int elem_idx = tid / Quadrature::N_TOTAL_QP;
                        int qp_idx = tid % Quadrature::N_TOTAL_QP;
                        if (d_data->type == TYPE_3243)
                        {
                            ancf3243_compute_p(elem_idx, qp_idx, static_cast<GPU_ANCF3243_Data *>(d_data));
                        }
                        else if (d_data->type == TYPE_3443)
                        {
                            ancf3443_compute_p(elem_idx, qp_idx, static_cast<GPU_ANCF3443_Data *>(d_data));
                        }
                    }

                    grid.sync();

                    if (tid < d_data->get_n_beam() * Quadrature::N_SHAPE)
                    {
                        int elem_idx = tid / Quadrature::N_SHAPE;
                        int node_idx = tid % Quadrature::N_SHAPE;
                        if (d_data->type == TYPE_3243)
                        {
                            ancf3243_compute_internal_force(elem_idx, node_idx, static_cast<GPU_ANCF3243_Data *>(d_data));
                        }
                        else if (d_data->type == TYPE_3443)
                        {
                            ancf3443_compute_internal_force(elem_idx, node_idx, static_cast<GPU_ANCF3443_Data *>(d_data));
                        }
                    }

                    grid.sync();

                    // if (tid == 0)
                    // {
                    //     printf("post f_elem_out");
                    //     for (int i = 0; i < 3 * d_data->get_n_coef(); i++)
                    //     {
                    //         printf("%f ", d_nesterov_solver->g()(i));
                    //     }
                    //     printf("\n");
                    // }

                    if (tid == 0)
                    {
                        if (d_data->type == TYPE_3243)
                            ancf3243_compute_constraint_data(static_cast<GPU_ANCF3243_Data *>(d_data));
                        else if (d_data->type == TYPE_3443)
                            ancf3443_compute_constraint_data(static_cast<GPU_ANCF3443_Data *>(d_data));
                    }

                    grid.sync();

                    if (tid < d_data->get_n_coef() * 3)
                    {
                        double g = solver_grad_L(tid, d_data, d_nesterov_solver);
                        d_nesterov_solver->g()[tid] = g;
                    }

                    grid.sync();

                    // print v_guess
                    // if (tid == 0)
                    // {
                    //     printf("v_guess: ");
                    //     for (int i = 0; i < 3 * d_data->get_n_coef(); i++)
                    //     {
                    //         printf("%f ", d_nesterov_solver->v_guess()(i));
                    //     }
                    //     printf("\n");
                    // }

                    // // print g
                    // if (tid == 0)
                    // {
                    //     printf("solver_grad_l g: ");
                    //     for (int i = 0; i < 3 * d_data->get_n_coef(); i++)
                    //     {
                    //         printf("%f ", d_nesterov_solver->g()(i));
                    //     }
                    //     printf("\n");
                    // }

                    if (tid == 0)
                    {
                        // calculate norm of g
                        double norm_g = 0.0;
                        for (int i = 0; i < 3 * d_data->get_n_coef(); i++)
                        {
                            norm_g += d_nesterov_solver->g()(i) * d_nesterov_solver->g()(i);
                        }
                        *d_nesterov_solver->norm_g() = sqrt(norm_g);
                        printf("norm_g: %.17f\n", *d_nesterov_solver->norm_g());

                        if (inner_iter > 0 && abs(*d_nesterov_solver->norm_g() - *d_nesterov_solver->prev_norm_g()) < d_nesterov_solver->solver_inner_tol())
                        {
                            printf("Converged diff: %.17f\n", *d_nesterov_solver->norm_g() - *d_nesterov_solver->prev_norm_g());
                            *d_nesterov_solver->inner_flag() = 1;
                        }
                    }

                    grid.sync();

                    // Step 4: Compute gradients and update velocities
                    if (tid < d_data->get_n_coef() * 3)
                    {
                        v_next = y - d_nesterov_solver->solver_alpha() * d_nesterov_solver->g()[tid];

                        d_nesterov_solver->v_next()[tid] = v_next;
                        d_nesterov_solver->v_k()[tid] = v_k;
                    }

                    grid.sync();

                    // set termination, abs of norm(v_next)-norm(v_k) < inner_tol
                    if (tid == 0)
                    {
                        double norm_v_next = 0.0;
                        double norm_v_k = 0.0;
                        for (int i = 0; i < 3 * d_data->get_n_coef(); i++)
                        {
                            norm_v_next += d_nesterov_solver->v_next()(i) * d_nesterov_solver->v_next()(i);
                            norm_v_k += d_nesterov_solver->v_k()(i) * d_nesterov_solver->v_k()(i);
                        }
                        norm_v_next = sqrt(norm_v_next);
                        norm_v_k = sqrt(norm_v_k);
                        printf("norm_v_next: %.17f, norm_v_k: %.17f\n", norm_v_next, norm_v_k);

                        if (inner_iter > 0 && abs(norm_v_next - norm_v_k) < d_nesterov_solver->solver_inner_tol())
                        {
                            printf("Converged velocity: %.17f\n", abs(norm_v_next - norm_v_k));
                            *d_nesterov_solver->inner_flag() = 1;
                        }
                    }

                    grid.sync();

                    if (tid < d_data->get_n_coef() * 3)
                    {
                        // Update for next iteration
                        v_km1 = v_k;
                        v_k = v_next;
                        t = t_next;

                        // Store final velocity
                        d_nesterov_solver->v_guess()[tid] = v_next;
                    }

                    grid.sync();

                    if (tid == 0)
                    {
                        *d_nesterov_solver->prev_norm_g() = *d_nesterov_solver->norm_g();
                    }

                    grid.sync();
                }
            }

            // After inner loop convergence, update v_prev for next outer iteration
            if (tid < d_data->get_n_coef() * 3)
            {
                d_nesterov_solver->v_prev()[tid] = d_nesterov_solver->v_guess()[tid];
            }

            grid.sync();

            // Update positions: q_new = q_prev + h * v (parallel across threads)
            if (tid < d_data->get_n_coef())
            {
                if (d_data->type == TYPE_3243)
                {
                    auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
                    data->x12()(tid) = d_nesterov_solver->x12_prev()(tid) + d_nesterov_solver->v_guess()(tid * 3 + 0) * d_nesterov_solver->solver_time_step();
                    data->y12()(tid) = d_nesterov_solver->y12_prev()(tid) + d_nesterov_solver->v_guess()(tid * 3 + 1) * d_nesterov_solver->solver_time_step();
                    data->z12()(tid) = d_nesterov_solver->z12_prev()(tid) + d_nesterov_solver->v_guess()(tid * 3 + 2) * d_nesterov_solver->solver_time_step();
                }
                else if (d_data->type == TYPE_3443)
                {
                    auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
                    data->x12()(tid) = d_nesterov_solver->x12_prev()(tid) + d_nesterov_solver->v_guess()(tid * 3 + 0) * d_nesterov_solver->solver_time_step();
                    data->y12()(tid) = d_nesterov_solver->y12_prev()(tid) + d_nesterov_solver->v_guess()(tid * 3 + 1) * d_nesterov_solver->solver_time_step();
                    data->z12()(tid) = d_nesterov_solver->z12_prev()(tid) + d_nesterov_solver->v_guess()(tid * 3 + 2) * d_nesterov_solver->solver_time_step();
                }
            }

            grid.sync();

            // Only thread 0 handles constraint computation and dual variable updates
            if (tid == 0)
            {
                // Compute constraints at new position
                if (d_data->type == TYPE_3243)
                {
                    ancf3243_compute_constraint_data(static_cast<GPU_ANCF3243_Data *>(d_data));
                }
                else if (d_data->type == TYPE_3443)
                {
                    ancf3443_compute_constraint_data(static_cast<GPU_ANCF3443_Data *>(d_data));
                }

                // Dual variable update: lam += rho * h * c(q_new)
                for (int i = 0; i < 12; i++)
                {
                    double constraint_val = 0.0;
                    if (d_data->type == TYPE_3243)
                    {
                        auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
                        constraint_val = data->constraint()[i];
                    }
                    else if (d_data->type == TYPE_3443)
                    {
                        auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
                        constraint_val = data->constraint()[i];
                    }
                    d_nesterov_solver->lambda_guess()[i] += *d_nesterov_solver->solver_rho() * d_nesterov_solver->solver_time_step() * constraint_val;
                }

                // Termination on the norm of constraint < outer_tol
                double norm_constraint = 0.0;
                for (int i = 0; i < 12; i++)
                {
                    double constraint_val = 0.0;
                    if (d_data->type == TYPE_3243)
                    {
                        auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
                        constraint_val = data->constraint()[i];
                    }
                    else if (d_data->type == TYPE_3443)
                    {
                        auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
                        constraint_val = data->constraint()[i];
                    }
                    norm_constraint += constraint_val * constraint_val;
                }
                norm_constraint = sqrt(norm_constraint);
                printf("norm_constraint: %.17f\n", norm_constraint);

                if (abs(norm_constraint) < d_nesterov_solver->solver_outer_tol())
                {
                    printf("Converged constraint: %.17f\n", abs(norm_constraint));
                    *d_nesterov_solver->outer_flag() = 1;
                }
            }

            grid.sync();
        }
        grid.sync();
    }

    // finally write data back to x12, y12, z12
    // explicit integration
    if (tid < d_data->get_n_coef())
    {
        if (d_data->type == TYPE_3243)
        {
            auto *data = static_cast<GPU_ANCF3243_Data *>(d_data);
            data->x12()(tid) = d_nesterov_solver->x12_prev()(tid) + d_nesterov_solver->v_guess()(tid * 3 + 0) * d_nesterov_solver->solver_time_step();
            data->y12()(tid) = d_nesterov_solver->y12_prev()(tid) + d_nesterov_solver->v_guess()(tid * 3 + 1) * d_nesterov_solver->solver_time_step();
            data->z12()(tid) = d_nesterov_solver->z12_prev()(tid) + d_nesterov_solver->v_guess()(tid * 3 + 2) * d_nesterov_solver->solver_time_step();
        }
        else if (d_data->type == TYPE_3443)
        {
            auto *data = static_cast<GPU_ANCF3443_Data *>(d_data);
            data->x12()(tid) = d_nesterov_solver->x12_prev()(tid) + d_nesterov_solver->v_guess()(tid * 3 + 0) * d_nesterov_solver->solver_time_step();
            data->y12()(tid) = d_nesterov_solver->y12_prev()(tid) + d_nesterov_solver->v_guess()(tid * 3 + 1) * d_nesterov_solver->solver_time_step();
            data->z12()(tid) = d_nesterov_solver->z12_prev()(tid) + d_nesterov_solver->v_guess()(tid * 3 + 2) * d_nesterov_solver->solver_time_step();
        }
    }

    grid.sync();
}

void SyncedNesterovSolver::OneStepNesterov()
{
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    int threads = 128;

    cudaDeviceProp props;
    HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

    int maxBlocksPerSm = 0;
    HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSm, one_step_nesterov_kernel, threads, 0));
    int maxCoopBlocks = maxBlocksPerSm * props.multiProcessorCount;

    int N = max(n_coef_ * 3, n_beam_ * Quadrature::N_TOTAL_QP);
    int blocksNeeded = (N + threads - 1) / threads;
    int blocks = std::min(blocksNeeded, maxCoopBlocks);

    ElementBase *element_data = nullptr;
    if (d_data_->type == TYPE_3243)
    {
        auto *typed_data = static_cast<GPU_ANCF3243_Data *>(d_data_);
        element_data = typed_data->d_data;
    }
    else if (d_data_->type == TYPE_3443)
    {
        auto *typed_data = static_cast<GPU_ANCF3443_Data *>(d_data_);
        element_data = typed_data->d_data;
    }
    void *args[] = {&element_data, &d_nesterov_solver_};

    HANDLE_ERROR(cudaEventRecord(start));
    HANDLE_ERROR(cudaLaunchCooperativeKernel((void *)one_step_nesterov_kernel, blocks, threads, args));
    HANDLE_ERROR(cudaEventRecord(stop));

    HANDLE_ERROR(cudaDeviceSynchronize());

    float milliseconds = 0;
    HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "OneStepNesterov kernel time: " << milliseconds << " ms" << std::endl;

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
}