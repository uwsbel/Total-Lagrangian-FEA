
#include <eigen3/Eigen/Dense>
#include <iostream>

// Definition of GPU_ANCF3243 and data access device functions
#pragma once

#ifndef HANDLE_ERROR_MACRO
#define HANDLE_ERROR_MACRO
static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#endif

// Macro definition

#define N_SHAPE 8
#define N_BEAM 2
#define N_COEF (N_SHAPE + 4 * (N_BEAM - 1))

// Define constants for array sizes
#define N_QP_6 6 // 6-point quadrature
#define N_QP_3 3 // 3-point quadrature
#define N_QP_2 2 // 2-point quadrature

#define N_TOTAL_QP (N_QP_3 * N_QP_2 * N_QP_2) // Total number of quadrature points

//
// define a SAP data strucutre
struct GPU_ANCF3243_Data
{
#if defined(__CUDACC__)

    // Const get functions
    __device__ const Eigen::Map<Eigen::MatrixXd> B_inv() const
    {
        int row_size = N_SHAPE;
        int col_size = N_SHAPE;
        return Eigen::Map<Eigen::MatrixXd>(d_B_inv, row_size, col_size);
    }

    __device__ Eigen::Map<Eigen::MatrixXd> ds_du_pre(int qp_idx) const
    {
        int row_size = N_SHAPE;
        int col_size = 3;
        double *qp_data = d_ds_du_pre + qp_idx * N_SHAPE * 3;
        return Eigen::Map<Eigen::MatrixXd>(qp_data, row_size, col_size);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> gauss_xi_m() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_gauss_xi_m, N_QP_6);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> gauss_xi() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_gauss_xi, N_QP_3);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> gauss_eta() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_gauss_eta, N_QP_2);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> gauss_zeta() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_gauss_zeta, N_QP_2);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> weight_xi_m() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_weight_xi_m, N_QP_6);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> weight_xi() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_weight_xi, N_QP_3);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> weight_eta() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_weight_eta, N_QP_2);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> weight_zeta() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_weight_zeta, N_QP_2);
    }

    __device__ Eigen::Map<Eigen::VectorXd> x12_jac()
    {
        return Eigen::Map<Eigen::VectorXd>(d_x12_jac, N_COEF);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const x12_jac() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_x12_jac, N_COEF);
    }

    __device__ Eigen::Map<Eigen::VectorXd> y12_jac()
    {
        return Eigen::Map<Eigen::VectorXd>(d_y12_jac, N_COEF);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const y12_jac() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_y12_jac, N_COEF);
    }

    __device__ Eigen::Map<Eigen::VectorXd> z12_jac()
    {
        return Eigen::Map<Eigen::VectorXd>(d_z12_jac, N_COEF);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const z12_jac() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_z12_jac, N_COEF);
    }

    __device__ Eigen::Map<Eigen::VectorXd> x12()
    {
        return Eigen::Map<Eigen::VectorXd>(d_x12, N_COEF);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const x12() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_x12, N_COEF);
    }

    __device__ Eigen::Map<Eigen::VectorXd> y12()
    {
        return Eigen::Map<Eigen::VectorXd>(d_y12, N_COEF);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const y12() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_y12, N_COEF);
    }

    __device__ Eigen::Map<Eigen::VectorXd> z12()
    {
        return Eigen::Map<Eigen::VectorXd>(d_z12, N_COEF);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const z12() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_z12, N_COEF);
    }

    // single element N_SHAPE retrieval function
    __device__ Eigen::Map<Eigen::VectorXd> x12(int elem)
    {
        return Eigen::Map<Eigen::VectorXd>(d_x12 + elem * (N_SHAPE / 2), N_SHAPE);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const x12(int elem) const
    {
        return Eigen::Map<Eigen::VectorXd>(d_x12 + elem * (N_SHAPE / 2), N_SHAPE);
    }

    __device__ Eigen::Map<Eigen::VectorXd> y12(int elem)
    {
        return Eigen::Map<Eigen::VectorXd>(d_y12 + elem * (N_SHAPE / 2), N_SHAPE);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const y12(int elem) const
    {
        return Eigen::Map<Eigen::VectorXd>(d_y12 + elem * (N_SHAPE / 2), N_SHAPE);
    }

    __device__ Eigen::Map<Eigen::VectorXd> z12(int elem)
    {
        return Eigen::Map<Eigen::VectorXd>(d_z12 + elem * (N_SHAPE / 2), N_SHAPE);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const z12(int elem) const
    {
        return Eigen::Map<Eigen::VectorXd>(d_z12 + elem * (N_SHAPE / 2), N_SHAPE);
    }

    // =================================

    __device__ Eigen::Map<Eigen::VectorXi> const offset_start() const
    {
        return Eigen::Map<Eigen::VectorXi>(d_offset_start, N_BEAM);
    }

    __device__ Eigen::Map<Eigen::VectorXi> const offset_end() const
    {
        return Eigen::Map<Eigen::VectorXi>(d_offset_end, N_BEAM);
    }

    __device__ double const L() const
    {
        return *d_L;
    }

    __device__ double const W() const
    {
        return *d_W;
    }

    __device__ double const H() const
    {
        return *d_H;
    }

    __device__ double const rho0() const
    {
        return *d_rho0;
    }

    __device__ double const nu() const
    {
        return *d_nu;
    }

    __device__ double const E() const
    {
        return *d_E;
    }

    //===========================================

    __device__ Eigen::Map<Eigen::MatrixXd> node_values()
    {
        return Eigen::Map<Eigen::MatrixXd>(d_node_values, N_COEF, N_COEF);
    }

    // __device__ double *node_values(int i, int j)
    // {
    //     return &Eigen::Map<Eigen::MatrixXd>(d_node_values, N_COEF, N_COEF)(i, j);
    // }

    __device__ double *node_values(int i, int j)
    {
        return d_node_values + j + i * N_COEF;
    }

#endif

    void Initialize()
    {
        HANDLE_ERROR(cudaMalloc(&d_B_inv, N_SHAPE * N_SHAPE * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_ds_du_pre, N_TOTAL_QP * N_SHAPE * 3 * sizeof(double)));

        HANDLE_ERROR(cudaMalloc(&d_gauss_xi_m, N_QP_6 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_gauss_xi, N_QP_3 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_gauss_eta, N_QP_2 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_gauss_zeta, N_QP_2 * sizeof(double)));

        HANDLE_ERROR(cudaMalloc(&d_weight_xi_m, N_QP_6 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_weight_xi, N_QP_3 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_weight_eta, N_QP_2 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_weight_zeta, N_QP_2 * sizeof(double)));

        HANDLE_ERROR(cudaMalloc(&d_x12_jac, N_COEF * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_y12_jac, N_COEF * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_z12_jac, N_COEF * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_x12, N_COEF * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_y12, N_COEF * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_z12, N_COEF * sizeof(double)));

        HANDLE_ERROR(cudaMalloc(&d_offset_start, N_BEAM * sizeof(int)));
        HANDLE_ERROR(cudaMalloc(&d_offset_end, N_BEAM * sizeof(int)));

        HANDLE_ERROR(cudaMalloc(&d_node_values, N_COEF * N_COEF * sizeof(double)));

        HANDLE_ERROR(cudaMalloc(&d_F, N_BEAM * N_TOTAL_QP * 3 * 3 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_f_elem_out, N_BEAM * 8 * 3 * sizeof(double)));

        // copy struct to device
        HANDLE_ERROR(cudaMalloc(&d_data, sizeof(GPU_ANCF3243_Data)));

        // beam data
        HANDLE_ERROR(cudaMalloc(&d_H, sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_W, sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_L, sizeof(double)));

        HANDLE_ERROR(cudaMalloc(&d_rho0, sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_nu, sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_E, sizeof(double)));
    }

    void Setup(double length,
               double width,
               double height,
               double rho0,
               double nu,
               double E,
               const Eigen::MatrixXd &h_B_inv,
               const Eigen::VectorXd &gauss_xi_m,
               const Eigen::VectorXd &gauss_xi,
               const Eigen::VectorXd &gauss_eta,
               const Eigen::VectorXd &gauss_zeta,
               const Eigen::VectorXd &weight_xi_m,
               const Eigen::VectorXd &weight_xi,
               const Eigen::VectorXd &weight_eta,
               const Eigen::VectorXd &weight_zeta,
               const Eigen::VectorXd &h_x12,
               const Eigen::VectorXd &h_y12,
               const Eigen::VectorXd &h_z12,
               const Eigen::VectorXi &h_offset_start,
               const Eigen::VectorXi &h_offset_end)
    {
        if (is_setup)
        {
            std::cerr << "GPU_ANCF3243_Data is already set up." << std::endl;
            return;
        }

        HANDLE_ERROR(cudaMemcpy(d_B_inv, h_B_inv.data(), N_SHAPE * N_SHAPE * sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(d_gauss_xi_m, gauss_xi_m.data(), N_QP_6 * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_gauss_xi, gauss_xi.data(), N_QP_3 * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_gauss_eta, gauss_eta.data(), N_QP_2 * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_gauss_zeta, gauss_zeta.data(), N_QP_2 * sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(d_weight_xi_m, weight_xi_m.data(), N_QP_6 * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_weight_xi, weight_xi.data(), N_QP_3 * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_weight_eta, weight_eta.data(), N_QP_2 * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_weight_zeta, weight_zeta.data(), N_QP_2 * sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(d_x12_jac, h_x12.data(), N_COEF * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_y12_jac, h_y12.data(), N_COEF * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_z12_jac, h_z12.data(), N_COEF * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_x12, h_x12.data(), N_COEF * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_y12, h_y12.data(), N_COEF * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_z12, h_z12.data(), N_COEF * sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(d_offset_start, h_offset_start.data(), N_BEAM * sizeof(int), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_offset_end, h_offset_end.data(), N_BEAM * sizeof(int), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemset(d_node_values, 0, N_COEF * N_COEF * sizeof(double)));

        cudaMemset(d_f_elem_out, 0, N_BEAM * 8 * 3 * sizeof(double));
        cudaMemset(d_F, 0, N_BEAM * N_TOTAL_QP * 3 * 3 * sizeof(double));

        HANDLE_ERROR(cudaMemcpy(d_H, &height, sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_W, &width, sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_L, &length, sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(d_rho0, &rho0, sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_nu, &nu, sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_E, &E, sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3243_Data), cudaMemcpyHostToDevice));

        is_setup = true;
    }

    // Free memory
    void Destroy()
    {
        HANDLE_ERROR(cudaFree(d_B_inv));
        HANDLE_ERROR(cudaFree(d_ds_du_pre));

        HANDLE_ERROR(cudaFree(d_gauss_xi_m));
        HANDLE_ERROR(cudaFree(d_gauss_xi));
        HANDLE_ERROR(cudaFree(d_gauss_eta));
        HANDLE_ERROR(cudaFree(d_gauss_zeta));
        HANDLE_ERROR(cudaFree(d_weight_xi_m));
        HANDLE_ERROR(cudaFree(d_weight_xi));
        HANDLE_ERROR(cudaFree(d_weight_eta));
        HANDLE_ERROR(cudaFree(d_weight_zeta));

        HANDLE_ERROR(cudaFree(d_x12_jac));
        HANDLE_ERROR(cudaFree(d_y12_jac));
        HANDLE_ERROR(cudaFree(d_z12_jac));
        HANDLE_ERROR(cudaFree(d_x12));
        HANDLE_ERROR(cudaFree(d_y12));
        HANDLE_ERROR(cudaFree(d_z12));

        HANDLE_ERROR(cudaFree(d_offset_start));
        HANDLE_ERROR(cudaFree(d_offset_end));

        HANDLE_ERROR(cudaFree(d_node_values));

        HANDLE_ERROR(cudaFree(d_F));
        HANDLE_ERROR(cudaFree(d_f_elem_out));

        HANDLE_ERROR(cudaFree(d_H));
        HANDLE_ERROR(cudaFree(d_W));
        HANDLE_ERROR(cudaFree(d_L));

        HANDLE_ERROR(cudaFree(d_rho0));
        HANDLE_ERROR(cudaFree(d_nu));
        HANDLE_ERROR(cudaFree(d_E));

        HANDLE_ERROR(cudaFree(d_data));
    }

    // void calc_int_force();

    void calc_ds_du_pre();

    void print_ds_du_pre();

    void print_mass_matrix();

    void calc_mass_matrix();

private:
    double *d_B_inv;
    double *d_ds_du_pre;
    double *d_gauss_xi_m, *d_gauss_xi, *d_gauss_eta, *d_gauss_zeta;
    double *d_weight_xi_m, *d_weight_xi, *d_weight_eta, *d_weight_zeta;

    double *d_x12_jac, *d_y12_jac, *d_z12_jac;
    double *d_x12, *d_y12, *d_z12;

    int *d_offset_start, *d_offset_end;

    double *d_node_values;

    double *d_F, *d_f_elem_out;

    double *d_H, *d_W, *d_L;

    double *d_rho0, *d_nu, *d_E;

    bool is_setup = false;

    GPU_ANCF3243_Data *d_data; // Storing GPU copy of SAPGPUData
};
