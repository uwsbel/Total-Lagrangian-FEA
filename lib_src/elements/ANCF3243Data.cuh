// Corrected Include Order

// STEP 1: Standard C++ and CUDA headers go FIRST.
// Example:
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include <Eigen/Dense>

// Add this include at the top:
#include "../../lib_utils/quadrature_utils.h"

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

//
// define a SAP data strucutre
struct GPU_ANCF3243_Data
{
#if defined(__CUDACC__)

    // Const get functions
    __device__ const Eigen::Map<Eigen::MatrixXd> B_inv() const
    {
        int row_size = Quadrature::N_SHAPE;
        int col_size = Quadrature::N_SHAPE;
        return Eigen::Map<Eigen::MatrixXd>(d_B_inv, row_size, col_size);
    }

    __device__ Eigen::Map<Eigen::MatrixXd> ds_du_pre(int qp_idx) const
    {
        int row_size = Quadrature::N_SHAPE;
        int col_size = 3;
        double *qp_data = d_ds_du_pre + qp_idx * Quadrature::N_SHAPE * 3;
        return Eigen::Map<Eigen::MatrixXd>(qp_data, row_size, col_size);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> gauss_xi_m() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_gauss_xi_m, Quadrature::N_QP_6);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> gauss_xi() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_gauss_xi, Quadrature::N_QP_3);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> gauss_eta() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_gauss_eta, Quadrature::N_QP_2);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> gauss_zeta() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_gauss_zeta, Quadrature::N_QP_2);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> weight_xi_m() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_weight_xi_m, Quadrature::N_QP_6);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> weight_xi() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_weight_xi, Quadrature::N_QP_3);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> weight_eta() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_weight_eta, Quadrature::N_QP_2);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> weight_zeta() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_weight_zeta, Quadrature::N_QP_2);
    }

    __device__ Eigen::Map<Eigen::VectorXd> x12_jac()
    {
        return Eigen::Map<Eigen::VectorXd>(d_x12_jac, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const x12_jac() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_x12_jac, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXd> y12_jac()
    {
        return Eigen::Map<Eigen::VectorXd>(d_y12_jac, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const y12_jac() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_y12_jac, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXd> z12_jac()
    {
        return Eigen::Map<Eigen::VectorXd>(d_z12_jac, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const z12_jac() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_z12_jac, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXd> x12()
    {
        return Eigen::Map<Eigen::VectorXd>(d_x12, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const x12() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_x12, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXd> y12()
    {
        return Eigen::Map<Eigen::VectorXd>(d_y12, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const y12() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_y12, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXd> z12()
    {
        return Eigen::Map<Eigen::VectorXd>(d_z12, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const z12() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_z12, n_coef);
    }

    __device__ Eigen::Map<Eigen::VectorXd> x12(int elem)
    {
        return Eigen::Map<Eigen::VectorXd>(d_x12 + elem * (Quadrature::N_SHAPE / 2), Quadrature::N_SHAPE);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const x12(int elem) const
    {
        return Eigen::Map<Eigen::VectorXd>(d_x12 + elem * (Quadrature::N_SHAPE / 2), Quadrature::N_SHAPE);
    }

    __device__ Eigen::Map<Eigen::VectorXd> y12(int elem)
    {
        return Eigen::Map<Eigen::VectorXd>(d_y12 + elem * (Quadrature::N_SHAPE / 2), Quadrature::N_SHAPE);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const y12(int elem) const
    {
        return Eigen::Map<Eigen::VectorXd>(d_y12 + elem * (Quadrature::N_SHAPE / 2), Quadrature::N_SHAPE);
    }

    __device__ Eigen::Map<Eigen::VectorXd> z12(int elem)
    {
        return Eigen::Map<Eigen::VectorXd>(d_z12 + elem * (Quadrature::N_SHAPE / 2), Quadrature::N_SHAPE);
    }

    __device__ Eigen::Map<Eigen::VectorXd> const z12(int elem) const
    {
        return Eigen::Map<Eigen::VectorXd>(d_z12 + elem * (Quadrature::N_SHAPE / 2), Quadrature::N_SHAPE);
    }

    __device__ Eigen::Map<Eigen::MatrixXd> F(int elem_idx, int qp_idx)
    {
        return Eigen::Map<Eigen::MatrixXd>(d_F + (elem_idx * Quadrature::N_TOTAL_QP + qp_idx) * 9, 3, 3);
    }

    __device__ const Eigen::Map<Eigen::MatrixXd> F(int elem_idx, int qp_idx) const
    {
        return Eigen::Map<Eigen::MatrixXd>(d_F + (elem_idx * Quadrature::N_TOTAL_QP + qp_idx) * 9, 3, 3);
    }

    __device__ Eigen::Map<Eigen::MatrixXd> P(int elem_idx, int qp_idx)
    {
        return Eigen::Map<Eigen::MatrixXd>(d_P + (elem_idx * Quadrature::N_TOTAL_QP + qp_idx) * 9, 3, 3);
    }

    __device__ const Eigen::Map<Eigen::MatrixXd> P(int elem_idx, int qp_idx) const
    {
        return Eigen::Map<Eigen::MatrixXd>(d_P + (elem_idx * Quadrature::N_TOTAL_QP + qp_idx) * 9, 3, 3);
    }

    __device__ Eigen::Map<Eigen::VectorXd> f_elem_out(int global_node_idx)
    {
        return Eigen::Map<Eigen::VectorXd>(d_f_elem_out + global_node_idx * 3, 3);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> f_elem_out(int global_node_idx) const
    {
        return Eigen::Map<Eigen::VectorXd>(d_f_elem_out + global_node_idx * 3, 3);
    }

    __device__ Eigen::Map<Eigen::VectorXd> f_elem_out()
    {
        return Eigen::Map<Eigen::VectorXd>(d_f_elem_out, n_coef * 3);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> f_elem_out() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_f_elem_out, n_coef * 3);
    }

    __device__ Eigen::Map<Eigen::VectorXd> constraint()
    {
        return Eigen::Map<Eigen::VectorXd>(d_constraint, 12);
    }

    __device__ const Eigen::Map<Eigen::VectorXd> constraint() const
    {
        return Eigen::Map<Eigen::VectorXd>(d_constraint, 12);
    }

    __device__ Eigen::Map<Eigen::MatrixXd> constraint_jac()
    {
        return Eigen::Map<Eigen::MatrixXd>(d_constraint_jac, 12, n_coef * 3);
    }

    __device__ const Eigen::Map<Eigen::MatrixXd> constraint_jac() const
    {
        return Eigen::Map<Eigen::MatrixXd>(d_constraint_jac, 12, n_coef * 3);
    }

    // ================================

    __device__ Eigen::Map<Eigen::VectorXi> const offset_start() const
    {
        return Eigen::Map<Eigen::VectorXi>(d_offset_start, n_beam);
    }

    __device__ Eigen::Map<Eigen::VectorXi> const offset_end() const
    {
        return Eigen::Map<Eigen::VectorXi>(d_offset_end, n_beam);
    }

    __device__ double L() const
    {
        return *d_L;
    }

    __device__ double W() const
    {
        return *d_W;
    }

    __device__ double H() const
    {
        return *d_H;
    }

    __device__ double rho0() const
    {
        return *d_rho0;
    }

    __device__ double nu() const
    {
        return *d_nu;
    }

    __device__ double E() const
    {
        return *d_E;
    }

    __device__ double lambda() const
    {
        return *d_lambda;
    }

    __device__ double mu() const
    {
        return *d_mu;
    }

    //===========================================

    __device__ Eigen::Map<Eigen::MatrixXd> node_values()
    {
        return Eigen::Map<Eigen::MatrixXd>(d_node_values, n_coef, n_coef);
    }

    __device__ double *node_values(int i, int j)
    {
        return d_node_values + j + i * n_coef;
    }

    __host__ __device__ int get_n_beam() const { return n_beam; }
    __host__ __device__ int get_n_coef() const { return n_coef; }
#else
    int get_n_beam() const { return n_beam; }
    int get_n_coef() const { return n_coef; }
#endif

    // Constructor
    GPU_ANCF3243_Data(int num_beams) : n_beam(num_beams)
    {
        n_coef = Quadrature::N_SHAPE + 4 * (n_beam - 1);
    }

    void Initialize()
    {
        HANDLE_ERROR(cudaMalloc(&d_B_inv, Quadrature::N_SHAPE * Quadrature::N_SHAPE * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_ds_du_pre, Quadrature::N_TOTAL_QP * Quadrature::N_SHAPE * 3 * sizeof(double)));

        HANDLE_ERROR(cudaMalloc(&d_gauss_xi_m, Quadrature::N_QP_6 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_gauss_xi, Quadrature::N_QP_3 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_gauss_eta, Quadrature::N_QP_2 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_gauss_zeta, Quadrature::N_QP_2 * sizeof(double)));

        HANDLE_ERROR(cudaMalloc(&d_weight_xi_m, Quadrature::N_QP_6 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_weight_xi, Quadrature::N_QP_3 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_weight_eta, Quadrature::N_QP_2 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_weight_zeta, Quadrature::N_QP_2 * sizeof(double)));

        HANDLE_ERROR(cudaMalloc(&d_x12_jac, n_coef * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_y12_jac, n_coef * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_z12_jac, n_coef * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_x12, n_coef * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_y12, n_coef * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_z12, n_coef * sizeof(double)));

        HANDLE_ERROR(cudaMalloc(&d_offset_start, n_beam * sizeof(int)));
        HANDLE_ERROR(cudaMalloc(&d_offset_end, n_beam * sizeof(int)));

        HANDLE_ERROR(cudaMalloc(&d_node_values, n_coef * n_coef * sizeof(double)));

        HANDLE_ERROR(cudaMalloc(&d_F, n_beam * Quadrature::N_TOTAL_QP * 3 * 3 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_P, n_beam * Quadrature::N_TOTAL_QP * 3 * 3 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_f_elem_out, n_coef * 3 * sizeof(double)));

        // copy struct to device
        HANDLE_ERROR(cudaMalloc(&d_data, sizeof(GPU_ANCF3243_Data)));

        // beam data
        HANDLE_ERROR(cudaMalloc(&d_H, sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_W, sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_L, sizeof(double)));

        HANDLE_ERROR(cudaMalloc(&d_rho0, sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_nu, sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_E, sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_lambda, sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_mu, sizeof(double)));

        // constraint data
        HANDLE_ERROR(cudaMalloc(&d_constraint, 12 * sizeof(double)));
        HANDLE_ERROR(cudaMalloc(&d_constraint_jac, 12 * (n_coef * 3) * sizeof(double)));
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

        HANDLE_ERROR(cudaMemcpy(d_B_inv, h_B_inv.data(), Quadrature::N_SHAPE * Quadrature::N_SHAPE * sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(d_gauss_xi_m, gauss_xi_m.data(), Quadrature::N_QP_6 * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_gauss_xi, gauss_xi.data(), Quadrature::N_QP_3 * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_gauss_eta, gauss_eta.data(), Quadrature::N_QP_2 * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_gauss_zeta, gauss_zeta.data(), Quadrature::N_QP_2 * sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(d_weight_xi_m, weight_xi_m.data(), Quadrature::N_QP_6 * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_weight_xi, weight_xi.data(), Quadrature::N_QP_3 * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_weight_eta, weight_eta.data(), Quadrature::N_QP_2 * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_weight_zeta, weight_zeta.data(), Quadrature::N_QP_2 * sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(d_x12_jac, h_x12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_y12_jac, h_y12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_z12_jac, h_z12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_x12, h_x12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_y12, h_y12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_z12, h_z12.data(), n_coef * sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(d_offset_start, h_offset_start.data(), n_beam * sizeof(int), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_offset_end, h_offset_end.data(), n_beam * sizeof(int), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemset(d_node_values, 0, n_coef * n_coef * sizeof(double)));

        cudaMemset(d_f_elem_out, 0, n_coef * 3 * sizeof(double));
        cudaMemset(d_F, 0, n_beam * Quadrature::N_TOTAL_QP * 3 * 3 * sizeof(double));
        cudaMemset(d_P, 0, n_beam * Quadrature::N_TOTAL_QP * 3 * 3 * sizeof(double));

        HANDLE_ERROR(cudaMemcpy(d_H, &height, sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_W, &width, sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_L, &length, sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(d_rho0, &rho0, sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_nu, &nu, sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_E, &E, sizeof(double), cudaMemcpyHostToDevice));
        double mu = E / (2 * (1 + nu)); // Shear modulus μ
        double lambda =
            (E * nu) / ((1 + nu) * (1 - 2 * nu)); // Lamé’s first parameter λ
        HANDLE_ERROR(cudaMemcpy(d_mu, &mu, sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_lambda, &lambda, sizeof(double), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3243_Data), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemset(d_constraint, 0, 12 * sizeof(double)));
        HANDLE_ERROR(cudaMemset(d_constraint_jac, 0, 12 * (n_coef * 3) * sizeof(double)));

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
        HANDLE_ERROR(cudaFree(d_P));
        HANDLE_ERROR(cudaFree(d_f_elem_out));

        HANDLE_ERROR(cudaFree(d_H));
        HANDLE_ERROR(cudaFree(d_W));
        HANDLE_ERROR(cudaFree(d_L));

        HANDLE_ERROR(cudaFree(d_rho0));
        HANDLE_ERROR(cudaFree(d_nu));
        HANDLE_ERROR(cudaFree(d_E));
        HANDLE_ERROR(cudaFree(d_lambda));
        HANDLE_ERROR(cudaFree(d_mu));

        HANDLE_ERROR(cudaFree(d_data));

        HANDLE_ERROR(cudaFree(d_constraint));
        HANDLE_ERROR(cudaFree(d_constraint_jac));
    }

    void CalcDsDuPre();

    void CalcMassMatrix();

    void CalcDeformationGradient();

    void CalcPFromF();

    void CalcInternalForce();

    void CalcConstraintData();

    void PrintDsDuPre();

    void RetrieveMassMatrixToCPU(Eigen::MatrixXd &mass_matrix);

    void RetrieveDeformationGradientToCPU(std::vector<std::vector<Eigen::MatrixXd>> &deformation_gradient);

    void RetrievePFromFToCPU(std::vector<std::vector<Eigen::MatrixXd>> &p_from_F);

    void RetrieveInternalForceToCPU(Eigen::VectorXd &internal_force);

    void RetrieveConstraintDataToCPU(Eigen::VectorXd &constraint);

    void RetrieveConstraintJacobianToCPU(Eigen::MatrixXd &constraint_jac);

    void RetrievePositionToCPU(Eigen::VectorXd &x12, Eigen::VectorXd &y12, Eigen::VectorXd &z12);

    GPU_ANCF3243_Data *d_data; // Storing GPU copy of SAPGPUData

private:
    double *d_B_inv;
    double *d_ds_du_pre;
    double *d_gauss_xi_m, *d_gauss_xi, *d_gauss_eta, *d_gauss_zeta;
    double *d_weight_xi_m, *d_weight_xi, *d_weight_eta, *d_weight_zeta;

    double *d_x12_jac, *d_y12_jac, *d_z12_jac;
    double *d_x12, *d_y12, *d_z12;

    int *d_offset_start, *d_offset_end;

    double *d_node_values;

    double *d_F, *d_P, *d_f_elem_out;

    double *d_H, *d_W, *d_L;

    double *d_rho0, *d_nu, *d_E, *d_lambda, *d_mu;

    double *d_constraint, *d_constraint_jac;

    bool is_setup = false;

    int n_beam;
    int n_coef;
};
