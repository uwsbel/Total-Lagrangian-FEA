#include "GPUMemoryManager.cuh"
#include <iomanip>

// Device function: matrix-vector multiply (8x8 * 8x1)
__device__ void mat_vec_mul8(Eigen::Map<Eigen::MatrixXd> A, const double *x, double *out)
{
    for (int i = 0; i < N_SHAPE; ++i)
    {
        out[i] = 0.0;
        for (int j = 0; j < N_SHAPE; ++j)
        {
            out[i] += A(i, j) * x[j];
        }
    }
}

// Device function to compute determinant of 3x3 matrix
__device__ double det3x3(const double *J)
{
    return J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) + J[2] * (J[3] * J[7] - J[4] * J[6]);
}

// Kernel: one thread per quadrature point, computes 8x3 ds_du_pre
__global__ void ds_du_pre_kernel(double L, double W, double H, GPU_ANCF3243_Data *d_data)
{
    int ixi = blockIdx.x;
    int ieta = blockIdx.y;
    int izeta = threadIdx.x;
    int idx = ixi * N_QP_2 * N_QP_2 + ieta * N_QP_2 + izeta;

    double xi = d_data->gauss_xi()(ixi);
    double eta = d_data->gauss_eta()(ieta);
    double zeta = d_data->gauss_zeta()(izeta);

    double u = L * xi / 2.0;
    double v = W * eta / 2.0;
    double w = H * zeta / 2.0;

    double db_du[N_SHAPE] = {0, 1, 0, 0, v, w, 2 * u, 3 * u * u};
    double db_dv[N_SHAPE] = {0, 0, 1, 0, u, 0, 0, 0};
    double db_dw[N_SHAPE] = {0, 0, 0, 1, 0, u, 0, 0};

    double ds_du[N_SHAPE], ds_dv[N_SHAPE], ds_dw[N_SHAPE];
    mat_vec_mul8(d_data->B_inv(), db_du, ds_du);
    mat_vec_mul8(d_data->B_inv(), db_dv, ds_dv);
    mat_vec_mul8(d_data->B_inv(), db_dw, ds_dw);

    // Store as 8x3 matrix: for each i in 0..7, store ds_du, ds_dv, ds_dw as columns
    for (int i = 0; i < N_SHAPE; ++i)
    {
        d_data->ds_du_pre(idx)(i, 0) = ds_du[i];
        d_data->ds_du_pre(idx)(i, 1) = ds_dv[i];
        d_data->ds_du_pre(idx)(i, 2) = ds_dw[i];
    }
}

__device__ void b_vec(double u, double v, double w, double *out)
{
    out[0] = 1.0;
    out[1] = u;
    out[2] = v;
    out[3] = w;
    out[4] = u * v;
    out[5] = u * w;
    out[6] = u * u;
    out[7] = u * u * u;
}

__device__ void b_vec_xi(double xi, double eta, double zeta, double L, double W, double H, double *out)
{
    double u = L * xi / 2.0;
    double v = W * eta / 2.0;
    double w = H * zeta / 2.0;
    b_vec(u, v, w, out);
}

// Device function for Jacobian determinant in normalized coordinates
__device__ void calc_det_J_xi(double xi,
                              double eta,
                              double zeta,
                              Eigen::Map<Eigen::MatrixXd> B_inv,
                              Eigen::Map<Eigen::VectorXd> x12_jac,
                              Eigen::Map<Eigen::VectorXd> y12_jac,
                              Eigen::Map<Eigen::VectorXd> z12_jac,
                              double L,
                              double W,
                              double H,
                              double *J_out)
{
    double db_dxi[N_SHAPE] = {
        0.0, L / 2, 0.0, 0.0, (L * W / 4) * eta, (L * H / 4) * zeta, (L * L / 2) * xi, (3 * L * L * L / 8) * xi * xi};
    double db_deta[N_SHAPE] = {0.0, 0.0, W / 2, 0.0, (L * W / 4) * xi, 0.0, 0.0, 0.0};
    double db_dzeta[N_SHAPE] = {0.0, 0.0, 0.0, H / 2, 0.0, (L * H / 4) * xi, 0.0, 0.0};

    double ds_dxi[N_SHAPE], ds_deta[N_SHAPE], ds_dzeta[N_SHAPE];
    mat_vec_mul8(B_inv, db_dxi, ds_dxi);
    mat_vec_mul8(B_inv, db_deta, ds_deta);
    mat_vec_mul8(B_inv, db_dzeta, ds_dzeta);

    // Nodal matrix: 3 × 8
    // J = N_mat_jac @ np.column_stack([ds_dxi, ds_deta, ds_dzeta])
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            J_out[i * 3 + j] = 0.0;

    for (int i = 0; i < N_SHAPE; ++i)
    {
        J_out[0 * 3 + 0] += x12_jac(i) * ds_dxi[i];
        J_out[1 * 3 + 0] += y12_jac(i) * ds_dxi[i];
        J_out[2 * 3 + 0] += z12_jac(i) * ds_dxi[i];

        J_out[0 * 3 + 1] += x12_jac(i) * ds_deta[i];
        J_out[1 * 3 + 1] += y12_jac(i) * ds_deta[i];
        J_out[2 * 3 + 1] += z12_jac(i) * ds_deta[i];

        J_out[0 * 3 + 2] += x12_jac(i) * ds_dzeta[i];
        J_out[1 * 3 + 2] += y12_jac(i) * ds_dzeta[i];
        J_out[2 * 3 + 2] += z12_jac(i) * ds_dzeta[i];
    }
}

__global__ void mass_matrix_qp_kernel(GPU_ANCF3243_Data *d_data)
{
    int n_qp_per_elem = N_QP_6 * N_QP_2 * N_QP_2;
    int thread_global = blockIdx.x * blockDim.x + threadIdx.x;
    int elem = thread_global / (N_SHAPE * N_SHAPE);
    int item_local = thread_global % (N_SHAPE * N_SHAPE);
    if (elem >= N_BEAM)
        return;

    for (int qp_local = 0; qp_local < n_qp_per_elem; qp_local++)
    {
        // Decode qp_local into (ixi, ieta, izeta)
        int ixi = qp_local / (N_QP_2 * N_QP_2);
        int ieta = (qp_local / N_QP_2) % N_QP_2;
        int izeta = qp_local % N_QP_2;

        double xi = d_data->gauss_xi_m()(ixi);
        double eta = d_data->gauss_eta()(ieta);
        double zeta = d_data->gauss_zeta()(izeta);
        double weight = d_data->weight_xi_m()(ixi) * d_data->weight_eta()(ieta) * d_data->weight_zeta()(izeta);

        // Get element's node offset
        int node_offset = d_data->offset_start()(elem);

        // Get local nodal coordinates for this element
        Eigen::Map<Eigen::VectorXd> x_loc = d_data->x12(elem);
        Eigen::Map<Eigen::VectorXd> y_loc = d_data->y12(elem);
        Eigen::Map<Eigen::VectorXd> z_loc = d_data->z12(elem);

        // Compute shape function at this QP
        double b[8];
        b_vec_xi(xi, eta, zeta, d_data->L(), d_data->W(), d_data->H(), b);
        //b_vec_xi(xi, eta, zeta, 2.0, 1.0, 1.0, b);

        // Compute s = B_inv @ b
        double s[8];
        mat_vec_mul8(d_data->B_inv(), b, s);

        // Compute Jacobian determinant
        double J[9];
        calc_det_J_xi(xi, eta, zeta, d_data->B_inv(), x_loc, y_loc, z_loc, d_data->L(), d_data->W(), d_data->H(), J);
        double detJ = det3x3(J);

        // For each local node, output (global_node, value)
        int i_local = item_local / N_SHAPE;                    // Local node index (0-7)
        int j_local = item_local % N_SHAPE;                    // Local shape function index (0-7)
        int i_global = d_data->offset_start()(elem) + i_local; // Global node index
        int j_global = d_data->offset_start()(elem) + j_local; // Global shape function index

        atomicAdd(d_data->node_values(i_global, j_global), d_data->rho0() * s[i_local] * s[j_local] * weight * detJ);
    }
}

void GPU_ANCF3243_Data::calc_ds_du_pre()
{
    // Launch kernel
    dim3 blocks_pre(N_QP_3, N_QP_2);
    dim3 threads_pre(N_QP_2);
    ds_du_pre_kernel<<<blocks_pre, threads_pre>>>(2.0, 1.0, 1.0, d_data);
    cudaDeviceSynchronize();
}

void GPU_ANCF3243_Data::print_ds_du_pre()
{
    // Allocate host memory for all quadrature points
    const int total_size = N_TOTAL_QP * N_SHAPE * 3;
    double *h_ds_du_pre_raw = new double[total_size];

    // Copy from device to host
    HANDLE_ERROR(cudaMemcpy(h_ds_du_pre_raw, d_ds_du_pre, total_size * sizeof(double), cudaMemcpyDeviceToHost));

    // Print each quadrature point's matrix
    for (int qp = 0; qp < N_TOTAL_QP; ++qp)
    {
        std::cout << "\n=== Quadrature Point " << qp << " ===" << std::endl;

        // Create Eigen::Map for this quadrature point's data
        double *qp_data = h_ds_du_pre_raw + qp * N_SHAPE * 3;
        Eigen::Map<Eigen::MatrixXd> ds_du_matrix(qp_data, N_SHAPE, 3);

        // Print the 8x3 matrix with column headers
        std::cout << "        ds/du       ds/dv       ds/dw" << std::endl;
        for (int i = 0; i < N_SHAPE; ++i)
        {
            std::cout << "Node " << i << ": ";
            for (int j = 0; j < 3; ++j)
            {
                std::cout << std::setw(10) << std::fixed << std::setprecision(6) << ds_du_matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    delete[] h_ds_du_pre_raw;
}

void GPU_ANCF3243_Data::calc_mass_matrix()
{
    // Mass terms computation
    const int N_QP = N_QP_6 * N_QP_2 * N_QP_2;
    const int N_OUT = N_BEAM * N_SHAPE * N_SHAPE;

    // Launch kernel
    int threads = 128;
    int blocks = (N_OUT + threads - 1) / threads;
    mass_matrix_qp_kernel<<<blocks, threads>>>(d_data);

    cudaDeviceSynchronize();
}

void GPU_ANCF3243_Data::print_mass_matrix()
{
    // Allocate host memory for all quadrature points
    const int total_size = N_COEF * N_COEF;
    double *h_node_values_raw = new double[total_size];

    // Copy from device to host
    HANDLE_ERROR(cudaMemcpy(h_node_values_raw, d_node_values, total_size * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "mass matrix:" << std::endl;

    // Print each quadrature point's matrix
    for (int i = 0; i < N_COEF; ++i)
    {
            for (int j = 0; j < N_COEF; ++j)
            {
                std::cout << std::setw(10) << std::fixed << std::setprecision(6) << h_node_values_raw[j * N_COEF + i] << " ";
            }
            std::cout << std::endl;
    }

    delete[] h_node_values_raw;
}

// void GPU_ANCF3243_Data::calc_int_force() {
//     threads = 128;
//     blocks = (N_QP_3 * N_QP_2 * N_QP_2 * N_BEAM + threads - 1) / threads;

//     compute_internal_force_kernel<<<blocks, threads>>>(
//         d_B_inv, d_ds_du_pre, d_x12_jac, d_y12_jac, d_z12_jac, d_x12, d_y12, d_z12, d_offset_start, d_offset_end,
//         N_BEAM, N_SHAPE, total_qp, d_F, d_weight_xi, d_weight_eta, d_weight_zeta, L, W, H, mu, lam_param,
//         f_elem_out);

//     cudaDeviceSynchronize();
// }