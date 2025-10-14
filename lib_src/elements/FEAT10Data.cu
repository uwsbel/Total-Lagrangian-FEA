#include <cooperative_groups.h>

#include <iomanip>

#include "FEAT10Data.cuh"
#include "FEAT10DataFunc.cuh"

namespace cg = cooperative_groups;

__global__ void dn_du_pre_kernel(GPU_FEAT10_Data *d_data) {
  // Get global thread index
  int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate element index and quadrature point index
  int elem_idx = global_thread_idx / Quadrature::N_QP_T10_5;
  int qp_idx   = global_thread_idx % Quadrature::N_QP_T10_5;

  // Bounds check
  if (elem_idx >= d_data->gpu_n_elem() || qp_idx >= Quadrature::N_QP_T10_5) {
    return;
  }

  // Get quadrature point coordinates (xi, eta, zeta)
  double xi   = d_data->tet5pt_x(qp_idx);  // L2 in Python code
  double eta  = d_data->tet5pt_y(qp_idx);  // L3 in Python code
  double zeta = d_data->tet5pt_z(qp_idx);  // L4 in Python code

  // Compute barycentric coordinates
  double L1   = 1.0 - xi - eta - zeta;
  double L2   = xi;
  double L3   = eta;
  double L4   = zeta;
  double L[4] = {L1, L2, L3, L4};

  // Derivatives of barycentric coordinates (dL matrix from Python)
  double dL[4][3] = {
      {-1.0, -1.0, -1.0},  // dL1/dxi, dL1/deta, dL1/dzeta
      {1.0, 0.0, 0.0},     // dL2/dxi, dL2/deta, dL2/dzeta
      {0.0, 1.0, 0.0},     // dL3/dxi, dL3/deta, dL3/dzeta
      {0.0, 0.0, 1.0}      // dL4/dxi, dL4/deta, dL4/dzeta
  };

  // Compute shape function derivatives dN_dxi for all 10 nodes
  double dN_dxi[10][3];

  // Corner nodes (0-3): dN_dxi[i, :] = (4*L[i]-1)*dL[i, :]
  for (int i = 0; i < 4; i++) {
    double factor = 4.0 * L[i] - 1.0;
    for (int j = 0; j < 3; j++) {
      dN_dxi[i][j] = factor * dL[i][j];
    }
  }

  // Edge nodes (4-9): dN_dxi[k, :] = 4*(L[i]*dL[j, :] + L[j]*dL[i, :])
  // Edge connectivity: [(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]
  int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};

  for (int k = 0; k < 6; k++) {
    int i = edges[k][0];
    int j = edges[k][1];

    for (int d = 0; d < 3; d++) {
      dN_dxi[k + 4][d] = 4.0 * (L[i] * dL[j][d] + L[j] * dL[i][d]);
    }
  }

  // Get element node coordinates for this element
  double X_elem[10][3];  // 10 nodes × 3 coordinates
  for (int node = 0; node < 10; node++) {
int global_node_idx = d_data->element_connectivity()(elem_idx, node);
X_elem[node][0] = d_data->x12()(global_node_idx);  // x coordinate
X_elem[node][1] = d_data->y12()(global_node_idx);  // y coordinate
X_elem[node][2] = d_data->z12()(global_node_idx);  // z coordinate
  }

  // Compute Jacobian matrix J = Σ(X_node ⊗ dN_dxi)
  double J[3][3] = {{0.0}};  // Initialize to zero
  for (int a = 0; a < 10; a++) {
    for (int i = 0; i < 3; i++) {    // Node coordinate components
      for (int j = 0; j < 3; j++) {  // Natural coordinate derivatives
        J[i][j] += X_elem[a][i] * dN_dxi[a][j];  // Outer product
      }
    }
  }

  // Compute J^T (transpose)
  double JT[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      JT[i][j] = J[j][i];
    }
  }

  // Solve JT * grad_N = dN_dxi for each shape function
  double grad_N[10][3];
  for (int a = 0; a < 10; a++) {
    // Solve 3×3 system: JT * grad_N[a] = dN_dxi[a]
    // You'll need a 3×3 linear solver here (LU decomposition, Gaussian elimination, etc.)
    solve_3x3_system(JT, dN_dxi[a], grad_N[a]);
  }

  // Store the PHYSICAL gradients in grad_N_ref
  for (int i = 0; i < 10; i++) {
    d_data->grad_N_ref(elem_idx, qp_idx)(i, 0) = grad_N[i][0];  // ∂N_i/∂x
    d_data->grad_N_ref(elem_idx, qp_idx)(i, 1) = grad_N[i][1];  // ∂N_i/∂y
    d_data->grad_N_ref(elem_idx, qp_idx)(i, 2) = grad_N[i][2];  // ∂N_i/∂z
  }

}

void GPU_FEAT10_Data::CalcDnDuPre() {
  int total_threads = n_elem * Quadrature::N_QP_T10_5;

  int threads_per_block = 128;  // or another suitable block size
  int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  dn_du_pre_kernel<<<blocks, threads_per_block>>>(d_data);
  cudaDeviceSynchronize();
}
