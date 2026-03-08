#include "multiitem_collision_kernels.h"

#include <cuda_runtime.h>

namespace {

__global__ void gather_nodes_col_major_kernel(double* d_nodes_col_major,
                                              int n_nodes,
                                              const double* d_x12,
                                              const double* d_y12,
                                              const double* d_z12,
                                              const int* d_coef_idx,
                                              const double* d_z_offset) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_nodes) {
    return;
  }
  const int coef = d_coef_idx[i];
  d_nodes_col_major[i]               = d_x12[coef];
  d_nodes_col_major[n_nodes + i]     = d_y12[coef];
  const double z_off =
      (d_z_offset != nullptr) ? d_z_offset[i] : 0.0;
  d_nodes_col_major[2 * n_nodes + i] = d_z12[coef] + z_off;
}

__global__ void scatter_forces_interleaved_kernel(
    const double* d_f_coll_interleaved, int n_nodes, const int* d_coef_idx,
    double* d_f_ext_interleaved) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_nodes) {
    return;
  }
  const int coef  = d_coef_idx[i];
  const double fx = d_f_coll_interleaved[3 * i + 0];
  const double fy = d_f_coll_interleaved[3 * i + 1];
  const double fz = d_f_coll_interleaved[3 * i + 2];
  atomicAdd(&d_f_ext_interleaved[3 * coef + 0], fx);
  atomicAdd(&d_f_ext_interleaved[3 * coef + 1], fy);
  atomicAdd(&d_f_ext_interleaved[3 * coef + 2], fz);
}

}  // namespace

void GatherCollisionNodesColumnMajor(double* d_nodes_col_major, int n_nodes,
                                     const double* d_x12, const double* d_y12,
                                     const double* d_z12,
                                     const int* d_coef_idx,
                                     const double* d_z_offset) {
  if (d_nodes_col_major == nullptr || d_x12 == nullptr || d_y12 == nullptr ||
      d_z12 == nullptr || d_coef_idx == nullptr || n_nodes <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks      = (n_nodes + threads - 1) / threads;
  gather_nodes_col_major_kernel<<<blocks, threads>>>(
      d_nodes_col_major, n_nodes, d_x12, d_y12, d_z12, d_coef_idx, d_z_offset);
}

void ScatterCollisionForcesToExternal(const double* d_f_coll_interleaved,
                                      int n_nodes, const int* d_coef_idx,
                                      double* d_f_ext_interleaved) {
  if (d_f_coll_interleaved == nullptr || d_coef_idx == nullptr ||
      d_f_ext_interleaved == nullptr || n_nodes <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks      = (n_nodes + threads - 1) / threads;
  scatter_forces_interleaved_kernel<<<blocks, threads>>>(
      d_f_coll_interleaved, n_nodes, d_coef_idx, d_f_ext_interleaved);
}

