#include "prescribed_shake.h"

#include <cuda_runtime.h>

namespace PrescribedShake {

__global__ void offset_nodes_and_targets_kernel(double* d_pos_axis,
                                                double* d_target_axis,
                                                const int* d_node_ids,
                                                int n_node_ids, double delta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_node_ids) {
    return;
  }
  const int node = d_node_ids[i];
  d_pos_axis[node] += delta;
  d_target_axis[node] += delta;
}

void OffsetNodesAndTargets(double* d_pos_axis, double* d_target_axis,
                           const int* d_node_ids, int n_node_ids,
                           double delta) {
  if (d_pos_axis == nullptr || d_target_axis == nullptr || d_node_ids == nullptr ||
      n_node_ids <= 0 || delta == 0.0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks      = (n_node_ids + threads - 1) / threads;
  offset_nodes_and_targets_kernel<<<blocks, threads>>>(d_pos_axis, d_target_axis,
                                                       d_node_ids, n_node_ids,
                                                       delta);
}

}  // namespace PrescribedShake

