#pragma once

/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    BroadphaseFunc.cuh
 * Brief:   CUDA device and kernel functions used by the Broadphase module:
 *          element AABB construction, extraction of sort keys, AABB
 *          reordering, neighbor filtering by hashed pairs, and generation of
 *          non-neighbor collision pairs on the GPU.
 *==============================================================
 *==============================================================*/

#include "Broadphase.cuh"

// Device / kernel functions for Broadphase

// Kernel to compute AABB for each element
__global__ void computeAABBKernel(Broadphase* bp, AABB* aabbs, int n_elems) {
  int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (elem_idx >= n_elems)
    return;

  // Get first node ID using element getter
  int first_node_id = bp->element_node(elem_idx, 0);

  // Access node coordinates using getters - use double precision
  double3 min_pt =
      make_double3(bp->node_x(first_node_id), bp->node_y(first_node_id),
                   bp->node_z(first_node_id));
  double3 max_pt = min_pt;

  // Iterate through all nodes of the element
  for (int i = 1; i < bp->nodesPerElement; i++) {
    int node_id = bp->element_node(elem_idx, i);

    double x = bp->node_x(node_id);
    double y = bp->node_y(node_id);
    double z = bp->node_z(node_id);

    min_pt.x = fmin(min_pt.x, x);
    min_pt.y = fmin(min_pt.y, y);
    min_pt.z = fmin(min_pt.z, z);

    max_pt.x = fmax(max_pt.x, x);
    max_pt.y = fmax(max_pt.y, y);
    max_pt.z = fmax(max_pt.z, z);
  }

  // Store AABB
  aabbs[elem_idx].min      = min_pt;
  aabbs[elem_idx].max      = max_pt;
  aabbs[elem_idx].objectId = elem_idx;
}

__global__ void countSameMeshPairsKernel(const CollisionPair* pairs,
                                         int numPairs,
                                         const int* elementMeshIds,
                                         int* outCount) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numPairs) {
    return;
  }

  int a = pairs[idx].idA;
  int b = pairs[idx].idB;
  if (a < 0 || b < 0) {
    return;
  }

  if (elementMeshIds[a] == elementMeshIds[b]) {
    atomicAdd(outCount, 1);
  }
}

// Kernel to extract sort keys from AABBs
__global__ void extractSortKeysKernel(const AABB* aabbs, double* keys,
                                      int* indices, int axis, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Extract min value for the specified axis
    if (axis == 0) {
      keys[idx] = aabbs[idx].min.x;
    } else if (axis == 1) {
      keys[idx] = aabbs[idx].min.y;
    } else {
      keys[idx] = aabbs[idx].min.z;
    }
    indices[idx] = idx;
  }
}

// Kernel to reorder AABBs based on sorted indices
__global__ void reorderAABBsKernel(const AABB* input, AABB* output,
                                   const int* indices, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[indices[idx]];
  }
}

// Device function: binary search in sorted hash array
__device__ bool isNeighborPair(int idA, int idB,
                               const long long* neighborHashes, int numHashes) {
  // Handle case with no neighbor data
  if (numHashes == 0 || neighborHashes == nullptr)
    return false;

  // Ensure idA < idB for consistent hashing
  if (idA > idB) {
    int temp = idA;
    idA      = idB;
    idB      = temp;
  }

  long long hash = ((long long)idA << 32) | idB;

  // Binary search
  int left = 0, right = numHashes - 1;
  while (left <= right) {
    int mid = (left + right) / 2;
    if (neighborHashes[mid] == hash)
      return true;
    if (neighborHashes[mid] < hash)
      left = mid + 1;
    else
      right = mid - 1;
  }
  return false;
}

// Kernel to count potential collisions per element (with neighbor filtering)
__global__ void countCollisionsKernel(const AABB* sortedAABBs,
                                      int* collisionCounts, int n,
                                      const long long* neighborHashes,
                                      int numHashes,
                                      const int* elementMeshIds,
                                      int enableSelfCollision) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  const AABB& Ai = sortedAABBs[i];
  int count      = 0;

  for (int j = i + 1; j < n; ++j) {
    const AABB& Aj = sortedAABBs[j];

    if (Aj.min.x > Ai.max.x)
      break;

    bool overlapX = (Ai.min.x <= Aj.max.x && Aj.min.x <= Ai.max.x);
    bool overlapY = (Ai.min.y <= Aj.max.y && Aj.min.y <= Ai.max.y);
    bool overlapZ = (Ai.min.z <= Aj.max.z && Aj.min.z <= Ai.max.z);

    if (overlapX && overlapY && overlapZ) {
      if (!enableSelfCollision && elementMeshIds != nullptr) {
        int meshIdA = elementMeshIds[Ai.objectId];
        int meshIdB = elementMeshIds[Aj.objectId];
        if (meshIdA == meshIdB) {
          continue;
        }
      }
      // Check if they are neighbors - skip if true
      if (!isNeighborPair(Ai.objectId, Aj.objectId, neighborHashes,
                          numHashes)) {
        count++;
      }
    }
  }

  collisionCounts[i] = count;
}

// Kernel to generate collision pairs (with neighbor filtering)
__global__ void generateCollisionPairsKernel(const AABB* sortedAABBs,
                                             const int* collisionOffsets,
                                             CollisionPair* collisionPairs,
                                             int n,
                                             const long long* neighborHashes,
                                             int numHashes,
                                             const int* elementMeshIds,
                                             int enableSelfCollision) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  const AABB& Ai = sortedAABBs[i];
  int writeIdx   = collisionOffsets[i];

  for (int j = i + 1; j < n; ++j) {
    const AABB& Aj = sortedAABBs[j];

    if (Aj.min.x > Ai.max.x)
      break;

    bool overlapX = (Ai.min.x <= Aj.max.x && Aj.min.x <= Ai.max.x);
    bool overlapY = (Ai.min.y <= Aj.max.y && Aj.min.y <= Ai.max.y);
    bool overlapZ = (Ai.min.z <= Aj.max.z && Aj.min.z <= Ai.max.z);

    if (overlapX && overlapY && overlapZ) {
      if (!enableSelfCollision && elementMeshIds != nullptr) {
        int meshIdA = elementMeshIds[Ai.objectId];
        int meshIdB = elementMeshIds[Aj.objectId];
        if (meshIdA == meshIdB) {
          continue;
        }
      }
      // Check if they are neighbors - skip if true
      if (!isNeighborPair(Ai.objectId, Aj.objectId, neighborHashes,
                          numHashes)) {
        collisionPairs[writeIdx++] = CollisionPair(Ai.objectId, Aj.objectId);
      }
    }
  }
}
