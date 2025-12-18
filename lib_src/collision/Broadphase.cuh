/* Broadphase.cuh
 * Author: Json Zhou (zzhou292@wisc.edu)
 *
 * GPU broadphase collision detection: AABBs, sweep-and-prune, neighbor
 * filtering, and host/device data for mesh overlap queries.
 */

#include <cuda_runtime.h>
#include <cusparse.h>

#include <Eigen/Dense>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/quadrature_utils.h"

#include "CollisionTypes.cuh"

namespace ANCFCPUUtils {
class MeshManager;
}

// Definition of GPU_ANCF3243 and data access device functions
#pragma once

struct AABB {
  double3 min;
  double3 max;
  int objectId;
};

// Hash function for pair (for CPU unordered_set)
struct PairHash {
  std::size_t operator()(const std::pair<int, int>& p) const {
    return std::hash<long long>()(((long long)p.first << 32) | p.second);
  }
};

struct Broadphase {
#if defined(__CUDACC__)
  // Device getters for node coordinates (column-major indexing)
  __device__ double node_x(int node_id) const {
    return d_nodes[node_id];
  }

  __device__ double node_y(int node_id) const {
    return d_nodes[node_id + n_nodes];
  }

  __device__ double node_z(int node_id) const {
    return d_nodes[node_id + 2 * n_nodes];
  }

  // Device getter for element node IDs (column-major indexing)
  __device__ int element_node(int elem_id, int local_node_idx) const {
    return d_elements[elem_id + local_node_idx * n_elems];
  }
#endif

  // Host data
  std::vector<AABB> h_aabbs;
  std::vector<CollisionPair> h_collisionPairs;
  int numObjects;
  int numCollisions;

  // Mesh data (host)
  Eigen::MatrixXd h_nodes;
  Eigen::MatrixXi h_elements;
  int n_nodes;
  int n_elems;

  // Neighbor tracking (host)
  std::unordered_set<std::pair<int, int>, PairHash> h_neighborPairs;

  // Device data
  AABB* d_aabbs;

  // Device mesh data
  double* d_nodes;
  int* d_elements;
  int nodesPerElement;

  // Element-to-mesh mapping (device). Used to optionally filter out same-mesh
  // (self) collisions efficiently in broadphase.
  int* d_elementMeshIds;

  // If false, collision pairs with elementMeshIds[a] == elementMeshIds[b] are
  // skipped in broadphase.
  bool enableSelfCollision;

  // Host-side copy of element mesh IDs (optional, used for diagnostics).
  std::vector<int> h_elementMeshIds;

  // Device pointer to this struct
  Broadphase* d_bp;

  // Sorting data for sweep and prune
  double* d_sortKeys;    // Sort keys (e.g., min.x values)
  int* d_sortIndices;    // Original indices
  double* d_sortedKeys;  // Sorted keys (output)
  int* d_sortedIndices;  // Sorted indices (output)
  AABB* d_sortedAABBs;   // Sorted AABBs
  void* d_tempStorage;   // Temporary storage for CUB
  size_t tempStorageBytes;

  // Collision detection data
  CollisionPair* d_collisionPairs;  // Device collision pairs

  int* d_sameMeshPairsCount;

  // Neighbor filter data (compact representation for GPU)
  long long* d_neighborPairHashes;  // Sorted array of hashed pairs
  int numNeighborPairs;

  // Constructor
  Broadphase();

  // Destructor
  ~Broadphase();

  // Initialize GPU resources with mesh data
  void Initialize(const Eigen::MatrixXd& nodes,
                  const Eigen::MatrixXi& elements,
                  const Eigen::VectorXi& elementMeshIds = Eigen::VectorXi());

  // Convenience overload: build element-to-mesh mapping from MeshManager.
  void Initialize(const ANCFCPUUtils::MeshManager& mesh_manager);

  int GetElementMeshIdHost(int elem_id) const {
    if (elem_id < 0 || elem_id >= static_cast<int>(h_elementMeshIds.size())) {
      return 0;
    }
    return h_elementMeshIds[elem_id];
  }

  void EnableSelfCollision(bool enable);

  // Update only nodal positions on the device while reusing existing
  // topology, neighbor maps, and allocated buffers.
  // - Expect same number of nodes and 3 columns.
  // - Does NOT rebuild neighbor maps or reallocate device memory.
  void UpdateNodes(const Eigen::MatrixXd& nodes);

  void RetrieveAABBandPrints();

  // Destroy/cleanup GPU resources
  void Destroy();

  // Create/update AABBs from mesh data
  void CreateAABB();

  // Sort AABBs along specified axis (0=x, 1=y, 2=z)
  void SortAABBs(int axis = 0);

  // Print sorted AABBs for verification
  void PrintSortedAABBs(int axis = 0);

  // Build neighbor connectivity map
  void BuildNeighborMap();

  // Detect collisions using sweep and prune (with neighbor filtering)
  void DetectCollisions(bool copyPairsToHost = false);

  int CountSameMeshPairsDevice() const;

  const CollisionPair* GetCollisionPairsDevicePtr() const {
    return d_collisionPairs;
  }

  // Print collision pairs
  void PrintCollisionPairs();
};
