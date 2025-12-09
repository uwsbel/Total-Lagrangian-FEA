#include <cuda_runtime.h>
#include <cusparse.h>

#include <Eigen/Dense>
#include <iostream>
#include <unordered_set>
#include <vector>
#include "../../lib_utils/cuda_utils.h"
#include "../../lib_utils/quadrature_utils.h"

// Definition of GPU_ANCF3243 and data access device functions
#pragma once

struct AABB {
  double3 min;
  double3 max;
  int objectId;
};

// Collision pair struct
struct CollisionPair {
  int idA;
  int idB;

  __host__ __device__ CollisionPair() : idA(-1), idB(-1) {}
  __host__ __device__ CollisionPair(int a, int b) : idA(a), idB(b) {}

  // Check if pair is valid (not sentinel)
  __host__ __device__ bool isValid() const {
    return idA >= 0 && idB >= 0;
  }
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

  // Device pointer to this struct
  Broadphase* d_bp;

  // Sorting data for sweep and prune
  double* d_sortKeys;     // Sort keys (e.g., min.x values)
  int* d_sortIndices;    // Original indices
  double* d_sortedKeys;   // Sorted keys (output)
  int* d_sortedIndices;  // Sorted indices (output)
  AABB* d_sortedAABBs;   // Sorted AABBs
  void* d_tempStorage;   // Temporary storage for CUB
  size_t tempStorageBytes;

  // Collision detection data
  CollisionPair* d_collisionPairs;  // Device collision pairs

  // Neighbor filter data (compact representation for GPU)
  long long* d_neighborPairHashes;  // Sorted array of hashed pairs
  int numNeighborPairs;

  // Constructor
  Broadphase();

  // Destructor
  ~Broadphase();

  // Initialize GPU resources with mesh data
  void Initialize(const Eigen::MatrixXd& nodes,
                  const Eigen::MatrixXi& elements);

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
  void DetectCollisions();

  // Print collision pairs
  void PrintCollisionPairs();
};
