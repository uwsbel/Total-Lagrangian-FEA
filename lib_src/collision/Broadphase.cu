#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cub/cub.cuh>  // Include CUB only in .cu file

#include "Broadphase.cuh"
#include "BroadphaseFunc.cuh"

// Constructor
Broadphase::Broadphase()
    : numObjects(0),
      numCollisions(0),
      d_aabbs(nullptr),
      d_nodes(nullptr),
      d_elements(nullptr),
      d_bp(nullptr),
      d_sortKeys(nullptr),
      d_sortIndices(nullptr),
      d_sortedKeys(nullptr),
      d_sortedIndices(nullptr),
      d_sortedAABBs(nullptr),
      d_tempStorage(nullptr),
      tempStorageBytes(0),
      d_collisionPairs(nullptr),
      d_neighborPairHashes(nullptr),
      numNeighborPairs(0),
      n_nodes(0),
      n_elems(0),
      nodesPerElement(0) {}

// Destructor
Broadphase::~Broadphase() {
  Destroy();
}

// Initialize GPU resources with mesh data
void Broadphase::Initialize(const Eigen::MatrixXd& nodes,
                            const Eigen::MatrixXi& elements) {
  // Store mesh dimensions
  n_nodes         = nodes.rows();
  n_elems         = elements.rows();
  nodesPerElement = elements.cols();

  // Store host mesh data
  h_nodes    = nodes;
  h_elements = elements;

  // Use n_elems as the actual number of objects
  numObjects = n_elems;

  // Allocate device memory for AABBs (one per element)
  cudaMalloc(&d_aabbs, n_elems * sizeof(AABB));

  // Allocate sorting arrays (input and output buffers)
  cudaMalloc(&d_sortKeys, n_elems * sizeof(double));
  cudaMalloc(&d_sortIndices, n_elems * sizeof(int));
  cudaMalloc(&d_sortedKeys, n_elems * sizeof(double));
  cudaMalloc(&d_sortedIndices, n_elems * sizeof(int));
  cudaMalloc(&d_sortedAABBs, n_elems * sizeof(AABB));

  // Allocate temporary storage for CUB sorting
  d_tempStorage = nullptr;
  cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageBytes, d_sortKeys,
                                  d_sortedKeys, d_sortIndices, d_sortedIndices,
                                  n_elems);
  cudaMalloc(&d_tempStorage, tempStorageBytes);

  // Allocate and copy mesh data to device
  // Nodes: n_nodes x 3 (column-major)
  cudaMalloc(&d_nodes, n_nodes * 3 * sizeof(double));
  cudaMemcpy(d_nodes, nodes.data(), n_nodes * 3 * sizeof(double),
             cudaMemcpyHostToDevice);

  // Elements: n_elems x nodesPerElement (column-major)
  cudaMalloc(&d_elements, n_elems * nodesPerElement * sizeof(int));
  cudaMemcpy(d_elements, elements.data(),
             n_elems * nodesPerElement * sizeof(int), cudaMemcpyHostToDevice);

  // Allocate device copy of this struct and copy to device
  cudaMalloc(&d_bp, sizeof(Broadphase));
  cudaMemcpy(d_bp, this, sizeof(Broadphase), cudaMemcpyHostToDevice);

  std::cout << "Broadphase initialized with " << n_nodes << " nodes and "
            << n_elems << " elements" << std::endl;
}

// Destroy/cleanup GPU resources
void Broadphase::Destroy() {
  if (d_aabbs) {
    cudaFree(d_aabbs);
    d_aabbs = nullptr;
  }

  if (d_sortKeys) {
    cudaFree(d_sortKeys);
    d_sortKeys = nullptr;
  }

  if (d_sortIndices) {
    cudaFree(d_sortIndices);
    d_sortIndices = nullptr;
  }

  if (d_sortedKeys) {
    cudaFree(d_sortedKeys);
    d_sortedKeys = nullptr;
  }

  if (d_sortedIndices) {
    cudaFree(d_sortedIndices);
    d_sortedIndices = nullptr;
  }

  if (d_sortedAABBs) {
    cudaFree(d_sortedAABBs);
    d_sortedAABBs = nullptr;
  }

  if (d_tempStorage) {
    cudaFree(d_tempStorage);
    d_tempStorage = nullptr;
  }

  if (d_collisionPairs) {
    cudaFree(d_collisionPairs);
    d_collisionPairs = nullptr;
  }

  if (d_neighborPairHashes) {
    cudaFree(d_neighborPairHashes);
    d_neighborPairHashes = nullptr;
  }

  if (d_nodes) {
    cudaFree(d_nodes);
    d_nodes = nullptr;
  }

  if (d_elements) {
    cudaFree(d_elements);
    d_elements = nullptr;
  }

  if (d_bp) {
    cudaFree(d_bp);
    d_bp = nullptr;
  }

  numObjects       = 0;
  numCollisions    = 0;
  numNeighborPairs = 0;
  n_nodes          = 0;
  n_elems          = 0;
}

// Create/update AABBs from mesh data
void Broadphase::CreateAABB() {
  if (n_elems == 0 || d_nodes == nullptr || d_elements == nullptr) {
    std::cerr << "Error: Mesh data not initialized" << std::endl;
    return;
  }

  // Launch kernel to compute AABBs using the device copy of this struct
  int blockSize = 256;
  int gridSize  = (n_elems + blockSize - 1) / blockSize;
  computeAABBKernel<<<gridSize, blockSize>>>(d_bp, d_aabbs, n_elems);

  cudaDeviceSynchronize();

  // Update host AABBs
  h_aabbs.resize(n_elems);
  cudaMemcpy(h_aabbs.data(), d_aabbs, n_elems * sizeof(AABB),
             cudaMemcpyDeviceToHost);

  numObjects = n_elems;

  std::cout << "Created " << numObjects << " AABBs from mesh elements"
            << std::endl;
}

void Broadphase::RetrieveAABBandPrints() {
  if (n_elems == 0 || h_aabbs.empty()) {
    std::cerr << "Error: No AABBs to print" << std::endl;
    return;
  }

  std::cout << "\n========== AABB Results ==========\n" << std::endl;

  for (int elem_idx = 0; elem_idx < n_elems; elem_idx++) {
    std::cout << "Element " << elem_idx << ":" << std::endl;

    // Print all nodes of this element
    std::cout << "  Nodes:" << std::endl;
    for (int i = 0; i < nodesPerElement; i++) {
      int node_id = h_elements(elem_idx, i);
      std::cout << "    Node " << node_id << ": (" << h_nodes(node_id, 0)
                << ", " << h_nodes(node_id, 1) << ", " << h_nodes(node_id, 2)
                << ")" << std::endl;
    }

    // Print AABB for this element
    AABB aabb = h_aabbs[elem_idx];
    std::cout << "  AABB:" << std::endl;
    std::cout << "    Min: (" << aabb.min.x << ", " << aabb.min.y << ", "
              << aabb.min.z << ")" << std::endl;
    std::cout << "    Max: (" << aabb.max.x << ", " << aabb.max.y << ", "
              << aabb.max.z << ")" << std::endl;
    std::cout << "    ObjectId: " << aabb.objectId << std::endl;
    std::cout << std::endl;
  }

  std::cout << "========== End of AABB Results ==========\n" << std::endl;
}

// Sort AABBs along specified axis
void Broadphase::SortAABBs(int axis) {
  if (numObjects == 0)
    return;

  axis = std::min(std::max(axis, 0), 2);  // Clamp to [0, 2]

  // Extract sort keys and initialize indices
  int blockSize = 256;
  int gridSize  = (numObjects + blockSize - 1) / blockSize;
  extractSortKeysKernel<<<gridSize, blockSize>>>(
      d_aabbs, d_sortKeys, d_sortIndices, axis, numObjects);

  // Sort using CUB (separate input and output buffers)
  cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageBytes, d_sortKeys,
                                  d_sortedKeys, d_sortIndices, d_sortedIndices,
                                  numObjects);

  // Reorder AABBs based on sorted indices
  reorderAABBsKernel<<<gridSize, blockSize>>>(d_aabbs, d_sortedAABBs,
                                              d_sortedIndices, numObjects);

  cudaDeviceSynchronize();

  std::cout << "Sorted " << numObjects << " AABBs along axis " << axis
            << std::endl;
}

// Print sorted AABBs for verification
void Broadphase::PrintSortedAABBs(int axis) {
  if (numObjects == 0) {
    std::cerr << "Error: No sorted AABBs to print" << std::endl;
    return;
  }

  axis = std::min(std::max(axis, 0), 2);

  // Copy sorted AABBs and indices to host
  std::vector<AABB> h_sortedAABBs(numObjects);
  std::vector<double> h_sortedKeys(numObjects);
  std::vector<int> h_sortedIndices(numObjects);

  cudaMemcpy(h_sortedAABBs.data(), d_sortedAABBs, numObjects * sizeof(AABB),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sortedKeys.data(), d_sortedKeys, numObjects * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sortedIndices.data(), d_sortedIndices, numObjects * sizeof(int),
             cudaMemcpyDeviceToHost);

  const char* axis_names[] = {"X", "Y", "Z"};
  std::cout << "\n========== Sorted AABBs (Axis: " << axis_names[axis]
            << ") ==========\n"
            << std::endl;

  for (int i = 0; i < numObjects; i++) {
    AABB aabb = h_sortedAABBs[i];
    std::cout << "Sorted Index " << i << " (Original Element "
              << h_sortedIndices[i] << "):" << std::endl;
    std::cout << "  Sort Key: " << h_sortedKeys[i] << std::endl;
    std::cout << "  AABB Min: (" << aabb.min.x << ", " << aabb.min.y << ", "
              << aabb.min.z << ")" << std::endl;
    std::cout << "  AABB Max: (" << aabb.max.x << ", " << aabb.max.y << ", "
              << aabb.max.z << ")" << std::endl;
    std::cout << "  ObjectId: " << aabb.objectId << std::endl;

    // Verify sort key matches AABB min value
    double expected_key = (axis == 0)   ? aabb.min.x
                          : (axis == 1) ? aabb.min.y
                                        : aabb.min.z;
    if (fabs(expected_key - h_sortedKeys[i]) > 1e-12) {
      std::cout << "  ⚠️  WARNING: Sort key mismatch! Expected " << expected_key
                << " but got " << h_sortedKeys[i] << std::endl;
    }

    // Check if sorted correctly (should be in ascending order)
    if (i > 0 && h_sortedKeys[i] < h_sortedKeys[i - 1]) {
      std::cout << "  ❌ ERROR: Sort order violated! Key " << h_sortedKeys[i]
                << " < previous key " << h_sortedKeys[i - 1] << std::endl;
    }

    std::cout << std::endl;
  }

  std::cout << "========== End of Sorted AABBs ==========\n" << std::endl;

  // Summary check
  bool is_sorted = true;
  for (int i = 1; i < numObjects; i++) {
    if (h_sortedKeys[i] < h_sortedKeys[i - 1]) {
      is_sorted = false;
      break;
    }
  }

  if (is_sorted) {
    std::cout << "✅ Sorting verification PASSED: AABBs are correctly sorted"
              << std::endl;
  } else {
    std::cout << "❌ Sorting verification FAILED: AABBs are NOT correctly "
                 "sorted"
              << std::endl;
  }
}

// Build neighbor connectivity map (CPU)
void Broadphase::BuildNeighborMap() {
  h_neighborPairs.clear();

  std::cout << "Building neighbor map..." << std::endl;

  // For each element, get its nodes
  for (int elemA = 0; elemA < n_elems; elemA++) {
    // Get nodes of element A
    std::unordered_set<int> nodesA;
    for (int i = 0; i < nodesPerElement; i++) {
      nodesA.insert(h_elements(elemA, i));
    }

    // Check against all other elements
    for (int elemB = elemA + 1; elemB < n_elems; elemB++) {
      // Check if elements share any nodes
      bool shareNode = false;
      for (int j = 0; j < nodesPerElement; j++) {
        if (nodesA.count(h_elements(elemB, j)) > 0) {
          shareNode = true;
          break;
        }
      }

      if (shareNode) {
        h_neighborPairs.insert({elemA, elemB});
      }
    }
  }

  std::cout << "Found " << h_neighborPairs.size() << " neighbor pairs"
            << std::endl;

  // Convert to sorted array of hashes for GPU binary search
  std::vector<long long> hashes;
  hashes.reserve(h_neighborPairs.size());

  for (const auto& pair : h_neighborPairs) {
    // Hash: combine two integers into one long long
    long long hash = ((long long)pair.first << 32) | pair.second;
    hashes.push_back(hash);
  }

  std::sort(hashes.begin(), hashes.end());

  // Copy to device
  numNeighborPairs = hashes.size();
  if (d_neighborPairHashes)
    cudaFree(d_neighborPairHashes);

  if (numNeighborPairs > 0) {
    cudaMalloc(&d_neighborPairHashes, numNeighborPairs * sizeof(long long));
    cudaMemcpy(d_neighborPairHashes, hashes.data(),
               numNeighborPairs * sizeof(long long), cudaMemcpyHostToDevice);
  }

  std::cout << "Neighbor map uploaded to GPU (" << numNeighborPairs << " pairs)"
            << std::endl;
}

// Detect collisions using two-pass sweep and prune (with neighbor filtering)
void Broadphase::DetectCollisions() {
  if (numObjects == 0)
    return;

  int blockSize = 256;
  int gridSize  = (numObjects + blockSize - 1) / blockSize;

  // Pass 1: Count collisions per element
  int* d_collisionCounts;
  int* d_collisionOffsets;
  cudaMalloc(&d_collisionCounts, numObjects * sizeof(int));
  cudaMalloc(&d_collisionOffsets, (numObjects + 1) * sizeof(int));

  countCollisionsKernel<<<gridSize, blockSize>>>(
      d_sortedAABBs, d_collisionCounts, numObjects, d_neighborPairHashes,
      numNeighborPairs);
  cudaDeviceSynchronize();

  // Compute prefix sum
  void* d_tempScan     = nullptr;
  size_t tempScanBytes = 0;

  cub::DeviceScan::ExclusiveSum(d_tempScan, tempScanBytes, d_collisionCounts,
                                d_collisionOffsets, numObjects + 1);
  cudaMalloc(&d_tempScan, tempScanBytes);

  cub::DeviceScan::ExclusiveSum(d_tempScan, tempScanBytes, d_collisionCounts,
                                d_collisionOffsets, numObjects + 1);
  cudaDeviceSynchronize();

  // Get total number of collisions
  cudaMemcpy(&numCollisions, &d_collisionOffsets[numObjects], sizeof(int),
             cudaMemcpyDeviceToHost);

  std::cout << "Total non-neighbor collisions found: " << numCollisions
            << std::endl;

  if (numCollisions == 0) {
    cudaFree(d_collisionCounts);
    cudaFree(d_collisionOffsets);
    cudaFree(d_tempScan);
    h_collisionPairs.clear();
    return;
  }

  // Pass 2: Generate pairs
  if (d_collisionPairs)
    cudaFree(d_collisionPairs);
  cudaMalloc(&d_collisionPairs, numCollisions * sizeof(CollisionPair));

  generateCollisionPairsKernel<<<gridSize, blockSize>>>(
      d_sortedAABBs, d_collisionOffsets, d_collisionPairs, numObjects,
      d_neighborPairHashes, numNeighborPairs);
  cudaDeviceSynchronize();

  // Copy results to host
  h_collisionPairs.resize(numCollisions);
  cudaMemcpy(h_collisionPairs.data(), d_collisionPairs,
             numCollisions * sizeof(CollisionPair), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_collisionCounts);
  cudaFree(d_collisionOffsets);
  cudaFree(d_tempScan);

  std::cout << "Detected " << numCollisions
            << " collision pairs (neighbors filtered)" << std::endl;
}

// Print collision pairs
void Broadphase::PrintCollisionPairs() {
  std::cout << "\n========== Collision Pairs (Non-Neighbors) ==========\n"
            << std::endl;

  // for (int i = 0; i < numCollisions; i++) {
  //   std::cout << "Pair " << i << ": Element " << h_collisionPairs[i].idA
  //             << " <-> Element " << h_collisionPairs[i].idB << std::endl;
  // }

  std::cout << "Total non-neighbor collisions: " << numCollisions << std::endl;

  std::cout << "\n========== End of Collision Pairs ==========\n" << std::endl;
}
