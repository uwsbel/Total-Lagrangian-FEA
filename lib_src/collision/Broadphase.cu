/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    Broadphase.cu
 * Brief:   Implements the GPU broadphase collision stage for tetrahedral
 *          meshes. Builds AABBs for elements, performs sweep-and-prune along
 *          a chosen axis, filters out mesh neighbors, and uses CUB-based
 *          prefix sums to allocate and generate non-neighbor collision pairs
 *          for the narrowphase.
 *==============================================================
 *==============================================================*/

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cub/cub.cuh>  // Include CUB only in .cu file

#include "Broadphase.cuh"
#include "BroadphaseFunc.cuh"

#include "lib_utils/mesh_manager.h"

// Constructor
Broadphase::Broadphase()
    : numObjects(0),
      numCollisions(0),
      d_aabbs(nullptr),
      d_nodes(nullptr),
      d_elements(nullptr),
      d_elementMeshIds(nullptr),
      enableSelfCollision(true),
      d_bp(nullptr),
      d_sortKeys(nullptr),
      d_sortIndices(nullptr),
      d_sortedKeys(nullptr),
      d_sortedIndices(nullptr),
      d_sortedAABBs(nullptr),
      d_tempStorage(nullptr),
      tempStorageBytes(0),
      d_collisionPairs(nullptr),
      d_sameMeshPairsCount(nullptr),
      d_neighborPairHashes(nullptr),
      numNeighborPairs(0),
      n_nodes(0),
      n_elems(0),
      nodesPerElement(0),
      ownsNodes(true),
      d_collisionCounts(nullptr),
      d_collisionOffsets(nullptr),
      collisionCountCapacity(0),
      d_scanTempStorage(nullptr),
      scanTempStorageBytes(0),
      collisionPairsCapacity(0),
      verbose(false) {}

// Destructor
Broadphase::~Broadphase() {
  Destroy();
}

static void BestEffortCudaFree(void *ptr, const char *name) {
  if (!ptr) {
    return;
  }
  cudaError_t err = cudaFree(ptr);
  if (err == cudaSuccess || err == cudaErrorCudartUnloading) {
    return;
  }
  std::cerr << "cudaFree(" << name << ") failed: " << cudaGetErrorString(err)
            << std::endl;
}

// Initialize GPU resources with mesh data
void Broadphase::Initialize(const Eigen::MatrixXd& nodes,
                            const Eigen::MatrixXi& elements,
                            const Eigen::VectorXi& elementMeshIds) {
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
  HANDLE_ERROR(cudaMalloc(&d_aabbs, n_elems * sizeof(AABB)));

  // Allocate sorting arrays (input and output buffers)
  HANDLE_ERROR(cudaMalloc(&d_sortKeys, n_elems * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_sortIndices, n_elems * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_sortedKeys, n_elems * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_sortedIndices, n_elems * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&d_sortedAABBs, n_elems * sizeof(AABB)));

  // Allocate temporary storage for CUB sorting
  d_tempStorage = nullptr;
  HANDLE_ERROR(cub::DeviceRadixSort::SortPairs(
      d_tempStorage, tempStorageBytes, d_sortKeys, d_sortedKeys, d_sortIndices,
      d_sortedIndices, n_elems));
  HANDLE_ERROR(cudaMalloc(&d_tempStorage, tempStorageBytes));

  // Allocate and copy mesh data to device
  // Nodes: n_nodes x 3 (column-major)
  HANDLE_ERROR(cudaMalloc(&d_nodes, n_nodes * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_nodes, nodes.data(), n_nodes * 3 * sizeof(double),
                          cudaMemcpyHostToDevice));
  ownsNodes = true;

  // Elements: n_elems x nodesPerElement (column-major)
  HANDLE_ERROR(cudaMalloc(&d_elements, n_elems * nodesPerElement * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_elements, elements.data(),
                          n_elems * nodesPerElement * sizeof(int),
                          cudaMemcpyHostToDevice));

  // Allocate and copy element mesh IDs
  h_elementMeshIds.assign(n_elems, 0);
  if (elementMeshIds.size() == n_elems) {
    for (int i = 0; i < n_elems; ++i) {
      h_elementMeshIds[i] = elementMeshIds(i);
    }
  }

  HANDLE_ERROR(cudaMalloc(&d_elementMeshIds, n_elems * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_elementMeshIds, h_elementMeshIds.data(),
                          n_elems * sizeof(int), cudaMemcpyHostToDevice));

  // Allocate device copy of this struct and copy to device
  HANDLE_ERROR(cudaMalloc(&d_bp, sizeof(Broadphase)));
  HANDLE_ERROR(cudaMemcpy(d_bp, this, sizeof(Broadphase),
                          cudaMemcpyHostToDevice));

  if (d_sameMeshPairsCount == nullptr) {
    HANDLE_ERROR(cudaMalloc(&d_sameMeshPairsCount, sizeof(int)));
  }

  std::cout << "Broadphase initialized with " << n_nodes << " nodes and "
            << n_elems << " elements" << std::endl;
}

void Broadphase::EnableSelfCollision(bool enable) {
  enableSelfCollision = enable;

  if (d_bp) {
    HANDLE_ERROR(cudaMemcpy(d_bp, this, sizeof(Broadphase),
                            cudaMemcpyHostToDevice));
  }
}

// Destroy/cleanup GPU resources
void Broadphase::Destroy() {
  if (d_aabbs) {
    BestEffortCudaFree(d_aabbs, "d_aabbs");
    d_aabbs = nullptr;
  }

  if (d_sortKeys) {
    BestEffortCudaFree(d_sortKeys, "d_sortKeys");
    d_sortKeys = nullptr;
  }

  if (d_sortIndices) {
    BestEffortCudaFree(d_sortIndices, "d_sortIndices");
    d_sortIndices = nullptr;
  }

  if (d_sortedKeys) {
    BestEffortCudaFree(d_sortedKeys, "d_sortedKeys");
    d_sortedKeys = nullptr;
  }

  if (d_sortedIndices) {
    BestEffortCudaFree(d_sortedIndices, "d_sortedIndices");
    d_sortedIndices = nullptr;
  }

  if (d_sortedAABBs) {
    BestEffortCudaFree(d_sortedAABBs, "d_sortedAABBs");
    d_sortedAABBs = nullptr;
  }

  if (d_tempStorage) {
    BestEffortCudaFree(d_tempStorage, "d_tempStorage");
    d_tempStorage = nullptr;
  }

  if (d_collisionPairs) {
    BestEffortCudaFree(d_collisionPairs, "d_collisionPairs");
    d_collisionPairs = nullptr;
  }
  collisionPairsCapacity = 0;

  if (d_sameMeshPairsCount) {
    BestEffortCudaFree(d_sameMeshPairsCount, "d_sameMeshPairsCount");
    d_sameMeshPairsCount = nullptr;
  }

  if (d_neighborPairHashes) {
    BestEffortCudaFree(d_neighborPairHashes, "d_neighborPairHashes");
    d_neighborPairHashes = nullptr;
  }

  if (d_collisionCounts) {
    BestEffortCudaFree(d_collisionCounts, "d_collisionCounts");
    d_collisionCounts = nullptr;
  }
  if (d_collisionOffsets) {
    BestEffortCudaFree(d_collisionOffsets, "d_collisionOffsets");
    d_collisionOffsets = nullptr;
  }
  collisionCountCapacity = 0;

  if (d_scanTempStorage) {
    BestEffortCudaFree(d_scanTempStorage, "d_scanTempStorage");
    d_scanTempStorage = nullptr;
  }
  scanTempStorageBytes = 0;

  if (d_nodes) {
    if (ownsNodes) {
      BestEffortCudaFree(d_nodes, "d_nodes");
    }
    d_nodes    = nullptr;
    ownsNodes  = true;
  }

  if (d_elementMeshIds) {
    BestEffortCudaFree(d_elementMeshIds, "d_elementMeshIds");
    d_elementMeshIds = nullptr;
  }

  if (d_elements) {
    BestEffortCudaFree(d_elements, "d_elements");
    d_elements = nullptr;
  }

  if (d_bp) {
    BestEffortCudaFree(d_bp, "d_bp");
    d_bp = nullptr;
  }

  h_elementMeshIds.clear();
  enableSelfCollision = true;

  numObjects       = 0;
  numCollisions    = 0;
  numNeighborPairs = 0;
  n_nodes          = 0;
  n_elems          = 0;
}

void Broadphase::Initialize(const ANCFCPUUtils::MeshManager& mesh_manager) {
  const Eigen::MatrixXd& nodes    = mesh_manager.GetAllNodes();
  const Eigen::MatrixXi& elements = mesh_manager.GetAllElements();

  Eigen::VectorXi elementMeshIds(mesh_manager.GetTotalElements());
  for (int i = 0; i < mesh_manager.GetNumMeshes(); ++i) {
    const auto& instance = mesh_manager.GetMeshInstance(i);
    for (int e = 0; e < instance.num_elements; ++e) {
      elementMeshIds(instance.element_offset + e) = i;
    }
  }

  Initialize(nodes, elements, elementMeshIds);
}

// Update node positions on device without changing topology or neighbor data
void Broadphase::UpdateNodes(const Eigen::MatrixXd& nodes) {
  if (n_nodes == 0 || d_nodes == nullptr) {
    std::cerr << "Broadphase::UpdateNodes called before Initialize" << std::endl;
    return;
  }

  if (nodes.rows() != n_nodes || nodes.cols() != 3) {
    std::cerr << "Broadphase::UpdateNodes: node matrix size mismatch" << std::endl;
    return;
  }

  // Update host copy (optional but keeps diagnostics consistent)
  h_nodes = nodes;

  // Copy updated positions to device; connectivity and neighbor map are reused
  HANDLE_ERROR(cudaMemcpy(d_nodes, nodes.data(), n_nodes * 3 * sizeof(double),
                          cudaMemcpyHostToDevice));
}

// Create/update AABBs from mesh data
void Broadphase::CreateAABB(bool copyToHost) {
  if (n_elems == 0 || d_nodes == nullptr || d_elements == nullptr) {
    std::cerr << "Error: Mesh data not initialized" << std::endl;
    return;
  }

  // Launch kernel to compute AABBs using the device copy of this struct
  int blockSize = 256;
  int gridSize  = (n_elems + blockSize - 1) / blockSize;
  computeAABBKernel<<<gridSize, blockSize>>>(d_bp, d_aabbs, n_elems);
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    std::cerr << "computeAABBKernel launch error: " << cudaGetErrorString(err)
              << std::endl;
  }

  if (copyToHost) {
    h_aabbs.resize(n_elems);
    HANDLE_ERROR(cudaMemcpy(h_aabbs.data(), d_aabbs, n_elems * sizeof(AABB),
                            cudaMemcpyDeviceToHost));
  }

  numObjects = n_elems;

  if (verbose) {
    std::cout << "Created " << numObjects << " AABBs from mesh elements\n";
  }
}

void Broadphase::RetrieveAABBandPrints() {
  if (n_elems == 0 || d_aabbs == nullptr) {
    std::cerr << "Error: No AABBs to print" << std::endl;
    return;
  }

  h_aabbs.resize(n_elems);
  HANDLE_ERROR(cudaMemcpy(h_aabbs.data(), d_aabbs, n_elems * sizeof(AABB),
                          cudaMemcpyDeviceToHost));

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

void Broadphase::BindNodesDevicePtr(double* d_nodes_external) {
  if (n_nodes == 0 || d_bp == nullptr) {
    std::cerr << "Broadphase::BindNodesDevicePtr called before Initialize"
              << std::endl;
    return;
  }
  if (d_nodes_external == nullptr) {
    std::cerr << "Broadphase::BindNodesDevicePtr: null device pointer"
              << std::endl;
    return;
  }

  if (d_nodes && ownsNodes) {
    HANDLE_ERROR(cudaFree(d_nodes));
  }
  d_nodes   = d_nodes_external;
  ownsNodes = false;

  HANDLE_ERROR(cudaMemcpy(d_bp, this, sizeof(Broadphase),
                          cudaMemcpyHostToDevice));
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
  HANDLE_ERROR(cudaPeekAtLastError());

  // Sort using CUB (separate input and output buffers)
  HANDLE_ERROR(cub::DeviceRadixSort::SortPairs(
      d_tempStorage, tempStorageBytes, d_sortKeys, d_sortedKeys, d_sortIndices,
      d_sortedIndices, numObjects));

  // Reorder AABBs based on sorted indices
  reorderAABBsKernel<<<gridSize, blockSize>>>(d_aabbs, d_sortedAABBs,
                                              d_sortedIndices, numObjects);
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    std::cerr << "SortAABBs kernel launch error: " << cudaGetErrorString(err)
              << std::endl;
  }

  if (verbose) {
    std::cout << "Sorted " << numObjects << " AABBs along axis " << axis << "\n";
  }
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

  HANDLE_ERROR(cudaMemcpy(h_sortedAABBs.data(), d_sortedAABBs,
                          numObjects * sizeof(AABB), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(h_sortedKeys.data(), d_sortedKeys,
                          numObjects * sizeof(double), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(h_sortedIndices.data(), d_sortedIndices,
                          numObjects * sizeof(int), cudaMemcpyDeviceToHost));

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
      std::cout << "  WARNING: Sort key mismatch. Expected " << expected_key
                << " but got " << h_sortedKeys[i] << std::endl;
    }

    // Check if sorted correctly (should be in ascending order)
    if (i > 0 && h_sortedKeys[i] < h_sortedKeys[i - 1]) {
      std::cout << "  ERROR: Sort order violated. Key " << h_sortedKeys[i]
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
    std::cout << "Sorting verification PASSED: AABBs are correctly sorted"
              << std::endl;
  } else {
    std::cout << "Sorting verification FAILED: AABBs are NOT correctly "
                 "sorted"
              << std::endl;
  }
}

// Build neighbor connectivity map (CPU).
// Optimized using a node-to-element map:
//   - Previous approach: O(n_elems^2) all-pairs element neighbor search.
//   - Current approach: O(n_nodes * avg_elements_per_node^2),
//     since we only compare elements that share a node.
// For typical tetrahedral meshes where each node belongs to few elements,
// this yields a substantial reduction in work compared to the naive method.
void Broadphase::BuildNeighborMap() {
  h_neighborPairs.clear();

  std::cout << "Building neighbor map..." << std::endl;

  // Step 1: Build node-to-element map (which elements contain each node)
  std::vector<std::vector<int>> nodeToElements(n_nodes);
  for (int elem = 0; elem < n_elems; elem++) {
    for (int i = 0; i < nodesPerElement; i++) {
      int nodeId = h_elements(elem, i);
      nodeToElements[nodeId].push_back(elem);
    }
  }

  // Step 2: For each node, all elements sharing that node are neighbors
  for (int nodeId = 0; nodeId < n_nodes; nodeId++) {
    const std::vector<int>& elems = nodeToElements[nodeId];
    int numElems = elems.size();
    
    // All pairs of elements sharing this node are neighbors
    for (int i = 0; i < numElems; i++) {
      for (int j = i + 1; j < numElems; j++) {
        int elemA = elems[i];
        int elemB = elems[j];
        // Ensure consistent ordering (smaller id first)
        if (elemA > elemB) std::swap(elemA, elemB);
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
    HANDLE_ERROR(cudaFree(d_neighborPairHashes));

  if (numNeighborPairs > 0) {
    HANDLE_ERROR(
        cudaMalloc(&d_neighborPairHashes, numNeighborPairs * sizeof(long long)));
    HANDLE_ERROR(cudaMemcpy(d_neighborPairHashes, hashes.data(),
                            numNeighborPairs * sizeof(long long),
                            cudaMemcpyHostToDevice));
  }

  std::cout << "Neighbor map uploaded to GPU (" << numNeighborPairs << " pairs)"
            << std::endl;
}

// Detect collisions using two-pass sweep and prune (with neighbor filtering)
void Broadphase::DetectCollisions(bool copyPairsToHost) {
  if (numObjects == 0)
    return;

  int blockSize = 256;
  int gridSize  = (numObjects + blockSize - 1) / blockSize;

  // Pass 1: Count collisions per element (reuse buffers; avoid per-step malloc/free)
  if (d_collisionCounts == nullptr || d_collisionOffsets == nullptr ||
      collisionCountCapacity < (numObjects + 1)) {
    if (d_collisionCounts)
      HANDLE_ERROR(cudaFree(d_collisionCounts));
    if (d_collisionOffsets)
      HANDLE_ERROR(cudaFree(d_collisionOffsets));
    HANDLE_ERROR(cudaMalloc(&d_collisionCounts, (numObjects + 1) * sizeof(int)));
    HANDLE_ERROR(
        cudaMalloc(&d_collisionOffsets, (numObjects + 1) * sizeof(int)));
    collisionCountCapacity = numObjects + 1;  // includes scan sentinel slot
  }

  countCollisionsKernel<<<gridSize, blockSize>>>(
      d_sortedAABBs, d_collisionCounts, numObjects, d_neighborPairHashes,
      numNeighborPairs, d_elementMeshIds, enableSelfCollision ? 1 : 0);
  HANDLE_ERROR(cudaPeekAtLastError());
  // `countCollisionsKernel` writes counts[0..numObjects-1] only. We scan
  // (numObjects + 1) entries so offsets[numObjects] becomes the total number of
  // collision pairs; therefore counts[numObjects] is a sentinel and must be 0.
  HANDLE_ERROR(cudaMemset(&d_collisionCounts[numObjects], 0, sizeof(int)));

  // Exclusive scan over (numObjects + 1) so offsets[numObjects] == total collisions.
  size_t requiredScanBytes = 0;
  HANDLE_ERROR(cub::DeviceScan::ExclusiveSum(nullptr, requiredScanBytes,
                                             d_collisionCounts,
                                             d_collisionOffsets,
                                             numObjects + 1));
  if (d_scanTempStorage == nullptr || scanTempStorageBytes < requiredScanBytes) {
    if (d_scanTempStorage)
      HANDLE_ERROR(cudaFree(d_scanTempStorage));
    HANDLE_ERROR(cudaMalloc(&d_scanTempStorage, requiredScanBytes));
    scanTempStorageBytes = requiredScanBytes;
  }

  HANDLE_ERROR(cub::DeviceScan::ExclusiveSum(d_scanTempStorage,
                                             scanTempStorageBytes,
                                             d_collisionCounts,
                                             d_collisionOffsets,
                                             numObjects + 1));

  // Minimal device->host transfer: just the total collision count (one int).
  HANDLE_ERROR(cudaMemcpy(&numCollisions, &d_collisionOffsets[numObjects],
                          sizeof(int), cudaMemcpyDeviceToHost));

  if (verbose) {
    std::cout << "Total non-neighbor collisions found: " << numCollisions
              << "\n";
  }

  if (numCollisions == 0) {
    h_collisionPairs.clear();
    return;
  }

  // Pass 2: Generate pairs
  if (d_collisionPairs == nullptr || collisionPairsCapacity < numCollisions) {
    if (d_collisionPairs)
      HANDLE_ERROR(cudaFree(d_collisionPairs));
    HANDLE_ERROR(
        cudaMalloc(&d_collisionPairs, numCollisions * sizeof(CollisionPair)));
    collisionPairsCapacity = numCollisions;
  }

  generateCollisionPairsKernel<<<gridSize, blockSize>>>(
      d_sortedAABBs, d_collisionOffsets, d_collisionPairs, numObjects,
      d_neighborPairHashes, numNeighborPairs, d_elementMeshIds,
      enableSelfCollision ? 1 : 0);
  HANDLE_ERROR(cudaPeekAtLastError());

  if (copyPairsToHost) {
    h_collisionPairs.resize(numCollisions);
    HANDLE_ERROR(cudaMemcpy(h_collisionPairs.data(), d_collisionPairs,
                            numCollisions * sizeof(CollisionPair),
                            cudaMemcpyDeviceToHost));
  } else {
    h_collisionPairs.clear();
  }

  if (verbose) {
    std::cout << "Detected " << numCollisions
              << " collision pairs (neighbors filtered)\n";
  }
}

int Broadphase::CountSameMeshPairsDevice() const {
  if (numCollisions == 0 || d_collisionPairs == nullptr || d_elementMeshIds == nullptr ||
      d_sameMeshPairsCount == nullptr) {
    return 0;
  }

  HANDLE_ERROR(cudaMemset(d_sameMeshPairsCount, 0, sizeof(int)));

  int blockSize = 256;
  int gridSize  = (numCollisions + blockSize - 1) / blockSize;

  countSameMeshPairsKernel<<<gridSize, blockSize>>>(
      d_collisionPairs, numCollisions, d_elementMeshIds, d_sameMeshPairsCount);
  HANDLE_ERROR(cudaPeekAtLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  int count = 0;
  HANDLE_ERROR(cudaMemcpy(&count, d_sameMeshPairsCount, sizeof(int),
                          cudaMemcpyDeviceToHost));
  return count;
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
