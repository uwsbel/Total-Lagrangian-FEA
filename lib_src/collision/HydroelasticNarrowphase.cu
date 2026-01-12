/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    HydroelasticNarrowphase.cu
 * Brief:   Host-side narrowphase collision pipeline. Manages GPU data for
 *          tetrahedral meshes and pressures, launches CUDA kernels to compute
 *          iso-pressure contact patches, retrieves patch data to the host,
 *          and assembles nodal external forces from contact patches on the
 *          GPU for use by the structural solver.
 *==============================================================
 *==============================================================*/

#include <algorithm>
#include <cmath>
#include <iostream>

#include "HydroelasticNarrowphase.cuh"
#include "HydroelasticNarrowphaseFunc.cuh"

#include "lib_utils/cuda_utils.h"

// ============================================================================
// Host implementation
// ============================================================================

Narrowphase::Narrowphase()
    : numPatches(0),
      d_contactPatches(nullptr),
      contactPatchCapacity(0),
      d_nodes(nullptr),
      d_elements(nullptr),
      d_pressure(nullptr),
      ownsNodes(true),
      n_nodes(0),
      n_elems(0),
      nodesPerElement(4),
      d_elementMeshIds(nullptr),
      enableSelfCollision(true),
      verbose(false),
      d_collisionPairs(nullptr),
      numCollisionPairs(0),
      ownsCollisionPairs(false),
      d_np(nullptr),
      d_f_ext(nullptr),
      f_ext_size(0) {}

Narrowphase::~Narrowphase() {
  Destroy();
}

void Narrowphase::Initialize(const Eigen::MatrixXd& nodes,
                             const Eigen::MatrixXi& elements,
                             const Eigen::VectorXd& pressure,
                             const Eigen::VectorXi& elementMeshIds) {
  n_nodes         = nodes.rows();
  n_elems         = elements.rows();
  nodesPerElement = elements.cols();

  // For 10-node tets, we only use first 4 corners
  // The kernel will only access indices 0-3

  // Allocate and copy nodes (column-major)
  HANDLE_ERROR(cudaMalloc(&d_nodes, n_nodes * 3 * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_nodes, nodes.data(), n_nodes * 3 * sizeof(double),
                          cudaMemcpyHostToDevice));

  // Allocate and copy elements (column-major)
  HANDLE_ERROR(cudaMalloc(&d_elements, n_elems * nodesPerElement * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_elements, elements.data(),
                          n_elems * nodesPerElement * sizeof(int),
                          cudaMemcpyHostToDevice));

  // Allocate and copy pressure
  HANDLE_ERROR(cudaMalloc(&d_pressure, n_nodes * sizeof(double)));
  HANDLE_ERROR(cudaMemcpy(d_pressure, pressure.data(), n_nodes * sizeof(double),
                          cudaMemcpyHostToDevice));

  // Allocate and copy element mesh IDs
  HANDLE_ERROR(cudaMalloc(&d_elementMeshIds, n_elems * sizeof(int)));
  if (elementMeshIds.size() == n_elems) {
    // Use provided mesh IDs
    HANDLE_ERROR(cudaMemcpy(d_elementMeshIds, elementMeshIds.data(),
                            n_elems * sizeof(int), cudaMemcpyHostToDevice));
  } else {
    // Default: all elements belong to mesh 0
    std::vector<int> defaultIds(n_elems, 0);
    HANDLE_ERROR(cudaMemcpy(d_elementMeshIds, defaultIds.data(),
                            n_elems * sizeof(int), cudaMemcpyHostToDevice));
  }

  // Allocate device copy of this struct
  HANDLE_ERROR(cudaMalloc(&d_np, sizeof(Narrowphase)));
  HANDLE_ERROR(cudaMemcpy(d_np, this, sizeof(Narrowphase),
                          cudaMemcpyHostToDevice));

  std::cout << "Narrowphase initialized with " << n_nodes << " nodes, "
            << n_elems << " elements" << std::endl;
}

void Narrowphase::EnableSelfCollision(bool enable) {
  enableSelfCollision = enable;

  if (d_np) {
    HANDLE_ERROR(cudaMemcpy(d_np, this, sizeof(Narrowphase),
                            cudaMemcpyHostToDevice));
  }
}

void Narrowphase::UpdateNodes(const Eigen::MatrixXd& nodes) {
  if (n_nodes == 0 || d_nodes == nullptr) {
    std::cerr << "Narrowphase::UpdateNodes called before Initialize" << std::endl;
    return;
  }

  if (nodes.rows() != n_nodes || nodes.cols() != 3) {
    std::cerr << "Narrowphase::UpdateNodes: node matrix size mismatch" << std::endl;
    return;
  }

  HANDLE_ERROR(cudaMemcpy(d_nodes, nodes.data(), n_nodes * 3 * sizeof(double),
                          cudaMemcpyHostToDevice));
}

void Narrowphase::BindNodesDevicePtr(double* d_nodes_external) {
  if (n_nodes == 0 || d_np == nullptr) {
    std::cerr << "Narrowphase::BindNodesDevicePtr called before Initialize"
              << std::endl;
    return;
  }
  if (d_nodes_external == nullptr) {
    std::cerr << "Narrowphase::BindNodesDevicePtr: null device pointer"
              << std::endl;
    return;
  }

  if (d_nodes && ownsNodes) {
    HANDLE_ERROR(cudaFree(d_nodes));
  }
  d_nodes   = d_nodes_external;
  ownsNodes = false;

  HANDLE_ERROR(cudaMemcpy(d_np, this, sizeof(Narrowphase),
                          cudaMemcpyHostToDevice));
}

void Narrowphase::SetCollisionPairs(
    const std::vector<std::pair<int, int>>& pairs) {
  if (d_collisionPairs && ownsCollisionPairs) {
    HANDLE_ERROR(cudaFree(d_collisionPairs));
    d_collisionPairs = nullptr;
  }

  numCollisionPairs = static_cast<int>(pairs.size());

  if (numCollisionPairs == 0) {
    ownsCollisionPairs = false;
    if (verbose) {
      std::cout << "Narrowphase: No collision pairs to process\n";
    }
    return;
  }

  ownsCollisionPairs = true;

  std::vector<CollisionPair> devicePairs(numCollisionPairs);
  for (int i = 0; i < numCollisionPairs; i++) {
    devicePairs[i] = CollisionPair(pairs[i].first, pairs[i].second);
  }

  // Allocate and copy collision pairs (owned)
  HANDLE_ERROR(cudaMalloc(&d_collisionPairs,
                          numCollisionPairs * sizeof(CollisionPair)));
  HANDLE_ERROR(cudaMemcpy(d_collisionPairs, devicePairs.data(),
                          numCollisionPairs * sizeof(CollisionPair),
                          cudaMemcpyHostToDevice));

  // Allocate contact patches (reuse capacity)
  if (d_contactPatches == nullptr || contactPatchCapacity < numCollisionPairs) {
    if (d_contactPatches) {
      HANDLE_ERROR(cudaFree(d_contactPatches));
    }
    HANDLE_ERROR(
        cudaMalloc(&d_contactPatches, numCollisionPairs * sizeof(ContactPatch)));
    contactPatchCapacity = numCollisionPairs;
  }

  if (verbose) {
    std::cout << "Narrowphase: Set " << numCollisionPairs << " collision pairs\n";
  }
}

void Narrowphase::SetCollisionPairsDevice(const CollisionPair* d_pairs,
                                          int numPairs) {
  if (d_collisionPairs && ownsCollisionPairs) {
    HANDLE_ERROR(cudaFree(d_collisionPairs));
  }

  numCollisionPairs  = std::max(numPairs, 0);
  ownsCollisionPairs = false;

  if (numCollisionPairs == 0) {
    d_collisionPairs = nullptr;
    if (verbose) {
      std::cout << "Narrowphase: No collision pairs to process\n";
    }
    return;
  }

  d_collisionPairs = const_cast<CollisionPair*>(d_pairs);

  // Allocate contact patches (reuse capacity)
  if (d_contactPatches == nullptr || contactPatchCapacity < numCollisionPairs) {
    if (d_contactPatches) {
      HANDLE_ERROR(cudaFree(d_contactPatches));
    }
    HANDLE_ERROR(
        cudaMalloc(&d_contactPatches, numCollisionPairs * sizeof(ContactPatch)));
    contactPatchCapacity = numCollisionPairs;
  }

  if (verbose) {
    std::cout << "Narrowphase: Set " << numCollisionPairs
              << " collision pairs (device)\n";
  }
}

void Narrowphase::ComputeContactPatches() {
  if (numCollisionPairs == 0) {
    if (verbose) {
      std::cout << "Narrowphase: No collision pairs to compute\n";
    }
    return;
  }

  // Check for any prior CUDA errors
  cudaError_t prior_err = cudaGetLastError();
  if (prior_err != cudaSuccess) {
    std::cerr << "Prior CUDA error before ComputeContactPatches: "
              << cudaGetErrorString(prior_err) << std::endl;
  }

  // Update device struct pointer
  HANDLE_ERROR(cudaMemcpy(d_np, this, sizeof(Narrowphase),
                          cudaMemcpyHostToDevice));

  int blockSize = 256;
  int gridSize  = (numCollisionPairs + blockSize - 1) / blockSize;

  computeContactPatchesKernel<<<gridSize, blockSize>>>(
      d_np, d_contactPatches, d_collisionPairs, numCollisionPairs);
  // Check for launch errors without forcing a full device sync.
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    std::cerr << "Narrowphase kernel error: " << cudaGetErrorString(err)
              << std::endl;
  }
}

void Narrowphase::RetrieveResults() {
  if (numCollisionPairs == 0) {
    h_contactPatches.clear();
    numPatches = 0;
    return;
  }

  h_contactPatches.resize(numCollisionPairs);
  HANDLE_ERROR(cudaMemcpy(h_contactPatches.data(), d_contactPatches,
                          numCollisionPairs * sizeof(ContactPatch),
                          cudaMemcpyDeviceToHost));

  // Count valid patches
  numPatches = 0;
  for (const auto& patch : h_contactPatches) {
    if (patch.isValid)
      numPatches++;
  }

  std::cout << "Narrowphase: " << numPatches << " valid contact patches out of "
            << numCollisionPairs << " pairs" << std::endl;
}

void Narrowphase::PrintContactPatches(bool verbose) {
  std::cout << "\n========== Contact Patches ==========\n" << std::endl;

  int validCount         = 0;
  int invalidOrientation = 0;

  for (size_t i = 0; i < h_contactPatches.size(); i++) {
    const auto& patch = h_contactPatches[i];

    if (!patch.isValid)
      continue;

    validCount++;
    if (!patch.validOrientation)
      invalidOrientation++;

    if (verbose) {
      std::cout << "Patch " << validCount << " (Tet " << patch.tetA_idx
                << " <-> Tet " << patch.tetB_idx << "):" << std::endl;
      std::cout << "  Vertices: " << patch.numVertices << std::endl;
      std::cout << "  Normal: (" << patch.normal.x << ", " << patch.normal.y
                << ", " << patch.normal.z << ")" << std::endl;
      std::cout << "  Centroid: (" << patch.centroid.x << ", "
                << patch.centroid.y << ", " << patch.centroid.z << ")"
                << std::endl;
      std::cout << "  Area: " << patch.area << std::endl;
      std::cout << "  g_A: " << patch.g_A << ", g_B: " << patch.g_B
                << std::endl;
      std::cout << "  p_equilibrium: " << patch.p_equilibrium << std::endl;
      std::cout << "  Valid orientation: "
                << (patch.validOrientation ? "Yes" : "No") << std::endl;
      std::cout << std::endl;
    }
  }

  std::cout << "Total valid patches: " << validCount << std::endl;
  std::cout << "Patches with invalid orientation: " << invalidOrientation
            << std::endl;
  std::cout << "\n========== End Contact Patches ==========\n" << std::endl;
}

std::vector<ContactPatch> Narrowphase::GetValidPatches() const {
  std::vector<ContactPatch> valid;
  for (const auto& patch : h_contactPatches) {
    if (patch.isValid) {
      valid.push_back(patch);
    }
  }
  return valid;
}

void Narrowphase::Destroy() {
  if (d_contactPatches) {
    HANDLE_ERROR(cudaFree(d_contactPatches));
    d_contactPatches = nullptr;
  }
  contactPatchCapacity = 0;

  if (d_nodes) {
    if (ownsNodes) {
      HANDLE_ERROR(cudaFree(d_nodes));
    }
    d_nodes   = nullptr;
    ownsNodes = true;
  }

  if (d_elements) {
    HANDLE_ERROR(cudaFree(d_elements));
    d_elements = nullptr;
  }

  if (d_pressure) {
    HANDLE_ERROR(cudaFree(d_pressure));
    d_pressure = nullptr;
  }

  if (d_elementMeshIds) {
    HANDLE_ERROR(cudaFree(d_elementMeshIds));
    d_elementMeshIds = nullptr;
  }

  if (d_collisionPairs && ownsCollisionPairs) {
    HANDLE_ERROR(cudaFree(d_collisionPairs));
  }
  d_collisionPairs     = nullptr;
  ownsCollisionPairs   = false;
  numCollisionPairs    = 0;

  if (d_np) {
    HANDLE_ERROR(cudaFree(d_np));
    d_np = nullptr;
  }

  if (d_f_ext) {
    HANDLE_ERROR(cudaFree(d_f_ext));
    d_f_ext = nullptr;
  }
  f_ext_size = 0;

  h_contactPatches.clear();
  numPatches        = 0;
  numCollisionPairs = 0;
  n_nodes           = 0;
  n_elems           = 0;
}

Eigen::VectorXd Narrowphase::ComputeExternalForcesGPU(const double* d_vel,
                                                      double damping,
                                                      double friction) {
  Eigen::VectorXd f_ext = Eigen::VectorXd::Zero(3 * n_nodes);
  ComputeExternalForcesGPUDevice(d_vel, damping, friction);
  if (d_f_ext == nullptr || f_ext_size != 3 * n_nodes) {
    return f_ext;
  }
  HANDLE_ERROR(cudaMemcpy(f_ext.data(), d_f_ext, f_ext_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
  return f_ext;
}

void Narrowphase::ComputeExternalForcesGPUDevice(const double* d_vel,
                                                 double damping,
                                                 double friction) {
  // Allocate or reallocate d_f_ext if needed
  int required_size = 3 * n_nodes;
  if (d_f_ext == nullptr || f_ext_size != required_size) {
    if (d_f_ext) {
      HANDLE_ERROR(cudaFree(d_f_ext));
    }
    HANDLE_ERROR(cudaMalloc(&d_f_ext, required_size * sizeof(double)));
    f_ext_size = required_size;
  }

  // Zero the force buffer
  HANDLE_ERROR(cudaMemset(d_f_ext, 0, required_size * sizeof(double)));

  if (numCollisionPairs == 0 || d_contactPatches == nullptr) {
    return;
  }

  // Launch kernel - one thread per collision pair (patch)
  int blockSize = 256;
  int gridSize  = (numCollisionPairs + blockSize - 1) / blockSize;

  computeExternalForcesKernel<<<gridSize, blockSize>>>(
      d_contactPatches, numCollisionPairs, d_nodes, d_vel, d_elements, n_nodes,
      n_elems, damping, friction, d_f_ext);
  // Check for launch errors without forcing a full device sync.
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    std::cerr << "ComputeExternalForcesGPUDevice kernel error: "
              << cudaGetErrorString(err) << std::endl;
  }
}
