#include <algorithm>
#include <cmath>
#include <iostream>

#include "Narrowphase.cuh"
#include "NarrowphaseFunc.cuh"

// ============================================================================
// Host implementation
// ============================================================================

Narrowphase::Narrowphase()
    : numPatches(0),
      d_contactPatches(nullptr),
      d_nodes(nullptr),
      d_elements(nullptr),
      d_pressure(nullptr),
      n_nodes(0),
      n_elems(0),
      nodesPerElement(4),
      d_elementMeshIds(nullptr),
      d_collisionPairs(nullptr),
      numCollisionPairs(0),
      d_np(nullptr) {}

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
  cudaMalloc(&d_nodes, n_nodes * 3 * sizeof(double));
  cudaMemcpy(d_nodes, nodes.data(), n_nodes * 3 * sizeof(double),
             cudaMemcpyHostToDevice);

  // Allocate and copy elements (column-major)
  cudaMalloc(&d_elements, n_elems * nodesPerElement * sizeof(int));
  cudaMemcpy(d_elements, elements.data(),
             n_elems * nodesPerElement * sizeof(int), cudaMemcpyHostToDevice);

  // Allocate and copy pressure
  cudaMalloc(&d_pressure, n_nodes * sizeof(double));
  cudaMemcpy(d_pressure, pressure.data(), n_nodes * sizeof(double),
             cudaMemcpyHostToDevice);

  // Allocate and copy element mesh IDs
  cudaMalloc(&d_elementMeshIds, n_elems * sizeof(int));
  if (elementMeshIds.size() == n_elems) {
    // Use provided mesh IDs
    cudaMemcpy(d_elementMeshIds, elementMeshIds.data(), n_elems * sizeof(int),
               cudaMemcpyHostToDevice);
  } else {
    // Default: all elements belong to mesh 0
    std::vector<int> defaultIds(n_elems, 0);
    cudaMemcpy(d_elementMeshIds, defaultIds.data(), n_elems * sizeof(int),
               cudaMemcpyHostToDevice);
  }

  // Allocate device copy of this struct
  cudaMalloc(&d_np, sizeof(Narrowphase));
  cudaMemcpy(d_np, this, sizeof(Narrowphase), cudaMemcpyHostToDevice);

  std::cout << "Narrowphase initialized with " << n_nodes << " nodes, "
            << n_elems << " elements" << std::endl;
}

void Narrowphase::SetCollisionPairs(
    const std::vector<std::pair<int, int>>& pairs) {
  numCollisionPairs = pairs.size();

  if (numCollisionPairs == 0) {
    std::cout << "Narrowphase: No collision pairs to process" << std::endl;
    return;
  }

  // Flatten pairs for GPU
  std::vector<int> flatPairs(2 * numCollisionPairs);
  for (int i = 0; i < numCollisionPairs; i++) {
    flatPairs[2 * i]     = pairs[i].first;
    flatPairs[2 * i + 1] = pairs[i].second;
  }

  // Allocate and copy collision pairs
  if (d_collisionPairs)
    cudaFree(d_collisionPairs);
  cudaMalloc(&d_collisionPairs, 2 * numCollisionPairs * sizeof(int));
  cudaMemcpy(d_collisionPairs, flatPairs.data(),
             2 * numCollisionPairs * sizeof(int), cudaMemcpyHostToDevice);

  // Allocate contact patches
  if (d_contactPatches)
    cudaFree(d_contactPatches);
  cudaMalloc(&d_contactPatches, numCollisionPairs * sizeof(ContactPatch));

  // Initialize patches to invalid
  h_contactPatches.resize(numCollisionPairs);
  cudaMemcpy(d_contactPatches, h_contactPatches.data(),
             numCollisionPairs * sizeof(ContactPatch), cudaMemcpyHostToDevice);

  std::cout << "Narrowphase: Set " << numCollisionPairs << " collision pairs"
            << std::endl;
}

void Narrowphase::ComputeContactPatches() {
  if (numCollisionPairs == 0) {
    std::cout << "Narrowphase: No collision pairs to compute" << std::endl;
    return;
  }

  // Update device struct pointer
  cudaMemcpy(d_np, this, sizeof(Narrowphase), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize  = (numCollisionPairs + blockSize - 1) / blockSize;

  computeContactPatchesKernel<<<gridSize, blockSize>>>(
      d_np, d_contactPatches, d_collisionPairs, numCollisionPairs);

  cudaDeviceSynchronize();

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Narrowphase kernel error: " << cudaGetErrorString(err)
              << std::endl;
  }
}

void Narrowphase::RetrieveResults() {
  if (numCollisionPairs == 0)
    return;

  h_contactPatches.resize(numCollisionPairs);
  cudaMemcpy(h_contactPatches.data(), d_contactPatches,
             numCollisionPairs * sizeof(ContactPatch), cudaMemcpyDeviceToHost);

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
    cudaFree(d_contactPatches);
    d_contactPatches = nullptr;
  }

  if (d_nodes) {
    cudaFree(d_nodes);
    d_nodes = nullptr;
  }

  if (d_elements) {
    cudaFree(d_elements);
    d_elements = nullptr;
  }

  if (d_pressure) {
    cudaFree(d_pressure);
    d_pressure = nullptr;
  }

  if (d_elementMeshIds) {
    cudaFree(d_elementMeshIds);
    d_elementMeshIds = nullptr;
  }

  if (d_collisionPairs) {
    cudaFree(d_collisionPairs);
    d_collisionPairs = nullptr;
  }

  if (d_np) {
    cudaFree(d_np);
    d_np = nullptr;
  }

  h_contactPatches.clear();
  numPatches        = 0;
  numCollisionPairs = 0;
}

namespace {

Eigen::Vector4d ComputeBarycentricCoordinates(const Eigen::Vector3d& x,
                                              const Eigen::Vector3d& v0,
                                              const Eigen::Vector3d& v1,
                                              const Eigen::Vector3d& v2,
                                              const Eigen::Vector3d& v3) {
  Eigen::Matrix3d M;
  M.col(0)                  = v1 - v0;
  M.col(1)                  = v2 - v0;
  M.col(2)                  = v3 - v0;
  Eigen::Vector3d rhs       = x - v0;
  Eigen::Vector3d lambda123 = M.colPivHouseholderQr().solve(rhs);
  Eigen::Vector4d lambda;
  lambda[1] = lambda123[0];
  lambda[2] = lambda123[1];
  lambda[3] = lambda123[2];
  lambda[0] = 1.0 - lambda[1] - lambda[2] - lambda[3];
  return lambda;
}

}  // namespace

Eigen::VectorXd ComputeExternalForcesFromContactPatches(
    const Eigen::MatrixXd& nodes, const Eigen::MatrixXi& elements,
    const std::vector<ContactPatch>& patches) {
  const int n_nodes         = static_cast<int>(nodes.rows());
  const int nodesPerElement = static_cast<int>(elements.cols());
  Eigen::VectorXd f_ext     = Eigen::VectorXd::Zero(3 * n_nodes);

  if (nodesPerElement < 4) {
    return f_ext;
  }

  const double area_eps = 1e-18;

  for (const auto& patch : patches) {
    if (!patch.isValid) {
      continue;
    }
    if (!patch.validOrientation) {
      continue;
    }
    if (patch.area <= area_eps) {
      continue;
    }

    Eigen::Vector3d normal(patch.normal.x, patch.normal.y, patch.normal.z);
    Eigen::Vector3d centroid(patch.centroid.x, patch.centroid.y,
                             patch.centroid.z);
    double p_eq = patch.p_equilibrium;
    double A    = patch.area;

    Eigen::Vector3d F_patch = p_eq * A * normal;

    int tetA = patch.tetA_idx;
    int tetB = patch.tetB_idx;
    if (tetA < 0 || tetA >= elements.rows() || tetB < 0 ||
        tetB >= elements.rows()) {
      continue;
    }

    Eigen::Vector4i elemA;
    Eigen::Vector4i elemB;
    for (int i = 0; i < 4; ++i) {
      elemA[i] = elements(tetA, i);
      elemB[i] = elements(tetB, i);
    }

    Eigen::Vector3d vA0 = nodes.row(elemA[0]).transpose();
    Eigen::Vector3d vA1 = nodes.row(elemA[1]).transpose();
    Eigen::Vector3d vA2 = nodes.row(elemA[2]).transpose();
    Eigen::Vector3d vA3 = nodes.row(elemA[3]).transpose();

    Eigen::Vector3d vB0 = nodes.row(elemB[0]).transpose();
    Eigen::Vector3d vB1 = nodes.row(elemB[1]).transpose();
    Eigen::Vector3d vB2 = nodes.row(elemB[2]).transpose();
    Eigen::Vector3d vB3 = nodes.row(elemB[3]).transpose();

    Eigen::Vector4d N_A =
        ComputeBarycentricCoordinates(centroid, vA0, vA1, vA2, vA3);
    Eigen::Vector4d N_B =
        ComputeBarycentricCoordinates(centroid, vB0, vB1, vB2, vB3);

    // Contact force is repulsive: normal points from A to B
    // tetA is pushed opposite to normal (away from B) -> -F_patch
    // tetB is pushed along normal (away from A) -> +F_patch
    for (int i = 0; i < 4; ++i) {
      int nodeA = elemA[i];
      int nodeB = elemB[i];
      if (nodeA >= 0 && nodeA < n_nodes) {
        Eigen::Vector3d fA = N_A[i] * (-F_patch);  // Push A away from B
        f_ext.segment<3>(3 * nodeA) += fA;
      }
      if (nodeB >= 0 && nodeB < n_nodes) {
        Eigen::Vector3d fB = N_B[i] * F_patch;  // Push B away from A
        f_ext.segment<3>(3 * nodeB) += fB;
      }
    }
  }

  return f_ext;
}
