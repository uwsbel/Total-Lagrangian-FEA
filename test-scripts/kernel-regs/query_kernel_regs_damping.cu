/*
 * Standalone compilation unit for querying register usage of the
 * FEAT10 fused internal force kernel.
 * NOTE: Not unit tested.
 *
 * Usage:
 *   nvcc -arch=sm_86 --ptxas-options=-v -c test-scripts/kernel-regs/query_kernel_regs_damping.cu \
 *        -o /dev/null
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

// Material parameters for Mooney-Rivlin
constexpr float kMu10 = 80000.0f;
constexpr float kMu01 = 20000.0f;
constexpr float kBulkK = 1.0e6f;
constexpr float kMinJthreshold = 1e-6f;
constexpr int kTotalElements = 1000;

// Viscous damping parameters (Kelvin-Voigt)
constexpr float kEtaDamp = 0.0f;
constexpr float kLambdaDamp = 0.0f;

/**
 * Warp shuffle reduction and atomic add helper.
 *
 * Reduces a value across a 4-thread tile using warp shuffles,
 * scales by the force scaling factor, then atomically adds to global memory.
 */
template <typename TileType>
__device__ __forceinline__ void reduce_scale_and_atomicAdd(
    const TileType& tile, int lane_in_tile,
    float* __restrict__ pInternalForceNodes, int whichGlobalNode,
    int component,  // 0=x, 1=y, 2=z
    float internalForce, float forceScalingFactor) {
  // Scale per-QP contribution
  internalForce *= forceScalingFactor;

  // 4-lane tile reduction: 4 -> 2 -> 1
  internalForce += tile.shfl_down(internalForce, 2);
  internalForce += tile.shfl_down(internalForce, 1);

  // Only lane 0 writes out
  if (lane_in_tile == 0) {
    atomicAdd(&pInternalForceNodes[3 * whichGlobalNode + component],
              internalForce);
  }
}

/**
 * Internal force kernel for T10 elements with Mooney-Rivlin + Kelvin-Voigt damping,
 * 4-point quadrature. 4 threads per element (one per QP), 64 threads per block.
 */
__global__ void internalF_MooneyRivlin_4QP_kelvin_voigt(
    const double* __restrict__ pPosNodes,
    const double* __restrict__ pVelNodes,
    const int* __restrict__ pElement_NodeIndexes,
    const float* __restrict__ pIsoMapInverse,
    float* __restrict__ pInternalForceNodes,
    float* __restrict__ pDeformationGradientF,
    bool writeOutDefGradientF) {
  // Define a tile of four threads; each thread handles one quadrature point
  constexpr int TILE = 4;
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<TILE> tile = cg::tiled_partition<TILE>(block);
  const int lane_in_tile = tile.thread_rank();

  // Calculate which element this tile of threads is responsible for
  const int elements_per_block = blockDim.x / TILE;
  const int element_idx =
      blockIdx.x * elements_per_block + tile.meta_group_rank();

  // Shared memory for F, invJacobian, PK1, and Edot
  extern __shared__ float shMem[];
  float* s_F = shMem;
  float* s_invJacobian = s_F + 9 * blockDim.x;
  float* s_PK1 = s_invJacobian + 9 * blockDim.x;
  float* s_Edot = s_PK1 + 9 * blockDim.x;

  // Early exit for padded elements (they have degenerate geometry)
  if (element_idx >= kTotalElements) {
    return;
  }

  // QP coordinates for the 4-point rule on the T10 tet element
  // Lane mapping (canonical 4-point tet rule: permutations of (b, a, a, a)):
  //  lane 0: (a,a,a)
  //  lane 1: (b,a,a)
  //  lane 2: (a,b,a)
  //  lane 3: (a,a,b)
  constexpr float a = 0.1381966011250105f;
  constexpr float b = 0.5854101966249685f;

  // Canonical 4-point tet rule: permutations of (b, a, a, a)
  const float xi = (lane_in_tile == 1) ? b : a;
  const float eta = (lane_in_tile == 2) ? b : a;
  const float zeta = (lane_in_tile == 3) ? b : a;

  // Index arithmetic for coalesced memory access
  // Blocked SoA layout: within each block, components are grouped
  const int baseIdx = blockIdx.x * TILE * elements_per_block * 9 + threadIdx.x;
  constexpr int nodes_per_element = 10;
  const int baseIdxNodes =
      blockIdx.x * nodes_per_element * elements_per_block + tile.meta_group_rank();
  const int* __restrict__ pElementNodes = pElement_NodeIndexes + baseIdxNodes;

  // ============================================================
  // Compute deformation gradient F
  // ============================================================
  {
    // Load inverse Jacobian for this QP into shared memory (coalesced reads)
    s_invJacobian[threadIdx.x + 0 * blockDim.x] =
        __ldg(&pIsoMapInverse[baseIdx + 0 * blockDim.x]);
    s_invJacobian[threadIdx.x + 1 * blockDim.x] =
        __ldg(&pIsoMapInverse[baseIdx + 1 * blockDim.x]);
    s_invJacobian[threadIdx.x + 2 * blockDim.x] =
        __ldg(&pIsoMapInverse[baseIdx + 2 * blockDim.x]);
    s_invJacobian[threadIdx.x + 3 * blockDim.x] =
        __ldg(&pIsoMapInverse[baseIdx + 3 * blockDim.x]);
    s_invJacobian[threadIdx.x + 4 * blockDim.x] =
        __ldg(&pIsoMapInverse[baseIdx + 4 * blockDim.x]);
    s_invJacobian[threadIdx.x + 5 * blockDim.x] =
        __ldg(&pIsoMapInverse[baseIdx + 5 * blockDim.x]);
    s_invJacobian[threadIdx.x + 6 * blockDim.x] =
        __ldg(&pIsoMapInverse[baseIdx + 6 * blockDim.x]);
    s_invJacobian[threadIdx.x + 7 * blockDim.x] =
        __ldg(&pIsoMapInverse[baseIdx + 7 * blockDim.x]);
    s_invJacobian[threadIdx.x + 8 * blockDim.x] =
        __ldg(&pIsoMapInverse[baseIdx + 8 * blockDim.x]);

// Macros for convenient access to shared memory arrays
#define F00 s_F[threadIdx.x + 0 * blockDim.x]
#define F01 s_F[threadIdx.x + 1 * blockDim.x]
#define F02 s_F[threadIdx.x + 2 * blockDim.x]
#define F10 s_F[threadIdx.x + 3 * blockDim.x]
#define F11 s_F[threadIdx.x + 4 * blockDim.x]
#define F12 s_F[threadIdx.x + 5 * blockDim.x]
#define F20 s_F[threadIdx.x + 6 * blockDim.x]
#define F21 s_F[threadIdx.x + 7 * blockDim.x]
#define F22 s_F[threadIdx.x + 8 * blockDim.x]

#define isoJacInv00 s_invJacobian[threadIdx.x + 0 * blockDim.x]
#define isoJacInv01 s_invJacobian[threadIdx.x + 1 * blockDim.x]
#define isoJacInv02 s_invJacobian[threadIdx.x + 2 * blockDim.x]
#define isoJacInv10 s_invJacobian[threadIdx.x + 3 * blockDim.x]
#define isoJacInv11 s_invJacobian[threadIdx.x + 4 * blockDim.x]
#define isoJacInv12 s_invJacobian[threadIdx.x + 5 * blockDim.x]
#define isoJacInv20 s_invJacobian[threadIdx.x + 6 * blockDim.x]
#define isoJacInv21 s_invJacobian[threadIdx.x + 7 * blockDim.x]
#define isoJacInv22 s_invJacobian[threadIdx.x + 8 * blockDim.x]

#define PKone_00 s_PK1[threadIdx.x + 0 * blockDim.x]
#define PKone_01 s_PK1[threadIdx.x + 1 * blockDim.x]
#define PKone_02 s_PK1[threadIdx.x + 2 * blockDim.x]
#define PKone_10 s_PK1[threadIdx.x + 3 * blockDim.x]
#define PKone_11 s_PK1[threadIdx.x + 4 * blockDim.x]
#define PKone_12 s_PK1[threadIdx.x + 5 * blockDim.x]
#define PKone_20 s_PK1[threadIdx.x + 6 * blockDim.x]
#define PKone_21 s_PK1[threadIdx.x + 7 * blockDim.x]
#define PKone_22 s_PK1[threadIdx.x + 8 * blockDim.x]

#define Edot_00 s_Edot[threadIdx.x + 0 * blockDim.x]
#define Edot_01 s_Edot[threadIdx.x + 1 * blockDim.x]
#define Edot_02 s_Edot[threadIdx.x + 2 * blockDim.x]
#define Edot_11 s_Edot[threadIdx.x + 3 * blockDim.x]
#define Edot_12 s_Edot[threadIdx.x + 4 * blockDim.x]
#define Edot_22 s_Edot[threadIdx.x + 5 * blockDim.x]

    // Node 0 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[0 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = 4.f * eta + 4.f * xi + 4.f * zeta - 3.f;
      // (dN/dX)_j = (dN/dxi)_i * Jinv_ij: use columns of Jinv
      const float dummy0 = h0 * (isoJacInv00 + isoJacInv10 + isoJacInv20);
      const float dummy1 = h0 * (isoJacInv01 + isoJacInv11 + isoJacInv21);
      const float dummy2 = h0 * (isoJacInv02 + isoJacInv12 + isoJacInv22);

      F00 = NUx * dummy0;
      F01 = NUx * dummy1;
      F02 = NUx * dummy2;
      F10 = NUy * dummy0;
      F11 = NUy * dummy1;
      F12 = NUy * dummy2;
      F20 = NUz * dummy0;
      F21 = NUz * dummy1;
      F22 = NUz * dummy2;
    }

    // Node 1 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[1 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = 4.f * xi - 1.f;
      const float dummy0 = h0 * isoJacInv00;
      const float dummy1 = h0 * isoJacInv01;
      const float dummy2 = h0 * isoJacInv02;

      F00 += NUx * dummy0;
      F01 += NUx * dummy1;
      F02 += NUx * dummy2;
      F10 += NUy * dummy0;
      F11 += NUy * dummy1;
      F12 += NUy * dummy2;
      F20 += NUz * dummy0;
      F21 += NUz * dummy1;
      F22 += NUz * dummy2;
    }

    // Node 2 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[2 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h1 = 4.f * eta - 1.f;
      const float dummy0 = h1 * isoJacInv10;
      const float dummy1 = h1 * isoJacInv11;
      const float dummy2 = h1 * isoJacInv12;

      F00 += NUx * dummy0;
      F01 += NUx * dummy1;
      F02 += NUx * dummy2;
      F10 += NUy * dummy0;
      F11 += NUy * dummy1;
      F12 += NUy * dummy2;
      F20 += NUz * dummy0;
      F21 += NUz * dummy1;
      F22 += NUz * dummy2;
    }

    // Node 3 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[3 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h2 = 4.f * zeta - 1.f;
      const float dummy0 = h2 * isoJacInv20;
      const float dummy1 = h2 * isoJacInv21;
      const float dummy2 = h2 * isoJacInv22;

      F00 += NUx * dummy0;
      F01 += NUx * dummy1;
      F02 += NUx * dummy2;
      F10 += NUy * dummy0;
      F11 += NUy * dummy1;
      F12 += NUy * dummy2;
      F20 += NUz * dummy0;
      F21 += NUz * dummy1;
      F22 += NUz * dummy2;
    }

    // Node 4 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[4 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = -4.f * eta - 8.f * xi - 4.f * zeta + 4.f;
      const float h1 = -4.f * xi;
      const float h2 = -4.f * xi;
      const float dummy0 = h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float dummy1 = h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float dummy2 = h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      F00 += NUx * dummy0;
      F01 += NUx * dummy1;
      F02 += NUx * dummy2;
      F10 += NUy * dummy0;
      F11 += NUy * dummy1;
      F12 += NUy * dummy2;
      F20 += NUz * dummy0;
      F21 += NUz * dummy1;
      F22 += NUz * dummy2;
    }

    // Node 5 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[5 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = 4.f * eta;
      const float h1 = 4.f * xi;
      const float dummy0 = h0 * isoJacInv00 + h1 * isoJacInv10;
      const float dummy1 = h0 * isoJacInv01 + h1 * isoJacInv11;
      const float dummy2 = h0 * isoJacInv02 + h1 * isoJacInv12;

      F00 += NUx * dummy0;
      F01 += NUx * dummy1;
      F02 += NUx * dummy2;
      F10 += NUy * dummy0;
      F11 += NUy * dummy1;
      F12 += NUy * dummy2;
      F20 += NUz * dummy0;
      F21 += NUz * dummy1;
      F22 += NUz * dummy2;
    }

    // Node 6 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[6 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = -4.f * eta;
      const float h1 = -8.f * eta - 4.f * xi - 4.f * zeta + 4.f;
      const float h2 = -4.f * eta;
      const float dummy0 =
          h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float dummy1 =
          h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float dummy2 =
          h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      F00 += NUx * dummy0;
      F01 += NUx * dummy1;
      F02 += NUx * dummy2;
      F10 += NUy * dummy0;
      F11 += NUy * dummy1;
      F12 += NUy * dummy2;
      F20 += NUz * dummy0;
      F21 += NUz * dummy1;
      F22 += NUz * dummy2;
    }

    // Node 7 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[7 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = -4.f * zeta;
      const float h1 = -4.f * zeta;
      const float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
      const float dummy0 = h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float dummy1 = h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float dummy2 = h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      F00 += NUx * dummy0;
      F01 += NUx * dummy1;
      F02 += NUx * dummy2;
      F10 += NUy * dummy0;
      F11 += NUy * dummy1;
      F12 += NUy * dummy2;
      F20 += NUz * dummy0;
      F21 += NUz * dummy1;
      F22 += NUz * dummy2;
    }

    // Node 8 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[8 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = 4.f * zeta;
      const float h2 = 4.f * xi;
      const float dummy0 = h0 * isoJacInv00 + h2 * isoJacInv20;
      const float dummy1 = h0 * isoJacInv01 + h2 * isoJacInv21;
      const float dummy2 = h0 * isoJacInv02 + h2 * isoJacInv22;

      F00 += NUx * dummy0;
      F01 += NUx * dummy1;
      F02 += NUx * dummy2;
      F10 += NUy * dummy0;
      F11 += NUy * dummy1;
      F12 += NUy * dummy2;
      F20 += NUz * dummy0;
      F21 += NUz * dummy1;
      F22 += NUz * dummy2;
    }

    // Node 9 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[9 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h1 = 4.f * zeta;
      const float h2 = 4.f * eta;
      const float dummy0 = h1 * isoJacInv10 + h2 * isoJacInv20;
      const float dummy1 = h1 * isoJacInv11 + h2 * isoJacInv21;
      const float dummy2 = h1 * isoJacInv12 + h2 * isoJacInv22;

      F00 += NUx * dummy0;
      F01 += NUx * dummy1;
      F02 += NUx * dummy2;
      F10 += NUy * dummy0;
      F11 += NUy * dummy1;
      F12 += NUy * dummy2;
      F20 += NUz * dummy0;
      F21 += NUz * dummy1;
      F22 += NUz * dummy2;
    }

    if (writeOutDefGradientF && pDeformationGradientF != nullptr) {
      pDeformationGradientF[baseIdx + 0 * blockDim.x] = F00;
      pDeformationGradientF[baseIdx + 1 * blockDim.x] = F01;
      pDeformationGradientF[baseIdx + 2 * blockDim.x] = F02;
      pDeformationGradientF[baseIdx + 3 * blockDim.x] = F10;
      pDeformationGradientF[baseIdx + 4 * blockDim.x] = F11;
      pDeformationGradientF[baseIdx + 5 * blockDim.x] = F12;
      pDeformationGradientF[baseIdx + 6 * blockDim.x] = F20;
      pDeformationGradientF[baseIdx + 7 * blockDim.x] = F21;
      pDeformationGradientF[baseIdx + 8 * blockDim.x] = F22;
    }
  }
  // End of deformation gradient F computation

  // ============================================================
  // Compute Edot = 0.5*(Fdot^T F + F^T Fdot) incrementally
  // Accumulate directly into s_Edot shared memory.
  // ============================================================
  {
    Edot_00 = 0.0f; Edot_01 = 0.0f; Edot_02 = 0.0f;
    Edot_11 = 0.0f; Edot_12 = 0.0f; Edot_22 = 0.0f;

    // Node 0 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[0 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = 4.f * eta + 4.f * xi + 4.f * zeta - 3.f;
      const float gx = h0 * (isoJacInv00 + isoJacInv10 + isoJacInv20);
      const float gy = h0 * (isoJacInv01 + isoJacInv11 + isoJacInv21);
      const float gz = h0 * (isoJacInv02 + isoJacInv12 + isoJacInv22);

      const float wx = F00 * vx + F10 * vy + F20 * vz;
      const float wy = F01 * vx + F11 * vy + F21 * vz;
      const float wz = F02 * vx + F12 * vy + F22 * vz;

      Edot_00 += gx * wx;
      Edot_01 += 0.5f * (gx * wy + gy * wx);
      Edot_02 += 0.5f * (gx * wz + gz * wx);
      Edot_11 += gy * wy;
      Edot_12 += 0.5f * (gy * wz + gz * wy);
      Edot_22 += gz * wz;
    }

    // Node 1 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[1 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = 4.f * xi - 1.f;
      const float gx = h0 * isoJacInv00;
      const float gy = h0 * isoJacInv01;
      const float gz = h0 * isoJacInv02;

      const float wx = F00 * vx + F10 * vy + F20 * vz;
      const float wy = F01 * vx + F11 * vy + F21 * vz;
      const float wz = F02 * vx + F12 * vy + F22 * vz;

      Edot_00 += gx * wx;
      Edot_01 += 0.5f * (gx * wy + gy * wx);
      Edot_02 += 0.5f * (gx * wz + gz * wx);
      Edot_11 += gy * wy;
      Edot_12 += 0.5f * (gy * wz + gz * wy);
      Edot_22 += gz * wz;
    }

    // Node 2 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[2 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h1 = 4.f * eta - 1.f;
      const float gx = h1 * isoJacInv10;
      const float gy = h1 * isoJacInv11;
      const float gz = h1 * isoJacInv12;

      const float wx = F00 * vx + F10 * vy + F20 * vz;
      const float wy = F01 * vx + F11 * vy + F21 * vz;
      const float wz = F02 * vx + F12 * vy + F22 * vz;

      Edot_00 += gx * wx;
      Edot_01 += 0.5f * (gx * wy + gy * wx);
      Edot_02 += 0.5f * (gx * wz + gz * wx);
      Edot_11 += gy * wy;
      Edot_12 += 0.5f * (gy * wz + gz * wy);
      Edot_22 += gz * wz;
    }

    // Node 3 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[3 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h2 = 4.f * zeta - 1.f;
      const float gx = h2 * isoJacInv20;
      const float gy = h2 * isoJacInv21;
      const float gz = h2 * isoJacInv22;

      const float wx = F00 * vx + F10 * vy + F20 * vz;
      const float wy = F01 * vx + F11 * vy + F21 * vz;
      const float wz = F02 * vx + F12 * vy + F22 * vz;

      Edot_00 += gx * wx;
      Edot_01 += 0.5f * (gx * wy + gy * wx);
      Edot_02 += 0.5f * (gx * wz + gz * wx);
      Edot_11 += gy * wy;
      Edot_12 += 0.5f * (gy * wz + gz * wy);
      Edot_22 += gz * wz;
    }

    // Node 4 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[4 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = -4.f * eta - 8.f * xi - 4.f * zeta + 4.f;
      const float h1 = -4.f * xi;
      const float h2 = -4.f * xi;
      const float gx = h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float gy = h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float gz = h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      const float wx = F00 * vx + F10 * vy + F20 * vz;
      const float wy = F01 * vx + F11 * vy + F21 * vz;
      const float wz = F02 * vx + F12 * vy + F22 * vz;

      Edot_00 += gx * wx;
      Edot_01 += 0.5f * (gx * wy + gy * wx);
      Edot_02 += 0.5f * (gx * wz + gz * wx);
      Edot_11 += gy * wy;
      Edot_12 += 0.5f * (gy * wz + gz * wy);
      Edot_22 += gz * wz;
    }

    // Node 5 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[5 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = 4.f * eta;
      const float h1 = 4.f * xi;
      const float gx = h0 * isoJacInv00 + h1 * isoJacInv10;
      const float gy = h0 * isoJacInv01 + h1 * isoJacInv11;
      const float gz = h0 * isoJacInv02 + h1 * isoJacInv12;

      const float wx = F00 * vx + F10 * vy + F20 * vz;
      const float wy = F01 * vx + F11 * vy + F21 * vz;
      const float wz = F02 * vx + F12 * vy + F22 * vz;

      Edot_00 += gx * wx;
      Edot_01 += 0.5f * (gx * wy + gy * wx);
      Edot_02 += 0.5f * (gx * wz + gz * wx);
      Edot_11 += gy * wy;
      Edot_12 += 0.5f * (gy * wz + gz * wy);
      Edot_22 += gz * wz;
    }

    // Node 6 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[6 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = -4.f * eta;
      const float h1 = -8.f * eta - 4.f * xi - 4.f * zeta + 4.f;
      const float h2 = -4.f * eta;
      const float gx = h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float gy = h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float gz = h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      const float wx = F00 * vx + F10 * vy + F20 * vz;
      const float wy = F01 * vx + F11 * vy + F21 * vz;
      const float wz = F02 * vx + F12 * vy + F22 * vz;

      Edot_00 += gx * wx;
      Edot_01 += 0.5f * (gx * wy + gy * wx);
      Edot_02 += 0.5f * (gx * wz + gz * wx);
      Edot_11 += gy * wy;
      Edot_12 += 0.5f * (gy * wz + gz * wy);
      Edot_22 += gz * wz;
    }

    // Node 7 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[7 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = -4.f * zeta;
      const float h1 = -4.f * zeta;
      const float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
      const float gx = h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float gy = h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float gz = h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      const float wx = F00 * vx + F10 * vy + F20 * vz;
      const float wy = F01 * vx + F11 * vy + F21 * vz;
      const float wz = F02 * vx + F12 * vy + F22 * vz;

      Edot_00 += gx * wx;
      Edot_01 += 0.5f * (gx * wy + gy * wx);
      Edot_02 += 0.5f * (gx * wz + gz * wx);
      Edot_11 += gy * wy;
      Edot_12 += 0.5f * (gy * wz + gz * wy);
      Edot_22 += gz * wz;
    }

    // Node 8 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[8 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = 4.f * zeta;
      const float h2 = 4.f * xi;
      const float gx = h0 * isoJacInv00 + h2 * isoJacInv20;
      const float gy = h0 * isoJacInv01 + h2 * isoJacInv21;
      const float gz = h0 * isoJacInv02 + h2 * isoJacInv22;

      const float wx = F00 * vx + F10 * vy + F20 * vz;
      const float wy = F01 * vx + F11 * vy + F21 * vz;
      const float wz = F02 * vx + F12 * vy + F22 * vz;

      Edot_00 += gx * wx;
      Edot_01 += 0.5f * (gx * wy + gy * wx);
      Edot_02 += 0.5f * (gx * wz + gz * wx);
      Edot_11 += gy * wy;
      Edot_12 += 0.5f * (gy * wz + gz * wy);
      Edot_22 += gz * wz;
    }

    // Node 9 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[9 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h1 = 4.f * zeta;
      const float h2 = 4.f * eta;
      const float gx = h1 * isoJacInv10 + h2 * isoJacInv20;
      const float gy = h1 * isoJacInv11 + h2 * isoJacInv21;
      const float gz = h1 * isoJacInv12 + h2 * isoJacInv22;

      const float wx = F00 * vx + F10 * vy + F20 * vz;
      const float wy = F01 * vx + F11 * vy + F21 * vz;
      const float wz = F02 * vx + F12 * vy + F22 * vz;

      Edot_00 += gx * wx;
      Edot_01 += 0.5f * (gx * wy + gy * wx);
      Edot_02 += 0.5f * (gx * wz + gz * wx);
      Edot_11 += gy * wy;
      Edot_12 += 0.5f * (gy * wz + gz * wy);
      Edot_22 += gz * wz;
    }
  }

  // ============================================================
  // Compute 1st Piola-Kirchhoff stress tensor P (Mooney-Rivlin)
  // P = 2*hatJ*(alpha*I - mu01*hatJ*F*F^T)F + beta*F^{-T}
  // ============================================================
  {
    // Compute determinant of F, pad with minJthreshold to avoid singularity
    float dummy = F00 * (F11 * F22 - F12 * F21) - F01 * (F10 * F22 - F12 * F20) +
                  F02 * (F10 * F21 - F11 * F20);
    const float J = (dummy < kMinJthreshold ? kMinJthreshold : dummy);
    float invJ = 1.0f / J;

    dummy = cbrtf(J);
    dummy = 1.0f / dummy;
    float hatJ = dummy * dummy;  // J^{-2/3}

    // PK1 temporarily holds B = F * F^T (symmetric)
    PKone_00 = F00 * F00 + F01 * F01 + F02 * F02;
    PKone_11 = F10 * F10 + F11 * F11 + F12 * F12;
    PKone_22 = F20 * F20 + F21 * F21 + F22 * F22;
    PKone_01 = F00 * F10 + F01 * F11 + F02 * F12;
    PKone_10 = PKone_01;
    PKone_02 = F00 * F20 + F01 * F21 + F02 * F22;
    PKone_20 = PKone_02;
    PKone_12 = F10 * F20 + F11 * F21 + F12 * F22;
    PKone_21 = PKone_12;

    // I1 = tr(B), I2 = 0.5*(I1^2 - tr(B*B))
    dummy = PKone_00 + PKone_11 + PKone_22;  // I1
    float bibi = PKone_00 * PKone_00 + PKone_11 * PKone_11 +
                 PKone_22 * PKone_22 +
                 2.0f * (PKone_01 * PKone_01 + PKone_02 * PKone_02 +
                         PKone_12 * PKone_12);  // tr(B*B)
    bibi = 0.5f * (dummy * dummy - bibi);       // I2
    float I1 = dummy;
    float I2 = bibi;
    float alpha = kMu10 + kMu01 * I1 * hatJ;
    float beta = kBulkK * (J - 1.0f) * J -
                 (2.0f / 3.0f) * hatJ * (kMu10 * I1 + 2.0f * kMu01 * I2 * hatJ);

    // Update PK1 to hold the matrix that multiplies F from the left
    bibi = -kMu01 * hatJ;
    bibi *= (2.0f * hatJ);
    alpha *= (2.0f * hatJ);
    // Absorb isotropic viscous contribution: lambda_d * tr(Edot)
    alpha += kLambdaDamp * (Edot_00 + Edot_11 + Edot_22);
    PKone_00 = alpha + bibi * PKone_00;
    PKone_11 = alpha + bibi * PKone_11;
    PKone_22 = alpha + bibi * PKone_22;
    PKone_01 = bibi * PKone_01;
    PKone_10 = bibi * PKone_10;
    PKone_02 = bibi * PKone_02;
    PKone_20 = bibi * PKone_20;
    PKone_12 = bibi * PKone_12;
    PKone_21 = bibi * PKone_21;

// Reuse local variables for intermediate symmetric matrix
#define intermediate_Matrix00 dummy
#define intermediate_Matrix01 bibi
#define intermediate_Matrix02 I1
#define intermediate_Matrix11 I2
#define intermediate_Matrix12 alpha
#define intermediate_Matrix22 hatJ

    intermediate_Matrix00 = PKone_00;
    intermediate_Matrix01 = PKone_01;
    intermediate_Matrix02 = PKone_02;
    intermediate_Matrix11 = PKone_11;
    intermediate_Matrix12 = PKone_12;
    intermediate_Matrix22 = PKone_22;

    // P = intermediate_Matrix * F + beta * F^{-T}
    invJ *= beta;

    // Row 0 of PK1
    PKone_00 = fmaf(F11, F22, -(F12 * F21)) * invJ;  // F^{-T} component
    PKone_00 += intermediate_Matrix00 * F00 + intermediate_Matrix01 * F10 +
                intermediate_Matrix02 * F20;

    PKone_01 = fmaf(F12, F20, -(F10 * F22)) * invJ;
    PKone_01 += intermediate_Matrix00 * F01 + intermediate_Matrix01 * F11 +
                intermediate_Matrix02 * F21;

    PKone_02 = fmaf(F10, F21, -(F11 * F20)) * invJ;
    PKone_02 += intermediate_Matrix00 * F02 + intermediate_Matrix01 * F12 +
                intermediate_Matrix02 * F22;

    // Row 1 of PK1 (use symmetry of intermediate_Matrix)
    PKone_10 = fmaf(F02, F21, -(F01 * F22)) * invJ;
    PKone_10 += intermediate_Matrix01 * F00 + intermediate_Matrix11 * F10 +
                intermediate_Matrix12 * F20;

    PKone_11 = fmaf(F00, F22, -(F02 * F20)) * invJ;
    PKone_11 += intermediate_Matrix01 * F01 + intermediate_Matrix11 * F11 +
                intermediate_Matrix12 * F21;

    PKone_12 = fmaf(F01, F20, -(F00 * F21)) * invJ;
    PKone_12 += intermediate_Matrix01 * F02 + intermediate_Matrix11 * F12 +
                intermediate_Matrix12 * F22;

    // Row 2 of PK1
    PKone_20 = fmaf(F01, F12, -(F02 * F11)) * invJ;
    PKone_20 += intermediate_Matrix02 * F00 + intermediate_Matrix12 * F10 +
                intermediate_Matrix22 * F20;

    PKone_21 = fmaf(F02, F10, -(F00 * F12)) * invJ;
    PKone_21 += intermediate_Matrix02 * F01 + intermediate_Matrix12 * F11 +
                intermediate_Matrix22 * F21;

    PKone_22 = fmaf(F00, F11, -(F01 * F10)) * invJ;
    PKone_22 += intermediate_Matrix02 * F02 + intermediate_Matrix12 * F12 +
                intermediate_Matrix22 * F22;

    // Add deviatoric viscous contribution: P_vis += 2*eta * F * Edot
    {
      constexpr float two_eta = 2.0f * kEtaDamp;
      PKone_00 += two_eta * (F00 * Edot_00 + F01 * Edot_01 + F02 * Edot_02);
      PKone_01 += two_eta * (F00 * Edot_01 + F01 * Edot_11 + F02 * Edot_12);
      PKone_02 += two_eta * (F00 * Edot_02 + F01 * Edot_12 + F02 * Edot_22);
      PKone_10 += two_eta * (F10 * Edot_00 + F11 * Edot_01 + F12 * Edot_02);
      PKone_11 += two_eta * (F10 * Edot_01 + F11 * Edot_11 + F12 * Edot_12);
      PKone_12 += two_eta * (F10 * Edot_02 + F11 * Edot_12 + F12 * Edot_22);
      PKone_20 += two_eta * (F20 * Edot_00 + F21 * Edot_01 + F22 * Edot_02);
      PKone_21 += two_eta * (F20 * Edot_01 + F21 * Edot_11 + F22 * Edot_12);
      PKone_22 += two_eta * (F20 * Edot_02 + F21 * Edot_12 + F22 * Edot_22);
    }

  }
  // End of PK1 computation

  // ============================================================
  // Compute internal force and accumulate with atomic adds
  // ============================================================
  {
    float forceScalingFactor = 0.f;
    {
      // Compute determinant of inverse Jacobian, then invert to get det(J)
      forceScalingFactor +=
          isoJacInv00 * (isoJacInv11 * isoJacInv22 - isoJacInv12 * isoJacInv21);
      forceScalingFactor -=
          isoJacInv01 * (isoJacInv10 * isoJacInv22 - isoJacInv12 * isoJacInv20);
      forceScalingFactor +=
          isoJacInv02 * (isoJacInv10 * isoJacInv21 - isoJacInv11 * isoJacInv20);
      forceScalingFactor = 1.f / forceScalingFactor;

      // Weight for 4-point quadrature (all QPs have same weight)
      constexpr float weightQP = 1.0f / 24.0f;
      forceScalingFactor *= weightQP;
    }

    // Node 0
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[0 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      const float h0 = 4.f * eta + 4.f * xi + 4.f * zeta - 3.f;

      const float hx = h0 * (isoJacInv00 + isoJacInv10 + isoJacInv20);
      const float hy = h0 * (isoJacInv01 + isoJacInv11 + isoJacInv21);
      const float hz = h0 * (isoJacInv02 + isoJacInv12 + isoJacInv22);

      float internalForce;

      internalForce = PKone_00 * hx + PKone_01 * hy + PKone_02 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_10 * hx + PKone_11 * hy + PKone_12 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_20 * hx + PKone_21 * hy + PKone_22 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 2, internalForce,
                                 forceScalingFactor);
    }

    // Node 1
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[1 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      const float h0 = 4.f * xi - 1.f;

      float internalForce;

      internalForce =
          h0 * (PKone_00 * isoJacInv00 + PKone_01 * isoJacInv01 +
                PKone_02 * isoJacInv02);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce =
          h0 * (PKone_10 * isoJacInv00 + PKone_11 * isoJacInv01 +
                PKone_12 * isoJacInv02);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce =
          h0 * (PKone_20 * isoJacInv00 + PKone_21 * isoJacInv01 +
                PKone_22 * isoJacInv02);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 2, internalForce,
                                 forceScalingFactor);
    }

    // Node 2
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[2 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      const float h1 = 4.f * eta - 1.f;

      float internalForce;

      internalForce =
          h1 * (PKone_00 * isoJacInv10 + PKone_01 * isoJacInv11 +
                PKone_02 * isoJacInv12);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce =
          h1 * (PKone_10 * isoJacInv10 + PKone_11 * isoJacInv11 +
                PKone_12 * isoJacInv12);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce =
          h1 * (PKone_20 * isoJacInv10 + PKone_21 * isoJacInv11 +
                PKone_22 * isoJacInv12);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 2, internalForce,
                                 forceScalingFactor);
    }

    // Node 3
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[3 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      const float h2 = 4.f * zeta - 1.f;

      float internalForce;

      internalForce =
          h2 * (PKone_00 * isoJacInv20 + PKone_01 * isoJacInv21 +
                PKone_02 * isoJacInv22);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce =
          h2 * (PKone_10 * isoJacInv20 + PKone_11 * isoJacInv21 +
                PKone_12 * isoJacInv22);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce =
          h2 * (PKone_20 * isoJacInv20 + PKone_21 * isoJacInv21 +
                PKone_22 * isoJacInv22);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 2, internalForce,
                                 forceScalingFactor);
    }

    // Node 4
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[4 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      const float h0 = -4.f * eta - 8.f * xi - 4.f * zeta + 4.f;
      const float h1 = -4.f * xi;
      const float h2 = -4.f * xi;

      const float hx = h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float hy = h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float hz = h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      float internalForce;

      internalForce = PKone_00 * hx + PKone_01 * hy + PKone_02 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_10 * hx + PKone_11 * hy + PKone_12 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_20 * hx + PKone_21 * hy + PKone_22 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 2, internalForce,
                                 forceScalingFactor);
    }

    // Node 5
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[5 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      const float h0 = 4.f * eta;
      const float h1 = 4.f * xi;
      const float hx = h0 * isoJacInv00 + h1 * isoJacInv10;
      const float hy = h0 * isoJacInv01 + h1 * isoJacInv11;
      const float hz = h0 * isoJacInv02 + h1 * isoJacInv12;

      float internalForce;

      internalForce = PKone_00 * hx + PKone_01 * hy + PKone_02 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_10 * hx + PKone_11 * hy + PKone_12 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_20 * hx + PKone_21 * hy + PKone_22 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 2, internalForce,
                                 forceScalingFactor);
    }

    // Node 6
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[6 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      const float h0 = -4.f * eta;
      const float h1 = -8.f * eta - 4.f * xi - 4.f * zeta + 4.f;
      const float h2 = -4.f * eta;
      const float hx =
          h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float hy =
          h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float hz =
          h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      float internalForce;

      internalForce = PKone_00 * hx + PKone_01 * hy + PKone_02 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_10 * hx + PKone_11 * hy + PKone_12 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_20 * hx + PKone_21 * hy + PKone_22 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 2, internalForce,
                                 forceScalingFactor);
    }

    // Node 7
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[7 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      const float h0 = -4.f * zeta;
      const float h1 = -4.f * zeta;
      const float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
      const float hx = h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float hy = h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float hz = h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      float internalForce;

      internalForce = PKone_00 * hx + PKone_01 * hy + PKone_02 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_10 * hx + PKone_11 * hy + PKone_12 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_20 * hx + PKone_21 * hy + PKone_22 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 2, internalForce,
                                 forceScalingFactor);
    }

    // Node 8
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[8 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      const float h0 = 4.f * zeta;
      const float h2 = 4.f * xi;
      const float hx = h0 * isoJacInv00 + h2 * isoJacInv20;
      const float hy = h0 * isoJacInv01 + h2 * isoJacInv21;
      const float hz = h0 * isoJacInv02 + h2 * isoJacInv22;

      float internalForce;

      internalForce = PKone_00 * hx + PKone_01 * hy + PKone_02 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_10 * hx + PKone_11 * hy + PKone_12 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_20 * hx + PKone_21 * hy + PKone_22 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 2, internalForce,
                                 forceScalingFactor);
    }

    // Node 9
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[9 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      const float h1 = 4.f * zeta;
      const float h2 = 4.f * eta;
      const float hx = h1 * isoJacInv10 + h2 * isoJacInv20;
      const float hy = h1 * isoJacInv11 + h2 * isoJacInv21;
      const float hz = h1 * isoJacInv12 + h2 * isoJacInv22;

      float internalForce;

      internalForce = PKone_00 * hx + PKone_01 * hy + PKone_02 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_10 * hx + PKone_11 * hy + PKone_12 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce = PKone_20 * hx + PKone_21 * hy + PKone_22 * hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 2, internalForce,
                                 forceScalingFactor);
    }
  }
  // End of internal force computation

// Clean up macros
#undef F00
#undef F01
#undef F02
#undef F10
#undef F11
#undef F12
#undef F20
#undef F21
#undef F22
#undef isoJacInv00
#undef isoJacInv01
#undef isoJacInv02
#undef isoJacInv10
#undef isoJacInv11
#undef isoJacInv12
#undef isoJacInv20
#undef isoJacInv21
#undef isoJacInv22
#undef PKone_00
#undef PKone_01
#undef PKone_02
#undef PKone_10
#undef PKone_11
#undef PKone_12
#undef PKone_20
#undef PKone_21
#undef PKone_22
#undef intermediate_Matrix00
#undef intermediate_Matrix01
#undef intermediate_Matrix02
#undef intermediate_Matrix11
#undef intermediate_Matrix12
#undef intermediate_Matrix22
#undef Edot_00
#undef Edot_01
#undef Edot_02
#undef Edot_11
#undef Edot_12
#undef Edot_22
}

/**
 * Returns required shared memory size for the internal force kernel.
 *
 * Shared memory is used for F, invJacobian, PK1, and Edot matrices
 * (each 9 floats per thread for F/invJacobian/PK1, 6 for Edot).
 *
 * @param blockSize Number of threads per block (should be 64)
 * @return Required shared memory in bytes
 */
inline size_t getInternalForceKernelSharedMemSize(int blockSize = 64) {
  return (3 * 9 + 6) * blockSize * sizeof(float);  // F + invJ + PK1 + Edot
}
