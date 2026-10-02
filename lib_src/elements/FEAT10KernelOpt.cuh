#pragma once

/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Dan Negrut, Ganesh Arivoli
 * Email:   negrut@wisc.edu, arivoli@wisc.edu
 * File:    FEAT10KernelOpt.cuh
 * Brief:   Fused internal force kernel for T10 elements with
 *          Mooney-Rivlin (+ optional Kelvin-Voigt damping) and 4-point
 *          quadrature.
 *==============================================================
 *==============================================================*/

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "FEAT10DataOpt.cuh"

// Material parameters for Mooney-Rivlin, fixed at compile time and declared in
// FEAT10DataOpt.cuh so callers and tests can read what the kernel will use.
// GPU_FEAT10Opt_Data::SetMooneyRivlin rejects any other values.
constexpr float kMu10 = GPU_FEAT10Opt_Data::kMu10;
constexpr float kMu01 = GPU_FEAT10Opt_Data::kMu01;
constexpr float kBulkK = GPU_FEAT10Opt_Data::kBulkK;
constexpr float kMinJthreshold = 1e-6f;
// Kelvin-Voigt damping is fixed at compile time for now to keep the fused
// kernel's register usage down; setting both to 0 compiles the damping path
// out. To be made settable later, like GPU_FEAT10_Data::SetDamping.
// Build with --config=opt_nodamp (defines FEAT10OPT_NO_DAMPING) for the
// undamped kernel.
#ifdef FEAT10OPT_NO_DAMPING
constexpr float kEtaDamp = 0.0f;
constexpr float kLambdaDamp = 0.0f;
#else
constexpr float kEtaDamp = 1.0e2f;    // Kelvin-Voigt shear damping (Pa·s)
constexpr float kLambdaDamp = 1.0e2f;  // Kelvin-Voigt volumetric damping (Pa·s)
#endif
constexpr bool kUseKelvinVoigtDamping =
    (kEtaDamp != 0.0f) || (kLambdaDamp != 0.0f);

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
 * Internal force kernel for T10 elements with Mooney-Rivlin (+ optional
 * Kelvin-Voigt damping), 4-point quadrature.
 * 4 threads per element (one per QP), 64 threads per block. Uses warp shuffle reduction.
 *
 * Works with the displacement gradient H = F - I instead of F. H is built
 * from displacements relative to node 0 of the element, formed in double and
 * cast to float once, and the stress is written so that every term that
 * vanishes at rest is a product of H terms. In float this removes the
 * cancellation that F = I + O(1e-7) suffers at small strain, where the bulk
 * modulus turned the rounding of F into hundreds of Pa of spurious pressure.
 */
__global__ void internalF_MooneyRivlin_4QP(
    int n_elem,
    const double* __restrict__ pPosNodes,
    const double* __restrict__ pPosNodesRef,  // reference X, for u = x - X
    const double* __restrict__ pVelNodes,
    const int* __restrict__ pElement_NodeIndexes,
    const float* __restrict__ pIsoMapInverse,
    float* __restrict__ pInternalForceNodes,
    float* __restrict__ pDeformationGradientF,
    float* __restrict__ pPiolaStressP,
    bool writeOutDefGradientF, bool writeOutPiolaP) {
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

  // Shared memory for H and invJacobian (each 9 components * blockDim.x);
  // P stays in registers.
  extern __shared__ float shMem[];
  float* s_H = shMem;
  float* s_invJacobian = s_H + 9 * blockDim.x;
  float PKone_00, PKone_01, PKone_02, PKone_10, PKone_11, PKone_12, PKone_20,
      PKone_21, PKone_22;

  // Early exit for padded elements (they have degenerate geometry)
  if (element_idx >= n_elem) {
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
  // Node 0 of the element is the origin for the relative displacements
  // u_a - u_0 = (x_a - x_0) - (X_a - X_0); one component per lane.
  // TODO: have the explicit solver integrate u = x - X directly and pass u
  // instead of x and X. That removes the 10 reference loads and 20 FP64
  // subtractions per thread (measured: kernel 64 us vs 73 us at res16).
  const int originNode = pElementNodes[0];
  double x0 = 0.0, X0 = 0.0;
  if (lane_in_tile < 3) {
    x0 = pPosNodes[3 * originNode + lane_in_tile];
    X0 = pPosNodesRef[3 * originNode + lane_in_tile];
  }

  // Thread-local Edot components (symmetric tensor storage).
  float Edot_00 = 0.0f, Edot_01 = 0.0f, Edot_02 = 0.0f;
  float Edot_11 = 0.0f, Edot_12 = 0.0f, Edot_22 = 0.0f;

  // ============================================================
  // Compute displacement gradient H = F - I = sum_a (u_a - u_0) (dN_a/dX)
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
#define H00 s_H[threadIdx.x + 0 * blockDim.x]
#define H01 s_H[threadIdx.x + 1 * blockDim.x]
#define H02 s_H[threadIdx.x + 2 * blockDim.x]
#define H10 s_H[threadIdx.x + 3 * blockDim.x]
#define H11 s_H[threadIdx.x + 4 * blockDim.x]
#define H12 s_H[threadIdx.x + 5 * blockDim.x]
#define H20 s_H[threadIdx.x + 6 * blockDim.x]
#define H21 s_H[threadIdx.x + 7 * blockDim.x]
#define H22 s_H[threadIdx.x + 8 * blockDim.x]

#define isoJacInv00 s_invJacobian[threadIdx.x + 0 * blockDim.x]
#define isoJacInv01 s_invJacobian[threadIdx.x + 1 * blockDim.x]
#define isoJacInv02 s_invJacobian[threadIdx.x + 2 * blockDim.x]
#define isoJacInv10 s_invJacobian[threadIdx.x + 3 * blockDim.x]
#define isoJacInv11 s_invJacobian[threadIdx.x + 4 * blockDim.x]
#define isoJacInv12 s_invJacobian[threadIdx.x + 5 * blockDim.x]
#define isoJacInv20 s_invJacobian[threadIdx.x + 6 * blockDim.x]
#define isoJacInv21 s_invJacobian[threadIdx.x + 7 * blockDim.x]
#define isoJacInv22 s_invJacobian[threadIdx.x + 8 * blockDim.x]

    // Node 0 (of 0-9)
    {
      const int whichGlobalNode = pElementNodes[0 * elements_per_block];
      // Relative displacement u_a - u_0 in double, cast once (same for nodes 1-9)
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)((pPosNodes[3 * whichGlobalNode + lane_in_tile] - x0) -
                        (pPosNodesRef[3 * whichGlobalNode + lane_in_tile] - X0));

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = 4.f * eta + 4.f * xi + 4.f * zeta - 3.f;
      // (dN/dX)_j = (dN/dxi)_i * Jinv_ij: use columns of Jinv
      const float dummy0 = h0 * (isoJacInv00 + isoJacInv10 + isoJacInv20);
      const float dummy1 = h0 * (isoJacInv01 + isoJacInv11 + isoJacInv21);
      const float dummy2 = h0 * (isoJacInv02 + isoJacInv12 + isoJacInv22);

      H00 = NUx * dummy0;
      H01 = NUx * dummy1;
      H02 = NUx * dummy2;
      H10 = NUy * dummy0;
      H11 = NUy * dummy1;
      H12 = NUy * dummy2;
      H20 = NUz * dummy0;
      H21 = NUz * dummy1;
      H22 = NUz * dummy2;
    }

    // Node 1 (of 0-9)
    {
      const int whichGlobalNode = pElementNodes[1 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)((pPosNodes[3 * whichGlobalNode + lane_in_tile] - x0) -
                        (pPosNodesRef[3 * whichGlobalNode + lane_in_tile] - X0));

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = 4.f * xi - 1.f;
      const float dummy0 = h0 * isoJacInv00;
      const float dummy1 = h0 * isoJacInv01;
      const float dummy2 = h0 * isoJacInv02;

      H00 += NUx * dummy0;
      H01 += NUx * dummy1;
      H02 += NUx * dummy2;
      H10 += NUy * dummy0;
      H11 += NUy * dummy1;
      H12 += NUy * dummy2;
      H20 += NUz * dummy0;
      H21 += NUz * dummy1;
      H22 += NUz * dummy2;
    }

    // Node 2 (of 0-9)
    {
      const int whichGlobalNode = pElementNodes[2 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)((pPosNodes[3 * whichGlobalNode + lane_in_tile] - x0) -
                        (pPosNodesRef[3 * whichGlobalNode + lane_in_tile] - X0));

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h1 = 4.f * eta - 1.f;
      const float dummy0 = h1 * isoJacInv10;
      const float dummy1 = h1 * isoJacInv11;
      const float dummy2 = h1 * isoJacInv12;

      H00 += NUx * dummy0;
      H01 += NUx * dummy1;
      H02 += NUx * dummy2;
      H10 += NUy * dummy0;
      H11 += NUy * dummy1;
      H12 += NUy * dummy2;
      H20 += NUz * dummy0;
      H21 += NUz * dummy1;
      H22 += NUz * dummy2;
    }

    // Node 3 (of 0-9)
    {
      const int whichGlobalNode = pElementNodes[3 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)((pPosNodes[3 * whichGlobalNode + lane_in_tile] - x0) -
                        (pPosNodesRef[3 * whichGlobalNode + lane_in_tile] - X0));

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h2 = 4.f * zeta - 1.f;
      const float dummy0 = h2 * isoJacInv20;
      const float dummy1 = h2 * isoJacInv21;
      const float dummy2 = h2 * isoJacInv22;

      H00 += NUx * dummy0;
      H01 += NUx * dummy1;
      H02 += NUx * dummy2;
      H10 += NUy * dummy0;
      H11 += NUy * dummy1;
      H12 += NUy * dummy2;
      H20 += NUz * dummy0;
      H21 += NUz * dummy1;
      H22 += NUz * dummy2;
    }

    // Node 4 (of 0-9)
    {
      const int whichGlobalNode = pElementNodes[4 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)((pPosNodes[3 * whichGlobalNode + lane_in_tile] - x0) -
                        (pPosNodesRef[3 * whichGlobalNode + lane_in_tile] - X0));

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = -4.f * eta - 8.f * xi - 4.f * zeta + 4.f;
      const float h1 = -4.f * xi;
      const float h2 = -4.f * xi;
      const float dummy0 = h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float dummy1 = h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float dummy2 = h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      H00 += NUx * dummy0;
      H01 += NUx * dummy1;
      H02 += NUx * dummy2;
      H10 += NUy * dummy0;
      H11 += NUy * dummy1;
      H12 += NUy * dummy2;
      H20 += NUz * dummy0;
      H21 += NUz * dummy1;
      H22 += NUz * dummy2;
    }

    // Node 5 (of 0-9)
    {
      const int whichGlobalNode = pElementNodes[5 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)((pPosNodes[3 * whichGlobalNode + lane_in_tile] - x0) -
                        (pPosNodesRef[3 * whichGlobalNode + lane_in_tile] - X0));

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = 4.f * eta;
      const float h1 = 4.f * xi;
      const float dummy0 = h0 * isoJacInv00 + h1 * isoJacInv10;
      const float dummy1 = h0 * isoJacInv01 + h1 * isoJacInv11;
      const float dummy2 = h0 * isoJacInv02 + h1 * isoJacInv12;

      H00 += NUx * dummy0;
      H01 += NUx * dummy1;
      H02 += NUx * dummy2;
      H10 += NUy * dummy0;
      H11 += NUy * dummy1;
      H12 += NUy * dummy2;
      H20 += NUz * dummy0;
      H21 += NUz * dummy1;
      H22 += NUz * dummy2;
    }

    // Node 6 (of 0-9)
    {
      const int whichGlobalNode = pElementNodes[6 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)((pPosNodes[3 * whichGlobalNode + lane_in_tile] - x0) -
                        (pPosNodesRef[3 * whichGlobalNode + lane_in_tile] - X0));

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

      H00 += NUx * dummy0;
      H01 += NUx * dummy1;
      H02 += NUx * dummy2;
      H10 += NUy * dummy0;
      H11 += NUy * dummy1;
      H12 += NUy * dummy2;
      H20 += NUz * dummy0;
      H21 += NUz * dummy1;
      H22 += NUz * dummy2;
    }

    // Node 7 (of 0-9)
    {
      const int whichGlobalNode = pElementNodes[7 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)((pPosNodes[3 * whichGlobalNode + lane_in_tile] - x0) -
                        (pPosNodesRef[3 * whichGlobalNode + lane_in_tile] - X0));

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = -4.f * zeta;
      const float h1 = -4.f * zeta;
      const float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
      const float dummy0 = h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float dummy1 = h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float dummy2 = h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      H00 += NUx * dummy0;
      H01 += NUx * dummy1;
      H02 += NUx * dummy2;
      H10 += NUy * dummy0;
      H11 += NUy * dummy1;
      H12 += NUy * dummy2;
      H20 += NUz * dummy0;
      H21 += NUz * dummy1;
      H22 += NUz * dummy2;
    }

    // Node 8 (of 0-9)
    {
      const int whichGlobalNode = pElementNodes[8 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)((pPosNodes[3 * whichGlobalNode + lane_in_tile] - x0) -
                        (pPosNodesRef[3 * whichGlobalNode + lane_in_tile] - X0));

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = 4.f * zeta;
      const float h2 = 4.f * xi;
      const float dummy0 = h0 * isoJacInv00 + h2 * isoJacInv20;
      const float dummy1 = h0 * isoJacInv01 + h2 * isoJacInv21;
      const float dummy2 = h0 * isoJacInv02 + h2 * isoJacInv22;

      H00 += NUx * dummy0;
      H01 += NUx * dummy1;
      H02 += NUx * dummy2;
      H10 += NUy * dummy0;
      H11 += NUy * dummy1;
      H12 += NUy * dummy2;
      H20 += NUz * dummy0;
      H21 += NUz * dummy1;
      H22 += NUz * dummy2;
    }

    // Node 9 (of 0-9)
    {
      const int whichGlobalNode = pElementNodes[9 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)((pPosNodes[3 * whichGlobalNode + lane_in_tile] - x0) -
                        (pPosNodesRef[3 * whichGlobalNode + lane_in_tile] - X0));

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h1 = 4.f * zeta;
      const float h2 = 4.f * eta;
      const float dummy0 = h1 * isoJacInv10 + h2 * isoJacInv20;
      const float dummy1 = h1 * isoJacInv11 + h2 * isoJacInv21;
      const float dummy2 = h1 * isoJacInv12 + h2 * isoJacInv22;

      H00 += NUx * dummy0;
      H01 += NUx * dummy1;
      H02 += NUx * dummy2;
      H10 += NUy * dummy0;
      H11 += NUy * dummy1;
      H12 += NUy * dummy2;
      H20 += NUz * dummy0;
      H21 += NUz * dummy1;
      H22 += NUz * dummy2;
    }

    // Write F = I + H to global memory if requested (for unit tests)
    if (writeOutDefGradientF && pDeformationGradientF != nullptr) {
      pDeformationGradientF[baseIdx + 0 * blockDim.x] = H00 + 1.0f;
      pDeformationGradientF[baseIdx + 1 * blockDim.x] = H01;
      pDeformationGradientF[baseIdx + 2 * blockDim.x] = H02;
      pDeformationGradientF[baseIdx + 3 * blockDim.x] = H10;
      pDeformationGradientF[baseIdx + 4 * blockDim.x] = H11 + 1.0f;
      pDeformationGradientF[baseIdx + 5 * blockDim.x] = H12;
      pDeformationGradientF[baseIdx + 6 * blockDim.x] = H20;
      pDeformationGradientF[baseIdx + 7 * blockDim.x] = H21;
      pDeformationGradientF[baseIdx + 8 * blockDim.x] = H22 + 1.0f;
    }
  }
  // End of displacement gradient H computation

  // ============================================================
  // Compute Edot = 0.5*(Fdot^T F + F^T Fdot) incrementally, F = I + H
  // Accumulate into thread-local symmetric Edot components.
  // ============================================================
  if (kUseKelvinVoigtDamping && pVelNodes != nullptr) {
    Edot_00 = 0.0f;
    Edot_01 = 0.0f;
    Edot_02 = 0.0f;
    Edot_11 = 0.0f;
    Edot_12 = 0.0f;
    Edot_22 = 0.0f;

    // Node 0 (of 0-9)
    {
      int whichGlobalNode = 0;
      if (lane_in_tile == 0) {
        whichGlobalNode = __ldg(&pElementNodes[0 * elements_per_block]);
      }
      whichGlobalNode = tile.shfl(whichGlobalNode, 0);

      float value = 0.0f;
      if (lane_in_tile < 3) {
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      }
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = 4.f * eta + 4.f * xi + 4.f * zeta - 3.f;
      const float gx = h0 * (isoJacInv00 + isoJacInv10 + isoJacInv20);
      const float gy = h0 * (isoJacInv01 + isoJacInv11 + isoJacInv21);
      const float gz = h0 * (isoJacInv02 + isoJacInv12 + isoJacInv22);

      // w = F^T v with F = I + H (same for nodes 1-9)
      const float wx = vx + H00 * vx + H10 * vy + H20 * vz;
      const float wy = vy + H01 * vx + H11 * vy + H21 * vz;
      const float wz = vz + H02 * vx + H12 * vy + H22 * vz;

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
      if (lane_in_tile < 3) {
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      }
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = 4.f * xi - 1.f;
      const float gx = h0 * isoJacInv00;
      const float gy = h0 * isoJacInv01;
      const float gz = h0 * isoJacInv02;

      const float wx = vx + H00 * vx + H10 * vy + H20 * vz;
      const float wy = vy + H01 * vx + H11 * vy + H21 * vz;
      const float wz = vz + H02 * vx + H12 * vy + H22 * vz;

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
      if (lane_in_tile < 3) {
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      }
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h1 = 4.f * eta - 1.f;
      const float gx = h1 * isoJacInv10;
      const float gy = h1 * isoJacInv11;
      const float gz = h1 * isoJacInv12;

      const float wx = vx + H00 * vx + H10 * vy + H20 * vz;
      const float wy = vy + H01 * vx + H11 * vy + H21 * vz;
      const float wz = vz + H02 * vx + H12 * vy + H22 * vz;

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
      if (lane_in_tile < 3) {
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      }
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h2 = 4.f * zeta - 1.f;
      const float gx = h2 * isoJacInv20;
      const float gy = h2 * isoJacInv21;
      const float gz = h2 * isoJacInv22;

      const float wx = vx + H00 * vx + H10 * vy + H20 * vz;
      const float wy = vy + H01 * vx + H11 * vy + H21 * vz;
      const float wz = vz + H02 * vx + H12 * vy + H22 * vz;

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
      if (lane_in_tile < 3) {
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      }
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = -4.f * eta - 8.f * xi - 4.f * zeta + 4.f;
      const float h1 = -4.f * xi;
      const float h2 = -4.f * xi;
      const float gx = h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float gy = h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float gz = h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      const float wx = vx + H00 * vx + H10 * vy + H20 * vz;
      const float wy = vy + H01 * vx + H11 * vy + H21 * vz;
      const float wz = vz + H02 * vx + H12 * vy + H22 * vz;

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
      if (lane_in_tile < 3) {
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      }
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = 4.f * eta;
      const float h1 = 4.f * xi;
      const float gx = h0 * isoJacInv00 + h1 * isoJacInv10;
      const float gy = h0 * isoJacInv01 + h1 * isoJacInv11;
      const float gz = h0 * isoJacInv02 + h1 * isoJacInv12;

      const float wx = vx + H00 * vx + H10 * vy + H20 * vz;
      const float wy = vy + H01 * vx + H11 * vy + H21 * vz;
      const float wz = vz + H02 * vx + H12 * vy + H22 * vz;

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
      if (lane_in_tile < 3) {
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      }
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = -4.f * eta;
      const float h1 = -8.f * eta - 4.f * xi - 4.f * zeta + 4.f;
      const float h2 = -4.f * eta;
      const float gx = h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float gy = h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float gz = h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      const float wx = vx + H00 * vx + H10 * vy + H20 * vz;
      const float wy = vy + H01 * vx + H11 * vy + H21 * vz;
      const float wz = vz + H02 * vx + H12 * vy + H22 * vz;

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
      if (lane_in_tile < 3) {
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      }
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = -4.f * zeta;
      const float h1 = -4.f * zeta;
      const float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
      const float gx = h0 * isoJacInv00 + h1 * isoJacInv10 + h2 * isoJacInv20;
      const float gy = h0 * isoJacInv01 + h1 * isoJacInv11 + h2 * isoJacInv21;
      const float gz = h0 * isoJacInv02 + h1 * isoJacInv12 + h2 * isoJacInv22;

      const float wx = vx + H00 * vx + H10 * vy + H20 * vz;
      const float wy = vy + H01 * vx + H11 * vy + H21 * vz;
      const float wz = vz + H02 * vx + H12 * vy + H22 * vz;

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
      if (lane_in_tile < 3) {
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      }
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h0 = 4.f * zeta;
      const float h2 = 4.f * xi;
      const float gx = h0 * isoJacInv00 + h2 * isoJacInv20;
      const float gy = h0 * isoJacInv01 + h2 * isoJacInv21;
      const float gz = h0 * isoJacInv02 + h2 * isoJacInv22;

      const float wx = vx + H00 * vx + H10 * vy + H20 * vz;
      const float wy = vy + H01 * vx + H11 * vy + H21 * vz;
      const float wz = vz + H02 * vx + H12 * vy + H22 * vz;

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
      if (lane_in_tile < 3) {
        value = (float)pVelNodes[3 * whichGlobalNode + lane_in_tile];
      }
      const float vx = tile.shfl(value, 0);
      const float vy = tile.shfl(value, 1);
      const float vz = tile.shfl(value, 2);

      const float h1 = 4.f * zeta;
      const float h2 = 4.f * eta;
      const float gx = h1 * isoJacInv10 + h2 * isoJacInv20;
      const float gy = h1 * isoJacInv11 + h2 * isoJacInv21;
      const float gz = h1 * isoJacInv12 + h2 * isoJacInv22;

      const float wx = vx + H00 * vx + H10 * vy + H20 * vz;
      const float wy = vy + H01 * vx + H11 * vy + H21 * vz;
      const float wz = vz + H02 * vx + H12 * vy + H22 * vz;

      Edot_00 += gx * wx;
      Edot_01 += 0.5f * (gx * wy + gy * wx);
      Edot_02 += 0.5f * (gx * wz + gz * wx);
      Edot_11 += gy * wy;
      Edot_12 += 0.5f * (gy * wz + gz * wy);
      Edot_22 += gz * wz;
    }
  }

  // ============================================================
  // Compute 1st Piola-Kirchhoff stress tensor P (Mooney-Rivlin) from H.
  // With F = I + H:
  //   S     = F F^T - I = H + H^T + H H^T,          tr S = I1 - 3
  //   J - 1 = tr H + (principal 2x2 minors of H) + det H
  //   F - F^{-T} = H + H^T F^{-T}
  //   P = cW (F - F^{-T}) + c01 (trS I - S) F + cInv F^{-T}
  // which equals P = 2*hatJ*(alpha*I - mu01*hatJ*F*F^T)*F + beta*F^{-T}
  // with the cancelling constant parts of alpha and beta removed.
  // ============================================================
  {
    float Jm1 = (H00 + H11 + H22) +
                (H00 * H11 - H01 * H10) + (H00 * H22 - H02 * H20) +
                (H11 * H22 - H12 * H21) +
                H00 * (H11 * H22 - H12 * H21) - H01 * (H10 * H22 - H12 * H20) +
                H02 * (H10 * H21 - H11 * H20);
    float J = 1.0f + Jm1;
    if (J < kMinJthreshold) {  // pad to avoid singularity
      J = kMinJthreshold;
      Jm1 = J - 1.0f;
    }
    const float invJ = 1.0f / J;
    float hatJ = 1.0f / cbrtf(J);
    hatJ *= hatJ;  // J^{-2/3}

    // S = H + H^T + H H^T (symmetric)
    float S00 = 2.0f * H00 + H00 * H00 + H01 * H01 + H02 * H02;
    float S11 = 2.0f * H11 + H10 * H10 + H11 * H11 + H12 * H12;
    float S22 = 2.0f * H22 + H20 * H20 + H21 * H21 + H22 * H22;
    float S01 = H01 + H10 + H00 * H10 + H01 * H11 + H02 * H12;
    float S02 = H02 + H20 + H00 * H20 + H01 * H21 + H02 * H22;
    float S12 = H12 + H21 + H10 * H20 + H11 * H21 + H12 * H22;
    const float trS = S00 + S11 + S22;  // I1 - 3
    const float trS2 = S00 * S00 + S11 * S11 + S22 * S22 +
                       2.0f * (S01 * S01 + S02 * S02 + S12 * S12);
    const float I2m3 = 2.0f * trS + 0.5f * (trS * trS - trS2);  // I2 - 3

    const float cW = 2.0f * hatJ * (kMu10 + 2.0f * hatJ * kMu01);
    const float c01 = 2.0f * hatJ * hatJ * kMu01;
    const float cInv = kBulkK * Jm1 * J -
                       (2.0f / 3.0f) * hatJ *
                           (kMu10 * trS + 2.0f * hatJ * kMu01 * I2m3);

    // T = trS I - S, reusing the S registers
    S00 = trS - S00;
    S11 = trS - S11;
    S22 = trS - S22;
    S01 = -S01;
    S02 = -S02;
    S12 = -S12;

    // F^{-T} = cof(F) / J, parked in the P registers until P is assembled
    PKone_00 = ((1.0f + H11) * (1.0f + H22) - H12 * H21) * invJ;
    PKone_01 = (H12 * H20 - H10 * (1.0f + H22)) * invJ;
    PKone_02 = (H10 * H21 - (1.0f + H11) * H20) * invJ;
    PKone_10 = (H02 * H21 - H01 * (1.0f + H22)) * invJ;
    PKone_11 = ((1.0f + H00) * (1.0f + H22) - H02 * H20) * invJ;
    PKone_12 = (H01 * H20 - (1.0f + H00) * H21) * invJ;
    PKone_20 = (H01 * H12 - H02 * (1.0f + H11)) * invJ;
    PKone_21 = (H02 * H10 - (1.0f + H00) * H12) * invJ;
    PKone_22 = ((1.0f + H00) * (1.0f + H11) - H01 * H10) * invJ;

    // P_ij = cW (H_ij + sum_k H_ki FinvT_kj) + c01 (T_ij + sum_k T_ik H_kj)
    //        + cInv FinvT_ij
    const float p00 = cW * (H00 + H00 * PKone_00 + H10 * PKone_10 + H20 * PKone_20) +
                      c01 * (S00 + S00 * H00 + S01 * H10 + S02 * H20) + cInv * PKone_00;
    const float p01 = cW * (H01 + H00 * PKone_01 + H10 * PKone_11 + H20 * PKone_21) +
                      c01 * (S01 + S00 * H01 + S01 * H11 + S02 * H21) + cInv * PKone_01;
    const float p02 = cW * (H02 + H00 * PKone_02 + H10 * PKone_12 + H20 * PKone_22) +
                      c01 * (S02 + S00 * H02 + S01 * H12 + S02 * H22) + cInv * PKone_02;
    const float p10 = cW * (H10 + H01 * PKone_00 + H11 * PKone_10 + H21 * PKone_20) +
                      c01 * (S01 + S01 * H00 + S11 * H10 + S12 * H20) + cInv * PKone_10;
    const float p11 = cW * (H11 + H01 * PKone_01 + H11 * PKone_11 + H21 * PKone_21) +
                      c01 * (S11 + S01 * H01 + S11 * H11 + S12 * H21) + cInv * PKone_11;
    const float p12 = cW * (H12 + H01 * PKone_02 + H11 * PKone_12 + H21 * PKone_22) +
                      c01 * (S12 + S01 * H02 + S11 * H12 + S12 * H22) + cInv * PKone_12;
    const float p20 = cW * (H20 + H02 * PKone_00 + H12 * PKone_10 + H22 * PKone_20) +
                      c01 * (S02 + S02 * H00 + S12 * H10 + S22 * H20) + cInv * PKone_20;
    const float p21 = cW * (H21 + H02 * PKone_01 + H12 * PKone_11 + H22 * PKone_21) +
                      c01 * (S12 + S02 * H01 + S12 * H11 + S22 * H21) + cInv * PKone_21;
    const float p22 = cW * (H22 + H02 * PKone_02 + H12 * PKone_12 + H22 * PKone_22) +
                      c01 * (S22 + S02 * H02 + S12 * H12 + S22 * H22) + cInv * PKone_22;
    PKone_00 = p00;
    PKone_01 = p01;
    PKone_02 = p02;
    PKone_10 = p10;
    PKone_11 = p11;
    PKone_12 = p12;
    PKone_20 = p20;
    PKone_21 = p21;
    PKone_22 = p22;

    if (kUseKelvinVoigtDamping) {
      // P += lambda tr(Edot) F + 2 eta F Edot, with F = I + H
      constexpr float two_eta = 2.0f * kEtaDamp;
      const float lamTr = kLambdaDamp * (Edot_00 + Edot_11 + Edot_22);
      const float F00 = 1.0f + H00, F11 = 1.0f + H11, F22 = 1.0f + H22;
      PKone_00 += lamTr * F00 + two_eta * (F00 * Edot_00 + H01 * Edot_01 + H02 * Edot_02);
      PKone_01 += lamTr * H01 + two_eta * (F00 * Edot_01 + H01 * Edot_11 + H02 * Edot_12);
      PKone_02 += lamTr * H02 + two_eta * (F00 * Edot_02 + H01 * Edot_12 + H02 * Edot_22);
      PKone_10 += lamTr * H10 + two_eta * (H10 * Edot_00 + F11 * Edot_01 + H12 * Edot_02);
      PKone_11 += lamTr * F11 + two_eta * (H10 * Edot_01 + F11 * Edot_11 + H12 * Edot_12);
      PKone_12 += lamTr * H12 + two_eta * (H10 * Edot_02 + F11 * Edot_12 + H12 * Edot_22);
      PKone_20 += lamTr * H20 + two_eta * (H20 * Edot_00 + H21 * Edot_01 + F22 * Edot_02);
      PKone_21 += lamTr * H21 + two_eta * (H20 * Edot_01 + H21 * Edot_11 + F22 * Edot_12);
      PKone_22 += lamTr * F22 + two_eta * (H20 * Edot_02 + H21 * Edot_12 + F22 * Edot_22);
    }

    // Write P to global memory if requested (for unit tests)
    if (writeOutPiolaP && pPiolaStressP != nullptr) {
      pPiolaStressP[baseIdx + 0 * blockDim.x] = PKone_00;
      pPiolaStressP[baseIdx + 1 * blockDim.x] = PKone_01;
      pPiolaStressP[baseIdx + 2 * blockDim.x] = PKone_02;
      pPiolaStressP[baseIdx + 3 * blockDim.x] = PKone_10;
      pPiolaStressP[baseIdx + 4 * blockDim.x] = PKone_11;
      pPiolaStressP[baseIdx + 5 * blockDim.x] = PKone_12;
      pPiolaStressP[baseIdx + 6 * blockDim.x] = PKone_20;
      pPiolaStressP[baseIdx + 7 * blockDim.x] = PKone_21;
      pPiolaStressP[baseIdx + 8 * blockDim.x] = PKone_22;
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
      const int whichGlobalNode = pElementNodes[0 * elements_per_block];
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
      const int whichGlobalNode = pElementNodes[1 * elements_per_block];
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
      const int whichGlobalNode = pElementNodes[2 * elements_per_block];
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
      const int whichGlobalNode = pElementNodes[3 * elements_per_block];
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
      const int whichGlobalNode = pElementNodes[4 * elements_per_block];

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
      const int whichGlobalNode = pElementNodes[5 * elements_per_block];
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
      const int whichGlobalNode = pElementNodes[6 * elements_per_block];
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
      const int whichGlobalNode = pElementNodes[7 * elements_per_block];
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
      const int whichGlobalNode = pElementNodes[8 * elements_per_block];
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
      const int whichGlobalNode = pElementNodes[9 * elements_per_block];
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
#undef H00
#undef H01
#undef H02
#undef H10
#undef H11
#undef H12
#undef H20
#undef H21
#undef H22
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
}

/**
 * Returns required shared memory size for the internal force kernel.
 *
 * Shared memory is used for the H and invJacobian matrices (each 9 floats per
 * thread); P stays in registers.
 *
 * @param blockSize Number of threads per block (should be 64)
 * @return Required shared memory in bytes
 */
inline size_t getInternalForceKernelSharedMemSize(int blockSize = 64) {
  return 2 * 9 * blockSize * sizeof(float);  // 2 matrices * 9 components *
                                              // blockSize threads
}
