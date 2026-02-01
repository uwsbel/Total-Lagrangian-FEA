/*==============================================================
 *==============================================================
 * Project: TL-FEA
 * File:    FEAT10KernelOpt.cuh
 * Brief:   Fused internal force kernel for T10 elements with
 *          Mooney-Rivlin material and 4-point quadrature.
 *
 * Adapted from: future_work/internalF-MR-smem.cu
 *
 * Key features:
 * - 4 threads per element (one per quadrature point)
 * - 64 threads per block (16 elements)
 * - Cooperative groups with warp shuffle reduction
 * - Shared memory for F, invJacobian, and PK1
 * - Float compute, double positions
 *==============================================================
 *==============================================================*/

#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "FEAT10DataOpt.cuh"

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
 * Internal force kernel for T10 elements with Mooney-Rivlin, 4-point quadrature.
 * 4 threads per element (one per QP), 64 threads per block. Uses warp shuffle reduction.
 */
__global__ void internalF_MooneyRivlin_4QP(const GPU_FEAT10Opt_Data* __restrict__ d_data,
                                           bool writeOutDefGradientF,
                                           bool writeOutPiolaP) {
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

  // Shared memory for F, invJacobian, and PK1 (each 9 components * blockDim.x)
  extern __shared__ float shMem[];
  float* s_F = shMem;
  float* s_invJacobian = s_F + 9 * blockDim.x;
  float* s_PK1 = s_invJacobian + 9 * blockDim.x;

  // Early exit for padded elements (they have degenerate geometry)
  if (element_idx >= d_data->n_elem) {
    return;
  }

  // Load material parameters from the data struct
  const float mu10 = d_data->mu10;
  const float mu01 = d_data->mu01;
  const float bulkK = d_data->bulkK;
  const float minJthreshold = d_data->minJthreshold;

  // Load device pointers
  const double* __restrict__ pPosNodes = d_data->d_pos_nodes;
  const int* __restrict__ pElement_NodeIndexes = d_data->d_elem_nodes_soa;
  const float* __restrict__ pIsoMapInverse = d_data->d_iso_map_inv;
  float* __restrict__ pInternalForceNodes = d_data->d_internal_force;
  float* __restrict__ pDeformationGradientF = d_data->d_deformation_grad_F;
  float* __restrict__ pPiolaStressP = d_data->d_piola_stress_P;

  // QP coordinates for the 4-point rule on the T10 tet element
  // Lane mapping:
  //  lane 0: (a,a,a)
  //  lane 1: (b,a,a)
  //  lane 2: (b,b,a)
  //  lane 3: (b,b,b)
  constexpr float a = 0.1381966011250105f;
  constexpr float b = 0.5854101966249685f;

  const float xi = (lane_in_tile == 0) ? a : b;
  const float eta = (lane_in_tile < 2) ? a : b;
  const float zeta = (lane_in_tile < 3) ? a : b;

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

    // Node 0 (of 0-9)
    {
      const int whichGlobalNode = pElementNodes[0 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = 4.f * eta + 4.f * xi + 4.f * zeta - 3.f;
      const float dummy0 = h0 * (isoJacInv00 + isoJacInv01 + isoJacInv02);
      const float dummy1 = h0 * (isoJacInv10 + isoJacInv11 + isoJacInv12);
      const float dummy2 = h0 * (isoJacInv20 + isoJacInv21 + isoJacInv22);

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
      const int whichGlobalNode = pElementNodes[1 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = 4.f * xi - 1.f;
      const float dummy0 = h0 * isoJacInv00;
      const float dummy1 = h0 * isoJacInv10;
      const float dummy2 = h0 * isoJacInv20;

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
      const int whichGlobalNode = pElementNodes[2 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h1 = 4.f * eta - 1.f;
      const float dummy0 = h1 * isoJacInv01;
      const float dummy1 = h1 * isoJacInv11;
      const float dummy2 = h1 * isoJacInv21;

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
      const int whichGlobalNode = pElementNodes[3 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h2 = 4.f * zeta - 1.f;
      const float dummy0 = h2 * isoJacInv02;
      const float dummy1 = h2 * isoJacInv12;
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
      const int whichGlobalNode = pElementNodes[4 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = -4.f * eta - 8.f * xi - 4.f * zeta + 4.f;
      const float h1 = -4.f * xi;
      const float dummy0 = h0 * isoJacInv00 + h1 * (isoJacInv01 + isoJacInv02);
      const float dummy1 = h0 * isoJacInv10 + h1 * (isoJacInv11 + isoJacInv12);
      const float dummy2 = h0 * isoJacInv20 + h1 * (isoJacInv21 + isoJacInv22);

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
      const int whichGlobalNode = pElementNodes[5 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = 4.f * eta;
      const float h1 = 4.f * xi;
      const float dummy0 = h0 * isoJacInv00 + h1 * isoJacInv01;
      const float dummy1 = h0 * isoJacInv10 + h1 * isoJacInv11;
      const float dummy2 = h0 * isoJacInv20 + h1 * isoJacInv21;

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
      const int whichGlobalNode = pElementNodes[6 * elements_per_block];
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
          h0 * isoJacInv00 + h1 * isoJacInv01 + h2 * isoJacInv02;
      const float dummy1 =
          h0 * isoJacInv10 + h1 * isoJacInv11 + h2 * isoJacInv12;
      const float dummy2 =
          h0 * isoJacInv20 + h1 * isoJacInv21 + h2 * isoJacInv22;

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
      const int whichGlobalNode = pElementNodes[7 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = -4.f * zeta;
      const float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
      const float dummy0 = h0 * (isoJacInv00 + isoJacInv01) + h2 * isoJacInv02;
      const float dummy1 = h0 * (isoJacInv10 + isoJacInv11) + h2 * isoJacInv12;
      const float dummy2 = h0 * (isoJacInv20 + isoJacInv21) + h2 * isoJacInv22;

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
      const int whichGlobalNode = pElementNodes[8 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h0 = 4.f * zeta;
      const float h2 = 4.f * xi;
      const float dummy0 = h0 * isoJacInv00 + h2 * isoJacInv02;
      const float dummy1 = h0 * isoJacInv10 + h2 * isoJacInv12;
      const float dummy2 = h0 * isoJacInv20 + h2 * isoJacInv22;

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
      const int whichGlobalNode = pElementNodes[9 * elements_per_block];
      float value = 0.0f;
      if (lane_in_tile < 3)
        value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];

      const float NUx = tile.shfl(value, 0);
      const float NUy = tile.shfl(value, 1);
      const float NUz = tile.shfl(value, 2);

      const float h1 = 4.f * zeta;
      const float h2 = 4.f * eta;
      const float dummy0 = h1 * isoJacInv01 + h2 * isoJacInv02;
      const float dummy1 = h1 * isoJacInv11 + h2 * isoJacInv12;
      const float dummy2 = h1 * isoJacInv21 + h2 * isoJacInv22;

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

    // Write F to global memory if requested (for unit tests)
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
  // Compute 1st Piola-Kirchhoff stress tensor P (Mooney-Rivlin)
  // P = 2*hatJ*(alpha*I - mu01*hatJ*F*F^T)F + beta*F^{-T}
  // ============================================================
  {
    // Compute determinant of F, pad with minJthreshold to avoid singularity
    float dummy = F00 * (F11 * F22 - F12 * F21) - F01 * (F10 * F22 - F12 * F20) +
                  F02 * (F10 * F21 - F11 * F20);
    const float J = (dummy < minJthreshold ? minJthreshold : dummy);
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
    float alpha = mu10 + mu01 * I1 * hatJ;
    float beta = bulkK * (J - 1.0f) * J -
                 (2.0f / 3.0f) * hatJ * (mu10 * I1 + 2.0f * mu01 * I2 * hatJ);

    // Update PK1 to hold the matrix that multiplies F from the left
    bibi = -mu01 * hatJ;
    bibi *= (2.0f * hatJ);
    alpha *= (2.0f * hatJ);
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

      const float hx = h0 * (isoJacInv00 + isoJacInv01 + isoJacInv02);
      const float hy = h0 * (isoJacInv10 + isoJacInv11 + isoJacInv12);
      const float hz = h0 * (isoJacInv20 + isoJacInv21 + isoJacInv22);

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
          h0 * (PKone_00 * isoJacInv00 + PKone_01 * isoJacInv10 +
                PKone_02 * isoJacInv20);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce =
          h0 * (PKone_10 * isoJacInv00 + PKone_11 * isoJacInv10 +
                PKone_12 * isoJacInv20);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce =
          h0 * (PKone_20 * isoJacInv00 + PKone_21 * isoJacInv10 +
                PKone_22 * isoJacInv20);
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
          h1 * (PKone_00 * isoJacInv01 + PKone_01 * isoJacInv11 +
                PKone_02 * isoJacInv21);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce =
          h1 * (PKone_10 * isoJacInv01 + PKone_11 * isoJacInv11 +
                PKone_12 * isoJacInv21);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce =
          h1 * (PKone_20 * isoJacInv01 + PKone_21 * isoJacInv11 +
                PKone_22 * isoJacInv21);
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
          h2 * (PKone_00 * isoJacInv02 + PKone_01 * isoJacInv12 +
                PKone_02 * isoJacInv22);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 0, internalForce,
                                 forceScalingFactor);

      internalForce =
          h2 * (PKone_10 * isoJacInv02 + PKone_11 * isoJacInv12 +
                PKone_12 * isoJacInv22);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes,
                                 whichGlobalNode, 1, internalForce,
                                 forceScalingFactor);

      internalForce =
          h2 * (PKone_20 * isoJacInv02 + PKone_21 * isoJacInv12 +
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

      const float hx = h0 * isoJacInv00 + h1 * (isoJacInv01 + isoJacInv02);
      const float hy = h0 * isoJacInv10 + h1 * (isoJacInv11 + isoJacInv12);
      const float hz = h0 * isoJacInv20 + h1 * (isoJacInv21 + isoJacInv22);

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
      const float hx = h0 * isoJacInv00 + h1 * isoJacInv01;
      const float hy = h0 * isoJacInv10 + h1 * isoJacInv11;
      const float hz = h0 * isoJacInv20 + h1 * isoJacInv21;

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
          h0 * isoJacInv00 + h1 * isoJacInv01 + h2 * isoJacInv02;
      const float hy =
          h0 * isoJacInv10 + h1 * isoJacInv11 + h2 * isoJacInv12;
      const float hz =
          h0 * isoJacInv20 + h1 * isoJacInv21 + h2 * isoJacInv22;

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
      const float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
      const float hx = h0 * (isoJacInv00 + isoJacInv01) + h2 * isoJacInv02;
      const float hy = h0 * (isoJacInv10 + isoJacInv11) + h2 * isoJacInv12;
      const float hz = h0 * (isoJacInv20 + isoJacInv21) + h2 * isoJacInv22;

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
      const float hx = h0 * isoJacInv00 + h2 * isoJacInv02;
      const float hy = h0 * isoJacInv10 + h2 * isoJacInv12;
      const float hz = h0 * isoJacInv20 + h2 * isoJacInv22;

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
      const float hx = h1 * isoJacInv01 + h2 * isoJacInv02;
      const float hy = h1 * isoJacInv11 + h2 * isoJacInv12;
      const float hz = h1 * isoJacInv21 + h2 * isoJacInv22;

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
}

/**
 * Returns required shared memory size for the internal force kernel.
 *
 * Shared memory is used for F, invJacobian, and PK1 matrices (each 9 floats per
 * thread).
 *
 * @param blockSize Number of threads per block (should be 64)
 * @return Required shared memory in bytes
 */
inline size_t getInternalForceKernelSharedMemSize(int blockSize = 64) {
  return 3 * 9 * blockSize * sizeof(float);  // 3 matrices * 9 components *
                                              // blockSize threads
}
