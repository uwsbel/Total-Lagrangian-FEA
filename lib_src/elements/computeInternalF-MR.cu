/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2026, Simulation Based Engineering Lab
 * All rights reserved.
 *
 * Contributors:
 *   - Dan Negrut
 *   - <Name 2>
 *   - <Name 3>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <cooperative_groups.h>
#include "tlfea_problem_parameters.cuh" // stores parameters associated with the TL_FEA model for this particular problem

// The arrays below are used to coordinate the four threads in a tile.

// The NU information is stored as AoS, xyz-xyz-xyz-xyz-etc.
// The arrays below are used to coordinate the four threads in a tile.
__device__ __constant__ int nodeOffsets_T10tet[8][4] = {
    {0, 0, 0, 3}, // first pass: node 0 (x,y,z), node 3 (x)
    {1, 1, 1, 3}, // second pass: node 1 (x,y,z), node 3 (y)
    {2, 2, 2, 3}, // third pass: node 2 (x,y,z), node 3 (z)
    {4, 4, 4, 7}, // fourth pass: node 4 (x,y,z), node 7 (x); jump from 2 to 4 since 3 has been read 
    {5, 5, 5, 7}, // fifth pass: node 5 (x,y,z), node 7 (y)
    {6, 6, 6, 7}, // sixth pass: node 6 (x,y,z), node 7 (z)
    {8, 8, 8, 8}, // seventh pass: node 8 (x,y,z), node 8 (**bogus**); the 8 is bogus, not used
    {9, 9, 9, 9}  // eighth pass: node 9 (x,y,z), node 9 (**bogus**); the 9 is bogus, not used
};

__device__ __constant__ int xyzFieldOffsets_T10tet[8][4] = {
    {0, 1, 2, 0}, // first pass: node 0 (x,y,z), node 3 (x)
    {0, 1, 2, 1}, // second pass: node 1 (x,y,z), node 3 (y)
    {0, 1, 2, 2}, // third pass: node 2 (x,y,z), node 3 (z)
    {0, 1, 2, 0}, // fourth pass: node 4 (x,y,z), node 7 (x); jump from 2 to 4 since 3 has been read 
    {0, 1, 2, 1}, // fifth pass: node 5 (x,y,z), node 7 (y)
    {0, 1, 2, 2}, // sixth pass: node 6 (x,y,z), node 7 (z)
    {0, 1, 2, 2}, // seventh pass: node 8 (x,y,z), node 8 (**bogus**); the 8 is bogus, not used
    {0, 1, 2, 2}  // eighth pass: node 9 (x,y,z), node 9 (**bogus**); the 9 is bogus, not used
};


template <typename TileType>
__device__ __inline__ void applyNodalForce(
//  __device__ __forceinline__ void applyNodalForce(
    const TileType& tile,
    int lane_in_tile,
    float* __restrict__ pInternalForceNodes,
    int whichGlobalNode,
    float hx, float hy, float hz,
    float F00, float F01, float F02,
    float F10, float F11, float F12,
    float F20, float F21, float F22,
    float invJ,
    float iso_scale,   
    float alpha,
    float beta,        
    float mu01_hatJ)
{
    // u = F*h (row-major)
    const float ux = fmaf(F00, hx, fmaf(F01, hy, F02 * hz));
    const float uy = fmaf(F10, hx, fmaf(F11, hy, F12 * hz));
    const float uz = fmaf(F20, hx, fmaf(F21, hy, F22 * hz));

    // v = F^T * u
    const float vx = fmaf(F00, ux, fmaf(F10, uy, F20 * uz));
    const float vy = fmaf(F01, ux, fmaf(F11, uy, F21 * uz));
    const float vz = fmaf(F02, ux, fmaf(F12, uy, F22 * uz));

    // w = F * v
    const float wx = fmaf(F00, vx, fmaf(F01, vy, F02 * vz));
    const float wy = fmaf(F10, vx, fmaf(F11, vy, F12 * vz));
    const float wz = fmaf(F20, vx, fmaf(F21, vy, F22 * vz));

    // X
    {
        const float cof_x = fmaf(F11, F22, -(F12 * F21));
        const float cof_y = fmaf(F12, F20, -(F10 * F22));
        const float cof_z = fmaf(F10, F21, -(F11 * F20));
        const float gtx   = fmaf(cof_x, hx, fmaf(cof_y, hy, cof_z * hz)) * invJ;

        float fX = fmaf(iso_scale, fmaf(alpha, ux, -mu01_hatJ * wx), beta * gtx);
        fX += tile.shfl_down(fX, 2);
        fX += tile.shfl_down(fX, 1);
        if (lane_in_tile == 0){
          constexpr float weightQP = 1.0f/24.0f; //NOTE: all weights are the same for the T10 tet element.
          fX *= weightQP;          
            atomicAdd(&pInternalForceNodes[3 * whichGlobalNode + 0], fX);
        }
    }

    // Y
    {
        const float cof_x = fmaf(F02, F21, -(F01 * F22));
        const float cof_y = fmaf(F00, F22, -(F02 * F20));
        const float cof_z = fmaf(F01, F20, -(F00 * F21));
        const float gty   = fmaf(cof_x, hx, fmaf(cof_y, hy, cof_z * hz)) * invJ;

        float fY = fmaf(iso_scale, fmaf(alpha, uy, -mu01_hatJ * wy), beta * gty);
        fY += tile.shfl_down(fY, 2);
        fY += tile.shfl_down(fY, 1);
        if (lane_in_tile == 0){
          constexpr float weightQP = 1.0f/24.0f; //NOTE: all weights are the same for the T10 tet element.
          fY *= weightQP;          
          atomicAdd(&pInternalForceNodes[3 * whichGlobalNode + 1], fY);
        } 
    }

    // Z
    {
        const float cof_x = fmaf(F01, F12, -(F02 * F11));
        const float cof_y = fmaf(F02, F10, -(F00 * F12));
        const float cof_z = fmaf(F00, F11, -(F01 * F10));
        const float gtz   = fmaf(cof_x, hx, fmaf(cof_y, hy, cof_z * hz)) * invJ;

        float fZ = fmaf(iso_scale, fmaf(alpha, uz, -mu01_hatJ * wz), beta * gtz);
        fZ += tile.shfl_down(fZ, 2);
        fZ += tile.shfl_down(fZ, 1);
        if (lane_in_tile == 0){
          constexpr float weightQP = 1.0f/24.0f; //NOTE: all weights are the same for the T10 tet element.
          fZ *= weightQP;          
          atomicAdd(&pInternalForceNodes[3 * whichGlobalNode + 2], fZ);
        }            
    }
}


/**
 * @brief Kernel computing the internal force associated with one element for Mooney-Rivlin material with 4 quadrature points.
 *
 * @details
 * This kernel computes the internal forces associated with all nodes of one element for Mooney-Rivlin material.
 * In general, one node belongs to multiple elements. As such, we use atomic add to accumulate the internal forces from all elements that a node belongs to.
 * The kernel operates per element, with four threads working together—each thread handles one quadrature point (QP).
 * Each element has four quadrature points. Computation is performed in float, although the position and velocity are in double precision.
 * Each element has 10 nodes. Four threads cooperate (one per QP) to compute QP-dependent quantities and assemble per-element contributions.
 * These threads communicate through shuffle operations to add their contributions to the internal force of each node.
 *
 * @param pPosNodes                The positions of the nodes.
 * @param pVelNodes                The velocities of the nodes.
 * @param pInternalForceNodes      The internal force of the nodes.
 * @param pElement_NodeIndexes     The indices of the nodes that make up each element, listed element by element.
 * @param pIsoMapInverseScaled     Pointer to the inverse of the isoparametric-element Jacobian. All entries are already scaled by the
 *                                 QP weight and determinant of the isoparametric map (computation can be done offline and in double precision,
 *                                 then saved as float, compromising less the precision).
 */

__global__ void computeInternalForceContributionPerElement_MR_4QP(
    const double* __restrict__ pPosNodes, const double* __restrict__ pVelNodes,
    const int* __restrict__ pElement_NodeIndexes,
    float* __restrict__ pInternalForceNodes,
    const float* __restrict__ pIsoMapInverseScaled) {
  // Four threads will load nodal unknowns from global memory, and then use them to compute the internal force contribution.

  // Define a tile of four threads; each thread handles one quadrature point (QP)
  constexpr int TILE = 4;
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<TILE> tile = cg::tiled_partition<TILE>(block);
  const int lane_in_tile = tile.thread_rank();

  // Calculate which element this tile of threads is responsible for
  const int elements_per_block = blockDim.x / TILE;
  const int element_idx = blockIdx.x * elements_per_block + tile.meta_group_rank();

  if (element_idx >= totalN_Elements) {
    return;
  }

  // QP coordinates for the 4-point rule on the T10 tet element canonical tetrahedron.
  // Lane mapping:
  //  lane 0: (a,a,a)
  //  lane 1: (b,a,a)
  //  lane 2: (b,b,a)
  //  lane 3: (b,b,b)  
  constexpr float a = 0.1381966011250105f;
  constexpr float b = 0.5854101966249685f;

  const float xi   = (lane_in_tile == 0) ? a : b;
  const float eta  = (lane_in_tile <  2) ? a : b;
  const float zeta = (lane_in_tile <  3) ? a : b;

  // start the computation of the deformation gradient F of the iso-parametric element.
  float F00, F01, F02, F10, F11, F12, F20, F21, F22;
  const int* __restrict__ pElementNodes = pElement_NodeIndexes + element_idx * 10; // 10 nodes per element

  // Define the total number of quadrature points globally
  constexpr int totalQPs = totalN_Elements * 4;
  const int globalQPIdx  = element_idx * 4 + lane_in_tile;

  // Load the inverse of the isoparametric-element parent-reference Jacobian for this QP.
  // globalQPIdx is consecutive within each 4-lane tile, so these loads are coalesced within a tile.
  const float isoJacInv00 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 0 * totalQPs]);
  const float isoJacInv01 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 1 * totalQPs]);
  const float isoJacInv02 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 2 * totalQPs]);
  const float isoJacInv10 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 3 * totalQPs]);
  const float isoJacInv11 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 4 * totalQPs]);
  const float isoJacInv12 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 5 * totalQPs]);
  const float isoJacInv20 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 6 * totalQPs]);
  const float isoJacInv21 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 7 * totalQPs]);
  const float isoJacInv22 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 8 * totalQPs]);

  {  
  
    // Used to store "serendipitous" (i.e., bonus) nodal unknowns.
    float NUx_bonus, NUy_bonus, NUz_bonus;

    // Node 0 (with node 3 bonus x)
    {
      const int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[0][lane_in_tile]];
      const float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[0][lane_in_tile]];
      const float NUx = tile.shfl(value, 0); // x of node 0
      const float NUy = tile.shfl(value, 1); // y of node 0
      const float NUz = tile.shfl(value, 2); // z of node 0
      NUx_bonus = tile.shfl(value, 3); // x of node 3
  
      const float h0 = 4.f * eta + 4.f * xi + 4.f * zeta - 3.f;
      // h1 = h0;
      // h2 = h0;
      const float dummy0 = h0*(isoJacInv00 + isoJacInv01 + isoJacInv02);
      const float dummy1 = h0*(isoJacInv10 + isoJacInv11 + isoJacInv12);
      const float dummy2 = h0*(isoJacInv20 + isoJacInv21 + isoJacInv22);
  
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
  
    // Node 1 (with node 3 bonus y)
    {
      const int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[1][lane_in_tile]];
      const float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[1][lane_in_tile]];
      const float NUx = tile.shfl(value, 0); // x of node 1
      const float NUy = tile.shfl(value, 1); // y of node 1
      const float NUz = tile.shfl(value, 2); // z of node 1
      NUy_bonus = tile.shfl(value, 3); // y of node 3
  
      const float h0 = 4.f * xi - 1.f;
      // h1 = 0.f;
      // h2 = 0.f;
      const float dummy0 = h0*isoJacInv00;
      const float dummy1 = h0*isoJacInv10;
      const float dummy2 = h0*isoJacInv20;
  
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
  
    // Node 2 (with node 3 bonus z)
    {
      const int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[2][lane_in_tile]];
      const float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[2][lane_in_tile]];
      const float NUx = tile.shfl(value, 0); // x of node 2
      const float NUy = tile.shfl(value, 1); // y of node 2
      const float NUz = tile.shfl(value, 2); // z of node 2
      NUz_bonus = tile.shfl(value, 3); // z of node 3
  
      // h0 = 0.f;
      const float h1 = 4.f * eta - 1.f;
      // h2 = 0.f;
      const float dummy0 = h1*isoJacInv01;
      const float dummy1 = h1*isoJacInv11;
      const float dummy2 = h1*isoJacInv21;
  
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
  
    // Node 3 (bonus, use shuffled bonuses)
    {
      // h0 = 0.f;
      // h1 = 0.f;
      const float h2 = 4.f * zeta - 1.f;
      const float dummy0 = h2*isoJacInv02;
      const float dummy1 = h2*isoJacInv12;
      const float dummy2 = h2*isoJacInv22;
  
      F00 += NUx_bonus * dummy0;
      F01 += NUx_bonus * dummy1;
      F02 += NUx_bonus * dummy2;
      F10 += NUy_bonus * dummy0;
      F11 += NUy_bonus * dummy1;
      F12 += NUy_bonus * dummy2;
      F20 += NUz_bonus * dummy0;
      F21 += NUz_bonus * dummy1;
      F22 += NUz_bonus * dummy2;
    }
  
    // Node 4 (with node 7 bonus x)
    {
      const int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[3][lane_in_tile]];
      const float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[3][lane_in_tile]];
      const float NUx = tile.shfl(value, 0); // x of node 4
      const float NUy = tile.shfl(value, 1); // y of node 4
      const float NUz = tile.shfl(value, 2); // z of node 4
      NUx_bonus = tile.shfl(value, 3); // x of node 7
  
      const float h0 = -4.f * eta - 8.f * xi - 4.f * zeta + 4.f;
      const float h1 = -4.f * xi;
      // h1 and h2 are the same
      const float dummy0 = h0*isoJacInv00 + h1*(isoJacInv01 + isoJacInv02);
      const float dummy1 = h0*isoJacInv10 + h1*(isoJacInv11 + isoJacInv12);
      const float dummy2 = h0*isoJacInv20 + h1*(isoJacInv21 + isoJacInv22);
  
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
  
    // Node 5 (with node 7 bonus y)
    {
      const int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[4][lane_in_tile]];
      const float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[4][lane_in_tile]];
      const float NUx = tile.shfl(value, 0); // x of node 5
      const float NUy = tile.shfl(value, 1); // y of node 5
      const float NUz = tile.shfl(value, 2); // z of node 5
      NUy_bonus = tile.shfl(value, 3); // y of node 7
  
      const float h0 = 4.f * eta;
      const float h1 = 4.f * xi;
      // h2 = 0.f;
      const float dummy0 = h0*isoJacInv00 + h1*isoJacInv01;
      const float dummy1 = h0*isoJacInv10 + h1*isoJacInv11;
      const float dummy2 = h0*isoJacInv20 + h1*isoJacInv21;
  
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
  
    // Node 6 (with node 7 bonus z)
    {
      const int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[5][lane_in_tile]];
      const float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[5][lane_in_tile]];
      const float NUx = tile.shfl(value, 0); // x of node 6
      const float NUy = tile.shfl(value, 1); // y of node 6
      const float NUz = tile.shfl(value, 2); // z of node 6
      NUz_bonus = tile.shfl(value, 3); // z of node 7
  
      const float h0 = -4.f * eta;
      const float h1 = -8.f * eta - 4.f * xi - 4.f * zeta + 4.f;
      const float h2 = -4.f * eta;
      const float dummy0 = h0*isoJacInv00 + h1*isoJacInv01 + h2*isoJacInv02;
      const float dummy1 = h0*isoJacInv10 + h1*isoJacInv11 + h2*isoJacInv12;
      const float dummy2 = h0*isoJacInv20 + h1*isoJacInv21 + h2*isoJacInv22;
  
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
  
    // Node 7 (bonus, use all shuffled bonuses)
    {
      const float h0 = -4.f * zeta;
      // h1 = h0;
      const float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
      const float dummy0 = h0*(isoJacInv00 + isoJacInv01) + h2*isoJacInv02;
      const float dummy1 = h0*(isoJacInv10 + isoJacInv11) + h2*isoJacInv12;
      const float dummy2 = h0*(isoJacInv20 + isoJacInv21) + h2*isoJacInv22;
  
      F00 += NUx_bonus * dummy0;
      F01 += NUx_bonus * dummy1;
      F02 += NUx_bonus * dummy2;
      F10 += NUy_bonus * dummy0;
      F11 += NUy_bonus * dummy1;
      F12 += NUy_bonus * dummy2;
      F20 += NUz_bonus * dummy0;
      F21 += NUz_bonus * dummy1;
      F22 += NUz_bonus * dummy2;
    }
  
    // Node 8 (no valid bonus, dummy)
    {
      const int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[6][lane_in_tile]];
      const float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[6][lane_in_tile]];
      const float NUx = tile.shfl(value, 0); // x of node 8
      const float NUy = tile.shfl(value, 1); // y of node 8
      const float NUz = tile.shfl(value, 2); // z of node 8
      NUz_bonus = tile.shfl(value, 3); // z of node 8; dummy
  
      const float h0 = 4.f * zeta;
      // h1 = 0.f;
      const float h2 = 4.f * xi;
      const float dummy0 = h0*isoJacInv00 + h2*isoJacInv02;
      const float dummy1 = h0*isoJacInv10 + h2*isoJacInv12;
      const float dummy2 = h0*isoJacInv20 + h2*isoJacInv22;
  
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
  
    // Node 9 (no valid bonus, dummy)
    {
      const int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[7][lane_in_tile]];
      const float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[7][lane_in_tile]];
      const float NUx = tile.shfl(value, 0); // x of node 9
      const float NUy = tile.shfl(value, 1); // y of node 9
      const float NUz = tile.shfl(value, 2); // z of node 9
      NUz_bonus = tile.shfl(value, 3); // z of node 9; dummy
  
      // h0 = 0.f;
      const float h1 = 4.f * zeta;
      const float h2 = 4.f * eta;
      const float dummy0 = h1*isoJacInv01 + h2*isoJacInv02;
      const float dummy1 = h1*isoJacInv11 + h2*isoJacInv12;
      const float dummy2 = h1*isoJacInv21 + h2*isoJacInv22;
  
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
  }
  // End of computation of deformation gradient F, for the iso-parametric T10 tet element - 10 nodes, 4 QPs.

  // Start of computation of the internal acceleration for the element.
  {

    // compute the determinant of the deformation gradient F; pad with minJthreshold to avoid division by zero.
    float dummy = F00 * (F11 * F22 - F12 * F21) - F01 * (F10 * F22 - F12 * F20) + F02 * (F10 * F21 - F11 * F20);
    const float J = (dummy<minJthreshold ? minJthreshold : dummy);

    dummy = cbrtf(J);
    dummy = 1.0f / dummy;
    const float hatJ = dummy * dummy;  // J^{-2/3}
    const float invJ = 1.0f / J;  
    const float iso_scale = 2.0f * hatJ;
    const float mu01_hatJ = mu01 * hatJ;

    float bibi;
    {
      // B = F * F^T (symmetric), then I1 = tr(B)
      const float B00 = F00*F00 + F01*F01 + F02*F02;
      const float B11 = F10*F10 + F11*F11 + F12*F12;
      const float B22 = F20*F20 + F21*F21 + F22*F22;
      const float B01 = F00*F10 + F01*F11 + F02*F12;
      const float B02 = F00*F20 + F01*F21 + F02*F22;
      const float B12 = F10*F20 + F11*F21 + F12*F22;
      dummy = B00 + B11 + B22;
      bibi = B00*B00 + B11*B11 + B22*B22 + 2.0f*(B01*B01 + B02*B02 + B12*B12); // tr(B*B)
      bibi = 0.5f * (dummy*dummy - bibi); // I2 = 1/2 (I1^2 - tr(B*B))
    }
    const float I1 = dummy; // first invariant of the deformation gradient
    const float I2 = bibi; // second invariant of the deformation gradient
    const float alpha = mu10 + mu01 * I1 * hatJ; // helper variable, see ME751 lecture notes
    const float beta = bulkK * (J - 1.0f) * J - (2.0f/3.0f) * hatJ * (mu10*I1 + 2.0f*mu01*I2*hatJ); // helper variable, see ME751 lecture notes


    // P*h = 2*hatJ*(alpha*u - mu01*hatJ*w) + beta*(F^{-T}*h)
    // start visiting the nodes and compute the internal force for each one. Use atomic adds to accumulate
    { //--------------------------------- Node 0 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[0]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = 4.f * eta + 4.f * xi + 4.f * zeta - 3.f;
      // h1 = h0;
      // h2 = h0;
      const float hx = h0*(isoJacInv00 + isoJacInv01 + isoJacInv02);
      const float hy = h0*(isoJacInv10 + isoJacInv11 + isoJacInv12);
      const float hz = h0*(isoJacInv20 + isoJacInv21 + isoJacInv22);

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        F00, F01, F02, F10, F11, F12, F20, F21, F22,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 1 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[1]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = 4.f * xi - 1.f;
      // h1 = h0;
      // h2 = h0;
      const float hx = h0*isoJacInv00;
      const float hy = h0*isoJacInv10;
      const float hz = h0*isoJacInv20;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        F00, F01, F02, F10, F11, F12, F20, F21, F22,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 2 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[2]; // why not storing in a reg? See comment at end of kernel.
      // h0 = 0.f;
      const float h1 = 4.f * eta - 1.f;
      // h2 = 0.f;
      const float hx = h1*isoJacInv01;
      const float hy = h1*isoJacInv11;
      const float hz = h1*isoJacInv21;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        F00, F01, F02, F10, F11, F12, F20, F21, F22,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 3 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[3]; // why not storing in a reg? See comment at end of kernel.
      // h0 = 0.f;
      // h1 = 0.f;
      const float h2 = 4.f * zeta - 1.f;
      const float hx = h2*isoJacInv02;
      const float hy = h2*isoJacInv12;
      const float hz = h2*isoJacInv22;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        F00, F01, F02, F10, F11, F12, F20, F21, F22,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 4 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[4]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = -4.f * eta - 8.f * xi - 4.f * zeta + 4.f;
      const float h1 = -4.f * xi;
      // h1 and h2 are the same
      const float hx = h0*isoJacInv00 + h1*(isoJacInv01 + isoJacInv02);
      const float hy = h0*isoJacInv10 + h1*(isoJacInv11 + isoJacInv12);
      const float hz = h0*isoJacInv20 + h1*(isoJacInv21 + isoJacInv22);

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        F00, F01, F02, F10, F11, F12, F20, F21, F22,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 5 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[5]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = 4.f * eta;
      const float h1 = 4.f * xi;
      // h2 = 0.f;
      const float hx = h0*isoJacInv00 + h1*isoJacInv01;
      const float hy = h0*isoJacInv10 + h1*isoJacInv11;
      const float hz = h0*isoJacInv20 + h1*isoJacInv21;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        F00, F01, F02, F10, F11, F12, F20, F21, F22,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 6 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[6]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = -4.f * eta;
      const float h1 = -8.f * eta - 4.f * xi - 4.f * zeta + 4.f;
      const float h2 = -4.f * eta;
      const float hx = h0*isoJacInv00 + h1*isoJacInv01 + h2*isoJacInv02;
      const float hy = h0*isoJacInv10 + h1*isoJacInv11 + h2*isoJacInv12;
      const float hz = h0*isoJacInv20 + h1*isoJacInv21 + h2*isoJacInv22;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        F00, F01, F02, F10, F11, F12, F20, F21, F22,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 7 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[7]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = -4.f * zeta;
      // h1 = h0;
      const float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
      const float hx = h0*(isoJacInv00 + isoJacInv01) + h2*isoJacInv02;
      const float hy = h0*(isoJacInv10 + isoJacInv11) + h2*isoJacInv12;
      const float hz = h0*(isoJacInv20 + isoJacInv21) + h2*isoJacInv22;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        F00, F01, F02, F10, F11, F12, F20, F21, F22,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 8 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[8]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = 4.f * zeta;
      // h1 = 0.f;
      const float h2 = 4.f * xi;
      const float hx = h0*isoJacInv00 + h2*isoJacInv02;
      const float hy = h0*isoJacInv10 + h2*isoJacInv12;
      const float hz = h0*isoJacInv20 + h2*isoJacInv22;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        F00, F01, F02, F10, F11, F12, F20, F21, F22,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 9 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[9]; // why not storing in a reg? See comment at end of kernel.
      // h0 = 0.f;
      const float h1 = 4.f * zeta;
      const float h2 = 4.f * eta;
      const float hx = h1*isoJacInv01 + h2*isoJacInv02;
      const float hy = h1*isoJacInv11 + h2*isoJacInv12;
      const float hz = h1*isoJacInv21 + h2*isoJacInv22;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        F00, F01, F02, F10, F11, F12, F20, F21, F22,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }   
  }  
  // end of computation of the internal acceleration for the element.
  
  return;
}
/*
 * DESIGN DECISION: Global Memory Reload vs. Register Storage
 * --------------------------------------------------------
 * For a T10 element (10 nodes), we purposefully choose to reload nodal indices 
 * and positions from global memory during the internal force assembly phase 
 * rather than storing them in registers. 
 *
 * 1. REGISTER PRESSURE & OCCUPANCY:
 * - Each element has 10 nodes. Storing 10 global indices and 30 coordinates (xyz)
 * per tile would consume ~40 registers per thread just for geometry storage.
 * - By reloading, we keep the register footprint low (target < 64 regs/thread), 
 * allowing for maximum occupancy (up to 512 elements/tiles per SM on RTX 4080/5080).
 * - Higher occupancy is critical for hiding the functional latency of the 
 * heavy floating-point math in the deformation gradient and stress calculations.
 *
 * 2. CACHE COHERENCY ("HOT" RELOAD):
 * - The pElement_NodeIndexes and pPosNodes data were accessed at the start 
 * of the kernel to compute the deformation gradient (F). 
 * - Total working set for 512 active elements is approx 80 KB (512 * 160B). 
 * Since modern SMs have 128 KB of L1/Shared memory, this data is 
 * guaranteed to be "hot" in the L1 or L2 cache.
 * - A cache-hit reload (latency ~30-50 cycles) is easily hidden by the 
 * scheduler given our high occupancy.
 *
 * 3. BROADCAST EFFICIENCY:
 * - Hardware-level constant memory and L1 controllers automatically handle 
 * broadcasting when multiple threads in a tile request the same index. 
 * Manual register-shuffling or broadcasting would increase instruction 
 * overhead without reducing memory transactions.
 */