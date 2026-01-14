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

template <typename TileType>
__device__ __forceinline__ void reduce_scale_and_atomicAdd(
    const TileType& tile,
    int lane_in_tile,
    float* __restrict__ pInternalForceNodes,
    int whichGlobalNode,
    int component,                 // 0=x, 1=y, 2=z
    float internalForce,
    float forceScalingFactor)
{
    // Scale per-QP contribution (do this before reduction if scale differs per lane/QP)
    internalForce *= forceScalingFactor;

    // 4-lane tile reduction: 4 -> 2 -> 1
    internalForce += tile.shfl_down(internalForce, 2);
    internalForce += tile.shfl_down(internalForce, 1);

    // Only lane 0 writes out
    if (lane_in_tile == 0) {
        atomicAdd(&pInternalForceNodes[3 * whichGlobalNode + component], internalForce);
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
  * @warning This kernel should be run with blocks of 64 threads because of the memory layout, to get coalesced memory accesses.
  *
  * @param pPosNodes                The positions of the nodes. [I]
  * @param pElement_NodeIndexes     The indices of the nodes that make up each element, listed element by element. [I]
  * @param pIsoMapInverse           Pointer to the inverse of the isoparametric-element Jacobian. [I]
  * @param writeOutDefGradientF     Whether to write out the deformation gradient F to global memory. [I]
  * @param pDeformationGradientF    The deformation gradient F of the element. [O]
  * @param pInternalForceNodes      The internal force of the nodes. [O]
 */
 
__global__ void internalF_KelvinVoigt_4QP(
     const double* __restrict__ pPosNodes,
     const int* __restrict__ pElement_NodeIndexes,
     const float* __restrict__ pIsoMapInverse,
     bool writeOutDefGradientF,
     float* __restrict__ pDeformationGradientF,
     float* __restrict__ pInternalForceNodes) {
   // Define a tile of four threads; each thread handles one quadrature point (QP) of one element.
   constexpr int TILE = 4;
   namespace cg = cooperative_groups;
   cg::thread_block block = cg::this_thread_block();
   cg::thread_block_tile<TILE> tile = cg::tiled_partition<TILE>(block);
   const int lane_in_tile = tile.thread_rank();
 
   // Calculate which element this tile of threads is responsible for
   const int elements_per_block = blockDim.x / TILE;
   const int element_idx = blockIdx.x * elements_per_block + tile.meta_group_rank(); // which element this tile is responsible for

   extern __shared__ float shMem[]; // 
   float *s_F = shMem;
   float *s_invJacobian = s_F + 9 * blockDim.x; // NOTE: blockDim.x = 4 QPs $\times$ N elements in this block.
   float *s_PK1 = s_invJacobian + 9 * blockDim.x; // Stores for this QP the PK1 stress tensor.
 
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
 
   
   // some index arithmetic, to pick up the nodes to read their positions from global memory.
   // We store all first nodes for all elements; then all second nodes for all elements; etc. 
   // This SoA (Structure of Arrays) layout leads to a coalesced memory access pattern. 
   // The threads in a block hit 16 successive indexes at each load. They are also nicely aligned in the global memory.
   const int baseIdx = blockIdx.x * TILE * elements_per_block * 9 + threadIdx.x; // this is where this thread starts in the global QP index space
   constexpr int nodes_per_element = 10;
   const int baseIdxNodes = blockIdx.x * nodes_per_element * elements_per_block + tile.meta_group_rank();
   const int* __restrict__ pElementNodes = pElement_NodeIndexes + baseIdxNodes; // 10 nodes per element
 
   // here we go, start computing the deformation gradient F of the iso-parametric element.
   {
    

   // Load the inverse of the isoparametric-element parent-reference Jacobian for this QP, into shared memory.
   // baseQPIdx is consecutive within the block, so these loads are coalesced within a block.
   // Consequently, the memory accesses are coalesced within a tile as well.
   // Stores all Jac00 for 64 matrices, then Jac01 for 64 matrices, etc. It's a SoA (Structure of Arrays) layout.
   s_invJacobian[threadIdx.x + 0 * blockDim.x] = __ldg(&pIsoMapInverse[baseIdx + 0 * blockDim.x]);
   s_invJacobian[threadIdx.x + 1 * blockDim.x] = __ldg(&pIsoMapInverse[baseIdx + 1 * blockDim.x]);
   s_invJacobian[threadIdx.x + 2 * blockDim.x] = __ldg(&pIsoMapInverse[baseIdx + 2 * blockDim.x]);
   s_invJacobian[threadIdx.x + 3 * blockDim.x] = __ldg(&pIsoMapInverse[baseIdx + 3 * blockDim.x]);
   s_invJacobian[threadIdx.x + 4 * blockDim.x] = __ldg(&pIsoMapInverse[baseIdx + 4 * blockDim.x]);
   s_invJacobian[threadIdx.x + 5 * blockDim.x] = __ldg(&pIsoMapInverse[baseIdx + 5 * blockDim.x]);
   s_invJacobian[threadIdx.x + 6 * blockDim.x] = __ldg(&pIsoMapInverse[baseIdx + 6 * blockDim.x]);
   s_invJacobian[threadIdx.x + 7 * blockDim.x] = __ldg(&pIsoMapInverse[baseIdx + 7 * blockDim.x]);
   s_invJacobian[threadIdx.x + 8 * blockDim.x] = __ldg(&pIsoMapInverse[baseIdx + 8 * blockDim.x]);

   // some nomenclature first, for convenience:
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
       // read from global memory; this is expensive since we read from all over the memory
       if(lane_in_tile<3) 
         value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];
 
       const float NUx = tile.shfl(value, 0); // x of node 0
       const float NUy = tile.shfl(value, 1); // y of node 0
       const float NUz = tile.shfl(value, 2); // z of node 0
   
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
   
     // Node 1 (of 0-9)
     {
       const int whichGlobalNode = pElementNodes[1 * elements_per_block];
       float value = 0.0f;
       // read from global memory; this is expensive since we read from all over the memory
       if(lane_in_tile<3) 
         value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];
 
       const float NUx = tile.shfl(value, 0); // x of node 1
       const float NUy = tile.shfl(value, 1); // y of node 1
       const float NUz = tile.shfl(value, 2); // z of node 1
   
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
   
     // Node 2 (of 0-9)
     {
       const int whichGlobalNode = pElementNodes[2 * elements_per_block];
       float value = 0.0f;
       if(lane_in_tile<3) 
         value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];
 
       const float NUx = tile.shfl(value, 0); // x of node 2
       const float NUy = tile.shfl(value, 1); // y of node 2
       const float NUz = tile.shfl(value, 2); // z of node 2
   
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
   
     // Node 3 (of 0-9)
     {
       const int whichGlobalNode = pElementNodes[3 * elements_per_block];
       float value = 0.0f;
       if(lane_in_tile<3) 
         value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];
 
       const float NUx = tile.shfl(value, 0); // x of node 3
       const float NUy = tile.shfl(value, 1); // y of node 3
       const float NUz = tile.shfl(value, 2); // z of node 3
 
       // h0 = 0.f;
       // h1 = 0.f;
       const float h2 = 4.f * zeta - 1.f;
       const float dummy0 = h2*isoJacInv02;
       const float dummy1 = h2*isoJacInv12;
       const float dummy2 = h2*isoJacInv22;
   
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
   
     // Node 4
     {
       const int whichGlobalNode = pElementNodes[4 * elements_per_block];
       float value = 0.0f;
       if(lane_in_tile<3) 
         value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];
 
       const float NUx = tile.shfl(value, 0); // x of node 4
       const float NUy = tile.shfl(value, 1); // y of node 4
       const float NUz = tile.shfl(value, 2); // z of node 4
   
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
   
     // Node 5 (of 0-9)
     {
       const int whichGlobalNode = pElementNodes[5 * elements_per_block];
       float value = 0.0f;
       if(lane_in_tile<3) 
         value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];
 
       const float NUx = tile.shfl(value, 0); // x of node 5
       const float NUy = tile.shfl(value, 1); // y of node 5
       const float NUz = tile.shfl(value, 2); // z of node 5
   
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
   
     // Node 6 (of 0-9)
     {
       const int whichGlobalNode = pElementNodes[6 * elements_per_block];
       float value = 0.0f;
       if(lane_in_tile<3) 
         value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];
 
       const float NUx = tile.shfl(value, 0); // x of node 6
       const float NUy = tile.shfl(value, 1); // y of node 6
       const float NUz = tile.shfl(value, 2); // z of node 6
   
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
   
     // Node 7 (of 0-9)
     {
       const int whichGlobalNode = pElementNodes[7 * elements_per_block];
       float value = 0.0f;
       if(lane_in_tile<3) 
         value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];
 
       const float NUx = tile.shfl(value, 0); // x of node 7
       const float NUy = tile.shfl(value, 1); // y of node 7
       const float NUz = tile.shfl(value, 2); // z of node 7
 
       const float h0 = -4.f * zeta;
       // h1 = h0;
       const float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
       const float dummy0 = h0*(isoJacInv00 + isoJacInv01) + h2*isoJacInv02;
       const float dummy1 = h0*(isoJacInv10 + isoJacInv11) + h2*isoJacInv12;
       const float dummy2 = h0*(isoJacInv20 + isoJacInv21) + h2*isoJacInv22;
   
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
       if(lane_in_tile<3) 
         value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];
 
       const float NUx = tile.shfl(value, 0); // x of node 8
       const float NUy = tile.shfl(value, 1); // y of node 8
       const float NUz = tile.shfl(value, 2); // z of node 8
 
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
   
     // Node 9 (of 0-9)
     {
       const int whichGlobalNode = pElementNodes[9 * elements_per_block];
       float value = 0.0f;
       if(lane_in_tile<3) 
         value = (float)pPosNodes[3 * whichGlobalNode + lane_in_tile];
 
       const float NUx = tile.shfl(value, 0); // x of node 9
       const float NUy = tile.shfl(value, 1); // y of node 9
       const float NUz = tile.shfl(value, 2); // z of node 9
 
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

     // Write back to global memory, for later used by other kernels, e.g., Kelvin-Voigt.
     // Loading in a different kernel is less painful since reads will be nicely coalesced.
     if (writeOutDefGradientF) {
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
   // End of computation of deformation gradient F, for the iso-parametric T10 tet element - 10 nodes, 4 QPs.

   // Compute the 1st Piola-Kirchhoff stress tensor P.
   // P = 2*hatJ*(alpha*I - mu01*hatJ*F*F^T)F + beta*F^{-T}
   { 
    // compute the determinant of the deformation gradient F; pad with minJthreshold to avoid division by zero.
    float dummy = F00 * (F11 * F22 - F12 * F21) - F01 * (F10 * F22 - F12 * F20) + F02 * (F10 * F21 - F11 * F20);
    const float J = (dummy<minJthreshold ? minJthreshold : dummy);
    float invJ = 1.0f / J;  

    dummy = cbrtf(J);
    dummy = 1.0f / dummy;
    float hatJ = dummy * dummy;  // this is J^{-2/3}

    // PK1 is first holding to B = F * F^T (symmetric), then I1 = tr(B)
    PKone_00 = F00*F00 + F01*F01 + F02*F02;
    PKone_11 = F10*F10 + F11*F11 + F12*F12;
    PKone_22 = F20*F20 + F21*F21 + F22*F22;
    PKone_01 = F00*F10 + F01*F11 + F02*F12;
    PKone_10 = PKone_01;
    PKone_02 = F00*F20 + F01*F21 + F02*F22;
    PKone_20 = PKone_02;
    PKone_12 = F10*F20 + F11*F21 + F12*F22;
    PKone_21 = PKone_12;
    dummy = PKone_00 + PKone_11 + PKone_22;
    float bibi = PKone_00*PKone_00 + PKone_11*PKone_11 + PKone_22*PKone_22 + 2.0f*(PKone_01*PKone_01 + PKone_02*PKone_02 + PKone_12*PKone_12); // tr(B*B)
    bibi = 0.5f * (dummy*dummy - bibi); // I2 = 1/2 (I1^2 - tr(B*B))
    float I1 = dummy; // first invariant of the deformation gradient
    float I2 = bibi; // second invariant of the deformation gradient
    float alpha = mu10 + mu01 * I1 * hatJ; // helper variable, see ME751 lecture notes
    float beta = bulkK * (J - 1.0f) * J - (2.0f/3.0f) * hatJ * (mu10*I1 + 2.0f*mu01*I2*hatJ); // helper variable, see ME751 lecture notes

    // update now to contain the matrix that multiplies F, from the left.
    bibi = -mu01*hatJ;
    bibi *= (2.0f * hatJ);
    alpha *= (2.0f * hatJ);
    PKone_00 = alpha + bibi * PKone_00; // there's an identity matrix that kicks in here
    PKone_11 = alpha + bibi * PKone_11; // there's an identity matrix that kicks in here
    PKone_22 = alpha + bibi * PKone_22; // there's an identity matrix that kicks in here

    PKone_01 = bibi * PKone_01;
    PKone_10 = bibi * PKone_10;
    PKone_02 = bibi * PKone_02;
    PKone_20 = bibi * PKone_20;
    PKone_12 = bibi * PKone_12;
    PKone_21 = bibi * PKone_21;

    // Note taht at this point PK1 hangs on to an intermediate matrix that is symmetric.
    // We'll use this to out advange, along with the fact that several variables are not 
    // needed at this point, and as such we'll recycle them.
    // NOTE: do not overwrite beta, it's still needed.
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

    // P = intermediate_Matrix * F + beta*F^{-T}
    // --- G = F^{-T} calculation snippet ---
    // Note: we overwrite 1.0f / J since it's not needed at this point.
    invJ *= beta;

    // Row 0 of G (using rows 1 and 2 of F)
    PKone_00 = fmaf(F11, F22, -(F12 * F21)) * invJ; // this is the F^{-T} component
    PKone_00 += intermediate_Matrix00*F00 + intermediate_Matrix01*F10 + intermediate_Matrix02*F20;
    
    PKone_01 = fmaf(F12, F20, -(F10 * F22)) * invJ; // this is the F^{-T} component
    PKone_01 += intermediate_Matrix00*F01 + intermediate_Matrix01*F11 + intermediate_Matrix02*F21;
    
    PKone_02 = fmaf(F10, F21, -(F11 * F20)) * invJ; // this is the F^{-T} component
    PKone_02 += intermediate_Matrix00*F02 + intermediate_Matrix01*F12 + intermediate_Matrix02*F22;


    //Row 1 (using rows 0 and 2 of F); use the symmetry of intermediate_Matrix
    PKone_10 = fmaf(F02, F21, -(F01 * F22)) * invJ; // this is the F^{-T} component
    PKone_10 += intermediate_Matrix01*F00 + intermediate_Matrix11*F10 + intermediate_Matrix12*F20;
    
    PKone_11 = fmaf(F00, F22, -(F02 * F20)) * invJ; // this is the F^{-T} component
    PKone_11 += intermediate_Matrix01*F01 + intermediate_Matrix11*F11 + intermediate_Matrix12*F21;
    
    PKone_12 = fmaf(F01, F20, -(F00 * F21)) * invJ; // this is the F^{-T} component
    PKone_12 += intermediate_Matrix01*F02 + intermediate_Matrix11*F12 + intermediate_Matrix12*F22;


    // Row 2 of G (using rows 0 and 1 of F); use the symmetry of intermediate_Matrix
    PKone_20 = fmaf(F01, F12, -(F02 * F11)) * invJ; // this is the F^{-T} component
    PKone_20 += intermediate_Matrix02*F00 + intermediate_Matrix12*F10 + intermediate_Matrix22*F20;
    PKone_21 = fmaf(F02, F10, -(F00 * F12)) * invJ; // this is the F^{-T} component
    PKone_21 += intermediate_Matrix02*F01 + intermediate_Matrix12*F11 + intermediate_Matrix22*F21;
    PKone_22 = fmaf(F00, F11, -(F01 * F10)) * invJ; // this is the F^{-T} component
    PKone_22 += intermediate_Matrix02*F02 + intermediate_Matrix12*F12 + intermediate_Matrix22*F22;

  }
  // end of computation of the 1st Piola-Kirchhoff stress tensor P.


  // Start of computation of the internal acceleration for the element.
  {
    float forceScalingFactor = 0.f;
    {
      // Need the jacobian of the parent-reference map, at this quadrature point.
      // Since we have the invJacobian in shared memory, we can use it here.
      forceScalingFactor += isoJacInv00 * (isoJacInv11 * isoJacInv22 - isoJacInv12 * isoJacInv21);
      forceScalingFactor -= isoJacInv01 * (isoJacInv10 * isoJacInv22 - isoJacInv12 * isoJacInv20);
      forceScalingFactor += isoJacInv02 * (isoJacInv10 * isoJacInv21 - isoJacInv11 * isoJacInv20);
      // At this point, forceScalingFactor is the determinant of inverse of the iso-map jacobian at this QP.
      // Invert to get the determinant of the iso-map at this QP.
      forceScalingFactor = 1.f / forceScalingFactor; // It better be that this is not blowing up here.

      // NOTE: all weights are the same for the T10 tet element.
      // Keep this here, with local scope to prevent others from inadvertently using it.
      constexpr float weightQP = 1.0f/24.0f; 

      // since we have the same weight for all QPs, we factor that in here:
      forceScalingFactor *= weightQP;
    }

    // start visiting the nodes and compute the internal force for each one. Use atomic adds to accumulate
    { //--------------------------------- Node 0 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[0 * elements_per_block];
      const float h0 = 4.f * eta + 4.f * xi + 4.f * zeta - 3.f;

      const float hx = h0*(isoJacInv00 + isoJacInv01 + isoJacInv02);
      const float hy = h0*(isoJacInv10 + isoJacInv11 + isoJacInv12);
      const float hz = h0*(isoJacInv20 + isoJacInv21 + isoJacInv22);

      float internalForce;

      internalForce = PKone_00*hx + PKone_01*hy + PKone_02*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 0, internalForce, forceScalingFactor);
      
      internalForce = PKone_10*hx + PKone_11*hy + PKone_12*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 1, internalForce, forceScalingFactor);
      
      internalForce = PKone_20*hx + PKone_21*hy + PKone_22*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 2, internalForce, forceScalingFactor);
    }


    
    { //--------------------------------- Node 1 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[1 * elements_per_block]; 
      const float h0 = 4.f * xi - 1.f;
      // h1 = h0;
      // h2 = h0;
      // hx = h0*isoJacInv00;
      // hy = h0*isoJacInv10;
      // hz = h0*isoJacInv20;

      float internalForce;

      internalForce = h0*(PKone_00*isoJacInv00 + PKone_01*isoJacInv10 + PKone_02*isoJacInv20);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 0, internalForce, forceScalingFactor);

      internalForce = h0*(PKone_10*isoJacInv00 + PKone_11*isoJacInv10 + PKone_12*isoJacInv20);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 1, internalForce, forceScalingFactor);

      internalForce = h0*(PKone_20*isoJacInv00 + PKone_21*isoJacInv10 + PKone_22*isoJacInv20);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 2, internalForce, forceScalingFactor);
    }

    
    { //--------------------------------- Node 2 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[2 * elements_per_block]; // why not storing in a reg? See comment at end of kernel.
      // h0 = 0.f;
      const float h1 = 4.f * eta - 1.f;
      // h2 = 0.f;
      // hx = h1*isoJacInv01;
      // hy = h1*isoJacInv11;
      // hz = h1*isoJacInv21;

      float internalForce;

      internalForce = h1*(PKone_00*isoJacInv01 + PKone_01*isoJacInv11 + PKone_02*isoJacInv21);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 0, internalForce, forceScalingFactor);

      internalForce = h1*(PKone_10*isoJacInv01 + PKone_11*isoJacInv11 + PKone_12*isoJacInv21);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 1, internalForce, forceScalingFactor);

      internalForce = h1*(PKone_20*isoJacInv01 + PKone_21*isoJacInv11 + PKone_22*isoJacInv21);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 2, internalForce, forceScalingFactor);
    }

    
    { //--------------------------------- Node 3 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[3 * elements_per_block]; // why not storing in a reg? See comment at end of kernel.
      // h0 = 0.f;
      // h1 = 0.f;
      const float h2 = 4.f * zeta - 1.f;
      // hx = h2*isoJacInv02;
      // hy = h2*isoJacInv12;
      // hz = h2*isoJacInv22;

      float internalForce;

      internalForce = h2*(PKone_00*isoJacInv02 + PKone_01*isoJacInv12 + PKone_02*isoJacInv22);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 0, internalForce, forceScalingFactor);

      internalForce = h2*(PKone_10*isoJacInv02 + PKone_11*isoJacInv12 + PKone_12*isoJacInv22);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 1, internalForce, forceScalingFactor);

      internalForce = h2*(PKone_20*isoJacInv02 + PKone_21*isoJacInv12 + PKone_22*isoJacInv22);
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 2, internalForce, forceScalingFactor);

    }

    
    { //--------------------------------- Node 4 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[4 * elements_per_block];

      const float h0 = -4.f * eta - 8.f * xi - 4.f * zeta + 4.f;
      const float h1 = -4.f * xi; // h1 == h2 for node 4

      const float hx = h0*isoJacInv00 + h1*(isoJacInv01 + isoJacInv02);
      const float hy = h0*isoJacInv10 + h1*(isoJacInv11 + isoJacInv12);
      const float hz = h0*isoJacInv20 + h1*(isoJacInv21 + isoJacInv22);

      float internalForce;

      internalForce = PKone_00*hx + PKone_01*hy + PKone_02*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 0, internalForce, forceScalingFactor);

      internalForce = PKone_10*hx + PKone_11*hy + PKone_12*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 1, internalForce, forceScalingFactor);

      internalForce = PKone_20*hx + PKone_21*hy + PKone_22*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 2, internalForce, forceScalingFactor);
    }


    
    { //--------------------------------- Node 5 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[5 * elements_per_block]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = 4.f * eta;
      const float h1 = 4.f * xi;
      // h2 = 0.f;
      const float hx = h0*isoJacInv00 + h1*isoJacInv01;
      const float hy = h0*isoJacInv10 + h1*isoJacInv11;
      const float hz = h0*isoJacInv20 + h1*isoJacInv21;

      float internalForce;

      internalForce = PKone_00*hx + PKone_01*hy + PKone_02*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 0, internalForce, forceScalingFactor);

      internalForce = PKone_10*hx + PKone_11*hy + PKone_12*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 1, internalForce, forceScalingFactor);

      internalForce = PKone_20*hx + PKone_21*hy + PKone_22*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 2, internalForce, forceScalingFactor);
    }

    
    { //--------------------------------- Node 6 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[6 * elements_per_block]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = -4.f * eta;
      const float h1 = -8.f * eta - 4.f * xi - 4.f * zeta + 4.f;
      const float h2 = -4.f * eta;
      const float hx = h0*isoJacInv00 + h1*isoJacInv01 + h2*isoJacInv02;
      const float hy = h0*isoJacInv10 + h1*isoJacInv11 + h2*isoJacInv12;
      const float hz = h0*isoJacInv20 + h1*isoJacInv21 + h2*isoJacInv22;

      float internalForce;

      internalForce = PKone_00*hx + PKone_01*hy + PKone_02*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 0, internalForce, forceScalingFactor);

      internalForce = PKone_10*hx + PKone_11*hy + PKone_12*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 1, internalForce, forceScalingFactor);

      internalForce = PKone_20*hx + PKone_21*hy + PKone_22*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 2, internalForce, forceScalingFactor);

    }

    
    { //--------------------------------- Node 7 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[7 * elements_per_block]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = -4.f * zeta;
      // h1 = h0;
      const float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
      const float hx = h0*(isoJacInv00 + isoJacInv01) + h2*isoJacInv02;
      const float hy = h0*(isoJacInv10 + isoJacInv11) + h2*isoJacInv12;
      const float hz = h0*(isoJacInv20 + isoJacInv21) + h2*isoJacInv22;

      float internalForce;

      internalForce = PKone_00*hx + PKone_01*hy + PKone_02*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 0, internalForce, forceScalingFactor);

      internalForce = PKone_10*hx + PKone_11*hy + PKone_12*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 1, internalForce, forceScalingFactor);

      internalForce = PKone_20*hx + PKone_21*hy + PKone_22*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 2, internalForce, forceScalingFactor);

    }

    
    { //--------------------------------- Node 8 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[8 * elements_per_block]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = 4.f * zeta;
      // h1 = 0.f;
      const float h2 = 4.f * xi;
      const float hx = h0*isoJacInv00 + h2*isoJacInv02;
      const float hy = h0*isoJacInv10 + h2*isoJacInv12;
      const float hz = h0*isoJacInv20 + h2*isoJacInv22;

      float internalForce;

      internalForce = PKone_00*hx + PKone_01*hy + PKone_02*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 0, internalForce, forceScalingFactor);

      internalForce = PKone_10*hx + PKone_11*hy + PKone_12*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 1, internalForce, forceScalingFactor);

      internalForce = PKone_20*hx + PKone_21*hy + PKone_22*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 2, internalForce, forceScalingFactor);

    }

    
    { //--------------------------------- Node 9 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[9 * elements_per_block]; // why not storing in a reg? See comment at end of kernel.
      // h0 = 0.f;
      const float h1 = 4.f * zeta;
      const float h2 = 4.f * eta;
      const float hx = h1*isoJacInv01 + h2*isoJacInv02;
      const float hy = h1*isoJacInv11 + h2*isoJacInv12;
      const float hz = h1*isoJacInv21 + h2*isoJacInv22;

      float internalForce;

      internalForce = PKone_00*hx + PKone_01*hy + PKone_02*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 0, internalForce, forceScalingFactor);

      internalForce = PKone_10*hx + PKone_11*hy + PKone_12*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 1, internalForce, forceScalingFactor);

      internalForce = PKone_20*hx + PKone_21*hy + PKone_22*hz;
      reduce_scale_and_atomicAdd(tile, lane_in_tile, pInternalForceNodes, whichGlobalNode, 2, internalForce, forceScalingFactor);

    }   
  }  
  // end of computation of the internal acceleration for the element.
  
  return;
}
 