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
__device__ __forceinline__ void applyNodalForce(
    const TileType& tile,
    int lane_in_tile,
    float* __restrict__ pInternalForceNodes,
    int whichGlobalNode,
    float hx, float hy, float hz,
    const float* __restrict__ s_F,
    float invJ,
    float iso_scale,   
    float alpha,
    float beta,        
    float mu01_hatJ)
{
    constexpr int offset = 64; //hardcoded :-). Strong assumption that we have 64 threads per block.
    const int baseIdxS = threadIdx.x;
    #define F00 s_F[baseIdxS + 0 * offset]
    #define F01 s_F[baseIdxS + 1 * offset]
    #define F02 s_F[baseIdxS + 2 * offset]
    #define F10 s_F[baseIdxS + 3 * offset]
    #define F11 s_F[baseIdxS + 4 * offset]
    #define F12 s_F[baseIdxS + 5 * offset]
    #define F20 s_F[baseIdxS + 6 * offset]
    #define F21 s_F[baseIdxS + 7 * offset]
    #define F22 s_F[baseIdxS + 8 * offset]  

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
        if (lane_in_tile == 0) {
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
        if (lane_in_tile == 0) {
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
        if (lane_in_tile == 0) {
            constexpr float weightQP = 1.0f/24.0f; //NOTE: all weights are the same for the T10 tet element.
            fZ *= weightQP;          
            atomicAdd(&pInternalForceNodes[3 * whichGlobalNode + 2], fZ);
        }            
    }
    #undef F00
    #undef F01
    #undef F02
    #undef F10
    #undef F11
    #undef F12
    #undef F20
    #undef F21
    #undef F22
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
   // In this block, we deal with 64 QPs: 16 elements, each with 4 QPs.
   // Define a tile of four threads; each thread handles one quadrature point (QP) of one element.
   constexpr int TILE = 4;
   namespace cg = cooperative_groups;
   cg::thread_block block = cg::this_thread_block();
   cg::thread_block_tile<TILE> tile = cg::tiled_partition<TILE>(block);
   const int lane_in_tile = tile.thread_rank();
 
   // Calculate which element this tile of threads is responsible for
   constexpr int elements_per_block = 16;
   const int element_idx = blockIdx.x * elements_per_block + tile.meta_group_rank();

   __shared__ float s_F[4 * elements_per_block * 9]; // 4 QPs, 16 elements, 9 entries in the deformation gradient F matrix.  
 
   if (element_idx >= totalN_Elements) {
     return;
   }
 
 
   // Define the total number of quadrature points in this block
   const int baseIdx = blockIdx.x * 4 * elements_per_block * 9 + threadIdx.x; // this is where this thread starts in the global QP index space
   constexpr int offset = elements_per_block * 4;

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

 
   // Load the inverse of the isoparametric-element parent-reference Jacobian for this QP.
   // baseQPIdx is consecutive within the block, so these loads are coalesced within a block.
   // Consequently, the memory accesses are coalesced within a tile as well.
   // Stores all Jac00 for 64 matrices, then Jac01 for 64 matrices, etc. It's a SoA (Structure of Arrays) layout.
   const float isoJacInv00 = __ldg(&pIsoMapInverse[baseIdx + 0 * offset]);
   const float isoJacInv01 = __ldg(&pIsoMapInverse[baseIdx + 1 * offset]);
   const float isoJacInv02 = __ldg(&pIsoMapInverse[baseIdx + 2 * offset]);
   const float isoJacInv10 = __ldg(&pIsoMapInverse[baseIdx + 3 * offset]);
   const float isoJacInv11 = __ldg(&pIsoMapInverse[baseIdx + 4 * offset]);
   const float isoJacInv12 = __ldg(&pIsoMapInverse[baseIdx + 5 * offset]);
   const float isoJacInv20 = __ldg(&pIsoMapInverse[baseIdx + 6 * offset]);
   const float isoJacInv21 = __ldg(&pIsoMapInverse[baseIdx + 7 * offset]);
   const float isoJacInv22 = __ldg(&pIsoMapInverse[baseIdx + 8 * offset]);
 
   // some index arithmetic, to pick up the nodes to read their positions from global memory.
   // We store all first nodes for all elements; then all second nodes for all elements; etc. 
   // This SoA (Structure of Arrays) layout leads to a coalesced memory access pattern. 
   // The threads in a block hit 16 successive indexes at each load. They are also nicely aligned in the global memory.
   constexpr int nodes_per_element = 10;
   constexpr int nodes_per_block = nodes_per_element * elements_per_block;
   const int baseIdxNodes = blockIdx.x * nodes_per_block + tile.meta_group_rank();
   constexpr int offsetNodes = elements_per_block;  
   const int* __restrict__ pElementNodes = pElement_NodeIndexes + baseIdxNodes; // 10 nodes per element
 
   // here we go, start computing the deformation gradient F of the iso-parametric element.
   {
    const int baseIdxS = threadIdx.x;
    #define F00 s_F[baseIdxS + 0 * offset]
    #define F01 s_F[baseIdxS + 1 * offset]
    #define F02 s_F[baseIdxS + 2 * offset]
    #define F10 s_F[baseIdxS + 3 * offset]
    #define F11 s_F[baseIdxS + 4 * offset]
    #define F12 s_F[baseIdxS + 5 * offset]
    #define F20 s_F[baseIdxS + 6 * offset]
    #define F21 s_F[baseIdxS + 7 * offset]
    #define F22 s_F[baseIdxS + 8 * offset]
 
 
     // Node 0 (of 0-9)
     {
       const int whichGlobalNode = pElementNodes[0 * offsetNodes];
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
       const int whichGlobalNode = pElementNodes[1 * offsetNodes];
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
       const int whichGlobalNode = pElementNodes[2 * offsetNodes];
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
       const int whichGlobalNode = pElementNodes[3 * offsetNodes];
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
       const int whichGlobalNode = pElementNodes[4 * offsetNodes];
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
       const int whichGlobalNode = pElementNodes[5 * offsetNodes];
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
       const int whichGlobalNode = pElementNodes[6 * offsetNodes];
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
       const int whichGlobalNode = pElementNodes[7 * offsetNodes];
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
       const int whichGlobalNode = pElementNodes[8 * offsetNodes];
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
       const int whichGlobalNode = pElementNodes[9 * offsetNodes];
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
         pDeformationGradientF[baseIdx + 0 * offset] = F00;
         pDeformationGradientF[baseIdx + 1 * offset] = F01;
         pDeformationGradientF[baseIdx + 2 * offset] = F02;
         pDeformationGradientF[baseIdx + 3 * offset] = F10;
         pDeformationGradientF[baseIdx + 4 * offset] = F11;
         pDeformationGradientF[baseIdx + 5 * offset] = F12;
         pDeformationGradientF[baseIdx + 6 * offset] = F20;
         pDeformationGradientF[baseIdx + 7 * offset] = F21;
         pDeformationGradientF[baseIdx + 8 * offset] = F22;
     }


    #undef F00
    #undef F01
    #undef F02
    #undef F10
    #undef F11
    #undef F12
    #undef F20
    #undef F21
    #undef F22    
   }
   // End of computation of deformation gradient F, for the iso-parametric T10 tet element - 10 nodes, 4 QPs.
 


  // Start of computation of the internal acceleration for the element.
  {
    const int baseIdxS = threadIdx.x;
    #define F00 s_F[baseIdxS + 0 * offset]
    #define F01 s_F[baseIdxS + 1 * offset]
    #define F02 s_F[baseIdxS + 2 * offset]
    #define F10 s_F[baseIdxS + 3 * offset]
    #define F11 s_F[baseIdxS + 4 * offset]
    #define F12 s_F[baseIdxS + 5 * offset]
    #define F20 s_F[baseIdxS + 6 * offset]
    #define F21 s_F[baseIdxS + 7 * offset]
    #define F22 s_F[baseIdxS + 8 * offset]


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
    #undef F00
    #undef F01
    #undef F02
    #undef F10
    #undef F11
    #undef F12
    #undef F20
    #undef F21
    #undef F22


    // P*h = 2*hatJ*(alpha*u - mu01*hatJ*w) + beta*(F^{-T}*h)
    // start visiting the nodes and compute the internal force for each one. Use atomic adds to accumulate
    { //--------------------------------- Node 0 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[0 * offsetNodes]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = 4.f * eta + 4.f * xi + 4.f * zeta - 3.f;
      // h1 = h0;
      // h2 = h0;
      const float hx = h0*(isoJacInv00 + isoJacInv01 + isoJacInv02);
      const float hy = h0*(isoJacInv10 + isoJacInv11 + isoJacInv12);
      const float hz = h0*(isoJacInv20 + isoJacInv21 + isoJacInv22);

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        s_F,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 1 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[1 * offsetNodes]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = 4.f * xi - 1.f;
      // h1 = h0;
      // h2 = h0;
      const float hx = h0*isoJacInv00;
      const float hy = h0*isoJacInv10;
      const float hz = h0*isoJacInv20;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        s_F,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 2 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[2 * offsetNodes]; // why not storing in a reg? See comment at end of kernel.
      // h0 = 0.f;
      const float h1 = 4.f * eta - 1.f;
      // h2 = 0.f;
      const float hx = h1*isoJacInv01;
      const float hy = h1*isoJacInv11;
      const float hz = h1*isoJacInv21;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        s_F,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 3 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[3 * offsetNodes]; // why not storing in a reg? See comment at end of kernel.
      // h0 = 0.f;
      // h1 = 0.f;
      const float h2 = 4.f * zeta - 1.f;
      const float hx = h2*isoJacInv02;
      const float hy = h2*isoJacInv12;
      const float hz = h2*isoJacInv22;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        s_F,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 4 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[4 * offsetNodes]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = -4.f * eta - 8.f * xi - 4.f * zeta + 4.f;
      const float h1 = -4.f * xi;
      // h1 and h2 are the same
      const float hx = h0*isoJacInv00 + h1*(isoJacInv01 + isoJacInv02);
      const float hy = h0*isoJacInv10 + h1*(isoJacInv11 + isoJacInv12);
      const float hz = h0*isoJacInv20 + h1*(isoJacInv21 + isoJacInv22);

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        s_F,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 5 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[5 * offsetNodes]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = 4.f * eta;
      const float h1 = 4.f * xi;
      // h2 = 0.f;
      const float hx = h0*isoJacInv00 + h1*isoJacInv01;
      const float hy = h0*isoJacInv10 + h1*isoJacInv11;
      const float hz = h0*isoJacInv20 + h1*isoJacInv21;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        s_F,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 6 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[6 * offsetNodes]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = -4.f * eta;
      const float h1 = -8.f * eta - 4.f * xi - 4.f * zeta + 4.f;
      const float h2 = -4.f * eta;
      const float hx = h0*isoJacInv00 + h1*isoJacInv01 + h2*isoJacInv02;
      const float hy = h0*isoJacInv10 + h1*isoJacInv11 + h2*isoJacInv12;
      const float hz = h0*isoJacInv20 + h1*isoJacInv21 + h2*isoJacInv22;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        s_F,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 7 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[7 * offsetNodes]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = -4.f * zeta;
      // h1 = h0;
      const float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
      const float hx = h0*(isoJacInv00 + isoJacInv01) + h2*isoJacInv02;
      const float hy = h0*(isoJacInv10 + isoJacInv11) + h2*isoJacInv12;
      const float hz = h0*(isoJacInv20 + isoJacInv21) + h2*isoJacInv22;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        s_F,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 8 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[8 * offsetNodes]; // why not storing in a reg? See comment at end of kernel.
      const float h0 = 4.f * zeta;
      // h1 = 0.f;
      const float h2 = 4.f * xi;
      const float hx = h0*isoJacInv00 + h2*isoJacInv02;
      const float hy = h0*isoJacInv10 + h2*isoJacInv12;
      const float hz = h0*isoJacInv20 + h2*isoJacInv22;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        s_F,
        invJ, 
        iso_scale,
        alpha, 
        beta,
        mu01_hatJ
      );
    }

    
    { //--------------------------------- Node 9 (of 0-9) ---------------------------------
      const int whichGlobalNode = pElementNodes[9 * offsetNodes]; // why not storing in a reg? See comment at end of kernel.
      // h0 = 0.f;
      const float h1 = 4.f * zeta;
      const float h2 = 4.f * eta;
      const float hx = h1*isoJacInv01 + h2*isoJacInv02;
      const float hy = h1*isoJacInv11 + h2*isoJacInv12;
      const float hz = h1*isoJacInv21 + h2*isoJacInv22;

      applyNodalForce(
        tile, lane_in_tile, pInternalForceNodes, whichGlobalNode,
        hx, hy, hz, 
        s_F,
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
 