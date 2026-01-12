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

  // define tiles with four threads each; one thread per QP
  constexpr int TILE = 4;
  namespace cg                  = cooperative_groups;
  cg::thread_block block        = cg::this_thread_block();
  cg::thread_block_tile<TILE> tile = cg::tiled_partition<TILE>(block);

  int elements_per_block = blockDim.x / TILE;
  int element_idx = blockIdx.x * elements_per_block + tile.meta_group_rank();

  if (element_idx >= totalN_Elements) {
    return;
  }

  // The coordinates of the four quadrature points.
  // The vertices of the canonical tetrahedron are:
  // (0,0,0), (1,0,0), (0,1,0), (0,0,1)
  // The coordinates of the four quadrature points are:
  // (0.1381966011250105, 0.1381966011250105, 0.1381966011250105)
  // (0.5854101966249685, 0.1381966011250105, 0.1381966011250105)
  // (0.5854101966249685, 0.5854101966249685, 0.1381966011250105)
  // (0.5854101966249685, 0.5854101966249685, 0.5854101966249685)

  float xi, eta, zeta;
  const int tile_thread_rank = tile.thread_rank();
  switch (tile_thread_rank) {
    case 0:
      xi   = 0.1381966011250105f;
      eta  = 0.1381966011250105f;
      zeta = 0.1381966011250105f;
      break;
    case 1:
      xi   = 0.5854101966249685f;
      eta  = 0.1381966011250105f;
      zeta = 0.1381966011250105f;
      break;
    case 2:
      xi   = 0.5854101966249685f;
      eta  = 0.5854101966249685f;
      zeta = 0.1381966011250105f;
      break;
    case 3:
      xi   = 0.5854101966249685f;
      eta  = 0.5854101966249685f;
      zeta = 0.5854101966249685f;
      break;
  }

  // start the computation of the deformation gradient F of the iso-parametric element.
  float isoJacInv00, isoJacInv01, isoJacInv02;
  float isoJacInv10, isoJacInv11, isoJacInv12;
  float isoJacInv20, isoJacInv21, isoJacInv22;

  {
    // Define the total number of quadrature points globally
    const int totalQPs    = totalN_Elements * 4;
    const int globalQPIdx = element_idx * 4 + tile_thread_rank;

    // Load the inverse of the scaled isoparametric-element Jacobian for this QP.
    // globalQPIdx is consecutive within each 4-lane tile, so these loads are coalesced within a tile.
    isoJacInv00 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 0 * totalQPs]);
    isoJacInv01 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 1 * totalQPs]);
    isoJacInv02 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 2 * totalQPs]);
    isoJacInv10 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 3 * totalQPs]);
    isoJacInv11 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 4 * totalQPs]);
    isoJacInv12 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 5 * totalQPs]);
    isoJacInv20 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 6 * totalQPs]);
    isoJacInv21 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 7 * totalQPs]);
    isoJacInv22 = __ldg(&pIsoMapInverseScaled[globalQPIdx + 8 * totalQPs]);
  }

  // Begin internal force contribution calculation using the Jacobian H.
  // The exact formula for H is given below.
  /*
      [See long multiline LaTeX comment above]
  */

  // Compute the deformation gradient F using the nodal unknowns.
  float F00, F01, F02, F10, F11, F12, F20, F21, F22;
  float NUx_bonus, NUy_bonus, NUz_bonus;
  const int* __restrict__ pElementNodes = pElement_NodeIndexes + element_idx * 10; // 10 nodes per element
  
  // Node 0 (with node 3 bonus x)
  {
    int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[0][tile_thread_rank]];
    float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[0][tile_thread_rank]];
    float NUx = tile.shfl(value, 0); // x of node 0
    float NUy = tile.shfl(value, 1); // y of node 0
    float NUz = tile.shfl(value, 2); // z of node 0
    NUx_bonus = tile.shfl(value, 3); // x of node 3

    float h0 = 4.f * eta + 4.f * xi + 4.f * zeta - 3.f;
    // h1 = h0;
    // h2 = h0;
    float dummy0 = h0*(isoJacInv00 + isoJacInv01 + isoJacInv02);
    float dummy1 = h0*(isoJacInv10 + isoJacInv11 + isoJacInv12);
    float dummy2 = h0*(isoJacInv20 + isoJacInv21 + isoJacInv22);

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
    int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[1][tile_thread_rank]];
    float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[1][tile_thread_rank]];
    float NUx = tile.shfl(value, 0); // x of node 1
    float NUy = tile.shfl(value, 1); // y of node 1
    float NUz = tile.shfl(value, 2); // z of node 1
    NUy_bonus = tile.shfl(value, 3); // y of node 3

    float h0 = 4.f * xi - 1.f;
    // h1 = 0.f;
    // h2 = 0.f;
    float dummy0 = h0*isoJacInv00;
    float dummy1 = h0*isoJacInv10;
    float dummy2 = h0*isoJacInv20;

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
    int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[2][tile_thread_rank]];
    float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[2][tile_thread_rank]];
    float NUx = tile.shfl(value, 0); // x of node 2
    float NUy = tile.shfl(value, 1); // y of node 2
    float NUz = tile.shfl(value, 2); // z of node 2
    NUz_bonus = tile.shfl(value, 3); // z of node 3

    // h0 = 0.f;
    float h1 = 4.f * eta - 1.f;
    // h2 = 0.f;
    float dummy0 = h1*isoJacInv01;
    float dummy1 = h1*isoJacInv11;
    float dummy2 = h1*isoJacInv21;

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
    float h2 = 4.f * zeta - 1.f;
    float dummy0 = h2*isoJacInv02;
    float dummy1 = h2*isoJacInv12;
    float dummy2 = h2*isoJacInv22;

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
    int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[3][tile_thread_rank]];
    float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[3][tile_thread_rank]];
    float NUx = tile.shfl(value, 0); // x of node 4
    float NUy = tile.shfl(value, 1); // y of node 4
    float NUz = tile.shfl(value, 2); // z of node 4
    NUx_bonus = tile.shfl(value, 3); // x of node 7

    float h0 = -4.f * eta - 8.f * xi - 4.f * zeta + 4.f;
    float h1 = -4.f * xi;
    // h1 and h2 are the same
    float dummy0 = h0*isoJacInv00 + h1*(isoJacInv01 + isoJacInv02);
    float dummy1 = h0*isoJacInv10 + h1*(isoJacInv11 + isoJacInv12);
    float dummy2 = h0*isoJacInv20 + h1*(isoJacInv21 + isoJacInv22);

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
    int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[4][tile_thread_rank]];
    float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[4][tile_thread_rank]];
    float NUx = tile.shfl(value, 0); // x of node 5
    float NUy = tile.shfl(value, 1); // y of node 5
    float NUz = tile.shfl(value, 2); // z of node 5
    NUy_bonus = tile.shfl(value, 3); // y of node 7

    float h0 = 4.f * eta;
    float h1 = 4.f * xi;
    // h2 = 0.f;
    float dummy0 = h0*isoJacInv00 + h1*isoJacInv01;
    float dummy1 = h0*isoJacInv10 + h1*isoJacInv11;
    float dummy2 = h0*isoJacInv20 + h1*isoJacInv21;

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
    int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[5][tile_thread_rank]];
    float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[5][tile_thread_rank]];
    float NUx = tile.shfl(value, 0); // x of node 6
    float NUy = tile.shfl(value, 1); // y of node 6
    float NUz = tile.shfl(value, 2); // z of node 6
    NUz_bonus = tile.shfl(value, 3); // z of node 7

    float h0 = -4.f * eta;
    float h1 = -8.f * eta - 4.f * xi - 4.f * zeta + 4.f;
    float h2 = -4.f * eta;
    float dummy0 = h0*isoJacInv00 + h1*isoJacInv01 + h2*isoJacInv02;
    float dummy1 = h0*isoJacInv10 + h1*isoJacInv11 + h2*isoJacInv12;
    float dummy2 = h0*isoJacInv20 + h1*isoJacInv21 + h2*isoJacInv22;

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
    float h0 = -4.f * zeta;
    // h1 = h0;
    float h2 = -4.f * eta - 4.f * xi - 8.f * zeta + 4.f;
    float dummy0 = h0*(isoJacInv00 + isoJacInv01) + h2*isoJacInv02;
    float dummy1 = h0*(isoJacInv10 + isoJacInv11) + h2*isoJacInv12;
    float dummy2 = h0*(isoJacInv20 + isoJacInv21) + h2*isoJacInv22;

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
    int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[6][tile_thread_rank]];
    float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[6][tile_thread_rank]];
    float NUx = tile.shfl(value, 0); // x of node 8
    float NUy = tile.shfl(value, 1); // y of node 8
    float NUz = tile.shfl(value, 2); // z of node 8
    NUz_bonus = tile.shfl(value, 3); // z of node 8; dummy

    float h0 = 4.f * zeta;
    // h1 = 0.f;
    float h2 = 4.f * xi;
    float dummy0 = h0*isoJacInv00 + h2*isoJacInv02;
    float dummy1 = h0*isoJacInv10 + h2*isoJacInv12;
    float dummy2 = h0*isoJacInv20 + h2*isoJacInv22;

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
    int whichGlobalNode = pElementNodes[nodeOffsets_T10tet[7][tile_thread_rank]];
    float value = (float)pPosNodes[3 * whichGlobalNode + xyzFieldOffsets_T10tet[7][tile_thread_rank]];
    float NUx = tile.shfl(value, 0); // x of node 9
    float NUy = tile.shfl(value, 1); // y of node 9
    float NUz = tile.shfl(value, 2); // z of node 9
    NUz_bonus = tile.shfl(value, 3); // z of node 9; dummy

    // h0 = 0.f;
    float h1 = 4.f * zeta;
    float h2 = 4.f * eta;
    float dummy0 = h1*isoJacInv01 + h2*isoJacInv02;
    float dummy1 = h1*isoJacInv11 + h2*isoJacInv12;
    float dummy2 = h1*isoJacInv21 + h2*isoJacInv22;

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
  // This conclude the computation of the deformation gradient F of the iso-parametric element.

  #define internalAcceleration_X NUx_bonus  // Alias the register; NUx_bonus not needed from this point on
  #define internalAcceleration_Y NUy_bonus  // Alias the register; NUy_bonus not needed from this point on
  #define internalAcceleration_Z NUz_bonus  // Alias the register; NUz_bonus not needed from this point on

  // Use the macro 'internalAcceleration_?' for internal acceleration calculation
  // (The following statement is a placeholder for demonstration)
  internalAcceleration_X = F00 + F01 + F02; //BOGUS FOR NOW, to get the atomic add operations going
  internalAcceleration_Y = F10 + F11 + F12; //BOGUS FOR NOW, to get the atomic add operations going
  internalAcceleration_Z = F20 + F21 + F22; //BOGUS FOR NOW, to get the atomic add operations going


  // At this point, we just computed the internal acceleration for the QP.
  // Need to do a reduce within the tile to get the internal acceleration for node "i" (of 0-9) in the element.
  // tree reduction: 4 -> 2 -> 1

  internalAcceleration_X += tile.shfl_down(internalAcceleration_X, 2);  
  internalAcceleration_Y += tile.shfl_down(internalAcceleration_Y, 2);  
  internalAcceleration_Z += tile.shfl_down(internalAcceleration_Z, 2);

  internalAcceleration_X += tile.shfl_down(internalAcceleration_X, 1);  
  internalAcceleration_Y += tile.shfl_down(internalAcceleration_Y, 1);  
  internalAcceleration_Z += tile.shfl_down(internalAcceleration_Z, 1);

  if (tile_thread_rank == 0) {
    const int whichGlobalNode = pElementNodes[0];
    atomicAdd(&pInternalForceNodes[3 * whichGlobalNode + 0], internalAcceleration_X);
    atomicAdd(&pInternalForceNodes[3 * whichGlobalNode + 1], internalAcceleration_Y);
    atomicAdd(&pInternalForceNodes[3 * whichGlobalNode + 2], internalAcceleration_Z);
  }


  // Remove the macro definition 
  #undef internalAcceleration_X
  #undef internalAcceleration_Y
  #undef internalAcceleration_Z

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