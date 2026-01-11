#include <cooperative_groups.h>

/**
 * @brief Kernel computing the internal force associated with one element for
 * Mooney-Rivlin material with 4 quadrature points.
 *
 * @details This kernel computes the internal forces associated with all nodes
 * of one element for Mooney-Rivlin material. Note that in general, one node
 * belongs to mutiple elements. As such, we use an atomic add to accumulate the
 * internal forces from all elements that the node belongs to. The kernel is per
 * element, and four threads work together, each thread handles one quadrature
 * point. This element has four quadrature points. Computation is done in float,
 * although the position and velocity are double precision. Each element has 4
 * nodes associated with it, each node working on one quadrature point. These
 * threads communicate through shuffle operations, to add their contributions to
 * the internal force of the node. To that end, the kernel uses
 *
 * @param pPosNodes The position of the nodes.
 * @param pVelNodes The velocity of the nodes.
 * @param pInternalForceNodes The internal force of the nodes.
 * @param pElement_NodeIndexes The indexes of the nodes that make up each element. Listed element by element.
 * @param pIsoMapInverseScaled Pointer to the inverse of the
 * isoparametric-element Jacobian. Note: all entries are alrady scaled by the
 * weight of the QP and the determinant of the isoparametric map. This is done
 * since the computation can be done offline and in double precision; then saved
 * as float, thus compromising less the precision.
 */
__global__ void computeInternalForceContributionPerElement_MR_4QP(
    const double* __restrict__ pPosNodes, const double* __restrict__ pVelNodes,
    const int* __restrict__ pElement_NodeIndexes,
    float* __restrict__ pInternalForceNodes,
    const float* __restrict__ pIsoMapInverseScaled) {
  // Four threads will load nodal unknowns from global memory, and then use them to compute the internal force contribution.
  // The arrays below are used to coordinate the four threads in a tile.
  // The NU information is stores as AoS, xyz-xyz-xyz-xyz-etc. The arrays below are used to coordinate the four threads in a tile.
  const int nodeOffsets[8][4] = {
    {0, 0, 0, 3}, // first pass to read in data from node 0 (x,y,z), and node 3 (x)
    {1, 1, 1, 3}, // second pass to read in data from node 1 (x,y,z), and node 3 (y)
    {2, 2, 2, 3}, // third pass to read in data from node 2 (x,y,z), and node 3 (z)
    {4, 4, 4, 7}, // fourth pass to read in data from node 4 (x,y,z), and node 7 (x); jumping from 2 to 4 since 3 has been read 
    {5, 5, 5, 7}, // fifth pass to read in data from node 5 (x,y,z), and node 7 (y)
    {6, 6, 6, 7}, // sixth pass to read in data from node 6 (x,y,z), and node 7 (z)
    {8, 8, 8, 8}, // seventh pass to read in data from node 8 (x,y,z), and node 8 (**bogus**); note that the 8 is bogus, not used
    {9, 9, 9, 9}, // eighth pass to read in data from last node of the element (x,y,z), and node 9 (**bogus**); note that the 9 is bogus, not used
  };

  const int xyzFieldOffsets[8][4] = {
    {0, 1, 2, 0}, // first pass to read in data from node 0 (x,y,z), and node 3 (x)
    {0, 1, 2, 1}, // second pass to read in data from node 1 (x,y,z), and node 3 (y)
    {0, 1, 2, 2}, // third pass to read in data from node 2 (x,y,z), and node 3 (z)
    {0, 1, 2, 0}, // fourth pass to read in data from node 4 (x,y,z), and node 7 (x); jumping from 2 to 4 since 3 has been read 
    {0, 1, 2, 1}, // fifth pass to read in data from node 5 (x,y,z), and node 7 (y)
    {0, 6, 2, 2}, // sixth pass to read in data from node 6 (x,y,z), and node 7 (z)
    {0, 1, 2, 2}, // seventh pass to read in data from node 8 (x,y,z), and node 8 (**bogus**); note that the 8 is bogus, not used
    {0, 1, 2, 2}, // eighth pass to read in data from node 9 (x,y,z), and node 9 (**bogus**); note that the 9 is bogus, not used
  };

  // define tiles with four threads each; one thread per QP
  namespace cg                  = cooperative_groups;
  cg::thread_block block        = cg::this_thread_block();
  cg::thread_block_tile<4> tile = cg::tiled_partition<4>(block);

  // there are 16 elements per block. The index of the element this tile is working on:
  int element_idx = blockIdx.x * 16 + tile.meta_group_rank();

  // the coordinates of the four quarature points.
  // The vertices of the canonical tetrahedron are:
  // (0,0,0), (1,0,0), (0,1,0), (0,0,1)
  // The coordinates of the four quarature points are:
  // (0.1381966011250105, 0.1381966011250105, 0.1381966011250105)
  // (0.5854101966249685, 0.1381966011250105, 0.1381966011250105)
  // (0.5854101966249685, 0.5854101966249685, 0.1381966011250105)
  // (0.5854101966249685, 0.5854101966249685, 0.5854101966249685)
  float xi, eta, zeta;
  switch (tile.thread_rank()) {
    case 0:
      xi   = 0.1381966011250105;
      eta  = 0.1381966011250105;
      zeta = 0.1381966011250105;
      break;
    case 1:
      xi   = 0.5854101966249685;
      eta  = 0.1381966011250105;
      zeta = 0.1381966011250105;
      break;
    case 2:
      xi   = 0.5854101966249685;
      eta  = 0.5854101966249685;
      zeta = 0.1381966011250105;
      break;
    case 3:
      xi   = 0.5854101966249685;
      eta  = 0.5854101966249685;
      zeta = 0.5854101966249685;
      break;
  }

  float isoJacInv00, isoJacInv01, isoJacInv02, isoJacInv10, isoJacInv11,
      isoJacInv12, isoJacInv20, isoJacInv21, isoJacInv22;
  // Next block of code brings in the inverse of the scaled
  // isoparametric-element Jacobian.
  {
    // Bring in the inverse of the scaled isoparametric-element Jacobian.
    // Data stored in AoS format, for the four QPs. Helps with cache coherence.
    int offset  = element_idx * 9 + tile.thread_rank();
    isoJacInv00 = pIsoMapInverseScaled[offset++];
    isoJacInv01 = pIsoMapInverseScaled[offset++];
    isoJacInv02 = pIsoMapInverseScaled[offset++];
    isoJacInv10 = pIsoMapInverseScaled[offset++];
    isoJacInv11 = pIsoMapInverseScaled[offset++];
    isoJacInv12 = pIsoMapInverseScaled[offset++];
    isoJacInv20 = pIsoMapInverseScaled[offset++];
    isoJacInv21 = pIsoMapInverseScaled[offset++];
    isoJacInv22 = pIsoMapInverseScaled[offset++];
  }  
  // end of block to bring in the inverse of the scaled isoparametric-element
  // Jacobian.

  // Begin internal force contribution calculation using the Jacobian H.
  // The exact formula for H is given below.
  /*
  \[ 
  \mathbf{H}(\xi, \eta, \zeta)
    =
    \left[
    \begin{matrix}
    4 \eta + 4 \xi + 4 \zeta - 3 & 4 \eta + 4 \xi + 4 \zeta - 3 & 4 \eta + 4 \xi + 4 \zeta - 3\\
    4 \xi - 1 & 0 & 0\\
    0 & 4 \eta - 1 & 0\\
    0 & 0 & 4 \zeta - 1\\
    - 4 \eta - 8 \xi - 4 \zeta + 4 & - 4 \xi & - 4 \xi\\
    4 \eta & 4 \xi & 0\\
    - 4 \eta & - 8 \eta - 4 \xi - 4 \zeta + 4 & - 4 \eta\\
    - 4 \zeta & - 4 \zeta & - 4 \eta - 4 \xi - 8 \zeta + 4\\
    4 \zeta & 0 & 4 \xi\\
    0 & 4 \zeta & 4 \eta
    \end{matrix}
    \right]
\] 
*/
// Compute the deformation gradient F using the nodal unknowns.
float F00, F01, F02, F10, F11, F12, F20, F21, F22;
{
    // The basic idea is to bring a Nodal Unknown (NU) in once, use it, and then discard it without asking for it again.
    // While the nodal unknowns are stored in double, we work with float here.
    const int* __restrict__ pElementNodes = pElement_NodeIndexes + element_idx * 10; // 10 nodes per element

    // Bring in the x-y-z coordinates of node 0, and the x coordinate of the node 3.
    int whichGlobalNode = pElementNodes[nodeOffsets[0][tile.thread_rank()]];
    float value = pPosNodes[3 * whichGlobalNode + xyzFieldOffsets[0][tile.thread_rank()]];
    float NUx = tile.shfl(value, 0); // the x of node 0 of the element
    float NUy = tile.shfl(value, 1); // the y of node 0 of the element
    float NUz = tile.shfl(value, 2); // the z of node 0 of the element
    float NU3x = tile.shfl(value, 3); // the x of node 3 of the element

    // Use the NU associated w/ node 0 to compute the deformation gradient.
    // I have to do N*H*inv(H_iso)h_i

    // Node zero, gradient of the shape function associated with node 0.
    // 4 \eta + 4 \xi + 4 \zeta - 3 & 4 \eta + 4 \xi + 4 \zeta - 3 & 4 \eta + 4 \xi + 4 \zeta - 3
    float h0 = 4.f * eta + 4.f * xi + 4.f * zeta - 3.f;
    float h1 = h0;
    float h2 = h0;

    // Left multiply the inverse of the isoparametric-element Jacobian by the gradient of the shape function associated with node 0. The results is a row vector with three components.
    float dummy0 = h0*isoJacInv00 + h1*isoJacInv01 + h2*isoJacInv02;
    float dummy1 = h0*isoJacInv10 + h1*isoJacInv11 + h2*isoJacInv12;
    float dummy2 = h0*isoJacInv20 + h1*isoJacInv21 + h2*isoJacInv22;

    // Outer produc to start building the deformation gradient.
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
    }