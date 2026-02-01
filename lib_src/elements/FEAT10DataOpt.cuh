/*==============================================================
 *==============================================================
 * Project: TL-FEA
 * File:    FEAT10DataOpt.cuh
 * Brief:   Defines GPU_FEAT10Opt_Data struct optimized for fused internal
 *          force computation with 4 QPs per element, SoA memory layout,
 *          and float compute with double precision positions.
 *==============================================================
 *==============================================================*/

#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "../../lib_utils/cuda_utils.h"

/**
 * GPU-optimized data structure for 10-node tetrahedral (T10) elements.
 *
 * Key design features:
 * - 4 quadrature points per element (vs 5 in FEAT10)
 * - Blocked SoA memory layout for coalesced access
 * - Float compute with double precision positions/velocities
 * - Designed for 64-thread blocks (16 elements × 4 QPs)
 * - Mooney-Rivlin material model only
 */
struct GPU_FEAT10Opt_Data {
  // ============================================================
  // Constants
  // ============================================================
  static constexpr int BLOCK_SIZE = 64;       // Threads per block
  static constexpr int TILE_SIZE = 4;         // Threads per element (= QPs)
  static constexpr int ELEMS_PER_BLOCK = 16;  // Elements per block
  static constexpr int NODES_PER_ELEM = 10;   // Nodes per T10 element
  static constexpr int QPS_PER_ELEM = 4;      // Quadrature points per element

  // ============================================================
  // Sizes
  // ============================================================
  int n_elem;         // Actual number of elements
  int n_nodes;        // Number of nodes
  int n_elem_padded;  // Padded to multiple of ELEMS_PER_BLOCK

  // ============================================================
  // Material Parameters (Mooney-Rivlin)
  // ============================================================
  float mu10;          // First Mooney-Rivlin coefficient
  float mu01;          // Second Mooney-Rivlin coefficient
  float bulkK;         // Bulk modulus (volumetric penalty)
  float minJthreshold; // Minimum J threshold to avoid singularity
  float rho;           // Density for mass computation

  // ============================================================
  // Device Pointers (Host-side storage of device addresses)
  // ============================================================

  // Positions - double precision, AoS interleaved [3 * n_nodes]
  // Layout: [x0, y0, z0, x1, y1, z1, ...]
  double* d_pos_nodes;

  // Reference positions - for precomputation (double) [3 * n_nodes]
  double* d_pos_nodes_ref;

  // Element connectivity - Blocked SoA layout [10 * n_elem_padded]
  // Within each 16-element block: all node0s, then all node1s, etc.
  // For block b, element e (0-15), node n: index = b*160 + n*16 + e
  int* d_elem_nodes_soa;

  // Inverse Jacobian (parent-to-reference) - Blocked SoA layout
  // [9 * 4 * n_elem_padded] floats
  // Within each 64-QP block: all Jac00s, then all Jac01s, etc.
  // For block b, thread t (0-63), component c (0-8): index = b*576 + c*64 + t
  float* d_iso_map_inv;

  // Internal force output - AoS interleaved [3 * n_nodes] floats
  // Layout: [fx0, fy0, fz0, fx1, fy1, fz1, ...]
  float* d_internal_force;

  // Deformation gradient F output (optional) - Blocked SoA layout
  // [9 * 4 * n_elem_padded] floats (same layout as d_iso_map_inv)
  float* d_deformation_grad_F;

  // Inverse lumped mass - [n_nodes] floats
  // Stores 1/m for direct multiplication in time integration
  float* d_inv_mass_lumped;

  // Device copy of this struct for kernel access
  GPU_FEAT10Opt_Data* d_data;

  // ============================================================
  // State Flags
  // ============================================================
  bool is_initialized;
  bool is_setup;
  bool is_precomputed;
  bool is_mass_computed;

  // ============================================================
  // Constructor
  // ============================================================
  GPU_FEAT10Opt_Data()
      : n_elem(0),
        n_nodes(0),
        n_elem_padded(0),
        mu10(0.0f),
        mu01(0.0f),
        bulkK(0.0f),
        minJthreshold(1e-6f),
        rho(1000.0f),
        d_pos_nodes(nullptr),
        d_pos_nodes_ref(nullptr),
        d_elem_nodes_soa(nullptr),
        d_iso_map_inv(nullptr),
        d_internal_force(nullptr),
        d_deformation_grad_F(nullptr),
        d_inv_mass_lumped(nullptr),
        d_data(nullptr),
        is_initialized(false),
        is_setup(false),
        is_precomputed(false),
        is_mass_computed(false) {}

  // ============================================================
  // Host Methods - Lifecycle
  // ============================================================

  /**
   * Allocate all device memory.
   * @param num_elements Number of elements
   * @param num_nodes Number of nodes
   */
  void Initialize(int num_elements, int num_nodes);

  /**
   * Setup element data from host arrays.
   * @param positions Node positions [n_nodes x 3] (row-major: x,y,z per node)
   * @param connectivity Element connectivity [n_elem x 10]
   */
  void Setup(const Eigen::MatrixXd& positions,
             const Eigen::MatrixXi& connectivity);

  /**
   * Free all device memory.
   */
  void Destroy();

  // ============================================================
  // Host Methods - Material
  // ============================================================

  /**
   * Set Mooney-Rivlin material parameters.
   * @param mu10_val First coefficient
   * @param mu01_val Second coefficient
   * @param kappa Bulk modulus
   */
  void SetMooneyRivlin(float mu10_val, float mu01_val, float kappa);

  /**
   * Set material density.
   * @param density Density value
   */
  void SetDensity(float density);

  // ============================================================
  // Host Methods - Computation
  // ============================================================

  /**
   * Compute inverse Jacobian at each quadrature point from reference config.
   * Must be called after Setup() and before any force computation.
   */
  void ComputePrecomputation();

  /**
   * Compute HRZ lumped mass using 4-point quadrature.
   * Stores inverse mass (1/m) for efficient time integration.
   */
  void ComputeLumpedMassHRZ();

  /**
   * Zero out the internal force buffer.
   */
  void ClearInternalForce();

  /**
   * Compute internal forces using the fused kernel.
   * @param writeOutF If true, also write deformation gradient F to device
   */
  void ComputeInternalForce(bool writeOutF = false);

  // ============================================================
  // Host Methods - Data Transfer
  // ============================================================

  /**
   * Update node positions on GPU.
   * @param positions New positions [n_nodes x 3]
   */
  void UpdatePositions(const Eigen::MatrixXd& positions);

  /**
   * Retrieve positions from GPU to CPU.
   * @param x Output x coordinates
   * @param y Output y coordinates
   * @param z Output z coordinates
   */
  void RetrievePositionToCPU(Eigen::VectorXd& x, Eigen::VectorXd& y,
                             Eigen::VectorXd& z);

  /**
   * Retrieve internal forces from GPU to CPU.
   * @param forces Output force vector [3 * n_nodes]
   */
  void RetrieveInternalForceToCPU(Eigen::VectorXd& forces);

  /**
   * Retrieve internal forces from GPU to CPU (float version).
   * @param forces Output force vector [3 * n_nodes]
   */
  void RetrieveInternalForceToCPU(Eigen::VectorXf& forces);

  /**
   * Retrieve inverse lumped mass from GPU to CPU.
   * @param inv_mass Output inverse mass vector [n_nodes]
   */
  void RetrieveInvLumpedMassToCPU(Eigen::VectorXf& inv_mass);

  /**
   * Retrieve deformation gradient F from GPU to CPU.
   * @param F Output: F[elem][qp] is 3x3 matrix
   */
  void RetrieveDeformationGradientToCPU(
      std::vector<std::vector<Eigen::Matrix3f>>& F);

  // ============================================================
  // Host Methods - Accessors
  // ============================================================

  int get_n_elem() const { return n_elem; }
  int get_n_nodes() const { return n_nodes; }
  int get_n_elem_padded() const { return n_elem_padded; }

  const float* GetInvMassDevicePtr() const { return d_inv_mass_lumped; }
  float* GetInternalForceDevicePtr() { return d_internal_force; }
  const float* GetInternalForceDevicePtr() const { return d_internal_force; }
  double* GetPositionDevicePtr() { return d_pos_nodes; }
  const double* GetPositionDevicePtr() const { return d_pos_nodes; }

  bool IsPrecomputed() const { return is_precomputed; }
  bool IsMassComputed() const { return is_mass_computed; }

  // ============================================================
  // Device Methods (only available in CUDA code)
  // ============================================================
#if defined(__CUDACC__)

  /**
   * Get position of a node (device-side).
   */
  __device__ void getNodePosition(int node_idx, double& x, double& y,
                                  double& z) const {
    x = d_pos_nodes[3 * node_idx + 0];
    y = d_pos_nodes[3 * node_idx + 1];
    z = d_pos_nodes[3 * node_idx + 2];
  }

  /**
   * Get reference position of a node (device-side).
   */
  __device__ void getNodePositionRef(int node_idx, double& x, double& y,
                                     double& z) const {
    x = d_pos_nodes_ref[3 * node_idx + 0];
    y = d_pos_nodes_ref[3 * node_idx + 1];
    z = d_pos_nodes_ref[3 * node_idx + 2];
  }

#endif
};

// ============================================================
// Kernel Launch Wrapper (declared here, defined in FEAT10DataOpt.cu)
// ============================================================

/**
 * Launch the fused internal force kernel.
 * @param d_data Device pointer to GPU_FEAT10Opt_Data
 * @param n_elem_padded Padded element count (must be multiple of 16)
 * @param writeOutF Whether to write deformation gradient F
 */
void launchInternalForceKernel_FEAT10Opt(GPU_FEAT10Opt_Data* d_data,
                                         int n_elem_padded, bool writeOutF);
