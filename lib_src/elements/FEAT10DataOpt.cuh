#pragma once

/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli
 * Email:   arivoli@wisc.edu
 * File:    FEAT10DataOpt.cuh
 * Brief:   Declares GPU_FEAT10Opt_Data optimized for fused
 *          FEAT10 internal force and mass kernels.
 *==============================================================
 *==============================================================*/

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "../../lib_utils/cuda_utils.h"

// GPU data layout for 10-node tetrahedral (T10) elements.
// Uses 4 QPs/elem, blocked SoA, float compute, double positions,
// and Mooney–Rivlin material, tuned for 64-thread blocks.
struct GPU_FEAT10Opt_Data {
  // Constants
  static constexpr int BLOCK_SIZE = 64;       // Threads per block
  static constexpr int TILE_SIZE = 4;         // Threads per element / QPs
  static constexpr int ELEMS_PER_BLOCK = 16;  // Elements per block
  static constexpr int NODES_PER_ELEM = 10;   // T10 nodes per element
  static constexpr int QPS_PER_ELEM = 4;      // Quadrature points per element

  // Sizes
  int n_elem;         // Elements
  int n_nodes;        // Nodes
  int n_elem_padded;  // Elements padded to ELEMS_PER_BLOCK

  // Material parameters (Mooney–Rivlin)
  float mu10;
  float mu01;
  float bulkK;         // Bulk modulus
  float minJthreshold; // Min J to avoid singularity
  float rho;           // Density

  // Device pointers (device memory owned here)

  // Positions, AoS interleaved [3 * n_nodes]:
  // [x0, y0, z0, x1, y1, z1, ...]
  double* d_pos_nodes;

  // Reference positions [3 * n_nodes]
  double* d_pos_nodes_ref;

  // Element connectivity, blocked SoA [10 * n_elem_padded]
  int* d_elem_nodes_soa;

  // Inverse Jacobian (parent → reference), blocked SoA
  // [9 * 4 * n_elem_padded] floats
  float* d_iso_map_inv;

  // Internal force, AoS interleaved [3 * n_nodes] floats
  float* d_internal_force;

  // External force, AoS interleaved [3 * n_nodes] floats
  float* d_external_force;

  // Deformation gradient F (optional), blocked SoA
  // [9 * 4 * n_elem_padded] floats
  float* d_deformation_grad_F;

  // Inverse lumped mass [n_nodes] (1 / m_i)
  float* d_inv_mass_lumped;

  // Device copy of this struct
  GPU_FEAT10Opt_Data* d_data;

  // State flags
  bool is_initialized;
  bool is_setup;
  bool is_precomputed;
  bool is_mass_computed;

  // Constructor
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
        d_external_force(nullptr),
        d_deformation_grad_F(nullptr),
        d_inv_mass_lumped(nullptr),
        d_data(nullptr),
        is_initialized(false),
        is_setup(false),
        is_precomputed(false),
        is_mass_computed(false) {}

  // Host methods – lifecycle
  // Allocate device memory.
  void Initialize(int num_elements, int num_nodes);

  // Upload element data from host arrays.
  void Setup(const Eigen::MatrixXd& positions,
             const Eigen::MatrixXi& connectivity);

  // Free device memory.
  void Destroy();

  // Host methods – material
  // Set Mooney–Rivlin parameters.
  void SetMooneyRivlin(float mu10_val, float mu01_val, float kappa);

  // Set material density.
  void SetDensity(float density);

  // Host methods – computation
  // Precompute inverse Jacobians per quadrature point.
  void ComputePrecomputation();

  // Compute HRZ lumped mass (stores 1 / m_i).
  void ComputeLumpedMassHRZ();

  // Zero internal force buffer.
  void ClearInternalForce();

  // Enable diagnostic output by pre-allocating F buffer.
  // Call this ONCE before simulation if you need diagnostics.
  void EnableDiagnostics(bool enableF = true);

  // Compute internal forces (optionally writing F).
  void ComputeInternalForce(bool writeOutF = false);

  // Set external force from CPU.
  void SetExternalForce(const Eigen::VectorXf& f_ext);

  // Clear external force to zero.
  void ClearExternalForce();

  // Host methods – data transfer
  // Update node positions on GPU.
  void UpdatePositions(const Eigen::MatrixXd& positions);

  // Download positions to CPU.
  void RetrievePositionToCPU(Eigen::VectorXd& x, Eigen::VectorXd& y,
                             Eigen::VectorXd& z);

  // Download internal forces to CPU (double).
  void RetrieveInternalForceToCPU(Eigen::VectorXd& forces);

  // Download internal forces to CPU (float).
  void RetrieveInternalForceToCPU(Eigen::VectorXf& forces);

  // Download inverse lumped mass to CPU.
  void RetrieveInvLumpedMassToCPU(Eigen::VectorXf& inv_mass);

  // Download deformation gradient F to CPU.
  void RetrieveDeformationGradientToCPU(
      std::vector<std::vector<Eigen::Matrix3f>>& F);

  // Host methods – accessors
  int get_n_elem() const { return n_elem; }
  int get_n_nodes() const { return n_nodes; }
  int get_n_elem_padded() const { return n_elem_padded; }

  const float* GetInvMassDevicePtr() const { return d_inv_mass_lumped; }
  float* GetInternalForceDevicePtr() { return d_internal_force; }
  const float* GetInternalForceDevicePtr() const { return d_internal_force; }
  float* GetExternalForceDevicePtr() { return d_external_force; }
  const float* GetExternalForceDevicePtr() const { return d_external_force; }
  double* GetPositionDevicePtr() { return d_pos_nodes; }
  const double* GetPositionDevicePtr() const { return d_pos_nodes; }

  bool IsPrecomputed() const { return is_precomputed; }
  bool IsMassComputed() const { return is_mass_computed; }

  // Device methods (CUDA only)
#if defined(__CUDACC__)

  // Get node position.
  __device__ void getNodePosition(int node_idx, double& x, double& y,
                                  double& z) const {
    x = d_pos_nodes[3 * node_idx + 0];
    y = d_pos_nodes[3 * node_idx + 1];
    z = d_pos_nodes[3 * node_idx + 2];
  }

  // Get reference node position.
  __device__ void getNodePositionRef(int node_idx, double& x, double& y,
                                     double& z) const {
    x = d_pos_nodes_ref[3 * node_idx + 0];
    y = d_pos_nodes_ref[3 * node_idx + 1];
    z = d_pos_nodes_ref[3 * node_idx + 2];
  }

#endif
};

// Kernel launch wrapper (declared here, defined in FEAT10DataOpt.cu).
// Launch fused internal force kernel using host-side struct.
// launchInternalForceKernel_FEAT10Opt removed - inlined in ComputeInternalForce()
