/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Ganesh Arivoli
 * Email:   arivoli@wisc.edu
 * File:    FEAT10DataOpt.cu
 * Brief:   Implements GPU_FEAT10Opt_Data for fast explicit
 *          dynamics with T10 elements and Mooney-Rivlin material.
 *==============================================================
 *==============================================================*/

#include "FEAT10DataOpt.cuh"

#include "FEAT10KernelOpt.cuh"

#include <cassert>
#include <iostream>
#include <vector>

// Precompute inverse Jacobian at each QP (one thread per QP).
__global__ void computeInverseJacobian_kernel(GPU_FEAT10Opt_Data* d_data) {
  constexpr int TILE = 4;
  constexpr int elements_per_block = 16;

  // Thread/element mapping
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int elem_idx = global_thread / TILE;
  const int qp_idx = global_thread % TILE;

  // Early exit for padded elements (they have degenerate geometry)
  if (elem_idx >= d_data->n_elem) {
    return;
  }

  // QP coordinates for the 4-point rule (canonical permutations of (b,a,a,a))
  constexpr float a = 0.1381966011250105f;
  constexpr float b = 0.5854101966249685f;

  const float xi = (qp_idx == 1) ? b : a;
  const float eta = (qp_idx == 2) ? b : a;
  const float zeta = (qp_idx == 3) ? b : a;

  // Compute barycentric coordinates
  const float L1 = 1.0f - xi - eta - zeta;
  const float L2 = xi;
  const float L3 = eta;
  const float L4 = zeta;
  const float L[4] = {L1, L2, L3, L4};

  // Derivatives of barycentric coordinates w.r.t. (xi, eta, zeta)
  const float dL[4][3] = {
      {-1.0f, -1.0f, -1.0f},  // dL1
      {1.0f, 0.0f, 0.0f},     // dL2
      {0.0f, 1.0f, 0.0f},     // dL3
      {0.0f, 0.0f, 1.0f}      // dL4
  };

  // Shape function derivatives dN/d(xi,eta,zeta) for T10 element
  float dN_dxi[10][3];

  // Corner nodes (0-3): dN_i = (4*L_i - 1) * dL_i
  for (int i = 0; i < 4; i++) {
    float factor = 4.0f * L[i] - 1.0f;
    for (int j = 0; j < 3; j++) {
      dN_dxi[i][j] = factor * dL[i][j];
    }
  }

  // Edge nodes (4-9): dN_k = 4*(L_i*dL_j + L_j*dL_i)
  // Edge connectivity: [(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]
  const int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};

  for (int k = 0; k < 6; k++) {
    int i = edges[k][0];
    int j = edges[k][1];
    for (int d = 0; d < 3; d++) {
      dN_dxi[k + 4][d] = 4.0f * (L[i] * dL[j][d] + L[j] * dL[i][d]);
    }
  }

  // Get node indices for this element (blocked SoA layout)
  const int block_idx = elem_idx / elements_per_block;
  const int elem_in_block = elem_idx % elements_per_block;
  const int* elem_nodes = d_data->d_elem_nodes_soa +
                          block_idx * 10 * elements_per_block + elem_in_block;

  // Load reference positions and compute Jacobian J = sum_a (X_a * dN_a/dxi)
  float J[3][3] = {{0.0f}};

  for (int a = 0; a < 10; a++) {
    int node_idx = elem_nodes[a * elements_per_block];
    float X[3];
    X[0] = (float)d_data->d_pos_nodes_ref[3 * node_idx + 0];
    X[1] = (float)d_data->d_pos_nodes_ref[3 * node_idx + 1];
    X[2] = (float)d_data->d_pos_nodes_ref[3 * node_idx + 2];

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        J[i][j] += X[i] * dN_dxi[a][j];
      }
    }
  }

  // Compute inverse of J using cofactor method
  // det(J)
  float detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
               J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
               J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

  float invDetJ = 1.0f / detJ;

  // Cofactor matrix / det = inverse
  float Jinv[3][3];
  Jinv[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * invDetJ;
  Jinv[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) * invDetJ;
  Jinv[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * invDetJ;
  Jinv[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) * invDetJ;
  Jinv[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * invDetJ;
  Jinv[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) * invDetJ;
  Jinv[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * invDetJ;
  Jinv[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) * invDetJ;
  Jinv[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * invDetJ;

  // Store in blocked SoA format
  // Block b, thread t (0-63), component c (0-8): index = b*576 + c*64 + t
  const int qp_block = (elem_idx * TILE) / 64;
  const int thread_in_block = (elem_idx * TILE + qp_idx) % 64;
  const int base = qp_block * 576 + thread_in_block;

  d_data->d_iso_map_inv[base + 0 * 64] = Jinv[0][0];
  d_data->d_iso_map_inv[base + 1 * 64] = Jinv[0][1];
  d_data->d_iso_map_inv[base + 2 * 64] = Jinv[0][2];
  d_data->d_iso_map_inv[base + 3 * 64] = Jinv[1][0];
  d_data->d_iso_map_inv[base + 4 * 64] = Jinv[1][1];
  d_data->d_iso_map_inv[base + 5 * 64] = Jinv[1][2];
  d_data->d_iso_map_inv[base + 6 * 64] = Jinv[2][0];
  d_data->d_iso_map_inv[base + 7 * 64] = Jinv[2][1];
  d_data->d_iso_map_inv[base + 8 * 64] = Jinv[2][2];
}

// HRZ lumped mass kernel (one thread per element).
__global__ void computeHRZLumpedMass_kernel(GPU_FEAT10Opt_Data* d_data,
                                             float* d_mass_lumped) {
  int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (elem_idx >= d_data->n_elem) return;

  constexpr int elements_per_block = 16;
  const float rho = d_data->rho;

  // QP coordinates and weight
  constexpr float a = 0.1381966011250105f;
  constexpr float b = 0.5854101966249685f;
  constexpr float weightQP = 1.0f / 24.0f;  // Same for all 4 QPs

  // Get node indices for this element
  const int block_idx = elem_idx / elements_per_block;
  const int elem_in_block = elem_idx % elements_per_block;
  const int* elem_nodes = d_data->d_elem_nodes_soa +
                          block_idx * 10 * elements_per_block + elem_in_block;

  // Accumulators
  float vol_elem = 0.0f;
  float diag_consistent[10] = {0.0f};

  // Loop over 4 quadrature points
  for (int qp = 0; qp < 4; qp++) {
    // Canonical 4-point tet rule: permutations of (b, a, a, a)
    const float xi = (qp == 1) ? b : a;
    const float eta = (qp == 2) ? b : a;
    const float zeta = (qp == 3) ? b : a;

    // Barycentric coordinates
    const float L1 = 1.0f - xi - eta - zeta;
    const float L2 = xi;
    const float L3 = eta;
    const float L4 = zeta;
    const float L[4] = {L1, L2, L3, L4};

    // Shape functions
    float N[10];

    // Corner nodes
    for (int k = 0; k < 4; k++) {
      N[k] = L[k] * (2.0f * L[k] - 1.0f);
    }

    // Edge nodes
    const int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
    for (int k = 0; k < 6; k++) {
      int ii = edges[k][0];
      int jj = edges[k][1];
      N[k + 4] = 4.0f * L[ii] * L[jj];
    }

    // Get detJ from the inverse Jacobian
    // We need to compute det(J) = 1/det(J^{-1})
    const int qp_block = (elem_idx * 4) / 64;
    const int thread_in_block = (elem_idx * 4 + qp) % 64;
    const int base = qp_block * 576 + thread_in_block;

    float Jinv00 = d_data->d_iso_map_inv[base + 0 * 64];
    float Jinv01 = d_data->d_iso_map_inv[base + 1 * 64];
    float Jinv02 = d_data->d_iso_map_inv[base + 2 * 64];
    float Jinv10 = d_data->d_iso_map_inv[base + 3 * 64];
    float Jinv11 = d_data->d_iso_map_inv[base + 4 * 64];
    float Jinv12 = d_data->d_iso_map_inv[base + 5 * 64];
    float Jinv20 = d_data->d_iso_map_inv[base + 6 * 64];
    float Jinv21 = d_data->d_iso_map_inv[base + 7 * 64];
    float Jinv22 = d_data->d_iso_map_inv[base + 8 * 64];

    float detJinv = Jinv00 * (Jinv11 * Jinv22 - Jinv12 * Jinv21) -
                    Jinv01 * (Jinv10 * Jinv22 - Jinv12 * Jinv20) +
                    Jinv02 * (Jinv10 * Jinv21 - Jinv11 * Jinv20);
    float detJ = 1.0f / detJinv;

    float dV = detJ * weightQP;
    vol_elem += dV;

    // Accumulate diagonal of consistent mass: ∫ N_i² dV
    for (int i = 0; i < 10; i++) {
      diag_consistent[i] += N[i] * N[i] * dV;
    }
  }

  // HRZ scaling: preserve total element mass
  float total_mass = rho * vol_elem;
  float sum_diag = 0.0f;
  for (int i = 0; i < 10; i++) {
    sum_diag += diag_consistent[i];
  }

  if (sum_diag < 1e-30f) return;

  float scale = total_mass / sum_diag;

  // Assemble to global mass vector using atomicAdd
  for (int i_local = 0; i_local < 10; i_local++) {
    int i_global = elem_nodes[i_local * elements_per_block];
    float m_lumped = diag_consistent[i_local] * scale;
    atomicAdd(&d_mass_lumped[i_global], m_lumped);
  }
}

// Invert lumped mass (1/m per node).
__global__ void invertLumpedMass_kernel(float* d_mass_lumped,
                                         float* d_inv_mass_lumped, int n_nodes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_nodes) return;

  float m = d_mass_lumped[idx];
  d_inv_mass_lumped[idx] = (m > 1e-30f) ? (1.0f / m) : 0.0f;
}

// GPU_FEAT10Opt_Data Implementation

void GPU_FEAT10Opt_Data::Initialize(int num_elements, int num_nodes) {
  if (is_initialized) {
    std::cerr << "GPU_FEAT10Opt_Data: Already initialized." << std::endl;
    return;
  }

  n_elem = num_elements;
  n_nodes = num_nodes;

  // Pad to multiple of ELEMS_PER_BLOCK (16)
  n_elem_padded = ((n_elem + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK;

  // Allocate positions (double, AoS interleaved)
  HANDLE_ERROR(cudaMalloc(&d_pos_nodes, 3 * n_nodes * sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&d_pos_nodes_ref, 3 * n_nodes * sizeof(double)));

  // Allocate connectivity (blocked SoA)
  HANDLE_ERROR(cudaMalloc(&d_elem_nodes_soa, 10 * n_elem_padded * sizeof(int)));

  // Allocate inverse Jacobian (9 components * 4 QPs * n_elem_padded)
  HANDLE_ERROR(cudaMalloc(&d_iso_map_inv, 9 * 4 * n_elem_padded * sizeof(float)));

  // Allocate internal force (float, AoS interleaved)
  HANDLE_ERROR(cudaMalloc(&d_internal_force, 3 * n_nodes * sizeof(float)));

  // Allocate external force (float, AoS interleaved)
  HANDLE_ERROR(cudaMalloc(&d_external_force, 3 * n_nodes * sizeof(float)));
  HANDLE_ERROR(cudaMemset(d_external_force, 0, 3 * n_nodes * sizeof(float)));

  // Allocate inverse lumped mass
  HANDLE_ERROR(cudaMalloc(&d_inv_mass_lumped, n_nodes * sizeof(float)));

  // Optional deformation gradient F (allocated on demand)
  d_deformation_grad_F = nullptr;

  // Allocate device copy of this struct
  HANDLE_ERROR(cudaMalloc(&d_data, sizeof(GPU_FEAT10Opt_Data)));

  is_initialized = true;
}

void GPU_FEAT10Opt_Data::Setup(const Eigen::MatrixXd& positions,
                              const Eigen::MatrixXi& connectivity) {
  if (!is_initialized) {
    std::cerr << "GPU_FEAT10Opt_Data: Must call Initialize() first." << std::endl;
    return;
  }
  if (is_setup) {
    std::cerr << "GPU_FEAT10Opt_Data: Already set up." << std::endl;
    return;
  }

  // Verify sizes
  if (positions.rows() != n_nodes || positions.cols() != 3) {
    std::cerr << "GPU_FEAT10Opt_Data: Position matrix size mismatch. Expected "
              << n_nodes << "x3, got " << positions.rows() << "x"
              << positions.cols() << std::endl;
    return;
  }
  if (connectivity.rows() != n_elem || connectivity.cols() != 10) {
    std::cerr << "GPU_FEAT10Opt_Data: Connectivity matrix size mismatch. Expected "
              << n_elem << "x10, got " << connectivity.rows() << "x"
              << connectivity.cols() << std::endl;
    return;
  }

  // Convert positions to AoS interleaved format [x0,y0,z0,x1,y1,z1,...]
  std::vector<double> pos_interleaved(3 * n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    pos_interleaved[3 * i + 0] = positions(i, 0);
    pos_interleaved[3 * i + 1] = positions(i, 1);
    pos_interleaved[3 * i + 2] = positions(i, 2);
  }

  HANDLE_ERROR(cudaMemcpy(d_pos_nodes, pos_interleaved.data(),
                          3 * n_nodes * sizeof(double), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_pos_nodes_ref, pos_interleaved.data(),
                          3 * n_nodes * sizeof(double), cudaMemcpyHostToDevice));

  // Convert connectivity to blocked SoA format
  // Within each 16-element block: all node0s, then node1s, etc.
  std::vector<int> conn_soa(10 * n_elem_padded, 0);

  for (int b = 0; b < n_elem_padded / ELEMS_PER_BLOCK; b++) {
    for (int e = 0; e < ELEMS_PER_BLOCK; e++) {
      int elem_idx = b * ELEMS_PER_BLOCK + e;
      for (int n = 0; n < 10; n++) {
        int soa_idx = b * 10 * ELEMS_PER_BLOCK + n * ELEMS_PER_BLOCK + e;
        if (elem_idx < n_elem) {
          conn_soa[soa_idx] = connectivity(elem_idx, n);
        } else {
          // Padded elements: use node 0 to avoid bad memory access
          conn_soa[soa_idx] = 0;
        }
      }
    }
  }

  HANDLE_ERROR(cudaMemcpy(d_elem_nodes_soa, conn_soa.data(),
                          10 * n_elem_padded * sizeof(int),
                          cudaMemcpyHostToDevice));

  // Zero internal force
  HANDLE_ERROR(cudaMemset(d_internal_force, 0, 3 * n_nodes * sizeof(float)));

  // Zero inverse mass
  HANDLE_ERROR(cudaMemset(d_inv_mass_lumped, 0, n_nodes * sizeof(float)));

  // Copy struct to device
  HANDLE_ERROR(
      cudaMemcpy(d_data, this, sizeof(GPU_FEAT10Opt_Data), cudaMemcpyHostToDevice));

  is_setup = true;
}

void GPU_FEAT10Opt_Data::Destroy() {
  if (d_pos_nodes) cudaFree(d_pos_nodes);
  if (d_pos_nodes_ref) cudaFree(d_pos_nodes_ref);
  if (d_elem_nodes_soa) cudaFree(d_elem_nodes_soa);
  if (d_iso_map_inv) cudaFree(d_iso_map_inv);
  if (d_internal_force) cudaFree(d_internal_force);
  if (d_external_force) cudaFree(d_external_force);
  if (d_deformation_grad_F) cudaFree(d_deformation_grad_F);
  if (d_inv_mass_lumped) cudaFree(d_inv_mass_lumped);
  if (d_data) cudaFree(d_data);

  d_pos_nodes = nullptr;
  d_pos_nodes_ref = nullptr;
  d_elem_nodes_soa = nullptr;
  d_iso_map_inv = nullptr;
  d_internal_force = nullptr;
  d_external_force = nullptr;
  d_deformation_grad_F = nullptr;
  d_inv_mass_lumped = nullptr;
  d_data = nullptr;

  is_initialized = false;
  is_setup = false;
  is_precomputed = false;
  is_mass_computed = false;
}

void GPU_FEAT10Opt_Data::SetMooneyRivlin(float mu10_val, float mu01_val,
                                        float kappa) {
  mu10 = mu10_val;
  mu01 = mu01_val;
  bulkK = kappa;

  // Update device copy
  if (d_data) {
    HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10Opt_Data),
                            cudaMemcpyHostToDevice));
  }
}

void GPU_FEAT10Opt_Data::SetDensity(float density) {
  rho = density;

  // Update device copy
  if (d_data) {
    HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10Opt_Data),
                            cudaMemcpyHostToDevice));
  }
}

void GPU_FEAT10Opt_Data::ComputePrecomputation() {
  if (!is_setup) {
    std::cerr << "GPU_FEAT10Opt_Data: Must call Setup() first." << std::endl;
    return;
  }
  if (is_precomputed) {
    return;  // Already done
  }

  // Launch kernel: one thread per QP (4 QPs per element)
  int total_threads = n_elem_padded * 4;
  int threads_per_block = 64;
  int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  computeInverseJacobian_kernel<<<blocks, threads_per_block>>>(d_data);
  HANDLE_ERROR(cudaDeviceSynchronize());

  is_precomputed = true;

  // Update device copy
  HANDLE_ERROR(
      cudaMemcpy(d_data, this, sizeof(GPU_FEAT10Opt_Data), cudaMemcpyHostToDevice));
}

void GPU_FEAT10Opt_Data::ComputeLumpedMassHRZ() {
  if (!is_precomputed) {
    std::cerr << "GPU_FEAT10Opt_Data: Must call ComputePrecomputation() first."
              << std::endl;
    return;
  }
  if (is_mass_computed) {
    return;  // Already done
  }

  // Allocate temporary mass buffer
  float* d_mass_temp;
  HANDLE_ERROR(cudaMalloc(&d_mass_temp, n_nodes * sizeof(float)));
  HANDLE_ERROR(cudaMemset(d_mass_temp, 0, n_nodes * sizeof(float)));

  // Launch HRZ kernel: one thread per element
  int threads_per_block = 128;
  int blocks = (n_elem + threads_per_block - 1) / threads_per_block;

  computeHRZLumpedMass_kernel<<<blocks, threads_per_block>>>(d_data, d_mass_temp);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Invert the mass
  blocks = (n_nodes + threads_per_block - 1) / threads_per_block;
  invertLumpedMass_kernel<<<blocks, threads_per_block>>>(
      d_mass_temp, d_inv_mass_lumped, n_nodes);
  HANDLE_ERROR(cudaDeviceSynchronize());

  cudaFree(d_mass_temp);

  is_mass_computed = true;

  // Update device copy
  HANDLE_ERROR(
      cudaMemcpy(d_data, this, sizeof(GPU_FEAT10Opt_Data), cudaMemcpyHostToDevice));
}

void GPU_FEAT10Opt_Data::ClearInternalForce(cudaStream_t stream) {
  HANDLE_ERROR(cudaMemsetAsync(d_internal_force, 0, 3 * n_nodes * sizeof(float), stream));
}

void GPU_FEAT10Opt_Data::EnableDiagnostics(bool enableF) {
  // Pre-allocate diagnostic buffer for F if needed
  if (enableF && d_deformation_grad_F == nullptr) {
    HANDLE_ERROR(
        cudaMalloc(&d_deformation_grad_F, 9 * 4 * n_elem_padded * sizeof(float)));
    // Update device copy with new pointer
    HANDLE_ERROR(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10Opt_Data),
                            cudaMemcpyHostToDevice));
  }
}

void GPU_FEAT10Opt_Data::ComputeInternalForce(bool writeOutF, cudaStream_t stream) {
  assert(is_precomputed && "Must call ComputePrecomputation() first");

  // Launch kernel directly - no wrapper overhead
  constexpr int BLOCK_SIZE = 64;
  const int num_blocks = n_elem_padded / 16;
  constexpr size_t shared_mem_size = 3 * 9 * BLOCK_SIZE * sizeof(float);

  internalF_MooneyRivlin_4QP<<<num_blocks, BLOCK_SIZE, shared_mem_size, stream>>>(
      n_elem, d_pos_nodes, d_elem_nodes_soa, d_iso_map_inv, d_internal_force,
      d_deformation_grad_F, writeOutF);
}

void GPU_FEAT10Opt_Data::SetExternalForce(const Eigen::VectorXf& f_ext) {
  assert(f_ext.size() == n_nodes * 3 && "External force size mismatch");
  
  HANDLE_ERROR(cudaMemcpy(d_external_force, f_ext.data(),
                          n_nodes * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void GPU_FEAT10Opt_Data::ClearExternalForce() {
  HANDLE_ERROR(cudaMemset(d_external_force, 0, 3 * n_nodes * sizeof(float)));
}

void GPU_FEAT10Opt_Data::UpdatePositions(const Eigen::MatrixXd& positions) {
  assert(positions.rows() == n_nodes && positions.cols() == 3 && 
         "Position matrix size mismatch");

  // Convert to AoS interleaved
  std::vector<double> pos_interleaved(3 * n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    pos_interleaved[3 * i + 0] = positions(i, 0);
    pos_interleaved[3 * i + 1] = positions(i, 1);
    pos_interleaved[3 * i + 2] = positions(i, 2);
  }

  HANDLE_ERROR(cudaMemcpy(d_pos_nodes, pos_interleaved.data(),
                          3 * n_nodes * sizeof(double), cudaMemcpyHostToDevice));
}

void GPU_FEAT10Opt_Data::RetrievePositionToCPU(Eigen::VectorXd& x,
                                              Eigen::VectorXd& y,
                                              Eigen::VectorXd& z) {
  x.resize(n_nodes);
  y.resize(n_nodes);
  z.resize(n_nodes);

  std::vector<double> pos_interleaved(3 * n_nodes);
  HANDLE_ERROR(cudaMemcpy(pos_interleaved.data(), d_pos_nodes,
                          3 * n_nodes * sizeof(double), cudaMemcpyDeviceToHost));

  for (int i = 0; i < n_nodes; i++) {
    x(i) = pos_interleaved[3 * i + 0];
    y(i) = pos_interleaved[3 * i + 1];
    z(i) = pos_interleaved[3 * i + 2];
  }
}

void GPU_FEAT10Opt_Data::RetrieveInternalForceToCPU(Eigen::VectorXd& forces) {
  forces.resize(3 * n_nodes);

  std::vector<float> forces_float(3 * n_nodes);
  HANDLE_ERROR(cudaMemcpy(forces_float.data(), d_internal_force,
                          3 * n_nodes * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < 3 * n_nodes; i++) {
    forces(i) = static_cast<double>(forces_float[i]);
  }
}

void GPU_FEAT10Opt_Data::RetrieveInternalForceToCPU(Eigen::VectorXf& forces) {
  forces.resize(3 * n_nodes);
  HANDLE_ERROR(cudaMemcpy(forces.data(), d_internal_force,
                          3 * n_nodes * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPU_FEAT10Opt_Data::RetrieveInvLumpedMassToCPU(Eigen::VectorXf& inv_mass) {
  inv_mass.resize(n_nodes);
  HANDLE_ERROR(cudaMemcpy(inv_mass.data(), d_inv_mass_lumped,
                          n_nodes * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPU_FEAT10Opt_Data::RetrieveDeformationGradientToCPU(
    std::vector<std::vector<Eigen::Matrix3f>>& F) {
  if (d_deformation_grad_F == nullptr) {
    std::cerr << "GPU_FEAT10Opt_Data: Deformation gradient F not computed."
              << std::endl;
    return;
  }

  F.resize(n_elem);

  // Copy all F data from device
  std::vector<float> F_flat(9 * 4 * n_elem_padded);
  HANDLE_ERROR(cudaMemcpy(F_flat.data(), d_deformation_grad_F,
                          9 * 4 * n_elem_padded * sizeof(float),
                          cudaMemcpyDeviceToHost));

  // Unpack from blocked SoA format
  for (int elem = 0; elem < n_elem; elem++) {
    F[elem].resize(4);

    for (int qp = 0; qp < 4; qp++) {
      // Find location in blocked SoA
      int global_qp = elem * 4 + qp;
      int qp_block = global_qp / 64;
      int thread_in_block = global_qp % 64;
      int base = qp_block * 576 + thread_in_block;

      F[elem][qp](0, 0) = F_flat[base + 0 * 64];
      F[elem][qp](0, 1) = F_flat[base + 1 * 64];
      F[elem][qp](0, 2) = F_flat[base + 2 * 64];
      F[elem][qp](1, 0) = F_flat[base + 3 * 64];
      F[elem][qp](1, 1) = F_flat[base + 4 * 64];
      F[elem][qp](1, 2) = F_flat[base + 5 * 64];
      F[elem][qp](2, 0) = F_flat[base + 6 * 64];
      F[elem][qp](2, 1) = F_flat[base + 7 * 64];
      F[elem][qp](2, 2) = F_flat[base + 8 * 64];
    }
  }
}

// Kernel launch wrapper: pass explicit parameters from host struct to reduce
// kernel register pressure (parameters stay in constant/parameter memory).

// launchInternalForceKernel_FEAT10Opt removed - now inlined in ComputeInternalForce()
