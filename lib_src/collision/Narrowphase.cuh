/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    Narrowphase.cuh
 * Brief:   Declares the Narrowphase class and supporting data structures for
 *          pressure-based contact detection between tetrahedra. Defines the
 *          ContactPatch representation, intermediate clipping polygons, GPU
 *          mesh and pressure storage, and host API for computing patches and
 *          external contact forces.
 *==============================================================
 *==============================================================*/

#pragma once

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <vector>

// Maximum vertices in a contact polygon (triangle to hexagon possible)
#define MAX_POLYGON_VERTS 8

// ============================================================================
// Data structures for narrowphase collision detection
// ============================================================================

/**
 * Contact patch result from narrowphase collision detection.
 * Represents the iso-pressure surface between two tetrahedra.
 */
struct ContactPatch {
  // Polygon vertices (up to MAX_POLYGON_VERTS)
  double3 vertices[MAX_POLYGON_VERTS];
  int numVertices;

  // Contact normal (points from tet A into tet B)
  double3 normal;

  // Centroid of the contact patch
  double3 centroid;

  // Patch area
  double area;

  // Directional pressure gradients (Drake convention)
  // g_A = -∂p_A/∂n (moving into A increases p_A)
  // g_B = ∂p_B/∂n (moving into B increases p_B)
  double g_A;
  double g_B;

  // Equilibrium pressure at patch centroid
  double p_equilibrium;

  // Tet pair indices (from broadphase)
  int tetA_idx;
  int tetB_idx;

  // Validity flags
  bool isValid;           // True if patch has >= 3 vertices
  bool validOrientation;  // True if g_A > 0 and g_B > 0 (Drake convention)

  __host__ __device__ ContactPatch()
      : numVertices(0),
        area(0.0),
        g_A(0.0),
        g_B(0.0),
        p_equilibrium(0.0),
        tetA_idx(-1),
        tetB_idx(-1),
        isValid(false),
        validOrientation(false) {
    normal   = make_double3(0.0, 0.0, 0.0);
    centroid = make_double3(0.0, 0.0, 0.0);
  }
};

/**
 * Intermediate polygon structure for clipping operations.
 * Used during Sutherland-Hodgman clipping.
 */
struct ClipPolygon {
  double3 verts[MAX_POLYGON_VERTS];
  int count;

  __host__ __device__ ClipPolygon() : count(0) {}

  __device__ void clear() {
    count = 0;
  }

  __device__ void addVertex(double3 v) {
    if (count < MAX_POLYGON_VERTS) {
      verts[count++] = v;
    }
  }
};

// ============================================================================
// Narrowphase collision detection class
// ============================================================================

struct Narrowphase {
  // Host data
  std::vector<ContactPatch> h_contactPatches;
  int numPatches;

  // Device data
  ContactPatch* d_contactPatches;

  // Mesh data pointers (shared with Broadphase or copied)
  double* d_nodes;     // (n_nodes * 3) node coordinates
  int* d_elements;     // (n_elems * nodesPerElement) element connectivity
  double* d_pressure;  // (n_nodes) pressure values at each node
  int n_nodes;
  int n_elems;
  int nodesPerElement;  // Should be 4 for linear tets (corners only)

  // Element-to-mesh mapping: elementMeshIds[elem_idx] = mesh_id
  // Used for correct normal direction (normal points from lower mesh ID to
  // higher)
  int* d_elementMeshIds;  // (n_elems) mesh ID for each element

  // Collision pair data (from broadphase)
  int* d_collisionPairs;  // Flattened pairs: [tetA0, tetB0, tetA1, tetB1, ...]
  int numCollisionPairs;

  // Device pointer to this struct for kernel access
  Narrowphase* d_np;

  // External forces buffer (reused across calls)
  double* d_f_ext;  // (3 * n_nodes) external forces on device
  int f_ext_size;   // Current allocated size (3 * n_nodes)

  // Constructor / Destructor
  Narrowphase();
  ~Narrowphase();

  // ========== Host API ==========

  /**
   * Initialize narrowphase with mesh data.
   * For 10-node tets, only the first 4 corner nodes are used.
   *
   * @param nodes Node coordinates (n_nodes x 3)
   * @param elements Element connectivity (n_elems x nodesPerElement)
   * @param pressure Pressure values at each node (n_nodes)
   * @param elementMeshIds Mesh ID for each element (n_elems). Used for normal
   * direction. Normal points from lower mesh ID to higher mesh ID. If empty,
   * all elements are treated as mesh 0.
   */
  void Initialize(const Eigen::MatrixXd& nodes, const Eigen::MatrixXi& elements,
                  const Eigen::VectorXd& pressure,
                  const Eigen::VectorXi& elementMeshIds = Eigen::VectorXi());

  /**
   * Set collision pairs from broadphase results.
   *
   * @param pairs Vector of (tetA, tetB) collision pairs
   */
  void SetCollisionPairs(const std::vector<std::pair<int, int>>& pairs);

  /**
   * Compute contact patches for all collision pairs.
   * This is the main narrowphase computation.
   */
  void ComputeContactPatches();

  /**
   * Retrieve contact patches to host memory.
   */
  void RetrieveResults();

  /**
   * Print contact patch results for debugging.
   */
  void PrintContactPatches(bool verbose = false);

  /**
   * Get valid contact patches (isValid == true).
   */
  std::vector<ContactPatch> GetValidPatches() const;

  /**
   * Cleanup GPU resources.
   */
  void Destroy();

  /**
   * Compute external forces from contact patches on GPU.
   * Uses already-on-device patch data (d_contactPatches) without CPU transfer.
   * Forces are computed using atomicAdd and returned as Eigen vector.
   *
   * @param d_vel   Optional device pointer to nodal velocities laid out as
   *                [vx0, vy0, vz0, vx1, ...]. If nullptr, no velocity-dependent
   *                damping is applied.
   * @param damping Normal damping coefficient (>= 0). If zero, no damping is
   *                applied even if d_vel is non-null.
   *
   * @return External forces vector of size 3 * n_nodes [fx0, fy0, fz0, fx1,
   * ...]
   */
  Eigen::VectorXd ComputeExternalForcesGPU(const double* d_vel = nullptr,
                                           double damping       = 0.0,
                                           double friction      = 0.0);

  /**
   * Get device pointer to external forces buffer.
   * Call ComputeExternalForcesGPU() first to populate the buffer.
   * Buffer layout: [fx0, fy0, fz0, fx1, fy1, fz1, ...]
   *
   * @return Device pointer to external forces (3 * n_nodes doubles)
   */
  double* GetExternalForcesDevicePtr() const {
    return d_f_ext;
  }

  // ========== Device accessors (for use in kernels) ==========
#if defined(__CUDACC__)
  // Node coordinate accessors (column-major)
  __device__ double node_x(int node_id) const {
    return d_nodes[node_id];
  }
  __device__ double node_y(int node_id) const {
    return d_nodes[node_id + n_nodes];
  }
  __device__ double node_z(int node_id) const {
    return d_nodes[node_id + 2 * n_nodes];
  }
  __device__ double3 getNode(int node_id) const {
    return make_double3(node_x(node_id), node_y(node_id), node_z(node_id));
  }

  // Element node accessor (column-major)
  __device__ int element_node(int elem_id, int local_node_idx) const {
    return d_elements[elem_id + local_node_idx * n_elems];
  }

  // Pressure accessor
  __device__ double getPressure(int node_id) const {
    return d_pressure[node_id];
  }
#endif
};

// ============================================================================
// Device helper functions (declared here, defined in .cu)
// ============================================================================

#if defined(__CUDACC__)

// ---------- Vector math utilities ----------
__device__ inline double3 operator+(double3 a, double3 b) {
  return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline double3 operator-(double3 a, double3 b) {
  return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline double3 operator*(double s, double3 v) {
  return make_double3(s * v.x, s * v.y, s * v.z);
}

__device__ inline double3 operator*(double3 v, double s) {
  return make_double3(v.x * s, v.y * s, v.z * s);
}

__device__ inline double dot(double3 a, double3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline double3 cross(double3 a, double3 b) {
  return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x);
}

__device__ inline double length(double3 v) {
  return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline double3 normalize(double3 v) {
  double len = length(v);
  if (len > 1e-12) {
    return (1.0 / len) * v;
  }
  return make_double3(0.0, 0.0, 0.0);
}

#endif  // __CUDACC__
