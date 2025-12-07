#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace ANCFCPUUtils {

/**
 * Structure representing a single mesh instance with offset information
 */
struct MeshInstance {
  int node_offset;     // Starting index in global node array
  int element_offset;  // Starting index in global element array
  int num_nodes;
  int num_elements;
  std::string name;  // Optional: for debugging/identification
};

/**
 * MeshManager class for handling multiple meshes with unified indexing
 *
 * This class manages loading multiple meshes and provides unified node/element
 * arrays where element indices are automatically shifted to account for
 * the combined node array. This is useful for collision detection and
 * other operations that need to work with multiple meshes simultaneously.
 */
class MeshManager {
 public:
  MeshManager() = default;

  /**
   * Load a mesh from TetGen .node and .ele files
   * @param node_file Path to the .node file
   * @param elem_file Path to the .ele file
   * @param name Optional name for the mesh instance
   * @return Mesh instance ID (>= 0 on success, -1 on failure)
   */
  int LoadMesh(const std::string& node_file, const std::string& elem_file,
               const std::string& name = "");

  /**
   * Apply a 4x4 transformation matrix to a specific mesh instance
   * @param mesh_id The mesh instance ID to transform
   * @param transform 4x4 transformation matrix (homogeneous coordinates)
   */
  void TransformMesh(int mesh_id, const Eigen::Matrix4d& transform);

  /**
   * Translate a specific mesh instance by a given offset
   * @param mesh_id The mesh instance ID to translate
   * @param dx Translation in X direction
   * @param dy Translation in Y direction
   * @param dz Translation in Z direction
   */
  void TranslateMesh(int mesh_id, double dx, double dy, double dz);

  /**
   * Get the unified node array containing all mesh nodes
   * @return Reference to the combined node matrix (n_total_nodes × 3)
   */
  const Eigen::MatrixXd& GetAllNodes() const {
    return all_nodes_;
  }

  /**
   * Get the unified element array with shifted indices
   * @return Reference to the combined element matrix (n_total_elements × 10)
   */
  const Eigen::MatrixXi& GetAllElements() const {
    return all_elements_;
  }

  /**
   * Get mesh instance information by ID
   * @param mesh_id The mesh instance ID
   * @return Reference to the MeshInstance structure
   */
  const MeshInstance& GetMeshInstance(int mesh_id) const;

  /**
   * Get the total number of loaded meshes
   * @return Number of mesh instances
   */
  int GetNumMeshes() const {
    return static_cast<int>(mesh_instances_.size());
  }

  /**
   * Get the total number of nodes across all meshes
   * @return Total node count
   */
  int GetTotalNodes() const {
    return total_nodes_;
  }

  /**
   * Get the total number of elements across all meshes
   * @return Total element count
   */
  int GetTotalElements() const {
    return total_elements_;
  }

  /**
   * Load scalar field from NPZ file for a specific mesh
   * The NPZ file should contain 'p_vertex' (or specified key) and
   * 'original_vertex_ids'
   * @param mesh_id The mesh instance ID to load scalar field for
   * @param npz_file Path to the .npz file
   * @param field_key Key in the NPZ file for the scalar field (default:
   * "p_vertex")
   * @return True on success, false on failure
   */
  bool LoadScalarFieldFromNpz(int mesh_id, const std::string& npz_file,
                              const std::string& field_key = "p_vertex");

  /**
   * Load scalar field from binary file for a specific mesh
   * @param mesh_id The mesh instance ID to load scalar field for
   * @param bin_file Path to the binary file (raw float64 array)
   * @param n_values Number of values to read
   * @return True on success, false on failure
   */
  bool LoadScalarFieldFromBinary(int mesh_id, const std::string& bin_file,
                                 int n_values);

  /**
   * Set scalar field directly for a mesh
   * @param mesh_id The mesh instance ID
   * @param field Scalar field values (must match mesh node count)
   * @return True on success, false on failure
   */
  bool SetScalarField(int mesh_id, const Eigen::VectorXd& field);

  /**
   * Get the unified scalar field array for all meshes
   * @return Reference to the combined scalar field vector
   */
  const Eigen::VectorXd& GetAllScalarFields() const {
    return all_scalar_fields_;
  }

  /**
   * Check if scalar fields have been loaded
   * @return True if scalar fields are available
   */
  bool HasScalarFields() const {
    return has_scalar_fields_;
  }

  /**
   * Given a global element index, find which mesh it belongs to
   * @param global_elem_idx The global element index
   * @return Mesh ID, or -1 if not found
   */
  int GetMeshIdFromElement(int global_elem_idx) const;

  /**
   * Given a global node index, find which mesh it belongs to
   * @param global_node_idx The global node index
   * @return Mesh ID, or -1 if not found
   */
  int GetMeshIdFromNode(int global_node_idx) const;

  /**
   * Clear all loaded meshes
   */
  void Clear();

 private:
  /**
   * Rebuild the unified node and element arrays from individual mesh buffers
   */
  void RebuildUnifiedArrays();

  /**
   * Rebuild the unified scalar field array from individual mesh buffers
   */
  void RebuildScalarFieldArray();

  std::vector<MeshInstance> mesh_instances_;
  std::vector<Eigen::MatrixXd> node_buffers_;  // Per-mesh nodes (local coords)
  std::vector<Eigen::MatrixXi>
      elem_buffers_;  // Per-mesh elements (local indices)
  std::vector<Eigen::VectorXd> scalar_field_buffers_;  // Per-mesh scalar fields

  Eigen::MatrixXd all_nodes_;  // Unified node array
  Eigen::MatrixXi
      all_elements_;  // Unified element array (with shifted indices)
  Eigen::VectorXd all_scalar_fields_;  // Unified scalar field array

  int total_nodes_        = 0;
  int total_elements_     = 0;
  bool has_scalar_fields_ = false;
};

}  // namespace ANCFCPUUtils
