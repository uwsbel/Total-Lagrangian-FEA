#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace ANCFCPUUtils {

// A linear triangle soup surface mesh using global node IDs as vertex keys.
//
// Intended as an element-agnostic representation that can be extracted from
// different FE discretizations (T10 today; ANCF later) and consumed by
// collision backends (DEME, etc).
struct SurfaceTriMesh {
  // Unique global node IDs used by this surface mesh.
  // For T10: actual mesh node IDs.
  // For ANCF3243: synthetic IDs (collision system uses vertex positions).
  std::vector<int> global_node_ids;

  // Vertex positions (same length/order as global_node_ids).
  std::vector<Eigen::Vector3d> vertices;

  // Triangles as indices into `vertices` (0-based).
  std::vector<Eigen::Vector3i> triangles;

  // For ANCF elements: maps each surface vertex to the ANCF node ID whose
  // position DOF should receive the collision force. Empty for T10 meshes.
  // Same length as vertices when populated.
  std::vector<int> ancf_node_ids;
};

bool WriteObj(const SurfaceTriMesh& mesh, const std::string& path);

}  // namespace ANCFCPUUtils
