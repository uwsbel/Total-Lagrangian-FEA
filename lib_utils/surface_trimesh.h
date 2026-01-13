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
  std::vector<int> global_node_ids;

  // Vertex positions (same length/order as global_node_ids).
  std::vector<Eigen::Vector3d> vertices;

  // Triangles as indices into `vertices` (0-based).
  std::vector<Eigen::Vector3i> triangles;
};

bool WriteObj(const SurfaceTriMesh& mesh, const std::string& path);

}  // namespace ANCFCPUUtils
