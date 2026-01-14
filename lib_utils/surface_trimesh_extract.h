#pragma once

#include <Eigen/Dense>

#include "lib_utils/mesh_manager.h"
#include "lib_utils/surface_trimesh.h"

namespace ANCFCPUUtils {

// Hint for how to interpret `elements` when extracting a surface triangle mesh.
// `kAuto` uses cheap heuristics (currently: `elements.cols() == 10` => T10).
enum class SurfaceTriMeshExtractionHint {
  kAuto     = 0,
  kANCF3243 = 1,
  kANCF3443 = 2,
};

// Extract a linear triangle soup surface mesh for collision/rendering.
//
// Today this supports 10-node quadratic tetrahedral meshes. ANCF extractors
// will be added later.
SurfaceTriMesh ExtractSurfaceTriMesh(
    const Eigen::MatrixXd& nodes, const Eigen::MatrixXi& elements,
    const MeshInstance& inst,
    SurfaceTriMeshExtractionHint hint = SurfaceTriMeshExtractionHint::kAuto);

// Extract a surface triangle mesh from ANCF3243 beam elements.
//
// Each beam is represented as a rectangular tube with cross-section
// dimensions (width, height). The function generates triangles for
// the outer surface of all beam elements.
//
// Parameters:
//   x12, y12, z12: ANCF coefficient vectors (size = 4 * n_nodes)
//   connectivity: Element connectivity matrix (n_elements x 2), node IDs
//   width, height: Cross-section dimensions
//
// Returns:
//   SurfaceTriMesh with global node IDs corresponding to ANCF coefficient
//   indices (to allow proper force mapping from collision system).
SurfaceTriMesh ExtractSurfaceTriMeshFromANCF3243(
    const Eigen::VectorXd& x12, const Eigen::VectorXd& y12,
    const Eigen::VectorXd& z12,
    const Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>& connectivity,
    double width, double height);

// Merge coincident vertices in a surface mesh within a given tolerance.
// This eliminates small overlaps at beam-beam connections that can cause
// excessive self-collision detection in DEME.
//
// Parameters:
//   mesh: Input surface mesh (modified in place)
//   tolerance: Distance threshold for merging vertices (default 1e-6)
void MergeCoincidentVertices(SurfaceTriMesh& mesh, double tolerance = 1e-6);

}  // namespace ANCFCPUUtils
