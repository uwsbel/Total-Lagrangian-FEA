#pragma once

#include <Eigen/Dense>

#include "lib_utils/mesh_manager.h"
#include "lib_utils/surface_trimesh.h"

namespace ANCFCPUUtils {

// Hint for how to interpret `elements` when extracting a surface triangle mesh.
// `kAuto` uses cheap heuristics (currently: `elements.cols() == 10` => T10).
enum class SurfaceTriMeshExtractionHint {
  kAuto = 0,
  kANCF3243 = 1,
  kANCF3443 = 2,
};

// Extract a linear triangle soup surface mesh for collision/rendering.
//
// Today this supports 10-node quadratic tetrahedral meshes. ANCF extractors will be added later.
SurfaceTriMesh ExtractSurfaceTriMesh(
    const Eigen::MatrixXd& nodes,
    const Eigen::MatrixXi& elements,
    const MeshInstance& inst,
    SurfaceTriMeshExtractionHint hint = SurfaceTriMeshExtractionHint::kAuto);

}  // namespace ANCFCPUUtils
