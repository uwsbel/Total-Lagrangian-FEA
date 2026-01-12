#include "surface_trimesh.h"

#include <fstream>

namespace ANCFCPUUtils {

bool WriteObj(const SurfaceTriMesh& mesh, const std::string& path) {
  std::ofstream out(path);
  if (!out.is_open()) return false;

  for (const auto& v : mesh.vertices) {
    out << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
  }
  for (const auto& t : mesh.triangles) {
    // OBJ is 1-based.
    out << "f " << (t.x() + 1) << " " << (t.y() + 1) << " " << (t.z() + 1)
        << "\n";
  }
  return true;
}

}  // namespace ANCFCPUUtils

