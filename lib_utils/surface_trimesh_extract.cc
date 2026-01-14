#include "surface_trimesh_extract.h"

#include <array>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ANCFCPUUtils {
namespace {

struct Array3Hash {
  size_t operator()(const std::array<int, 3>& k) const noexcept {
    size_t h = 1469598103934665603ull;
    for (int v : k) {
      h ^= static_cast<size_t>(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    }
    return h;
  }
};

static std::array<int, 3> SortedCorners(int a, int b, int c) {
  std::array<int, 3> k{a, b, c};
  if (k[0] > k[1])
    std::swap(k[0], k[1]);
  if (k[1] > k[2])
    std::swap(k[1], k[2]);
  if (k[0] > k[1])
    std::swap(k[0], k[1]);
  return k;
}

struct FaceRecord {
  // Oriented corners [a,b,c].
  std::array<int, 3> corners;
  // Mid-edge nodes corresponding to [ab, bc, ca] for the oriented corners.
  std::array<int, 3> mids;
  // Opposite tetra corner node (for outward orientation).
  int opp = -1;
};

static Eigen::Vector3d NodePos(const Eigen::MatrixXd& nodes, int global_id) {
  return nodes.row(global_id).transpose();
}

static void EnsureVertex(int global_id, const Eigen::MatrixXd& nodes,
                         std::unordered_map<int, int>& global_to_local,
                         SurfaceTriMesh& out) {
  const auto it = global_to_local.find(global_id);
  if (it != global_to_local.end())
    return;
  const int local_id = static_cast<int>(out.vertices.size());
  global_to_local.emplace(global_id, local_id);
  out.global_node_ids.push_back(global_id);
  out.vertices.push_back(NodePos(nodes, global_id));
}

static int LocalIndexForGlobal(
    int global_id, const std::unordered_map<int, int>& global_to_local) {
  const auto it = global_to_local.find(global_id);
  return (it == global_to_local.end()) ? -1 : it->second;
}

static SurfaceTriMesh ExtractSurfaceTriMeshFromT10(
    const Eigen::MatrixXd& nodes, const Eigen::MatrixXi& elements,
    const MeshInstance& inst) {
  SurfaceTriMesh out;
  std::unordered_map<int, int> global_to_local;

  std::unordered_map<std::array<int, 3>, std::vector<FaceRecord>, Array3Hash>
      faces;

  for (int e = 0; e < inst.num_elements; ++e) {
    const int elem_idx = inst.element_offset + e;
    const auto tet     = elements.row(elem_idx);

    const int n0 = tet(0);
    const int n1 = tet(1);
    const int n2 = tet(2);
    const int n3 = tet(3);

    // T10 ordering (see `lib_src/elements/FEAT10Data.cu`):
    // edge nodes (4..9): (0,1),(1,2),(0,2),(0,3),(1,3),(2,3)
    const int e01 = tet(4);
    const int e12 = tet(5);
    const int e02 = tet(6);
    const int e03 = tet(7);
    const int e13 = tet(8);
    const int e23 = tet(9);

    auto add_face = [&](int a, int b, int c, int ab, int bc, int ca, int opp) {
      FaceRecord r;
      r.corners = {a, b, c};
      r.mids    = {ab, bc, ca};
      r.opp     = opp;
      faces[SortedCorners(a, b, c)].push_back(r);
    };

    add_face(n0, n1, n2, e01, e12, e02, n3);  // opposite n3
    add_face(n0, n1, n3, e01, e13, e03, n2);  // opposite n2
    add_face(n1, n2, n3, e12, e23, e13, n0);  // opposite n0
    add_face(n0, n2, n3, e02, e23, e03, n1);  // opposite n1
  }

  for (const auto& kv : faces) {
    if (kv.second.size() != 1)
      continue;  // interior face
    FaceRecord f = kv.second.front();

    const Eigen::Vector3d pa = NodePos(nodes, f.corners[0]);
    const Eigen::Vector3d pb = NodePos(nodes, f.corners[1]);
    const Eigen::Vector3d pc = NodePos(nodes, f.corners[2]);
    const Eigen::Vector3d pd = NodePos(nodes, f.opp);

    Eigen::Vector3d n = (pb - pa).cross(pc - pa);
    if (n.dot(pd - pa) > 0.0) {
      // Flip so the normal points away from the tetra interior.
      std::swap(f.corners[1], f.corners[2]);
      // mids are [ab, bc, ca] -> after swapping b<->c, become [ca, bc, ab]
      f.mids = {f.mids[2], f.mids[1], f.mids[0]};
    }

    const int a  = f.corners[0];
    const int b  = f.corners[1];
    const int c  = f.corners[2];
    const int ab = f.mids[0];
    const int bc = f.mids[1];
    const int ca = f.mids[2];

    for (int v : {a, b, c, ab, bc, ca}) {
      EnsureVertex(v, nodes, global_to_local, out);
    }

    const int ia  = LocalIndexForGlobal(a, global_to_local);
    const int ib  = LocalIndexForGlobal(b, global_to_local);
    const int ic  = LocalIndexForGlobal(c, global_to_local);
    const int iab = LocalIndexForGlobal(ab, global_to_local);
    const int ibc = LocalIndexForGlobal(bc, global_to_local);
    const int ica = LocalIndexForGlobal(ca, global_to_local);

    out.triangles.emplace_back(ia, iab, ica);
    out.triangles.emplace_back(iab, ib, ibc);
    out.triangles.emplace_back(ica, ibc, ic);
    out.triangles.emplace_back(iab, ibc, ica);
  }

  return out;
}

}  // namespace

SurfaceTriMesh ExtractSurfaceTriMesh(const Eigen::MatrixXd& nodes,
                                     const Eigen::MatrixXi& elements,
                                     const MeshInstance& inst,
                                     SurfaceTriMeshExtractionHint hint) {
  if (hint == SurfaceTriMeshExtractionHint::kAuto) {
    if (elements.cols() == 10) {
      return ExtractSurfaceTriMeshFromT10(nodes, elements, inst);
    }
  }

  switch (hint) {
    case SurfaceTriMeshExtractionHint::kANCF3243:
      throw std::runtime_error(
          "ExtractSurfaceTriMesh: ANCF3243 extractor not implemented yet");
    case SurfaceTriMeshExtractionHint::kANCF3443:
      throw std::runtime_error(
          "ExtractSurfaceTriMesh: ANCF3443 extractor not implemented yet");
    case SurfaceTriMeshExtractionHint::kAuto:
      break;
  }
  throw std::invalid_argument(
      "ExtractSurfaceTriMesh: unable to infer discretization (pass an explicit "
      "SurfaceTriMeshExtractionHint)");
}

SurfaceTriMesh ExtractSurfaceTriMeshFromANCF3243(
    const Eigen::VectorXd& x12, const Eigen::VectorXd& y12,
    const Eigen::VectorXd& z12,
    const Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>& connectivity,
    double width, double height) {
  SurfaceTriMesh out;
  std::unordered_map<int, int> global_to_local;

  const int n_elements = static_cast<int>(connectivity.rows());
  const double hw      = 0.5 * width;
  const double hh      = 0.5 * height;

  // For each beam element, we generate a rectangular tube with 8 corner
  // vertices (4 at each end). The tube has 4 faces (top, bottom, left, right),
  // each face consists of 2 triangles.
  //
  // ANCF3243: 2 nodes per element, 4 DOFs per node (position + gradient).
  // DOF layout: [r, r,xi] per node. Position is the first DOF (index 0).

  // Helper to get node position from ANCF coefficients.
  // For node with ID `node_id`, position coefficients are at indices:
  // x12[4*node_id], y12[4*node_id], z12[4*node_id]
  auto get_node_pos = [&](int node_id) -> Eigen::Vector3d {
    const int coef_idx = 4 * node_id;
    return Eigen::Vector3d(x12(coef_idx), y12(coef_idx), z12(coef_idx));
  };

  // Helper to get tangent direction from ANCF gradient coefficients.
  // The gradient r,xi is at coefficient index 1 of each node.
  auto get_node_tangent = [&](int node_id) -> Eigen::Vector3d {
    const int coef_idx = 4 * node_id + 1;
    Eigen::Vector3d t(x12(coef_idx), y12(coef_idx), z12(coef_idx));
    double norm = t.norm();
    if (norm < 1e-12) {
      return Eigen::Vector3d(1.0, 0.0, 0.0);  // Fallback
    }
    return t / norm;
  };

  // For each beam element, generate tube surface.
  for (int e = 0; e < n_elements; ++e) {
    const int node0 = connectivity(e, 0);
    const int node1 = connectivity(e, 1);

    const Eigen::Vector3d p0 = get_node_pos(node0);
    const Eigen::Vector3d p1 = get_node_pos(node1);
    const Eigen::Vector3d t0 = get_node_tangent(node0);
    const Eigen::Vector3d t1 = get_node_tangent(node1);

    // Use averaged beam direction as reference.
    Eigen::Vector3d beam_dir = (p1 - p0).normalized();
    if (beam_dir.norm() < 1e-12) {
      beam_dir = t0;
    }

    // Compute local coordinate system at each end.
    auto compute_local_axes = [](const Eigen::Vector3d& tangent)
        -> std::pair<Eigen::Vector3d, Eigen::Vector3d> {
      Eigen::Vector3d up(0.0, 0.0, 1.0);
      if (std::abs(tangent.dot(up)) > 0.99) {
        up = Eigen::Vector3d(1.0, 0.0, 0.0);
      }
      Eigen::Vector3d local_y = tangent.cross(up).normalized();
      Eigen::Vector3d local_z = local_y.cross(tangent).normalized();
      return {local_y, local_z};
    };

    auto [y0, z0] = compute_local_axes(t0);
    auto [y1, z1] = compute_local_axes(t1);

    // Generate 4 corner vertices at each end of the beam.
    std::array<Eigen::Vector3d, 4> corners0 = {
        p0 + hw * y0 + hh * z0,
        p0 - hw * y0 + hh * z0,
        p0 - hw * y0 - hh * z0,
        p0 + hw * y0 - hh * z0};

    std::array<Eigen::Vector3d, 4> corners1 = {
        p1 + hw * y1 + hh * z1,
        p1 - hw * y1 + hh * z1,
        p1 - hw * y1 - hh * z1,
        p1 + hw * y1 - hh * z1};

    // Synthetic global IDs for collision system vertex tracking.
    const int base_id0 = 10000 * (e + 1);
    const int base_id1 = base_id0 + 4;

    std::array<int, 4> local_idx0, local_idx1;

    for (int c = 0; c < 4; ++c) {
      int gid0 = base_id0 + c;
      int gid1 = base_id1 + c;

      int lid0              = static_cast<int>(out.vertices.size());
      global_to_local[gid0] = lid0;
      out.global_node_ids.push_back(gid0);
      out.vertices.push_back(corners0[static_cast<size_t>(c)]);
      out.ancf_node_ids.push_back(node0);  // Map to ANCF node at this end
      local_idx0[static_cast<size_t>(c)] = lid0;

      int lid1              = static_cast<int>(out.vertices.size());
      global_to_local[gid1] = lid1;
      out.global_node_ids.push_back(gid1);
      out.vertices.push_back(corners1[static_cast<size_t>(c)]);
      out.ancf_node_ids.push_back(node1);  // Map to ANCF node at this end
      local_idx1[static_cast<size_t>(c)] = lid1;
    }

    // Generate triangles for the 4 side faces.
    for (int i = 0; i < 4; ++i) {
      int j   = (i + 1) % 4;
      int v00 = local_idx0[static_cast<size_t>(i)];
      int v01 = local_idx0[static_cast<size_t>(j)];
      int v10 = local_idx1[static_cast<size_t>(i)];
      int v11 = local_idx1[static_cast<size_t>(j)];

      out.triangles.emplace_back(v00, v10, v01);
      out.triangles.emplace_back(v01, v10, v11);
    }
  }

  return out;
}

}  // namespace ANCFCPUUtils

