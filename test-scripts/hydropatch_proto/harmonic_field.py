#!/usr/bin/env python3
import numpy as np

# ------------------------------------------------------------
# Mesh I/O (TetGen .node + .ele -> linear tet mesh)
# ------------------------------------------------------------

def load_tetgen_linear_mesh(node_path, ele_path):
    """
    Load a TetGen ASCII .node + .ele mesh and convert to a
    linear tetrahedral mesh.

    Handles:
      * 0-based or 1-based node indices in the .node/.ele files.
      * Higher-order tets (e.g., 10-node) by using only the first 4
        corner nodes of each element.

    Returns
    -------
    verts : (Nv, 3) float64
        Vertex coordinates.
    tets  : (Ne, 4) int64
        Tetrahedra as 0-based indices into verts.
    original_vertex_ids : (Nv,) int64
        For each row in 'verts', this stores the original TetGen
        node index from the .node file.
    """
    # --- Parse nodes ---
    with open(node_path, "r") as f:
        lines = [ln.strip() for ln in f
                 if ln.strip() and not ln.strip().startswith("#")]

    header = lines[0].split()
    n_points = int(header[0])
    dim = int(header[1])
    assert dim == 3, f"Expected 3D nodes, got dim={dim}"

    orig_ids = []
    coords = []
    for i in range(n_points):
        parts = lines[1 + i].split()
        idx = int(parts[0])   # might start at 0 or 1
        x, y, z = map(float, parts[1:4])
        orig_ids.append(idx)
        coords.append((x, y, z))
    orig_ids = np.array(orig_ids, dtype=np.int64)
    coords = np.array(coords, dtype=np.float64)

    # Map original node ID -> coordinate
    id_to_coord = {int(i): coords[k] for k, i in enumerate(orig_ids)}

    # --- Parse elements ---
    with open(ele_path, "r") as f:
        elines = [ln.strip() for ln in f
                  if ln.strip() and not ln.strip().startswith("#")]

    eheader = elines[0].split()
    n_tets = int(eheader[0])
    nodes_per_tet = int(eheader[1])
    assert nodes_per_tet >= 4, "Expected at least 4 nodes per tet"

    # Take the first 4 (corner) node IDs for each tet, in original ID space.
    corner_ids = np.zeros((n_tets, 4), dtype=np.int64)
    for e in range(n_tets):
        parts = elines[1 + e].split()
        # parts[0] is tet ID (0- or 1-based). We don't care here.
        for j in range(4):
            corner_ids[e, j] = int(parts[1 + j])

    # The linearized mesh only needs the corner nodes.
    used_orig_ids = np.unique(corner_ids.ravel())

    # Build mapping original node ID -> new contiguous 0..Nv-1
    id_to_new = {int(old_id): new_id
                 for new_id, old_id in enumerate(used_orig_ids)}

    Nv = len(used_orig_ids)
    verts = np.zeros((Nv, 3), dtype=np.float64)
    for old_id in used_orig_ids:
        verts[id_to_new[int(old_id)]] = id_to_coord[int(old_id)]

    Ne = n_tets
    tets = np.zeros((Ne, 4), dtype=np.int64)
    for e in range(Ne):
        for j in range(4):
            tets[e, j] = id_to_new[int(corner_ids[e, j])]

    return verts, tets, used_orig_ids


# ------------------------------------------------------------
# Boundary faces / vertices
# ------------------------------------------------------------

def build_boundary_faces(tets):
    """
    Given (Ne, 4) tets, construct boundary triangle faces and vertices.

    Returns
    -------
    boundary_faces : (K, 3) int64
        Triangle vertex indices (0-based into verts).
    boundary_verts : (Kb,) int64
        Unique vertex indices on the boundary.
    """
    from collections import defaultdict
    face_count = defaultdict(int)

    def add_face(a, b, c):
        face = tuple(sorted((a, b, c)))
        face_count[face] += 1

    Ne = tets.shape[0]
    for e in range(Ne):
        i0, i1, i2, i3 = tets[e]
        add_face(i0, i1, i2)
        add_face(i0, i1, i3)
        add_face(i0, i2, i3)
        add_face(i1, i2, i3)

    boundary_faces = [f for f, cnt in face_count.items() if cnt == 1]
    boundary_faces = np.array(boundary_faces, dtype=np.int64)
    boundary_verts = np.unique(boundary_faces.ravel())
    return boundary_faces, boundary_verts


# ------------------------------------------------------------
# Geometric helpers: point → segment / triangle distance
# ------------------------------------------------------------

def distance2_point_segments(p, a, b):
    """
    Squared distance from a point p to a batch of segments [a_k, b_k].

    Parameters
    ----------
    p : (3,) float
    a, b : (K, 3) float

    Returns
    -------
    dist2 : (K,) float
        Squared distances to each segment.
    """
    ab = b - a          # (K,3)
    ap = p[None, :] - a # (K,3)
    ab_dot_ab = np.einsum("ij,ij->i", ab, ab)  # (K,)
    eps = 1e-15
    denom = np.where(ab_dot_ab > eps, ab_dot_ab, 1.0)
    t = np.einsum("ij,ij->i", ap, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    diff = p[None, :] - closest
    return np.einsum("ij,ij->i", diff, diff)


def point_to_triangles_distance(p, tri0, tri1, tri2):
    """
    Shortest distance from point p to a set of triangles.

    Parameters
    ----------
    p : (3,)
        Query point.
    tri0, tri1, tri2 : (K,3)
        Triangle vertices for each triangle k.

    Returns
    -------
    dist : float
        Shortest distance from p to the union of all triangles.
    """
    # 1) Distance to edges (baseline)
    dist2_edges_ab = distance2_point_segments(p, tri0, tri1)
    dist2_edges_bc = distance2_point_segments(p, tri1, tri2)
    dist2_edges_ca = distance2_point_segments(p, tri2, tri0)
    dist2 = np.minimum(dist2_edges_ab,
                       np.minimum(dist2_edges_bc, dist2_edges_ca))

    # 2) For triangles where perpendicular projection falls inside,
    #    distance is just |signed distance to plane|.
    v0 = tri1 - tri0
    v1 = tri2 - tri0
    n = np.cross(v0, v1)
    n_norm = np.linalg.norm(n, axis=1)
    eps = 1e-15
    valid = n_norm > eps
    if np.any(valid):
        n_hat = np.zeros_like(n)
        n_hat[valid] = n[valid] / n_norm[valid, None]

        # vector from tri0 to p
        p_vec = p[None, :] - tri0  # (K,3)
        # signed distance along normal
        d_signed = np.einsum("ij,ij->i", p_vec, n_hat)
        p_proj = p[None, :] - d_signed[:, None] * n_hat

        # barycentric coordinates on the projected triangle
        w_vec = p_proj - tri0
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", w_vec, v0)
        d21 = np.einsum("ij,ij->i", w_vec, v1)
        denom = d00 * d11 - d01 * d01
        denom_valid = np.abs(denom) > eps
        inside_mask = valid & denom_valid

        if np.any(inside_mask):
            v = np.zeros_like(denom)
            w_b = np.zeros_like(denom)

            idx = inside_mask
            v[idx] = (d11[idx] * d20[idx] - d01[idx] * d21[idx]) / denom[idx]
            w_b[idx] = (d00[idx] * d21[idx] - d01[idx] * d20[idx]) / denom[idx]
            u = 1.0 - v - w_b

            # Allow tiny negative due to numerical noise
            inside_tri = (inside_mask &
                          (u >= -1e-12) &
                          (v >= -1e-12) &
                          (w_b >= -1e-12))

            if np.any(inside_tri):
                plane_dist2 = d_signed[inside_tri] ** 2
                dist2_inside = dist2[inside_tri]
                # True distance is perpendicular distance inside triangle
                dist2[inside_tri] = np.minimum(dist2_inside, plane_dist2)

    return float(np.sqrt(np.min(dist2)))


# ------------------------------------------------------------
# Distance-to-boundary field (using boundary triangles)
# ------------------------------------------------------------

def compute_distance_field_surface(verts, boundary_faces):
    """
    Compute distance to boundary *surface* (triangles) for each vertex,
    and normalize to [0,1].

    Parameters
    ----------
    verts : (Nv,3)
    boundary_faces : (K,3) int
        Indices into verts.

    Returns
    -------
    e_vertex : (Nv,) float64
        Normalized distance field in [0,1].
    dists : (Nv,) float64
        Raw distances to boundary surface.
    """
    Nv = verts.shape[0]
    # Pre-gather triangle vertex positions
    tri0 = verts[boundary_faces[:, 0]]
    tri1 = verts[boundary_faces[:, 1]]
    tri2 = verts[boundary_faces[:, 2]]

    dists = np.zeros(Nv, dtype=np.float64)
    for i in range(Nv):
        p = verts[i]
        dists[i] = point_to_triangles_distance(p, tri0, tri1, tri2)

    maxd = dists.max()
    if maxd > 0.0:
        e_vertex = dists / maxd
    else:
        e_vertex = np.zeros_like(dists)

    e_vertex = np.clip(e_vertex, 0.0, 1.0)
    return e_vertex, dists


def compute_per_tet_scalar(tets, e_vertex):
    """Per-tet scalar = average of its 4 vertex values."""
    return e_vertex[tets].mean(axis=1)


# ------------------------------------------------------------
# High-level pipeline
# ------------------------------------------------------------

def build_hydroelastic_scalar(node_path,
                              ele_path,
                              Eh=10000.0):
    """
    1) Load TetGen mesh and convert to linear tets (4-node).
    2) Extract boundary faces & verts.
    3) Compute per-vertex *surface* distance field d(x):
         d(x) = shortest distance to boundary triangles.
    4) Normalize: e = d / max(d).
    5) Define p = Eh * e.
    6) Compute per-tet values (simple averages).

    Parameters
    ----------
    node_path : str
    ele_path  : str
    Eh : float
        Hydroelastic modulus / pressure scale.

    Returns
    -------
    verts : (Nv, 3)
    tets  : (Ne, 4)
    e_vertex : (Nv,)
    e_tet    : (Ne,)
    p_vertex : (Nv,)
    p_tet    : (Ne,)
    boundary_faces : (K,3)
    boundary_verts : (Kb,)
    dists : (Nv,)
    original_vertex_ids : (Nv,)
    """
    verts, tets, original_vertex_ids = load_tetgen_linear_mesh(node_path, ele_path)
    boundary_faces, boundary_verts = build_boundary_faces(tets)

    e_vertex, dists = compute_distance_field_surface(verts, boundary_faces)
    e_tet = compute_per_tet_scalar(tets, e_vertex)

    p_vertex = Eh * e_vertex
    p_tet = Eh * e_tet

    return (verts, tets,
            e_vertex, e_tet,
            p_vertex, p_tet,
            boundary_faces, boundary_verts,
            dists, original_vertex_ids)


# ------------------------------------------------------------
# CSV export for ParaView (optional)
# ------------------------------------------------------------

def save_vertex_csv(csv_path, verts, scalars_dict):
    """
    Save per-vertex data to CSV for ParaView.

    Parameters
    ----------
    csv_path : str
        Output CSV filename.
    verts : (Nv, 3) float64
        Vertex coordinates.
    scalars_dict : dict[str, (Nv,) array_like]
        Mapping from scalar name to per-vertex array.
        e.g. {"e": e_vertex, "p": p_vertex}
    """
    cols = ["x", "y", "z"] + list(scalars_dict.keys())
    arrays = [verts] + [np.asarray(v).reshape(-1, 1) for v in scalars_dict.values()]
    data = np.hstack(arrays)

    header = ",".join(cols)
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")


# ------------------------------------------------------------
# VTU export for ParaView (unstructured grid)
# ------------------------------------------------------------

def save_vtu(vtu_path, verts, tets, e_vertex, p_vertex, e_tet, dists=None):
    """
    Save a VTK unstructured grid (.vtu) with tetrahedra and scalar fields.

    Parameters
    ----------
    vtu_path : str
        Output filename (.vtu).
    verts : (Nv, 3)
    tets  : (Ne, 4)
    e_vertex : (Nv,)
        Normalized distance field at vertices.
    p_vertex : (Nv,)
        Pressure-like scalar at vertices.
    e_tet : (Ne,)
        Per-tet scalar (e.g. averaged e).
    dists : (Nv,) or None
        Optional distance-to-boundary per vertex.
    """
    try:
        import meshio
    except ImportError:
        print("WARNING: meshio not installed; run `pip install meshio` to enable VTU export.")
        return

    cells = [("tetra", tets)]

    point_data = {
        "e": e_vertex,
        "p": p_vertex,
    }
    if dists is not None:
        point_data["dist"] = dists

    cell_data = {"e_tet": [e_tet]}

    mesh = meshio.Mesh(points=verts,
                       cells=cells,
                       point_data=point_data,
                       cell_data=cell_data)
    mesh.write(vtu_path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build surface-distance-based hydroelastic-like scalar field on a TetGen mesh."
    )
    parser.add_argument("--node", type=str, required=True, help="Path to .node file")
    parser.add_argument("--ele", type=str, required=True, help="Path to .ele file")
    parser.add_argument("--Eh", type=float, default=1.0,
                        help="Hydroelastic modulus / pressure scale (Pa or similar).")
    parser.add_argument("--out-prefix", type=str, default="field",
                        help="Prefix for output files (NPZ + CSV + VTU)")

    args = parser.parse_args()

    (verts, tets,
     e_v, e_t,
     p_v, p_t,
     boundary_faces, boundary_verts,
     dists, original_vertex_ids) = build_hydroelastic_scalar(
        args.node,
        args.ele,
        Eh=args.Eh,
    )

    # 1) Save full data as NPZ
    out_npz = f"{args.out_prefix}.npz"
    np.savez(
        out_npz,
        verts=verts,
        tets=tets,
        e_vertex=e_v,
        e_tet=e_t,
        p_vertex=p_v,
        p_tet=p_t,
        boundary_faces=boundary_faces,
        boundary_verts=boundary_verts,
        dists=dists,
        original_vertex_ids=original_vertex_ids,
    )
    print("Saved scalar field to", out_npz)

    # 2) Optional: per-vertex CSV
    out_csv = f"{args.out_prefix}_vertices.csv"
    save_vertex_csv(out_csv, verts, {"e": e_v, "p": p_v, "dist": dists})
    print("Saved per-vertex CSV to", out_csv)

    # 3) VTU for ParaView
    out_vtu = f"{args.out_prefix}.vtu"
    save_vtu(out_vtu, verts, tets, e_v, p_v, e_t, dists=dists)
    print("Saved VTU to", out_vtu)

    print("Num vertices (linear mesh):", verts.shape[0],
          "Num tets:", tets.shape[0])
    print("Num boundary verts:", len(boundary_verts),
          "Num boundary faces:", boundary_faces.shape[0])
