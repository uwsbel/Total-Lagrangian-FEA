#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# ------------------------------------------------------------
# TetGen .node + .ele reader (10-node -> linear 4-node tets)
# ------------------------------------------------------------

def load_tetgen_linear_mesh(node_path, ele_path):
    """
    Load TetGen ASCII .node + .ele mesh and convert to a
    linear tetrahedral mesh (4-node tets).

    Returns
    -------
    verts : (Nv, 3) float64
        Vertex coordinates.
    tets  : (Ne, 4) int64
        Tetrahedra as 0-based indices into verts.
    used_orig_ids : (Nv,) int64
        Original TetGen node IDs used as corners.
    """
    # --- Parse nodes ---
    with open(node_path, "r") as f:
        lines = [ln.strip() for ln in f
                 if ln.strip() and not ln.strip().startswith("#")]

    header = lines[0].split()
    n_points = int(header[0])
    dim = int(header[1])
    # header[2] = num_attrs, header[3] = has_boundary_markers
    assert dim == 3, f"Expected 3D nodes, got dim={dim}"

    orig_ids = []
    coords = []
    for i in range(n_points):
        parts = lines[1 + i].split()
        idx = int(parts[0])   # may be 0-based or 1-based; we keep it as-is
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
    # eheader[2] = num_attrs (ignored)
    assert nodes_per_tet >= 4, "Expected at least 4 nodes per tet"

    # Take the first 4 (corner) node IDs for each tet, in original ID space.
    corner_ids = np.zeros((n_tets, 4), dtype=np.int64)
    for e in range(n_tets):
        parts = elines[1 + e].split()
        # parts[0] is tet ID; ignore
        for j in range(4):
            corner_ids[e, j] = int(parts[1 + j])

    # Linearized mesh only needs the corner nodes.
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
# Slice: intersect tets with plane x = x0
# ------------------------------------------------------------

def slice_mesh_with_plane_x(verts, tets, scalar, x0=0.0, eps=1e-9):
    """
    Intersect a tetrahedral mesh with the plane x = x0.

    Parameters
    ----------
    verts : (Nv,3)
    tets  : (Ne,4)
    scalar : (Nv,)
        Per-vertex scalar (e.g., pressure).
    x0 : float
        Plane location along x.
    eps : float
        Tolerance around the plane.

    Returns
    -------
    slice_tris : (M,3,3)
        Triangles lying on the plane (3D coordinates).
    slice_tri_vals : (M,)
        Average scalar for each triangle.
    """
    edge_pairs = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
    ], dtype=int)

    slice_tris = []
    slice_vals = []

    for tet in tets:
        v = verts[tet]      # (4,3)
        s = scalar[tet]     # (4,)
        d = v[:, 0] - x0    # signed distance to plane along x

        # If all on one side, no intersection
        if np.all(d > eps) or np.all(d < -eps):
            continue

        pts = []
        vals = []

        # 1) Vertices exactly on the plane (within eps)
        for i in range(4):
            if abs(d[i]) <= eps:
                pts.append(v[i].copy())
                vals.append(s[i])

        # 2) Edge intersections
        for (i, j) in edge_pairs:
            di, dj = d[i], d[j]
            # Opposite signs => crosses plane
            if di * dj < -eps * eps:
                t = di / (di - dj)  # interpolation parameter
                p = v[i] + t * (v[j] - v[i])
                val = s[i] + t * (s[j] - s[i])
                pts.append(p)
                vals.append(val)

        if len(pts) < 3:
            continue

        pts = np.array(pts)
        vals = np.array(vals)

        # Deduplicate approximate duplicates in (y,z)
        yz = pts[:, 1:3]
        tol = 1e-8
        yz_rounded = np.round(yz / tol) * tol
        _, unique_idx = np.unique(yz_rounded, axis=0, return_index=True)
        pts = pts[unique_idx]
        vals = vals[unique_idx]

        if len(pts) < 3:
            continue

        # Order polygon vertices around centroid in yz plane
        yz = pts[:, 1:3]
        centroid = yz.mean(axis=0)
        angles = np.arctan2(yz[:, 1] - centroid[1], yz[:, 0] - centroid[0])
        order = np.argsort(angles)
        pts = pts[order]
        vals = vals[order]

        n = len(pts)
        if n < 3:
            continue

        # Triangulate polygon as fan (0, i, i+1)
        for i in range(1, n - 1):
            tri = np.vstack([pts[0], pts[i], pts[i + 1]])
            tri_val = np.mean([vals[0], vals[i], vals[i + 1]])
            slice_tris.append(tri)
            slice_vals.append(tri_val)

    if len(slice_tris) == 0:
        return np.zeros((0, 3, 3)), np.zeros((0,))
    return np.array(slice_tris), np.array(slice_vals)


# ------------------------------------------------------------
# 2D visualization in YZ plane
# ------------------------------------------------------------

def plot_slice_yz(slice_tris, slice_vals, title="Slice at x = 0"):
    """
    Plot the cross-section in the y-z plane, colored by scalar.
    """
    polys_yz = [tri[:, 1:3] for tri in slice_tris]  # drop x

    fig, ax = plt.subplots(figsize=(6, 6))
    pc = PolyCollection(polys_yz, array=slice_vals, cmap="viridis", edgecolors="k", linewidths=0.2)
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal", "box")

    cbar = fig.colorbar(pc, ax=ax, shrink=0.8)
    cbar.set_label("pressure")

    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Main: load mesh + pressure from field.npz, slice at x=0
# ------------------------------------------------------------

if __name__ == "__main__":
    node_path = "bunny_ascii_26.1.node"
    ele_path = "bunny_ascii_26.1.ele"
    npz_path = "field.npz"   # produced by your hydroelastic script

    # 1) Read TetGen mesh and linearize
    verts, tets, used_orig_ids = load_tetgen_linear_mesh(node_path, ele_path)
    print("Loaded mesh: verts =", verts.shape[0], ", tets =", tets.shape[0])

    # 2) Load pressure from field.npz
    data = np.load(npz_path)
    p_vertex = data["p_vertex"]                # (Nv,)
    original_ids_npz = data["original_vertex_ids"]

    # Sanity check: mapping must match
    if not np.array_equal(used_orig_ids, original_ids_npz):
        raise RuntimeError(
            "Vertex mapping mismatch between TetGen loader and field.npz!\n"
            "Make sure field.npz was generated from the same .node/.ele."
        )

    print("Loaded pressure for", p_vertex.shape[0], "vertices.")

    # 3) Slice at x = 0
    slice_tris, slice_vals = slice_mesh_with_plane_x(verts, tets, p_vertex, x0=0.0)
    print("Slice triangles:", slice_tris.shape[0])

    if slice_tris.shape[0] == 0:
        print("No intersection with plane x=0 (check your mesh's x-range).")
    else:
        plot_slice_yz(slice_tris, slice_vals, title="Bunny slice at x = 0")
