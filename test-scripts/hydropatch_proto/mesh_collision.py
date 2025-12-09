#!/usr/bin/env python3
"""
Iso-pressure contact patches between two translated spheres.

Assumes you have a TetGen linear mesh:
    sphere.1.node, sphere.1.ele
generated from the same geometry (sphere radius ~0.15).

Uses:
  - load_tetgen_linear_mesh from reader.py
  - geometry helpers from tet_intersect.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from reader import load_tetgen_linear_mesh
from tet_intersect import affine_from_tet, plane_tet_intersection, clip_polygon_with_tet


# ------------------------------------------------------------
# Transform utilities
# ------------------------------------------------------------

def transform_mesh(verts, rotation_matrix=None, translation=None):
    """
    Apply rotation and translation to mesh vertices.
    Rotation is about the centroid.
    """
    transformed = verts.copy()

    if rotation_matrix is not None:
        centroid = verts.mean(axis=0)
        transformed = (transformed - centroid) @ rotation_matrix.T + centroid

    if translation is not None:
        transformed = transformed + translation

    return transformed


# ------------------------------------------------------------
# AABB utilities for broad-phase
# ------------------------------------------------------------

def compute_tet_aabb(verts, tet_indices):
    tet_verts = verts[tet_indices]
    return tet_verts.min(axis=0), tet_verts.max(axis=0)


def aabb_overlap(aabb1, aabb2, eps=1e-9):
    min1, max1 = aabb1
    min2, max2 = aabb2
    for i in range(3):
        if max1[i] < min2[i] - eps or max2[i] < min1[i] - eps:
            return False
    return True


# ------------------------------------------------------------
# Hydro-like pressure field for a sphere
# ------------------------------------------------------------

def hydro_pressure(verts, center, R, Eh=1.0):
    """
    Simple hydroelastic-like field:
        p(x) = Eh * max(0, R - ||x - center||).

    For two identical spheres with same R and Eh but different centers,
    p_A(x) = p_B(x) => ||x-c_A|| = ||x-c_B||, i.e. the perpendicular
    bisector plane between centers.
    """
    r = np.linalg.norm(verts - center, axis=1)
    return Eh * np.maximum(0.0, R - r)


# ------------------------------------------------------------
# Iso-pressure patch for a tet pair
# ------------------------------------------------------------

def compute_iso_pressure_patch(tet_A_verts, tet_B_verts, p_A, p_B, eps=1e-9):
    """
    Compute the iso-pressure surface patch for two tetrahedra:
        p_A(x) = p_B(x).

    p_A, p_B are 4-vertex values; we build affine fields
      p(x) = a·x + b
    inside each tet, then intersect the plane
      (a_A - a_B)·x + (b_A - b_B) = 0
    with both tets.

    Returns:
      patch: (k,3) vertices (possibly empty),
      nhat : (3,) oriented normal from A into B,
      g_A  : -∂p_A/∂n,
      g_B  :  ∂p_B/∂n,
      p_e  : equilibrium pressure at centroid,
      valid_orientation: True iff g_A>0 and g_B>0.
    """
    try:
        a_A, b_A = affine_from_tet(tet_A_verts, p_A)
        a_B, b_B = affine_from_tet(tet_B_verts, p_B)
    except np.linalg.LinAlgError:
        return np.zeros((0, 3)), None, 0.0, 0.0, 0.0, False

    n = a_A - a_B
    c = b_A - b_B

    n_norm = np.linalg.norm(n)
    if n_norm < eps:
        # parallel fields -> no iso-plane
        return np.zeros((0, 3)), None, 0.0, 0.0, 0.0, False

    # plane ∩ tet A
    patch_A = plane_tet_intersection(tet_A_verts, n, c, eps=eps)
    if patch_A.shape[0] < 3:
        return np.zeros((0, 3)), None, 0.0, 0.0, 0.0, False

    # clip with tet B
    patch = clip_polygon_with_tet(patch_A, tet_B_verts, eps=eps)
    if patch.shape[0] < 3:
        return np.zeros((0, 3)), None, 0.0, 0.0, 0.0, False

    nhat = n / n_norm

    # Directional gradients
    g_A = -np.dot(a_A, nhat)
    g_B = np.dot(a_B, nhat)

    valid_orientation = True
    if g_A <= 0 or g_B <= 0:
        nhat = -nhat
        g_A = -np.dot(a_A, nhat)
        g_B = np.dot(a_B, nhat)
        if g_A <= 0 or g_B <= 0:
            valid_orientation = False

    centroid = patch.mean(axis=0)
    p_e = np.dot(a_A, centroid) + b_A

    return patch, nhat, g_A, g_B, p_e, valid_orientation


# ------------------------------------------------------------
# Collision detection
# ------------------------------------------------------------

def detect_mesh_collisions(verts_A, tets_A, pressure_A,
                           verts_B, tets_B, pressure_B,
                           use_broadphase=True, verbose=True):
    n_tets_A = tets_A.shape[0]
    n_tets_B = tets_B.shape[0]

    if verbose:
        print(f"Checking {n_tets_A} x {n_tets_B} tet pairs...")

    if use_broadphase:
        aabbs_A = [compute_tet_aabb(verts_A, tets_A[i]) for i in range(n_tets_A)]
        aabbs_B = [compute_tet_aabb(verts_B, tets_B[i]) for i in range(n_tets_B)]

    patches = []
    patch_info = []
    patch_normals = []
    patch_data = []
    checked = 0
    skipped = 0
    invalid_normals = 0

    for i in range(n_tets_A):
        tet_A_verts = verts_A[tets_A[i]]
        p_A = pressure_A[tets_A[i]]

        for j in range(n_tets_B):
            if use_broadphase and not aabb_overlap(aabbs_A[i], aabbs_B[j]):
                skipped += 1
                continue

            checked += 1
            tet_B_verts = verts_B[tets_B[j]]
            p_B = pressure_B[tets_B[j]]

            patch, nhat, g_A, g_B, p_e, valid_orientation = compute_iso_pressure_patch(
                tet_A_verts, tet_B_verts, p_A, p_B
            )

            if patch.shape[0] >= 3:
                patches.append(patch)
                patch_info.append((i, j))
                patch_normals.append(nhat)
                patch_data.append({
                    "g_A": g_A,
                    "g_B": g_B,
                    "p_e": p_e,
                    "centroid": patch.mean(axis=0),
                    "valid_orientation": valid_orientation,
                })
                if not valid_orientation:
                    invalid_normals += 1

    if verbose:
        print(f"  Broad-phase skipped: {skipped}")
        print(f"  Narrow-phase checked: {checked}")
        print(f"  Found {len(patches)} iso-pressure patches")
        print(f"  Patches with valid normals: {len(patches) - invalid_normals}")

    return patches, patch_info, patch_normals, patch_data


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

def _get_tet_faces(verts, tet_indices):
    v = verts[tet_indices]
    return [
        [v[0], v[1], v[2]],
        [v[0], v[1], v[3]],
        [v[0], v[2], v[3]],
        [v[1], v[2], v[3]],
    ]


def plot_meshes_and_patches(verts_A, tets_A, verts_B, tets_B, patches,
                            patch_normals=None, patch_data=None,
                            title="Mesh Collision: Iso-Pressure Surfaces",
                            show_tets=True, tet_alpha=0.05,
                            show_normals=True, normal_scale=0.5):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    if show_tets:
        max_tets_show = min(500, tets_A.shape[0])
        step_A = max(1, tets_A.shape[0] // max_tets_show)
        faces_A = []
        for i in range(0, tets_A.shape[0], step_A):
            faces_A.extend(_get_tet_faces(verts_A, tets_A[i]))

        poly_A = Poly3DCollection(faces_A, alpha=tet_alpha,
                                  edgecolor="blue", linewidths=0.1)
        poly_A.set_facecolor("tab:blue")
        ax.add_collection3d(poly_A)

        max_tets_show = min(500, tets_B.shape[0])
        step_B = max(1, tets_B.shape[0] // max_tets_show)
        faces_B = []
        for i in range(0, tets_B.shape[0], step_B):
            faces_B.extend(_get_tet_faces(verts_B, tets_B[i]))

        poly_B = Poly3DCollection(faces_B, alpha=tet_alpha,
                                  edgecolor="orange", linewidths=0.1)
        poly_B.set_facecolor("tab:orange")
        ax.add_collection3d(poly_B)

    if patches:
        patch_polys = Poly3DCollection(patches, alpha=0.8,
                                       edgecolor="k", linewidths=0.5)
        patch_polys.set_facecolor("tab:green")
        ax.add_collection3d(patch_polys)

    if show_normals and patch_normals is not None and patch_data is not None:
        valid_count = 0
        invalid_count = 0
        for nhat, data in zip(patch_normals, patch_data):
            if nhat is None:
                continue
            centroid = data["centroid"]
            valid = data.get("valid_orientation", True)
            color = "red" if valid else "yellow"
            ax.quiver(centroid[0], centroid[1], centroid[2],
                      nhat[0] * normal_scale,
                      nhat[1] * normal_scale,
                      nhat[2] * normal_scale,
                      color=color, arrow_length_ratio=0.3, linewidth=1.5)
            if valid:
                valid_count += 1
            else:
                invalid_count += 1
        print(f"    Plotted {valid_count} valid (red) + {invalid_count} invalid (yellow) normals")

    all_verts = np.vstack([verts_A, verts_B])
    mins = all_verts.min(axis=0) - 0.05
    maxs = all_verts.max(axis=0) + 0.05
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="tab:blue", alpha=0.3, label="Mesh A"),
        Patch(facecolor="tab:orange", alpha=0.3, label="Mesh B (transformed)"),
        Patch(facecolor="tab:green", alpha=0.8, label="Iso-pressure surfaces"),
        Line2D([0], [0], color="red", linewidth=2, label="Valid normals (A→B)"),
        Line2D([0], [0], color="yellow", linewidth=2, label="Invalid normals"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    return fig, ax


def plot_patches_only(patches, patch_normals=None, patch_data=None,
                      title="Iso-Pressure Surface Patches",
                      show_normals=True, normal_scale=0.5):
    if not patches:
        print("No patches to visualize.")
        return None, None

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))
    for i, patch in enumerate(patches):
        if patch.shape[0] >= 3:
            poly = Poly3DCollection([patch], alpha=0.7,
                                    edgecolor="k", linewidths=0.3)
            poly.set_facecolor(colors[i])
            ax.add_collection3d(poly)

    if show_normals and patch_normals is not None and patch_data is not None:
        for nhat, data in zip(patch_normals, patch_data):
            if nhat is None:
                continue
            centroid = data["centroid"]
            valid = data.get("valid_orientation", True)
            color = "red" if valid else "yellow"
            ax.quiver(centroid[0], centroid[1], centroid[2],
                      nhat[0] * normal_scale,
                      nhat[1] * normal_scale,
                      nhat[2] * normal_scale,
                      color=color, arrow_length_ratio=0.3, linewidth=1.5)

    all_pts = np.vstack(patches)
    mins = all_pts.min(axis=0) - 0.1
    maxs = all_pts.max(axis=0) + 0.1
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title} ({len(patches)} patches)")

    plt.tight_layout()
    return fig, ax


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    node_path = "sphere.1.node"
    ele_path = "sphere.1.ele"

    print("=" * 60)
    print("Iso-Pressure Contact for Two Spheres")
    print("=" * 60)

    # 1) Load mesh
    print("\n[1] Loading mesh...")
    verts, tets, used_orig_ids = load_tetgen_linear_mesh(node_path, ele_path)
    print(f"    Loaded mesh: {verts.shape[0]} vertices, {tets.shape[0]} tetrahedra")

    # 2) Build meshes A and B
    print("\n[2] Setting up Mesh A & B (translated spheres)...")
    verts_A = verts.copy()
    tets_A = tets.copy()

    translation = np.array([0.2, 0.0, 0.1])
    verts_B = transform_mesh(verts, rotation_matrix=None, translation=translation)
    tets_B = tets.copy()

    c_A = verts_A.mean(axis=0)
    c_B = verts_B.mean(axis=0)
    # Estimate radius from mesh
    R_est = np.max(np.linalg.norm(verts_A - c_A, axis=1))
    R = R_est
    print(f"    Estimated radius R ≈ {R:.5f}")
    print(f"    Centers: c_A = {c_A}, c_B = {c_B}")

    pressure_A = hydro_pressure(verts_A, c_A, R)
    pressure_B = hydro_pressure(verts_B, c_B, R)

    print(f"    pressure_A range: [{pressure_A.min():.4e}, {pressure_A.max():.4e}]")
    print(f"    pressure_B range: [{pressure_B.min():.4e}, {pressure_B.max():.4e}]")

    # 3) Detect collisions
    print("\n[3] Detecting iso-pressure patches...")
    patches, patch_info, patch_normals, patch_data = detect_mesh_collisions(
        verts_A, tets_A, pressure_A,
        verts_B, tets_B, pressure_B,
        use_broadphase=True,
        verbose=True,
    )

    # 4) Visualize
    print("\n[4] Visualizing...")
    if patches:
        fig1, ax1 = plot_meshes_and_patches(
            verts_A, tets_A, verts_B, tets_B, patches,
            patch_normals=patch_normals,
            patch_data=patch_data,
            title=f"Sphere Collision: {len(patches)} Iso-Pressure Patches",
            show_tets=True,
            tet_alpha=0.06,
            show_normals=True,
            normal_scale=0.03,
        )

        fig2, ax2 = plot_patches_only(
            patches,
            patch_normals=patch_normals,
            patch_data=patch_data,
            title="Iso-Pressure Surface Patches (Contact Region)",
            show_normals=True,
            normal_scale=0.03,
        )

        # area stats
        total_area = 0.0
        for patch in patches:
            if patch.shape[0] < 3:
                continue
            centroid = patch.mean(axis=0)
            area = 0.0
            for i in range(patch.shape[0]):
                v1 = patch[i] - centroid
                v2 = patch[(i + 1) % patch.shape[0] - 0] - centroid
                area += 0.5 * np.linalg.norm(np.cross(v1, v2))
            total_area += area
        print(f"    Total iso-pressure surface area: {total_area:.6f}")
    else:
        print("    No iso-pressure patches found.")
        fig1, ax1 = plot_meshes_and_patches(
            verts_A, tets_A, verts_B, tets_B, [],
            patch_normals=None,
            patch_data=None,
            title="Spheres (No Patches)",
            show_tets=True,
            tet_alpha=0.1,
            show_normals=False,
        )

    # Save figs
    fig1.savefig("collision_meshes.png", dpi=150, bbox_inches="tight")
    if patches:
        fig2.savefig("collision_patches.png", dpi=150, bbox_inches="tight")
    print("    Saved figures.")

    plt.show()
    print("\n[Done]")


if __name__ == "__main__":
    main()
