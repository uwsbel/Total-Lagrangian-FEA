#!/usr/bin/env python3
"""
Compare CUDA narrowphase results with Python prototype.

This script:
1. Runs the Python prototype collision detection
2. Loads CUDA results from JSON
3. Compares the results

Uses the same sphere mesh and translation as the CUDA test.
"""

import numpy as np
import json
import os
import sys

# Change to the project root directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'test-scripts/hydropatch_proto')

from reader import load_tetgen_linear_mesh
from tet_intersect import affine_from_tet, plane_tet_intersection, clip_polygon_with_tet


def hydro_pressure(verts, center, R, Eh=1.0):
    """Hydroelastic pressure field: p(x) = Eh * max(0, R - ||x - center||)"""
    r = np.linalg.norm(verts - center, axis=1)
    return Eh * np.maximum(0.0, R - r)


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


def compute_iso_pressure_patch(tet_A_verts, tet_B_verts, p_A, p_B, eps=1e-9):
    """Compute iso-pressure surface patch for two tetrahedra."""
    try:
        a_A, b_A = affine_from_tet(tet_A_verts, p_A)
        a_B, b_B = affine_from_tet(tet_B_verts, p_B)
    except np.linalg.LinAlgError:
        return np.zeros((0, 3)), None, 0.0, 0.0, 0.0, False

    n = a_A - a_B
    c = b_A - b_B

    n_norm = np.linalg.norm(n)
    if n_norm < eps:
        return np.zeros((0, 3)), None, 0.0, 0.0, 0.0, False

    patch_A = plane_tet_intersection(tet_A_verts, n, c, eps=eps)
    if patch_A.shape[0] < 3:
        return np.zeros((0, 3)), None, 0.0, 0.0, 0.0, False

    patch = clip_polygon_with_tet(patch_A, tet_B_verts, eps=eps)
    if patch.shape[0] < 3:
        return np.zeros((0, 3)), None, 0.0, 0.0, 0.0, False

    nhat = n / n_norm
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


def compute_polygon_area(poly):
    """Compute area of a convex polygon."""
    if poly.shape[0] < 3:
        return 0.0
    
    centroid = poly.mean(axis=0)
    area = 0.0
    for i in range(poly.shape[0]):
        v1 = poly[i] - centroid
        v2 = poly[(i + 1) % poly.shape[0]] - centroid
        area += 0.5 * np.linalg.norm(np.cross(v1, v2))
    return area


def load_t10_nodes(node_path):
    """Load ALL T10 nodes (not just corners)."""
    with open(node_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]
    
    header = lines[0].split()
    n_nodes = int(header[0])
    
    nodes = np.zeros((n_nodes, 3), dtype=np.float64)
    for i in range(n_nodes):
        parts = lines[1 + i].split()
        idx = int(parts[0]) - 1  # Convert to 0-indexed
        nodes[idx] = [float(parts[1]), float(parts[2]), float(parts[3])]
    
    return nodes


def main():
    print("="*60)
    print("Python vs CUDA Narrowphase Comparison")
    print("="*60)
    
    # Load mesh (linearized - only corner nodes for tets)
    node_path = "data/meshes/T10/sphere.1.node"
    ele_path = "data/meshes/T10/sphere.1.ele"
    
    print("\n[1] Loading mesh...")
    verts, tets, used_orig_ids = load_tetgen_linear_mesh(node_path, ele_path)
    print(f"    Loaded: {verts.shape[0]} vertices, {tets.shape[0]} tets (linearized)")
    
    # Also load ALL T10 nodes for center/radius computation (to match CUDA)
    t10_nodes = load_t10_nodes(node_path)
    print(f"    T10 mesh: {t10_nodes.shape[0]} total nodes")
    
    # Setup meshes A and B
    verts_A = verts.copy()
    tets_A = tets.copy()
    
    translation = np.array([0.1, 0.0, 0.0])  # Same as CUDA test
    verts_B = verts.copy() + translation
    tets_B = tets.copy()
    
    # Also translate T10 nodes for center computation
    t10_nodes_A = t10_nodes.copy()
    t10_nodes_B = t10_nodes.copy() + translation
    
    # Compute centers from ALL T10 nodes (same as CUDA)
    center_A = t10_nodes_A.mean(axis=0)
    center_B = t10_nodes_B.mean(axis=0)
    
    # Estimate radius from ALL T10 nodes
    R = np.max(np.linalg.norm(t10_nodes_A - center_A, axis=1))
    
    print(f"\n[2] Mesh setup:")
    print(f"    Center A: {center_A}")
    print(f"    Center B: {center_B}")
    print(f"    Radius R: {R:.6f}")
    print(f"    Translation: {translation}")
    
    # Compute pressure fields (centered at each mesh's center)
    pressure_A = hydro_pressure(verts_A, center_A, R)
    pressure_B = hydro_pressure(verts_B, center_B, R)  # Uses center_B!
    
    print(f"\n[3] Pressure fields:")
    print(f"    Pressure A range: [{pressure_A.min():.6e}, {pressure_A.max():.6e}]")
    print(f"    Pressure B range: [{pressure_B.min():.6e}, {pressure_B.max():.6e}]")
    
    # Run Python narrowphase
    print("\n[4] Running Python narrowphase...")
    n_tets = tets.shape[0]
    
    aabbs_A = [compute_tet_aabb(verts_A, tets_A[i]) for i in range(n_tets)]
    aabbs_B = [compute_tet_aabb(verts_B, tets_B[i]) for i in range(n_tets)]
    
    py_patches = []
    py_patch_info = []
    checked = 0
    skipped = 0
    
    for i in range(n_tets):
        for j in range(n_tets):
            if not aabb_overlap(aabbs_A[i], aabbs_B[j]):
                skipped += 1
                continue
            
            checked += 1
            tet_A_verts = verts_A[tets_A[i]]
            tet_B_verts = verts_B[tets_B[j]]
            p_A = pressure_A[tets_A[i]]
            p_B = pressure_B[tets_B[j]]
            
            patch, nhat, g_A, g_B, p_e, valid = compute_iso_pressure_patch(
                tet_A_verts, tet_B_verts, p_A, p_B
            )
            
            if patch.shape[0] >= 3:
                area = compute_polygon_area(patch)
                py_patches.append({
                    'tetA': i,
                    'tetB': j + n_tets,  # Offset for mesh B (matches CUDA indexing)
                    'area': area,
                    'centroid': patch.mean(axis=0),
                    'normal': nhat,
                    'g_A': g_A,
                    'g_B': g_B,
                    'p_e': p_e,
                    'valid': valid,
                    'n_verts': patch.shape[0]
                })
    
    print(f"    Broadphase skipped: {skipped}")
    print(f"    Narrowphase checked: {checked}")
    print(f"    Found {len(py_patches)} patches")
    
    # Load CUDA results
    print("\n[5] Loading CUDA results...")
    with open("output/contact_patches.json", 'r') as f:
        cuda_data = json.load(f)
    
    cuda_patches = cuda_data['contact_patches']
    print(f"    CUDA found {len(cuda_patches)} patches")
    
    # Compare statistics
    print("\n[6] Comparison:")
    print(f"    Python patches: {len(py_patches)}")
    print(f"    CUDA patches:   {len(cuda_patches)}")
    
    py_total_area = sum(p['area'] for p in py_patches)
    cuda_total_area = sum(p['area'] for p in cuda_patches)
    print(f"\n    Python total area: {py_total_area:.6e}")
    print(f"    CUDA total area:   {cuda_total_area:.6e}")
    print(f"    Difference:        {abs(py_total_area - cuda_total_area):.6e}")
    
    py_valid = sum(1 for p in py_patches if p['valid'])
    cuda_valid = sum(1 for p in cuda_patches if p['valid_orientation'])
    print(f"\n    Python valid orientations: {py_valid}")
    print(f"    CUDA valid orientations:   {cuda_valid}")
    
    # Check if we can match tet pairs
    py_pairs = set((p['tetA'], p['tetB']) for p in py_patches)
    cuda_pairs = set((p['tetA_idx'], p['tetB_idx']) for p in cuda_patches)
    
    common_pairs = py_pairs & cuda_pairs
    py_only = py_pairs - cuda_pairs
    cuda_only = cuda_pairs - py_pairs
    
    print(f"\n    Common tet pairs: {len(common_pairs)}")
    print(f"    Python-only pairs: {len(py_only)}")
    print(f"    CUDA-only pairs: {len(cuda_only)}")
    
    print("\n" + "="*60)
    print("Note: Differences may arise from:")
    print("  - Python uses linearized tets (4 corners only)")
    print("  - CUDA uses T10 elements (first 4 of 10 nodes)")
    print("  - Numerical precision differences")
    print("  - Different tolerance handling at boundaries")
    print("="*60)


if __name__ == "__main__":
    main()
