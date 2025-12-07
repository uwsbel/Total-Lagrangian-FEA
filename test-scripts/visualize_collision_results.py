#!/usr/bin/env python3
"""
Visualize collision detection results: meshes + contact patches in 3D.

This script loads:
1. Mesh data from TetGen .node/.ele files
2. Contact patch data from JSON export (includes full vertex geometry)

And creates 3D visualizations similar to mesh_collision.py showing:
- Both meshes (as transparent tetrahedra)
- Contact patches (as solid polygons)
- Contact normals (as arrows)

Usage:
    python visualize_collision_results.py --json output/contact_patches.json \\
        --mesh1 data/meshes/T10/sphere.1 --mesh2 data/meshes/T10/sphere.1 \\
        --translate2 0.1 0 0
        
For hydroelastic spheres, the iso-pressure surface should be the perpendicular
bisector plane between the two sphere centers (when using identical hydroelastic
pressure fields centered at each sphere's center).
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ------------------------------------------------------------
# Mesh loading utilities
# ------------------------------------------------------------

def load_tetgen_node_file(filename):
    """
    Load vertices from TetGen .node file.
    
    Returns
    -------
    verts : (N, 3) array
        Vertex coordinates
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split()
    n_verts = int(header[0])
    dim = int(header[1])
    
    if dim != 3:
        raise ValueError(f"Expected 3D mesh, got {dim}D")
    
    verts = np.zeros((n_verts, 3))
    
    # Find min vertex ID for offset
    min_id = float('inf')
    vert_data = []
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 4:
            vid = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            min_id = min(min_id, vid)
            vert_data.append((vid, x, y, z))
    
    # Fill vertices with adaptive offset
    offset = 0 if min_id == 0 else 1
    for vid, x, y, z in vert_data:
        idx = vid - offset
        if 0 <= idx < n_verts:
            verts[idx] = [x, y, z]
    
    return verts


def load_tetgen_ele_file(filename):
    """
    Load tetrahedra from TetGen .ele file.
    
    Returns
    -------
    tets : (M, 4) array for linear or (M, 10) for quadratic
        Element connectivity (0-indexed)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split()
    n_tets = int(header[0])
    nodes_per_tet = int(header[1])
    
    tets = np.zeros((n_tets, nodes_per_tet), dtype=int)
    
    # Find min element and node IDs for offset
    min_elem_id = float('inf')
    min_node_id = float('inf')
    elem_data = []
    
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= nodes_per_tet + 1:
            eid = int(parts[0])
            node_ids = [int(parts[i+1]) for i in range(nodes_per_tet)]
            min_elem_id = min(min_elem_id, eid)
            min_node_id = min(min_node_id, min(node_ids))
            elem_data.append((eid, node_ids))
    
    # Fill elements with adaptive offset
    elem_offset = 0 if min_elem_id == 0 else 1
    node_offset = 0 if min_node_id == 0 else 1
    
    for eid, node_ids in elem_data:
        idx = eid - elem_offset
        if 0 <= idx < n_tets:
            tets[idx] = [nid - node_offset for nid in node_ids]
    
    return tets


def load_mesh(base_path):
    """
    Load mesh from TetGen files.
    
    Parameters
    ----------
    base_path : str
        Base path without extension (e.g., 'data/meshes/T10/sphere.1')
        
    Returns
    -------
    verts : (N, 3) array
    tets : (M, 4) or (M, 10) array
    """
    node_file = base_path + '.node'
    ele_file = base_path + '.ele'
    
    verts = load_tetgen_node_file(node_file)
    tets = load_tetgen_ele_file(ele_file)
    
    return verts, tets


# ------------------------------------------------------------
# Contact patch loading
# ------------------------------------------------------------

def load_json_patches(filename):
    """Load contact patches from JSON file with full vertex geometry."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    patches = []
    for p in data['contact_patches']:
        patch = {
            'tetA_idx': p['tetA_idx'],
            'tetB_idx': p['tetB_idx'],
            'centroid': np.array(p['centroid']),
            'normal': np.array(p['normal']),
            'area': p['area'],
            'g_A': p['g_A'],
            'g_B': p['g_B'],
            'p_equilibrium': p['p_equilibrium'],
            'vertices': np.array(p['vertices']),
            'valid_orientation': p['valid_orientation']
        }
        patches.append(patch)
    return patches


# ------------------------------------------------------------
# Visualization functions
# ------------------------------------------------------------

def get_tet_faces(verts, tet_indices):
    """Get the 4 triangular faces of a tetrahedron."""
    # Use only first 4 nodes for T10 elements
    v = verts[tet_indices[:4]]
    return [
        [v[0], v[1], v[2]],
        [v[0], v[1], v[3]],
        [v[0], v[2], v[3]],
        [v[1], v[2], v[3]],
    ]


def plot_meshes_and_patches(verts_A, tets_A, verts_B, tets_B, patches,
                            title="Mesh Collision: Iso-Pressure Surfaces",
                            show_tets=True, tet_alpha=0.05,
                            show_normals=True, normal_scale=0.02,
                            max_tets_show=500):
    """
    Visualize two meshes and their iso-pressure surface patches.
    
    Parameters
    ----------
    verts_A, tets_A : Mesh A vertices and elements
    verts_B, tets_B : Mesh B vertices and elements
    patches : list of patch dicts with 'vertices', 'centroid', 'normal', etc.
    show_tets : bool
        Whether to draw tet faces
    tet_alpha : float
        Transparency for tet faces
    show_normals : bool
        Whether to draw normal arrows
    normal_scale : float
        Scale factor for normal arrow length
    max_tets_show : int
        Maximum tetrahedra to display (for performance)
    """
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot mesh A (subset for performance)
    if show_tets and tets_A is not None:
        step_A = max(1, tets_A.shape[0] // max_tets_show)
        faces_A = []
        for i in range(0, tets_A.shape[0], step_A):
            faces_A.extend(get_tet_faces(verts_A, tets_A[i]))
        
        poly_A = Poly3DCollection(faces_A, alpha=tet_alpha, 
                                   edgecolor='blue', linewidths=0.1)
        poly_A.set_facecolor('tab:blue')
        ax.add_collection3d(poly_A)
    
    # Plot mesh B
    if show_tets and tets_B is not None:
        step_B = max(1, tets_B.shape[0] // max_tets_show)
        faces_B = []
        for i in range(0, tets_B.shape[0], step_B):
            faces_B.extend(get_tet_faces(verts_B, tets_B[i]))
        
        poly_B = Poly3DCollection(faces_B, alpha=tet_alpha, 
                                   edgecolor='orange', linewidths=0.1)
        poly_B.set_facecolor('tab:orange')
        ax.add_collection3d(poly_B)
    
    # Plot iso-pressure patches
    valid_count = 0
    invalid_count = 0
    
    if patches:
        patch_verts = [p['vertices'] for p in patches if 'vertices' in p]
        if patch_verts:
            patch_polys = Poly3DCollection(patch_verts, alpha=0.8, 
                                           edgecolor='k', linewidths=0.5)
            patch_polys.set_facecolor('tab:green')
            ax.add_collection3d(patch_polys)
        
        # Plot contact normals as arrows
        if show_normals:
            for patch in patches:
                centroid = patch['centroid']
                normal = patch['normal']
                valid = patch.get('valid_orientation', True)
                color = 'red' if valid else 'yellow'
                
                ax.quiver(
                    centroid[0], centroid[1], centroid[2],
                    normal[0] * normal_scale, 
                    normal[1] * normal_scale, 
                    normal[2] * normal_scale,
                    color=color, arrow_length_ratio=0.3, linewidth=1.5
                )
                
                if valid:
                    valid_count += 1
                else:
                    invalid_count += 1
    
    if show_normals and patches:
        print(f"  Plotted {valid_count} valid (red) + {invalid_count} invalid (yellow) normal arrows")
    
    # Set axis limits
    all_verts = []
    if verts_A is not None:
        all_verts.append(verts_A)
    if verts_B is not None:
        all_verts.append(verts_B)
    
    if all_verts:
        all_verts = np.vstack(all_verts)
        mins = all_verts.min(axis=0) - 0.02
        maxs = all_verts.max(axis=0) + 0.02
        
        # Make axes equal
        max_range = (maxs - mins).max() / 2
        mid = (mins + maxs) / 2
        
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='tab:blue', alpha=0.3, label='Mesh A'),
        Patch(facecolor='tab:orange', alpha=0.3, label='Mesh B (transformed)'),
        Patch(facecolor='tab:green', alpha=0.8, label='Iso-pressure surfaces'),
        Line2D([0], [0], color='red', linewidth=2, label='Valid normals (A→B)'),
        Line2D([0], [0], color='yellow', linewidth=2, label='Invalid normals'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    return fig, ax


def plot_patches_only(patches, title="Iso-Pressure Surface Patches",
                      show_normals=True, normal_scale=0.02,
                      color_by='area'):
    """
    Visualize only the iso-pressure patches with optional normals.
    
    Parameters
    ----------
    patches : list of patch dicts
    show_normals : bool
    normal_scale : float
    color_by : str
        'area', 'g_A', 'g_B', 'p_equilibrium', or 'index'
    """
    if not patches:
        print("No patches to visualize.")
        return None, None
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get color values
    if color_by == 'area':
        values = [p['area'] for p in patches]
        cbar_label = 'Contact Patch Area'
    elif color_by == 'g_A':
        values = [p['g_A'] for p in patches]
        cbar_label = 'g_A (Pressure Gradient A)'
    elif color_by == 'g_B':
        values = [p['g_B'] for p in patches]
        cbar_label = 'g_B (Pressure Gradient B)'
    elif color_by == 'p_equilibrium':
        values = [p['p_equilibrium'] for p in patches]
        cbar_label = 'Equilibrium Pressure'
    else:  # index
        values = list(range(len(patches)))
        cbar_label = 'Patch Index'
    
    # Normalize colors
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-10:
        vmax = vmin + 1e-10
    norm_values = [(v - vmin) / (vmax - vmin) for v in values]
    
    cmap = plt.cm.viridis
    
    # Plot each patch with color
    valid_count = 0
    invalid_count = 0
    
    for i, patch in enumerate(patches):
        if 'vertices' in patch and patch['vertices'].shape[0] >= 3:
            verts = [patch['vertices']]
            color = cmap(norm_values[i])
            poly = Poly3DCollection(verts, alpha=0.7, 
                                    edgecolor='k', linewidths=0.3)
            poly.set_facecolor(color)
            ax.add_collection3d(poly)
    
    # Plot contact normals as arrows
    if show_normals:
        for patch in patches:
            centroid = patch['centroid']
            normal = patch['normal']
            valid = patch.get('valid_orientation', True)
            color = 'red' if valid else 'yellow'
            
            ax.quiver(
                centroid[0], centroid[1], centroid[2],
                normal[0] * normal_scale, 
                normal[1] * normal_scale, 
                normal[2] * normal_scale,
                color=color, arrow_length_ratio=0.3, linewidth=1.5
            )
            
            if valid:
                valid_count += 1
            else:
                invalid_count += 1
        
        print(f"  Plotted {valid_count} valid (red) + {invalid_count} invalid (yellow) normals")
    
    # Set axis limits
    all_pts = np.vstack([p['vertices'] for p in patches if 'vertices' in p])
    mins = all_pts.min(axis=0) - 0.01
    maxs = all_pts.max(axis=0) + 0.01
    
    # Make axes equal
    max_range = (maxs - mins).max() / 2
    mid = (mins + maxs) / 2
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title} ({len(patches)} patches)")
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label(cbar_label)
    
    plt.tight_layout()
    return fig, ax


def print_summary(patches):
    """Print summary statistics of contact patches."""
    print("\n" + "="*60)
    print("CONTACT PATCH SUMMARY")
    print("="*60)
    
    areas = [p['area'] for p in patches]
    g_A = [p['g_A'] for p in patches]
    g_B = [p['g_B'] for p in patches]
    p_eq = [p['p_equilibrium'] for p in patches]
    
    print(f"Total patches: {len(patches)}")
    
    # Compute total area
    total_area = sum(areas)
    
    print(f"\nArea statistics:")
    print(f"  Min:    {np.min(areas):.6e}")
    print(f"  Max:    {np.max(areas):.6e}")
    print(f"  Mean:   {np.mean(areas):.6e}")
    print(f"  Std:    {np.std(areas):.6e}")
    print(f"  Total:  {total_area:.6e}")
    
    print(f"\ng_A statistics:")
    print(f"  Min:    {np.min(g_A):.6e}")
    print(f"  Max:    {np.max(g_A):.6e}")
    print(f"  Mean:   {np.mean(g_A):.6e}")
    
    print(f"\ng_B statistics:")
    print(f"  Min:    {np.min(g_B):.6e}")
    print(f"  Max:    {np.max(g_B):.6e}")
    print(f"  Mean:   {np.mean(g_B):.6e}")
    
    print(f"\nEquilibrium pressure statistics:")
    print(f"  Min:    {np.min(p_eq):.6e}")
    print(f"  Max:    {np.max(p_eq):.6e}")
    print(f"  Mean:   {np.mean(p_eq):.6e}")
    
    # Count vertex distribution
    n_verts = [p['vertices'].shape[0] for p in patches if 'vertices' in p]
    print(f"\nPatch vertex count:")
    print(f"  Min:    {np.min(n_verts)}")
    print(f"  Max:    {np.max(n_verts)}")
    print(f"  Mean:   {np.mean(n_verts):.1f}")
    
    valid_count = sum(1 for p in patches if p.get('valid_orientation', True))
    print(f"\nOrientation validity:")
    print(f"  Valid:   {valid_count} ({100*valid_count/len(patches):.1f}%)")
    print(f"  Invalid: {len(patches)-valid_count} ({100*(len(patches)-valid_count)/len(patches):.1f}%)")
    print("="*60 + "\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Visualize collision detection results: meshes + contact patches')
    
    # Required
    parser.add_argument('--json', type=str, required=True,
                        help='Path to contact patches JSON file')
    
    # Mesh files (optional - for showing meshes)
    parser.add_argument('--mesh1', type=str, default=None,
                        help='Base path for mesh 1 (e.g., data/meshes/T10/sphere.1)')
    parser.add_argument('--mesh2', type=str, default=None,
                        help='Base path for mesh 2 (same format as mesh1)')
    
    # Mesh 2 transformation
    parser.add_argument('--translate2', type=float, nargs=3, default=[0, 0, 0],
                        metavar=('X', 'Y', 'Z'),
                        help='Translation for mesh 2 (default: 0 0 0)')
    
    # Visualization options
    parser.add_argument('--no-tets', action='store_true',
                        help='Do not show tetrahedral mesh faces')
    parser.add_argument('--no-normals', action='store_true',
                        help='Do not show normal vectors')
    parser.add_argument('--tet-alpha', type=float, default=0.05,
                        help='Transparency for tet faces (default: 0.05)')
    parser.add_argument('--normal-scale', type=float, default=0.02,
                        help='Scale factor for normal arrows (default: 0.02)')
    parser.add_argument('--color-by', type=str, default='area',
                        choices=['area', 'g_A', 'g_B', 'p_equilibrium', 'index'],
                        help='Property to color patches by (default: area)')
    parser.add_argument('--max-tets', type=int, default=500,
                        help='Max tetrahedra to display (default: 500)')
    
    # Output
    parser.add_argument('--save', type=str, default=None,
                        help='Save figure to file')
    parser.add_argument('--patches-only', action='store_true',
                        help='Show only patches (no mesh)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print statistics, no visualization')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Collision Detection Results Visualization")
    print("="*60)
    
    # Load patches
    print(f"\n[1] Loading contact patches from {args.json}...")
    patches = load_json_patches(args.json)
    print(f"    Loaded {len(patches)} contact patches")
    
    # Print summary
    print_summary(patches)
    
    if args.stats_only:
        return
    
    # Load meshes if provided
    verts_A, tets_A = None, None
    verts_B, tets_B = None, None
    
    if args.mesh1:
        print(f"\n[2] Loading mesh 1 from {args.mesh1}...")
        verts_A, tets_A = load_mesh(args.mesh1)
        print(f"    Loaded: {verts_A.shape[0]} vertices, {tets_A.shape[0]} elements")
    
    if args.mesh2:
        print(f"\n[3] Loading mesh 2 from {args.mesh2}...")
        verts_B, tets_B = load_mesh(args.mesh2)
        print(f"    Loaded: {verts_B.shape[0]} vertices, {tets_B.shape[0]} elements")
        
        # Apply translation
        translation = np.array(args.translate2)
        if np.any(translation != 0):
            verts_B = verts_B + translation
            print(f"    Applied translation: {translation}")
    
    # Visualization
    print(f"\n[4] Creating visualization...")
    
    if args.patches_only or (args.mesh1 is None and args.mesh2 is None):
        # Show only patches
        fig, ax = plot_patches_only(
            patches,
            title="Iso-Pressure Surface Patches",
            show_normals=not args.no_normals,
            normal_scale=args.normal_scale,
            color_by=args.color_by
        )
    else:
        # Show meshes and patches
        fig, ax = plot_meshes_and_patches(
            verts_A, tets_A, verts_B, tets_B, patches,
            title=f"Mesh Collision: {len(patches)} Iso-Pressure Patches",
            show_tets=not args.no_tets,
            tet_alpha=args.tet_alpha,
            show_normals=not args.no_normals,
            normal_scale=args.normal_scale,
            max_tets_show=args.max_tets
        )
    
    # Save figure
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"    Saved figure to {args.save}")
    
    plt.show()
    print("\n[Done]")


if __name__ == "__main__":
    main()
