#!/usr/bin/env python3
"""
Visualization script for three spheres collision test results.
Displays contact patches colored by mesh pair.

Usage:
    python3 test-scripts/visualize_three_spheres.py
    python3 test-scripts/visualize_three_spheres.py --json output/three_spheres_patches.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch


def get_mesh_id(tet_idx, elems_per_mesh=256):
    """Determine which mesh a tetrahedron belongs to based on its index."""
    return tet_idx // elems_per_mesh


def main():
    parser = argparse.ArgumentParser(description='Visualize three spheres collision results')
    parser.add_argument('--json', type=str, default='output/three_spheres_patches.json',
                        help='Path to contact patches JSON file')
    parser.add_argument('--elems-per-mesh', type=int, default=256,
                        help='Number of elements per mesh (default: 256)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save figure to file instead of showing')
    args = parser.parse_args()

    # Load contact patches
    with open(args.json, 'r') as f:
        data = json.load(f)

    patches = data['contact_patches']
    print(f"Loaded {len(patches)} contact patches from {args.json}")

    # Sphere centers (from test output)
    centers = [
        np.array([-0.000189511, 0.000372958, -0.00110778]),  # Mesh 0 (origin)
        np.array([0.19981, 0.000372958, 0.0988922]),          # Mesh 1 [0.2, 0, 0.1]
        np.array([0.0998105, 0.200373, 0.0988922])            # Mesh 2 [0.1, 0.2, 0.1]
    ]
    R = 0.151177

    # Count patches by mesh pair
    mesh_pair_patches = {(0,1): [], (0,2): [], (1,2): []}
    for p in patches:
        meshA = get_mesh_id(p['tetA_idx'], args.elems_per_mesh)
        meshB = get_mesh_id(p['tetB_idx'], args.elems_per_mesh)
        key = (min(meshA, meshB), max(meshA, meshB))
        if key in mesh_pair_patches:
            mesh_pair_patches[key].append(p)

    print("\nPatches by mesh pair:")
    for k, v in mesh_pair_patches.items():
        print(f"  Mesh {k[0]} <-> Mesh {k[1]}: {len(v)} patches")

    # Stats
    areas = [p['area'] for p in patches]
    print(f"\nTotal area: {sum(areas):.6f}")
    print(f"All valid orientation: {all(p['valid_orientation'] for p in patches)}")

    # Create visualization
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere wireframes
    sphere_colors = ['blue', 'orange', 'green']
    sphere_labels = ['Sphere 0 (origin)', 'Sphere 1 [0.2, 0, 0.1]', 'Sphere 2 [0.1, 0.2, 0.1]']
    
    for i, (center, color) in enumerate(zip(centers, sphere_colors)):
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = center[0] + R * np.outer(np.cos(u), np.sin(v))
        y = center[1] + R * np.outer(np.sin(u), np.sin(v))
        z = center[2] + R * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color=color, alpha=0.15, linewidth=0.5)
        ax.scatter(*center, color=color, s=100, marker='o')

    # Draw contact patches with different colors per mesh pair
    patch_colors = {(0,1): 'red', (0,2): 'purple', (1,2): 'cyan'}
    for key, patch_list in mesh_pair_patches.items():
        for p in patch_list:
            verts = np.array(p['vertices'])
            if len(verts) >= 3:
                poly = Poly3DCollection([verts], alpha=0.7, edgecolor='k', linewidth=0.3)
                poly.set_facecolor(patch_colors[key])
                ax.add_collection3d(poly)

    # Draw normal arrows (sample every 10th to avoid clutter)
    for i, p in enumerate(patches[::10]):
        centroid = np.array(p['centroid'])
        normal = np.array(p['normal'])
        ax.quiver(centroid[0], centroid[1], centroid[2],
                  normal[0]*0.03, normal[1]*0.03, normal[2]*0.03,
                  color='black', arrow_length_ratio=0.3, linewidth=1)

    # Set axis limits
    all_centers = np.array(centers)
    margin = R * 1.5
    ax.set_xlim([all_centers[:,0].min() - margin, all_centers[:,0].max() + margin])
    ax.set_ylim([all_centers[:,1].min() - margin, all_centers[:,1].max() + margin])
    ax.set_zlim([all_centers[:,2].min() - margin, all_centers[:,2].max() + margin])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Three Spheres Contact: {len(patches)} patches\n'
                 f'Red: 0↔1 ({len(mesh_pair_patches[(0,1)])}), '
                 f'Purple: 0↔2 ({len(mesh_pair_patches[(0,2)])}), '
                 f'Cyan: 1↔2 ({len(mesh_pair_patches[(1,2)])})')

    # Custom legend
    legend_elements = [
        Patch(facecolor='blue', alpha=0.3, label=sphere_labels[0]),
        Patch(facecolor='orange', alpha=0.3, label=sphere_labels[1]),
        Patch(facecolor='green', alpha=0.3, label=sphere_labels[2]),
        Patch(facecolor='red', alpha=0.7, label=f'Contact 0↔1 ({len(mesh_pair_patches[(0,1)])})'),
        Patch(facecolor='purple', alpha=0.7, label=f'Contact 0↔2 ({len(mesh_pair_patches[(0,2)])})'),
        Patch(facecolor='cyan', alpha=0.7, label=f'Contact 1↔2 ({len(mesh_pair_patches[(1,2)])})'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
