#!/usr/bin/env python3
"""
Visualize contact patches from collision detection.

This script loads contact patch data exported by the C++ visualization utilities
and creates interactive 3D visualizations using matplotlib or exports to formats
compatible with ParaView.

Usage:
    python visualize_contact_patches.py --csv output/contact_patches.csv
    python visualize_contact_patches.py --json output/contact_patches.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd


def load_csv_patches(filename):
    """Load contact patches from CSV file."""
    df = pd.read_csv(filename)
    patches = []
    for _, row in df.iterrows():
        patch = {
            'tetA_idx': int(row['tetA_idx']),
            'tetB_idx': int(row['tetB_idx']),
            'centroid': np.array([row['centroid_x'], row['centroid_y'], row['centroid_z']]),
            'normal': np.array([row['normal_x'], row['normal_y'], row['normal_z']]),
            'area': row['area'],
            'g_A': row['g_A'],
            'g_B': row['g_B'],
            'p_equilibrium': row['p_equilibrium'],
            'num_vertices': int(row['num_vertices']),
            'valid_orientation': bool(row['valid_orientation'])
        }
        patches.append(patch)
    return patches


def load_json_patches(filename):
    """Load contact patches from JSON file."""
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


def visualize_patches_3d(patches, show_normals=True, color_by='area'):
    """
    Create 3D visualization of contact patches.
    
    Args:
        patches: List of patch dictionaries (from JSON, which includes vertices)
        show_normals: Whether to draw normal vectors
        color_by: 'area', 'g_A', 'g_B', or 'p_equilibrium'
    """
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
    else:
        values = [p['p_equilibrium'] for p in patches]
        cbar_label = 'Equilibrium Pressure'
    
    # Normalize colors
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-10:
        vmax = vmin + 1e-10
    norm_values = [(v - vmin) / (vmax - vmin) for v in values]
    
    # Create colormap
    cmap = plt.cm.viridis
    
    # Draw patches as polygons
    for i, patch in enumerate(patches):
        if 'vertices' in patch:
            verts = [patch['vertices']]
            color = cmap(norm_values[i])
            poly = Poly3DCollection(verts, alpha=0.7, facecolor=color, 
                                   edgecolor='black', linewidth=0.5)
            ax.add_collection3d(poly)
        
        # Draw normal vector
        if show_normals:
            c = patch['centroid']
            n = patch['normal']
            arrow_scale = 0.02
            ax.quiver(c[0], c[1], c[2], 
                     n[0]*arrow_scale, n[1]*arrow_scale, n[2]*arrow_scale,
                     color='red', arrow_length_ratio=0.3, linewidth=1)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Contact Patches ({len(patches)} patches)\nColored by {color_by}')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label(cbar_label)
    
    # Set equal aspect ratio
    centroids = np.array([p['centroid'] for p in patches])
    if len(centroids) > 0:
        max_range = np.max(centroids.max(axis=0) - centroids.min(axis=0)) / 2
        mid = centroids.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    plt.tight_layout()
    return fig, ax


def plot_patch_statistics(patches):
    """Create statistical plots of contact patch data."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    areas = [p['area'] for p in patches]
    g_A = [p['g_A'] for p in patches]
    g_B = [p['g_B'] for p in patches]
    p_eq = [p['p_equilibrium'] for p in patches]
    
    # Area histogram
    axes[0, 0].hist(areas, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Contact Patch Area')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Patch Areas')
    axes[0, 0].axvline(np.mean(areas), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(areas):.2e}')
    axes[0, 0].legend()
    
    # g_A vs g_B scatter
    axes[0, 1].scatter(g_A, g_B, alpha=0.5, c=areas, cmap='viridis')
    axes[0, 1].set_xlabel('g_A')
    axes[0, 1].set_ylabel('g_B')
    axes[0, 1].set_title('Pressure Gradients (colored by area)')
    axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Equilibrium pressure histogram
    axes[1, 0].hist(p_eq, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Equilibrium Pressure')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Equilibrium Pressure')
    
    # Total area vs patch count
    valid_orient = [p for p in patches if p.get('valid_orientation', True)]
    labels = ['Valid Orientation', 'Invalid Orientation']
    sizes = [len(valid_orient), len(patches) - len(valid_orient)]
    colors = ['#2ecc71', '#e74c3c']
    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90)
    axes[1, 1].set_title('Patch Orientation Validity')
    
    plt.tight_layout()
    return fig


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
    print(f"\nArea statistics:")
    print(f"  Min:    {np.min(areas):.6e}")
    print(f"  Max:    {np.max(areas):.6e}")
    print(f"  Mean:   {np.mean(areas):.6e}")
    print(f"  Std:    {np.std(areas):.6e}")
    print(f"  Total:  {np.sum(areas):.6e}")
    
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
    
    valid_count = sum(1 for p in patches if p.get('valid_orientation', True))
    print(f"\nOrientation validity:")
    print(f"  Valid:   {valid_count} ({100*valid_count/len(patches):.1f}%)")
    print(f"  Invalid: {len(patches)-valid_count} ({100*(len(patches)-valid_count)/len(patches):.1f}%)")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize contact patches from collision detection')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--json', type=str, help='Path to JSON file')
    parser.add_argument('--color-by', type=str, default='area',
                        choices=['area', 'g_A', 'g_B', 'p_equilibrium'],
                        help='Property to color patches by')
    parser.add_argument('--no-normals', action='store_true',
                        help='Do not show normal vectors')
    parser.add_argument('--save', type=str, help='Save figure to file')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print statistics, no visualization')
    args = parser.parse_args()
    
    # Load patches
    if args.json:
        patches = load_json_patches(args.json)
        has_vertices = True
    elif args.csv:
        patches = load_csv_patches(args.csv)
        has_vertices = False
    else:
        print("Error: Please specify either --csv or --json file")
        return
    
    print(f"Loaded {len(patches)} contact patches")
    
    # Print summary
    print_summary(patches)
    
    if args.stats_only:
        return
    
    # Create visualizations
    if has_vertices:
        fig1, ax1 = visualize_patches_3d(patches, 
                                         show_normals=not args.no_normals,
                                         color_by=args.color_by)
        if args.save:
            fig1.savefig(args.save.replace('.', '_3d.'), dpi=150, bbox_inches='tight')
            print(f"Saved 3D visualization to {args.save.replace('.', '_3d.')}")
    else:
        print("Note: CSV does not contain vertex data. 3D polygon visualization unavailable.")
        print("Use JSON export for full polygon visualization.")
    
    # Plot statistics
    fig2 = plot_patch_statistics(patches)
    if args.save:
        stats_file = args.save.replace('.png', '_stats.png').replace('.pdf', '_stats.pdf')
        fig2.savefig(stats_file, dpi=150, bbox_inches='tight')
        print(f"Saved statistics plot to {stats_file}")
    
    plt.show()


if __name__ == '__main__':
    main()
