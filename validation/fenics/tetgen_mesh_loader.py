"""Load TetGen mesh files (.node and .ele) and create FEniCS meshes."""

import os
import numpy as np
from mpi4py import MPI
from dolfinx import mesh
import basix.ufl


def remap_tetgen_to_fenics_tet10(tetgen_elem):
    """Remap TetGen T10 node ordering to FEniCS/Basix ordering."""
    fenics_to_tetgen = [0, 1, 2, 3, 4, 8, 7, 5, 9, 6]
    return [tetgen_elem[i] for i in fenics_to_tetgen]


def read_tetgen_node_file(fname):
    """Read TetGen .node file and return node coordinates."""
    with open(fname, 'r') as f:
        n_nodes, dim = map(int, f.readline().split()[:2])
        x = np.zeros((n_nodes, dim))
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if parts:
                    node_id = int(parts[0]) - 1
                    x[node_id] = [float(parts[i]) for i in range(1, dim + 1)]
    return x


def read_tetgen_ele_file(fname):
    """Read TetGen .ele file and return cell connectivity."""
    with open(fname, 'r') as f:
        n_elements, nodes_per_elem = map(int, f.readline().split()[:2])
        cells = []
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if parts:
                    node_indices = [int(parts[i]) - 1 for i in range(1, nodes_per_elem + 1)]
                    cells.append(remap_tetgen_to_fenics_tet10(node_indices))
    return np.array(cells, dtype=np.int64)


def load_tetgen_mesh(mesh_name, resolution=0, mesh_dir=None):
    """Load TetGen mesh files and create a FEniCS mesh.
    
    Args:
        mesh_name: Base name of mesh files (e.g., "beam_3x2x1")
        resolution: Resolution level (0, 2, or 4). Default is 0 (RES_0).
        mesh_dir: Directory containing mesh files (default: data/meshes/T10/resolution relative to project root)
    
    Returns:
        tuple: (mesh, x_tetgen) - FEniCS mesh and original TetGen coordinates
    
    Examples:
        load_tetgen_mesh("beam_3x2x1")  # Loads beam_3x2x1_res0.1 (RES_0)
        load_tetgen_mesh("beam_3x2x1", resolution=2)  # Loads beam_3x2x1_res2.1 (RES_2)
        load_tetgen_mesh("beam_3x2x1", resolution=4)  # Loads beam_3x2x1_res4.1 (RES_4)
    """
    # Strip .1 suffix if present
    if mesh_name.endswith(".1"):
        base_name = mesh_name[:-2]
    else:
        base_name = mesh_name
    
    if mesh_dir is None:
        # Default to data/meshes/T10/resolution relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir))
        mesh_dir = os.path.join(project_root, "data", "meshes", "T10", "resolution")
    
    mesh_name = f"{base_name}_res{resolution}.1"
    node_file = os.path.join(mesh_dir, f"{mesh_name}.node")
    ele_file = os.path.join(mesh_dir, f"{mesh_name}.ele")
    
    x_tetgen = read_tetgen_node_file(node_file)
    cells = read_tetgen_ele_file(ele_file)
    element = basix.ufl.element("Lagrange", "tetrahedron", 2, shape=(3,), dtype=x_tetgen.dtype)
    msh = mesh.create_mesh(MPI.COMM_WORLD, cells, element, x_tetgen)
    
    return msh, x_tetgen

