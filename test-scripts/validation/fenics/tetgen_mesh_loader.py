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


def read_tetgen_node_file(fname, return_offset=False):
    """Read TetGen .node file and return node coordinates.
    
    Automatically detects and handles both 0-based and 1-based indexing.
    
    Args:
        fname: Node file path
        return_offset: If True, also returns the detected index offset (0 or 1)
    
    Returns:
        x: Node coordinates array
        offset: (optional) Detected index offset (0 for 0-based, 1 for 1-based files)
    """
    with open(fname, 'r') as f:
        n_nodes, dim = map(int, f.readline().split()[:2])
        x = np.zeros((n_nodes, dim))
        index_offset = None
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if parts:
                    node_id_raw = int(parts[0])
                    # Auto-detect indexing on first data line
                    if index_offset is None:
                        index_offset = node_id_raw  # 0 for 0-based, 1 for 1-based
                    node_id = node_id_raw - index_offset
                    x[node_id] = [float(parts[i]) for i in range(1, dim + 1)]
    if return_offset:
        return x, index_offset
    return x


def read_tetgen_ele_file(fname, node_index_offset=0):
    """Read TetGen .ele file and return cell connectivity.
    
    Automatically handles both 0-based and 1-based node indexing.
    
    Args:
        fname: Element file path
        node_index_offset: Offset to apply to node indices (0 for 0-based, 1 for 1-based files)
    """
    with open(fname, 'r') as f:
        n_elements, nodes_per_elem = map(int, f.readline().split()[:2])
        cells = []
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if parts:
                    # Read all node indices for this element and apply offset
                    node_indices = [int(parts[i]) - node_index_offset for i in range(1, nodes_per_elem + 1)]
                    cells.append(remap_tetgen_to_fenics_tet10(node_indices))
    return np.array(cells, dtype=np.int64)


def load_tetgen_mesh_from_files(node_file, ele_file):
    """Load TetGen mesh from .node and .ele files.
    
    Automatically detects and handles both 0-based and 1-based indexing in mesh files.
    Properly handles parallel MPI execution by reading files only on rank 0.
    
    Args:
        node_file: Path to .node file (absolute or relative)
        ele_file: Path to .ele file (absolute or relative)
    
    Returns:
        tuple: (mesh, x_nodes) - DOLFINx mesh and node coordinate array
    
    Raises:
        FileNotFoundError: If either the node file or element file does not exist
    
    Examples:
        load_tetgen_mesh_from_files(
            "data/meshes/T10/bunny_ascii_26.1.node",
            "data/meshes/T10/bunny_ascii_26.1.ele"
        )
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Only rank 0 reads files and validates they exist
    if rank == 0:
        if not os.path.exists(node_file):
            raise FileNotFoundError(f"Node file not found: {node_file}")
        if not os.path.exists(ele_file):
            raise FileNotFoundError(f"Element file not found: {ele_file}")
        
        # Read nodes and detect indexing offset (0 or 1)
        x_nodes, index_offset = read_tetgen_node_file(node_file, return_offset=True)
        
        # Read elements using the detected offset
        cells = read_tetgen_ele_file(ele_file, node_index_offset=index_offset)
    else:
        # Other ranks receive empty arrays - DOLFINx will distribute the mesh
        x_nodes = np.empty((0, 3), dtype=np.float64)
        cells = np.empty((0, 10), dtype=np.int64)
    
    # Create DOLFINx mesh with P2 tetrahedral elements
    # DOLFINx automatically distributes the mesh across all ranks
    element = basix.ufl.element("Lagrange", "tetrahedron", 2, shape=(3,), dtype=np.float64)
    msh = mesh.create_mesh(comm, cells, element, x_nodes)
    
    return msh, x_nodes

