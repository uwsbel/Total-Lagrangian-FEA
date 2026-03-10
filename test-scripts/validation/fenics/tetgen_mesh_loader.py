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


def remap_standard_to_fenics_tet10(std_elem):
    """Remap 'standard' T10 ordering (as used in C++ FEAT10 teapot/tire meshes) to FEniCS/Basix.

    Node ordering conventions:
      TetGen:   [v0,v1,v2,v3, m23,m03,m01,m12,m13,m02]
      Standard: [v0,v1,v2,v3, m01,m12,m02,m03,m13,m23]  <- teapot/tire files
      FEniCS:   [v0,v1,v2,v3, m23,m13,m12,m03,m02,m01]

    Permutation: FEniCS[i] = Standard[fenics_to_standard[i]]
    """
    fenics_to_standard = [0, 1, 2, 3, 9, 8, 5, 7, 6, 4]
    return [std_elem[i] for i in fenics_to_standard]


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


def read_tetgen_ele_file(fname, node_index_offset=0, tetgen_order=True):
    """Read TetGen .ele file and return cell connectivity.
    
    Automatically handles both 0-based and 1-based node indexing.
    
    Args:
        fname: Element file path
        node_index_offset: Offset to apply to node indices (0 for 0-based, 1 for 1-based files)
        tetgen_order: If True (default), assumes TetGen node ordering and remaps accordingly.
                      If False, assumes 'standard' ordering (as used in teapot/tire C++ meshes).
    """
    remap_fn = remap_tetgen_to_fenics_tet10 if tetgen_order else remap_standard_to_fenics_tet10
    with open(fname, 'r') as f:
        n_elements, nodes_per_elem = map(int, f.readline().split()[:2])
        cells = []
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if parts:
                    # Read all node indices for this element and apply offset
                    node_indices = [int(parts[i]) - node_index_offset for i in range(1, nodes_per_elem + 1)]
                    cells.append(remap_fn(node_indices))
    return np.array(cells, dtype=np.int64)


def load_tetgen_mesh_from_files(node_file, ele_file, tetgen_order=True):
    """Load TetGen mesh from .node and .ele files.
    
    Automatically detects and handles both 0-based and 1-based indexing in mesh files.
    Properly handles parallel MPI execution by reading files only on rank 0.
    
    Args:
        node_file: Path to .node file (absolute or relative)
        ele_file: Path to .ele file (absolute or relative)
        tetgen_order: If True (default), assumes TetGen node ordering.
                      If False, assumes 'standard' ordering (tire mesh).
    
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
        
        # Read elements using the detected offset and requested ordering
        cells = read_tetgen_ele_file(ele_file, node_index_offset=index_offset,
                                     tetgen_order=tetgen_order)
    else:
        # Other ranks receive empty arrays - DOLFINx will distribute the mesh
        x_nodes = np.empty((0, 3), dtype=np.float64)
        cells = np.empty((0, 10), dtype=np.int64)
    
    # Create DOLFINx mesh with P2 tetrahedral elements
    # DOLFINx automatically distributes the mesh across all ranks
    element = basix.ufl.element("Lagrange", "tetrahedron", 2, shape=(3,), dtype=np.float64)
    msh = mesh.create_mesh(comm, cells, element, x_nodes)
    
    return msh, x_nodes


def write_vtk_frame(domain, V, u, filename):
    """Write deformed T10 mesh as ASCII VTK (matches C++ demo format).

    Deformed position = reference coord + displacement, baked into geometry.
    VTK cell type 24 = VTK_QUADRATIC_TETRA (10-node quadratic tetrahedron).
    Safe for MPI runs: gathers mesh to rank-0 before writing.
    """
    import numpy as np

    imap = V.dofmap.index_map
    n_owned = imap.size_local
    n_local_cells = domain.topology.index_map(domain.topology.dim).size_local

    # Reference positions and displacement for owned nodes only
    dof_coords = V.tabulate_dof_coordinates()[:n_owned]          # (n_owned, 3)
    disp = u.x.array[:n_owned * 3].reshape(n_owned, 3)           # (n_owned, 3)
    deformed_local = dof_coords + disp                            # (n_owned, 3)

    # Connectivity for owned cells, converted to global node indices
    cells_local = np.asarray(V.dofmap.list).reshape(-1, 10)[:n_local_cells]
    cells_global = imap.local_to_global(cells_local.ravel()).reshape(-1, 10)

    # Gather on rank-0
    all_deformed = domain.comm.gather(deformed_local, root=0)
    all_cells    = domain.comm.gather(cells_global,   root=0)

    if domain.comm.rank == 0:
        pts  = np.vstack(all_deformed)   # (N_global_nodes, 3)
        conn = np.vstack(all_cells)      # (N_global_cells, 10)
        n_pts, n_cells = pts.shape[0], conn.shape[0]

        # Remap FEniCS/Basix → Standard/FEAT10 node ordering (matches Newton VTK output)
        fenics_to_std = np.array([0, 1, 2, 3, 9, 6, 8, 7, 5, 4])
        conn = conn[:, fenics_to_std]

        with open(filename, 'w') as out:
            out.write("# vtk DataFile Version 3.0\n")
            out.write("FEniCS T10 mesh output\n")
            out.write("ASCII\n")
            out.write("DATASET UNSTRUCTURED_GRID\n")
            out.write(f"POINTS {n_pts} double\n")
            for i in range(n_pts):
                out.write(f"{pts[i,0]} {pts[i,1]} {pts[i,2]}\n")
            out.write(f"CELLS {n_cells} {n_cells * 11}\n")
            for i in range(n_cells):
                out.write("10 " + " ".join(str(c) for c in conn[i]) + "\n")
            out.write(f"CELL_TYPES {n_cells}\n")
            out.write("24\n" * n_cells)

