"""Utilities for loading TetGen T10 meshes and locating tracked raw node ids."""

from __future__ import annotations

import os

import basix.ufl
import numpy as np
from dolfinx import mesh
from mpi4py import MPI


def remap_tetgen_to_fenics_tet10(tetgen_elem):
    """Remap TetGen T10 node ordering to FEniCS/Basix ordering."""
    fenics_to_tetgen = [0, 1, 2, 3, 4, 8, 7, 5, 9, 6]
    return [tetgen_elem[i] for i in fenics_to_tetgen]


def remap_standard_to_fenics_tet10(std_elem):
    """Remap an alternate 'standard' T10 ordering to FEniCS/Basix."""
    fenics_to_standard = [0, 1, 2, 3, 9, 8, 5, 7, 6, 4]
    return [std_elem[i] for i in fenics_to_standard]


def read_tetgen_node_file(fname, return_offset=False):
    """Read TetGen .node file and return node coordinates."""
    with open(fname, "r") as f:
        n_nodes, dim = map(int, f.readline().split()[:2])
        x = np.zeros((n_nodes, dim))
        index_offset = None
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                if parts:
                    node_id_raw = int(parts[0])
                    if index_offset is None:
                        index_offset = node_id_raw
                    node_id = node_id_raw - index_offset
                    x[node_id] = [float(parts[i]) for i in range(1, dim + 1)]
    if return_offset:
        return x, index_offset
    return x


def read_tetgen_ele_file(fname, node_index_offset=0, tetgen_order=True):
    """Read TetGen .ele file and return cell connectivity."""
    remap_fn = remap_tetgen_to_fenics_tet10 if tetgen_order else remap_standard_to_fenics_tet10
    with open(fname, "r") as f:
        n_elements, nodes_per_elem = map(int, f.readline().split()[:2])
        cells = []
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                if parts:
                    node_indices = [
                        int(parts[i]) - node_index_offset for i in range(1, nodes_per_elem + 1)
                    ]
                    cells.append(remap_fn(node_indices))
    return np.array(cells, dtype=np.int64)


def load_tetgen_mesh_from_files(node_file, ele_file, tetgen_order=True):
    """Load TetGen mesh from .node and .ele files."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        if not os.path.exists(node_file):
            raise FileNotFoundError(f"Node file not found: {node_file}")
        if not os.path.exists(ele_file):
            raise FileNotFoundError(f"Element file not found: {ele_file}")

        x_nodes, index_offset = read_tetgen_node_file(node_file, return_offset=True)
        cells = read_tetgen_ele_file(
            ele_file, node_index_offset=index_offset, tetgen_order=tetgen_order
        )
    else:
        x_nodes = np.empty((0, 3), dtype=np.float64)
        cells = np.empty((0, 10), dtype=np.int64)

    element = basix.ufl.element("Lagrange", "tetrahedron", 2, shape=(3,), dtype=np.float64)
    msh = mesh.create_mesh(comm, cells, element, x_nodes)
    return msh, x_nodes


def get_raw_node_coordinate(comm, node_file, raw_node_id):
    """Broadcast the exact mesh-node coordinate for a raw .node id."""
    if comm.rank == 0:
        x_nodes, index_offset = read_tetgen_node_file(node_file, return_offset=True)
        node_idx = raw_node_id - index_offset
        if node_idx < 0 or node_idx >= len(x_nodes):
            raise RuntimeError(
                f"Raw node id {raw_node_id} is out of range for {node_file} "
                f"(detected offset {index_offset})."
            )
        coord = x_nodes[node_idx].copy()
    else:
        coord = None
    return np.array(comm.bcast(coord, root=0))


def find_owned_dof_by_coord(comm, rank, dof_coords, num_owned, target, tol=1e-10):
    """Find the unique owned DOF that matches a mesh-node coordinate."""
    local_candidate = None
    for i, coord in enumerate(dof_coords[:num_owned]):
        if np.allclose(coord, target, atol=tol, rtol=0.0):
            local_candidate = {
                "rank": rank,
                "dof": int(i),
                "coord": [float(v) for v in coord.copy()],
            }
            break

    candidates = comm.gather(local_candidate, root=0)
    chosen = None
    if rank == 0:
        valid = [c for c in candidates if c is not None]
        if len(valid) != 1:
            raise RuntimeError(
                f"Expected exactly one tracked-node owner for coordinate {target}, "
                f"found {len(valid)}."
            )
        chosen = valid[0]
    chosen = comm.bcast(chosen, root=0)
    if rank == chosen["rank"]:
        return chosen["dof"], np.array(chosen["coord"]), chosen["rank"]
    return None, np.array(chosen["coord"]), chosen["rank"]


def locate_raw_node_dof(domain, rank, dof_coords, num_owned, node_file, raw_node_id, tol=1e-10):
    """Resolve a raw TetGen node id to its owned FEniCS DOF and coordinates."""
    target = get_raw_node_coordinate(domain.comm, node_file, raw_node_id)
    dof, coord, owner = find_owned_dof_by_coord(
        domain.comm, rank, dof_coords, num_owned, target, tol=tol
    )
    return target, dof, coord, owner


def write_vtk_frame(domain, V, u, filename):
    """Write deformed T10 mesh as ASCII VTK."""
    imap = V.dofmap.index_map
    n_owned = imap.size_local
    n_local_cells = domain.topology.index_map(domain.topology.dim).size_local

    dof_coords = V.tabulate_dof_coordinates()[:n_owned]
    disp = u.x.array[: n_owned * 3].reshape(n_owned, 3)
    deformed_local = dof_coords + disp

    cells_local = np.asarray(V.dofmap.list).reshape(-1, 10)[:n_local_cells]
    cells_global = imap.local_to_global(cells_local.ravel()).reshape(-1, 10)

    all_deformed = domain.comm.gather(deformed_local, root=0)
    all_cells = domain.comm.gather(cells_global, root=0)

    if domain.comm.rank == 0:
        pts = np.vstack(all_deformed)
        conn = np.vstack(all_cells)
        n_pts, n_cells = pts.shape[0], conn.shape[0]

        fenics_to_std = np.array([0, 1, 2, 3, 9, 6, 8, 7, 5, 4])
        conn = conn[:, fenics_to_std]

        with open(filename, "w") as out:
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
