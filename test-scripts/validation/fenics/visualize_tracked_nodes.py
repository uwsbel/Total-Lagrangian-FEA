"""
CLI viewer for validation meshes and calculated candidate tracked nodes.

Usage (from repository root):
  conda run -n fenicsx python test-scripts/validation/fenics/visualize_tracked_nodes.py
  conda run -n fenicsx python test-scripts/validation/fenics/visualize_tracked_nodes.py --mesh teapot --res 0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pyvista as pv

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "meshes", "T10")

MESH_LABELS = {
    "bunny_ascii": "Bunny ascii",
    "bunny_scaling": "Bunny scaling",
    "beam": "Beam",
    "teapot": "Teapot",
    "tire": "Tire",
}

MESH_RES_OPTIONS = {
    "bunny_ascii": [None],
    "bunny_scaling": [0, 2, 4, 8, 16],
    "beam": [0, 2, 4, 8, 16, 32],
    "teapot": [0, 2, 4, 8, 16],
    "tire": [0, 2, 4, 8, 16],
}


@dataclass
class Marker:
    idx: int
    label: str
    color: str
    radius: float


@dataclass
class Scene:
    title: str
    subtitle: str
    coords: np.ndarray
    grid: pv.UnstructuredGrid
    markers: list[Marker]
    details: list[str]


def read_node_file(fname):
    """Read TetGen .node file, return 0-based coords array."""
    with open(fname) as f:
        n_nodes, dim = map(int, f.readline().split()[:2])
        coords = np.zeros((n_nodes, dim))
        offset = None
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            idx = int(parts[0])
            if offset is None:
                offset = idx
            coords[idx - offset] = [float(parts[i]) for i in range(1, dim + 1)]
    return coords


def read_ele_file_for_vtk(fname):
    """Read TetGen .ele file and return 0-based T10 connectivity in VTK ordering."""
    with open(fname) as f:
        header = f.readline().split()
        _, nodes_per = int(header[0]), int(header[1])
        raw = []
        offset = None
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            if offset is None:
                offset = int(parts[0])
            ids = [int(p) - offset for p in parts[1 : nodes_per + 1]]
            raw.append(ids)
    raw = np.array(raw)

    node_fname = fname.rsplit(".", 1)[0] + ".node"
    if not os.path.exists(node_fname):
        node_fname = fname[: fname.rfind(".")] + ".node"
    coords = read_node_file(node_fname)

    tet_edges = [(2, 3), (0, 3), (0, 1), (1, 2), (1, 3), (0, 2)]
    std_edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]

    n_check = min(len(raw), 200)
    err_tet = 0.0
    err_std = 0.0
    count = 0
    for ei in range(n_check):
        v = raw[ei, :4]
        m = raw[ei, 4:]
        if np.any(v < 0) or np.any(v >= len(coords)):
            continue
        if np.any(m < 0) or np.any(m >= len(coords)):
            continue
        for j in range(6):
            mid_coord = coords[m[j]]
            tet_mid = 0.5 * (coords[v[tet_edges[j][0]]] + coords[v[tet_edges[j][1]]])
            std_mid = 0.5 * (coords[v[std_edges[j][0]]] + coords[v[std_edges[j][1]]])
            err_tet += np.linalg.norm(mid_coord - tet_mid)
            err_std += np.linalg.norm(mid_coord - std_mid)
            count += 1

    is_tetgen = err_tet <= err_std if count > 0 else True
    if is_tetgen:
        vtk_order = [0, 1, 2, 3, 6, 7, 9, 5, 8, 4]
        elems = raw[:, vtk_order]
    else:
        elems = raw

    return elems, is_tetgen


def make_grid(coords, elems):
    """Build pyvista UnstructuredGrid from T10 coords and VTK-ordered elements."""
    n_elem = elems.shape[0]
    cells = np.hstack([np.full((n_elem, 1), 10, dtype=int), elems]).ravel()
    cell_type = np.full(n_elem, 24, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_type, np.array(coords, dtype=float))


def add_corner_csys(plotter):
    """Add a small orientation widget in the lower-left corner."""
    plotter.add_axes(viewport=(0.0, 0.0, 0.18, 0.18))


def mesh_base(mesh_name: str, res: int | None) -> str:
    if mesh_name == "bunny_ascii":
        return os.path.join(DATA, "bunny_ascii_26.1")
    if mesh_name == "bunny_scaling":
        return os.path.join(DATA, "bunny_scaling", f"bunny_res{res}.1")
    if mesh_name == "beam":
        return os.path.join(DATA, "resolution", f"beam_3x2x1_res{res}.1")
    if mesh_name == "teapot":
        return os.path.join(DATA, "teapot_scaling", f"teapot_res{res}.1")
    if mesh_name == "tire":
        return os.path.join(DATA, "tire_scaling", f"tire_res{res}.1")
    raise ValueError(f"Unknown mesh: {mesh_name}")


def default_mesh_views() -> list[tuple[str, int | None]]:
    return [
        ("bunny_ascii", None),
        ("bunny_scaling", 0),
        ("beam", 0),
        ("teapot", 0),
        ("tire", 0),
    ]


def vertex_indices(elems: np.ndarray) -> np.ndarray:
    return np.array(sorted(set(np.asarray(elems)[:, :4].ravel())), dtype=int)


def nearest_vertex(candidates: np.ndarray, coords: np.ndarray, target: np.ndarray) -> int:
    candidate_coords = coords[candidates]
    return int(candidates[np.argmin(np.linalg.norm(candidate_coords - target, axis=1))])


def build_scene(mesh_name: str, res: int | None) -> Scene:
    base = mesh_base(mesh_name, res)
    coords = read_node_file(base + ".node")
    elems, is_tetgen = read_ele_file_for_vtk(base + ".ele")
    grid = make_grid(coords, elems)

    z = coords[:, 2]
    x = coords[:, 0]
    vertices = vertex_indices(elems)
    markers: list[Marker] = []

    if mesh_name == "bunny_ascii":
        force_mask = z > 4.0
        fixed_mask = z < -4.0
        right_verts = vertices[(z[vertices] > 6.0) & (x[vertices] > 0.0)]
        left_verts = vertices[(z[vertices] > 6.0) & (x[vertices] < 0.0)]
        i_right_ear = int(right_verts[np.argmax(z[right_verts])])
        i_left_ear = int(left_verts[np.argmax(z[left_verts])])
        markers = [
            Marker(i_right_ear, f"Right ear #{i_right_ear}", "red", 0.15),
            Marker(i_left_ear, f"Left ear #{i_left_ear}", "blue", 0.15),
        ]
        title = "Bunny ascii — Calculated Candidate Nodes"
        subtitle = "Blue=fixed (z<-4)  Gray=free  Salmon=force (z>4)"
    elif mesh_name == "bunny_scaling":
        force_mask = z > 0.4
        fixed_mask = z < -0.4
        right_verts = vertices[(z[vertices] > 0.6) & (x[vertices] > 0.05)]
        left_verts = vertices[(z[vertices] > 0.6) & (x[vertices] < -0.1)]
        i_right_ear = int(right_verts[np.argmax(z[right_verts])])
        i_left_ear = int(left_verts[np.argmax(z[left_verts])])
        markers = [
            Marker(i_right_ear, f"Right ear #{i_right_ear}", "red", 0.02),
            Marker(i_left_ear, f"Left ear #{i_left_ear}", "blue", 0.02),
        ]
        title = f"Bunny res{res} — Calculated Candidate Nodes"
        subtitle = "Blue=fixed (z<-0.4)  Gray=free  Salmon=force (z>0.4)"
    elif mesh_name == "beam":
        force_mask = np.isclose(x, x.max(), atol=1e-8)
        fixed_mask = np.isclose(x, x.min(), atol=1e-8)
        vcoords = coords[vertices]
        target = np.array([coords[:, 0].max(), coords[:, 1].max(), coords[:, 2].max()])
        i_tracked = int(vertices[np.argmin(np.linalg.norm(vcoords - target, axis=1))])
        markers = [Marker(i_tracked, f"Tracked corner #{i_tracked}", "red", 0.04)]
        title = f"Beam res{res} — Calculated Candidate Node"
        subtitle = "Blue=fixed (x=0)  Gray=free  Salmon=force (x=max)"
    elif mesh_name == "teapot":
        z_min, z_max = z.min(), z.max()
        z_rng = z_max - z_min
        z_fix = z_min + 0.2 * z_rng
        z_top = z_min + 0.8 * z_rng
        force_mask = z >= z_top
        fixed_mask = z <= z_fix
        x_max = x.max()
        x_rng = x.max() - x.min()
        spout_verts = vertices[x[vertices] >= x_max - 0.08 * x_rng]
        spout_target = np.array([x_max, 0.0, z[spout_verts].max()])
        lid_target = np.array([0.0, 0.0, z_max])
        i_spout = nearest_vertex(spout_verts, coords, spout_target)
        i_lid = nearest_vertex(vertices, coords, lid_target)
        markers = [
            Marker(i_spout, f"Spout tip #{i_spout}", "red", 0.01),
            Marker(i_lid, f"Lid knob #{i_lid}", "blue", 0.01),
        ]
        title = f"Teapot res{res} — Calculated Candidate Nodes"
        subtitle = f"Blue=fixed (z<={z_fix:.3f})  Gray=free  Salmon=force (z>={z_top:.3f})"
    elif mesh_name == "tire":
        z_min, z_max = z.min(), z.max()
        z_rng = z_max - z_min
        z_fix = z_min + 0.1 * z_rng
        z_top = z_min + 0.9 * z_rng
        force_mask = z >= z_top
        fixed_mask = z <= z_fix
        i_crown = int(vertices[np.argmax(z[vertices])])
        markers = [Marker(i_crown, f"Crown #{i_crown}", "red", 0.02)]
        title = f"Tire res{res} — Calculated Candidate Node"
        subtitle = f"Blue=fixed (z<={z_fix:.3f})  Gray=free  Salmon=force (z>={z_top:.3f})"
    else:
        raise ValueError(f"Unsupported mesh: {mesh_name}")

    region = np.ones(len(coords))
    region[fixed_mask] = 0
    region[force_mask] = 2
    grid.point_data["region"] = region

    details = [
        f"mesh={mesh_name}" + ("" if res is None else f" res={res}"),
        f"nodes={len(coords)} elems={len(elems)} order={'TetGen' if is_tetgen else 'Standard'}",
    ]
    for marker in markers:
        c = coords[marker.idx]
        details.append(
            f"{marker.label}: ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})"
        )

    return Scene(
        title=title,
        subtitle=subtitle,
        coords=coords,
        grid=grid,
        markers=markers,
        details=details,
    )


def render_scene(plotter, scene: Scene, show_regions: bool, show_edges: bool, show_nodes: bool):
    plotter.clear()
    plotter.set_background("white")

    surface = scene.grid.extract_surface()
    mesh_kwargs = {
        "opacity": 0.4,
        "show_scalar_bar": False,
        "show_edges": show_edges,
        "edge_color": "black",
        "line_width": 0.6,
    }
    if show_regions:
        plotter.add_mesh(
            surface,
            scalars="region",
            cmap=["steelblue", "lightgray", "salmon"],
            **mesh_kwargs,
        )
    else:
        plotter.add_mesh(surface, color="lightgray", **mesh_kwargs)

    if show_nodes:
        for marker in scene.markers:
            plotter.add_mesh(
                pv.Sphere(radius=marker.radius, center=scene.coords[marker.idx]),
                color=marker.color,
                label=marker.label,
            )
        if scene.markers:
            plotter.add_legend(bcolor="white", face="circle", loc="upper left")

    plotter.add_title(scene.title, font_size=14)
    plotter.add_text(scene.subtitle, position="lower_left", font_size=10)
    add_corner_csys(plotter)
    plotter.camera_position = "xz"


def show_static_scene(mesh_name: str, res: int | None, show_regions: bool, show_edges: bool, show_nodes: bool):
    scene = build_scene(mesh_name, res)
    for line in scene.details:
        print(line)
    plotter = pv.Plotter()
    render_scene(plotter, scene, show_regions, show_edges, show_nodes)
    plotter.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh",
        choices=list(MESH_LABELS),
        help="Mesh family to display. If omitted, show all mesh families one by one.",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=0,
        help="Resolution for res-based mesh families.",
    )
    parser.add_argument("--hide-regions", action="store_true")
    parser.add_argument("--hide-edges", action="store_true")
    parser.add_argument("--hide-nodes", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mesh_views = (
        default_mesh_views()
        if args.mesh is None
        else [(args.mesh, None if args.mesh == "bunny_ascii" else args.res)]
    )
    for mesh_name, res in mesh_views:
        show_static_scene(
            mesh_name,
            res,
            show_regions=not args.hide_regions,
            show_edges=not args.hide_edges,
            show_nodes=not args.hide_nodes,
        )
