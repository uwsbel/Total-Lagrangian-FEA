#!/usr/bin/env python3
"""
Generate a simple ANCF3243 "net" (orthogonal beam grid) mesh with per-crossing
constraints in a small, line-based format.

Why duplicate nodes at crossings?
- "Pinned" joints require *position-only* continuity between strands, so each
  strand must keep its own nodal gradients (r_u, r_v, r_w). That means we store
  two nodes per grid point: one for the horizontal strand family (H), one for
  the vertical family (V), and tie them with constraints.

File format (.ancf3243mesh), v1 (line-based, comments start with '#'):

  ancf3243_mesh 1
  grid nx <nx> ny <ny> L <L> origin <ox> <oy> <oz>

  nodes <N>
  # id family x0 x1 x2 x3 y0 y1 y2 y3 z0 z1 z2 z3
  <id> <H|V> <12 floats...>

  elements <M>
  # id family n0 n1
  <id> <H|V> <node0> <node1>

  constraints <K>
  # pinned a b
  # welded a b q00 q01 q02 q10 q11 q12 q20 q21 q22
  <...>

Constraint semantics (for later consumption):
- pinned a b:
    r(b) = r(a)
- welded a b Q (Q row-major):
    r(b)     = r(a)
    r_u(b)   = Q r_u(a)
    r_v(b)   = Q r_v(a)
    r_w(b)   = Q r_w(a)

For this generator:
- H nodes use basis: r_u=(1,0,0), r_v=(0,1,0), r_w=(0,0,1)
- V nodes use basis: r_u=(0,1,0), r_v=(-1,0,0), r_w=(0,0,1)
  which corresponds to Q = Rz(+90deg) mapping H -> V.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]


@dataclass(frozen=True)
class Node:
    node_id: int
    family: str  # "H" or "V"
    x12: Tuple[float, float, float, float]
    y12: Tuple[float, float, float, float]
    z12: Tuple[float, float, float, float]


@dataclass(frozen=True)
class Element:
    elem_id: int
    family: str  # "H" or "V"
    n0: int
    n1: int


@dataclass(frozen=True)
class Constraint:
    kind: str  # "pinned" | "welded"
    a: int
    b: int
    Q: Mat3 | None = None  # only for welded


def _matmul33(A: Mat3, B: Mat3) -> Mat3:
    out: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = sum(A[i][k] * B[k][j] for k in range(3))
    return (tuple(out[0]), tuple(out[1]), tuple(out[2]))  # type: ignore[return-value]


def _transpose33(A: Mat3) -> Mat3:
    return (
        (A[0][0], A[1][0], A[2][0]),
        (A[0][1], A[1][1], A[2][1]),
        (A[0][2], A[1][2], A[2][2]),
    )


def _basis_matrix(ru: Vec3, rv: Vec3, rw: Vec3) -> Mat3:
    # Columns are the basis vectors in global coordinates.
    return (
        (ru[0], rv[0], rw[0]),
        (ru[1], rv[1], rw[1]),
        (ru[2], rv[2], rw[2]),
    )


def _rotation_from_bases(R_from: Mat3, R_to: Mat3) -> Mat3:
    # Map vectors expressed in "from" frame to "to" frame at the joint:
    # Q = R_to * R_from^T
    return _matmul33(R_to, _transpose33(R_from))


def _node_dofs(position: Vec3, ru: Vec3, rv: Vec3, rw: Vec3) -> Tuple[Tuple[float, float, float, float], ...]:
    x0, y0, z0 = position
    x12 = (x0, ru[0], rv[0], rw[0])
    y12 = (y0, ru[1], rv[1], rw[1])
    z12 = (z0, ru[2], rv[2], rw[2])
    return x12, y12, z12


def _point_id(i: int, j: int, nx: int) -> int:
    return j * (nx + 1) + i


def _node_id(i: int, j: int, nx: int, family: str) -> int:
    pid = _point_id(i, j, nx)
    return 2 * pid + (0 if family == "H" else 1)


def generate_net(nx: int, ny: int, L: float, origin: Vec3, joint: str) -> Tuple[List[Node], List[Element], List[Constraint]]:
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1 for a net.")
    if not (L > 0.0) or math.isinf(L) or math.isnan(L):
        raise ValueError("L must be a finite positive number.")
    if joint not in ("pinned", "welded"):
        raise ValueError("joint must be 'pinned' or 'welded'.")

    ox, oy, oz = origin

    # Strand-local bases (in global coordinates).
    ru_h: Vec3 = (1.0, 0.0, 0.0)
    rv_h: Vec3 = (0.0, 1.0, 0.0)
    rw_h: Vec3 = (0.0, 0.0, 1.0)

    ru_v: Vec3 = (0.0, 1.0, 0.0)
    rv_v: Vec3 = (-1.0, 0.0, 0.0)
    rw_v: Vec3 = (0.0, 0.0, 1.0)

    R_h = _basis_matrix(ru_h, rv_h, rw_h)
    R_v = _basis_matrix(ru_v, rv_v, rw_v)
    Q_h_to_v = _rotation_from_bases(R_h, R_v)

    nodes: List[Node] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = ox + i * L
            y = oy + j * L
            z = oz

            nid_h = _node_id(i, j, nx, "H")
            x12_h, y12_h, z12_h = _node_dofs((x, y, z), ru_h, rv_h, rw_h)
            nodes.append(Node(nid_h, "H", x12_h, y12_h, z12_h))

            nid_v = _node_id(i, j, nx, "V")
            x12_v, y12_v, z12_v = _node_dofs((x, y, z), ru_v, rv_v, rw_v)
            nodes.append(Node(nid_v, "V", x12_v, y12_v, z12_v))

    nodes.sort(key=lambda n: n.node_id)

    elements: List[Element] = []
    eid = 0
    # Horizontal elements (H family): along +x for each row j.
    for j in range(ny + 1):
        for i in range(nx):
            n0 = _node_id(i, j, nx, "H")
            n1 = _node_id(i + 1, j, nx, "H")
            elements.append(Element(eid, "H", n0, n1))
            eid += 1
    # Vertical elements (V family): along +y for each column i.
    for i in range(nx + 1):
        for j in range(ny):
            n0 = _node_id(i, j, nx, "V")
            n1 = _node_id(i, j + 1, nx, "V")
            elements.append(Element(eid, "V", n0, n1))
            eid += 1

    constraints: List[Constraint] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            a = _node_id(i, j, nx, "H")
            b = _node_id(i, j, nx, "V")
            if joint == "pinned":
                constraints.append(Constraint("pinned", a, b, None))
            else:
                constraints.append(Constraint("welded", a, b, Q_h_to_v))

    return nodes, elements, constraints


def _fmt_f(x: float) -> str:
    # Deterministic, parse-friendly formatting.
    return f"{x:.17g}"


def write_ancf3243mesh(path: Path, nx: int, ny: int, L: float, origin: Vec3,
                       nodes: Sequence[Node], elements: Sequence[Element], constraints: Sequence[Constraint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("ancf3243_mesh 1\n")
        f.write(f"grid nx {nx} ny {ny} L {_fmt_f(L)} origin {_fmt_f(origin[0])} {_fmt_f(origin[1])} {_fmt_f(origin[2])}\n")
        f.write("\n")

        f.write(f"nodes {len(nodes)}\n")
        f.write("# id family x0 x1 x2 x3 y0 y1 y2 y3 z0 z1 z2 z3\n")
        for n in nodes:
            parts: List[str] = [str(n.node_id), n.family]
            parts += [_fmt_f(v) for v in (*n.x12, *n.y12, *n.z12)]
            f.write(" ".join(parts) + "\n")
        f.write("\n")

        f.write(f"elements {len(elements)}\n")
        f.write("# id family n0 n1\n")
        for e in elements:
            f.write(f"{e.elem_id} {e.family} {e.n0} {e.n1}\n")
        f.write("\n")

        f.write(f"constraints {len(constraints)}\n")
        f.write("# pinned a b\n")
        f.write("# welded a b q00 q01 q02 q10 q11 q12 q20 q21 q22\n")
        for c in constraints:
            if c.kind == "pinned":
                f.write(f"pinned {c.a} {c.b}\n")
            elif c.kind == "welded":
                if c.Q is None:
                    raise ValueError("welded constraint missing Q")
                q = c.Q
                qflat = (q[0][0], q[0][1], q[0][2], q[1][0], q[1][1], q[1][2], q[2][0], q[2][1], q[2][2])
                f.write("welded " + " ".join([str(c.a), str(c.b)] + [_fmt_f(v) for v in qflat]) + "\n")
            else:
                raise ValueError(f"unknown constraint kind: {c.kind}")


def _default_out_path(script_dir: Path, nx: int, ny: int, L: float, joint: str) -> Path:
    name = f"net_{joint}_nx{nx}_ny{ny}_L{_fmt_f(L)}.ancf3243mesh"
    return script_dir / name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an ANCF3243 net mesh (.ancf3243mesh).")
    parser.add_argument("--nx", type=int, default=10, help="Number of elements along x per strand (>=1).")
    parser.add_argument("--ny", type=int, default=10, help="Number of elements along y per strand (>=1).")
    parser.add_argument("--L", type=float, default=0.5, help="Element length / grid spacing (meters).")
    parser.add_argument("--origin", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("OX", "OY", "OZ"),
                        help="Origin of the net (meters).")
    parser.add_argument("--joint", choices=("pinned", "welded"), default="pinned",
                        help="Joint model at crossings.")
    parser.add_argument("--out", type=str, default="",
                        help="Output path (default: next to this script).")

    args = parser.parse_args(argv)

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    out_path = Path(args.out) if args.out else _default_out_path(script_dir, args.nx, args.ny, args.L, args.joint)

    nodes, elements, constraints = generate_net(args.nx, args.ny, args.L, tuple(args.origin), args.joint)
    write_ancf3243mesh(out_path, args.nx, args.ny, args.L, tuple(args.origin), nodes, elements, constraints)

    n_points = (args.nx + 1) * (args.ny + 1)
    print(f"Wrote {out_path}")
    print(f"grid points={n_points} nodes={len(nodes)} elements={len(elements)} constraints={len(constraints)} joint={args.joint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
