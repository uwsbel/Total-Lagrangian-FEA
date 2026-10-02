"""Generate the HRZ lumped-mass reference for utest_feat10 (data/utest/hrz_mass_reference.csv).

For a straight-sided T10 element the isoparametric mapping is affine, so the
HRZ-lumped mass shares are exact constants: each of the 4 corner nodes carries
1/36 of the element mass and each of the 6 edge nodes 4/27 (4/36 + 24/27 = 1).
These follow from the consistent-mass diagonal int N_i^2 dV, which is a degree-4
integrand; quadrature rules of lower degree (for example the 5-point Keast rule,
degree 3 with a negative centroid weight) get the corner shares badly wrong.

Usage: python3 gen_hrz_mass_reference.py [--rho RHO] [--out PATH]
"""
import argparse
import os

CORNER_FRACTION = 1.0 / 36.0
EDGE_FRACTION = 4.0 / 27.0

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))


def read_node(path):
    with open(path) as fh:
        lines = [l for l in fh if l.strip() and not l.lstrip().startswith("#")]
    n_nodes = int(lines[0].split()[0])
    coords = {}
    for line in lines[1:n_nodes + 1]:
        f = line.split()
        coords[int(f[0])] = tuple(float(v) for v in f[1:4])
    return coords


def read_ele(path):
    with open(path) as fh:
        lines = [l for l in fh if l.strip() and not l.lstrip().startswith("#")]
    n_elem = int(lines[0].split()[0])
    return [[int(v) for v in line.split()[1:11]] for line in lines[1:n_elem + 1]]


def tet_volume(p0, p1, p2, p3):
    a = [p1[i] - p0[i] for i in range(3)]
    b = [p2[i] - p0[i] for i in range(3)]
    c = [p3[i] - p0[i] for i in range(3)]
    det = (a[0] * (b[1] * c[2] - b[2] * c[1])
           - a[1] * (b[0] * c[2] - b[2] * c[0])
           + a[2] * (b[0] * c[1] - b[1] * c[0]))
    return abs(det) / 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=2700.0)
    ap.add_argument("--mesh", default=os.path.join(ROOT, "data", "meshes", "T10", "beam_3x2x1.1"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "utest", "hrz_mass_reference.csv"))
    args = ap.parse_args()

    coords = read_node(args.mesh + ".node")
    elements = read_ele(args.mesh + ".ele")

    mass = {nid: 0.0 for nid in coords}
    total_volume = 0.0
    for nodes in elements:
        vol = tet_volume(*(coords[n] for n in nodes[:4]))
        total_volume += vol
        m_elem = args.rho * vol
        for local, nid in enumerate(nodes):
            mass[nid] += m_elem * (CORNER_FRACTION if local < 4 else EDGE_FRACTION)

    with open(args.out, "w") as fh:
        for nid in sorted(mass):
            fh.write(f"{mass[nid]:.15e}\n")

    print(f"{len(mass)} nodes, {len(elements)} elements, volume {total_volume:.6f}")
    print(f"total mass {sum(mass.values()):.6f} (rho * V = {args.rho * total_volume:.6f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
