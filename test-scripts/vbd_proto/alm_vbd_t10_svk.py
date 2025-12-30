#!/usr/bin/env python3
"""
ALM outer loop + TRUE VBD inner loop for TL-SVK with T10 tetrahedra.
- Outer loop: classic augmented Lagrangian multiplier update (same structure as your code).
- Inner loop: colored per-node 3×3 block updates on velocity v (NO global Hessian assembly).

Your Newton residual/Jacobian (for reference):
  R(v) = (M/h)(v - v_prev) + f_int(x) - f_ext + h T^T (lam + rho c)
  J(v) = (M/h) + h K_t(x) + h^2 T^T (rho I) T
  x = q_prev + h v

This VBD inner solve approximates J(v) by its per-node 3×3 diagonal blocks:
  H_i = (M_ii/h) + h * sum_{e∋i} K_ii^(e) + h^2 * rho * I   (only for constrained nodes)
and updates:
  v_i <- v_i - H_i^{-1} R_i

Constraint curvature:
- For your *pin* constraints, c(x_i)=x_i-X_i is linear, so the ONLY "constraint Hessian"
  is rho * T^T T. Per constrained node, its 3×3 block is simply rho*I.
- Therefore in v-space it contributes exactly +h^2*rho*I to H_i (constrained nodes only).
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from itertools import combinations
from typing import List, Tuple, Dict

I3 = np.eye(3, dtype=np.float64)

# ----------------------------
# TetGen I/O
# ----------------------------
def _read_nonempty_lines(path: str) -> List[str]:
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line)
    return out

def read_tetgen_node(path: str) -> Tuple[np.ndarray, np.ndarray]:
    lines = _read_nonempty_lines(path)
    n_pts, dim, *_ = map(int, lines[0].split()[:4])
    if dim != 3:
        raise ValueError(f"Expected dim=3, got {dim}")
    ids = np.empty(n_pts, dtype=np.int64)
    X = np.empty((n_pts, 3), dtype=np.float64)
    for i in range(n_pts):
        parts = lines[1 + i].split()
        ids[i] = int(parts[0])
        X[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return ids, X

def read_tetgen_ele(path: str) -> np.ndarray:
    lines = _read_nonempty_lines(path)
    n_tet, nodes_per, *_ = map(int, lines[0].split()[:3])
    conn = np.empty((n_tet, nodes_per), dtype=np.int64)
    for i in range(n_tet):
        parts = lines[1 + i].split()
        conn[i] = [int(x) for x in parts[1:1 + nodes_per]]
    return conn

# ----------------------------
# Quadrature (5-point Keast)
# ----------------------------
def tet5pt_quadrature():
    Lc = np.array([0.25, 0.25, 0.25, 0.25])
    a = 0.5
    b = 1.0/6.0
    pts_bary = np.array([
        Lc,
        [a, b, b, b],
        [b, a, b, b],
        [b, b, a, b],
        [b, b, b, a]
    ])
    w = np.array([-4.0/5.0, 9.0/20.0, 9.0/20.0, 9.0/20.0, 9.0/20.0]) * (1.0/6.0)
    pts_xyz = pts_bary[:, 1:4]
    return pts_xyz, pts_bary, w

# ----------------------------
# T10 shape gradients (CANONICAL)
#   edges: (0,1),(1,2),(0,2),(0,3),(1,3),(2,3)
# ----------------------------
_CANON_EDGES = [(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]

def tet10_shape_function_gradients_canon(xi, eta, zeta):
    L2, L3, L4 = xi, eta, zeta
    L1 = 1.0 - xi - eta - zeta
    L = [L1, L2, L3, L4]
    dL = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0],
    ], dtype=np.float64)
    dN = np.zeros((10,3), dtype=np.float64)
    for i in range(4):
        dN[i,:] = (4*L[i]-1)*dL[i,:]
    for k,(i,j) in enumerate(_CANON_EDGES, start=4):
        dN[k,:] = 4*(L[i]*dL[j,:] + L[j]*dL[i,:])
    return dN

def tet10_shape_functions_canon(xi, eta, zeta):
    L2, L3, L4 = xi, eta, zeta
    L1 = 1.0 - xi - eta - zeta
    L = [L1, L2, L3, L4]
    N = np.zeros(10, dtype=np.float64)
    N[0:4] = [L[i]*(2*L[i]-1) for i in range(4)]
    for k,(i,j) in enumerate(_CANON_EDGES, start=4):
        N[k] = 4*L[i]*L[j]
    return N

# ----------------------------
# Reorder each T10 element to canonical mid-edge order
# (keeps vertex order as given; just permutes nodes 4..9)
# ----------------------------
def reorder_t10_elements_to_canon(X: np.ndarray, conn_idx: np.ndarray, tol=1e-10) -> np.ndarray:
    conn_out = conn_idx.copy()
    for e in range(conn_idx.shape[0]):
        nodes = conn_idx[e]
        Xe = X[nodes]
        Xv = Xe[:4]
        Xm = Xe[4:]
        # map mid-edge node -> which edge (i,j)
        edge_for_mid = {}
        for k in range(6):
            m = Xm[k]
            best = None
            for (i,j) in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
                err = float(np.linalg.norm(m - 0.5*(Xv[i]+Xv[j])))
                if best is None or err < best[0]:
                    best = (err,(i,j))
            if best is None or best[0] > tol:
                raise RuntimeError(f"Elem {e}: mid-edge node did not match midpoint within tol. best_err={best[0] if best else None}")
            edge_for_mid[best[1]] = nodes[4+k]  # global node index

        mids = [edge_for_mid[edge] for edge in _CANON_EDGES]
        conn_out[e] = np.array([nodes[0],nodes[1],nodes[2],nodes[3], *mids], dtype=np.int64)
    return conn_out

# ----------------------------
# Precompute reference grads per element (store grad_N and dV)
# ----------------------------
def tet10_precompute_reference_mesh(X_nodes: np.ndarray, X_elem: np.ndarray):
    pts_xyz, _, w = tet5pt_quadrature()
    pre_mesh = []
    for e in range(X_elem.shape[0]):
        Xe = X_nodes[X_elem[e]]  # (10,3)
        pre = []
        for (xi,eta,zeta), wq in zip(pts_xyz, w):
            dN_dxi = tet10_shape_function_gradients_canon(xi,eta,zeta)  # (10,3)
            J = Xe.T @ dN_dxi
            detJ = float(np.linalg.det(J))
            grad_N = np.linalg.solve(J.T, dN_dxi.T).T  # (10,3)
            dV = abs(detJ) * wq
            pre.append({"grad_N": grad_N, "dV": dV})
        pre_mesh.append(pre)
    return pre_mesh

# ----------------------------
# TL-SVK internal force (local and global)
# ----------------------------
def svk_P(F: np.ndarray, lam: float, mu: float):
    FtF = F.T @ F
    trFtF = float(np.trace(FtF))
    return lam*(0.5*trFtF - 1.5)*F + mu*(F @ F.T @ F - F), FtF

def local_internal_force_and_Kii(node_i: int,
                                x_nodes: np.ndarray,
                                X_elem: np.ndarray,
                                pre_mesh,
                                incidence_i,
                                lam: float,
                                mu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      f_int_i : (3,) internal force contribution at node_i
      Kii_sum : (3,3) diagonal tangent block sum_{e∋i} K_ii^(e)
    computed ONLY from incident elements.
    """
    f_i = np.zeros(3, dtype=np.float64)
    Kii = np.zeros((3,3), dtype=np.float64)

    for (e, a_local) in incidence_i:
        idx = X_elem[e]
        xe = x_nodes[idx]  # (10,3)
        for q in pre_mesh[e]:
            H = q["grad_N"]  # (10,3)
            dV = q["dV"]

            # F = sum_a x_a ⊗ h_a^T  ==>  F = X^T H  (with rows as node vectors)
            F = xe.T @ H
            P, FtF = svk_P(F, lam, mu)

            ha = H[a_local]
            f_i += (P @ ha) * dV

            # diagonal tangent block K_aa for this quadrature point
            trE = 0.5*(float(np.trace(FtF)) - 3.0)
            FFT = F @ F.T
            g   = F @ ha
            s   = float(ha @ ha)
            g2  = float(g @ g)

            Kaa = (lam + mu) * np.outer(g, g) \
                  + lam * trE * s * I3 \
                  + mu  * g2 * I3 \
                  + mu  * s * (FFT - I3)
            Kii += Kaa * dV

    # symmetrize numeric noise
    Kii = 0.5*(Kii + Kii.T)
    return f_i, Kii

# ----------------------------
# Consistent mass (dense) – optional; for VBD we only use diagonal 3×3 blocks
# ----------------------------
def tet10_consistent_mass(X_nodes_elem: np.ndarray, rho: float) -> np.ndarray:
    pts_xyz, _, w = tet5pt_quadrature()
    Msc = np.zeros((10, 10), dtype=np.float64)
    for (xi,eta,zeta), wq in zip(pts_xyz, w):
        dN_dxi = tet10_shape_function_gradients_canon(xi,eta,zeta)
        J = X_nodes_elem.T @ dN_dxi
        detJ = abs(float(np.linalg.det(J)))
        N = tet10_shape_functions_canon(xi,eta,zeta)
        Msc += rho * np.outer(N, N) * detJ * wq
    M30 = np.zeros((30,30), dtype=np.float64)
    for i in range(10):
        for j in range(10):
            M30[3*i:3*i+3, 3*j:3*j+3] = Msc[i,j] * I3
    return M30

def tet10_consistent_mass_mesh_diag_blocks(X_nodes: np.ndarray, X_elem: np.ndarray, rho: float) -> np.ndarray:
    """
    Returns per-node diagonal 3×3 blocks M_ii assembled from consistent element mass.
    This avoids building the full 3N×3N matrix and matches the VBD "local mass" spirit.
    """
    n_nodes = X_nodes.shape[0]
    Mdiag = np.zeros((n_nodes,3,3), dtype=np.float64)
    for e in range(X_elem.shape[0]):
        idx = X_elem[e]
        Me = tet10_consistent_mass(X_nodes[idx], rho)  # (30,30)
        for a_local, a_global in enumerate(idx):
            Mdiag[a_global] += Me[3*a_local:3*a_local+3, 3*a_local:3*a_local+3]
    return Mdiag

# ----------------------------
# Coloring and incidence
# ----------------------------
def build_incidence(X_elem: np.ndarray, n_nodes: int):
    inc = [[] for _ in range(n_nodes)]
    for e in range(X_elem.shape[0]):
        for a_local, g in enumerate(X_elem[e]):
            inc[g].append((e, a_local))
    return inc

def build_vertex_adjacency(X_elem: np.ndarray, n_nodes: int):
    adj = [set() for _ in range(n_nodes)]
    for tet in X_elem:
        tet = np.unique(tet)
        for a,b in combinations(tet,2):
            adj[a].add(b); adj[b].add(a)
    return adj

def greedy_vertex_coloring(adj):
    n = len(adj)
    deg = np.array([len(s) for s in adj], dtype=np.int64)
    order = np.argsort(-deg)
    colors = -np.ones(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    for v in order:
        used[:] = False
        for nb in adj[v]:
            c = colors[nb]
            if c >= 0:
                used[c] = True
        c = 0
        while used[c]:
            c += 1
        colors[v] = c
    return colors

def validate_coloring(X_elem: np.ndarray, colors: np.ndarray):
    for e, tet in enumerate(X_elem):
        c = colors[tet]
        if len(np.unique(c)) != len(c):
            raise RuntimeError(f"Invalid coloring: element {e} repeats a color.")
    return True

# ----------------------------
# Pin constraints exactly matching your code:
#   fixed nodes are those with X[:,0]==xmin (or ==0 if you prefer)
#   c(qA) stacks (x_i - X_i) for fixed nodes
# ----------------------------
def get_fixed_nodes(X_nodes: np.ndarray, xmin: float | None = None):
    if xmin is None:
        xmin = float(X_nodes[:,0].min())
    return np.where(np.isclose(X_nodes[:,0], xmin))[0]

def build_fixed_map(fixed_nodes: np.ndarray, n_nodes: int) -> np.ndarray:
    m = -np.ones(n_nodes, dtype=np.int64)
    for k, node in enumerate(fixed_nodes):
        m[node] = k
    return m

# ----------------------------
# VBD inner solve in velocity space (NO global Hessian)
# ----------------------------
def vbd_inner(v0_flat: np.ndarray,
              q_prev_flat: np.ndarray,
              v_prev_flat: np.ndarray,
              Mdiag_blocks: np.ndarray,          # (N,3,3) diag blocks
              X_nodes: np.ndarray,
              X_elem: np.ndarray,
              pre_mesh,
              f_ext_flat: np.ndarray,
              h: float,
              lam: float,
              mu: float,
              lam_mult: np.ndarray,              # (3*n_fixed,)
              rho_bb: float,
              fixed_nodes: np.ndarray,
              fixed_map: np.ndarray,             # node -> k in fixed_nodes
              colors_to_nodes: List[np.ndarray],
              incidence,
              max_sweeps: int = 50,
              omega: float = 1.0,
              tol_R: float = 1e-8,
              hess_eps: float = 1e-12,
              verbose: bool = True) -> Tuple[np.ndarray, int]:
    """
    Approximate solve R(v)=0 at fixed multipliers using colored per-node 3×3 updates.
    Returns v_flat and number of sweeps performed.
    """
    N = X_nodes.shape[0]
    v = v0_flat.reshape(N,3).copy()
    v_prev = v_prev_flat.reshape(N,3)
    q_prev = q_prev_flat.reshape(N,3)
    f_ext = f_ext_flat.reshape(N,3)

    # pre-slice multipliers per fixed node
    lam_mult = lam_mult.reshape(-1,3)  # (n_fixed,3)

    inv_h = 1.0 / h

    def compute_residual_norm_full() -> float:
        # Full (global) residual norm check ||R(v)|| over all nodes.
        x = q_prev + h*v

        f_int = np.zeros((N, 3), dtype=np.float64)
        for e in range(X_elem.shape[0]):
            idx = X_elem[e]
            xe = x[idx]  # (10,3)
            for q in pre_mesh[e]:
                H = q["grad_N"]  # (10,3)
                dV = q["dV"]
                F = xe.T @ H
                P, _ = svk_P(F, lam, mu)
                fe = (P @ H.T).T * dV  # (10,3)
                f_int[idx] += fe

        inertia = np.einsum("nij,nj->ni", Mdiag_blocks, (v - v_prev)) * inv_h
        R = inertia + f_int - f_ext

        for i in fixed_nodes:
            k = fixed_map[i]
            if k >= 0:
                c_i = (q_prev[i] + h*v[i]) - X_nodes[i]
                R[i] += h * (lam_mult[k] + rho_bb * c_i)

        return float(np.linalg.norm(R.reshape(-1)))

    sweeps_done = 0
    for sweep in range(max_sweeps):
        sweeps_done += 1

        # color-by-color updates
        for nodes_c in colors_to_nodes:
            # "parallel within color": compute updates based on current v before this color updates
            v_old = v.copy()
            x_old = q_prev + h*v_old

            for i in nodes_c:
                # local internal force and local diagonal stiffness
                f_int_i, Kii = local_internal_force_and_Kii(i, x_old, X_elem, pre_mesh, incidence[i], lam, mu)

                # residual block
                Ri = (Mdiag_blocks[i] @ (v_old[i] - v_prev[i])) * inv_h \
                     + f_int_i - f_ext[i]

                # local Hessian block in v-space
                Hi = (Mdiag_blocks[i] * inv_h) + (h * Kii)

                # ALM constraint contributions (pins only):
                k = fixed_map[i]
                if k >= 0:
                    c_i = (q_prev[i] + h*v_old[i]) - X_nodes[i]  # (3,)
                    Ri += h * (lam_mult[k] + rho_bb * c_i)
                    Hi += (h*h) * rho_bb * I3

                # stabilize Hi
                Hi = 0.5*(Hi + Hi.T)
                Hi += (hess_eps * max(1.0, float(np.trace(Hi)))) * I3

                dv = -np.linalg.solve(Hi, Ri)
                v[i] = v_old[i] + omega*dv

        if verbose and (sweep % 5 == 0 or sweep == max_sweeps-1):
            rn = compute_residual_norm_full()
            print(f"    VBD sweep {sweep:03d}: total ||R|| = {rn:.3e}")
            if rn < tol_R*(1.0 + float(np.linalg.norm(v))):
                break

    return v.reshape(-1), sweeps_done

# ----------------------------
# ALM outer loop with VBD inner loop (same structure as your Newton-based ALM)
# ----------------------------
def alm_vbd_step(v_guess: np.ndarray,
                 lam_guess: np.ndarray,
                 v_prev: np.ndarray,
                 q_prev: np.ndarray,
                 Mdiag_blocks: np.ndarray,
                 X_nodes: np.ndarray,
                 X_elem: np.ndarray,
                 pre_mesh,
                 f_ext: np.ndarray,
                 h: float,
                 rho_bb: float,
                 lam: float,
                 mu: float,
                 fixed_nodes: np.ndarray,
                 fixed_map: np.ndarray,
                 colors_to_nodes: List[np.ndarray],
                 incidence,
                 max_outer: int = 5,
                 outer_tol: float = 1e-6,
                 inner_sweeps: int = 50,
                 omega: float = 1.0) -> Tuple[np.ndarray, np.ndarray, int, int]:
    v = v_guess.copy()
    lam_mult = lam_guess.copy()

    actual_outer_iters = 0
    total_inner_sweeps = 0

    for outer_iter in range(max_outer):
        actual_outer_iters += 1

        # ---- INNER (VBD) at fixed multipliers ----
        v, sweeps = vbd_inner(
            v0_flat=v,
            q_prev_flat=q_prev,
            v_prev_flat=v_prev,
            Mdiag_blocks=Mdiag_blocks,
            X_nodes=X_nodes,
            X_elem=X_elem,
            pre_mesh=pre_mesh,
            f_ext_flat=f_ext,
            h=h,
            lam=lam,
            mu=mu,
            lam_mult=lam_mult,
            rho_bb=rho_bb,
            fixed_nodes=fixed_nodes,
            fixed_map=fixed_map,
            colors_to_nodes=colors_to_nodes,
            incidence=incidence,
            max_sweeps=inner_sweeps,
            omega=omega,
            tol_R=1e-8,
            verbose=True
        )
        total_inner_sweeps += sweeps

        # ---- OUTER multiplier update ----
        N = X_nodes.shape[0]
        qA = (q_prev + h*v).reshape(N,3)
        cA = (qA[fixed_nodes] - X_nodes[fixed_nodes]).reshape(-1)  # (3*n_fixed,)
        lam_mult += rho_bb * cA

        cn = float(np.linalg.norm(cA))
        print(f">>>>> OUTER {outer_iter}: ||c||={cn:.3e}  (inner sweeps={sweeps})")
        if cn < outer_tol:
            break

    return v, lam_mult, actual_outer_iters, total_inner_sweeps

# ----------------------------
# Demo main: similar to your simulation loop
# ----------------------------
def main():
    # Set these paths as needed
    node_path = "beam_3x2x1_res4.1.node"
    ele_path  = "beam_3x2x1_res4.1.ele"

    node_ids, X_nodes = read_tetgen_node(node_path)
    conn = read_tetgen_ele(ele_path)
    if conn.shape[1] != 10:
        raise ValueError("Expected T10 .ele with 10 nodes per element")

    # map ids -> indices
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    X_elem = np.vectorize(lambda nid: id2idx[nid])(conn)

    # reorder to canonical mid-edge order
    X_elem = reorder_t10_elements_to_canon(X_nodes, X_elem, tol=1e-10)

    # material (your Material 1)
    E_mat = 7e8
    nu = 0.33
    mu = E_mat / (2*(1+nu))
    lam = E_mat*nu/((1+nu)*(1-2*nu))
    rho0 = 2700.0

    # precompute reference data
    pre_mesh = tet10_precompute_reference_mesh(X_nodes, X_elem)

    # for VBD, we only need diagonal mass blocks
    Mdiag_blocks = tet10_consistent_mass_mesh_diag_blocks(X_nodes, X_elem, rho0)

    # constraints: pins at xmin
    fixed_nodes = get_fixed_nodes(X_nodes)  # uses xmin
    fixed_map = build_fixed_map(fixed_nodes, X_nodes.shape[0])
    print(f"Fixed nodes: {len(fixed_nodes)}")

    # coloring
    adj = build_vertex_adjacency(X_elem, X_nodes.shape[0])
    colors = greedy_vertex_coloring(adj)
    validate_coloring(X_elem, colors)
    n_colors = int(colors.max()+1)
    colors_to_nodes = [np.where(colors == c)[0] for c in range(n_colors)]
    print(f"Coloring: {n_colors} colors")

    incidence = build_incidence(X_elem, X_nodes.shape[0])

    # simulation params
    h = 1e-3
    rho_bb = 1e14

    q_prev = X_nodes.reshape(-1).copy()
    v_prev = np.zeros_like(q_prev)
    v_guess = v_prev.copy()
    lam_guess = np.zeros(3*len(fixed_nodes), dtype=np.float64)

    # external force: replicate your example (node 19 +x)
    f_ext = np.zeros(3*X_nodes.shape[0], dtype=np.float64)
    if X_nodes.shape[0] > 20:
        f_ext[3*19 + 0] = 1000.0

    Nt = 60
    track = []

    for step in range(Nt):
        if step > 30 and X_nodes.shape[0] > 20:
            f_ext[3*19 + 0] = 0.0

        v_res, lam_res, outer_iters, inner_sweeps = alm_vbd_step(
            v_guess, lam_guess,
            v_prev, q_prev,
            Mdiag_blocks,
            X_nodes, X_elem, pre_mesh,
            f_ext, h, rho_bb,
            lam, mu,
            fixed_nodes, fixed_map,
            colors_to_nodes, incidence,
            max_outer=5,
            outer_tol=1e-6,
            inner_sweeps=100,
            omega=1.0
        )

        v_guess, lam_guess = v_res.copy(), lam_res.copy()
        q_new = q_prev + h*v_guess
        x_nodes = q_new.reshape(X_nodes.shape[0], 3)

        if X_nodes.shape[0] > 20:
            print(f"Step {step}: node 19 = {x_nodes[19]}  outer={outer_iters}  inner_sweeps={inner_sweeps}")
            track.append(x_nodes[19,0])

        q_prev = q_new.copy()
        v_prev = v_guess.copy()

    # plot displacement magnitude and node track
    x_final = q_prev.reshape(X_nodes.shape[0],3)
    u = np.linalg.norm(x_final - X_nodes, axis=1)

    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(1,2,1, projection="3d")
    p1 = ax1.scatter(X_nodes[:,0], X_nodes[:,1], X_nodes[:,2], c=u, s=4)
    ax1.set_title("Final displacement |u| (ALM+VBD)")
    plt.colorbar(p1, ax=ax1, shrink=0.6)

    ax2 = fig.add_subplot(1,2,2)
    if len(track) > 0:
        ax2.plot(track, linewidth=2)
        ax2.set_title("Node 19 x over time")
        ax2.set_xlabel("step"); ax2.set_ylabel("x")
        ax2.grid(True)
    else:
        ax2.text(0.1, 0.5, "Mesh too small to track node 19.", transform=ax2.transAxes)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
