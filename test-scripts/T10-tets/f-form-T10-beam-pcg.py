import numpy as np
import matplotlib.pyplot as plt
from tet_mesh_reader import read_node, read_ele

# ----------------------------
# Material 1
# ----------------------------
E_mat = 7e8
nu = 0.33
mu = E_mat / (2*(1+nu))
lam = E_mat*nu/((1+nu)*(1-2*nu))
rho0 = 2700.0


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
# Shape functions & gradients (TET10)
# ----------------------------
def tet10_shape_function_gradients(xi, eta, zeta):
    L2, L3, L4 = xi, eta, zeta
    L1 = 1.0 - xi - eta - zeta
    L = [L1, L2, L3, L4]
    dL = np.array([
        [-1, -1, -1],
        [1,  0,  0],
        [0,  1,  0],
        [0,  0,  1],
    ])
    dN_dxi = np.zeros((10, 3))
    for i in range(4):
        dN_dxi[i, :] = (4*L[i]-1)*dL[i, :]
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    for k, (i, j) in enumerate(edges, start=4):
        dN_dxi[k, :] = 4*(L[i]*dL[j, :] + L[j]*dL[i, :])
    return dN_dxi

# ----------------------------
# Precompute reference gradients
# ----------------------------
def tet10_precompute_reference(X_elem_nodes):
    pts_xyz, _, w = tet5pt_quadrature()
    pre = []
    for (xi, eta, zeta), wq in zip(pts_xyz, w):
        dN_dxi = tet10_shape_function_gradients(xi, eta, zeta)
        J = np.zeros((3, 3))
        for a in range(10):
            J += np.outer(X_elem_nodes[a], dN_dxi[a])  # X ⊗ ∂N/∂ξ
        detJ = np.linalg.det(J)
        grad_N = np.zeros((10, 3))
        JT = J.T
        for a in range(10):
            grad_N[a, :] = np.linalg.solve(JT, dN_dxi[a])  # h_a = J^{-T} ∂N/∂ξ
        pre.append({"grad_N": grad_N, "detJ": detJ, "w": wq})
    return pre

def tet10_precompute_reference_mesh(X_nodes, X_elem):
    pre_list = []
    for elem_idx in range(X_elem.shape[0]):
        node_indices = X_elem[elem_idx]
        X_elem_nodes = X_nodes[node_indices]      # (10, 3)
        print("=====")
        print(X_elem_nodes)
        pre = tet10_precompute_reference(X_elem_nodes)
        pre_list.append(pre)
    return pre_list

# ----------------------------
# Internal force (SVK, TL)
# ----------------------------
def tet10_internal_force(x_nodes, pre, lam, mu):
    f = np.zeros((10, 3))
    for q in pre:
        grad_N = q["grad_N"]   # h_a
        detJ   = q["detJ"]
        wq     = q["w"]

        # F = sum_a x_a ⊗ h_a^T
        F = np.zeros((3, 3))
        for a in range(10):
            F += np.outer(x_nodes[a], grad_N[a])

        FtF   = F.T @ F
        trFtF = np.trace(FtF)
        # P for SVK in your code form:
        P = lam*(0.5*trFtF - 1.5)*F + mu*(F @ F.T @ F - F)

        dV = detJ * wq
        for a in range(10):
            f[a] += (P @ grad_N[a]) * dV
    return f

def tet10_internal_force_mesh(x_nodes, X_elem, pre_mesh, lam, mu):
    n_nodes = x_nodes.shape[0]
    f_int = np.zeros((n_nodes, 3))
    for elem_idx in range(X_elem.shape[0]):
        node_indices = X_elem[elem_idx]           # (10,)
        x_elem_nodes = x_nodes[node_indices]      # (10, 3)
        pre = pre_mesh[elem_idx]
        f_elem = tet10_internal_force(x_elem_nodes, pre, lam, mu)  # (10, 3)
        for a_local, a_global in enumerate(node_indices):
            f_int[a_global] += f_elem[a_local]
    return f_int.flatten()

# ----------------------------
# Consistent mass
# ----------------------------
def tet10_consistent_mass(X_nodes, rho):
    pts_xyz, _, w = tet5pt_quadrature()
    Msc = np.zeros((10, 10))
    for (xi, eta, zeta), wq in zip(pts_xyz, w):
        dN_dxi = tet10_shape_function_gradients(xi, eta, zeta)
        J = np.zeros((3, 3))
        for a in range(10):
            J += np.outer(X_nodes[a], dN_dxi[a])
        detJ = abs(np.linalg.det(J))
        N = np.zeros(10)
        L2, L3, L4 = xi, eta, zeta
        L1 = 1-xi-eta-zeta
        L = [L1, L2, L3, L4]
        N[0:4] = [L[i]*(2*L[i]-1) for i in range(4)]
        edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
        for k, (i, j) in enumerate(edges, start=4):
            N[k] = 4*L[i]*L[j]
        Msc += rho * np.outer(N, N) * detJ * wq
    M30 = np.zeros((30, 30))
    for i in range(10):
        for j in range(10):
            M30[3*i:3*i+3, 3*j:3*j+3] = Msc[i, j]*np.eye(3)
    return M30

def tet10_consistent_mass_mesh(X_nodes, X_elem, rho):
    n_nodes = X_nodes.shape[0]
    n_elements = X_elem.shape[0]
    M_full = np.zeros((3*n_nodes, 3*n_nodes))
    for elem_idx in range(n_elements):
        node_indices = X_elem[elem_idx]    # (10,)
        X_elem_nodes = X_nodes[node_indices]
        M_elem = tet10_consistent_mass(X_elem_nodes, rho)  # (30, 30)

        # scatter
        gdofs = []
        for ni in node_indices:
            gdofs.extend([3*ni, 3*ni+1, 3*ni+2])

        for i_local, i_global in enumerate(gdofs):
            for j_local, j_global in enumerate(gdofs):
                M_full[i_global, j_global] += M_elem[i_local, j_local]
    return M_full

# ----------------------------
# Constraints (pins at x==0 reference nodes)
# ----------------------------
def get_fixed_nodes(X_nodes):
    return np.where(np.isclose(X_nodes[:, 0], 0.0))[0]

def constraint(q):
    fixed_nodes = get_fixed_nodes(X_nodes)
    c = np.zeros(3 * len(fixed_nodes))
    for idx, node in enumerate(fixed_nodes):
        c[3*idx:3*idx+3] = q[3*node:3*node+3] - X_nodes[node]
    return c

def constraint_jacobian(q):
    fixed_nodes = get_fixed_nodes(X_nodes)
    J = np.zeros((3 * len(fixed_nodes), len(q)))
    for idx, node in enumerate(fixed_nodes):
        J[3*idx,   3*node]   = 1
        J[3*idx+1, 3*node+1] = 1
        J[3*idx+2, 3*node+2] = 1
    return J

# ============================================================
# TANGENT STIFFNESS (SVK)
# ============================================================

def tet10_tangent_svk(x_elem_nodes, pre, lam, mu):
    K = np.zeros((30, 30))
    I3 = np.eye(3)

    for q in pre:
        H  = q["grad_N"]                 # (10,3): rows are h_i^T
        dV = q["detJ"] * q["w"]

        # F = sum_a x_a ⊗ h_a^T
        F = np.zeros((3,3))
        for a in range(10):
            F += np.outer(x_elem_nodes[a], H[a])

        C    = F.T @ F
        trE  = 0.5*(np.trace(C) - 3.0)   # tr(E) where E=0.5(C-I)
        FFT  = F @ F.T

        # Precompute F h_i
        Fh = np.zeros((10,3))
        for i in range(10):
            Fh[i] = F @ H[i]

        # Assemble 3x3 blocks
        for i in range(10):
            for j in range(10):
                hij  = float(H[j] @ H[i])              # (h_j)^T h_i (scalar)

                A    = lam * np.outer(Fh[i], Fh[j])    # λ (Fh_i)(Fh_j)^T
                B    = lam * trE * hij * I3            # λ tr(E) (h_j^T h_i) I
                C1   = mu  * float(Fh[j] @ Fh[i]) * I3 # μ (Fh_j)^T(Fh_i) I
                D    = mu  * np.outer(Fh[j], Fh[i])    # μ (Fh_j)(Fh_i)^T
                Etrm = mu  * hij * FFT                 # μ (h_j^T h_i) F F^T
                Ftrm = -mu * hij * I3                  # −μ (h_j^T h_i) I

                K_ij = (A + B + C1 + D + Etrm + Ftrm) * dV

                I0 = 3*i; J0 = 3*j
                K[I0:I0+3, J0:J0+3] += K_ij
    return K

def tet10_tangent_svk_mesh(X_nodes, x_nodes, X_elem, pre_mesh, lam, mu):
    n_nodes = X_nodes.shape[0]
    K_full = np.zeros((3*n_nodes, 3*n_nodes))
    for e in range(X_elem.shape[0]):
        idx = X_elem[e]                     # (10,)
        x_e = x_nodes[idx]                  # (10,3)
        Ke  = tet10_tangent_svk(x_e, pre_mesh[e], lam, mu)  # (30,30)

        # scatter
        gdofs = []
        for ni in idx:
            gdofs.extend([3*ni, 3*ni+1, 3*ni+2])
        for aL, aG in enumerate(gdofs):
            for bL, bG in enumerate(gdofs):
                K_full[aG, bG] += Ke[aL, bL]
    return K_full

# ============================================================
# BLOCK-DIAGONAL PRECONDITIONED CONJUGATE GRADIENT
# ============================================================

def extract_block_diag_precond(H, n_nodes, eps=1e-8):
    """
    Extract 3x3 diagonal blocks from H, symmetrize, regularize, invert.
    """
    M_inv_blocks = np.zeros((n_nodes, 3, 3))
    for i in range(n_nodes):
        blk = H[3*i:3*i+3, 3*i:3*i+3].copy()

        # Symmetrize
        blk = 0.5 * (blk + blk.T)

        # Regularize: add eps * max(1, |trace|) to diagonal
        trace = blk[0, 0] + blk[1, 1] + blk[2, 2]
        reg = eps * max(1.0, abs(trace))
        blk[0, 0] += reg
        blk[1, 1] += reg
        blk[2, 2] += reg

        # Invert
        det = np.linalg.det(blk)
        if abs(det) < 1e-30:
            M_inv_blocks[i] = np.eye(3)
        else:
            M_inv_blocks[i] = np.linalg.inv(blk)

    return M_inv_blocks

def apply_precond(M_inv_blocks, r, n_nodes):
    """
    Block-diagonal matrix-vector product: z[3i:3i+3] = M_inv_blocks[i] @ r[3i:3i+3].
    """
    z = np.zeros_like(r)
    for i in range(n_nodes):
        z[3*i:3*i+3] = M_inv_blocks[i] @ r[3*i:3*i+3]
    return z

def pcg_inner(v0, q_prev, v_prev, M, f_int_func, f_ext, h,
              X_nodes, X_elem, pre_mesh, lam, mu,
              lam_mult, rho_bb,
              max_newton=10, max_pcg=200, pcg_rtol=1e-4,
              inner_atol=1e-4, inner_rtol=1e-4):
    """
    Inexact Newton with PCG inner solve using block-diagonal preconditioner.
    """
    v = v0.copy()
    N = M.shape[0] // 3
    total_pcg_iters = 0

    norm_g0 = None

    for it in range(max_newton):
        qA = q_prev + h*v
        x  = qA.reshape(N, 3)

        # Residual (gradient of Lagrangian)
        f_int = f_int_func(x, X_elem, pre_mesh, lam, mu)
        R_mech = (M @ (v - v_prev)) / h + f_int - f_ext

        T = constraint_jacobian(qA)
        c = constraint(qA)
        g = R_mech + h * (T.T @ (lam_mult + rho_bb * c))

        ng = np.linalg.norm(g)
        if norm_g0 is None:
            norm_g0 = ng
        rel_g = ng / norm_g0 if norm_g0 > 0 else 0.0
        print(f"    Newton {it:02d}: ||g|| = {ng:.3e}, ||g||/||g0|| = {rel_g:.3e}")
        if ng < inner_atol or (inner_rtol > 0.0 and norm_g0 > 0.0 and ng <= inner_rtol * norm_g0):
            break

        # Hessian: H = M/h + h*Kt + h^2*rho*T^T*T
        Kt = tet10_tangent_svk_mesh(X_nodes, x, X_elem, pre_mesh, lam, mu)
        H = (M / h) + h * Kt + (h**2) * (T.T @ (rho_bb * T))

        # Extract block-diagonal preconditioner
        M_inv_blocks = extract_block_diag_precond(H, N)

        # PCG solve: H @ dv = -g
        r = -g.copy()
        dv = np.zeros_like(g)
        z = apply_precond(M_inv_blocks, r, N)
        p = z.copy()
        rz = r @ z
        r_norm0 = np.linalg.norm(r)
        r_norm = r_norm0
        pcg_converged = False
        pcg_bad_curvature = False

        pcg_iters = 0
        for pcg_it in range(max_pcg):
            Hp = H @ p
            denom = p @ Hp
            if denom <= 0.0:
                pcg_bad_curvature = True
                break
            if abs(denom) < 1e-30:
                break

            alpha = rz / denom
            dv += alpha * p
            r -= alpha * Hp

            r_norm = np.linalg.norm(r)
            pcg_iters = pcg_it + 1
            if r_norm < pcg_rtol * r_norm0 or r_norm < 1e-15:
                pcg_converged = True
                break

            z_new = apply_precond(M_inv_blocks, r, N)
            rz_new = r @ z_new
            if abs(rz) < 1e-30:
                break

            beta = rz_new / rz
            p = z_new + beta * p
            z = z_new
            rz = rz_new

        if not pcg_converged:
            if pcg_bad_curvature:
                print("      WARNING: PCG stopped due to nonpositive p^T H p")
            else:
                print(f"      WARNING: PCG did not converge, ||r|| = {r_norm:.3e}")
        total_pcg_iters += pcg_iters
        print(f"      PCG converged in {pcg_iters} iters, ||r|| = {np.linalg.norm(r):.3e}")

        v += dv

    return v, it+1, total_pcg_iters

def alm_pcg_step(v_guess, lam_guess, v_prev, q_prev, M, f_int_func, f_ext, h, rho_bb,
                 X_nodes, X_elem, pre_mesh, lam, mu,
                 max_outer=5, outer_tol=1e-6):
    """
    ALM outer loop with PCG inner solves.
    """
    v = v_guess.copy()
    lam_mult = lam_guess.copy()

    actual_outer_iters = 0
    total_newton_iters = 0
    total_pcg_iters = 0

    for outer_iter in range(max_outer):
        actual_outer_iters += 1

        v, nit, pcg_iters = pcg_inner(v, q_prev, v_prev, M, f_int_func, f_ext, h,
                                       X_nodes, X_elem, pre_mesh, lam, mu,
                                       lam_mult, rho_bb,
                                       max_newton=10, max_pcg=200, pcg_rtol=1e-4,
                                       inner_atol=1e-4, inner_rtol=1e-4)
        total_newton_iters += nit
        total_pcg_iters += pcg_iters

        # ALM multiplier update
        qA = q_prev + h*v
        cA = constraint(qA)
        lam_mult += rho_bb * cA

        print(f">>>>> OUTER {outer_iter}: ||c||={np.linalg.norm(cA):.3e}")
        if np.linalg.norm(cA) < outer_tol:
            break

    return v, lam_mult, actual_outer_iters, total_newton_iters, total_pcg_iters

# ----------------------------
# Main simulation
# ----------------------------
if __name__ == "__main__":
    # shape: (n_nodes, 3)
    X_nodes = read_node("../../data/meshes/T10/beam_3x2x1.1.node")
    x_nodes = X_nodes.copy()
    X_elem = read_ele("../../data/meshes/T10/beam_3x2x1.1.ele")

    print(X_nodes)
    print(X_elem)
    pre_mesh = tet10_precompute_reference_mesh(X_nodes, X_elem)

    print(pre_mesh)

    np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)

    M_full = tet10_consistent_mass_mesh(X_nodes, X_elem, rho0)
    print("shape M_full:", M_full.shape)
    print(M_full)

    f_ext = np.zeros(3 * X_nodes.shape[0])
    f_ext[3*19 + 0] = 1000.0  # 1000 N force in x direction at node index 19
    time_step = 1e-3
    rho_bb = 1e14

    q_prev = x_nodes.flatten()
    v_prev = np.zeros_like(q_prev)
    v_guess = v_prev.copy()
    lam_guess = np.zeros(3 * len(get_fixed_nodes(X_nodes)))

    Nt = 200
    node19_x = []
    node20_x = []
    outer_iters_per_step = []
    inner_iters_per_step = []
    pcg_iters_per_step = []

    for step in range(Nt):
        if step > 50:
            f_ext[3*19 + 0] = 0.0  # remove force after step 50

        v_res, lam_res, outer_iters, inner_iters, pcg_iters = alm_pcg_step(
            v_guess, lam_guess, v_prev, q_prev, M_full,
            tet10_internal_force_mesh, f_ext, time_step, rho_bb,
            X_nodes, X_elem, pre_mesh, lam, mu)

        v_guess, lam_guess = v_res.copy(), lam_res.copy()
        q_new = q_prev + time_step * v_guess
        x_nodes = q_new.reshape(X_nodes.shape[0], 3)

        print(f"Step {step}: node 19 position = {x_nodes[19]}, node 20 position = {x_nodes[20]}")
        print(f"Step {step}: outer={outer_iters}, newton={inner_iters}, pcg={pcg_iters}")

        node19_x.append(x_nodes[19, 0])
        node20_x.append(x_nodes[20, 0])
        outer_iters_per_step.append(outer_iters)
        inner_iters_per_step.append(inner_iters)
        pcg_iters_per_step.append(pcg_iters)

        q_prev = q_new.copy()
        v_prev = v_guess.copy()

    # Plot after simulation
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    ax1.plot(range(Nt), node19_x, marker='o', linewidth=2, markersize=4)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Node 19 X Position")
    ax1.set_title("Node 19 X Position vs Step"); ax1.grid(True)

    ax2.plot(range(Nt), node20_x, marker='x', color='orange', linewidth=2, markersize=4)
    ax2.set_xlabel("Step"); ax2.set_ylabel("Node 20 X Position")
    ax2.set_title("Node 20 X Position vs Step"); ax2.grid(True)

    ax3.plot(range(Nt), outer_iters_per_step, marker='s', color='red', linewidth=2, markersize=4)
    ax3.set_xlabel("Step"); ax3.set_ylabel("Outer Iterations")
    ax3.set_title("Outer Loop Iterations per Step"); ax3.grid(True); ax3.set_ylim(bottom=0)

    ax4.plot(range(Nt), pcg_iters_per_step, marker='^', color='green', linewidth=2, markersize=4)
    ax4.set_xlabel("Step"); ax4.set_ylabel("PCG Iterations")
    ax4.set_title("Total PCG Iterations per Step"); ax4.grid(True); ax4.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "="*50)
    print("ITERATION SUMMARY STATISTICS")
    print("="*50)
    print(f"Total outer iterations: {sum(outer_iters_per_step)}")
    print(f"Total Newton iterations: {sum(inner_iters_per_step)}")
    print(f"Total PCG iterations: {sum(pcg_iters_per_step)}")
    print(f"Average outer iterations per step: {np.mean(outer_iters_per_step):.2f}")
    print(f"Average Newton iterations per step: {np.mean(inner_iters_per_step):.2f}")
    print(f"Average PCG iterations per step: {np.mean(pcg_iters_per_step):.2f}")
    print(f"Max outer iterations in a step: {max(outer_iters_per_step)}")
    print(f"Max Newton iterations in a step: {max(inner_iters_per_step)}")
    print(f"Max PCG iterations in a step: {max(pcg_iters_per_step)}")
    print(f"Min outer iterations in a step: {min(outer_iters_per_step)}")
    print(f"Min Newton iterations in a step: {min(inner_iters_per_step)}")
    print(f"Min PCG iterations in a step: {min(pcg_iters_per_step)}")

    # Cumulative iteration plot
    plt.figure(figsize=(10, 6))
    cumulative_outer = np.cumsum(outer_iters_per_step)
    cumulative_newton = np.cumsum(inner_iters_per_step)
    cumulative_pcg = np.cumsum(pcg_iters_per_step)
    plt.plot(range(Nt), cumulative_outer, label='Cumulative Outer Iterations', linewidth=2)
    plt.plot(range(Nt), cumulative_newton, label='Cumulative Newton Iterations', linewidth=2)
    plt.plot(range(Nt), cumulative_pcg, label='Cumulative PCG Iterations', linewidth=2)
    plt.xlabel("Step"); plt.ylabel("Cumulative Iterations")
    plt.title("Cumulative Iteration Count")
    plt.legend(); plt.grid(True)
    plt.show()
