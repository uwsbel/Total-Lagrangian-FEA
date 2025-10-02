import numpy as np

# ----------------------------
# Material
# ----------------------------
E = 7e8
nu = 0.33
mu = E / (2*(1+nu))
lam = E*nu/((1+nu)*(1-2*nu))
rho0 = 2700.0


# ----------------------------
# Quadrature (4-pt symmetric, positive weights)
# Exact for quadratics on the reference tetra
# ----------------------------
#def tet4pt_quadrature():
#    a = 0.58541020
#    b = 0.13819660
#    pts_bary = np.array([
#        [a, b, b, b],
#        [b, a, b, b],
#        [b, b, a, b],
#        [b, b, b, a]
#    ])
    # (xi, eta, zeta) := (L2, L3, L4) in your notation
#    pts_xyz = pts_bary[:, 1:4]
    # weights sum to 1/6 (volume of reference tetra)
#    w = np.ones(4) * (1.0/24.0)
#    return pts_xyz, pts_bary, w

def tet5pt_quadrature():
    """
    5-point quadrature rule for tetrahedron (Keast, degree-3 exactness).
    One centroid point with negative weight, four symmetric off-centroid points
    with positive weights. Weights sum to 1/6 (reference tetra volume).
    Returns:
        pts_xyz : (5,3) parent coords (xi,eta,zeta)
        pts_bary: (5,4) barycentric coords (L1..L4)
        w       : (5,) quadrature weights
    """
    # Centroid
    Lc = np.array([0.25, 0.25, 0.25, 0.25])

    # Four symmetric off-centroid points
    a = 0.5
    b = 1.0/6.0
    pts_bary = np.array([
        Lc,
        [a, b, b, b],
        [b, a, b, b],
        [b, b, a, b],
        [b, b, b, a]
    ])

    # Quadrature weights (Keast 5-point rule)
    w = np.array([-4.0/5.0, 9.0/20.0, 9.0/20.0, 9.0/20.0, 9.0/20.0]) * (1.0/6.0)

    # (xi, eta, zeta) = (L2, L3, L4)
    pts_xyz = pts_bary[:, 1:4]
    return pts_xyz, pts_bary, w



# ----------------------------
# Shape functions & gradients (TET10)
# xi,eta,zeta are parent coords with L2=xi,L3=eta,L4=zeta and L1=1-xi-eta-zeta
# ----------------------------
def tet10_shape_functions(xi, eta, zeta):
    L2, L3, L4 = xi, eta, zeta
    L1 = 1.0 - xi - eta - zeta
    N = np.zeros(10)
    # vertices
    N[0] = L1*(2*L1-1)
    N[1] = L2*(2*L2-1)
    N[2] = L3*(2*L3-1)
    N[3] = L4*(2*L4-1)
    # edge mids
    N[4] = 4*L1*L2
    N[5] = 4*L2*L3
    N[6] = 4*L1*L3
    N[7] = 4*L1*L4
    N[8] = 4*L2*L4
    N[9] = 4*L3*L4
    return N


def tet10_shape_function_gradients(xi, eta, zeta):
    L2, L3, L4 = xi, eta, zeta
    L1 = 1.0 - xi - eta - zeta
    L = [L1, L2, L3, L4]

    # dL/d(xi,eta,zeta)
    dL = np.array([
        [-1, -1, -1],  # dL1
        [1,  0,  0],  # dL2
        [0,  1,  0],  # dL3
        [0,  0,  1],  # dL4
    ])

    N = np.zeros(10)
    dN_dxi = np.zeros((10, 3))

    # vertices
    for i in range(4):
        N[i] = L[i]*(2*L[i]-1)
        dN_dxi[i, :] = (4*L[i]-1)*dL[i, :]

    # edges: (1-2),(2-3),(1-3),(1-4),(2-4),(3-4) in 0-based L indices
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    for k, (i, j) in enumerate(edges, start=4):
        N[k] = 4*L[i]*L[j]
        dN_dxi[k, :] = 4*(L[i]*dL[j, :] + L[j]*dL[i, :])

    return N, dN_dxi


# ----------------------------
# Precompute reference (grad_N, detJ, weight) for internal force integration
# ----------------------------
def tet10_precompute_reference(X_nodes):
    pts_xyz, _, w = tet5pt_quadrature()
    out = []
    for q, (xi, eta, zeta) in enumerate(pts_xyz):
        N, dN_dxi = tet10_shape_function_gradients(xi, eta, zeta)

        # Build reference Jacobian J = sum_a X_a ⊗ (∂N_a/∂ξ)
        J = np.zeros((3, 3))
        for a in range(10):
            J += np.outer(X_nodes[a], dN_dxi[a])

        detJ = np.linalg.det(J)

        # Map gradients to reference space: grad_N = J^{-T} * dN_dxi
        grad_N = np.zeros((10, 3))
        JT = J.T
        for a in range(10):
            grad_N[a, :] = np.linalg.solve(JT, dN_dxi[a])

        out.append({"N": N, "grad_N": grad_N, "detJ": detJ, "w": w[q]})
    return out


# ----------------------------
# Internal force (Total-Lagrangian, StVK)
# ----------------------------
def tet10_internal_force(x_nodes, pre, lam, mu):
    f = np.zeros((10, 3))
    for q in pre:
        grad_N = q["grad_N"]
        detJ = q["detJ"]
        wq = q["w"]

        # deformation gradient
        F = np.zeros((3, 3))
        for a in range(10):
            F += np.outer(x_nodes[a], grad_N[a])

        FtF = F.T @ F
        trFtF = np.trace(FtF)
        P = lam*(0.5*trFtF - 1.5)*F + mu*(F @ F.T @ F - F)

        dV = detJ * wq
        for a in range(10):
            f[a] += (P @ grad_N[a]) * dV
    return f

def assemble_external_force(num_nodes, force_node, F_node):
    """
    Assemble external point force at one node into the global RHS vector.

    Parameters:
      num_nodes  : number of nodes (10 for TET10)
      force_node : node index (0-based) where force is applied
      F_node     : np.array(3,), [Fx, Fy, Fz] external load

    Returns:
      f_ext : global external force vector (3*num_nodes,)
    """
    f_ext = np.zeros(3 * num_nodes)
    i = 3 * force_node
    f_ext[i:i+3] = F_node
    return f_ext



# ----------------------------
# Convenience: deformation gradients at each QP
# ----------------------------
def tet10_F_at_qpoints(x_nodes, pre):
    """Return list of F (3x3) at each quadrature point."""
    Fs = []
    for q in pre:
        grad_N = q["grad_N"]
        F = np.zeros((3, 3))
        for a in range(10):
            F += np.outer(x_nodes[a], grad_N[a])
        Fs.append(F)
    return Fs


# ----------------------------
# Volume and Mass (use |detJ| for volume/mass robustness)
# ----------------------------
def tet10_volume(X_nodes):
    pts_xyz, _, w = tet5pt_quadrature()
    V = 0.0
    for (xi, eta, zeta), wq in zip(pts_xyz, w):
        _, dN_dxi = tet10_shape_function_gradients(xi, eta, zeta)
        J = np.zeros((3, 3))
        for a in range(10):
            J += np.outer(X_nodes[a], dN_dxi[a])
        V += abs(np.linalg.det(J)) * wq
    return V


def tet10_lumped_mass_chrono(X_nodes, rho):
    V = tet10_volume(X_nodes)
    nodal_mass = rho * V / 10.0
    M30 = np.zeros((30, 30))
    for a in range(10):
        i = 3*a
        M30[i:i+3, i:i+3] = nodal_mass * np.eye(3)
    return M30, V


def tet10_consistent_mass(X_nodes, rho):
    """
    Compute consistent mass matrix for TET10 element.
    Returns:
      M30 (30x30) DOF mass matrix
      vol (float) element volume
    """
    pts_xyz, _, w = tet5pt_quadrature()
    Msc = np.zeros((10, 10))
    vol = 0.0

    for (xi, eta, zeta), wq in zip(pts_xyz, w):
        N, dN_dxi = tet10_shape_function_gradients(xi, eta, zeta)

        # Jacobian for volume
        J = np.zeros((3, 3))
        for a in range(10):
            J += np.outer(X_nodes[a], dN_dxi[a])

        dV = abs(np.linalg.det(J)) * wq
        Msc += rho * np.outer(N, N) * dV
        vol += dV

    # Expand to 30x30 with 3 DOFs/node
    M30 = np.zeros((30, 30))
    for i in range(10):
        for j in range(10):
            block = Msc[i, j] * np.eye(3)
            M30[3*i:3*i+3, 3*j:3*j+3] = block

    return M30, vol


# ----------------------------
# Simple geometry: unit-ish TET10 (scaled 0.1)
# ----------------------------
def make_unit_tet10():
    Xv = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
    ])
    mids = []
    for (i, j) in [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]:
        mids.append(0.5*(Xv[i]+Xv[j]))
    Xe = np.vstack([Xv, np.array(mids)])  # (10,3)
    base_vertex_idx = [0, 1, 2]
    base_mid_idx = [4, 5, 6]
    tip_idx = 3
    fixed_nodes = base_vertex_idx + base_mid_idx
    return Xe, tip_idx, fixed_nodes


def constraint(q):
    """
    Constrain nodes 1, 2, and 4 to their initial reference positions.
    Each node contributes 3 constraints (x,y,z).
    """
    c = np.zeros(9)

    # Node 1 (index 0)
    # fixed at (0,0,0)
    c[0] = q[0] - 0.0
    c[1] = q[1] - 0.0
    c[2] = q[2] - 0.0

    # Node 2 (index 1)
    # fixed at (0.1,0,0)
    c[3] = q[3] - 0.1
    c[4] = q[4] - 0.0
    c[5] = q[5] - 0.0

    # Node 4 (index 3)
    # fixed at (0,0,0.1)
    c[6] = q[9]  - 0.0
    c[7] = q[10] - 0.0
    c[8] = q[11] - 0.1

    return c


def constraint_jacobian(q):
    """
    Jacobian of constraints w.r.t q.
    9 constraints × len(q) DOFs.
    """
    J = np.zeros((9, len(q)))

    # Node 1
    J[0, 0] = 1
    J[1, 1] = 1
    J[2, 2] = 1

    # Node 2
    J[3, 3] = 1
    J[4, 4] = 1
    J[5, 5] = 1

    # Node 4
    J[6, 9]  = 1
    J[7, 10] = 1
    J[8, 11] = 1

    return J

if __name__ == "__main__":
    X_nodes, tip_idx, fixed_nodes = make_unit_tet10()
    x_nodes = X_nodes.copy()

    # --- Precompute reference data for internal forces / F ---
    pre = tet10_precompute_reference(X_nodes)

    # --- Deformation gradients at initial configuration ---
    Fs = tet10_F_at_qpoints(x_nodes, pre)
    print("\nDeformation gradient F at each quadrature point (initial config):")
    for k, F in enumerate(Fs):
        print(f"QP {k}:")
        print(F)
        print("||F - I||_F =", np.linalg.norm(F - np.eye(3)))
        print("-"*40)

    # --- Internal force at initial configuration ---
    f0 = tet10_internal_force(x_nodes, pre, lam, mu)
    print("\nInternal force at initial configuration (should be ~0):")
    print(f0)
    print("Max nodal force norm:", np.linalg.norm(f0, axis=1).max())

    # --- Chrono-style equal-share (30×30) ---
    #M30_chrono, V2 = tet10_lumped_mass_chrono(X_nodes, rho0)
    M30_consistent, V1 = tet10_consistent_mass(X_nodes, rho0)
    #print("\nChrono-style equal-share lumped mass matrix (30x30):")
    #print(M30_chrono)

    print("\nConsistent mass matrix (30x30):")
    print(M30_consistent)

    # --- Sanity checks ---
    #print("trace(M30_chrono) =", np.trace(M30_chrono))
    #print("Total mass (chrono)  =", M30_chrono.sum())

    print("trace(M30_consistent) =", np.trace(M30_consistent))
    print("Total mass (consistent)  =", M30_consistent.sum())


    time_step = 1e-3
    rho_bb = 1e14  # Augmented Lagrangian penalty parameter

    # --- Build mass matrix ---
    M30_consistent, V = tet10_consistent_mass(X_nodes, rho0)

    # Flatten mass to (30,30) DOF form
    M_full = M30_consistent.copy()

    # --- Internal & external force ---
    f_int0 = tet10_internal_force(x_nodes, pre, lam, mu).flatten()
    F_node = np.array([0.0, -1000.0, 0.0])  # Force at tip node
    f_ext0 = assemble_external_force(10, 2, F_node)

    # --- ALM+Nesterov wrapper ---
    def alm_nesterov_step(v_guess, lam_guess, v_prev, q_prev, M, f_int_func, f_ext, h, rho_bb):
        v = v_guess.copy()
        lam = lam_guess.copy()

        max_outer = 5
        max_inner = 500
        alpha = 1e-8  # fixed step size

        for outer_iter in range(max_outer):
            def grad_L(v_loc):
                qA = q_prev + h * v_loc
                x_new = qA[0::3]; y_new = qA[1::3]; z_new = qA[2::3]
                f_int_dyn = f_int_func(np.column_stack([x_new,y_new,z_new]))
                g_mech = (M @ (v_loc - v_prev)) / h - (-f_int_dyn.flatten() + f_ext)
                J = constraint_jacobian(qA)
                cA = constraint(qA)
                return g_mech + J.T @ (lam + rho_bb*h*cA)

            v_k   = v.copy()
            v_km1 = v.copy()
            t     = 1.0

            v_k_prev = None
            for inner_iter in range(max_inner):
                t_next = 0.5*(1+np.sqrt(1+4*t*t))
                beta   = (t-1)/t_next
                y = v_k + beta*(v_k - v_km1)
                g = grad_L(y)
                print("outer", outer_iter, "inner", inner_iter, "||g|| =", np.linalg.norm(g))
                v_next = y - alpha*g
                
                # Check convergence (skip first iteration)
                if v_k_prev is not None and np.linalg.norm(v_next - v_k_prev) < 1e-6:
                    v_k = v_next
                    print(f"  -> Converged after {inner_iter+1} iterations")
                    break
                    
                v_k_prev = v_k.copy()
                v_km1, v_k, t = v_k, v_next, t_next

            v = v_k
            qA = q_prev + h*v
            lam += rho_bb*h*constraint(qA)
            if np.linalg.norm(constraint(qA)) < 1e-6:
                break
        return v, lam

    # --- Simulation loop ---
    Nt = 10
    q_prev = x_nodes.flatten()
    v_prev = np.zeros_like(q_prev)
    v_guess = v_prev.copy()
    lam_guess = np.zeros(constraint(q_prev).shape)

    np.set_printoptions(suppress=True, precision=8)
    
    for step in range(Nt):
        v_res, lam_res = alm_nesterov_step(v_guess, lam_guess, v_prev, q_prev, 
                                        M_full, lambda x: tet10_internal_force(x, pre, lam, mu),
                                        f_ext0, time_step, rho_bb=1e14)

        v_guess, lam_guess = v_res.copy(), lam_res.copy()
        q_new = q_prev + time_step * v_guess

        # Update nodal positions
        for i in range(10):
            x_nodes[i] = q_new[3*i:3*i+3]

        # Print position of node 2
        print(f"Step {step}: Node 2 position = {x_nodes[2]}")


        q_prev = q_new.copy()
        v_prev = v_guess.copy()
