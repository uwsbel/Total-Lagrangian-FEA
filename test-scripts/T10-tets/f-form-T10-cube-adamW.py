import numpy as np
import matplotlib.pyplot as plt
from tet_mesh_reader import read_node, read_ele

# ----------------------------
# Material
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
    w = np.array([-4.0/5.0, 9.0/20.0, 9.0/20.0,
                 9.0/20.0, 9.0/20.0]) * (1.0/6.0)
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
            J += np.outer(X_elem_nodes[a], dN_dxi[a])  # <-- FIXED HERE
        detJ = np.linalg.det(J)
        grad_N = np.zeros((10, 3))
        JT = J.T
        for a in range(10):
            grad_N[a, :] = np.linalg.solve(JT, dN_dxi[a])
        pre.append({"grad_N": grad_N, "detJ": detJ, "w": wq})
    return pre


def tet10_precompute_reference_mesh(X_nodes, X_elem):
    """
    Precompute reference gradients for all elements in the mesh.

    Args:
        X_nodes: (n_nodes, 3) array of node coordinates.
        X_elem: (n_elements, 10) array of element connectivity (node indices).

    Returns:
        pre_list: list of precomputed reference data for each element.
    """
    pre_list = []
    for elem_idx in range(X_elem.shape[0]):
        # indices of the 10 nodes for this element
        node_indices = X_elem[elem_idx]
        X_elem_nodes = X_nodes[node_indices]      # shape: (10, 3)
        print("=====")
        print(X_elem_nodes)
        pre = tet10_precompute_reference(X_elem_nodes)
        pre_list.append(pre)
    return pre_list


# ----------------------------
# Internal force
# ----------------------------
def tet10_internal_force(x_nodes, pre, lam, mu):
    f = np.zeros((10, 3))
    for q in pre:
        grad_N = q["grad_N"]
        detJ = q["detJ"]
        wq = q["w"]
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


def tet10_internal_force_mesh(x_nodes, X_elem, pre_mesh, lam, mu):
    """
    Assemble the global internal force vector for the mesh.

    Args:
        x_nodes: (n_nodes, 3) current node positions
        X_elem: (n_elements, 10) element connectivity
        pre_mesh: list of precomputed reference data for each element
        lam, mu: material parameters

    Returns:
        f_int: (3*n_nodes,) global internal force vector
    """
    n_nodes = x_nodes.shape[0]
    f_int = np.zeros((n_nodes, 3))
    for elem_idx in range(X_elem.shape[0]):
        node_indices = X_elem[elem_idx]           # (10,)
        x_elem_nodes = x_nodes[node_indices]      # (10, 3)
        # reference gradients for this element
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
    """
    Assemble the global consistent mass matrix for a TET10 mesh.

    Args:
        X_nodes: (n_nodes, 3) array of node coordinates.
        X_elem: (n_elements, 10) array of element connectivity (node indices).
        rho: material density.

    Returns:
        M_full: (3*n_nodes, 3*n_nodes) global mass matrix.
    """
    n_nodes = X_nodes.shape[0]
    n_elements = X_elem.shape[0]
    M_full = np.zeros((3*n_nodes, 3*n_nodes))

    for elem_idx in range(n_elements):
        node_indices = X_elem[elem_idx]  # (10,)
        X_elem_nodes = X_nodes[node_indices]  # (10, 3)
        M_elem = tet10_consistent_mass(X_elem_nodes, rho)  # (30, 30)

        # Map local element DOFs to global DOFs
        global_dof_indices = []
        for ni in node_indices:
            # x, y, z for each node
            global_dof_indices.extend([3*ni, 3*ni+1, 3*ni+2])

        # Assemble
        for i_local, i_global in enumerate(global_dof_indices):
            for j_local, j_global in enumerate(global_dof_indices):
                M_full[i_global, j_global] += M_elem[i_local, j_local]

    return M_full


# ----------------------------
# Geometry
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
    Xe = np.vstack([Xv, mids])
    return Xe

# Constraints


def get_fixed_nodes(X_nodes):
    # Returns indices of nodes with z == 0
    return np.where(np.isclose(X_nodes[:, 2], 0.0))[0]


def constraint(q):
    # Use global X_nodes
    fixed_nodes = get_fixed_nodes(X_nodes)
    c = np.zeros(3 * len(fixed_nodes))
    for idx, node in enumerate(fixed_nodes):
        c[3*idx:3*idx+3] = q[3*node:3*node+3] - X_nodes[node]
    return c


def constraint_jacobian(q):
    fixed_nodes = get_fixed_nodes(X_nodes)
    J = np.zeros((3 * len(fixed_nodes), len(q)))
    for idx, node in enumerate(fixed_nodes):
        J[3*idx,   3*node] = 1
        J[3*idx+1, 3*node+1] = 1
        J[3*idx+2, 3*node+2] = 1
    return J


# ----------------------------
# ALM + AdamW solver
# ----------------------------
def alm_adamw_step(v_guess, lam_guess, v_prev, q_prev, M, f_int_func, f_ext, h, rho_bb, X_elem, pre_mesh, lam, mu):
    v = v_guess.copy()
    lam_mult = lam_guess.copy()
    n_nodes = M.shape[0] // 3

    max_outer = 5
    max_inner = 500
    lr = 2e-4
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    weight_decay = 1e-4
    inner_tol = 1e-1
    outer_tol = 1e-6

    actual_outer_iters = 0
    total_inner_iters = 0

    for outer_iter in range(max_outer):
        actual_outer_iters += 1

        def grad_L(v_loc):
            qA = q_prev + h*v_loc
            x_new = qA.reshape(n_nodes, 3)
            f_int_dyn = f_int_func(x_new, X_elem, pre_mesh, lam, mu)
            g_mech = (M @ (v_loc - v_prev)) / h - (-f_int_dyn + f_ext)
            J = constraint_jacobian(qA)
            cA = constraint(qA)
            return g_mech + J.T @ (lam_mult + rho_bb*h*cA)

        m_t = np.zeros_like(v)
        v_t = np.zeros_like(v)
        t = 0
        v_curr = v.copy()

        for inner_iter in range(max_inner):
            t += 1
            total_inner_iters += 1
            g = grad_L(v_curr)
            m_t = beta1*m_t + (1-beta1)*g
            v_t = beta2*v_t + (1-beta2)*(g*g)
            m_hat = m_t / (1 - beta1**t)
            v_hat = v_t / (1 - beta2**t)
            v_curr -= lr * (m_hat / (np.sqrt(v_hat) + eps) +
                            weight_decay*v_curr)

            gnorm = np.linalg.norm(g)
            if gnorm <= inner_tol*(1+np.linalg.norm(v_curr)):
                print(f"[inner {inner_iter}] ||g||={gnorm:.3e} (stop)")
                break
            if inner_iter % 10 == 0:
                print(f"[inner {inner_iter}] ||g||={gnorm:.3e}")

        v = v_curr
        qA = q_prev + h*v
        cA = constraint(qA)
        lam_mult += rho_bb*h*cA
        print(f">>>>> OUTER {outer_iter}: ||c||={np.linalg.norm(cA):.3e}")
        if np.linalg.norm(cA) < outer_tol:
            break

    return v, lam_mult, actual_outer_iters, total_inner_iters


# ----------------------------
# Main simulation
# ----------------------------
if __name__ == "__main__":
    # shape: (n_nodes, 3)
    X_nodes = read_node("../../data/meshes/T10/cube.1.node")
    x_nodes = X_nodes.copy()
    X_elem = read_ele("../../data/meshes/T10/cube.1.ele")

    print(X_nodes)
    print(X_elem)
    pre_mesh = tet10_precompute_reference_mesh(X_nodes, X_elem)

    # print the shape of pre_mesh
    print("shape pre_mesh:", len(pre_mesh))
    print("shape pre_mesh[0]:", len(pre_mesh[0]))

    print(pre_mesh)

    import numpy as np
    np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)

    M_full = tet10_consistent_mass_mesh(X_nodes, X_elem, rho0)
    print("shape M_full:", M_full.shape)
    print(M_full)

    f_ext = np.zeros(3 * X_nodes.shape[0])
    f_ext[3*6 + 0] = 1000.0  # 1000 N force in x direction at node index 6
    time_step = 1e-3
    rho_bb = 1e14

    q_prev = x_nodes.flatten()
    v_prev = np.zeros_like(q_prev)
    v_guess = v_prev.copy()
    lam_guess = np.zeros(3 * len(get_fixed_nodes(X_nodes)))

    Nt = 100
    node6_x = []  # List to store y position of node index 6
    node8_x = []  # List to store x position of node index 8
    outer_iters_per_step = []  # List to store outer iterations for each step
    inner_iters_per_step = []  # List to store inner iterations for each step

    for step in range(Nt):
        if step > 50:
            f_ext[3*6 + 0] = 0.0  # Remove force after step 15
        # else:
        #    f_ext[3*6 + 0] = 1000.0
        v_res, lam_res, outer_iters, inner_iters = alm_adamw_step(
            v_guess, lam_guess, v_prev, q_prev, M_full,
            tet10_internal_force_mesh, f_ext, time_step, rho_bb,
            X_elem, pre_mesh, lam, mu)

        v_guess, lam_guess = v_res.copy(), lam_res.copy()
        q_new = q_prev + time_step * v_guess
        x_nodes = q_new.reshape(X_nodes.shape[0], 3)
        print(
            f"Step {step}: node 6 position = {x_nodes[6]}, node 8 position = {x_nodes[8]}")
        print(
            f"Step {step}: outer iters = {outer_iters}, inner iters = {inner_iters}")
        node6_x.append(x_nodes[6, 0])  # Save node 6 x position
        node8_x.append(x_nodes[8, 0])  # Save node 8 x position
        outer_iters_per_step.append(outer_iters)  # Save outer iterations
        inner_iters_per_step.append(inner_iters)  # Save inner iterations
        q_prev = q_new.copy()
        v_prev = v_guess.copy()

    # Plot after simulation

    # Create subplots for better visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Node positions plots
    ax1.plot(range(Nt), node6_x, marker='o', linewidth=2, markersize=4)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Node 6 X Position")
    ax1.set_title("Node 6 X Position vs Step")
    ax1.grid(True)

    ax2.plot(range(Nt), node8_x, marker='x',
             color='orange', linewidth=2, markersize=4)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Node 8 X Position")
    ax2.set_title("Node 8 X Position vs Step")
    ax2.grid(True)

    # Iteration count plots
    ax3.plot(range(Nt), outer_iters_per_step, marker='s',
             color='red', linewidth=2, markersize=4)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Outer Iterations")
    ax3.set_title("Outer Loop Iterations per Step")
    ax3.grid(True)
    ax3.set_ylim(bottom=0)

    ax4.plot(range(Nt), inner_iters_per_step, marker='^',
             color='green', linewidth=2, markersize=4)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Inner Iterations")
    ax4.set_title("Inner Loop Iterations per Step")
    ax4.grid(True)
    ax4.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "="*50)
    print("ITERATION SUMMARY STATISTICS")
    print("="*50)
    print(f"Total outer iterations: {sum(outer_iters_per_step)}")
    print(f"Total inner iterations: {sum(inner_iters_per_step)}")
    print(
        f"Average outer iterations per step: {np.mean(outer_iters_per_step):.2f}")
    print(
        f"Average inner iterations per step: {np.mean(inner_iters_per_step):.2f}")
    print(f"Max outer iterations in a step: {max(outer_iters_per_step)}")
    print(f"Max inner iterations in a step: {max(inner_iters_per_step)}")
    print(f"Min outer iterations in a step: {min(outer_iters_per_step)}")
    print(f"Min inner iterations in a step: {min(inner_iters_per_step)}")

    # Additional plot showing cumulative iterations
    plt.figure(figsize=(10, 6))
    cumulative_outer = np.cumsum(outer_iters_per_step)
    cumulative_inner = np.cumsum(inner_iters_per_step)

    plt.plot(range(Nt), cumulative_outer,
             label='Cumulative Outer Iterations', linewidth=2)
    plt.plot(range(Nt), cumulative_inner,
             label='Cumulative Inner Iterations', linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Iterations")
    plt.title("Cumulative Iteration Count")
    plt.legend()
    plt.grid(True)
    plt.show()
