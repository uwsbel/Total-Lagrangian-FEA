import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Material
# ----------------------------
E = 7e8
nu = 0.33
mu = E / (2*(1+nu))
lam = E*nu/((1+nu)*(1-2*nu))
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
    edges = [(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]
    for k,(i,j) in enumerate(edges, start=4):
        dN_dxi[k,:] = 4*(L[i]*dL[j,:] + L[j]*dL[i,:])
    return dN_dxi


# ----------------------------
# Precompute reference gradients
# ----------------------------
def tet10_precompute_reference(X_nodes):
    pts_xyz, _, w = tet5pt_quadrature()
    pre = []
    for (xi, eta, zeta), wq in zip(pts_xyz, w):
        dN_dxi = tet10_shape_function_gradients(xi, eta, zeta)
        J = np.zeros((3, 3))
        for a in range(10):
            J += np.outer(X_nodes[a], dN_dxi[a])
        detJ = np.linalg.det(J)
        grad_N = np.zeros((10, 3))
        JT = J.T
        for a in range(10):
            grad_N[a,:] = np.linalg.solve(JT, dN_dxi[a])
        pre.append({"grad_N": grad_N, "detJ": detJ, "w": wq})
    return pre


# ----------------------------
# Internal force
# ----------------------------
def tet10_internal_force(x_nodes, pre, lam, mu):
    f = np.zeros((10, 3))
    for q in pre:
        grad_N = q["grad_N"]
        detJ = q["detJ"]
        wq = q["w"]
        F = np.zeros((3,3))
        for a in range(10):
            F += np.outer(x_nodes[a], grad_N[a])
        FtF = F.T @ F
        trFtF = np.trace(FtF)
        P = lam*(0.5*trFtF - 1.5)*F + mu*(F @ F.T @ F - F)
        dV = detJ * wq
        for a in range(10):
            f[a] += (P @ grad_N[a]) * dV
    return f


# ----------------------------
# Consistent mass
# ----------------------------
def tet10_consistent_mass(X_nodes, rho):
    pts_xyz, _, w = tet5pt_quadrature()
    Msc = np.zeros((10,10))
    for (xi,eta,zeta), wq in zip(pts_xyz, w):
        dN_dxi = tet10_shape_function_gradients(xi, eta, zeta)
        J = np.zeros((3,3))
        for a in range(10):
            J += np.outer(X_nodes[a], dN_dxi[a])
        detJ = abs(np.linalg.det(J))
        N = np.zeros(10)
        L2,L3,L4 = xi,eta,zeta
        L1 = 1-xi-eta-zeta
        L = [L1,L2,L3,L4]
        N[0:4] = [L[i]*(2*L[i]-1) for i in range(4)]
        edges = [(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]
        for k,(i,j) in enumerate(edges, start=4):
            N[k] = 4*L[i]*L[j]
        Msc += rho * np.outer(N,N) * detJ * wq
    M30 = np.zeros((30,30))
    for i in range(10):
        for j in range(10):
            M30[3*i:3*i+3, 3*j:3*j+3] = Msc[i,j]*np.eye(3)
    return M30


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
    for (i,j) in [(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]:
        mids.append(0.5*(Xv[i]+Xv[j]))
    Xe = np.vstack([Xv, mids])
    return Xe


# ----------------------------
# Constraints
# ----------------------------
def constraint(q):
    c = np.zeros(9)
    c[0:3] = q[0:3] - np.array([0.0,0.0,0.0])
    c[3:6] = q[3:6] - np.array([0.1,0.0,0.0])
    c[6:9] = q[9:12] - np.array([0.0,0.0,0.1])
    return c

def constraint_jacobian(q):
    J = np.zeros((9,len(q)))
    J[0,0]=J[1,1]=J[2,2]=1
    J[3,3]=J[4,4]=J[5,5]=1
    J[6,9]=J[7,10]=J[8,11]=1
    return J


# ----------------------------
# ALM + AdamW solver
# ----------------------------
def alm_adamw_step(v_guess, lam_guess, v_prev, q_prev, M, f_int_func, f_ext, h, rho_bb, pre, lam, mu):
    v = v_guess.copy()
    lam_mult = lam_guess.copy()

    max_outer = 5
    max_inner = 500
    lr = 1e-3
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    weight_decay = 1e-4
    inner_tol = 1e-2
    outer_tol = 1e-6

    for outer_iter in range(max_outer):

        def grad_L(v_loc):
            qA = q_prev + h*v_loc
            x_new = qA.reshape(10,3)
            f_int_dyn = f_int_func(x_new, pre, lam, mu)
            g_mech = (M @ (v_loc - v_prev)) / h - (-f_int_dyn.flatten() + f_ext)
            J = constraint_jacobian(qA)
            cA = constraint(qA)
            return g_mech + J.T @ (lam_mult + rho_bb*h*cA)

        m_t = np.zeros_like(v)
        v_t = np.zeros_like(v)
        t = 0
        v_curr = v.copy()

        for inner_iter in range(max_inner):
            t += 1
            g = grad_L(v_curr)
            m_t = beta1*m_t + (1-beta1)*g
            v_t = beta2*v_t + (1-beta2)*(g*g)
            m_hat = m_t / (1 - beta1**t)
            v_hat = v_t / (1 - beta2**t)
            v_curr -= lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay*v_curr)

            gnorm = np.linalg.norm(g)
            if gnorm <= inner_tol*(1+np.linalg.norm(v_curr)):
                print(f"[inner {inner_iter}] ||g||={gnorm:.3e} (stop)")
                break
            if inner_iter % 20 == 0:
                print(f"[inner {inner_iter}] ||g||={gnorm:.3e}")

        v = v_curr
        qA = q_prev + h*v
        cA = constraint(qA)
        lam_mult += rho_bb*h*cA
        print(f">>>>> OUTER {outer_iter}: ||c||={np.linalg.norm(cA):.3e}")
        if np.linalg.norm(cA) < outer_tol:
            break

    return v, lam_mult


# ----------------------------
# Main simulation
# ----------------------------
if __name__ == "__main__":
    X_nodes = make_unit_tet10()
    x_nodes = X_nodes.copy()
    pre = tet10_precompute_reference(X_nodes)
    M_full = tet10_consistent_mass(X_nodes, rho0)
    f_ext = np.zeros(30)
    f_ext[3*2 + 1] = -1000.0  # downward force at node 3 (index 2)
    time_step = 1e-3
    rho_bb = 1e14

    q_prev = x_nodes.flatten()
    v_prev = np.zeros_like(q_prev)
    v_guess = v_prev.copy()
    lam_guess = np.zeros(9)

    Nt = 30
    node2_y = []  # List to store y position of node 2

    for step in range(Nt):
        if step > 15 or step == 0:
            f_ext[3*2 + 1] = 0.0  # Remove force after step 15
        else:
            f_ext[3*2 + 1] = -1000.0
        v_res, lam_res = alm_adamw_step(v_guess, lam_guess, v_prev, q_prev, M_full,
                                        tet10_internal_force, f_ext, time_step, rho_bb,
                                        pre, lam, mu)

        v_guess, lam_guess = v_res.copy(), lam_res.copy()
        q_new = q_prev + time_step * v_guess
        x_nodes = q_new.reshape(10,3)
        print(f"Step {step}: node 2 position = {x_nodes[2]}")
        node2_y.append(x_nodes[2, 1])  # Save y position
        q_prev = q_new.copy()
        v_prev = v_guess.copy()

    # Plot after simulation
    plt.plot(range(Nt), node2_y, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Node 2 Y Position")
    plt.title("Node 2 Y Position vs Step")
    plt.grid(True)
    plt.show()
