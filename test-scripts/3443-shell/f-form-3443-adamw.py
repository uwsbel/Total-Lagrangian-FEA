import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# note that we have a very big assumption that detJ is the same for all elements, we only precompute one value
# in the correct case, we shall have one detJ for each element.

uvw = np.array([
    [-1.0, -0.5, -0.5],
    [-1.0, -0.5,  0.5],
    [-1.0,  0.5, -0.5],
    [-1.0,  0.5,  0.5],
    [-0.6, -0.5, -0.5],
    [-0.6, -0.5,  0.5],
    [-0.6,  0.5, -0.5],
    [-0.6,  0.5,  0.5],
    [-0.2, -0.5, -0.5],
    [-0.2, -0.5,  0.5],
    [-0.2,  0.5, -0.5],
    [-0.2,  0.5,  0.5],
    [0.2, -0.5, -0.5],
    [0.2, -0.5,  0.5],
    [0.2,  0.5, -0.5],
    [0.2,  0.5,  0.5],
    [0.6, -0.5, -0.5],
    [0.6, -0.5,  0.5],
    [0.6,  0.5, -0.5],
    [0.6,  0.5,  0.5],
    [1.0, -0.5, -0.5],
    [1.0, -0.5,  0.5],
    [1.0,  0.5, -0.5],
    [1.0,  0.5,  0.5]
])

# 7-point Gauss-Legendre quadrature (symmetric) for mass assembly
gauss_xi_m = np.array([
    -0.949107912342759,
    -0.741531185599394,
    -0.405845151377397,
    0.0,
    0.405845151377397,
    0.741531185599394,
    0.949107912342759
])

weight_xi_m = np.array([
    0.129484966168870,
    0.279705391489277,
    0.381830050505119,
    0.417959183673469,
    0.381830050505119,
    0.279705391489277,
    0.129484966168870
])

gauss_eta_m = gauss_xi_m.copy()
weight_eta_m = weight_xi_m.copy()

# 3-point Gauss-Legendre quadrature (zeta) for mass assembly
gauss_zeta_m = np.array([
    -np.sqrt(3/5), 0.0, np.sqrt(3/5)
])
weight_zeta_m = np.array([
    5/9, 8/9, 5/9
])

# 4-point Gauss-Legendre quadrature for internal forces
gauss_xi = np.array([
    -0.861136311594053,
    -0.339981043584856,
    0.339981043584856,
    0.861136311594053
])
weight_xi = np.array([
    0.347854845137454,
    0.652145154862546,
    0.652145154862546,
    0.347854845137454
])

gauss_eta = gauss_xi.copy()
weight_eta = weight_xi.copy()

# 3-point Gauss-Legendre quadrature (zeta) for internal forces
gauss_zeta = np.array([
    -np.sqrt(3/5), 0.0, np.sqrt(3/5)
])
weight_zeta = np.array([
    5/9, 8/9, 5/9
])

E = 7e8
nu = 0.33
rho0 = 2700

mu = E / (2 * (1 + nu))              # Shear modulus μ
lam_param = (E * nu) / ((1 + nu)*(1 - 2*nu))  # Lamé’s first parameter λ

H = 1.0  # Height
W = 1.0  # Width
L = 2.0  # Length

n_shell = 2  # Number of shell elements

# Define element connectivity structure (4 nodes per element, each node contributes 4 coefficients)
element_connectivity = np.array([
    [0, 1, 2, 3],  # Element 0: nodes 0, 1, 2, 3
    [1, 4, 5, 2]   # Element 1: nodes 1, 4, 5, 2
])


def get_element_nodes(element_id):
    if element_id < len(element_connectivity):
        return element_connectivity[element_id]
    else:
        raise ValueError(f"Element {element_id} does not exist")


# Each vector is 24 x 1 (6 nodes × 4 coefficients per node)
x12 = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0,
               0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0])
y12 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
               0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
z12 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
               1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

x12_ref = x12.copy()
y12_ref = y12.copy()
z12_ref = z12.copy()

# Velocities (all zero)
x12_dot = np.zeros_like(x12)
y12_dot = np.zeros_like(y12)
z12_dot = np.zeros_like(z12)

time_step = 1e-3


def b_vec(u, v, w):
    return np.array([
        1,
        u,
        v,
        w,
        u*v,
        u*w,
        v*w,
        u*v*w,
        u**2,
        v**2,
        (u**2)*v,
        u*(v**2),
        u**3,
        v**3,
        (u**3)*v,
        u*(v**3)
    ])


def b_vec_xi(xi, eta, zeta, L, W, H):
    u = L * xi / 2.0
    v = W * eta / 2.0
    w = H * zeta / 2.0
    return b_vec(u, v, w)

# Chain-rule to parametric space (xi, eta, zeta)


def db_dxi(xi, eta, zeta, L, W, H):
    u = 0.5*L*xi
    v = 0.5*W*eta
    w = 0.5*H*zeta
    return 0.5*L * db_du(u, v, w)


def db_deta(xi, eta, zeta, L, W, H):
    u = 0.5*L*xi
    v = 0.5*W*eta
    w = 0.5*H*zeta
    return 0.5*W * db_dv(u, v, w)


def db_dzeta(xi, eta, zeta, L, W, H):
    u = 0.5*L*xi
    v = 0.5*W*eta
    w = 0.5*H*zeta
    return 0.5*H * db_dw(u, v, w)


def db_du(u, v, w):
    return np.array([
        0.0,
        1.0,
        0.0,
        0.0,
        v,
        w,
        0.0,
        v*w,
        2.0*u,
        0.0,
        2.0*u*v,
        v**2,
        3.0*u**2,
        0.0,
        3.0*u**2*v,
        v**3
    ])


def db_dv(u, v, w):
    return np.array([
        0.0,
        0.0,
        1.0,
        0.0,
        u,
        0.0,
        w,
        u*w,
        0.0,
        2.0*v,
        u**2,
        2.0*u*v,
        0.0,
        3.0*v**2,
        u**3,
        3.0*u*v**2
    ])


def db_dw(u, v, w):
    return np.array([
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        u,
        v,
        u*v,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ])


def B12_matrix(L, W, H):
    # Reference coordinates of 4 corner points on the mid-surface (w=0)
    u1, v1, w1 = -L/2.0, -W/2.0, 0.0
    u2, v2, w2 = L/2.0, -W/2.0, 0.0
    u3, v3, w3 = L/2.0,  W/2.0, 0.0
    u4, v4, w4 = -L/2.0,  W/2.0, 0.0

    b1 = b_vec(u1, v1, w1)
    b2 = b_vec(u2, v2, w2)
    b3 = b_vec(u3, v3, w3)
    b4 = b_vec(u4, v4, w4)

    B = np.vstack([
        b1,
        db_du(u1, v1, w1),
        db_dv(u1, v1, w1),
        db_dw(u1, v1, w1),
        b2,
        db_du(u2, v2, w2),
        db_dv(u2, v2, w2),
        db_dw(u2, v2, w2),
        b3,
        db_du(u3, v3, w3),
        db_dv(u3, v3, w3),
        db_dw(u3, v3, w3),
        b4,
        db_du(u4, v4, w4),
        db_dv(u4, v4, w4),
        db_dw(u4, v4, w4)
    ]).T  # 16x16
    return B


B = B12_matrix(L, W, H)
print("rank(B) =", np.linalg.matrix_rank(B))
B_inv = np.linalg.inv(B)


def constraint(q):
    c = np.zeros(24)
    c[0] = q[0] - 0.0
    c[1] = q[1] - 0.0
    c[2] = q[2] - 0.0

    c[3] = q[3] - 1.0
    c[4] = q[4] - 0.0
    c[5] = q[5] - 0.0

    c[6] = q[6] - 0.0
    c[7] = q[7] - 1.0
    c[8] = q[8] - 0.0

    c[9] = q[9] - 0.0
    c[10] = q[10] - 0.0
    c[11] = q[11] - 1.0

    c[12] = q[36] - 0.0
    c[13] = q[37] - 1.0
    c[14] = q[38] - 0.0

    c[15] = q[39] - 1.0
    c[16] = q[40] - 0.0
    c[17] = q[41] - 0.0

    c[18] = q[42] - 0.0
    c[19] = q[43] - 1.0
    c[20] = q[44] - 0.0

    c[21] = q[45] - 0.0
    c[22] = q[46] - 0.0
    c[23] = q[47] - 1.0
    return c


def constraint_jacobian(q):
    constrained_indices = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
    ]
    J = np.zeros((len(constrained_indices), len(q)))
    for i, idx in enumerate(constrained_indices):
        J[i, idx] = 1.0
    return J


def ds_du_mat(u, v, w, B_inv):
    """Returns 16×3 matrix: each row is ∇s_i w.r.t [u, v, w]"""
    db_du_mat = db_du(u, v, w)
    db_dv_mat = db_dv(u, v, w)
    db_dw_mat = db_dw(u, v, w)
    ds_du = B_inv @ db_du_mat
    ds_dv = B_inv @ db_dv_mat
    ds_dw = B_inv @ db_dw_mat
    ds = np.stack([ds_du, ds_dv, ds_dw], axis=-1)  # (16,3)
    return ds


def calc_det_J(u, v, w, B_inv, x12_jac, y12_jac, z12_jac):
    ds_du = B_inv @ db_du(u, v, w)
    ds_dv = B_inv @ db_dv(u, v, w)
    ds_dw = B_inv @ db_dw(u, v, w)
    # use first 16 coeffs for Jacobian (reference)
    x12_jac_cut = x12_ref[:16]
    y12_jac_cut = y12_ref[:16]
    z12_jac_cut = z12_ref[:16]
    N_mat_jac = np.vstack([x12_jac_cut, y12_jac_cut, z12_jac_cut])
    J = N_mat_jac @ np.column_stack([ds_du, ds_dv, ds_dw])
    return J


def calc_det_J_xi(xi, eta, zeta, B_inv, x12_jac, y12_jac, z12_jac, L, W, H, idx):
    ds_dxi = B_inv @ db_dxi(xi, eta, zeta, L, W, H)
    ds_deta = B_inv @ db_deta(xi, eta, zeta, L, W, H)
    ds_dzeta = B_inv @ db_dzeta(xi, eta, zeta, L, W, H)
    N_mat_jac = np.vstack([x12_jac, y12_jac, z12_jac])
    J = N_mat_jac @ np.column_stack([ds_dxi, ds_deta, ds_dzeta])
    return J


# Global DOF count: 16 + 8*(n_shell-1)
N_coef = 16 + 8 * (n_shell - 1)
m = np.zeros((N_coef, N_coef))

# Assemble lumped-like mass (via s_i s_j)
for elem in range(n_shell):
    elem_idx = element_connectivity[elem]
    idx = np.concatenate([np.arange(node*4, (node+1)*4) for node in elem_idx])
    x_loc = np.concatenate([x12[node*4:(node+1)*4] for node in elem_idx])
    y_loc = np.concatenate([y12[node*4:(node+1)*4] for node in elem_idx])
    z_loc = np.concatenate([z12[node*4:(node+1)*4] for node in elem_idx])

    for ixi, xi in enumerate(gauss_xi_m):
        weight_u = weight_xi_m[ixi]
        for ieta, eta in enumerate(gauss_eta_m):
            weight_v = weight_eta_m[ieta]
            for izeta, zeta in enumerate(gauss_zeta_m):
                weight_w = weight_zeta_m[izeta]
                b = b_vec_xi(xi, eta, zeta, L, W, H)
                s = B_inv @ b
                J = calc_det_J_xi(xi, eta, zeta, B_inv, x_loc,
                                  y_loc, z_loc, L, W, H, idx)
                detJ = np.linalg.det(J)
                for i in range(16):
                    gi = idx[i]
                    for j in range(16):
                        gj = idx[j]
                        m[gi, gj] += (rho0 * s[i] * s[j] *
                                      weight_u * weight_v * weight_w * detJ)

print("Mass matrix shape:", m.shape)
print(pd.DataFrame(m).to_string(float_format="%.3f"))

Nt = 20  # Number of time steps

# Precompute ds/du at force quadrature points
ds_du_pre = {}
for ixi, xi in enumerate(gauss_xi):
    for ieta, eta in enumerate(gauss_eta):
        for izeta, zeta in enumerate(gauss_zeta):
            u = L * xi / 2
            v = W * eta / 2
            w = H * zeta / 2
            ds_du_pre[(ixi, ieta, izeta)] = ds_du_mat(u, v, w, B_inv)

# Precompute detJ at force quadrature points (reference)
detJ_pre = {}
for ixi, xi in enumerate(gauss_xi):
    for ieta, eta in enumerate(gauss_eta):
        for izeta, zeta in enumerate(gauss_zeta):
            u = L * xi / 2
            v = W * eta / 2
            w = H * zeta / 2
            J = calc_det_J(u, v, w, B_inv, x12_ref, y12_ref, z12_ref)
            detJ_pre[(ixi, ieta, izeta)] = np.linalg.det(J)

n_gen_coord = 3 * N_coef
n_constr = 24

v = np.zeros(n_gen_coord)
lam_bb = np.zeros(n_constr)
v_guess = v.copy()
lam_bb_guess = lam_bb.copy()
rho_bb = 1e14

endz_pos = []


def compute_deformation_gradient(e, grad_s):
    """F = sum_i e_i ⊗ ∇s_i; e: list of 16 (3,) vectors, grad_s: (16,3)"""
    F = np.zeros((3, 3))
    for i in range(16):
        F += np.outer(e[i], grad_s[i])
    return F


def compute_green_lagrange_strain(F):
    I = np.eye(3)
    C = F.T @ F
    E = 0.5 * (C - I)
    return E


def compute_internal_force(x12_cur, y12_cur, z12_cur):
    f_int = np.zeros((3 * N_coef,))
    for elem in range(n_shell):
        elem_idx = element_connectivity[elem]
        idx = np.concatenate([np.arange(node*4, (node+1)*4)
                             for node in elem_idx])
        x12_loc = np.concatenate([x12_cur[node*4:(node+1)*4]
                                 for node in elem_idx])
        y12_loc = np.concatenate([y12_cur[node*4:(node+1)*4]
                                 for node in elem_idx])
        z12_loc = np.concatenate([z12_cur[node*4:(node+1)*4]
                                 for node in elem_idx])
        e = [np.array([x12_loc[i], y12_loc[i], z12_loc[i]]) for i in range(16)]
        f_elem = np.zeros((16, 3))

        for ixi, xi in enumerate(gauss_xi):
            weight_u = weight_xi[ixi]
            for ieta, eta in enumerate(gauss_eta):
                weight_v = weight_eta[ieta]
                for izeta, zeta in enumerate(gauss_zeta):
                    weight_w = weight_zeta[izeta]
                    scale = weight_u * weight_v * weight_w
                    u = L * xi / 2
                    v = W * eta / 2
                    w = H * zeta / 2
                    ds = ds_du_pre[(ixi, ieta, izeta)]      # (16,3)
                    detJ = detJ_pre[(ixi, ieta, izeta)]
                    F = compute_deformation_gradient(e, ds)
                    FtF = F.T @ F
                    tr_FtF = np.trace(FtF)
                    FFF = F @ F.T @ F
                    stress_term = lam_param * \
                        (0.5 * tr_FtF - 1.5) * F + mu * (FFF - F)

                    for i in range(16):
                        grad_si = ds[i]
                        force_i = stress_term @ grad_si
                        f_elem[i] += force_i * scale * detJ * (L * W * H / 8.0)

        for i in range(16):
            gi = idx[i]
            f_int[3*gi + 0] += f_elem[i, 0]
            f_int[3*gi + 1] += f_elem[i, 1]
            f_int[3*gi + 2] += f_elem[i, 2]
    return f_int


def alm_adamw_step(v_guess, lam_guess, v_prev, q_prev, M, f_int_func, f_int, f_ext, h, rho):
    """ALM outer with AdamW inner, with verbose prints for outer/inner and ||g||."""
    def scaled_grad_stop(grad, x, tol_base):
        return np.linalg.norm(grad) <= tol_base * (1.0 + np.linalg.norm(x))

    v = v_guess.copy()
    lam = lam_guess.copy()

    max_outer = 5
    max_inner = 400
    inner_tol_base = 1e-2
    outer_tol = 1e-6

    # AdamW hyperparams
    lr = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.0   # keep 0 for velocity unless you want damping
    clip_norm = 1e6

    for outer_iter in range(max_outer):
        def grad_L(v_loc):
            qA = q_prev + h * v_loc
            x_new = qA[0::3]
            y_new = qA[1::3]
            z_new = qA[2::3]
            f_int_dyn = f_int_func(x_new, y_new, z_new)
            g_mech = (M @ (v_loc - v_prev)) / h - (-f_int_dyn + f_ext)
            J = constraint_jacobian(qA)
            cA = constraint(qA)
            return g_mech + J.T @ (lam + rho * h * cA)

        m_adam = np.zeros_like(v)
        v_adam = np.zeros_like(v)

        v_current = v.copy()
        for inner_iter in range(max_inner):
            lr = lr * 0.998
            g = grad_L(v_current)
            gnorm = np.linalg.norm(g)

            # print like your Nesterov version
            print(f"outer iter: {outer_iter:2d} inner iter: {inner_iter:3d}")
            print(f"g norm:  {gnorm}")

            if gnorm > clip_norm:
                g = g * (clip_norm / max(gnorm, 1e-12))

            m_adam = beta1 * m_adam + (1 - beta1) * g
            v_adam = beta2 * v_adam + (1 - beta2) * (g * g)
            t = inner_iter + 1
            m_hat = m_adam / (1 - beta1**t)
            v_hat = v_adam / (1 - beta2**t)

            # AdamW (decoupled)
            v_current = v_current - lr * \
                (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * v_current)

            if scaled_grad_stop(g, v_current, inner_tol_base):
                print(
                    f"[inner {inner_iter:3d}] scaled stop: ||∇L||={gnorm:.6e}")
                break

        v = v_current
        qA = q_prev + h * v
        cA = constraint(qA)
        lam += rho * h * cA
        print(
            f">>>>> End of OUTER STEP #{outer_iter}; ||c(qA)|| = {np.linalg.norm(cA):.6e}")

        if np.linalg.norm(cA) < outer_tol:
            break

    return v, lam


for step in range(Nt):
    # Internal forces at current state
    f_int = compute_internal_force(x12, y12, z12)

    # External load application point in physical coordinates
    u_P, v_P, w_P = 1.0, 0.0, 0.0
    b = b_vec(u_P, v_P, w_P)
    s_at_P = B_inv @ b

    # External force (z only in this example)
    if step <= 200:
        f_P = np.array([0.0, 0.0, 1000.0])
    else:
        f_P = np.array([0.0, 0.0, 0.0])

    # Distribute to the 16 local dofs of the LAST element
    elem = n_shell - 1
    # (fix: ensure elem_idx defined here)
    elem_idx = element_connectivity[elem]
    idx = np.concatenate([np.arange(node*4, (node+1)*4) for node in elem_idx])
    f_ext = [s_at_P[i] * f_P for i in range(16)]

    f_ext_vec = np.zeros((3 * N_coef,))
    for i_local, global_idx in enumerate(idx):
        row_idx = slice(3 * global_idx, 3 * (global_idx + 1))
        f_ext_vec[row_idx] = f_ext[i_local]

    # Expand mass matrix to xyz blocks
    M_full = np.zeros((3 * N_coef, 3 * N_coef))
    for i in range(N_coef):
        for j in range(N_coef):
            block = m[i, j] * np.eye(3)
            row_idx = slice(3 * i, 3 * (i + 1))
            col_idx = slice(3 * j, 3 * (j + 1))
            M_full[row_idx, col_idx] += block

    v_prev = v_guess.copy()
    q_prev = np.zeros((3 * N_coef,))
    for i in range(N_coef):
        q_prev[3 * i + 0] = x12[i]
        q_prev[3 * i + 1] = y12[i]
        q_prev[3 * i + 2] = z12[i]

    # === AdamW-based ALM inner solver ===
    v_res, lam_bb_res = alm_adamw_step(
        v_guess, lam_bb_guess, v_prev, q_prev, M_full,
        compute_internal_force, f_int, f_ext_vec, time_step, rho_bb
    )

    v_guess = v_res.copy()
    lam_bb_guess = lam_bb_res.copy()

    q_new = q_prev + time_step * v_guess
    for i in range(N_coef):
        x12[i] = q_new[3 * i + 0]
        y12[i] = q_new[3 * i + 1]
        z12[i] = q_new[3 * i + 2]

    # track some tip dof (example uses coefficient index 20)
    endz_pos.append(z12[20])

    print("x12: ")
    print(x12)
    print("y12: ")
    print(y12)
    print("z12: ")
    print(z12)

print(endz_pos)
