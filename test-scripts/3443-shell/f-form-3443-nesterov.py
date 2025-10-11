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

# 4-point Gauss-Legendre quadrature in 1D

# 7-point Gauss-Legendre quadrature (symmetric)
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


# 7-point Gauss-Legendre quadrature (symmetric)
gauss_eta_m = np.array([
    -0.949107912342759,
    -0.741531185599394,
    -0.405845151377397,
    0.0,
    0.405845151377397,
    0.741531185599394,
    0.949107912342759
])

weight_eta_m = np.array([
    0.129484966168870,
    0.279705391489277,
    0.381830050505119,
    0.417959183673469,
    0.381830050505119,
    0.279705391489277,
    0.129484966168870
])


# 3-point Gauss-Legendre quadrature
gauss_zeta_m = np.array([
    -np.sqrt(3/5), 0.0, np.sqrt(3/5)
])

weight_zeta_m = np.array([
    5/9, 8/9, 5/9
])


# 4-point Gauss-Legendre quadrature
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

# 4-point Gauss-Legendre quadrature
gauss_eta = np.array([
    -0.861136311594053,
    -0.339981043584856,
    0.339981043584856,
    0.861136311594053
])

weight_eta = np.array([
    0.347854845137454,
    0.652145154862546,
    0.652145154862546,
    0.347854845137454
])

# 3-point Gauss-Legendre quadrature (for eta and zeta directions)
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

# Define element connectivity structure
# Each row represents an element, columns are the node indices
element_connectivity = np.array([
    [0, 1, 2, 3],  # Element 0: nodes 0, 1, 2, 3
    [1, 4, 5, 2]   # Element 1: nodes 1, 4, 5, 2
])


# Function to get nodes for a specific element
def get_element_nodes(element_id):
    """Get the node indices for a given element"""
    if element_id < len(element_connectivity):
        return element_connectivity[element_id]
    else:
        raise ValueError(f"Element {element_id} does not exist")

# ====================================================


# Each vector is 24 x 1
x12 = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0,
               0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0])
y12 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
               0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
z12 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
               1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

x12_ref = x12
y12_ref = y12
z12_ref = z12

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
    """Evaluate shape function b in physical coordinates from normalized (ξ,η,ζ)"""
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
        0.0,          # d/du 1
        1.0,          # d/du u
        0.0,          # d/du v
        0.0,          # d/du w
        v,            # d/du uv
        w,            # d/du uw
        0.0,          # d/du vw
        v*w,          # d/du uvw
        2.0*u,        # d/du u^2
        0.0,          # d/du v^2
        2.0*u*v,      # d/du u^2 v
        v**2,         # d/du u v^2
        3.0*u**2,     # d/du u^3
        0.0,          # d/du v^3
        3.0*u**2*v,   # d/du u^3 v
        v**3          # d/du u v^3
    ])


def db_dv(u, v, w):
    return np.array([
        0.0,          # d/dv 1
        0.0,          # d/dv u
        1.0,          # d/dv v
        0.0,          # d/dv w
        u,            # d/dv uv
        0.0,          # d/dv uw
        w,            # d/dv vw
        u*w,          # d/dv uvw
        0.0,          # d/dv u^2
        2.0*v,        # d/dv v^2
        u**2,         # d/dv u^2 v
        2.0*u*v,      # d/dv u v^2
        0.0,          # d/dv u^3
        3.0*v**2,     # d/dv v^3
        u**3,         # d/dv u^3 v
        3.0*u*v**2    # d/dv u v^3
    ])


def db_dw(u, v, w):
    return np.array([
        0.0,      # d/dw 1
        0.0,      # d/dw u
        0.0,      # d/dw v
        1.0,      # d/dw w
        0.0,      # d/dw uv
        u,        # d/dw uw
        v,        # d/dw vw
        u*v,      # d/dw uvw
        0.0,      # d/dw u^2
        0.0,      # d/dw v^2
        0.0,      # d/dw u^2 v
        0.0,      # d/dw u v^2
        0.0,      # d/dw u^3
        0.0,      # d/dw v^3
        0.0,      # d/dw u^3 v
        0.0       # d/dw u v^3
    ])


def B12_matrix(L, W, H):
    # Reference coordinates of points P1 and P2
    u1 = -L / 2.0
    v1 = -W / 2.0
    w1 = 0.0

    u2 = L / 2.0
    v2 = -W / 2.0
    w2 = 0.0

    u3 = L / 2.0
    v3 = W / 2.0
    w3 = 0.0

    u4 = -L / 2.0
    v4 = W / 2.0
    w4 = 0.0

    # Basis function evaluations
    b1 = b_vec(u1, v1, w1)
    b2 = b_vec(u2, v2, w2)
    b3 = b_vec(u3, v3, w3)
    b4 = b_vec(u4, v4, w4)

    # Construct B12 matrix
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
    ]).T  # shape: (16, 16)

    return B


B = B12_matrix(L, W, H)
print("B12 =\n", B)

print(np.linalg.matrix_rank(B))

B_inv = np.linalg.inv(B)
print("B12_inv =\n", B_inv)


def constraint(q):
    c = np.zeros(24)
    c[0] = q[0] - 0.0  # x0 - (-1) = x0 + 1
    c[1] = q[1] - 0.0  # y0 - (1)
    c[2] = q[2] - 0.0  # z0 - (0)

    c[3] = q[3] - 1.0  # x1 - (1)
    c[4] = q[4] - 0.0  # y1 - (0)
    c[5] = q[5] - 0.0  # z1 - (0)

    c[6] = q[6] - 0.0  # x2 - (0)
    c[7] = q[7] - 1.0  # y2 - (1)
    c[8] = q[8] - 0.0  # z2 - (0)

    c[9] = q[9] - 0.0  # x3 - (0)
    c[10] = q[10] - 0.0  # y3 - (0)
    c[11] = q[11] - 1.0  # z3 - (1)

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
    # Indices of constrained DOFs (from your constraint function)
    constrained_indices = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
    ]
    J = np.zeros((len(constrained_indices), len(q)))
    for i, idx in enumerate(constrained_indices):
        J[i, idx] = 1
    return J


def ds_du_mat(u, v, w, B_inv):
    """Returns 8×3 matrix: each row is gradient of s_i w.r.t [u, v, w]"""
    db_du_mat = db_du(u, v, w)
    db_dv_mat = db_dv(u, v, w)
    db_dw_mat = db_dw(u, v, w)

    ds_du = B_inv @ db_du_mat
    ds_dv = B_inv @ db_dv_mat
    ds_dw = B_inv @ db_dw_mat

    # Stack gradients row-by-row: 8 × 3
    ds = np.stack([ds_du, ds_dv, ds_dw], axis=-1)  # shape (8, 3)
    return ds


def calc_det_J(u, v, w, B_inv, x12_jac, y12_jac, z12_jac):
    # Basis vector derivatives
    db_du_temp = db_du(u, v, w)
    db_dv_temp = db_dv(u, v, w)
    db_dw_temp = db_dw(u, v, w)

    # Shape function derivatives
    ds_du = B_inv @ db_du_temp
    ds_dv = B_inv @ db_dv_temp
    ds_dw = B_inv @ db_dw_temp

    # Nodal matrix: 3 × 16
    # take the first 16 columns of x12, y12, z12
    x12_jac_cut = x12_ref[:16]
    y12_jac_cut = y12_ref[:16]
    z12_jac_cut = z12_ref[:16]
    N_mat_jac = np.vstack([x12_jac_cut, y12_jac_cut, z12_jac_cut])

    # Compute Jacobian
    J = N_mat_jac @ np.column_stack([ds_du, ds_dv, ds_dw])
    return J


def calc_det_J_xi(xi, eta, zeta, B_inv, x12_jac, y12_jac, z12_jac, L, W, H, idx):
    # Shape function derivatives (length 16)
    ds_dxi = B_inv @ db_dxi(xi, eta, zeta, L, W, H)
    ds_deta = B_inv @ db_deta(xi, eta, zeta, L, W, H)
    ds_dzeta = B_inv @ db_dzeta(xi, eta, zeta, L, W, H)

    # Nodal matrix: 3 × 4
    N_mat_jac = np.vstack([x12_jac, y12_jac, z12_jac])

    # Construct Jacobian
    J = N_mat_jac @ np.column_stack([ds_dxi, ds_deta, ds_dzeta])
    return J


n_shell = 2
N_coef = 16 + 8 * (n_shell - 1)
m = np.zeros((N_coef, N_coef))


for elem in range(n_shell):
    elem_idx = element_connectivity[elem]
    idx = np.concatenate([np.arange(node*4, (node+1)*4) for node in elem_idx])
    x_loc = np.concatenate([x12[node*4:(node+1)*4] for node in elem_idx])
    y_loc = np.concatenate([y12[node*4:(node+1)*4] for node in elem_idx])
    z_loc = np.concatenate([z12[node*4:(node+1)*4] for node in elem_idx])

    print("idx: ", idx)
    print("x_loc: ", x_loc)
    print("y_loc: ", y_loc)
    print("z_loc: ", z_loc)

    for ixi, xi in enumerate(gauss_xi_m):
        weight_u = weight_xi_m[ixi]

        for ieta, eta in enumerate(gauss_eta_m):
            weight_v = weight_eta_m[ieta]

            for izeta, zeta in enumerate(gauss_zeta_m):
                weight_w = weight_zeta_m[izeta]

                # Compute b and s
                b = b_vec_xi(xi, eta, zeta, L, W, H)
                s = B_inv @ b

                # Jacobian determinant
                J = calc_det_J_xi(xi, eta, zeta, B_inv, x_loc,
                                  y_loc, z_loc, L, W, H, idx)
                detJ = np.linalg.det(J)

                # Assemble local to global
                for i in range(16):  # 4 nodes per element
                    global_i = idx[i]
                    for j in range(16):
                        global_j = idx[j]
                        m[global_i, global_j] += (rho0 * s[i] * s[j]
                                                  * weight_u * weight_v * weight_w * detJ)

print("mass matrix: ")
# print shape

print("Mass matrix shape:", m.shape)
df = pd.DataFrame(m)
print("Mass matrix:\n", df.to_string(float_format="%.3f"))


Nt = 20  # Number of time steps

ds_du_pre = {}  # Precomputed ∂s/∂(u,v,w) at each quadrature point


for ixi, xi in enumerate(gauss_xi):
    weight_u = weight_xi[ixi]
    for ieta, eta in enumerate(gauss_eta):
        weight_v = weight_eta[ieta]
        for izeta, zeta in enumerate(gauss_zeta):
            weight_w = weight_zeta[izeta]
            u = L * xi / 2
            v = W * eta / 2
            w = H * zeta / 2
            ds_du_pre[(ixi, ieta, izeta)] = ds_du_mat(u, v, w, B_inv)

print("Precomputed ds/du at quadrature points:")
print(ds_du_pre)


detJ_pre = {}
for ixi, xi in enumerate(gauss_xi):
    for ieta, eta in enumerate(gauss_eta):
        for izeta, zeta in enumerate(gauss_zeta):
            u = L * xi / 2
            v = W * eta / 2
            w = H * zeta / 2
            J = calc_det_J(u, v, w, B_inv, x12_ref, y12_ref, z12_ref)
            detJ_pre[(ixi, ieta, izeta)] = np.linalg.det(J)


n_gen_coord = 3 * N_coef   # total degrees of freedom (x, y, z per node)
n_constr = 24              # 4 nodes × 3D fixed DOFs

v = np.zeros(n_gen_coord)
lam_bb = np.zeros(n_constr)
v_guess = v.copy()  # Initial guess for velocity
lam_bb_guess = lam_bb.copy()    # Initial guess for Lagrange multipliers
rho_bb = 1e14

endz_pos = []


for step in range(Nt):
    def compute_deformation_gradient(e, grad_s):
        """
        Compute the deformation gradient F = sum_i e_i ⊗ ∇s_i
        where:
        - e: list of 16 node vectors (each shape (3,))
        - grad_s: numpy array of shape (16, 3), ∇s_i as rows
        Returns:
        - F: numpy array of shape (3, 3)
        """
        F = np.zeros((3, 3))
        for i in range(16):
            F += np.outer(e[i], grad_s[i])  # e_i (3,) ⊗ ∇s_i (3,) → (3×3)
        return F

    def compute_green_lagrange_strain(F):
        """
        Compute Green-Lagrange strain tensor:
            E = 0.5 * (F^T F - I)
        """
        I = np.eye(3)
        C = F.T @ F
        E = 0.5 * (C - I)
        return E

    def compute_internal_force(x12, y12, z12):
        f_int = np.zeros((3 * N_coef,))  # global internal force vector

        for elem in range(n_shell):
            elem_idx = element_connectivity[elem]
            idx = np.concatenate([np.arange(node*4, (node+1)*4)
                                 for node in elem_idx])
            x12_loc = np.concatenate([x12[node*4:(node+1)*4]
                                     for node in elem_idx])
            y12_loc = np.concatenate([y12[node*4:(node+1)*4]
                                     for node in elem_idx])
            z12_loc = np.concatenate([z12[node*4:(node+1)*4]
                                     for node in elem_idx])
            e = [np.array([x12_loc[i], y12_loc[i], z12_loc[i]])
                 for i in range(16)]

            f_elem = np.zeros((16, 3))  # local internal force for this element

            for ixi, xi in enumerate(gauss_xi):
                weight_u = weight_xi[ixi]
                for ieta, eta in enumerate(gauss_eta):
                    weight_v = weight_eta[ieta]
                    for izeta, zeta in enumerate(gauss_zeta):
                        weight_w = weight_zeta[izeta]
                        scale = weight_u * weight_v * weight_w

                        # Compute local coordinates and shape functions
                        u = L * xi / 2
                        v = W * eta / 2
                        w = H * zeta / 2
                        b = b_vec(u, v, w)
                        # shape function values (not used here but kept if needed)
                        s = B_inv @ b

                        ds = ds_du_pre[(ixi, ieta, izeta)]  # shape (16, 3)
                        detJ = detJ_pre[(ixi, ieta, izeta)]

                        # Compute deformation gradient
                        F = compute_deformation_gradient(e, ds)
                        FtF = F.T @ F
                        tr_FtF = np.trace(FtF)
                        FFF = F @ F.T @ F

                        # Stress-like integrand
                        stress_term = (
                            lam_param * (0.5 * tr_FtF - 1.5) * F +
                            mu * (FFF - F)
                        )

                        # Integrate internal force: f_i += stress_term @ grad_s_i
                        for i in range(16):
                            grad_si = ds[i]  # ∇s_i as 3-vector
                            force_i = stress_term @ grad_si  # shape (3,)
                            f_elem[i] += force_i * scale * \
                                detJ * (L * W * H / 8.0)

            # Assemble local to global
            for i in range(16):  # 4 nodes per element
                local_i = i
                global_i = idx[i]
                f_int[3 * global_i + 0] += f_elem[local_i, 0]
                f_int[3 * global_i + 1] += f_elem[local_i, 1]
                f_int[3 * global_i + 2] += f_elem[local_i, 2]

        return f_int

    def alm_nesterov_step(v_guess, lam_guess, v_prev, q_prev, M, f_int_func, f_int, f_ext, h, rho):
        """
        One ALM step with true Nesterov acceleration (FISTA schedule), no restart.
        - Uses scaled duals: lam <- lam + rho*h*c(qA), and grad uses J^T(lam + rho*h*c(qA)).
        - Fixed stepsize alpha (pick conservatively).
        """
        v = v_guess.copy()
        lam = lam_guess.copy()

        max_outer = 5
        max_inner = 300
        inner_tol = 1e-6   # tolerance on iterate change (can relax)
        outer_tol = 1e-6

        # --- choose a fixed stepsize alpha ---
        # Option A (keep your old small value):
        alpha = 1.0e-8

        # Option B (one-time crude Lipschitz estimate; uncomment to try):
        # L_M = np.max(np.sum(np.abs(M), axis=1)) / h          # ||M||_inf / h
        # J0 = constraint_jacobian(q_prev)
        # L_J = (np.max(np.sum(np.abs(J0), axis=1)) ** 2)      # ||J||_inf^2
        # L_est = L_M + rho * h * L_J
        # alpha = 1.0 / (10.0 * max(L_est, 1e-12))             # safety factor 10

        for outer_iter in range(max_outer):
            def grad_L(v_loc):
                # State at look-ahead
                qA = q_prev + h * v_loc

                # unpack qA -> x,y,z nodal arrays
                x_new = qA[0::3]
                y_new = qA[1::3]
                z_new = qA[2::3]

                f_int_dyn = f_int_func(x_new, y_new, z_new)
                g_mech = (M @ (v_loc - v_prev)) / h - (-f_int_dyn + f_ext)

                J = constraint_jacobian(qA)
                cA = constraint(qA)

                return g_mech + J.T @ (lam + rho * h * cA)

            # ---- True Nesterov/FISTA inner loop (fixed schedule, no restart) ----
            v_k = v.copy()
            v_km1 = v.copy()   # zero momentum at first step
            t = 1.0

            for inner_iter in range(max_inner):
                t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
                beta = (t - 1.0) / t_next

                # Nesterov look-ahead
                y = v_k + beta * (v_k - v_km1)

                # Gradient at look-ahead
                print("outer iter: ", outer_iter, "inner iter: ", inner_iter)
                g = grad_L(y)
                # print("g: ", g)
                print("g norm: ", np.linalg.norm(g))

                # Fixed stepsize update
                v_next = y - alpha * g

                # stopping by iterate change (cheap)
                if abs(np.linalg.norm(v_next) - np.linalg.norm(v_k)) < inner_tol:
                    v_k = v_next
                    break

                v_km1, v_k, t = v_k, v_next, t_next

            # OUTER dual update (scaled multipliers)
            v = v_k
            qA = q_prev + h * v
            cA = constraint(qA)
            lam += rho * h * cA

            if np.linalg.norm(cA) < outer_tol:
                break

        return v, lam

    # Now call the internal force computation
    f_int = compute_internal_force(x12, y12, z12)

    # Location where the external force is applied (in physical coordinates)
    u_P, v_P, w_P = 1.0, 0.0, 0.0

    # Basis and shape functions
    b = b_vec(u_P, v_P, w_P)
    s_at_P = B_inv @ b

    # External force at point P
    if step <= 200:
        f_P = np.array([0.0, 0.0, 1000.0])
    else:
        f_P = np.array([0.0, 0.0, 0.0])

    # External force distribution to the 16 DOFs
    f_ext = [s_at_P[i] * f_P for i in range(16)]  # List of (3,) vectors

    # if apply external force this need to be changed
    f_ext_vec = np.zeros((3 * N_coef,))
    elem = n_shell - 1
    idx = np.concatenate([np.arange(node*4, (node+1)*4) for node in elem_idx])
    for i_local, global_idx in enumerate(idx):
        row_idx = slice(3 * global_idx, 3 * (global_idx + 1))
        f_ext_vec[row_idx] = f_ext[i_local]

    print(f_ext_vec)

    # ========== tillhere

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

    v_res, lam_bb_res = alm_nesterov_step(
        v_guess, lam_bb_guess, v_prev, q_prev, M_full, compute_internal_force, f_int, f_ext_vec, time_step, rho_bb)

    v_guess = v_res.copy()
    lam_bb_guess = lam_bb_res.copy()

    q_new = q_prev + time_step * v_guess

    for i in range(N_coef):
        x12[i] = q_new[3 * i + 0]
        y12[i] = q_new[3 * i + 1]
        z12[i] = q_new[3 * i + 2]

    endz_pos.append(z12[20])

    print("x12: ")
    print(x12)
    print("y12: ")
    print(y12)
    print("z12: ")
    print(z12)


print(endz_pos)
