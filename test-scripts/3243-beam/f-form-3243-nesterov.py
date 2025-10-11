import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    [ 0.2, -0.5, -0.5],
    [ 0.2, -0.5,  0.5],
    [ 0.2,  0.5, -0.5],
    [ 0.2,  0.5,  0.5],
    [ 0.6, -0.5, -0.5],
    [ 0.6, -0.5,  0.5],
    [ 0.6,  0.5, -0.5],
    [ 0.6,  0.5,  0.5],
    [ 1.0, -0.5, -0.5],
    [ 1.0, -0.5,  0.5],
    [ 1.0,  0.5, -0.5],
    [ 1.0,  0.5,  0.5]
])

# 4-point Gauss-Legendre quadrature in 1D

# 6-point Gauss-Legendre quadrature (symmetric)
gauss_xi_m = np.array([
    -0.932469514203152, -0.661209386466265, -0.238619186083197,
     0.238619186083197,  0.661209386466265,  0.932469514203152
])

weight_xi_m = np.array([
    0.171324492379170, 0.360761573048139, 0.467913934572691,
    0.467913934572691, 0.360761573048139, 0.171324492379170
])

# 3-point Gauss-Legendre quadrature
gauss_xi = np.array([
    -np.sqrt(3/5), 0.0, np.sqrt(3/5)
])

weight_xi = np.array([
    5/9, 8/9, 5/9
])

# 2-point Gauss-Legendre quadrature (for eta and zeta directions)
gauss_eta = np.array([
    -1/np.sqrt(3), 1/np.sqrt(3)
])

weight_eta = np.array([
    1.0, 1.0
])

gauss_zeta = np.array([
    -1/np.sqrt(3), 1/np.sqrt(3)
])

weight_zeta = np.array([
    1.0, 1.0
])


E = 7e8
nu = 0.33
rho0 = 2700

mu = E / (2 * (1 + nu))              # Shear modulus μ
lam_param = (E * nu) / ((1 + nu)*(1 - 2*nu))  # Lamé’s first parameter λ

H = 1.0  # Height
W = 1.0  # Width
L = 2.0  # Length

n_beam = 3  # Number of beam elements


# Each vector is 8x1
x12 = np.array([-1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
y12 = np.array([ 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
z12 = np.array([ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

x12_ref = x12
y12_ref = y12
z12_ref = z12

# Velocities (all zero)
x12_dot = np.zeros_like(x12)
y12_dot = np.zeros_like(y12)
z12_dot = np.zeros_like(z12)

# Append new blocks for additional beams (besides the first one)
for i in range(2, n_beam + 1):  # MATLAB loop is inclusive
    x_offset = 2.0

    # Only shift the first entry of the last beam by x_offset
    x_block = x12[-4:].copy()
    x_block[0] += x_offset

    # Last 4 entries for y and z
    y_block = y12[-4:]
    z_block = z12[-4:]

    # Append to the nodal arrays
    x12 = np.concatenate([x12, x_block])
    y12 = np.concatenate([y12, y_block])
    z12 = np.concatenate([z12, z_block])

    # Append to velocity arrays
    x12_dot = np.concatenate([x12_dot, np.zeros(4)])
    y12_dot = np.concatenate([y12_dot, np.zeros(4)])
    z12_dot = np.concatenate([z12_dot, np.zeros(4)])


offset_start = np.zeros(n_beam, dtype=int)
offset_end   = np.zeros(n_beam, dtype=int)

for i in range(n_beam):
    offset_start[i] = i * 4
    offset_end[i]   = offset_start[i] + 7


time_step = 1e-3


def b_vec(u, v, w):
    return np.array([
        1,
        u,
        v,
        w,
        u * v,
        u * w,
        u ** 2,
        u ** 3
    ])

def b_vec_xi(xi, eta, zeta, L, W, H):
    """Evaluate shape function b in physical coordinates from normalized (ξ,η,ζ)"""
    u = L * xi / 2.0
    v = W * eta / 2.0
    w = H * zeta / 2.0
    return b_vec(u, v, w)

def B12_matrix(L, W, H):
    # Reference coordinates of points P1 and P2
    u1 = -L / 2.0
    u2 =  L / 2.0
    v = 0.0
    w = 0.0

    # Basis function evaluations
    b1 = b_vec(u1, v, w)
    b2 = b_vec(u2, v, w)

    # Partial derivatives
    def db_du(u): return np.array([0.0, 1.0, 0.0, 0.0, v, w, 2*u, 3*u**2])
    def db_dv(u): return np.array([0.0, 0.0, 1.0, 0.0, u, 0.0, 0.0, 0.0])
    def db_dw(u): return np.array([0.0, 0.0, 0.0, 1.0, 0.0, u, 0.0, 0.0])

    # Construct B12 matrix
    B = np.vstack([
        b1,
        db_du(u1),
        db_dv(u1),
        db_dw(u1),
        b2,
        db_du(u2),
        db_dv(u2),
        db_dw(u2)
    ]).T  # shape: (8, 8)
    
    return B

B = B12_matrix(L, W, H)
print("B12 =\n", B)

B_inv = np.linalg.inv(B)
print("B12_inv =\n", B_inv)

def calc_det_J(u, v, w, B_inv, x12_jac, y12_jac, z12_jac):
    # Basis vector derivatives
    db_du = np.array([0, 1, 0, 0, v, w, 2*u, 3*u**2])
    db_dv = np.array([0, 0, 1, 0, u, 0, 0, 0])
    db_dw = np.array([0, 0, 0, 1, 0, u, 0, 0])

    # Shape function derivatives
    ds_du = B_inv @ db_du
    ds_dv = B_inv @ db_dv
    ds_dw = B_inv @ db_dw

    # Nodal matrix: 3 × 8
    N_mat_jac = np.vstack([x12_jac, y12_jac, z12_jac])

    # Compute Jacobian
    J = N_mat_jac @ np.column_stack([ds_du, ds_dv, ds_dw])
    return J

def calc_det_J_xi(xi, eta, zeta, B_inv, x12_jac, y12_jac, z12_jac, L, W, H):
    # Basis derivatives in normalized coordinates
    db_dxi = np.array([
        0.0,
        L / 2,
        0.0,
        0.0,
        (L * W / 4) * eta,
        (L * H / 4) * zeta,
        (L ** 2) / 2 * xi,
        (3 * L ** 3) / 8 * xi ** 2
    ])
    db_deta = np.array([
        0.0,
        0.0,
        W / 2,
        0.0,
        (L * W / 4) * xi,
        0.0,
        0.0,
        0.0
    ])
    db_dzeta = np.array([
        0.0,
        0.0,
        0.0,
        H / 2,
        0.0,
        (L * H / 4) * xi,
        0.0,
        0.0
    ])

    # Shape function derivatives
    ds_dxi   = B_inv @ db_dxi
    ds_deta  = B_inv @ db_deta
    ds_dzeta = B_inv @ db_dzeta

    # Nodal matrix: 3 × 8
    N_mat_jac = np.vstack([x12_jac, y12_jac, z12_jac])

    # Construct Jacobian
    J = N_mat_jac @ np.column_stack([ds_dxi, ds_deta, ds_dzeta])
    return J

def ds_du_mat(u, v, w, B_inv):
    """Returns 8×3 matrix: each row is gradient of s_i w.r.t [u, v, w]"""
    db_du = np.array([0, 1, 0, 0, v, w, 2*u, 3*u**2])
    db_dv = np.array([0, 0, 1, 0, u, 0, 0, 0])
    db_dw = np.array([0, 0, 0, 1, 0, u, 0, 0])

    ds_du = B_inv @ db_du
    ds_dv = B_inv @ db_dv
    ds_dw = B_inv @ db_dw

    # Stack gradients row-by-row: 8 × 3
    ds = np.stack([ds_du, ds_dv, ds_dw], axis=-1)  # shape (8, 3)
    return ds

def ds_dxi_mat(xi, eta, zeta, B_inv, L, W, H):
    """Returns 8×3 matrix: each row is gradient of s_i w.r.t [ξ, η, ζ]"""
    db_dxi = np.array([
        0.0,
        L / 2,
        0.0,
        0.0,
        (L * W / 4) * eta,
        (L * H / 4) * zeta,
        (L ** 2 / 2) * xi,
        (3 * L ** 3 / 8) * xi ** 2
    ])
    db_deta = np.array([
        0.0,
        0.0,
        W / 2,
        0.0,
        (L * W / 4) * xi,
        0.0,
        0.0,
        0.0
    ])
    db_dzeta = np.array([
        0.0,
        0.0,
        0.0,
        H / 2,
        0.0,
        (L * H / 4) * xi,
        0.0,
        0.0
    ])

    ds_dxi = B_inv @ db_dxi
    ds_deta = B_inv @ db_deta
    ds_dzeta = B_inv @ db_dzeta

    # Stack gradients row-by-row: 8 × 3
    ds = np.stack([ds_dxi, ds_deta, ds_dzeta], axis=-1)  # shape (8, 3)
    return ds


def constraint(q):
    c = np.zeros(12)
    c[0] = q[0] + 1.0  # x0 - (-1) = x0 + 1
    c[1] = q[1] - 1.0  # y0 - (1)
    c[2] = q[2] - 0.0  # z0 - (0)

    c[3] = q[3] - 1.0  # x1 - (1)
    c[4] = q[4] - 0.0  # y1 - (0)
    c[5] = q[5] - 0.0  # z1 - (0)

    c[6] = q[6] - 0.0  # x2 - (0)
    c[7] = q[7] - 1.0  # y2 - (1)
    c[8] = q[8] - 0.0  # z2 - (0)

    c[9]  = q[9] - 0.0  # x3 - (0)
    c[10] = q[10] - 0.0 # y3 - (0)
    c[11] = q[11] - 1.0 # z3 - (1)

    return c


def constraint_jacobian(q):
    J = np.zeros((12, len(q)))
    for i in range(12):
        J[i, i] = 1
    return J



N_coef = 8 + 4 * (n_beam - 1)
m = np.zeros((N_coef, N_coef))



for elem in range(n_beam):
    idx = np.arange(offset_start[elem], offset_end[elem] + 1)
    x_loc = x12[idx]
    y_loc = y12[idx]
    z_loc = z12[idx]

    for ixi, xi in enumerate(gauss_xi_m):
        weight_u = weight_xi_m[ixi]

        for ieta, eta in enumerate(gauss_eta):
            weight_v = weight_eta[ieta]

            for izeta, zeta in enumerate(gauss_zeta):
                weight_w = weight_zeta[izeta]

                # Compute b and s
                b = b_vec_xi(xi, eta, zeta, L, W, H)
                s = B_inv @ b  # shape: (8,)

                # Jacobian determinant
                J = calc_det_J_xi(xi, eta, zeta, B_inv, x_loc, y_loc, z_loc, L, W, H)
                detJ = np.linalg.det(J)

                # Assemble local to global
                for i in range(8):
                    global_i = offset_start[elem] + i
                    for j in range(8):
                        global_j = offset_start[elem] + j
                        m[global_i, global_j] += (
                            rho0 * s[i] * s[j] * weight_u * weight_v * weight_w * detJ
                        )


print("Mass matrix m_{ij}:")
print(m)

from numpy.linalg import matrix_rank
r = matrix_rank(m)
print("Rank of mass matrix:")
print(r)


n_sample = uvw.shape[0]
r_ref = np.zeros((n_beam, n_sample, 3))     # reference positions
r_global = np.zeros((n_beam, n_sample, 3))  # deformed/global positions (if needed later)

n_sample = uvw.shape[0]
r_ref = np.zeros((n_beam, n_sample, 3))  # beam × sample × (x,y,z)

        


for a in range(n_beam):
    idx = np.arange(offset_start[a], offset_end[a] + 1)
    x12_loc = x12[idx]
    y12_loc = y12[idx]
    z12_loc = z12[idx]

    for i in range(n_sample):
        u, v, w = uvw[i]

        b_sample = np.array([1, u, v, w, u*v, u*w, u**2, u**3])
        s_sample = B_inv @ b_sample

        N_mat = np.vstack([x12_loc, y12_loc, z12_loc])
        r_ref[a, i, :] = N_mat @ s_sample

        


Nt = 40  # Number of time steps

end_x = np.zeros((n_beam, Nt))
end_y = np.zeros((n_beam, Nt))
end_z = np.zeros((n_beam, Nt))
end_x_du = np.zeros((n_beam, Nt))
end_y_du = np.zeros((n_beam, Nt))
end_z_du = np.zeros((n_beam, Nt))



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
n_constr = 12              # 4 nodes × 3D fixed DOFs

v = np.zeros(n_gen_coord)            
lam_bb = np.zeros(n_constr)
v_guess = v.copy()  # Initial guess for velocity
lam_bb_guess = lam_bb.copy()    # Initial guess for Lagrange multipliers
rho_bb = 1e14


for step in range(Nt):
    def compute_deformation_gradient(e, grad_s):
        """
        Compute the deformation gradient F = sum_i e_i ⊗ ∇s_i
        where:
        - e: list of 8 node vectors (each shape (3,))
        - grad_s: numpy array of shape (8, 3), ∇s_i as rows
        Returns:
        - F: numpy array of shape (3, 3)
        """
        F = np.zeros((3, 3))
        for i in range(8):
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

        for elem in range(n_beam):
            idx = np.arange(offset_start[elem], offset_end[elem] + 1)
            x12_loc = x12[idx]
            y12_loc = y12[idx]
            z12_loc = z12[idx]
            e = [np.array([x12_loc[i], y12_loc[i], z12_loc[i]]) for i in range(8)]

            f_elem = np.zeros((8, 3))  # local internal force for this element

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
                        s = B_inv @ b  # shape function values (not used here but kept if needed)

                        ds = ds_du_pre[(ixi, ieta, izeta)]  # shape (8, 3)
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
                        for i in range(8):
                            grad_si = ds[i]  # ∇s_i as 3-vector
                            force_i = stress_term @ grad_si  # shape (3,)
                            f_elem[i] += force_i * scale * detJ * (L * W * H / 8.0)

            # Assemble local into global internal force vector
            for a_local, a_global in enumerate(idx):
                f_int[3 * a_global + 0] += f_elem[a_local, 0]
                f_int[3 * a_global + 1] += f_elem[a_local, 1]
                f_int[3 * a_global + 2] += f_elem[a_local, 2]

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
        max_inner = 200
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
            v_k   = v.copy()
            v_km1 = v.copy()   # zero momentum at first step
            t     = 1.0

            for inner_iter in range(max_inner):
                t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
                beta   = (t - 1.0) / t_next

                # Nesterov look-ahead
                y = v_k + beta * (v_k - v_km1)

                # Gradient at look-ahead
                print("outer iter: ", outer_iter, "inner iter: ", inner_iter)
                g = grad_L(y)
                #print("g: ", g)
                print("g: ", np.linalg.norm(g))

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
    b = b_vec(u_P, v_P, w_P)        # (8,)
    s_at_P = B_inv @ b              # (8,)

    # External force at point P
    if step <= 200:
        f_P = np.array([200.0, -10.0, 3100.0])
    else:
        f_P = np.array([0.0, 0.0, 0.0])



    # External force distribution to the 8 DOFs
    f_ext = [s_at_P[i] * f_P for i in range(8)]  # List of (3,) vectors

    f_ext_vec = np.zeros((3 * N_coef,))
    elem = n_beam - 1
    idx = np.arange(offset_start[elem], offset_end[elem] + 1)
    for i_local, global_idx in enumerate(idx):
        row_idx = slice(3 * global_idx, 3 * (global_idx + 1))
        f_ext_vec[row_idx] = f_ext[i_local]

    print(f_ext_vec)

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


    v_res, lam_bb_res = alm_nesterov_step(v_guess, lam_bb_guess, v_prev, q_prev, M_full, compute_internal_force,f_int, f_ext_vec, time_step, rho_bb)


    v_guess = v_res.copy()
    lam_bb_guess = lam_bb_res.copy()

    q_new = q_prev + time_step * v_guess

    for i in range(N_coef):
        x12[i] = q_new[3 * i + 0]
        y12[i] = q_new[3 * i + 1]
        z12[i] = q_new[3 * i + 2]

    # Tip index = node 5 of last element (MATLAB: end - 3)
    tip_idx = offset_end[-1] - 3

    end_x[-1, step] = x12[tip_idx] - 3.0
    end_y[-1, step] = y12[tip_idx]
    end_z[-1, step] = z12[tip_idx]

    end_x_du[-1, step] = x12[tip_idx + 1]
    end_y_du[-1, step] = y12[tip_idx + 1]
    end_z_du[-1, step] = z12[tip_idx + 1]

    print("x12")
    print(x12)

    print("y12")
    print(y12)

    print("z12")
    print(z12)

    


plt.figure()
plt.plot(np.arange(Nt), end_z[-1], '-o', label='end_z')
#plt.plot(np.arange(Nt), end_x_du[-1], '-s', label='end_z_du')
plt.xlabel('Time step')
plt.ylabel('z displacement')
plt.legend()
plt.title('Vertical displacement of nodes 5 and 6')
plt.grid(True)
plt.tight_layout()
plt.show()





