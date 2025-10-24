import numpy as np

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

n_beam = 2  # Number of beam elements


# Each vector is 8x1
# The values are x, derivative wrt u, derivative wrt v, derivative wrt w for the two nodes
x12 = np.array([-1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
y12 = np.array([ 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
z12 = np.array([ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])


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



offset_start = np.zeros(n_beam, dtype=int)
offset_end   = np.zeros(n_beam, dtype=int)

for i in range(n_beam):
    offset_start[i] = i * 4
    offset_end[i]   = offset_start[i] + 7


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
print(f"Size of mass matrix: {m.shape[0]} x {m.shape[1]}")
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        print(f"{float(m[i,j]):6.1f}", end=" ")
    print()


# from numpy.linalg import matrix_rank
# r = matrix_rank(m)
# print("Rank of mass matrix:")
# print(r)
