import numpy as np
import pandas as pd
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

# Each vector is 8x1
x12 = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0])
y12 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
z12 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

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

    u2 =  L / 2.0
    v2 = -W / 2.0
    w2 = 0.0

    u3 =  L / 2.0
    v3 =  W / 2.0
    w3 = 0.0

    u4 = -L / 2.0
    v4 =  W / 2.0
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



def calc_det_J_xi(xi, eta, zeta, B_inv, x12_jac, y12_jac, z12_jac, L, W, H, idx):
    # Shape function derivatives (length 16)
    ds_dxi   = B_inv @ db_dxi(xi, eta, zeta, L, W, H)
    ds_deta  = B_inv @ db_deta(xi, eta, zeta, L, W, H)
    ds_dzeta = B_inv @ db_dzeta(xi, eta, zeta, L, W, H)


    # Nodal matrix: 3 × 4
    N_mat_jac = np.vstack([x12_jac, y12_jac, z12_jac])

    # Construct Jacobian
    J = N_mat_jac @ np.column_stack([ds_dxi, ds_deta, ds_dzeta])
    return J

n_shell = 1
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

        for ieta, eta in enumerate(gauss_eta):
            weight_v = weight_eta[ieta]

            for izeta, zeta in enumerate(gauss_zeta):
                weight_w = weight_zeta[izeta]

                # Compute b and s
                b = b_vec_xi(xi, eta, zeta, L, W, H)
                s = B_inv @ b 

                # Jacobian determinant
                J = calc_det_J_xi(xi, eta, zeta, B_inv, x_loc, y_loc, z_loc, L, W, H, idx)
                detJ = np.linalg.det(J)


                # Assemble local to global
                for i in range(16):  # 4 nodes per element
                    global_i = idx[i]
                    for j in range(16):
                        global_j = idx[j]
                        m[global_i, global_j] += (rho0 * s[i] * s[j] * weight_u * weight_v * weight_w * detJ)

print("mass matrix: ")
# print shape

print("Mass matrix shape:", m.shape)
df = pd.DataFrame(m)
print("Mass matrix:\n", df.to_string(float_format="%.3f"))