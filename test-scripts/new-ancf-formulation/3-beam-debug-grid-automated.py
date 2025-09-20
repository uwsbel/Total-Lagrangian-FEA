import numpy as np
from grid_mesh_generator import GridConfig, GridMeshGenerator

# Same quadrature rules as original
gauss_xi_m = np.array([
    -0.932469514203152, -0.661209386466265, -0.238619186083197,
     0.238619186083197,  0.661209386466265,  0.932469514203152
])

weight_xi_m = np.array([
    0.171324492379170, 0.360761573048139, 0.467913934572691,
    0.467913934572691, 0.360761573048139, 0.171324492379170
])

gauss_eta = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
weight_eta = np.array([1.0, 1.0])

gauss_zeta = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
weight_zeta = np.array([1.0, 1.0])

# Material properties (same as original)
E = 7e8
nu = 0.33
rho0 = 2700
mu = E / (2 * (1 + nu))
lam_param = (E * nu) / ((1 + nu)*(1 - 2*nu))

# Grid configuration
L = 2.0  # Element length
W = 1.0  # Element width  
H = 1.0  # Element height

# Create pure vertical grid (3 elements)
print("Creating pure vertical grid (0x3)...")
grid_config = GridConfig(nx=2, ny=0, element_length=L, element_width=W, element_height=H)
mesh_gen = GridMeshGenerator(grid_config)

# Generate mesh data
node_coords = mesh_gen.generate_node_coordinates()
elements = mesh_gen.generate_element_connectivity()
dof_mappings = mesh_gen.generate_dof_mapping()

print(f"Grid: {grid_config.nx}x{grid_config.ny}")
print(f"Nodes: {len(node_coords)}")
print(f"Elements: {len(elements)}")
print(f"Total DOFs: {grid_config.get_total_dofs()}")

# Same shape functions as original
def b_vec(u, v, w):
    return np.array([1, u, v, w, u*v, u*w, u**2, u**3])

def b_vec_xi(xi, eta, zeta, L, W, H):
    u = L * xi / 2.0
    v = W * eta / 2.0
    w = H * zeta / 2.0
    return b_vec(u, v, w)

def B12_matrix(L, W, H):
    # Same as original implementation
    u1, u2 = -L/2.0, L/2.0
    v, w = 0.0, 0.0
    
    b1 = b_vec(u1, v, w)
    b2 = b_vec(u2, v, w)
    
    def db_du(u): return np.array([0.0, 1.0, 0.0, 0.0, v, w, 2*u, 3*u**2])
    def db_dv(u): return np.array([0.0, 0.0, 1.0, 0.0, u, 0.0, 0.0, 0.0])
    def db_dw(u): return np.array([0.0, 0.0, 0.0, 1.0, 0.0, u, 0.0, 0.0])
    
    B = np.vstack([b1, db_du(u1), db_dv(u1), db_dw(u1), b2, db_du(u2), db_dv(u2), db_dw(u2)]).T
    return B

def calc_det_J_xi(xi, eta, zeta, B_inv, x12_jac, y12_jac, z12_jac, L, W, H):
    # Simplified version for 2-node beam elements
    # For 2-node beam, we need to create 8 coordinates (4 per node)
    # Each node has: [x, dx/du, dx/dv, dx/dw]
    
    # Create 8-coordinate arrays for the beam element
    x8 = np.array([x12_jac[0], 1.0, 0.0, 0.0, x12_jac[1], 1.0, 0.0, 0.0])
    y8 = np.array([y12_jac[0], 0.0, 1.0, 0.0, y12_jac[1], 0.0, 1.0, 0.0])
    z8 = np.array([z12_jac[0], 0.0, 0.0, 1.0, z12_jac[1], 0.0, 0.0, 1.0])
    
    # Same as original implementation
    db_dxi = np.array([0.0, L/2, 0.0, 0.0, (L*W/4)*eta, (L*H/4)*zeta, (L**2)/2*xi, (3*L**3)/8*xi**2])
    db_deta = np.array([0.0, 0.0, W/2, 0.0, (L*W/4)*xi, 0.0, 0.0, 0.0])
    db_dzeta = np.array([0.0, 0.0, 0.0, H/2, 0.0, (L*H/4)*xi, 0.0, 0.0])
    
    ds_dxi = B_inv @ db_dxi
    ds_deta = B_inv @ db_deta
    ds_dzeta = B_inv @ db_dzeta
    
    N_mat_jac = np.vstack([x8, y8, z8])
    J = N_mat_jac @ np.column_stack([ds_dxi, ds_deta, ds_dzeta])
    return J

# Initialize mass matrix
N_coef = grid_config.get_total_dofs()
m = np.zeros((N_coef, N_coef))

print(f"\nAssembling mass matrix for {len(elements)} elements...")
print(f"Mass matrix size: {N_coef} x {N_coef}")

# Element assembly loop
for elem_idx, elem_info in enumerate(dof_mappings):
    node1, node2 = elem_info['element_nodes']
    dof1_range, dof2_range = elem_info['dof_ranges']
    orientation = elem_info['orientation']
    
    # Get element dimensions based on orientation
    L_elem, W_elem, H_elem = mesh_gen.get_element_dimensions(orientation)
    
    # Get node coordinates
    x_coords = np.array([node_coords[node1][0], node_coords[node2][0]])
    y_coords = np.array([node_coords[node1][1], node_coords[node2][1]])
    z_coords = np.array([node_coords[node1][2], node_coords[node2][2]])
    
    print(f"Element {elem_idx}: nodes ({node1},{node2}), orientation: {orientation}")
    print(f"  DOF ranges: {dof1_range}, {dof2_range}")
    print(f"  Coords: x={x_coords}, y={y_coords}, z={z_coords}")
    print(f"  Dimensions: L={L_elem}, W={W_elem}, H={H_elem}")
    
    # Create B matrix for this element
    B = B12_matrix(L_elem, W_elem, H_elem)
    B_inv = np.linalg.inv(B)
    
    # Quadrature integration
    for ixi, xi in enumerate(gauss_xi_m):
        weight_u = weight_xi_m[ixi]
        for ieta, eta in enumerate(gauss_eta):
            weight_v = weight_eta[ieta]
            for izeta, zeta in enumerate(gauss_zeta):
                weight_w = weight_zeta[izeta]
                
                # Compute shape functions
                b = b_vec_xi(xi, eta, zeta, L_elem, W_elem, H_elem)
                s = B_inv @ b
                
                # Jacobian determinant
                J = calc_det_J_xi(xi, eta, zeta, B_inv, x_coords, y_coords, z_coords, L_elem, W_elem, H_elem)
                detJ = np.linalg.det(J)
                
                # Assemble to global mass matrix
                # Map local DOFs to global DOFs
                local_dofs = list(range(8))  # 8 local DOFs per element
                global_dofs = list(range(dof1_range[0], dof1_range[1]+1)) + list(range(dof2_range[0], dof2_range[1]+1))
                
                for i in range(8):
                    global_i = global_dofs[i]
                    for j in range(8):
                        global_j = global_dofs[j]
                        m[global_i, global_j] += (
                            rho0 * s[i] * s[j] * weight_u * weight_v * weight_w * detJ
                        )

print("\nMass matrix assembly complete!")
print(f"Mass matrix shape: {m.shape}")

# Print mass matrix (first 10x10 for readability)
print("\nMass matrix (first 10x10):")
for i in range(min(10, m.shape[0])):
    for j in range(min(10, m.shape[1])):
        print(f"{float(m[i,j]):6.1f}", end=" ")
    print()

# Print full mass matrix if small enough
if m.shape[0] <= 20:
    print("\nFull mass matrix:")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            print(f"{float(m[i,j]):6.1f}", end=" ")
        print()

# Check matrix properties
print(f"\nMatrix properties:")
print(f"  Shape: {m.shape}")
print(f"  Non-zero entries: {np.count_nonzero(m)}")
print(f"  Max value: {np.max(m):.2e}")
print(f"  Min value: {np.min(m):.2e}")
print(f"  Condition number: {np.linalg.cond(m):.2e}")

# Make mass matrix available for testing
if __name__ == "__main__":
    # This will be accessible when imported
    pass
