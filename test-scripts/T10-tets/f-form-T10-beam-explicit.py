#!/usr/bin/env python3
"""
Symplectic Euler (Semi-Implicit Euler) solver for T10 tetrahedra with SVK material.

Features:
- Lumped mass matrix (HRZ diagonal scaling method)
- Symplectic Euler time integration: 
    v_{n+1} = v_n + dt * M^{-1} * (f_ext - f_int)
    x_{n+1} = x_n + dt * v_{n+1}   <- uses NEW velocity (symplectic)
- Pin boundary conditions at x=0
- Stability analysis and error detection
- Better energy conservation than explicit Forward Euler
"""
import numpy as np
import matplotlib.pyplot as plt
from tet_mesh_reader import read_node, read_ele

# ----------------------------
# Material parameters
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

def tet10_shape_functions(xi, eta, zeta):
    L2, L3, L4 = xi, eta, zeta
    L1 = 1.0 - xi - eta - zeta
    L = [L1, L2, L3, L4]
    N = np.zeros(10)
    N[0:4] = [L[i]*(2*L[i]-1) for i in range(4)]
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    for k, (i, j) in enumerate(edges, start=4):
        N[k] = 4*L[i]*L[j]
    return N

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
            J += np.outer(X_elem_nodes[a], dN_dxi[a])
        detJ = np.linalg.det(J)
        grad_N = np.zeros((10, 3))
        JT = J.T
        for a in range(10):
            grad_N[a, :] = np.linalg.solve(JT, dN_dxi[a])
        pre.append({"grad_N": grad_N, "detJ": detJ, "w": wq})
    return pre

def tet10_precompute_reference_mesh(X_nodes, X_elem):
    pre_list = []
    for elem_idx in range(X_elem.shape[0]):
        node_indices = X_elem[elem_idx]
        X_elem_nodes = X_nodes[node_indices]
        pre = tet10_precompute_reference(X_elem_nodes)
        pre_list.append(pre)
    return pre_list

# ----------------------------
# Internal force (SVK, TL)
# ----------------------------
def tet10_internal_force(x_nodes, pre, lam, mu):
    f = np.zeros((10, 3))
    for q in pre:
        grad_N = q["grad_N"]
        detJ   = q["detJ"]
        wq     = q["w"]

        # F = sum_a x_a ⊗ h_a^T
        F = np.zeros((3, 3))
        for a in range(10):
            F += np.outer(x_nodes[a], grad_N[a])

        FtF   = F.T @ F
        trFtF = np.trace(FtF)
        # P for SVK
        P = lam*(0.5*trFtF - 1.5)*F + mu*(F @ F.T @ F - F)

        dV = detJ * wq
        for a in range(10):
            f[a] += (P @ grad_N[a]) * dV
    return f

def tet10_internal_force_mesh(x_nodes, X_elem, pre_mesh, lam, mu):
    n_nodes = x_nodes.shape[0]
    f_int = np.zeros((n_nodes, 3))
    for elem_idx in range(X_elem.shape[0]):
        node_indices = X_elem[elem_idx]
        x_elem_nodes = x_nodes[node_indices]
        pre = pre_mesh[elem_idx]
        f_elem = tet10_internal_force(x_elem_nodes, pre, lam, mu)
        for a_local, a_global in enumerate(node_indices):
            f_int[a_global] += f_elem[a_local]
    return f_int

# ----------------------------
# Lumped mass (HRZ diagonal scaling)
# ----------------------------
def tet10_lumped_mass_hrz(X_nodes, rho):
    """
    HRZ (Hinton-Rock-Zienkiewicz) lumped mass using diagonal scaling.
    Standard row-sum lumping fails for Tet10 (produces negative corner masses).
    Instead, we integrate diagonal terms N_i^2 and scale to preserve total mass.
    Returns diagonal mass vector (10,) for the element.
    """
    pts_xyz, _, w = tet5pt_quadrature()
    
    vol_elem = 0.0
    diag_consistent = np.zeros(10)
    
    # Integrate density and diagonal shape functions only
    for (xi, eta, zeta), wq in zip(pts_xyz, w):
        dN_dxi = tet10_shape_function_gradients(xi, eta, zeta)
        J = np.zeros((3, 3))
        for a in range(10):
            J += np.outer(X_nodes[a], dN_dxi[a])
        detJ = abs(np.linalg.det(J))
        
        # Shape functions
        N = tet10_shape_functions(xi, eta, zeta)
        
        vol_elem += detJ * wq
        # Only compute diagonal: ∫ ρ N_i² dV
        for i in range(10):
            diag_consistent[i] += (N[i]**2) * detJ * wq
    
    # HRZ Scaling: preserve total element mass
    total_mass = rho * vol_elem
    sum_diag = np.sum(diag_consistent)
    M_lumped = (total_mass / sum_diag) * diag_consistent
    
    return M_lumped

def tet10_lumped_mass_mesh(X_nodes, X_elem, rho):
    """
    Assemble lumped mass for entire mesh.
    Returns diagonal mass vector (3*n_nodes,) for x,y,z components.
    """
    n_nodes = X_nodes.shape[0]
    M_lumped = np.zeros(3 * n_nodes)
    
    for elem_idx in range(X_elem.shape[0]):
        node_indices = X_elem[elem_idx]
        X_elem_nodes = X_nodes[node_indices]
        M_elem = tet10_lumped_mass_hrz(X_elem_nodes, rho)
        
        for a_local, a_global in enumerate(node_indices):
            # Add to each component (x, y, z)
            M_lumped[3*a_global:3*a_global+3] += M_elem[a_local]
    
    return M_lumped

# ----------------------------
# Constraints (pins at x==0)
# ----------------------------
def get_fixed_nodes(X_nodes):
    return np.where(np.isclose(X_nodes[:, 0], 0.0))[0]

def apply_pin_constraints(v, fixed_nodes):
    """Zero out velocities at pinned nodes"""
    for node in fixed_nodes:
        v[3*node:3*node+3] = 0.0
    return v


# ----------------------------
# Main explicit solver
# ----------------------------
if __name__ == "__main__":
    import os
    # Get absolute path to mesh files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.join(script_dir, "../../data/meshes/T10")
    node_file = os.path.join(mesh_dir, "beam_3x2x1.1.node")
    ele_file = os.path.join(mesh_dir, "beam_3x2x1.1.ele")
    
    # Load mesh
    X_nodes = read_node(node_file)
    X_elem = read_ele(ele_file)
    
    print(f"Mesh: {X_nodes.shape[0]} nodes, {X_elem.shape[0]} elements")
    print(f"Material: E={E_mat:.2e} Pa, nu={nu}, rho={rho0} kg/m³")
    print(f"Lamé parameters: lambda={lam:.2e}, mu={mu:.2e}")
    
    # Precompute reference configuration data
    pre_mesh = tet10_precompute_reference_mesh(X_nodes, X_elem)
    
    # Compute lumped mass matrix
    print("\nComputing lumped mass matrix (HRZ)...")
    M_lumped = tet10_lumped_mass_mesh(X_nodes, X_elem, rho0)
    print(f"Total mass: {np.sum(M_lumped)/3:.6e} kg")
    print(f"Mass per node (avg): {np.mean(M_lumped):.6e} kg")
    
    # Get fixed nodes
    fixed_nodes = get_fixed_nodes(X_nodes)
    print(f"Fixed nodes: {len(fixed_nodes)} nodes at x=0")
    
    # Initial conditions
    x_nodes = X_nodes.copy()
    v = np.zeros(3 * X_nodes.shape[0])  # velocity (flat)
    
    # External force
    f_ext = np.zeros(3 * X_nodes.shape[0])
    f_ext[3*19 + 0] = 1000.0  # 1000 N in x-direction on node 19
    
    # Time integration parameters
    dt = 1e-6  # Time step (larger for longer simulation)
    total_time = 0.005  # Total simulation time in seconds
    Nt = int(total_time / dt)  # Number of steps based on total_time and dt
    
    print(f"\nTime integration:")
    print(f"  dt = {dt:.2e} s")
    print(f"  Steps = {Nt}")
    print(f"  Total time = {Nt * dt:.4f} s")
    
    # Estimate critical time step for stability
    # For explicit methods: dt_crit ~ h / c, where c = sqrt(E/rho) is wave speed
    wave_speed = np.sqrt(E_mat / rho0)
    elem_size = np.mean([np.linalg.norm(X_nodes[X_elem[e,0]] - X_nodes[X_elem[e,1]]) 
                         for e in range(min(10, X_elem.shape[0]))])
    dt_crit = 0.5 * elem_size / wave_speed  # 0.5 for safety
    print(f"\nStability check:")
    print(f"  Wave speed c = {wave_speed:.2e} m/s")
    print(f"  Element size h ≈ {elem_size:.4f} m")
    print(f"  Critical dt ≈ {dt_crit:.2e} s")
    print(f"  Using dt = {dt:.2e} s ({'OK' if dt < dt_crit else 'WARNING: May be unstable!'})")
    
    # Storage for tracking
    node19_x = []
    node20_x = []
    
    t = 0.0
    force_removal_step = 2500  # Remove force after step 2500 (0.05s)
    
    print("\nStarting simulation...\n")
    
    for step in range(Nt):
        # Remove force after specified step
        if step > force_removal_step:
            f_ext[3*19 + 0] = 0.0
        
        # Compute internal forces at current configuration
        f_int = tet10_internal_force_mesh(x_nodes, X_elem, pre_mesh, lam, mu)
        f_int_flat = f_int.flatten()
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(f_int_flat)):
            print(f"\nERROR: Internal force contains NaN/Inf at step {step}")
            print(f"Max displacement: {np.max(np.abs(x_nodes.flatten() - X_nodes.flatten())):.3e}")
            break
        
        # Symplectic Euler: a = M^{-1} * (f_ext - f_int)
        a = (f_ext - f_int_flat) / M_lumped
        
        # Update velocity: v_{n+1} = v_n + dt * a_n
        v = v + dt * a
        
        # Apply boundary conditions (zero velocity at fixed nodes)
        v = apply_pin_constraints(v, fixed_nodes)
        
        # Update positions using NEW velocity: x_{n+1} = x_n + dt * v_{n+1} (symplectic)
        x_flat = x_nodes.flatten() + dt * v
        x_nodes = x_flat.reshape(-1, 3)
        
        # Check for instability
        max_disp = np.max(np.abs(x_nodes.flatten() - X_nodes.flatten()))
        if max_disp > 1.0:  # More than 1 meter displacement is suspicious
            print(f"\nWARNING: Large displacement detected at step {step}")
            print(f"Max displacement: {max_disp:.3e} m")
            break
        
        # Save data
        node19_x.append(x_nodes[19, 0])
        node20_x.append(x_nodes[20, 0])
        
        # Print progress every 500 steps
        if step % 500 == 0:
            print(f"Step {step}/{Nt}: t={step*dt:.4f}s, node 19 x={x_nodes[19,0]:.6e}, node 20 x={x_nodes[20,0]:.6e}")
    
    print("\nSimulation complete!\n")
    
    # Save results to file (in same directory as script)
    output_file = os.path.join(script_dir, "results_explicit.npz")
    time_array = np.arange(len(node19_x)) * dt
    np.savez(output_file,
             time=time_array,
             node19_x=np.array(node19_x),
             node20_x=np.array(node20_x),
             x_final=x_nodes,
             X_initial=X_nodes,
             dt=dt,
             Nt=Nt,
             force_removal_step=force_removal_step,
             total_mass=np.sum(M_lumped)/3)
    print(f"Results saved to: {output_file}")
    
    # Plot after simulation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    time_array = np.arange(len(node19_x)) * dt
    
    ax1.plot(time_array, node19_x, linewidth=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Node 19 X Position (m)")
    ax1.set_title("Node 19 X Position vs Time (Symplectic Euler)")
    ax1.grid(True)
    
    ax2.plot(time_array, node20_x, color='orange', linewidth=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Node 20 X Position (m)")
    ax2.set_title("Node 20 X Position vs Time (Symplectic Euler)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
