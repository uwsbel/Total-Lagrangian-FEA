"""
Nonlinear 3D bunny mesh dynamic analysis using Backward Euler time integration.
Matches C++ implementation: uses nodal forces, Backward Euler, and same tolerances.
For validation and debugging purposes.
"""
import os
import numpy as np
import ufl

from dolfinx import fem, mesh, log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from tetgen_mesh_loader import load_tetgen_mesh_from_files


# ============================================================================
# GEOMETRY AND MESH SETUP
# ============================================================================
# Construct path to mesh files
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir))
mesh_dir = os.path.join(project_root, "data", "meshes", "T10")

node_file = os.path.join(mesh_dir, "bunny_ascii_26.1.node")
ele_file = os.path.join(mesh_dir, "bunny_ascii_26.1.ele")

print(f"Loading mesh from:")
print(f"  Node file: {node_file}")
print(f"  Element file: {ele_file}")

# Load TetGen mesh
domain, x_nodes = load_tetgen_mesh_from_files(node_file, ele_file)
print(f"Loaded {len(x_nodes)} nodes")
print(f"Loaded mesh with P2 tetrahedral elements")

print(f"\nMesh created successfully!")
print(f"  Topology dimension: {domain.topology.dim}")
print(f"  Number of cells: {domain.topology.index_map(domain.topology.dim).size_local}")
print(f"  Geometry dimension: {domain.geometry.dim}")

# Create vector function space
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))

# Print function space information
topology_vertices = domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
total_elements = domain.topology.index_map(domain.topology.dim).size_local + domain.topology.index_map(domain.topology.dim).num_ghosts
dofmap = V.dofmap
total_dofs = dofmap.index_map.size_local + dofmap.index_map.num_ghosts
block_size = dofmap.index_map_bs
total_vector_dofs = total_dofs * block_size

print(f"Topology vertices: {topology_vertices}")
print(f"Function space DOFs (quadratic): {total_dofs}")
print(f"Total DOFs (including all vector components): {total_vector_dofs}")
print(f"Total elements: {total_elements}")

# Print coordinate bounds for verification
print(f"\nMesh bounds:")
print(f"  X: [{x_nodes[:, 0].min():.4f}, {x_nodes[:, 0].max():.4f}]")
print(f"  Y: [{x_nodes[:, 1].min():.4f}, {x_nodes[:, 1].max():.4f}]")
print(f"  Z: [{x_nodes[:, 2].min():.4f}, {x_nodes[:, 2].max():.4f}]")


# ============================================================================
# MATERIAL PROPERTIES
# ============================================================================
E = 5.0e7      # Pa (Young's modulus)
nu = 0.28      # Poisson's ratio
rho0 = 1200.0  # kg/m^3 (density)

# Derived Lamé parameters
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


# ============================================================================
# BOUNDARY CONDITIONS - Fix nodes with z < -4.0 (geometrical approach)
# ============================================================================
print("\nBOUNDARY CONDITIONS SETUP")

# Define a function to identify nodes at z < -4.0 (bottom of bunny)
def fixed_boundary(x):
    return x[2] < -4.0

# Locate DOFs on the fixed boundary (geometrical approach)
boundary_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)

# Verify: manually find all function space DOF nodes at z < -4.0
dof_coords = V.tabulate_dof_coordinates()
manual_bottom_dofs = []
tol = 1e-6
for i, coord in enumerate(dof_coords):
    if coord[2] < -4.0:
        manual_bottom_dofs.append(i)

# Create zero displacement boundary condition
u_zero = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
bc_fixed = fem.dirichletbc(u_zero, boundary_dofs, V)

print(f"Fixed boundary at z < -4.0 (bottom):")
print(f"  Number of DOFs found by locate_dofs_geometrical: {len(boundary_dofs)}")
print(f"  Number of DOFs manually found at z < -4.0: {len(manual_bottom_dofs)}")
print(f"  DOF indices match: {set(boundary_dofs) == set(manual_bottom_dofs)}")
print(f"  Boundary condition: u = {u_zero}")
print(f"  Constrained scalar DOFs: {len(boundary_dofs) * block_size}")
print(f"  Free scalar DOFs: {total_vector_dofs - len(boundary_dofs) * block_size}")


# ============================================================================
# APPLIED LOADS - Nodal forces at top nodes (z > 4.0)
# ============================================================================
print("\nAPPLIED LOADS SETUP")

# Apply force on nodes with z > 4.0 (top of bunny)
# Matching C++ implementation
force_value = -2.0  # N (negative = downward in z-direction)

# Create external force vector
f_ext = fem.Function(V)
f_ext.x.array[:] = 0.0  # Initialize to zero

# Apply forces to nodes with z > 4.0
num_force_nodes = 0
loaded_nodes = []
for i in range(len(x_nodes)):
    if x_nodes[i, 2] > 4.0:
        f_ext.x.array[i * block_size + 2] = force_value  # z-component
        num_force_nodes += 1
        loaded_nodes.append((i, x_nodes[i]))

f_ext.x.scatter_forward()

total_force = force_value * num_force_nodes

print(f"Load applied at z > 4.0 (top):")
print(f"  Force per node: {force_value} N (z direction)")
print(f"  Number of loaded nodes: {num_force_nodes}")
print(f"  Total force applied: {total_force:.1f} N")
print(f"  Distribution: Equal distribution across all top nodes")


# ============================================================================
# MATERIAL MODEL AND KINEMATICS
# ============================================================================
# Define test and trial functions
v = ufl.TestFunction(V)
u = fem.Function(V)  # displacement field (unknown)
u_old = fem.Function(V)  # Previous displacement
v_old = fem.Function(V)  # Previous velocity

# Body force (zero for this problem)
B = fem.Constant(domain, default_scalar_type((0, 0, 0)))

# Spatial dimension
d = len(u)
# Identity tensor
I = ufl.Identity(d)
# Deformation gradient
F = I + ufl.grad(u)
# Right Cauchy-Green tensor
C = F.T * F
# Trace of C = F^T * F
trFtF = ufl.tr(C)
# Compute F * F^T * F
FFt = F * F.T
FFtF = FFt * F

# Material model (St. Venant-Kirchhoff):
# P = λ*(0.5*tr(F^T*F) - 1.5)*F + μ*(F*F^T*F - F)
lambda_factor = lmbda * (0.5 * trFtF - 1.5)
P = lambda_factor * F + mu * (FFtF - F)


# ============================================================================
# TIME INTEGRATION SETUP (Backward Euler method)
# ============================================================================
dt = 1e-2  # Time step (0.01 seconds)
n_steps = 100  # Number of time steps
t_final = n_steps * dt  # Total simulation time

print("\nTIME INTEGRATION SETUP")
print(f"Method: Backward Euler")
print(f"Time step (dt): {dt} s")
print(f"Number of steps: {n_steps}")
print(f"Total simulation time: {t_final} s")


# ============================================================================
# VARIATIONAL FORM (Backward Euler)
# ============================================================================
# Quadrature degree 3 for tetrahedra = 5-point quadrature rule
# This matches C++ which uses tet5pt (5-point tetrahedral quadrature)
metadata = {"quadrature_degree": 3}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Backward Euler time discretization:
# Current velocity: v = (u - u_old) / dt
v_current = (u - u_old) / dt

# Current acceleration: a = (v_current - v_old) / dt
a_current = (v_current - v_old) / dt

# Backward Euler variational form:
# ρ ∫ a·v dx + ∫ ∇v:P dx = ∫ v·B dx + v·f_ext
F_form = (rho0 * ufl.inner(a_current, v) * dx +
          ufl.inner(ufl.grad(v), P) * dx -
          ufl.inner(v, B) * dx -
          ufl.inner(v, f_ext) * dx)


# ============================================================================
# SOLVER SETUP
# ============================================================================
# Tolerances matching C++ (1e-6 outer_tol)
problem = NonlinearProblem(
    F_form,
    u,
    bcs=[bc_fixed],
    petsc_options={
        "snes_type": "newtonls",
        "snes_atol": 1e-6,  # Matching C++ outer_tol
        "snes_rtol": 1e-6,  # Matching C++ outer_tol
        "snes_stol": 1e-6,  # Matching C++ outer_tol
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    petsc_options_prefix="bunny_debug",
)

print("\nNONLINEAR SOLVER SETUP")
print(f"Solver type: Newton line search (SNES)")
print(f"Linear solver: Direct LU (MUMPS)")
print(f"Absolute tolerance: 1e-6")
print(f"Relative tolerance: 1e-6")


# ============================================================================
# TIME STEPPING LOOP
# ============================================================================
# log.set_log_level(log.LogLevel.INFO)

print("\nSTARTING DYNAMIC ANALYSIS")

# Initialize state variables (bunny starts from rest)
u_old.x.array[:] = 0.0
v_old.x.array[:] = 0.0

# Time stepping loop
for n in range(n_steps):
    t = n * dt
    
    # Solve for displacement at current time step
    problem.solve()
    converged = problem.solver.getConvergedReason()
    num_its = problem.solver.getIterationNumber()
    assert converged > 0, f"Newton solver did not converge (reason {converged})."
    u.x.scatter_forward()
    
    # Update velocity using Backward Euler
    v_new = (u.x.array - u_old.x.array) / dt
    
    # Update old values for next time step
    u_old.x.array[:] = u.x.array[:]
    v_old.x.array[:] = v_new[:]
    
    # Print progress
    if n % 10 == 0 or n < 5:
        max_disp = np.max(np.linalg.norm(u.x.array.reshape(-1, 3), axis=1))
        max_vel = np.max(np.linalg.norm(v_old.x.array.reshape(-1, 3), axis=1))
        print(f"Step {n:3d}, Time {t:.4f} s, Iterations {num_its}")
        print(f"  Max displacement: {max_disp:.6e} m, Max velocity: {max_vel:.6e} m/s")

print("\nDYNAMIC ANALYSIS COMPLETE")


# ============================================================================
# SAVE FINAL POSITIONS (Matching C++ output format)
# ============================================================================
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Get final deformed positions (like C++ does)
x_final = x_nodes[:, 0] + u.x.array[0::3]
y_final = x_nodes[:, 1] + u.x.array[1::3]
z_final = x_nodes[:, 2] + u.x.array[2::3]

# Save final positions to file for comparison with C++
positions_file = os.path.join(output_dir, "bunny_debug_final_positions.txt")
with open(positions_file, 'w') as f:
    f.write("# Final node positions after dynamic deformation\n")
    f.write(f"# Total nodes: {len(x_nodes)}\n")
    f.write(f"# Time steps: {n_steps}, Final time: {t_final} s\n")
    f.write(f"# Format: node_id x y z\n")
    for i in range(len(x_nodes)):
        f.write(f"{i} {x_final[i]:.17e} {y_final[i]:.17e} {z_final[i]:.17e}\n")

print(f"\nWrote final node positions to {positions_file}")
print(f"  Total nodes: {len(x_nodes)}")
print(f"  Format: node_id x y z (17 decimal precision)")
print(f"  For comparison with C++ output")
