"""
Nonlinear 3D bunny mesh dynamic analysis (St. Venant-Kirchhoff) solved with DOLFINx.
Uses Generalized-α time integration with constant load and time stepping.
"""
from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import pyvista
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, mesh
from tetgen_mesh_loader import load_tetgen_mesh_from_files

import os


# ============================================================================
# CONFIGURATION
# ============================================================================
# Set to True to show interactive visualization, else just save GIF
VIS = True  

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
print(f"  Function space DOFs: {V.dofmap.index_map.size_local * V.dofmap.index_map_bs}")

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

print(f"\nMaterial Properties:")
print(f"  E  = {E:.2e} Pa")
print(f"  nu = {nu}")
print(f"  rho = {rho0} kg/m^3")
print(f"  Derived:")
print(f"    μ = {mu:.4e} Pa")
print(f"    λ = {lmbda:.4e} Pa")


# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================
# Fix nodes with z < -4.0 (bottom of bunny)
def bottom(x):
    return x[2] < -4.0

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, bottom)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

u_bc = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_bc, boundary_dofs, V)
bcs = [bc]

print(f"\nBoundary Conditions:")
print(f"  Fixed DOFs (z < -4.0): {len(boundary_dofs)}")


# ============================================================================
# EXTERNAL FORCES
# ============================================================================
# Apply force on nodes with z > 4.0 (top of bunny)
force_value = -50.0  # N (negative = downward in z-direction)

# Create external force vector
f_ext = fem.Function(V)
f_ext_array = f_ext.x.array.reshape(-1, 3)

# Apply forces to nodes with z > 4.0
for i in range(len(x_nodes)):
    if x_nodes[i, 2] > 4.0:
        f_ext_array[i, 2] = force_value

f_ext.x.scatter_forward()

print(f"\nApplied force: {force_value} N in z-direction")
print(f"Number of nodes with z > 4.0: {np.sum(x_nodes[:, 2] > 4.0)}")
print(f"Total external force magnitude: {np.linalg.norm(f_ext_array):.4f} N")


# ============================================================================
# TIME INTEGRATION SETUP (Generalized-α method)
# ============================================================================
# Time integration variables
dt = 1e-2  # Time step (0.01 seconds)
n_steps = 100  # Number of time steps
t_final = n_steps * dt  # Total simulation time

# Generalized-α method parameters
alpha_m_val = 0.2  # Mass parameter
alpha_f_val = 0.4  # Force parameter
gamma_val = 0.5 + alpha_f_val - alpha_m_val  # Velocity parameter
beta_val = (gamma_val + 0.5)**2 / 4.0  # Displacement parameter

alpha_m = fem.Constant(domain, alpha_m_val)
alpha_f = fem.Constant(domain, alpha_f_val)
gamma = fem.Constant(domain, gamma_val)
beta = fem.Constant(domain, beta_val)

print(f"\nTime Integration (Generalized-α):")
print(f"  Time step: {dt} s")
print(f"  Total steps: {n_steps}")
print(f"  Total time: {t_final} s")
print(f"  α_m = {alpha_m_val}, α_f = {alpha_f_val}")
print(f"  β = {beta_val:.4f}, γ = {gamma_val:.4f}")

# State variables for time integration
u_old = fem.Function(V)  # Previous displacement
v_old = fem.Function(V)  # Previous velocity
a_old = fem.Function(V)  # Previous acceleration


# ============================================================================
# DAMPING
# ============================================================================
# Rayleigh damping coefficients
eta_m = fem.Constant(domain, 0.0)  # Mass proportional damping
eta_k = fem.Constant(domain, 0.0)  # Stiffness proportional damping


# ============================================================================
# KINEMATICS AND MATERIAL MODEL
# ============================================================================
# Define test and trial functions
v = ufl.TestFunction(V)
u = fem.Function(V)  # displacement field (unknown)

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

# Material model (St. Venant-Kirchhoff)
# P = λ*(0.5*tr(F^T*F) - 1.5)*F + μ*(F*F^T*F - F)
lambda_factor = lmbda * (0.5 * trFtF - 1.5)
P = lambda_factor * F + mu * (FFtF - F)


# ============================================================================
# VARIATIONAL FORM
# ============================================================================
# Set quadrature degree
metadata = {"quadrature_degree": 4}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Mass matrix form
def m(u, v):
    return rho0 * ufl.inner(u, v) * dx

# Damping form
def c(u, v):
    return eta_m * m(u, v) + eta_k * ufl.inner(ufl.grad(v), P) * dx

# Generalized-α update functions
def update_a(u_new, u_old, v_old, a_old):
    """Update acceleration using Generalized-α method"""
    return (u_new - u_old - dt * v_old) / (beta * dt**2) - (1 - 2*beta) / (2*beta) * a_old

def update_v(a_new, u_old, v_old, a_old):
    """Update velocity using Generalized-α method"""
    return v_old + dt * ((1 - gamma) * a_old + gamma * a_new)

def avg(x_old, x_new, alpha):
    """Compute weighted average for Generalized-α method"""
    return alpha * x_old + (1 - alpha) * x_new

# Compute acceleration and velocity
a_new = update_a(u, u_old, v_old, a_old)
v_new = update_v(a_new, u_old, v_old, a_old)

# Dynamic form with Generalized-α method: m(ü,v) + c(ẋ,v) + k(u,v) = L(v)
F_form = (m(avg(a_old, a_new, alpha_m), v) + 
          c(avg(v_old, v_new, alpha_f), v) + 
          ufl.inner(ufl.grad(v), P) * dx - 
          ufl.inner(v, f_ext) * dx)

print("Using 4 point quadrature")
print("Problem Setup Complete")


# ============================================================================
# SOLVER SETUP
# ============================================================================
# Create nonlinear problem using new API
problem = NonlinearProblem(
    F_form,
    u,
    bcs=bcs,
    petsc_options={
        "snes_type": "newtonls",
        "snes_monitor": None,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_stol": 1e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    petsc_options_prefix="bunny_dynamic",
)

print("\nNewton Solver Setup:")
print("  Type: newtonls")
print("  Linear solver: LU (MUMPS)")
print("  Tolerances: atol=1e-8, rtol=1e-8, stol=1e-8")


# ============================================================================
# OUTPUT DIRECTORY SETUP
# ============================================================================
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)


# ============================================================================
# TIME STEPPING LOOP
# ============================================================================
log.set_log_level(log.LogLevel.INFO)

print(f"\nStarting dynamic time integration...")

# Initialize state variables (bunny starts from rest)
u_old.x.array[:] = 0.0
v_old.x.array[:] = 0.0
a_old.x.array[:] = 0.0

# Storage for animation frames (store every Nth step to save memory)
frame_skip = max(1, n_steps // 50)  # Store ~50 frames
displacement_frames = []
time_frames = []

# Time stepping loop
for n in range(n_steps):
    t = n * dt
    
    # Solve for displacement at current time step
    problem.solve()
    converged = problem.solver.getConvergedReason()
    num_its = problem.solver.getIterationNumber()
    assert converged > 0, f"Newton solver did not converge (reason {converged})."
    u.x.scatter_forward()
    
    # Update fields using Generalized-α method
    def update_fields(u_new, u_old, v_old, a_old):
        """Update fields at the end of each time step using Generalized-α method"""
        # Get vectors (references)
        u_vec, u0_vec = u_new.x.array, u_old.x.array
        v0_vec, a0_vec = v_old.x.array, a_old.x.array
        
        # Use update functions with vector arguments
        a_vec = (u_vec - u0_vec - dt * v0_vec) / (beta_val * dt**2) - (1 - 2*beta_val) / (2*beta_val) * a0_vec
        v_vec = v0_vec + dt * ((1 - gamma_val) * a0_vec + gamma_val * a_vec)
        
        # Update (u_old <- u_new)
        v_old.x.array[:] = v_vec
        a_old.x.array[:] = a_vec
        u_old.x.array[:] = u_new.x.array
    
    # Update fields for next time step
    update_fields(u, u_old, v_old, a_old)
    
    # Print progress every 10 steps
    if n % 10 == 0 or n < 5:
        max_disp = np.max(np.linalg.norm(u.x.array.reshape(-1, 3), axis=1))
        max_vel = np.max(np.linalg.norm(v_old.x.array.reshape(-1, 3), axis=1))
        print(f"Time step {n:3d}, Time {t:.4f} s, Iterations {num_its}")
        print(f"  Max displacement: {max_disp:.6f} m")
        print(f"  Max velocity: {max_vel:.6f} m/s")
    
    # Store frames for animation
    if n % frame_skip == 0 or n == n_steps - 1:
        displacement_frames.append(u.x.array.copy())
        time_frames.append(t)


# ============================================================================
# VISUALIZATION AND ANIMATION
# ============================================================================
print("\nGenerating visualization...")

# Create PyVista mesh from DOLFINx domain
import dolfinx.plot as plot
topology, cells, geometry = plot.vtk_mesh(V)

# Create animated GIF with time evolution
plotter = pyvista.Plotter(off_screen=not VIS)
gif_path = os.path.join(output_dir, "deformation_bunny_dynamic.gif")
plotter.open_gif(gif_path, fps=10)

# Animation loop through stored frames
for frame_idx, (disp_array, t) in enumerate(zip(displacement_frames, time_frames)):
    plotter.clear()
    
    # Create mesh for this time step
    function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)
    
    # Add displacement field
    displacement_3d = disp_array.reshape(-1, 3)
    function_grid["u"] = displacement_3d
    
    # Compute magnitude
    magnitude_array = np.linalg.norm(displacement_3d, axis=1)
    function_grid["mag"] = magnitude_array
    
    # Create warped mesh for visualization
    warped = function_grid.warp_by_vector("u", factor=1.0)
    warped.set_active_scalars("mag")
    
    # Add mesh to plotter
    plotter.add_mesh(warped, show_edges=True, lighting=False, scalars="mag",
                     cmap="viridis", scalar_bar_args={'title': 'Displacement (m)'})
    plotter.add_text(f"Bunny Dynamic (t = {t:.3f} s)", 
                     position='upper_edge', font_size=12)
    
    # Write frame
    plotter.write_frame()

plotter.close()
print(f"Saved animated GIF to: {gif_path}")


# ============================================================================
# FINAL OUTPUT AND COMPARISON DATA
# ============================================================================
# Get final deformed positions
x_final = x_nodes[:, 0] + u.x.array[0::3]
y_final = x_nodes[:, 1] + u.x.array[1::3]
z_final = x_nodes[:, 2] + u.x.array[2::3]

# Save final positions to file for comparison
positions_file = os.path.join(output_dir, "bunny_dynamic_final_positions.txt")
with open(positions_file, 'w') as f:
    f.write("# Final node positions (x, y, z) after dynamic deformation\n")
    f.write(f"# Total nodes: {len(x_nodes)}\n")
    f.write(f"# Time steps: {n_steps}, Final time: {t_final} s\n")
    f.write(f"# Format: node_id x y z\n")
    for i in range(len(x_nodes)):
        f.write(f"{i} {x_final[i]:.17e} {y_final[i]:.17e} {z_final[i]:.17e}\n")

print(f"Saved final node positions to: {positions_file}")

# Print final statistics
print(f"\nFinal Statistics:")
print(f"  Max displacement: {np.max(np.linalg.norm(displacement_3d, axis=1)):.6f} m")
print(f"  Max velocity: {np.max(np.linalg.norm(v_old.x.array.reshape(-1, 3), axis=1)):.6f} m/s")

