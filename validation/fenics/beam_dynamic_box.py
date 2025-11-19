"""
Nonlinear 3D beam dynamic analysis solved with DOLFINx.
Uses Backward Euler time integration with constant load and time stepping.
This version uses create_box to generate the mesh instead of loading from TetGen files.
"""
import os
import numpy as np
import ufl
import pyvista
import matplotlib.pyplot as plt
from mpi4py import MPI

from dolfinx import fem, mesh, plot, log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem


# ============================================================================
# GEOMETRY AND MESH SETUP
# ============================================================================
# Beam dimensions
L = 5.0   # Length (x)
W = 2.0   # Width (y)
H = 1.0   # Height (z)

# Mesh resolution: number of divisions along each axis
# Higher values = finer mesh
nx, ny, nz = 3, 2, 1  # Adjust these for different mesh resolutions

# Create box mesh using create_box
# create_box(comm, [p0, p1], [nx, ny, nz], cell_type=CellType.tetrahedron)
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [L, W, H]],
    [nx, ny, nz],
    cell_type=mesh.CellType.tetrahedron
)

# Create function space with quadratic elements (Lagrange P2)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))

# Print total nodes and elements (all ranks print)
topology_vertices = domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
total_elements = domain.topology.index_map(domain.topology.dim).size_local + domain.topology.index_map(domain.topology.dim).num_ghosts
# Function space DOFs (quadratic elements - includes mid-edge nodes)
dofmap = V.dofmap
total_dofs = dofmap.index_map.size_local + dofmap.index_map.num_ghosts
print(f"Created box mesh: {nx}x{ny}x{nz} divisions")
print(f"Topology vertices: {topology_vertices}")
print(f"Function space DOFs (quadratic): {total_dofs}")
print(f"Total elements: {total_elements}")

# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================
def left(x):
    # x = 0 plane
    return np.isclose(x[0], 0, atol=1e-6)

def right(x):
    # x = L plane
    return np.isclose(x[0], L, atol=1e-6)

fdim = domain.topology.dim - 1
# Boundary facet sets: left (clamped) and right (traction)
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

# Print node indices on left and right boundary
# Get all unique vertex indices connected to the selected facets

def facet_vertices(domain, facets, fdim):
    geometry_dim = domain.geometry.dim
    topology = domain.topology
    topology.create_connectivity(fdim, 0)
    conn = topology.connectivity(fdim, 0)
    vertices = np.unique(np.hstack([conn.links(i) for i in facets]))
    return vertices

left_nodes = facet_vertices(domain, left_facets, fdim)
right_nodes = facet_vertices(domain, right_facets, fdim)

print("Left boundary node indices:", left_nodes)
print("Right boundary node indices:", right_nodes)

# Concatenate and sort by facet index. Left facets tagged 1, right facets tagged 2
marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

# Homogeneous Dirichlet BC (clamped) on left boundary: u = 0
u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)

left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs = [fem.dirichletbc(u_bc, left_dofs, V)]
# ============================================================================
# LOADS AND FUNCTION SPACE
# ============================================================================
# Body force (zero) and boundary traction (constant load applied on right boundary for dynamic analysis)
# TODO: Apply nodal forces matching C++ implementation
B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

v = ufl.TestFunction(V)
u = fem.Function(V)  # displacement field (unknown)


# ============================================================================
# TIME INTEGRATION SETUP (Backward Euler method)
# ============================================================================
# Time integration variables
dt = 1e-1  # Time step
n_steps = 50  # Number of time steps (matching C++ iterations)
t_final = n_steps * dt  # Total simulation time: 50 * 0.1 = 5.0 seconds

# Backward Euler: fully implicit, first-order accurate
# No parameters needed - formulas are directly implemented

# State variables for time integration
u_old = fem.Function(V)  # Previous displacement
v_old = fem.Function(V)  # Previous velocity
a_old = fem.Function(V)  # Previous acceleration


# ============================================================================
# DAMPING
# ============================================================================
# Rayleigh damping coefficients
# Not used here, but can be added if needed
eta_m = fem.Constant(domain, 0.0)  # Mass proportional damping
eta_k = fem.Constant(domain, 0.0)   # Stiffness proportional damping


# ============================================================================
# KINEMATICS
# ============================================================================
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
# Compute F * F^T * F (needed for material model)
FFt = F * F.T
FFtF = FFt * F


# ============================================================================
# MATERIAL MODEL
# ============================================================================
# Material properties
E = default_scalar_type(7.0e8)  # Young's modulus: 7×10⁸ Pa
nu = default_scalar_type(0.33)  # Poisson's ratio: 0.33
mu = fem.Constant(domain, E / (2 * (1 + nu)))  # Shear modulus
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))  # Lamé's first parameter
rho = fem.Constant(domain, 2700.0)  # Density: 2700 kg/m³

# Material model (St. Venant-Kirchhoff):
# P = λ*(0.5*tr(F^T*F) - 1.5)*F + μ*(F*F^T*F - F)
lambda_factor = lmbda * (0.5 * trFtF - 1.5)
P = lambda_factor * F + mu * (FFtF - F)

# Mass matrix form
def m(u, v):
    return rho * ufl.inner(u, v) * dx

# Damping form
def c(u, v):
    return eta_m * m(u, v) + eta_k * ufl.inner(ufl.grad(v), P) * dx


# ============================================================================
# VARIATIONAL FORM
# ============================================================================
# Variational form with inertia terms for dynamic analysis
metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Backward Euler update functions
def update_a(u_new, u_old, v_old, a_old):
    """Update acceleration using Backward Euler method"""
    return (u_new - u_old - dt * v_old) / (dt**2)

def update_v(a_new, u_old, v_old, a_old):
    """Update velocity using Backward Euler method"""
    return v_old + dt * a_new

# For dynamic analysis, we need to solve for u at time t
# The acceleration term will be computed using the current and previous solutions
# We'll define the acceleration in terms of the unknown u and known previous states
a_new = update_a(u, u_old, v_old, a_old)
v_new = update_v(a_new, u_old, v_old, a_old)

# Dynamic form with Backward Euler method: m(ü,v) + c(ẋ,v) + k(u,v) = L(v)
# All terms are evaluated at time t_{n+1} (fully implicit)
F = (m(a_new, v) + 
     c(v_new, v) + 
     ufl.inner(ufl.grad(v), P) * dx - 
     ufl.inner(v, B) * dx - 
     ufl.inner(v, T) * ds(2))


# ============================================================================
# SOLVER SETUP
# ============================================================================
problem = NonlinearProblem(
    F,
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
    petsc_options_prefix="beam_dynamic_box",
)


# ============================================================================
# VISUALIZATION SETUP
# ============================================================================
# PyVista setup for animated GIF
plotter = pyvista.Plotter()
plotter.show_axes()  # Add coordinate system axes

# Ensure output directory exists, and use a path relative to script location
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
gif_path = os.path.join(output_dir, "deformation_dynamic_box_backward_euler.gif")
plotter.open_gif(gif_path, fps=10)

topology, cells, geometry = plot.vtk_mesh(u.function_space)
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)

values = np.zeros((geometry.shape[0], 3))
values[:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
function_grid["u"] = values
function_grid.set_active_vectors("u")

# Warp mesh by deformation
warped = function_grid.warp_by_vector("u", factor=1)

# Add mesh to plotter and visualize
plotter.add_mesh(warped, show_edges=True, lighting=False)

# Compute magnitude of displacement to visualize in GIF
Vs = fem.functionspace(domain, ("Lagrange", 2))
magnitude = fem.Function(Vs)
us = fem.Expression(ufl.sqrt(sum([u[i]**2 for i in range(len(u))])), Vs.element.interpolation_points)
magnitude.interpolate(us)
warped["mag"] = magnitude.x.array


# ============================================================================
# TIME STEPPING LOOP
# ============================================================================
# log.set_log_level(log.LogLevel.INFO)

# Set constant load (applied immediately at t=0)
# Load: 5000 N total in z-direction, distributed on right boundary (x = L)
# Right face area = W × H = 2.0 × 1.0 = 2.0 m²
# Traction (force per unit area) = 5000 N / 2.0 m² = 2500 N/m²
T.value[2] = -500000.0  # Traction in +z direction (gives total force of 5000 N)

print(f"Starting dynamic analysis:")
print(f"  Time step: {dt}")
print(f"  Total time: {t_final}")
print(f"  Number of steps: {n_steps}")
print(f"  Load: {T.value}")
print(f"  Integrator: Backward Euler")

# Initialize state variables (beam starts from rest)
u_old.x.array[:] = 0.0
v_old.x.array[:] = 0.0
a_old.x.array[:] = 0.0

# ============================================================================
# FIND MIDPOINT OF RIGHT FACE FOR DISPLACEMENT TRACKING
# ============================================================================
# Midpoint of right face: (L, W/2, H/2)
midpoint_target = np.array([L, W/2, H/2])

# Get DOF coordinates to find closest node
dof_coords = V.tabulate_dof_coordinates()
dof_distances = np.linalg.norm(dof_coords - midpoint_target, axis=1)
closest_dof_idx = np.argmin(dof_distances)
closest_dof_coord = dof_coords[closest_dof_idx]

print(f"Midpoint of right face (target): {midpoint_target}")
print(f"Closest DOF found: {closest_dof_coord} (DOF index: {closest_dof_idx}, distance: {dof_distances[closest_dof_idx]:.6f})")

# For vector function spaces, the displacement array is organized as:
# [u0_x, u0_y, u0_z, u1_x, u1_y, u1_z, ...]
# So for DOF index i, the node index is i // 3, and component is i % 3
# To get z-component: find the node index and access z (index 2)
node_idx = closest_dof_idx // 3
print(f"Node index: {node_idx} (DOF {closest_dof_idx} corresponds to node {node_idx}, component {closest_dof_idx % 3})")

# Arrays to store time and displacement data
time_history = []
z_displacement_history = []

# Time stepping loop
for n in range(n_steps):
    t = n * dt
    
    # Solve for displacement at current time step
    problem.solve()
    converged = problem.solver.getConvergedReason()
    num_its = problem.solver.getIterationNumber()
    assert converged > 0, f"Newton solver did not converge (reason {converged})."
    u.x.scatter_forward()
    
    # Update fields using Backward Euler method
    def update_fields(u_new, u_old, v_old, a_old):
        """Update fields at the end of each time step using Backward Euler method"""
        # Get vectors (references)
        u_vec, u0_vec = u_new.x.array, u_old.x.array
        v0_vec, a0_vec = v_old.x.array, a_old.x.array
        
        # Use update functions with vector arguments (Backward Euler)
        a_vec = (u_vec - u0_vec - dt * v0_vec) / (dt**2)
        v_vec = v0_vec + dt * a_vec
        
        # Update (u_old <- u_new)
        v_old.x.array[:] = v_vec
        a_old.x.array[:] = a_vec
        u_old.x.array[:] = u_new.x.array
    
    # Update fields for next time step
    update_fields(u, u_old, v_old, a_old)
    
    # Track z-displacement at midpoint of right face
    # Reshape displacement array: [u0x, u0y, u0z, u1x, u1y, u1z, ...] -> [[u0x, u0y, u0z], [u1x, u1y, u1z], ...]
    u_array = u.x.array.reshape(-1, 3)
    if node_idx < len(u_array):
        z_disp = u_array[node_idx, 2]  # z-component (index 2) of the node
        time_history.append(t)
        z_displacement_history.append(z_disp)
    
    # Print progress every 10 steps
    if n % 10 == 0 or n < 5:
        max_disp = np.max(np.linalg.norm(u.x.array.reshape(-1, 3), axis=1))
        max_vel = np.max(np.linalg.norm(v_old.x.array.reshape(-1, 3), axis=1))
        # print(f"Time step {n}, Time {t:.4f}, Iterations {num_its}")
        # print(f"  Max displacement: {max_disp:.6f}")
        # print(f"  Max velocity: {max_vel:.6f}")
    
    # Update visualization every 2 steps for GIF
    if n % 2 == 0:
        function_grid["u"][:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
        magnitude.interpolate(us)
        warped.set_active_scalars("mag")
        warped_n = function_grid.warp_by_vector(factor=1)
        warped.points[:, :] = warped_n.points
        warped.point_data["mag"][:] = magnitude.x.array
        plotter.update_scalar_bar_range([0, 1])
        plotter.write_frame()

plotter.close()
print(f"Saved GIF to {gif_path}")

# ============================================================================
# PLOT Z-DISPLACEMENT VS TIME
# ============================================================================
plt.figure(figsize=(10, 6))
plt.plot(time_history, z_displacement_history, 'b-', linewidth=2, label=f'Z-displacement at right face midpoint (Node {node_idx})')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Z-displacement (m)', fontsize=12)
plt.title(f'Z-displacement vs Time at Midpoint of Right Face\n(Node {node_idx}, DOF {closest_dof_idx}, at {closest_dof_coord})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()

# Save plot
plot_path = os.path.join(output_dir, "z_displacement_vs_time_backward_euler.png")
plt.savefig(plot_path, dpi=150)
print(f"Saved displacement plot to {plot_path}")
plt.show()

