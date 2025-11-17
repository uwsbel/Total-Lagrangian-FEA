"""
Nonlinear 3D beam dynamic analysis (compressible Neo-Hookean) solved with DOLFINx.
Uses Newmark-β time integration with constant load and time stepping.
Creates an animated GIF of the  deformation using PyVista.
"""
from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import pyvista
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, mesh, plot

import os


# ============================================================================
# GEOMETRY AND MESH SETUP
# ============================================================================
# TODO: Import mesh from a .ele and .node file (for now, create a mesh manually)
# Geometry and function space: length L in x, width W in y, height H in z
L = 10.0  # Length
W = 2.0   # Width (increased from 1)
H = 1.5   # Height (increased from 1)
domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, H]], [15, 6, 4], mesh.CellType.tetrahedron)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))


# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================
def left(x):
    # x = 0 plane
    return np.isclose(x[0], 0)


def right(x):
    # x = L plane
    return np.isclose(x[0], L)


fdim = domain.topology.dim - 1
# Boundary facet sets: left (clamped) and right (traction)
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

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
B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

v = ufl.TestFunction(V)
u = fem.Function(V)  # displacement field (unknown)


# ============================================================================
# TIME INTEGRATION SETUP (Generalized-α method)
# ============================================================================
# Time integration variables
dt = 1e-1  # Time step
t_final = 2.0  # Total simulation time
n_steps = int(t_final / dt)

# Generalized-α method parameters (from reference)
alpha_m_val = 0.2  # Mass parameter
alpha_f_val = 0.4  # Force parameter
gamma_val = 0.5 + alpha_f_val - alpha_m_val  # Velocity parameter
beta_val = (gamma_val + 0.5)**2 / 4.0  # Displacement parameter

alpha_m = fem.Constant(domain, alpha_m_val)
alpha_f = fem.Constant(domain, alpha_f_val)
gamma = fem.Constant(domain, gamma_val)
beta = fem.Constant(domain, beta_val)

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
I = ufl.variable(ufl.Identity(d))
# Deformation gradient
F = ufl.variable(I + ufl.grad(u))
# Right Cauchy-Green tensor
C = ufl.variable(F.T * F)
# Invariants
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))


# ============================================================================
# MATERIAL MODEL
# ============================================================================
# Material (compressible Neo-Hookean)
E = default_scalar_type(1.0e4)
nu = default_scalar_type(0.3)
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
rho = fem.Constant(domain, 1000.0)  # Density for dynamic analysis

# Stored strain energy density
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
# First Piola–Kirchhoff stress
P = ufl.diff(psi, F)

# Mass matrix form (for inertia and Rayleigh damping)
def m(u, v):
    return rho * ufl.inner(u, v) * dx

# Damping form (Rayleigh damping)
def c(u, v):
    return eta_m * m(u, v) + eta_k * ufl.inner(ufl.grad(v), P) * dx


# ============================================================================
# VARIATIONAL FORM
# ============================================================================
# Variational form with inertia terms for dynamic analysis
metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Generalized-α update functions (from reference)
def update_a(u_new, u_old, v_old, a_old):
    """Update acceleration using Generalized-α method"""
    return (u_new - u_old - dt * v_old) / (beta * dt**2) - (1 - 2*beta) / (2*beta) * a_old

def update_v(a_new, u_old, v_old, a_old):
    """Update velocity using Generalized-α method"""
    return v_old + dt * ((1 - gamma) * a_old + gamma * a_new)

def avg(x_old, x_new, alpha):
    """Compute weighted average for Generalized-α method"""
    return alpha * x_old + (1 - alpha) * x_new

# For dynamic analysis, we need to solve for u at time t
# The acceleration term will be computed using the current and previous solutions
# We'll define the acceleration in terms of the unknown u and known previous states
a_new = update_a(u, u_old, v_old, a_old)
v_new = update_v(a_new, u_old, v_old, a_old)

# Dynamic form with Generalized-α method: m(ü,v) + c(ẋ,v) + k(u,v) = L(v)
# where the terms are evaluated at intermediate times
F = (m(avg(a_old, a_new, alpha_m), v) + 
     c(avg(v_old, v_new, alpha_f), v) + 
     ufl.inner(ufl.grad(v), P) * dx - 
     ufl.inner(v, B) * dx - 
     ufl.inner(v, T) * ds(2))


# ============================================================================
# SOLVER SETUP
# ============================================================================
problem = NonlinearProblem(F, u, bcs)

solver = NewtonSolver(domain.comm, problem)
# Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"


# ============================================================================
# VISUALIZATION SETUP
# ============================================================================
# PyVista setup for animated GIF
plotter = pyvista.Plotter()

# Ensure output directory exists, and use a path relative to script location
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
gif_path = os.path.join(output_dir, "deformation_dynamic.gif")
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
us = fem.Expression(ufl.sqrt(sum([u[i]**2 for i in range(len(u))])), Vs.element.interpolation_points())
magnitude.interpolate(us)
warped["mag"] = magnitude.x.array


# ============================================================================
# TIME STEPPING LOOP
# ============================================================================
log.set_log_level(log.LogLevel.INFO)

# Set constant load (applied immediately at t=0)
T.value[1] = -200.0  # Constant load in -y direction

print(f"Starting dynamic analysis:")
print(f"  Time step: {dt}")
print(f"  Total time: {t_final}")
print(f"  Number of steps: {n_steps}")
print(f"  Load: {T.value}")
print(f"  Generalized-α: α_m={alpha_m_val}, α_f={alpha_f_val}, β={beta_val}, γ={gamma_val}")

# Initialize state variables (beam starts from rest)
u_old.x.array[:] = 0.0
v_old.x.array[:] = 0.0
a_old.x.array[:] = 0.0

# Time stepping loop
for n in range(n_steps):
    t = n * dt
    
    # Solve for displacement at current time step
    num_its, converged = solver.solve(u)
    assert (converged)
    u.x.scatter_forward()
    
    # Update fields using Generalized-α method (from reference)
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
        print(f"Time step {n}, Time {t:.4f}, Iterations {num_its}")
        print(f"  Max displacement: {max_disp:.6f}")
        print(f"  Max velocity: {max_vel:.6f}")
    
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
