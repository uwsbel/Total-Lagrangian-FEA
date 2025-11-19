"""
Nonlinear 3D beam dynamic analysis (compressible Neo-Hookean) solved with DOLFINx.
Uses Generalized-α time integration with constant load and time stepping.
"""
import os
import numpy as np
import ufl

from dolfinx import fem, mesh, plot, log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from tetgen_mesh_loader import load_tetgen_mesh

import pyvista as pv


# ============================================================================
# GEOMETRY AND MESH SETUP
# ============================================================================
# Resolution selection: 0 (RES_0), 2 (RES_2), or 4 (RES_4)
RES = 0
# VTK output option: Set to True to save VTK files, False to skip
SAVE_VTK = False

# Load TetGen mesh using the tetgen_mesh_loader.py module
domain, x_tetgen = load_tetgen_mesh("beam_3x2x1", resolution=RES)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))

# Beam dimensions (for boundary conditions)
L = 3.0   # Length (x)
W = 2.0   # Width (y)
H = 1.0   # Height (z)


# Print total nodes and elements (all ranks print)
topology_vertices = domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
total_elements = domain.topology.index_map(domain.topology.dim).size_local + domain.topology.index_map(domain.topology.dim).num_ghosts
# Function space DOFs (quadratic elements - includes mid-edge nodes)
dofmap = V.dofmap
total_dofs = dofmap.index_map.size_local + dofmap.index_map.num_ghosts
print(f"Loaded TetGen mesh: beam_3x2x1 (RES_{RES})")
print(f"Topology vertices: {topology_vertices}")
print(f"Function space DOFs (quadratic): {total_dofs}")
print(f"Total elements: {total_elements}")


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
# TODO: Apply nodal forces matching C++ implementation
B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

v = ufl.TestFunction(V)
u = fem.Function(V)  # displacement field (unknown)


# ============================================================================
# TIME INTEGRATION SETUP (Generalized-α method)
# ============================================================================
# Time integration variables
dt = 1e-1  # Time step
n_steps = 50  # Number of time steps (matching C++ iterations)
t_final = n_steps * dt  # Total simulation time: 50 * 0.1 = 5.0 seconds

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
    petsc_options_prefix="beam_dynamic",
)


# ============================================================================
# DATA OUTPUT SETUP (VTK)
# ============================================================================
if SAVE_VTK:
    # Ensure output directory exists, and use a path relative to script location
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Create VTK subdirectory for all VTK files
    vtk_dir = os.path.join(output_dir, "beam_dynamic_vtk")
    os.makedirs(vtk_dir, exist_ok=True)

    # Create VTK mesh from domain (mesh is static, only fields change over time)
    topology, cells, geometry = plot.vtk_mesh(u.function_space)
    pv_mesh = pv.UnstructuredGrid(topology, cells, geometry)

    # Save base mesh once
    mesh_path = os.path.join(vtk_dir, "beam_dynamic_mesh.vtk")
    pv_mesh.save(mesh_path)
    print(f"Saved base mesh to {mesh_path}")

    # Create function spaces for output
    Vs = fem.functionspace(domain, ("Lagrange", 2))  # Scalar space for magnitude
    magnitude = fem.Function(Vs)


# ============================================================================
# TIME STEPPING LOOP
# ============================================================================
log.set_log_level(log.LogLevel.INFO)

# Set constant load (applied immediately at t=0)
# Load: 5000 N total in x-direction, distributed on right boundary (x = L)
# Right face area = W × H = 2.0 × 1.0 = 2.0 m²
# Traction (force per unit area) = 5000 N / 2.0 m² = 2500 N/m²
T.value[0] = 2500.0  # Traction in +x direction (gives total force of 5000 N)

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
    problem.solve()
    converged = problem.solver.getConvergedReason()
    num_its = problem.solver.getIterationNumber()
    assert converged > 0, f"Newton solver did not converge (reason {converged})."
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
    
    # Save displacement and magnitude fields to VTK file for this time step
    if SAVE_VTK:
        # Update displacement field on mesh
        displacement_3d = u.x.array.reshape(-1, 3)
        pv_mesh["displacement"] = displacement_3d
        pv_mesh.set_active_vectors("displacement")
        
        # Compute and save magnitude field
        us = fem.Expression(
            ufl.sqrt(sum([u[i] ** 2 for i in range(len(u))])),
            Vs.element.interpolation_points,
        )
        magnitude.interpolate(us)
        pv_mesh["magnitude"] = magnitude.x.array
        
        # Save VTK file for this time step
        vtk_path = os.path.join(vtk_dir, f"beam_dynamic_step_{n:04d}.vtk")
        pv_mesh.save(vtk_path)

# Save final summary
if SAVE_VTK:
    print(f"Saved time series data to {vtk_dir}")
    print(f"  Base mesh: beam_dynamic_mesh.vtk")
    print(f"  Time step files: beam_dynamic_step_XXXX.vtk (total: {n_steps} files)")
