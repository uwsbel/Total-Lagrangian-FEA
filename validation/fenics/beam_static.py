"""
Nonlinear 3D beam (compressible Neo-Hookean) solved with DOLFINx.
Creates an animated GIF of the deformation using PyVista.
"""
from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
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
# Body force and boundary traction (traction updated in load loop)
B = fem.Constant(domain, default_scalar_type((0, 0, 0)))
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

v = ufl.TestFunction(V)
u = fem.Function(V)  # displacement field (unknown)


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
# Stored strain energy density
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
# First Piola–Kirchhoff stress
P = ufl.diff(psi, F)


# ============================================================================
# VARIATIONAL FORM
# ============================================================================
# Variational form: find u such that F(u) = 0 (traction on tag=2)
metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)
F = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, B) * dx - ufl.inner(v, T) * ds(2)


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
    petsc_options_prefix="beam_static",
)


# ============================================================================
# VISUALIZATION SETUP
# ============================================================================
# PyVista setup for animated GIF
plotter = pyvista.Plotter()

# Ensure output directory exists, and use a path relative to script location
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
gif_path = os.path.join(output_dir, "deformation_static.gif")
plotter.open_gif(gif_path, fps=3)

topology, cells, geometry = plot.vtk_mesh(u.function_space)
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)

values = np.zeros((geometry.shape[0], 3))
values[:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
function_grid["u"] = values
function_grid.set_active_vectors("u")

# Warp mesh by deformation
warped = function_grid.warp_by_vector("u", factor=1)

# Add mesh to plotter and visualize
plotter.add_mesh(warped, show_edges=True, lighting=False, clim=[0, 10])

# Compute magnitude of displacement to visualize in GIF
Vs = fem.functionspace(domain, ("Lagrange", 2))
magnitude = fem.Function(Vs)
us = fem.Expression(ufl.sqrt(sum([u[i]**2 for i in range(len(u))])), Vs.element.interpolation_points)
magnitude.interpolate(us)
warped["mag"] = magnitude.x.array


# ============================================================================
# LOAD STEPPING LOOP
# ============================================================================
log.set_log_level(log.LogLevel.INFO)
# Load stepping: increase traction in -y direction and solve; write GIF frames
tval0 = -2.0
for n in range(1, 12):
    T.value[1] = n * tval0
    problem.solve()
    converged = problem.solver.getConvergedReason()
    num_its = problem.solver.getIterationNumber()
    assert converged > 0, f"Newton solver did not converge (reason {converged})."
    u.x.scatter_forward()
    print(f"Time step {n}, Number of iterations {num_its}, Load {T.value}")
    function_grid["u"][:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
    magnitude.interpolate(us)
    warped.set_active_scalars("mag")
    warped_n = function_grid.warp_by_vector("u", factor=1)
    warped.points[:, :] = warped_n.points
    warped.point_data["mag"][:] = magnitude.x.array
    plotter.update_scalar_bar_range([0, 1])
    plotter.write_frame()
plotter.close()
print(f"Saved GIF to {gif_path}")

