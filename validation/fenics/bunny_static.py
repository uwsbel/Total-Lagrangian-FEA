"""
Nonlinear 3D bunny mesh (St. Venant-Kirchhoff) solved with DOLFINx.
Validates C++ AdamW solver results using FEniCS Newton solver.
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
VIS = False  

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
# MATERIAL PROPERTIES (Step 2)
# ============================================================================
# Match C++ code exactly: lines 12-14 of test_feat10_bunny_adamw.cc
E = default_scalar_type(5.0e7)      # Pa - Young's modulus
nu = default_scalar_type(0.28)      # - Poisson's ratio
rho0 = default_scalar_type(1200.0)  # kg/m^3 - Density (for reference)

# Compute Lamé parameters
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

# ============================================================================
# BOUNDARY CONDITIONS - FIXED NODES (Step 4)
# ============================================================================
# Replicate C++ constraints: fix all nodes with z < -4.0
def fixed_boundary(x):
    """Identify nodes with z < -4.0"""
    return x[2] < -4.0

fdim = domain.topology.dim - 1

# Locate DOFs on the fixed boundary
fixed_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)

# Apply homogeneous Dirichlet BC: u = (0, 0, 0)
u_bc = np.array((0, 0, 0), dtype=default_scalar_type)
bc_fixed = fem.dirichletbc(u_bc, fixed_dofs, V)
bcs = [bc_fixed]

print(f"Number of DOFs fixed (z < -4.0): {len(fixed_dofs)}")

# ============================================================================
# LOADS AND FUNCTION SPACE (Step 5)
# ============================================================================
# Apply point load of -2.0 N in z-direction for all nodes with z > 4.0

# Create external force vector
f_ext = fem.Function(V)
f_ext_array = f_ext.x.array

# Identify loaded nodes and apply forces
# Get coordinates of all DOF points
dof_coordinates = V.tabulate_dof_coordinates()
print(f"Total DOF coordinates shape: {dof_coordinates.shape}")

# Apply force to z-component of nodes with z > 4.0
force_value = -2.0  # N in z-direction
nodes_loaded = 0

for i in range(len(x_nodes)):
    if x_nodes[i, 2] > 4.0:
        # For each node, we need to find the corresponding DOF for z-component
        # In a P2 vector space, each physical node has 3 DOFs (x, y, z components)
        # The DOF layout depends on FEniCS internal ordering
        # We need to find which DOF index corresponds to this node's z-component
        
        # For now, let's use a simpler approach: iterate through all DOFs
        # and check their coordinates
        nodes_loaded += 1

# Alternative approach: directly set forces based on DOF coordinates
for dof_idx in range(0, len(f_ext_array), 3):
    # Get z-coordinate of this DOF group (every 3rd DOF starting from 2 is z-component)
    dof_coord_idx = dof_idx // 3
    if dof_coord_idx < len(dof_coordinates):
        z_coord = dof_coordinates[dof_coord_idx, 2]
        if z_coord > 4.0:
            f_ext_array[dof_idx + 2] = force_value  # z-component

f_ext.x.scatter_forward()

print(f"Applied force: {force_value} N in z-direction")
print(f"Number of nodes with z > 4.0: {np.sum(x_nodes[:, 2] > 4.0)}")
print(f"Total external force magnitude: {np.linalg.norm(f_ext_array):.4f} N")
    

# ============================================================================
# KINEMATICS AND MATERIAL MODEL (Step 3)
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
# Compute F * F^T * F (needed for St. Venant-Kirchhoff)
FFt = F * F.T
FFtF = FFt * F

# Material model (St. Venant-Kirchhoff) - matches C++ implementation
# P = λ*(0.5*tr(F^T*F) - 1.5)*F + μ*(F*F^T*F - F)
lambda_factor = lmbda * (0.5 * trFtF - 1.5)
P = lambda_factor * F + mu * (FFtF - F)


# ============================================================================
# VARIATIONAL FORM
# ============================================================================

# Set quadrature degree
metadata = {"quadrature_degree": 4}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Weak form: internal virtual work - external virtual work
F_form = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, f_ext) * dx

print("Using 4 point quadrature")
print("Problem Setup Complete")


# ============================================================================
# SOLVER SETUP AND SOLVE (Step 7)
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
    petsc_options_prefix="bunny_static",
)

print("Newton Solver Setup:")
print("  SNES type: newtonls")
print("  Absolute tolerance: 1e-8")
print("  Relative tolerance: 1e-8")
print("  Linear solver: LU (MUMPS)")

print("Solving Nonlinear System")

# Set log level to see solver progress
log.set_log_level(log.LogLevel.INFO)

# Solve the nonlinear problem
problem.solve()
converged = problem.solver.getConvergedReason()
num_its = problem.solver.getIterationNumber()

if converged > 0:
    print(f"\nSolution converged in {num_its} iterations")
    print(f"  Convergence reason: {converged}")
    u.x.scatter_forward()
    
    # Compute displacement statistics
    u_array = u.x.array
    u_magnitude = np.sqrt(u_array[0::3]**2 + u_array[1::3]**2 + u_array[2::3]**2)
        
    # Print displacement components statistics
    u_x = u_array[0::3]
    u_y = u_array[1::3]
    u_z = u_array[2::3]
else:
    print(f"\nSolution did NOT converge after {num_its} iterations")
    print(f"  Convergence reason: {converged}")
    
# ============================================================================
# VISUALIZATION SETUP (Step 8)
# ============================================================================
# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

# PyVista setup for visualization
from dolfinx import plot

topology, cells, geometry = plot.vtk_mesh(u.function_space)
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)

# Set displacement field
values = np.zeros((geometry.shape[0], 3))
values[:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
function_grid["u"] = values
function_grid.set_active_vectors("u")

# Compute displacement magnitude
Vs = fem.functionspace(domain, ("Lagrange", 2))
magnitude = fem.Function(Vs)
us = fem.Expression(ufl.sqrt(sum([u[i]**2 for i in range(len(u))])), Vs.element.interpolation_points)
magnitude.interpolate(us)
function_grid["mag"] = magnitude.x.array

# Create warped mesh for visualization
warped = function_grid.warp_by_vector("u", factor=1.0)
warped.set_active_scalars("mag")

# Create animated GIF with rotating view
plotter = pyvista.Plotter(off_screen=not VIS)
gif_path = os.path.join(output_dir, "deformation_bunny_static.gif")
plotter.open_gif(gif_path, fps=10)

plotter.add_mesh(warped, show_edges=True, lighting=False, scalars="mag",
                 cmap="viridis", scalar_bar_args={'title': 'Displacement (m)'})
plotter.add_text("Bunny Static Deformation (FEniCS)", position='upper_edge', font_size=12)

# Create rotating animation (360 degrees over 36 frames)
n_frames = 36
for i in range(n_frames):
    plotter.camera.azimuth = 10  # Rotate 10 degrees per frame
    plotter.write_frame()

plotter.close()
print(f"Saved animated GIF to: {gif_path}")


# ============================================================================
# FINAL OUTPUT AND COMPARISON DATA (Step 9)
# ============================================================================
# Get final deformed positions
x_final = x_nodes[:, 0] + u.x.array[0::3]
y_final = x_nodes[:, 1] + u.x.array[1::3]
z_final = x_nodes[:, 2] + u.x.array[2::3]

# Save final positions to file for comparison
positions_file = os.path.join(output_dir, "bunny_static_final_positions.txt")
with open(positions_file, 'w') as f:
    f.write("# Final node positions (x, y, z) after deformation\n")
    f.write(f"# Total nodes: {len(x_nodes)}\n")
    f.write(f"# Format: node_id x y z\n")
    for i in range(len(x_nodes)):
        f.write(f"{i} {x_final[i]:.17e} {y_final[i]:.17e} {z_final[i]:.17e}\n")

print(f"Saved final node positions to: {positions_file}")



