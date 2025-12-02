"""
Nonlinear 3D beam dynamic analysis using Backward Euler time integration.
Matches C++ implementation: uses nodal forces, Backward Euler, and same tolerances.
"""
import os
import numpy as np
import ufl

from dolfinx import fem, mesh, plot, log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, assemble_residual
from petsc4py import PETSc
from tetgen_mesh_loader import load_tetgen_mesh_from_files


# ============================================================================
# GEOMETRY AND MESH SETUP
# ============================================================================
# Resolution selection: 0 (RES_0), 2 (RES_2), or 4 (RES_4)
RES = 0
# VTK output option: Set to True to save VTK files, False to skip
SAVE_VTK = False

# Construct mesh file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir))
mesh_dir = os.path.join(project_root, "data", "meshes", "T10", "resolution")

node_file = os.path.join(mesh_dir, f"beam_3x2x1_res{RES}.1.node")
ele_file = os.path.join(mesh_dir, f"beam_3x2x1_res{RES}.1.ele")

# Load TetGen mesh
domain, x_tetgen = load_tetgen_mesh_from_files(node_file, ele_file)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))

# Print function space information
# print("\n" + "="*80)
# print("FUNCTION SPACE INFORMATION")
# print("="*80)
# print(f"Function space: {V}")
# print(f"Element family: Lagrange")
# print(f"Element degree: 2 (quadratic)")
# print(f"Value shape: {V.element.value_shape} (vector with {domain.geometry.dim} components)")
# print(f"Mesh dimension: {domain.geometry.dim}D")
# print("="*80 + "\n")

# # List all DOF nodes in function space (includes mid-edge nodes for quadratic)
# print("\n" + "="*80)
# print("FUNCTION SPACE DOF NODES (Quadratic - includes mid-edge nodes)")
# print("="*80)
# dof_coords = V.tabulate_dof_coordinates()
# print(f"DEBUG: tabulate_dof_coordinates() returned shape: {dof_coords.shape}")
# print(f"DEBUG: Total entries: {len(dof_coords)}")

# For a blocked vector function space, tabulate_dof_coordinates() returns 
# one entry per spatial DOF location (not repeated for each component)
# print(f"Total DOF nodes in function space: {len(dof_coords)}")
# print(f"\nDOF Node listing:")
# print(f"{'Index':<8} {'X':<12} {'Y':<12} {'Z':<12}")
# print("-"*80)
# for i, point in enumerate(dof_coords):
#     print(f"{i:<8} {point[0]:<12.6f} {point[1]:<12.6f} {point[2]:<12.6f}")
# print("="*80 + "\n")

# Beam dimensions (for boundary conditions)
L = 3.0   # Length (x)
W = 2.0   # Width (y)
H = 1.0   # Height (z)
tol = 1e-6

# Find DOF nodes at x = 0 and x = 3 (using function space DOFs)
# dof_nodes_x0 = []
# dof_nodes_x3 = []

# for i, point in enumerate(dof_coords):
#     if abs(point[0] - 0.0) < tol:
#         dof_nodes_x0.append((i, point))
#     elif abs(point[0] - L) < tol:
#         dof_nodes_x3.append((i, point))

# Print DOF nodes at x = 0
# print("\n" + "="*80)
# print("DOF NODES AT X = 0 (Fixed end)")
# print("="*80)
# print(f"Total DOF nodes at x=0: {len(dof_nodes_x0)}")
# print(f"\n{'DOF Index':<12} {'X':<12} {'Y':<12} {'Z':<12}")
# print("-"*80)
# for idx, point in sorted(dof_nodes_x0):
#     print(f"{idx:<12} {point[0]:<12.6f} {point[1]:<12.6f} {point[2]:<12.6f}")
# print("="*80 + "\n")

# Print DOF nodes at x = 3
# print("\n" + "="*80)
# print("DOF NODES AT X = 3 (Free end)")
# print("="*80)
# print(f"Total DOF nodes at x=3: {len(dof_nodes_x3)}")
# print(f"\n{'DOF Index':<12} {'X':<12} {'Y':<12} {'Z':<12}")
# print("-"*80)
# for idx, point in sorted(dof_nodes_x3):
#     print(f"{idx:<12} {point[0]:<12.6f} {point[1]:<12.6f} {point[2]:<12.6f}")
# print("="*80 + "\n")


# Print total nodes and elements (all ranks print)
topology_vertices = domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
total_elements = domain.topology.index_map(domain.topology.dim).size_local + domain.topology.index_map(domain.topology.dim).num_ghosts
# Function space DOFs (quadratic elements - includes mid-edge nodes)
dofmap = V.dofmap
total_dofs = dofmap.index_map.size_local + dofmap.index_map.num_ghosts
block_size = dofmap.index_map_bs
total_vector_dofs = total_dofs * block_size
print(f"Loaded TetGen mesh: beam_3x2x1 (RES_{RES})")
print(f"Topology vertices: {topology_vertices}")
print(f"Function space DOFs (quadratic): {total_dofs}")
print(f"Total DOFs (including all vector components): {total_vector_dofs}")
print(f"Total elements: {total_elements}")

# ============================================================================
# BOUNDARY CONDITIONS - Fix x=0 face
# ============================================================================
print("\nBOUNDARY CONDITIONS SETUP")

# Define a function to identify nodes at x = 0
def fixed_boundary(x):
    return np.isclose(x[0], 0.0, atol=1e-6)

# Locate DOFs on the fixed boundary
boundary_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)

# Verify: manually find all function space DOF nodes at x=0
dof_coords = V.tabulate_dof_coordinates()
manual_x0_dofs = []
tol = 1e-6
for i, coord in enumerate(dof_coords):
    if abs(coord[0] - 0.0) < tol:
        manual_x0_dofs.append(i)

# Create zero displacement boundary condition
u_zero = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
bc_fixed = fem.dirichletbc(u_zero, boundary_dofs, V)

print(f"Fixed boundary at x=0:")
print(f"  Number of DOFs found by locate_dofs_geometrical: {len(boundary_dofs)}")
print(f"  Number of DOFs manually found at x=0: {len(manual_x0_dofs)}")
print(f"  DOF indices match: {set(boundary_dofs) == set(manual_x0_dofs)}")
print(f"  Boundary condition: u = {u_zero}")
print(f"  Constrained scalar DOFs: {len(boundary_dofs) * block_size}")
print(f"  Free scalar DOFs: {total_vector_dofs - len(boundary_dofs) * block_size}")

# List the constrained DOF nodes
# print(f"\n  DOF nodes constrained at x=0:")
# print(f"  {'DOF Index':<12} {'X':<12} {'Y':<12} {'Z':<12}")
# print("  " + "-"*76)
# for dof_idx in sorted(boundary_dofs):
#     coord = dof_coords[dof_idx]
#     print(f"  {dof_idx:<12} {coord[0]:<12.6f} {coord[1]:<12.6f} {coord[2]:<12.6f}")


# ============================================================================
# 1. PREPARE THE EXTERNAL FORCE VECTOR (DIRECT POINT LOADS)
# ============================================================================
print("\nAPPLIED LOADS SETUP")

# Identify DOFs on the face x = L
dof_coords = V.tabulate_dof_coordinates()
force_dofs = []

# Find indices of nodes at x=L
# Note: In a VectorFunctionSpace, tabulate_dof_coordinates returns the 
# coordinate of the "node" associated with the block.
for i, coord in enumerate(dof_coords):
    if abs(coord[0] - L) < tol:
        force_dofs.append(i)

# Calculate Force Per Node (Matching C++ Logic)
total_force = 5000.0
num_force_nodes = len(force_dofs)
force_per_node = total_force / num_force_nodes if num_force_nodes > 0 else 0.0

print(f"Applying Lumped Force: {force_per_node} N per node on {num_force_nodes} nodes.")

# Create a global PETSc vector for the external force
# We use the same index map as the function space
f_temp = fem.Function(V)
f_temp.x.array[:] = 0.0

# Apply the force directly to the vector indices
# We must map the local 'force_dofs' to the global PETSc vector indices
dofmap = V.dofmap
bs = dofmap.index_map_bs  # Block size (should be 3)

# Get local-to-global map if running in parallel (or serial)
# For simple serial script, we can access the array directly via the Function wrapper
# but treating it as a raw PETSc vector is safer for the solver.
for node_idx in force_dofs:
    # Set X-component (index 0 in the block)
    # The C++ code sets: h_f_ext(3 * node_idx + 0) = force_per_node
    f_temp.x.array[node_idx * bs + 0] = force_per_node

# Move data to the PETSc vector
f_ext_vector = f_temp.x.petsc_vec.copy()
f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

print(f"Load applied at x=3:")
print(f"  Total force: {total_force} N (x direction)")
print(f"  Number of nodes at x=3: {num_force_nodes}")
print(f"  Force per node: {force_per_node:.6f} N")
print(f"  Total force applied: {force_per_node * num_force_nodes:.1f} N")
print(f"  Distribution: Equal distribution across all nodes")


# ============================================================================
# TRACKED NODE - Node at (3, 2, 0) for monitoring during simulation
# ============================================================================
print("\nTRACKED NODE SETUP")

# Find the DOF node at position (3, 2, 0)
tracked_node_position = np.array([3.0, 2.0, 0.0])
tracked_node_dof = None
tracked_node_coord = None

for i, coord in enumerate(dof_coords):
    if (abs(coord[0] - tracked_node_position[0]) < tol and 
        abs(coord[1] - tracked_node_position[1]) < tol and 
        abs(coord[2] - tracked_node_position[2]) < tol):
        tracked_node_dof = i
        tracked_node_coord = coord
        break

if tracked_node_dof is not None:
    # In a vector function space with block size 3, the DOF indices are organized as:
    # node 0: [ux, uy, uz], node 1: [ux, uy, uz], ...
    # For a blocked space, the node index IS the DOF index
    dof_x = tracked_node_dof  # This is the node index in the blocked space
    
    print(f"Tracked node found at position ({tracked_node_position[0]}, {tracked_node_position[1]}, {tracked_node_position[2]}):")
    print(f"  DOF node index: {tracked_node_dof}")
    print(f"  Actual coordinates: ({tracked_node_coord[0]:.6f}, {tracked_node_coord[1]:.6f}, {tracked_node_coord[2]:.6f})")
    print(f"  X-component DOF index: {dof_x}")
else:
    print(f"WARNING: No DOF node found at position ({tracked_node_position[0]}, {tracked_node_position[1]}, {tracked_node_position[2]})")

# ============================================================================
# MATERIAL MODEL AND KINEMATICS
# ============================================================================
# Material properties (matching C++ exactly)
E = default_scalar_type(7.0e8)  # Young's modulus: 7×10⁸ Pa
nu = default_scalar_type(0.33)  # Poisson's ratio: 0.33
mu = fem.Constant(domain, E / (2 * (1 + nu)))  # Shear modulus
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))  # Lamé's first parameter
rho = fem.Constant(domain, 2700.0)  # Density: 2700 kg/m³

# Function space setup for dynamics
v = ufl.TestFunction(V)
u = fem.Function(V)  # displacement field (unknown)
u_old = fem.Function(V)  # Previous displacement
v_old = fem.Function(V)  # Previous velocity

# Body force (zero for this problem)
B = fem.Constant(domain, default_scalar_type((0, 0, 0)))

# Kinematics
d = len(u)  # Spatial dimension
I = ufl.Identity(d)  # Identity tensor
F = I + ufl.grad(u)  # Deformation gradient
C = F.T * F  # Right Cauchy-Green tensor
trFtF = ufl.tr(C)  # Trace of C = F^T * F
FFt = F * F.T  # F * F^T
FFtF = FFt * F  # F * F^T * F

# Material model (St. Venant-Kirchhoff):
# P = λ*(0.5*tr(F^T*F) - 1.5)*F + μ*(F*F^T*F - F)
lambda_factor = lmbda * (0.5 * trFtF - 1.5)
P = lambda_factor * F + mu * (FFtF - F)


# ============================================================================
# TIME INTEGRATION SETUP (Backward Euler method)
# ============================================================================
dt = 1e-3  # Time step (0.1 seconds)
n_steps = 50  # Number of time steps
t_final = n_steps * dt  # Total simulation time: 5.0 seconds

print("\nTIME INTEGRATION SETUP")
print(f"Method: Backward Euler")
print(f"Time step (dt): {dt} s")
print(f"Number of steps: {n_steps}")
print(f"Total simulation time: {t_final} s")


# ============================================================================
# VARIATIONAL FORM (Backward Euler)
# ============================================================================
# Quadrature degree reduced to 3 (matching C++ more closely)
metadata = {"quadrature_degree": 5}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Backward Euler time discretization:
# Current velocity: v = (u - u_old) / dt
v_current = (u - u_old) / dt

# Current acceleration: a = (v_current - v_old) / dt
a_current = (v_current - v_old) / dt

# Backward Euler variational form:
# REMOVED: - ufl.inner(v, f_nodal) * dx
# We now only calculate Mass (Inertia) and Stiffness (Internal Stress) here.
F_form = (rho * ufl.inner(a_current, v) * dx +
          ufl.inner(ufl.grad(v), P) * dx -
          ufl.inner(v, B) * dx)


# ============================================================================
# 3. CUSTOM SOLVER PROBLEM
# ============================================================================
class PointLoadProblem(NonlinearProblem):
    def __init__(self, F, u, f_ext_vector, bcs=None, **kwargs):
        super().__init__(F, u, bcs=bcs, **kwargs)
        self.f_ext_vector = f_ext_vector
        self._bcs = bcs if bcs is not None else []
        
        # Set up custom residual function callback for SNES
        # This callback will be called by SNES during Newton iterations
        def residual_callback(snes, x, b, ctx=None):
            # First, assemble the standard residual (Mass + Stiffness) into b
            # assemble_residual handles SNES context and boundary conditions properly
            assemble_residual(self.u, self.F, self.J, self._bcs, snes, x, b)
            
            # Then subtract the external force vector (b = F_int - F_ext)
            b.axpy(-1.0, self.f_ext_vector)
        
        # Set the custom residual function
        # self.solver is the SNES object
        self.solver.setFunction(residual_callback, self.b)

# Initialize the custom problem
problem = PointLoadProblem(
    F_form,
    u,
    f_ext_vector=f_ext_vector,
    bcs=[bc_fixed],
    petsc_options={
        "snes_type": "newtonls",
        "snes_atol": 1e-6,
        "snes_rtol": 1e-6,
        "snes_stol": 1e-6,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    petsc_options_prefix="beam_dynamic_be",
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

# Initialize state variables (beam starts from rest)
u_old.x.array[:] = 0.0
v_old.x.array[:] = 0.0

# History for tracked node
node_x_history = []

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
    
    # Track node x-position (absolute position = initial + displacement)
    if tracked_node_dof is not None:
        # Get displacement at tracked node (x-component)
        u_x_at_node = u.x.array[tracked_node_dof * block_size + 0]
        x_position = tracked_node_coord[0] + u_x_at_node
        node_x_history.append(x_position)
    
    # Update old values for next time step
    u_old.x.array[:] = u.x.array[:]
    v_old.x.array[:] = v_new[:]
    
    # Print progress
    if n % 10 == 0 or n < 5:
        max_disp = np.max(np.linalg.norm(u.x.array.reshape(-1, 3), axis=1))
        max_vel = np.max(np.linalg.norm(v_old.x.array.reshape(-1, 3), axis=1))
        if tracked_node_dof is not None:
            print(f"Step {n}: tracked node x = {node_x_history[-1]:.17f}")
        print(f"  Time {t:.4f}, Iterations {num_its}")
        print(f"  Max displacement: {max_disp:.6e}, Max velocity: {max_vel:.6e}")

print("\nDYNAMIC ANALYSIS COMPLETE")


# ============================================================================
# SAVE CSV OUTPUT (Matching C++ format)
# ============================================================================
if tracked_node_dof is not None:
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"node_x_history_fenics_res{RES}.csv")
    with open(csv_path, 'w') as f:
        f.write("step,x_position\n")
        for i, x_val in enumerate(node_x_history):
            f.write(f"{i},{x_val:.17f}\n")
    
    print(f"Wrote tracked node x-position history to {csv_path}")
    print(f"  Node position: ({tracked_node_coord[0]:.1f}, {tracked_node_coord[1]:.1f}, {tracked_node_coord[2]:.1f})")
    print(f"  Format: step,x_position")
    print(f"  Total steps: {len(node_x_history)}\n")
