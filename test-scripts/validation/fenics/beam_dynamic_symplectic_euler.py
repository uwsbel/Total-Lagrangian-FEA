"""Nonlinear 3D beam dynamic analysis using Symplectic Euler (explicit) time integration."""
import os
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import assemble_vector, assemble_matrix
from petsc4py import PETSc
from tetgen_mesh_loader import load_tetgen_mesh_from_files

rank = MPI.COMM_WORLD.rank

if rank == 0:
    print(f"Running with {MPI.COMM_WORLD.size} MPI ranks")
# ============================================================================
# GEOMETRY AND MESH SETUP
# ============================================================================
# Resolution selection: 0, 2, 4, 8, 16
RES = 0
# Debug flag: Set to True to enable detailed debug output
DEBUG = False

# Construct mesh file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
mesh_dir = os.path.join(project_root, "data", "meshes", "T10", "resolution")

node_file = os.path.join(mesh_dir, f"beam_3x2x1_res{RES}.1.node")
ele_file = os.path.join(mesh_dir, f"beam_3x2x1_res{RES}.1.ele")

# Load TetGen mesh
domain, _ = load_tetgen_mesh_from_files(node_file, ele_file)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))

if DEBUG:
    # Print function space information
    print("\n" + "="*80)
    print("FUNCTION SPACE INFORMATION")
    print("="*80)
    print(f"Function space: {V}")
    print(f"Element family: Lagrange")
    print(f"Element degree: 2 (quadratic)")
    print(f"Value shape: {V.element.value_shape} (vector with {domain.geometry.dim} components)")
    print(f"Mesh dimension: {domain.geometry.dim}D")
    print("="*80 + "\n")
    
    # List all DOF nodes in function space (includes mid-edge nodes for quadratic)
    print("\n" + "="*80)
    print("FUNCTION SPACE DOF NODES (Quadratic - includes mid-edge nodes)")
    print("="*80)
    dof_coords_debug = V.tabulate_dof_coordinates()
    print(f"DEBUG: tabulate_dof_coordinates() returned shape: {dof_coords_debug.shape}")
    print(f"DEBUG: Total entries: {len(dof_coords_debug)}")
    
    # For a blocked vector function space, tabulate_dof_coordinates() returns 
    # one entry per spatial DOF location (not repeated for each component)
    print(f"Total DOF nodes in function space: {len(dof_coords_debug)}")
    print(f"\nDOF Node listing:")
    print(f"{'Index':<8} {'X':<12} {'Y':<12} {'Z':<12}")
    print("-"*80)
    for i, point in enumerate(dof_coords_debug):
        print(f"{i:<8} {point[0]:<12.6f} {point[1]:<12.6f} {point[2]:<12.6f}")
    print("="*80 + "\n")

# Beam dimensions (for boundary conditions)
# L = 3.0; W = 2.0; H = 1.0;
L = 3.0   # Length (x)
W = 2.0   # Width (y)
H = 1.0   # Height (z)


# Print total nodes and elements
topology_vertices = domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
total_elements = domain.topology.index_map(domain.topology.dim).size_local + domain.topology.index_map(domain.topology.dim).num_ghosts
# Function space DOFs (quadratic elements - includes mid-edge nodes)
dofmap = V.dofmap
total_dofs = dofmap.index_map.size_local + dofmap.index_map.num_ghosts
block_size = dofmap.index_map_bs
total_vector_dofs = total_dofs * block_size

if rank == 0:
    print(f"Loaded TetGen mesh: beam_3x2x1 (RES_{RES})")
    print(f"Topology vertices: {topology_vertices}")
    print(f"Function space DOFs (quadratic): {total_dofs}")
    print(f"Total DOFs (including all vector components): {total_vector_dofs}")
    print(f"Total elements: {total_elements}")

# ============================================================================
# BOUNDARY CONDITIONS - Fix x=0 face
# ============================================================================
if rank == 0:
    print("\nBOUNDARY CONDITIONS SETUP")

# Define a function to identify nodes at x = 0
def fixed_boundary(x):
    return np.isclose(x[0], 0.0, atol=1e-6)

# Locate DOFs on the fixed boundary
boundary_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)

# Verify: manually find all function space DOF nodes at x=0 (only owned, not ghosts)
dof_coords = V.tabulate_dof_coordinates()
dofmap = V.dofmap
num_owned_dofs = dofmap.index_map.size_local
manual_x0_dofs = []
for i, coord in enumerate(dof_coords):
    if i < num_owned_dofs and abs(coord[0] - 0.0) < 1e-6:
        manual_x0_dofs.append(i)

# Create zero displacement boundary condition
u_zero = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
bc_fixed = fem.dirichletbc(u_zero, boundary_dofs, V)

if rank == 0:
    print(f"Fixed boundary at x=0:")
    print(f"  Number of DOFs found by locate_dofs_geometrical: {len(boundary_dofs)}")
    print(f"  Number of DOFs manually found at x=0: {len(manual_x0_dofs)}")
    print(f"  DOF indices match: {set(boundary_dofs) == set(manual_x0_dofs)}")
    print(f"  Boundary condition: u = {u_zero}")
    print(f"  Constrained scalar DOFs: {len(boundary_dofs) * block_size}")
    print(f"  Free scalar DOFs: {total_vector_dofs - len(boundary_dofs) * block_size}")

if DEBUG:
    # List the constrained DOF nodes
    print(f"\n  DOF nodes constrained at x=0:")
    print(f"  {'DOF Index':<12} {'X':<12} {'Y':<12} {'Z':<12}")
    print("  " + "-"*76)
    for dof_idx in sorted(boundary_dofs):
        coord = dof_coords[dof_idx]
        print(f"  {dof_idx:<12} {coord[0]:<12.6f} {coord[1]:<12.6f} {coord[2]:<12.6f}")
    
    # Find and print DOF nodes at x = 0 and x = 3 (using function space DOFs, owned only)
    dof_nodes_x0 = []
    dof_nodes_x3 = []
    
    for i, point in enumerate(dof_coords):
        if i < num_owned_dofs:
            if abs(point[0] - 0.0) < 1e-6:
                dof_nodes_x0.append((i, point))
            elif abs(point[0] - L) < 1e-6:
                dof_nodes_x3.append((i, point))
    
    # Print DOF nodes at x = 0
    print("\n" + "="*80)
    print("DOF NODES AT X = 0 (Fixed end)")
    print("="*80)
    print(f"Total DOF nodes at x=0: {len(dof_nodes_x0)}")
    print(f"\n{'DOF Index':<12} {'X':<12} {'Y':<12} {'Z':<12}")
    print("-"*80)
    for idx, point in sorted(dof_nodes_x0):
        print(f"{idx:<12} {point[0]:<12.6f} {point[1]:<12.6f} {point[2]:<12.6f}")
    print("="*80 + "\n")
    
    # Print DOF nodes at x = 3
    print("\n" + "="*80)
    print("DOF NODES AT X = 3 (Free end)")
    print("="*80)
    print(f"Total DOF nodes at x=3: {len(dof_nodes_x3)}")
    print(f"\n{'DOF Index':<12} {'X':<12} {'Y':<12} {'Z':<12}")
    print("-"*80)
    for idx, point in sorted(dof_nodes_x3):
        print(f"{idx:<12} {point[0]:<12.6f} {point[1]:<12.6f} {point[2]:<12.6f}")
    print("="*80 + "\n")


# ============================================================================
# 1. PREPARE THE EXTERNAL FORCE VECTOR (DIRECT POINT LOADS)
# ============================================================================
if rank == 0:
    print("\nAPPLIED LOADS SETUP")

# Identify DOFs on the face x = L
dof_coords = V.tabulate_dof_coordinates()
force_dofs = []

# Get the number of owned DOFs (exclude ghosts)
num_owned_dofs = dofmap.index_map.size_local

# Find indices of nodes at x=L (only owned DOFs, not ghosts)
# Note: In a VectorFunctionSpace, tabulate_dof_coordinates returns the 
# coordinate of the "node" associated with the block.
for i, coord in enumerate(dof_coords):
    if i < num_owned_dofs and abs(coord[0] - L) < 1e-6:
        force_dofs.append(i)

# Compute GLOBAL total number of force nodes using MPI reduction
local_num_force_nodes = len(force_dofs)
global_num_force_nodes = domain.comm.allreduce(local_num_force_nodes, op=MPI.SUM)

# Calculate Force Per Node (Matching C++ Logic: +100000 N in +z for 1 second)
total_force = 100000.0
force_per_node = total_force / global_num_force_nodes if global_num_force_nodes > 0 else 0.0

if rank == 0:
    print(f"Applying Lumped Force (+z): {force_per_node} N per node on {global_num_force_nodes} nodes.")
    print(f"  Force will be applied for 1 second, then turned off.")

# Create a global PETSc vector for the external force
# We use the same index map as the function space
f_temp = fem.Function(V)
f_temp.x.array[:] = 0.0

# Apply the force directly to the vector indices
# We must map the local 'force_dofs' to the global PETSc vector indices
# Get local-to-global map if running in parallel (or serial)
# For simple serial script, we can access the array directly via the Function wrapper
# but treating it as a raw PETSc vector is safer for the solver.
for node_idx in force_dofs:
    # Set Z-component (index 2 in the block) to match C++ (+z direction)
    # The C++ code sets: h_f_ext(3 * node_idx + 2) = force_per_node
    f_temp.x.array[node_idx * block_size + 2] = force_per_node

# Move data to the PETSc vector
f_ext_vector = f_temp.x.petsc_vec.copy()
f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

if rank == 0:
    print(f"Load applied at x=3:")
    print(f"  Total force: {total_force} N (+z direction)")
    print(f"  Number of nodes at x=3: {global_num_force_nodes}")
    print(f"  Force per node: {force_per_node:.6f} N")
    print(f"  Total force applied: {force_per_node * global_num_force_nodes:.1f} N")
    print(f"  Distribution: Equal distribution across all nodes")


# ============================================================================
# TRACKED NODE - Match C++ TetGen node index for RES_0
# ============================================================================
if rank == 0:
    print("\nTRACKED NODE SETUP")

# Match current C++ run (which tracks the top corner at (3, 2, 1))
tracked_node_position = np.array([L, W, H])
tracked_node_dof = None
tracked_node_coord = None
tracked_node_rank = -1  # Which rank owns the tracked node

# Only search in owned DOFs, not ghosts
for i, coord in enumerate(dof_coords):
    if i < num_owned_dofs and (abs(coord[0] - tracked_node_position[0]) < 1e-6 and 
        abs(coord[1] - tracked_node_position[1]) < 1e-6 and 
        abs(coord[2] - tracked_node_position[2]) < 1e-6):
        tracked_node_dof = i
        tracked_node_coord = coord
        tracked_node_rank = rank
        break

# Use MPI to determine which rank has the tracked node
all_ranks_with_node = domain.comm.gather(tracked_node_rank, root=0)
if rank == 0:
    owner_rank = next((r for r in all_ranks_with_node if r >= 0), -1)
    if owner_rank >= 0:
        print(f"Tracked node found at position ({tracked_node_position[0]}, {tracked_node_position[1]}, {tracked_node_position[2]}):")
        print(f"  Owned by rank: {owner_rank}")
        if tracked_node_dof is not None:
            print(f"  DOF node index: {tracked_node_dof}")
            print(f"  Actual coordinates: ({tracked_node_coord[0]:.6f}, {tracked_node_coord[1]:.6f}, {tracked_node_coord[2]:.6f})")
    else:
        print(f"WARNING: No DOF node found at position ({tracked_node_position[0]}, {tracked_node_position[1]}, {tracked_node_position[2]})")

# ============================================================================
# MATERIAL MODEL AND KINEMATICS
# ============================================================================
# Select material model: "SVK" or "MOONEY_RIVLIN"
MATERIAL_MODEL = "SVK"

# Base material properties (matching C++ exactly)
E_val = 7.0e8     # Young's modulus: 7×10⁸ Pa
nu_val = 0.33     # Poisson's ratio: 0.33
rho_val = 2700.0  # Density: 2700 kg/m³
rho = fem.Constant(domain, rho_val)

E = default_scalar_type(E_val)
nu = default_scalar_type(nu_val)

# Function space setup for dynamics
u = fem.Function(V)      # Current displacement field
u_old = fem.Function(V)  # Previous displacement
v_old = fem.Function(V)  # Previous velocity
a = fem.Function(V)      # Acceleration (to be solved)

# Body force (zero for this problem)
B = fem.Constant(domain, default_scalar_type((0, 0, 0)))

# Kinematics
d = len(u)  # Spatial dimension
I = ufl.Identity(d)  # Identity tensor

# Wrap F in ufl.variable so we can differentiate psi w.r.t. F
F = ufl.variable(I + ufl.grad(u))  # Deformation gradient
C = F.T * F  # Right Cauchy-Green tensor

# Material model selection
if MATERIAL_MODEL == "SVK":
    # SVK parameters
    mu_svk = fem.Constant(domain, E / (2 * (1 + nu)))  # Shear modulus
    lmbda_svk = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))  # Lamé's first parameter

    trFtF = ufl.tr(C)  # Trace of C = F^T * F
    FFt = F * F.T  # F * F^T
    FFtF = FFt * F  # F * F^T * F

    # St. Venant-Kirchhoff:
    # P = λ*(0.5*tr(F^T*F) - 1.5)*F + μ*(F*F^T*F - F)
    lambda_factor = lmbda_svk * (0.5 * trFtF - 1.5)
    P = lambda_factor * F + mu_svk * (FFtF - F)
    if rank == 0:
        print("\nMATERIAL MODEL: St. Venant-Kirchhoff (SVK)")
else:
    # Compressible Mooney-Rivlin (isochoric invariants + volumetric penalty)
    # Match C++ logic: mu10 = 0.30*mu, mu01 = 0.20*mu, kappa = 1.5*K
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    K_val = E_val / (3.0 * (1.0 - 2.0 * nu_val))

    c1_val = 0.30 * mu_val
    c2_val = 0.20 * mu_val
    kappa_val = 1.5 * K_val

    C1 = fem.Constant(domain, default_scalar_type(c1_val))
    C2 = fem.Constant(domain, default_scalar_type(c2_val))
    kappa = fem.Constant(domain, default_scalar_type(kappa_val))

    if rank == 0:
        print("\nMATERIAL MODEL: Mooney-Rivlin")
        print(f"  C1 (mu10): {c1_val:.4e}")
        print(f"  C2 (mu01): {c2_val:.4e}")
        print(f"  Kappa:     {kappa_val:.4e}")

    # Invariants and isochoric invariants (built from variable F)
    I1 = ufl.tr(C)
    C_squared = C * C
    trC2 = ufl.tr(C_squared)
    I2 = 0.5 * (I1**2 - trC2)

    J = ufl.det(F)
    I1_bar = J**(-2.0 / 3.0) * I1
    I2_bar = J**(-4.0 / 3.0) * I2

    # Strain energy density
    psi = C1 * (I1_bar - 3.0) + C2 * (I2_bar - 3.0) + 0.5 * kappa * (J - 1.0) ** 2

    # First Piola-Kirchhoff stress via automatic differentiation
    P = ufl.diff(psi, F)



# ============================================================================
# TIME INTEGRATION SETUP (Symplectic Euler method)
# ============================================================================
dt = 1e-5  # Time step (matching C++ GPU explicit)
n_steps = 200000  # Number of time steps (2 seconds total to match C++ GPU solvers)
t_final = n_steps * dt 

if rank == 0:
    print("\nTIME INTEGRATION SETUP")
    print(f"Method: Symplectic Euler (Explicit)")
    print(f"Time step (dt): {dt} s")
    print(f"Number of steps: {n_steps}")
    print(f"Total simulation time: {t_final} s")
    print(f"Force will be applied for 1 second (steps 0-99999), then turned off")


# ============================================================================
# ASSEMBLE MASS MATRIX (One-time, at start)
# ============================================================================
if rank == 0:
    print("\nASSEMBLING MASS MATRIX")

# Quadrature degree reduced to 5 (matching C++ more closely)
metadata = {"quadrature_degree": 5}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Mass matrix form: M_ij = ∫ ρ φ_i · φ_j dx
u_trial = ufl.TrialFunction(V)
v_test = ufl.TestFunction(V)
M_form = fem.form(rho * ufl.inner(u_trial, v_test) * dx)

# Assemble mass matrix
M_matrix = assemble_matrix(M_form, bcs=[bc_fixed])
M_matrix.assemble()

if rank == 0:
    print(f"Mass matrix assembled: {M_matrix.size[0]} x {M_matrix.size[1]}")

# Set up linear solver for M * a = residual
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(M_matrix)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
ksp.setUp()

if rank == 0:
    print(f"Linear solver setup: Direct LU (MUMPS)")


# ============================================================================
# NOTE: ACCELERATION VS VELOCITY FORMULATION
# ============================================================================
# There are two equivalent ways to formulate explicit time integration:
#
# 1. ACCELERATION FORMULATION (current approach):
#    - Solve:  M · a = f_ext - f_int
#    - Update: v_new = v_old + dt · a
#    - Update: u_new = u_old + dt · v_new
#
# 2. VELOCITY FORMULATION (typical GPU approach):
#    - Rearranging: M · v_new = M · v_old + dt · (f_ext - f_int)
#    - For LUMPED mass, this becomes a fused update:
#        v_new = v_old + dt · M_inv · (f_ext - f_int)
#      No separate acceleration variable needed.
#
# ANALYSIS:
# - For LUMPED mass: Both are equivalent. Velocity formulation eliminates
#   the intermediate 'a' array (GPU codes typically use this fused form).
# - For CONSISTENT mass: Acceleration formulation is preferred because
#   velocity formulation requires an extra matrix-vector product (M · v_old).
#
# This script uses the ACCELERATION formulation which is optimal for
# consistent mass and equivalent for lumped mass.
# ============================================================================


# ============================================================================
# TIME STEPPING LOOP (Symplectic Euler - Explicit)
# ============================================================================
if rank == 0:
    print("\nSTARTING DYNAMIC ANALYSIS")

# Initialize state variables (beam starts from rest)
u_old.x.array[:] = 0.0
u_old.x.scatter_forward()

v_old.x.array[:] = 0.0
v_old.x.scatter_forward()

# History for tracked node (x, y, z positions)
node_xyz_history = []

# Create work vectors for explicit time stepping
residual = f_ext_vector.copy()
f_int = f_ext_vector.copy()

# Internal force form: f_int = ∫ ∇φ : P(u) dx
f_int_form = fem.form(ufl.inner(ufl.grad(v_test), P) * dx)

# Calculate step at which to turn off force (1 second)
force_off_step = int(1.0 / dt)
if rank == 0:
    print(f"Force will be turned off at step {force_off_step} (t = 1.0s)")

# Progress tracking
progress_interval = 1000  # Print progress every N steps

# Time stepping loop
for n in range(n_steps):
    t = n * dt
    
    # Turn off force after 1 second
    if n == force_off_step:
        f_ext_vector.zeroEntries()
        f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        if rank == 0:
            print(f"Force turned off at step {n} (t = {t:.4f}s)")
    
    # Compute internal force based on current displacement u_old
    # Note: u is used in P via F = I + grad(u), so we need to update u first
    u.x.array[:] = u_old.x.array[:]
    u.x.scatter_forward()
    
    # Assemble internal force vector
    with f_int.localForm() as f_int_local:
        f_int_local.set(0.0)
    assemble_vector(f_int, f_int_form)
    f_int.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    # Apply boundary conditions to internal force
    fem.petsc.set_bc(f_int, [bc_fixed])
    
    # Compute residual: M * a = f_ext - f_int
    residual.zeroEntries()
    residual.axpy(1.0, f_ext_vector)  # residual = f_ext
    residual.axpy(-1.0, f_int)        # residual = f_ext - f_int
    
    # Solve for acceleration: M * a = residual
    ksp.solve(residual, a.x.petsc_vec)
    a.x.scatter_forward()
    
    # Apply boundary conditions to acceleration
    fem.petsc.set_bc(a.x.petsc_vec, [bc_fixed])
    
    # Explicit updates (Symplectic Euler)
    # v_new = v_old + dt * a
    v_new = v_old.x.array[:] + dt * a.x.array[:]
    
    # u_new = u_old + dt * v_new
    u_new = u_old.x.array[:] + dt * v_new[:]
    
    # Track node x, y, z positions (absolute position = initial + displacement)
    local_position = None
    if tracked_node_dof is not None:
        # Get displacement at tracked node (all components)
        u_x_at_node = u_new[tracked_node_dof * block_size + 0]
        u_y_at_node = u_new[tracked_node_dof * block_size + 1]
        u_z_at_node = u_new[tracked_node_dof * block_size + 2]
        x_position = tracked_node_coord[0] + u_x_at_node
        y_position = tracked_node_coord[1] + u_y_at_node
        z_position = tracked_node_coord[2] + u_z_at_node
        local_position = [float(x_position), float(y_position), float(z_position)]
    
    # Gather tracked node data from all ranks to rank 0
    all_positions = domain.comm.gather(local_position, root=0)
    if rank == 0:
        # Find the non-None position (from the rank that owns the node)
        node_position = next((pos for pos in all_positions if pos is not None), None)
        if node_position is not None:
            node_xyz_history.append(node_position)
    
    # Update old values for next time step
    u_old.x.array[:] = u_new[:]
    u_old.x.scatter_forward()
    
    v_old.x.array[:] = v_new[:]
    v_old.x.scatter_forward()
    
    # Print progress
    if rank == 0 and (n % progress_interval == 0 or n < 5 or n == n_steps - 1):
        progress_pct = 100.0 * (n + 1) / n_steps
        max_disp = np.max(np.linalg.norm(u_old.x.array.reshape(-1, 3), axis=1))
        max_vel = np.max(np.linalg.norm(v_old.x.array.reshape(-1, 3), axis=1))
        
        print(f"[{progress_pct:5.1f}%] Step {n+1}/{n_steps} | t={t:.6f}s")
        
        if len(node_xyz_history) > 0:
            x_pos, y_pos, z_pos = node_xyz_history[-1]
            print(f"  Tracked node: ({x_pos:.10f}, {y_pos:.10f}, {z_pos:.10f})")
        
        print(f"  Max displacement: {max_disp:.6e} m | Max velocity: {max_vel:.6e} m/s")
        
        # Print force status
        if n < force_off_step:
            print(f"  Force: ON ({force_per_node:.1f} N/node)")
        else:
            print(f"  Force: OFF")

if rank == 0:
    print("\nDYNAMIC ANALYSIS COMPLETE")


# ============================================================================
# SAVE CSV OUTPUT (Matching C++ format)
# ============================================================================
# Only rank 0 writes the CSV file (it has gathered all the data)
if rank == 0 and len(node_xyz_history) > 0:
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Material suffix for file naming
    mat_suffix = "svk" if MATERIAL_MODEL == "SVK" else "mr"

    csv_path = os.path.join(
        output_dir, f"node_xyz_history_fenics_res{RES}_{mat_suffix}_fe.csv"
    )
    
    with open(csv_path, 'w') as f:
        f.write("step,x_position,y_position,z_position\n")
        for i, (x_val, y_val, z_val) in enumerate(node_xyz_history):
            f.write(f"{i},{x_val:.17f},{y_val:.17f},{z_val:.17f}\n")
    
    print(f"Wrote tracked node x,y,z position history to {csv_path}")
    print(f"  Node position: ({tracked_node_position[0]:.1f}, {tracked_node_position[1]:.1f}, {tracked_node_position[2]:.1f})")
    print(f"  Total steps: {len(node_xyz_history)}\n")
