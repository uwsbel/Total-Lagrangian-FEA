"""Nonlinear 3D beam dynamics with Symplectic Euler (unified consistent/lumped mass)."""
import argparse
import os
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import assemble_vector, assemble_matrix
from petsc4py import PETSc
from tetgen_mesh_loader import load_tetgen_mesh_from_files


def _parse_cli():
    parser = argparse.ArgumentParser(
        description="Nonlinear 3D beam dynamics with Symplectic Euler (unified mass matrix options)."
    )
    parser.add_argument(
        "--mat",
        type=str,
        default="svk",
        help="svk | mr (default: svk)",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=0,
        help="0 | 2 | 4 | 8 | 16 | 32 (default: 0)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1e-5,
        help="Time step (default: 1e-5)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Number of time steps (default: 5000)",
    )
    parser.add_argument(
        "--mass",
        type=str,
        default="lumped",
        choices=["consistent", "lumped"],
        help="Mass matrix type: consistent | lumped (default: lumped)",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Write CSV of tracked node history (optional PATH, default name if omitted)",
    )
    return parser.parse_args()


_args = _parse_cli()

rank = MPI.COMM_WORLD.rank

if rank == 0:
    print(f"Running with {MPI.COMM_WORLD.size} MPI ranks")

RES = _args.res
MASS_TYPE = _args.mass.lower()
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

# Beam dimensions (for boundary conditions)
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

# Get the number of owned DOFs (exclude ghosts)
num_owned_dofs = dofmap.index_map.size_local

# Create zero displacement boundary condition
u_zero = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
bc_fixed = fem.dirichletbc(u_zero, boundary_dofs, V)

if rank == 0:
    print(f"Fixed boundary at x=0:")
    print(f"  Number of DOFs found: {len(boundary_dofs)}")
    print(f"  Constrained scalar DOFs: {len(boundary_dofs) * block_size}")
    print(f"  Free scalar DOFs: {total_vector_dofs - len(boundary_dofs) * block_size}")


# ============================================================================
# 1. PREPARE THE EXTERNAL FORCE VECTOR (DIRECT POINT LOADS)
# ============================================================================
if rank == 0:
    print("\nAPPLIED LOADS SETUP")

# Identify DOFs on the face x = L
dof_coords = V.tabulate_dof_coordinates()
force_dofs = []

# Find indices of nodes at x=L (only owned DOFs, not ghosts)
for i, coord in enumerate(dof_coords):
    if i < num_owned_dofs and abs(coord[0] - L) < 1e-6:
        force_dofs.append(i)

# Compute GLOBAL total number of force nodes using MPI reduction
local_num_force_nodes = len(force_dofs)
global_num_force_nodes = domain.comm.allreduce(local_num_force_nodes, op=MPI.SUM)

# Calculate Force Per Node (Matching C++ explicit: +10000 N in +z)
total_force = 10000.0
force_per_node = total_force / global_num_force_nodes if global_num_force_nodes > 0 else 0.0

if rank == 0:
    print(f"Applying Lumped Force (+z): {force_per_node} N per node on {global_num_force_nodes} nodes.")
    print(f"  Force will be applied for 1 second, then turned off.")

# Create a global PETSc vector for the external force
f_temp = fem.Function(V)
f_temp.x.array[:] = 0.0

# Apply the force directly to the vector indices
for node_idx in force_dofs:
    # Set Z-component (index 2 in the block) to match C++ (+z direction)
    f_temp.x.array[node_idx * block_size + 2] = force_per_node

# Move data to the PETSc vector
f_ext_vector = f_temp.x.petsc_vec.copy()
f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

if rank == 0:
    print(f"Load applied at x=3:")
    print(f"  Total force: {total_force} N (+z direction)")
    print(f"  Number of nodes at x=3: {global_num_force_nodes}")
    print(f"  Force per node: {force_per_node:.6f} N")


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
        print(f"Tracked node found at position ({tracked_node_position[0]}, {tracked_node_position[1]}, {tracked_node_position[2]})")
        if tracked_node_dof is not None:
            print(f"  DOF node index: {tracked_node_dof}")
    else:
        print(f"WARNING: No DOF node found at tracked position")

# ============================================================================
# MATERIAL MODEL AND KINEMATICS
# ============================================================================
# Select material model from CLI: "SVK" or "MOONEY_RIVLIN"
MATERIAL_MODEL = "SVK" if _args.mat.lower() == "svk" else "MOONEY_RIVLIN"

# Material properties - choose based on material model
if MATERIAL_MODEL == "MOONEY_RIVLIN":
    # Mooney-Rivlin coefficients
    mu10_val = 80000.0    # First MR coefficient
    mu01_val = 20000.0    # Second MR coefficient
    kappa_val = 1.0e6     # Bulk modulus (volumetric penalty)
    rho_val = 1100.0      # Rubber density kg/m³
    # Approximate E and nu for reference (not used in MR formulation)
    E_val = 6.0 * (mu10_val + mu01_val)  # ~600000 Pa
    nu_val = 0.45
else:
    # SVK material properties (matching GPU test_feat10_explicit.cc)
    E_val = 7.0e8         # Young's modulus: 7×10⁸ Pa
    nu_val = 0.33         # Poisson's ratio: 0.33
    rho_val = 2700.0      # Density: 2700 kg/m³

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
    # Use direct coefficients defined at top (matching GPU code)
    C1 = fem.Constant(domain, default_scalar_type(mu10_val))
    C2 = fem.Constant(domain, default_scalar_type(mu01_val))
    kappa = fem.Constant(domain, default_scalar_type(kappa_val))

    if rank == 0:
        print("\nMATERIAL MODEL: Mooney-Rivlin")
        print(f"  C1 (mu10): {mu10_val:.4e}")
        print(f"  C2 (mu01): {mu01_val:.4e}")
        print(f"  Kappa:     {kappa_val:.4e}")
        print(f"  Density:   {rho_val:.1f} kg/m³")

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
dt = _args.dt  # Time step (matching C++ GPU explicit default)
n_steps = _args.steps  # Number of time steps (default 5000, matches C++)
t_final = n_steps * dt

if rank == 0:
    print("\nTIME INTEGRATION SETUP")
    print(f"Method: Symplectic Euler (Explicit) with {MASS_TYPE.upper()} MASS")
    print(f"Time step (dt): {dt} s")
    print(f"Number of steps: {n_steps}")
    print(f"Total simulation time: {t_final} s")
    print(f"Force will be applied for the first half of the steps, then turned off")


# ============================================================================
# ASSEMBLE MASS MATRIX (Consistent or Lumped based on --mass flag)
# ============================================================================
if rank == 0:
    print(f"\nASSEMBLING {MASS_TYPE.upper()} MASS MATRIX")

# Quadrature degree reduced to 5 (matching C++ more closely)
metadata = {"quadrature_degree": 5}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Mass matrix form: M_ij = ∫ ρ φ_i · φ_j dx
u_trial = ufl.TrialFunction(V)
v_test = ufl.TestFunction(V)
M_form = fem.form(rho * ufl.inner(u_trial, v_test) * dx)

# Branch based on mass type
ksp = None
M_lumped_inv_array = None

if MASS_TYPE == "consistent":
    # ========================================================================
    # CONSISTENT MASS: Assemble full matrix and set up direct solver
    # ========================================================================
    M_matrix = assemble_matrix(M_form, bcs=[bc_fixed])
    M_matrix.assemble()

    if rank == 0:
        print(f"Consistent mass matrix assembled: {M_matrix.size[0]} x {M_matrix.size[1]}")

    # Set up linear solver for M * a = residual (one solve per time step)
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(M_matrix)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    ksp.setUp()

    if rank == 0:
        print(f"Linear solver setup: Direct LU (MUMPS) for M*a = f_ext - f_int")

else:  # "lumped"
    # ========================================================================
    # HRZ LUMPED MASS: Assemble, lump, and compute inverse
    # ========================================================================
    # Assemble consistent mass matrix (without BC modification for lumping)
    M_matrix_no_bc = assemble_matrix(M_form)
    M_matrix_no_bc.assemble()

    if rank == 0:
        print(f"Consistent mass matrix assembled: {M_matrix_no_bc.size[0]} x {M_matrix_no_bc.size[1]}")

    # ========================================================================
    # HRZ LUMPING: Scale diagonal to preserve total mass
    # ========================================================================
    # HRZ (Hinton-Rock-Zienkiewicz) lumping:
    # 1. Extract diagonal of consistent mass matrix
    # 2. Compute row sums (total mass per row)
    # 3. Scale diagonal: M_lumped[i] = M_diag[i] * (total_mass / sum_diag)
    #
    # For quadratic elements, this avoids negative masses that row-sum gives.

    # Step 1: Get diagonal of mass matrix
    M_diag = M_matrix_no_bc.createVecLeft()
    M_matrix_no_bc.getDiagonal(M_diag)
    M_diag_array = M_diag.getArray().copy()

    # Step 2: Get row sums
    ones = M_matrix_no_bc.createVecRight()
    ones.set(1.0)
    M_rowsum = M_matrix_no_bc.createVecLeft()
    M_matrix_no_bc.mult(ones, M_rowsum)
    M_rowsum_array = M_rowsum.getArray().copy()
    ones.destroy()

    # Step 3: Compute total mass
    total_mass_consistent = M_rowsum.sum()

    # Step 4: Compute sum of diagonals
    diag_sum = M_diag.sum()

    # Scale factor to preserve total mass
    if abs(diag_sum) > 1e-30:
        scale_factor = total_mass_consistent / diag_sum
    else:
        scale_factor = 1.0

    # HRZ lumped mass = scaled diagonal (all positive since M_ii > 0 for mass matrix)
    M_lumped_array = M_diag_array * scale_factor

    # Compute inverse lumped mass
    M_lumped_inv_array = np.zeros_like(M_lumped_array)
    for i in range(len(M_lumped_array)):
        if M_lumped_array[i] > 1e-30:
            M_lumped_inv_array[i] = 1.0 / M_lumped_array[i]
        else:
            M_lumped_inv_array[i] = 0.0

    # Zero out inverse mass for boundary DOFs (they should have zero acceleration)
    for dof in boundary_dofs:
        for c in range(block_size):
            idx = dof * block_size + c
            if idx < len(M_lumped_inv_array):
                M_lumped_inv_array[idx] = 0.0

    # Clean up PETSc vectors
    M_diag.destroy()
    M_rowsum.destroy()
    M_matrix_no_bc.destroy()

    if rank == 0:
        print(f"HRZ lumped mass vector created")
        print(f"  Total mass (consistent): {total_mass_consistent:.6f} kg")
        print(f"  Scale factor: {scale_factor:.6f}")
        positive_masses = M_lumped_array[M_lumped_array > 0]
        if len(positive_masses) > 0:
            print(f"  Min lumped mass: {positive_masses.min():.6e}")
            print(f"  Max lumped mass: {M_lumped_array.max():.6e}")
        print(f"  Any negative masses: {np.any(M_lumped_array < 0)}")


# ============================================================================
# TIME STEPPING LOOP (Symplectic Euler - Explicit)
# ============================================================================
if rank == 0:
    print(f"\nSTARTING DYNAMIC ANALYSIS ({MASS_TYPE.upper()} MASS)")

# Initialize state variables (beam starts from rest)
u_old.x.array[:] = 0.0
u_old.x.scatter_forward()

v_old.x.array[:] = 0.0
v_old.x.scatter_forward()

# Create work vectors for explicit time stepping
residual = f_ext_vector.copy()
f_int = f_ext_vector.copy()

# Internal force form: f_int = ∫ ∇φ : P(u) dx
f_int_form = fem.form(ufl.inner(ufl.grad(v_test), P) * dx)

# Calculate step at which to turn off force (half the simulation, like C++)
force_off_step = n_steps // 2
if rank == 0:
    print(f"Force will be turned off at step {force_off_step} (t = {force_off_step * dt:.4f}s)")

# ============================================================================
# CSV OUTPUT SETUP (Streaming to file during time loop)
# ============================================================================
write_csv = _args.csv is not None
csv_file = None

if write_csv and rank == 0:
    # Determine output path
    if _args.csv == "":
        output_dir = os.path.join(script_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        mat_suffix = "svk" if MATERIAL_MODEL == "SVK" else "mr"
        csv_path = os.path.join(
            output_dir,
            f"node_xyz_history_fenics_res{RES}_{mat_suffix}_{MASS_TYPE}.csv",
        )
    else:
        csv_path = _args.csv

    # Open CSV file and write header
    csv_file = open(csv_path, "w")
    csv_file.write("step,x_position,y_position,z_position,internal_force_l2\n")
    print(f"Writing CSV to {csv_path}")

# Time stepping loop
for n in range(n_steps):
    t = n * dt

    # Turn off force after first half of steps
    if n == force_off_step:
        f_ext_vector.zeroEntries()
        f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        if rank == 0:
            print(f"Force turned off at step {n} (t = {t:.4f}s)")

    # Compute internal force based on current displacement u_old
    u.x.array[:] = u_old.x.array[:]
    u.x.scatter_forward()

    # Assemble internal force vector
    with f_int.localForm() as f_int_local:
        f_int_local.set(0.0)
    assemble_vector(f_int, f_int_form)
    f_int.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Apply boundary conditions to internal force
    fem.petsc.set_bc(f_int, [bc_fixed])

    # Compute L2 norm of internal force for CSV output
    internal_force_l2 = 0.0
    if write_csv:
        # Get local owned portion of internal force vector
        local_owned_size = dofmap.index_map.size_local * block_size
        with f_int.localForm() as f_int_local:
            local_f_int_array = f_int_local.array[:local_owned_size]
            # Compute local squared sum over owned DOFs
            local_sq = np.sum(local_f_int_array ** 2)

        # Reduce to get global squared sum
        global_sq = domain.comm.allreduce(local_sq, op=MPI.SUM)
        internal_force_l2 = np.sqrt(global_sq)

    # Compute residual: f_ext - f_int
    residual.zeroEntries()
    residual.axpy(1.0, f_ext_vector)  # residual = f_ext
    residual.axpy(-1.0, f_int)        # residual = f_ext - f_int

    # ========================================================================
    # SOLVE FOR ACCELERATION (branched by mass type)
    # ========================================================================
    if MASS_TYPE == "consistent":
        # Consistent mass: Solve M * a = residual using direct solver
        ksp.solve(residual, a.x.petsc_vec)
        a.x.scatter_forward()
    else:  # lumped
        # Lumped mass: a = M^{-1} * residual (element-wise multiplication)
        residual_array = residual.getArray()
        a.x.array[:] = residual_array * M_lumped_inv_array
        a.x.scatter_forward()

    # ========================================================================
    # SYMPLECTIC EULER UPDATES (same for both mass types)
    # ========================================================================
    # Step 1: v_new = v_old + dt * a
    v_new = v_old.x.array[:] + dt * a.x.array[:]

    # Step 2: Apply BC - zero velocity at fixed nodes
    for dof in boundary_dofs:
        for c in range(block_size):
            idx = dof * block_size + c
            if idx < len(v_new):
                v_new[idx] = 0.0

    # Step 3: u_new = u_old + dt * v_new
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

    # Process gathered data on rank 0
    node_position = None
    if rank == 0:
        # Find the non-None position (from the rank that owns the node)
        node_position = next((pos for pos in all_positions if pos is not None), None)

        # Write CSV row (streaming during loop)
        if write_csv and node_position is not None:
            x_pos, y_pos, z_pos = node_position
            csv_file.write(f"{n},{x_pos:.17f},{y_pos:.17f},{z_pos:.17f},{internal_force_l2:.17e}\n")

        # Print progress (every 10000 steps to avoid flooding output)
        if (n % 10000 == 0 or n < 5) and node_position is not None:
            x_pos, y_pos, z_pos = node_position
            max_disp = np.max(np.linalg.norm(u_old.x.array.reshape(-1, 3), axis=1))
            max_vel = np.max(np.linalg.norm(v_old.x.array.reshape(-1, 3), axis=1))
            print(f"Step {n}/{n_steps}: t={t:.4f}s, tracked node position = ({x_pos:.17f}, {y_pos:.17f}, {z_pos:.17f})")
            print(f"  Max displacement: {max_disp:.6e}, Max velocity: {max_vel:.6e}")

    # Update old values for next time step
    u_old.x.array[:] = u_new[:]
    u_old.x.scatter_forward()

    v_old.x.array[:] = v_new[:]
    v_old.x.scatter_forward()

# Close CSV file if opened
if rank == 0 and write_csv and csv_file is not None:
    csv_file.close()
    print(f"\nCSV output complete. Wrote {n_steps} steps to CSV file.")
    print(
        f"  Node position: ({tracked_node_position[0]:.1f}, {tracked_node_position[1]:.1f}, {tracked_node_position[2]:.1f})"
    )
    print(f"  Columns: step, x_position, y_position, z_position, internal_force_l2")

if rank == 0:
    print("\nDYNAMIC ANALYSIS COMPLETE")

# Clean up PETSc objects
residual.destroy()
f_int.destroy()
f_ext_vector.destroy()

if MASS_TYPE == "consistent" and ksp is not None:
    ksp.destroy()
    M_matrix.destroy()
