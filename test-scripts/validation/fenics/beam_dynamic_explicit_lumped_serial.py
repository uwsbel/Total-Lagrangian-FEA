"""Nonlinear 3D beam dynamics with Symplectic Euler and HRZ lumped mass (serial, no MPI)."""
import os
import numpy as np
import ufl

from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import assemble_vector, assemble_matrix
from petsc4py import PETSc
from tetgen_mesh_loader import load_tetgen_mesh_from_files_serial

# -----------------------------------------------------------------------------
# Config (edit these)
# -----------------------------------------------------------------------------
MAT = "mr"            # "svk" | "mr"
RES = 0               # 0 | 2 | 4 | 8 | 16 | 32
DT = 1e-5
STEPS = 10
CSV_PATH = None       # None = no CSV, "" = default in output/, or path
DUMP_IFORCE = None    # None = no dump, "" = default in output/, or path
DUMP_EVERY = 1
# -----------------------------------------------------------------------------

# Construct mesh file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
mesh_dir = os.path.join(project_root, "data", "meshes", "T10", "resolution")

node_file = os.path.join(mesh_dir, f"beam_3x2x1_res{RES}.1.node")
ele_file = os.path.join(mesh_dir, f"beam_3x2x1_res{RES}.1.ele")

# Load TetGen mesh (serial: COMM_SELF, no MPI distribution)
domain, _ = load_tetgen_mesh_from_files_serial(node_file, ele_file)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))

# Beam dimensions (for boundary conditions)
L = 3.0   # Length (x)
W = 2.0   # Width (y)
H = 1.0   # Height (z)

# Print total nodes and elements (serial: all data local)
topology_vertices = domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
total_elements = domain.topology.index_map(domain.topology.dim).size_local + domain.topology.index_map(domain.topology.dim).num_ghosts
dofmap = V.dofmap
total_dofs = dofmap.index_map.size_local + dofmap.index_map.num_ghosts
block_size = dofmap.index_map_bs
total_vector_dofs = total_dofs * block_size
num_owned_dofs = dofmap.index_map.size_local

print(f"Loaded TetGen mesh (serial): beam_3x2x1 (RES_{RES})")
print(f"Topology vertices: {topology_vertices}")
print(f"Function space DOFs (quadratic): {total_dofs}")
print(f"Total DOFs (including all vector components): {total_vector_dofs}")
print(f"Total elements: {total_elements}")

# ============================================================================
# BOUNDARY CONDITIONS - Fix x=0 face
# ============================================================================
print("\nBOUNDARY CONDITIONS SETUP")


def fixed_boundary(x):
    return np.isclose(x[0], 0.0, atol=1e-6)


boundary_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)
u_zero = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
bc_fixed = fem.dirichletbc(u_zero, boundary_dofs, V)

print(f"Fixed boundary at x=0:")
print(f"  Number of DOFs found: {len(boundary_dofs)}")
print(f"  Constrained scalar DOFs: {len(boundary_dofs) * block_size}")
print(f"  Free scalar DOFs: {total_vector_dofs - len(boundary_dofs) * block_size}")

# ============================================================================
# EXTERNAL FORCE VECTOR (DIRECT POINT LOADS)
# ============================================================================
print("\nAPPLIED LOADS SETUP")

dof_coords = V.tabulate_dof_coordinates()
force_dofs = []
for i, coord in enumerate(dof_coords):
    if i < num_owned_dofs and abs(coord[0] - L) < 1e-6:
        force_dofs.append(i)

num_force_nodes = len(force_dofs)
total_force = 10000.0
force_per_node = total_force / num_force_nodes if num_force_nodes > 0 else 0.0

print(f"Applying Lumped Force (+z): {force_per_node} N per node on {num_force_nodes} nodes.")
print(f"  Force will be applied for first half of steps, then turned off.")

f_temp = fem.Function(V)
f_temp.x.array[:] = 0.0
for node_idx in force_dofs:
    f_temp.x.array[node_idx * block_size + 2] = force_per_node

f_ext_vector = f_temp.x.petsc_vec.copy()
f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

print(f"Load applied at x=3:")
print(f"  Total force: {total_force} N (+z direction)")
print(f"  Number of nodes at x=3: {num_force_nodes}")
print(f"  Force per node: {force_per_node:.6f} N")

# ============================================================================
# TRACKED NODE - top corner (L, W, H) = (3, 2, 1)
# ============================================================================
print("\nTRACKED NODE SETUP")

tracked_node_position = np.array([L, W, H])
tracked_node_dof = None
tracked_node_coord = None

for i, coord in enumerate(dof_coords):
    if i < num_owned_dofs and (
        abs(coord[0] - tracked_node_position[0]) < 1e-6
        and abs(coord[1] - tracked_node_position[1]) < 1e-6
        and abs(coord[2] - tracked_node_position[2]) < 1e-6
    ):
        tracked_node_dof = i
        tracked_node_coord = coord.copy()
        break

if tracked_node_dof is not None:
    print(f"Tracked node at ({tracked_node_position[0]}, {tracked_node_position[1]}, {tracked_node_position[2]}), DOF index: {tracked_node_dof}")
else:
    print("WARNING: No DOF found at tracked position")

# ============================================================================
# MATERIAL MODEL AND KINEMATICS
# ============================================================================
MATERIAL_MODEL = "SVK" if MAT.lower() == "svk" else "MOONEY_RIVLIN"

if MATERIAL_MODEL == "MOONEY_RIVLIN":
    mu10_val = 80000.0
    mu01_val = 20000.0
    kappa_val = 1.0e6
    rho_val = 1100.0
    E_val = 6.0 * (mu10_val + mu01_val)
    nu_val = 0.45
else:
    E_val = 7.0e8
    nu_val = 0.33
    rho_val = 2700.0

rho = fem.Constant(domain, rho_val)
E = default_scalar_type(E_val)
nu = default_scalar_type(nu_val)

u = fem.Function(V)
u_old = fem.Function(V)
v_old = fem.Function(V)
a = fem.Function(V)
B = fem.Constant(domain, default_scalar_type((0, 0, 0)))

d = len(u)
I = ufl.Identity(d)
F = ufl.variable(I + ufl.grad(u))
C = F.T * F

if MATERIAL_MODEL == "SVK":
    mu_svk = fem.Constant(domain, E / (2 * (1 + nu)))
    lmbda_svk = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
    trFtF = ufl.tr(C)
    FFt = F * F.T
    FFtF = FFt * F
    lambda_factor = lmbda_svk * (0.5 * trFtF - 1.5)
    P = lambda_factor * F + mu_svk * (FFtF - F)
    print("\nMATERIAL MODEL: St. Venant-Kirchhoff (SVK)")
else:
    C1 = fem.Constant(domain, default_scalar_type(mu10_val))
    C2 = fem.Constant(domain, default_scalar_type(mu01_val))
    kappa = fem.Constant(domain, default_scalar_type(kappa_val))
    print("\nMATERIAL MODEL: Mooney-Rivlin")
    print(f"  C1 (mu10): {mu10_val:.4e}, C2 (mu01): {mu01_val:.4e}, Kappa: {kappa_val:.4e}, Density: {rho_val:.1f} kg/m³")
    I1 = ufl.tr(C)
    C_squared = C * C
    trC2 = ufl.tr(C_squared)
    I2 = 0.5 * (I1**2 - trC2)
    J = ufl.det(F)
    I1_bar = J**(-2.0 / 3.0) * I1
    I2_bar = J**(-4.0 / 3.0) * I2
    psi = C1 * (I1_bar - 3.0) + C2 * (I2_bar - 3.0) + 0.5 * kappa * (J - 1.0) ** 2
    P = ufl.diff(psi, F)

# ============================================================================
# TIME INTEGRATION SETUP
# ============================================================================
dt = DT
n_steps = STEPS
t_final = n_steps * dt

print("\nTIME INTEGRATION SETUP")
print(f"Method: Symplectic Euler (Explicit) with LUMPED MASS (serial)")
print(f"Time step (dt): {dt} s")
print(f"Number of steps: {n_steps}")
print(f"Total simulation time: {t_final} s")
print(f"Force turned off at step n_steps//2")

# ============================================================================
# ASSEMBLE AND LUMP MASS MATRIX (HRZ)
# ============================================================================
print("\nASSEMBLING AND LUMPING MASS MATRIX (HRZ Method)")

metadata = {"quadrature_degree": 5}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)
u_trial = ufl.TrialFunction(V)
v_test = ufl.TestFunction(V)
M_form = fem.form(rho * ufl.inner(u_trial, v_test) * dx)

M_matrix_no_bc = assemble_matrix(M_form)
M_matrix_no_bc.assemble()
print(f"Consistent mass matrix assembled: {M_matrix_no_bc.size[0]} x {M_matrix_no_bc.size[1]}")

M_diag = M_matrix_no_bc.createVecLeft()
M_matrix_no_bc.getDiagonal(M_diag)
M_diag_array = M_diag.getArray().copy()

ones = M_matrix_no_bc.createVecRight()
ones.set(1.0)
M_rowsum = M_matrix_no_bc.createVecLeft()
M_matrix_no_bc.mult(ones, M_rowsum)
M_rowsum_array = M_rowsum.getArray().copy()
ones.destroy()

total_mass_consistent = M_rowsum.sum()
diag_sum = M_diag.sum()
scale_factor = total_mass_consistent / diag_sum if abs(diag_sum) > 1e-30 else 1.0
M_lumped_array = M_diag_array * scale_factor

M_lumped_inv_array = np.zeros_like(M_lumped_array)
for i in range(len(M_lumped_array)):
    M_lumped_inv_array[i] = 1.0 / M_lumped_array[i] if M_lumped_array[i] > 1e-30 else 0.0

for dof in boundary_dofs:
    for c in range(block_size):
        idx = dof * block_size + c
        if idx < len(M_lumped_inv_array):
            M_lumped_inv_array[idx] = 0.0

M_diag.destroy()
M_rowsum.destroy()
M_matrix_no_bc.destroy()

# Mass conservation: HRZ scaling preserves total mass (sum of lumped diagonal = 1^T M 1)
total_mass_lumped = float(np.sum(M_lumped_array))
print(f"HRZ lumped mass: total mass {total_mass_consistent:.6f} kg, scale factor {scale_factor:.6f}")
print(f"  Mass conserved: sum(M_lumped) = {total_mass_lumped:.6f} kg (should equal {total_mass_consistent:.6f})")

# ============================================================================
# TIME STEPPING LOOP
# ============================================================================
print("\nSTARTING DYNAMIC ANALYSIS (LUMPED MASS)")

u_old.x.array[:] = 0.0
u_old.x.scatter_forward()
v_old.x.array[:] = 0.0
v_old.x.scatter_forward()

node_xyz_history = []
residual = f_ext_vector.copy()
f_int = f_ext_vector.copy()
f_int_form = fem.form(ufl.inner(ufl.grad(v_test), P) * dx)
force_off_step = n_steps // 2
print(f"Force turned off at step {force_off_step} (t = {force_off_step * dt:.4f}s)")

# Internal force dump
iforce_file = None
if DUMP_IFORCE is not None:
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    mat_suffix = "svk" if MATERIAL_MODEL == "SVK" else "mr"
    if DUMP_IFORCE == "":
        dump_path = os.path.join(
            output_dir, f"fenics_serial_iforce_res{RES}_{mat_suffix}.csv"
        )
    else:
        dump_path = DUMP_IFORCE
    iforce_file = open(dump_path, "w")
    iforce_file.write("step,node,fx,fy,fz\n")
    print(f"Internal force dump: {dump_path} every {DUMP_EVERY} step(s)")

for n in range(n_steps):
    t = n * dt

    if n == force_off_step:
        f_ext_vector.zeroEntries()
        f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        print(f"Force turned off at step {n} (t = {t:.4f}s)")

    u.x.array[:] = u_old.x.array[:]
    u.x.scatter_forward()

    with f_int.localForm() as f_int_local:
        f_int_local.set(0.0)
    assemble_vector(f_int, f_int_form)
    f_int.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(f_int, [bc_fixed])

    if iforce_file is not None and n % DUMP_EVERY == 0:
        f_int_array = f_int.getArray()
        n_blocks = len(f_int_array) // block_size
        for node in range(n_blocks):
            fx = f_int_array[block_size * node + 0]
            fy = f_int_array[block_size * node + 1]
            fz = f_int_array[block_size * node + 2]
            iforce_file.write(f"{n},{node},{fx:.17e},{fy:.17e},{fz:.17e}\n")

    residual.zeroEntries()
    residual.axpy(1.0, f_ext_vector)
    residual.axpy(-1.0, f_int)
    residual_array = residual.getArray()
    a.x.array[:] = residual_array * M_lumped_inv_array
    a.x.scatter_forward()

    v_new = v_old.x.array[:] + dt * a.x.array[:]
    for dof in boundary_dofs:
        for c in range(block_size):
            idx = dof * block_size + c
            if idx < len(v_new):
                v_new[idx] = 0.0

    u_new = u_old.x.array[:] + dt * v_new[:]

    if tracked_node_dof is not None:
        u_x_at_node = u_new[tracked_node_dof * block_size + 0]
        u_y_at_node = u_new[tracked_node_dof * block_size + 1]
        u_z_at_node = u_new[tracked_node_dof * block_size + 2]
        x_position = tracked_node_coord[0] + u_x_at_node
        y_position = tracked_node_coord[1] + u_y_at_node
        z_position = tracked_node_coord[2] + u_z_at_node
        node_xyz_history.append([float(x_position), float(y_position), float(z_position)])

    u_old.x.array[:] = u_new[:]
    u_old.x.scatter_forward()
    v_old.x.array[:] = v_new[:]
    v_old.x.scatter_forward()

    if n % 10000 == 0 or n < 5:
        max_disp = np.max(np.linalg.norm(u_old.x.array.reshape(-1, 3), axis=1))
        max_vel = np.max(np.linalg.norm(v_old.x.array.reshape(-1, 3), axis=1))
        if node_xyz_history:
            x_pos, y_pos, z_pos = node_xyz_history[-1]
            print(f"Step {n}/{n_steps}: t={t:.4f}s, tracked = ({x_pos:.17f}, {y_pos:.17f}, {z_pos:.17f}), max_disp={max_disp:.6e}, max_vel={max_vel:.6e}")

if iforce_file is not None:
    iforce_file.close()

print("\nDYNAMIC ANALYSIS COMPLETE")

# ============================================================================
# SAVE CSV
# ============================================================================
write_csv = CSV_PATH is not None
if write_csv and node_xyz_history:
    if CSV_PATH == "":
        output_dir = os.path.join(script_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        mat_suffix = "svk" if MATERIAL_MODEL == "SVK" else "mr"
        csv_path = os.path.join(output_dir, f"node_xyz_history_fenics_serial_res{RES}_{mat_suffix}_fe_lumped.csv")
    else:
        csv_path = CSV_PATH
    with open(csv_path, "w") as f:
        f.write("step,x_position,y_position,z_position\n")
        for i, (x_val, y_val, z_val) in enumerate(node_xyz_history):
            f.write(f"{i},{x_val:.17f},{y_val:.17f},{z_val:.17f}\n")
    print(f"Wrote {csv_path}")
    print(f"  Node position: ({tracked_node_position[0]:.1f}, {tracked_node_position[1]:.1f}, {tracked_node_position[2]:.1f}), steps: {len(node_xyz_history)}")

residual.destroy()
f_int.destroy()
f_ext_vector.destroy()
