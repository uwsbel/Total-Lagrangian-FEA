"""
Timing analysis script for nonlinear 3D beam dynamic analysis using Forward Euler.
Matches beam_dynamic_forward_euler_lumped.py: HRZ lumped mass, symplectic Euler,
explicit velocity zeroing at fixed nodes. Minimal version with only solver execution and timing.
"""
import os
import time
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
MAT = "svk"   # svk | mr

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
dofmap = V.dofmap
block_size = dofmap.index_map_bs

# ============================================================================
# BOUNDARY CONDITIONS - Fix x=0 face
# ============================================================================
def fixed_boundary(x):
    return np.isclose(x[0], 0.0, atol=1e-6)

boundary_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)
u_zero = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
bc_fixed = fem.dirichletbc(u_zero, boundary_dofs, V)

# ============================================================================
# APPLIED LOADS - Distribute 100000 N at x=3 face in +z direction
# ============================================================================
dof_coords = V.tabulate_dof_coordinates()
num_owned_dofs = dofmap.index_map.size_local

force_dofs = []
for i, coord in enumerate(dof_coords):
    if i < num_owned_dofs and abs(coord[0] - L) < 1e-6:
        force_dofs.append(i)

local_num_force_nodes = len(force_dofs)
global_num_force_nodes = domain.comm.allreduce(local_num_force_nodes, op=MPI.SUM)

total_force = 100000.0
force_per_node = total_force / global_num_force_nodes if global_num_force_nodes > 0 else 0.0

f_temp = fem.Function(V)
f_temp.x.array[:] = 0.0
for node_idx in force_dofs:
    f_temp.x.array[node_idx * block_size + 2] = force_per_node

f_ext_vector = f_temp.x.petsc_vec.copy()
f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

if rank == 0:
    print(f"Load applied at x=3: {total_force} N (+z direction)")

# ============================================================================
# MATERIAL MODEL AND KINEMATICS (SVK - matching lumped script)
# ============================================================================
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

d = len(u)
I = ufl.Identity(d)
F = ufl.variable(I + ufl.grad(u))
C = F.T * F

# SVK (St. Venant-Kirchhoff)
mu_svk = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda_svk = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

trFtF = ufl.tr(C)
FFt = F * F.T
FFtF = FFt * F
lambda_factor = lmbda_svk * (0.5 * trFtF - 1.5)
P = lambda_factor * F + mu_svk * (FFtF - F)

# ============================================================================
# TIME INTEGRATION SETUP (Forward Euler with lumped mass - matching lumped script)
# ============================================================================
dt = 1e-5
n_steps = 200000
t_final = n_steps * dt

# ============================================================================
# ASSEMBLE AND LUMP MASS MATRIX (HRZ - matching lumped script)
# ============================================================================
metadata = {"quadrature_degree": 5}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

u_trial = ufl.TrialFunction(V)
v_test = ufl.TestFunction(V)
M_form = fem.form(rho * ufl.inner(u_trial, v_test) * dx)

M_matrix_no_bc = assemble_matrix(M_form)
M_matrix_no_bc.assemble()

# HRZ lumping: diagonal scaling
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
    if M_lumped_array[i] > 1e-30:
        M_lumped_inv_array[i] = 1.0 / M_lumped_array[i]

for dof in boundary_dofs:
    for c in range(block_size):
        idx = dof * block_size + c
        if idx < len(M_lumped_inv_array):
            M_lumped_inv_array[idx] = 0.0

M_diag.destroy()
M_rowsum.destroy()
M_matrix_no_bc.destroy()

if rank == 0:
    print(f"HRZ lumped mass assembled (RES={RES}, MAT={MAT})")

# Internal force form
f_int_form = fem.form(ufl.inner(ufl.grad(v_test), P) * dx)

# ============================================================================
# TIME STEPPING LOOP WITH TIMING (matching lumped script)
# ============================================================================
u_old.x.array[:] = 0.0
u_old.x.scatter_forward()

v_old.x.array[:] = 0.0
v_old.x.scatter_forward()

residual = f_ext_vector.copy()
f_int = f_ext_vector.copy()

force_off_step = int(1.0 / dt)

# Start timing
start_time = time.perf_counter()

for n in range(n_steps):
    if n == force_off_step:
        f_ext_vector.zeroEntries()
        f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Internal force from current displacement
    u.x.array[:] = u_old.x.array[:]
    u.x.scatter_forward()

    with f_int.localForm() as f_int_local:
        f_int_local.set(0.0)
    assemble_vector(f_int, f_int_form)
    f_int.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    fem.petsc.set_bc(f_int, [bc_fixed])

    # Residual and acceleration (lumped mass: element-wise)
    residual.zeroEntries()
    residual.axpy(1.0, f_ext_vector)
    residual.axpy(-1.0, f_int)

    residual_array = residual.getArray()
    # With MPI: residual/M_lumped_inv are owned-only; a.x.array includes ghosts
    n_local = len(residual_array)
    a.x.array[:n_local] = residual_array * M_lumped_inv_array
    a.x.scatter_forward()

    # Symplectic Euler + explicit velocity zeroing at fixed nodes (matching GPU)
    v_new = v_old.x.array[:] + dt * a.x.array[:]

    for dof in boundary_dofs:
        for c in range(block_size):
            idx = dof * block_size + c
            if idx < len(v_new):
                v_new[idx] = 0.0

    u_new = u_old.x.array[:] + dt * v_new[:]

    u_old.x.array[:] = u_new[:]
    u_old.x.scatter_forward()

    v_old.x.array[:] = v_new[:]
    v_old.x.scatter_forward()

# End timing
end_time = time.perf_counter()
elapsed_time = end_time - start_time

if rank == 0:
    print(f"Solver execution time (s): {elapsed_time:.6f}")
    print(f"Average time per step (ms): {(elapsed_time / n_steps) * 1000:.3f}")

residual.destroy()
f_int.destroy()
f_ext_vector.destroy()
