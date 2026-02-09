"""
Timing analysis script for nonlinear 3D beam dynamic analysis using Symplectic Euler.
Matches test_feat10_explicit_opt.cc: Mooney-Rivlin material, HRZ lumped mass,
symplectic Euler, explicit velocity zeroing at fixed nodes.
Minimal version with only solver execution and timing.
"""
import argparse
import os
import sys
import time
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import assemble_vector, assemble_matrix
from petsc4py import PETSc
from tetgen_mesh_loader import load_tetgen_mesh_from_files

rank = MPI.COMM_WORLD.rank

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Beam dynamic explicit dynamics with lumped mass (timing version)'
    )
    parser.add_argument('--res', type=int, default=0,
                        choices=[0, 2, 4, 8, 16, 32],
                        help='Mesh resolution (0, 2, 4, 8, 16, 32)')
    parser.add_argument('--dt', type=float, default=1e-5,
                        help='Time step size (default: 1e-5)')
    parser.add_argument('--steps', type=int, default=5000,
                        help='Number of time steps (default: 5000)')
    parser.add_argument('--mat', type=str, default='mr',
                        choices=['svk', 'mr'],
                        help='Material model: svk or mr (default: mr)')
    return parser.parse_args()

args = parse_args()

if rank == 0:
    print(f"Running with {MPI.COMM_WORLD.size} MPI ranks")
    print(f"Parameters: res={args.res}, mat={args.mat}, dt={args.dt}, steps={args.steps}")

# ============================================================================
# GEOMETRY AND MESH SETUP
# ============================================================================
RES = args.res
MAT = args.mat

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
# APPLIED LOADS - Distribute 10000 N at x=3 face in +z direction
# ============================================================================
dof_coords = V.tabulate_dof_coordinates()
num_owned_dofs = dofmap.index_map.size_local

force_dofs = []
for i, coord in enumerate(dof_coords):
    if i < num_owned_dofs and abs(coord[0] - L) < 1e-6:
        force_dofs.append(i)

local_num_force_nodes = len(force_dofs)
global_num_force_nodes = domain.comm.allreduce(local_num_force_nodes, op=MPI.SUM)

total_force = 10000.0
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
# MATERIAL MODEL AND KINEMATICS
# ============================================================================
u = fem.Function(V)
u_old = fem.Function(V)
v_old = fem.Function(V)
a = fem.Function(V)

d = len(u)
I = ufl.Identity(d)
F = ufl.variable(I + ufl.grad(u))
C = F.T * F

if MAT == "mr":
    # Mooney-Rivlin parameters (match C++ FEAT10Opt test)
    mu10_val = 80000.0   # Pa
    mu01_val = 20000.0   # Pa
    kappa_val = 1e6      # Pa
    rho_val = 1100.0     # kg/m^3

    rho = fem.Constant(domain, rho_val)
    mu10 = fem.Constant(domain, default_scalar_type(mu10_val))
    mu01 = fem.Constant(domain, default_scalar_type(mu01_val))
    kappa = fem.Constant(domain, default_scalar_type(kappa_val))

    J = ufl.det(F)
    I1 = ufl.tr(C)
    I2 = 0.5 * (I1**2 - ufl.tr(C * C))

    # Strain energy function
    psi = mu10 * (I1 - 3) + mu01 * (I2 - 3) + 0.5 * kappa * (J - 1)**2

    # First Piola-Kirchhoff stress (automatic differentiation)
    P = ufl.diff(psi, F)

    if rank == 0:
        print(f"Material: Mooney-Rivlin (mu10={mu10_val}, mu01={mu01_val}, kappa={kappa_val}, rho={rho_val})")

elif MAT == "svk":
    # St. Venant-Kirchhoff parameters
    E_val = 7.0e8        # Pa
    nu_val = 0.33
    rho_val = 2700.0     # kg/m^3

    rho = fem.Constant(domain, rho_val)
    E = default_scalar_type(E_val)
    nu = default_scalar_type(nu_val)

    mu_svk = fem.Constant(domain, E / (2 * (1 + nu)))
    lmbda_svk = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

    trFtF = ufl.tr(C)
    FFt = F * F.T
    FFtF = FFt * F
    lambda_factor = lmbda_svk * (0.5 * trFtF - 1.5)
    P = lambda_factor * F + mu_svk * (FFtF - F)

    if rank == 0:
        print(f"Material: St. Venant-Kirchhoff (E={E_val}, nu={nu_val}, rho={rho_val})")

else:
    if rank == 0:
        print(f"Unknown material: {MAT}")
    sys.exit(1)

# ============================================================================
# TIME INTEGRATION SETUP (Symplectic Euler with lumped mass - matching C++ test)
# ============================================================================
dt = args.dt
n_steps = args.steps
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

force_off_step = n_steps // 2

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
    avg_step_time_ms = (elapsed_time / n_steps) * 1000.0
    throughput = 1000.0 / avg_step_time_ms
    print(f"\nTiming summary:")
    print(f"  Solver execution time: {elapsed_time:.6f} s")
    print(f"  Average step time: {avg_step_time_ms:.3f} ms")
    print(f"  Throughput: {throughput:.2f} steps/sec")

residual.destroy()
f_int.destroy()
f_ext_vector.destroy()
