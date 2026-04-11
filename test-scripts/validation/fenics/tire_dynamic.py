"""
Nonlinear 3D tire dynamic analysis using Backward Euler time integration.
SVK material + Kelvin-Voigt viscous damping.
Matches C++ tire benchmark: z-based BCs, -z force on top nodes, release at step 400.

Usage: mpirun -np N python tire_dynamic.py --res RES
"""
import argparse
import os
import sys
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, assemble_residual
from petsc4py import PETSc
from tet10_mesh_utils import load_tetgen_mesh_from_files, locate_raw_node_dof
from dolfinx.io import VTKFile

rank = MPI.COMM_WORLD.rank

# ---------------------------------------------------------------------------
# FLAGS
# ---------------------------------------------------------------------------
WRITE_VTK = False
DEBUG = False

# Line-buffer stdout on rank 0 so prints show under MPI (stdout is not a TTY).
if rank == 0:
    sys.stdout.reconfigure(line_buffering=True)
    print(f"Running with {MPI.COMM_WORLD.size} MPI ranks")

# ============================================================================
# GEOMETRY AND MESH SETUP
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=0)
args = parser.parse_args()
RES = args.res

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))

mesh_dir = os.path.join(project_root, "data", "meshes", "T10", "tire_scaling")
node_file = os.path.join(mesh_dir, f"tire_res{RES}.1.node")
ele_file  = os.path.join(mesh_dir, f"tire_res{RES}.1.ele")

domain, _ = load_tetgen_mesh_from_files(node_file, ele_file, tetgen_order=True)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))

# Print mesh statistics
topology_vertices = (domain.topology.index_map(0).size_local +
                     domain.topology.index_map(0).num_ghosts)
total_elements = (domain.topology.index_map(domain.topology.dim).size_local +
                  domain.topology.index_map(domain.topology.dim).num_ghosts)
dofmap = V.dofmap
total_dofs = dofmap.index_map.size_local + dofmap.index_map.num_ghosts
block_size = dofmap.index_map_bs
total_vector_dofs = total_dofs * block_size

if rank == 0:
    print(f"Loaded mesh: tire_res{RES} (TetGen ordering)")
    print(f"Topology vertices: {topology_vertices}")
    print(f"Function space DOFs (quadratic): {total_dofs}")
    print(f"Total DOFs (including all vector components): {total_vector_dofs}")
    print(f"Total elements: {total_elements}")

# ============================================================================
# Z-RANGE COMPUTATION (requires MPI reduction for parallel correctness)
# ============================================================================
z_coords = domain.geometry.x[:, 2]
local_z_min = float(np.min(z_coords)) if z_coords.size > 0 else np.inf
local_z_max = float(np.max(z_coords)) if z_coords.size > 0 else -np.inf
z_min = domain.comm.allreduce(local_z_min, op=MPI.MIN)
z_max = domain.comm.allreduce(local_z_max, op=MPI.MAX)
z_range = z_max - z_min

z_fix_thresh   = z_min + 0.1 * z_range   # Fix bottom 10%
z_force_thresh = z_min + 0.9 * z_range   # Apply force to top 10%

if rank == 0:
    print(f"\nZ-range: [{z_min:.4f}, {z_max:.4f}]  (range={z_range:.4f})")
    print(f"Fixed BC threshold:  z <= {z_fix_thresh:.4f}")
    print(f"Force region:        z >= {z_force_thresh:.4f}")

# ============================================================================
# BOUNDARY CONDITIONS — Fix nodes with z <= z_fix_thresh
# ============================================================================
if rank == 0:
    print("\nBOUNDARY CONDITIONS SETUP")

def fixed_boundary(x):
    return x[2] <= z_fix_thresh + 1e-8

boundary_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)
bc_fixed = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), boundary_dofs, V)

if rank == 0:
    print(f"Fixed boundary (z <= {z_fix_thresh:.4f}):")
    print(f"  Number of constrained DOFs: {len(boundary_dofs)}")

# ============================================================================
# EXTERNAL FORCE VECTOR — Lumped nodal forces in -z on top nodes
# ============================================================================
if rank == 0:
    print("\nAPPLIED LOADS SETUP")

dof_coords = V.tabulate_dof_coordinates()
num_owned_dofs = dofmap.index_map.size_local

force_dofs = []
for i, coord in enumerate(dof_coords):
    if i < num_owned_dofs and coord[2] >= z_force_thresh - 1e-8:
        force_dofs.append(i)

local_num_force_nodes = len(force_dofs)
global_num_force_nodes = domain.comm.allreduce(local_num_force_nodes, op=MPI.SUM)

total_force_z = 3000.0  # N, -z direction
force_per_node = total_force_z / global_num_force_nodes if global_num_force_nodes > 0 else 0.0

if rank == 0:
    print(f"Force region: z >= {z_force_thresh:.4f}")
    print(f"Force nodes (global): {global_num_force_nodes}")
    print(f"Total force: {total_force_z} N (-z)")
    print(f"Force per node: {force_per_node:.6f} N")

f_temp = fem.Function(V)
f_temp.x.array[:] = 0.0
for node_idx in force_dofs:
    f_temp.x.array[node_idx * block_size + 2] = -force_per_node  # -z direction

f_ext_vector = f_temp.x.petsc_vec.copy()
f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# ============================================================================
# TRACKED NODE — exact raw node id by resolution
# ============================================================================
if rank == 0:
    print("\nTRACKED NODE SETUP")


tire_tracked_node_raw_ids = {
    0: 416,
    2: 765,
    4: 1713,
    8: 3708,
    16: 28836,
}
if RES not in tire_tracked_node_raw_ids:
    raise RuntimeError(f"Unsupported RES={RES} for tire tracked-node lookup.")

tracked_node_raw_id = tire_tracked_node_raw_ids[RES]
tracked_node_position, tracked_node_dof, tracked_node_coord, tracked_node_rank = locate_raw_node_dof(
    domain, rank, dof_coords, num_owned_dofs, node_file, tracked_node_raw_id
)

if rank == 0:
    print("Tracked node (crown):")
    print(f"  Raw node id: {tracked_node_raw_id}")
    print(f"  Owner rank: {tracked_node_rank}")
    print(f"  Coordinates: ({tracked_node_coord[0]:.6f}, "
          f"{tracked_node_coord[1]:.6f}, {tracked_node_coord[2]:.6f})")
if tracked_node_dof is not None:
    print(f"  DOF index: {tracked_node_dof}")

# ============================================================================
# MATERIAL MODEL — SVK + Kelvin-Voigt damping
# ============================================================================
E_val  = 1.0e7
nu_val = 0.35
rho    = fem.Constant(domain, 1000.0)

E  = default_scalar_type(E_val)
nu = default_scalar_type(nu_val)

mu_svk     = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda_svk  = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
lmbda_damp = fem.Constant(domain, 5.0e3)
eta_damp   = fem.Constant(domain, 5.0e3)

if rank == 0:
    print(f"\nMATERIAL: SVK + Kelvin-Voigt damping")
    print(f"  E={E_val:.2e} Pa, nu={nu_val}, rho=1000 kg/m³")
    print(f"  eta_damp={5.0e3:.2e}, lambda_damp={5.0e3:.2e} Pa·s")

# ============================================================================
# FUNCTION SPACE SETUP
# ============================================================================
v_test = ufl.TestFunction(V)
u      = fem.Function(V)   # current displacement (unknown)
u.name = "displacement"
u_old  = fem.Function(V)   # previous displacement
v_old  = fem.Function(V)   # previous velocity
B      = fem.Constant(domain, default_scalar_type((0, 0, 0)))

# ============================================================================
# TIME INTEGRATION PARAMETERS
# ============================================================================
dt_val  = 5e-4
n_steps = 1200
vtk_interval = 20

dt = fem.Constant(domain, dt_val)

if rank == 0:
    print(f"\nTIME INTEGRATION: Backward Euler")
    print(f"  dt={dt_val} s, steps={n_steps}, total time={n_steps*dt_val:.3f} s")

# ============================================================================
# VARIATIONAL FORM — SVK stress + Kelvin-Voigt viscous stress
# ============================================================================
metadata = {"quadrature_degree": 5}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

d = len(u)
I_tensor = ufl.Identity(d)
F_grad = ufl.variable(I_tensor + ufl.grad(u))
C_tensor = F_grad.T * F_grad

# Backward Euler kinematics
v_current = (u - u_old) / dt
a_current = (v_current - v_old) / dt

# SVK first Piola-Kirchhoff stress
trFtF = ufl.tr(C_tensor)
FFtF  = F_grad * F_grad.T * F_grad
lambda_factor = lmbda_svk * (0.5 * trFtF - 1.5)
P_svk = lambda_factor * F_grad + mu_svk * (FFtF - F_grad)

# Kelvin-Voigt viscous stress (Backward Euler rate approximation)
F_rate = ufl.grad(u - u_old) / dt
E_rate = 0.5 * (F_grad.T * F_rate + F_rate.T * F_grad)
P_vis  = lmbda_damp * ufl.tr(E_rate) * F_grad + 2.0 * eta_damp * (F_grad * E_rate)

P_total = P_svk + P_vis

F_form = (rho * ufl.inner(a_current, v_test) * dx +
          ufl.inner(ufl.grad(v_test), P_total) * dx -
          ufl.inner(v_test, B) * dx)

# ============================================================================
# CUSTOM SOLVER — PointLoadProblem (adds external force vector to residual)
# ============================================================================
class PointLoadProblem(NonlinearProblem):
    def __init__(self, F, u, f_ext_vector, bcs=None, **kwargs):
        super().__init__(F, u, bcs=bcs, **kwargs)
        self.f_ext_vector = f_ext_vector
        self._bcs = bcs if bcs is not None else []

        def residual_callback(snes, x, b, ctx=None):
            assemble_residual(self.u, self.F, self.J, self._bcs, snes, x, b)
            b.axpy(-1.0, self.f_ext_vector)

        self.solver.setFunction(residual_callback, self.b)


problem = PointLoadProblem(
    F_form,
    u,
    f_ext_vector=f_ext_vector,
    bcs=[bc_fixed],
    petsc_options={
        "snes_type": "newtonls",
        "snes_atol": 1e-4,
        "snes_rtol": 1e-4,
        "snes_stol": 1e-6,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    petsc_options_prefix="tire_dynamic_be",
)

if rank == 0:
    print("\nNONLINEAR SOLVER: Newton line search (SNES), LU (MUMPS)")
    print("  atol=1e-4, rtol=1e-4, stol=1e-6")

# ============================================================================
# VTK AND CSV OUTPUT
# ============================================================================
root_output_dir = os.path.join(project_root, "output")
vtk_output_dir = os.path.join(root_output_dir, "tire_fenics_vtu")
os.makedirs(vtk_output_dir, exist_ok=True)
vtk_file = None
if WRITE_VTK:
    vtk_file = VTKFile(domain.comm,
                       os.path.join(vtk_output_dir, "tire_fenics.pvd"), "w")

# ============================================================================
# TIME STEPPING LOOP
# ============================================================================
if rank == 0:
    print("\nSTARTING DYNAMIC ANALYSIS — TIRE")

u_old.x.array[:] = 0.0
u_old.x.scatter_forward()
v_old.x.array[:] = 0.0
v_old.x.scatter_forward()

node_xyz_history = []

for n in range(n_steps):
    t = n * dt_val

    # Release force at step 400
    if n == 400:
        f_ext_vector.set(0.0)
        f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                  mode=PETSc.ScatterMode.FORWARD)
        if rank == 0:
            print(f"Step {n}: Force released.")

    # Solve nonlinear system
    problem.solve()
    converged = problem.solver.getConvergedReason()
    num_its = problem.solver.getIterationNumber()
    assert converged > 0, f"Newton solver did not converge at step {n} (reason {converged})."
    u.x.scatter_forward()

    # VTK output
    if vtk_file is not None and n % vtk_interval == 0:
        vtk_file.write_function([u], t)

    # Update velocity (Backward Euler)
    v_new = (u.x.array - u_old.x.array) / dt_val

    # Track node position
    local_position = None
    if tracked_node_dof is not None:
        u_x = u.x.array[tracked_node_dof * block_size + 0]
        u_y = u.x.array[tracked_node_dof * block_size + 1]
        u_z = u.x.array[tracked_node_dof * block_size + 2]
        x_pos = tracked_node_coord[0] + u_x
        y_pos = tracked_node_coord[1] + u_y
        z_pos = tracked_node_coord[2] + u_z
        local_position = [float(x_pos), float(y_pos), float(z_pos)]

    all_positions = domain.comm.gather(local_position, root=0)
    if rank == 0:
        node_position = next((pos for pos in all_positions if pos is not None), None)
        if node_position is not None:
            node_xyz_history.append(node_position)

    # Update state
    u_old.x.array[:] = u.x.array[:]
    u_old.x.scatter_forward()
    v_old.x.array[:] = v_new[:]
    v_old.x.scatter_forward()

    # Progress output every 10 steps (or first few)
    if rank == 0 and (n % 10 == 0 or n < 5):
        max_disp = np.max(np.linalg.norm(u.x.array.reshape(-1, 3), axis=1))
        if len(node_xyz_history) > 0:
            xp, yp, zp = node_xyz_history[-1]
            print(f"Step {n:4d}: tracked=({xp:.6f}, {yp:.6f}, {zp:.6f})  "
                  f"max_disp={max_disp:.4e}  iters={num_its}")

if vtk_file is not None:
    vtk_file.close()

if rank == 0:
    print("\nDYNAMIC ANALYSIS COMPLETE")

# ============================================================================
# CSV OUTPUT
# ============================================================================
if rank == 0 and len(node_xyz_history) > 0:
    csv_path = os.path.join(root_output_dir, f"node_xyz_history_fenics_tire_res{RES}_svk.csv")
    with open(csv_path, 'w') as f:
        f.write(f"# node: crown (raw id {tracked_node_raw_id})\n")
        f.write("step,x_position,y_position,z_position\n")
        for i, (x_val, y_val, z_val) in enumerate(node_xyz_history):
            f.write(f"{i},{x_val:.17f},{y_val:.17f},{z_val:.17f}\n")
    print(f"Wrote tracked node history to {csv_path}")
    print(f"  Total steps: {len(node_xyz_history)}")
