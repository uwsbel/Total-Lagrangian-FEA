"""
Nonlinear 3D bunny dynamic analysis using Backward Euler time integration.
SVK material, no damping (eta_damp = lambda_damp = 0).
Matches C++ bunny benchmark: absolute z-based BCs, -z force on ear nodes, release at step 1000.
"""
import os
import sys
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, assemble_residual
from petsc4py import PETSc
from tetgen_mesh_loader import load_tetgen_mesh_from_files, write_vtk_frame

rank = MPI.COMM_WORLD.rank

# ---------------------------------------------------------------------------
# FLAGS
# ---------------------------------------------------------------------------
WRITE_VTK = True
DEBUG = False

# Line-buffer stdout on rank 0 so prints show under MPI (stdout is not a TTY).
if rank == 0:
    sys.stdout.reconfigure(line_buffering=True)
    print(f"Running with {MPI.COMM_WORLD.size} MPI ranks")

# ============================================================================
# GEOMETRY AND MESH SETUP
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))

node_file = os.path.join(project_root, "data", "meshes", "T10", "bunny_ascii_26.1.node")
ele_file  = os.path.join(project_root, "data", "meshes", "T10", "bunny_ascii_26.1.ele")

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
    print(f"Loaded mesh: bunny_ascii_26.1 (TetGen ordering)")
    print(f"Topology vertices: {topology_vertices}")
    print(f"Function space DOFs (quadratic): {total_dofs}")
    print(f"Total DOFs (including all vector components): {total_vector_dofs}")
    print(f"Total elements: {total_elements}")

# ============================================================================
# Z-RANGE (for info only — thresholds are absolute constants)
# ============================================================================
z_coords = domain.geometry.x[:, 2]
z_min = domain.comm.allreduce(float(np.min(z_coords)), op=MPI.MIN)
z_max = domain.comm.allreduce(float(np.max(z_coords)), op=MPI.MAX)

z_fix_thresh   = -4.0   # Fixed BC: z <= -4.0 (absolute)
z_force_thresh =  4.0   # Force region: z >= +4.0 (absolute)

if rank == 0:
    print(f"\nZ-range: [{z_min:.4f}, {z_max:.4f}]")
    print(f"Fixed BC threshold:  z <= {z_fix_thresh:.4f}  (absolute)")
    print(f"Force region:        z >= {z_force_thresh:.4f}  (absolute)")

# ============================================================================
# BOUNDARY CONDITIONS — Fix nodes with z <= -4.0
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
# EXTERNAL FORCE VECTOR — Per-node, downward (-z) on ear nodes (z >= 4.0)
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

force_per_node = -35000.0   # N per node, -z direction (downward on ears)

if rank == 0:
    print(f"Force region: z >= {z_force_thresh:.4f}")
    print(f"Force nodes (global): {global_num_force_nodes}")
    print(f"Force per node: {force_per_node:.1f} N (-z)")

f_temp = fem.Function(V)
f_temp.x.array[:] = 0.0
for node_idx in force_dofs:
    f_temp.x.array[node_idx * block_size + 2] = force_per_node  # -z direction

f_ext_vector = f_temp.x.petsc_vec.copy()
f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# ============================================================================
# TRACKED NODE — highest-z DOF (in force region / ear area)
# ============================================================================
if rank == 0:
    print("\nTRACKED NODE SETUP")

max_z_local = -np.inf
tracked_node_dof = None
tracked_node_coord = None
tracked_node_rank = -1

for i, coord in enumerate(dof_coords):
    if i < num_owned_dofs and coord[2] > max_z_local:
        max_z_local = coord[2]
        tracked_node_dof = i
        tracked_node_coord = coord.copy()
        tracked_node_rank = rank

# Find the rank with the globally maximum z
all_max_z = domain.comm.gather(max_z_local, root=0)

if rank == 0:
    global_max_z = max(all_max_z)
    owner_rank = all_max_z.index(global_max_z)
    print(f"Tracked node: highest-z DOF (global max z = {global_max_z:.6f})")
    print(f"  Owned by rank: {owner_rank}")

# Broadcast global max z so every rank can decide if it owns the tracked node
global_max_z_bcast = domain.comm.bcast(
    max(all_max_z) if rank == 0 else None, root=0
)
if max_z_local < global_max_z_bcast - 1e-12:
    tracked_node_dof = None
    tracked_node_coord = None

# ============================================================================
# MATERIAL MODEL — SVK, no damping
# ============================================================================
E_val  = 3.0e8
nu_val = 0.40
rho    = fem.Constant(domain, 920.0)

E  = default_scalar_type(E_val)
nu = default_scalar_type(nu_val)

mu_svk     = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda_svk  = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
lmbda_damp = fem.Constant(domain, 0.0)
eta_damp   = fem.Constant(domain, 0.0)

if rank == 0:
    print(f"\nMATERIAL: SVK (no damping)")
    print(f"  E={E_val:.2e} Pa, nu={nu_val}, rho=920 kg/m³")
    print(f"  eta_damp=0.0, lambda_damp=0.0 Pa·s")

# ============================================================================
# FUNCTION SPACE SETUP
# ============================================================================
v_test = ufl.TestFunction(V)
u      = fem.Function(V)   # current displacement (unknown)
u_old  = fem.Function(V)   # previous displacement
v_old  = fem.Function(V)   # previous velocity
B      = fem.Constant(domain, default_scalar_type((0, 0, 0)))

# ============================================================================
# TIME INTEGRATION PARAMETERS
# ============================================================================
dt_val  = 1e-3
n_steps = 8000
vtk_interval = 10

dt = fem.Constant(domain, dt_val)

if rank == 0:
    print(f"\nTIME INTEGRATION: Backward Euler")
    print(f"  dt={dt_val} s, steps={n_steps}, total time={n_steps*dt_val:.3f} s")

# ============================================================================
# VARIATIONAL FORM — SVK stress + Kelvin-Voigt viscous stress (zero damping)
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

# Kelvin-Voigt viscous stress (evaluates to zero with zero damping constants)
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
        "snes_rtol": 1e-6,
        "snes_stol": 1e-6,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    petsc_options_prefix="bunny_dynamic_be",
)

if rank == 0:
    print("\nNONLINEAR SOLVER: Newton line search (SNES), LU (MUMPS)")
    print("  atol=1e-4, rtol=1e-6, stol=1e-4")

# ============================================================================
# VTK AND CSV OUTPUT
# ============================================================================
root_output_dir = os.path.join(project_root, "output")
vtk_output_dir = os.path.join(root_output_dir, "bunny_fenics_vtk")
if rank == 0:
    os.makedirs(root_output_dir, exist_ok=True)
    if WRITE_VTK:
        os.makedirs(vtk_output_dir, exist_ok=True)
output_frame = 0

# ============================================================================
# TIME STEPPING LOOP
# ============================================================================
if rank == 0:
    print("\nSTARTING DYNAMIC ANALYSIS — BUNNY")

u_old.x.array[:] = 0.0
u_old.x.scatter_forward()
v_old.x.array[:] = 0.0
v_old.x.scatter_forward()

node_xyz_history = []

for n in range(n_steps):
    t = n * dt_val

    # Release force at step 1000
    if n == 1000:
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

    # VTK output when WRITE_VTK is True
    if WRITE_VTK and n % vtk_interval == 0:
        vtk_path = os.path.join(vtk_output_dir, f"bunny_fenics_frame_{output_frame:04d}.vtk")
        write_vtk_frame(domain, V, u, vtk_path)
        output_frame += 1

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
            print(f"Step {n:5d}: tracked=({xp:.6f}, {yp:.6f}, {zp:.6f})  "
                  f"max_disp={max_disp:.4e}  iters={num_its}")

if rank == 0:
    print("\nDYNAMIC ANALYSIS COMPLETE")

# ============================================================================
# CSV OUTPUT
# ============================================================================
if rank == 0 and len(node_xyz_history) > 0:
    csv_path = os.path.join(root_output_dir, "node_xyz_history_fenics_bunny_svk.csv")
    with open(csv_path, 'w') as f:
        f.write("step,x_position,y_position,z_position\n")
        for i, (x_val, y_val, z_val) in enumerate(node_xyz_history):
            f.write(f"{i},{x_val:.17f},{y_val:.17f},{z_val:.17f}\n")
    print(f"Wrote tracked node history to {csv_path}")
    print(f"  Total steps: {len(node_xyz_history)}")
