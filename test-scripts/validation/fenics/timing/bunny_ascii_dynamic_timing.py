"""
Timing analysis for nonlinear 3D bunny (legacy ascii mesh) dynamic analysis using Backward Euler.
Minimal version with only solver execution and timing.

Usage: mpirun -np N python bunny_ascii_dynamic_timing.py
"""
import os
import sys
import time
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, assemble_residual
from petsc4py import PETSc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from tet10_mesh_utils import load_tetgen_mesh_from_files

rank = MPI.COMM_WORLD.rank

if rank == 0:
    print(f"Running with {MPI.COMM_WORLD.size} MPI ranks")

# ============================================================================
# GEOMETRY AND MESH SETUP
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir, os.pardir))

node_file = os.path.join(project_root, "data", "meshes", "T10", "bunny_ascii_26.1.node")
ele_file  = os.path.join(project_root, "data", "meshes", "T10", "bunny_ascii_26.1.ele")

domain, _ = load_tetgen_mesh_from_files(node_file, ele_file, tetgen_order=True)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))

# ============================================================================
# BOUNDARY CONDITIONS — Fix nodes with z <= -4.0 (absolute threshold)
# ============================================================================
z_fix_thresh   = -4.0   # Fixed BC: z <= -4.0 (absolute)
z_force_thresh =  4.0   # Force region: z >= +4.0 (absolute)

if rank == 0:
    print(f"Load applied: z >= {z_force_thresh:.4f}, -35000 N per node (-z)")

def fixed_boundary(x):
    return x[2] <= z_fix_thresh + 1e-8

boundary_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)
bc_fixed = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), boundary_dofs, V)

# ============================================================================
# EXTERNAL FORCE VECTOR — Per-node, downward (-z) on ear nodes (z >= 4.0)
# ============================================================================
dof_coords = V.tabulate_dof_coordinates()
dofmap = V.dofmap
num_owned_dofs = dofmap.index_map.size_local
block_size = dofmap.index_map_bs

force_dofs = []
for i, coord in enumerate(dof_coords):
    if i < num_owned_dofs and coord[2] >= z_force_thresh - 1e-8:
        force_dofs.append(i)

force_per_node = -35000.0   # N per node, -z direction (downward on ears)

f_temp = fem.Function(V)
f_temp.x.array[:] = 0.0
for node_idx in force_dofs:
    f_temp.x.array[node_idx * block_size + 2] = force_per_node  # -z direction

f_ext_vector = f_temp.x.petsc_vec.copy()
f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

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

# ============================================================================
# FUNCTION SPACE SETUP
# ============================================================================
v_test = ufl.TestFunction(V)
u      = fem.Function(V)
u_old  = fem.Function(V)
v_old  = fem.Function(V)

# ============================================================================
# TIME INTEGRATION PARAMETERS
# ============================================================================
dt_val  = 1e-3
n_steps = 8000

dt = fem.Constant(domain, dt_val)

# ============================================================================
# VARIATIONAL FORM — SVK stress + Kelvin-Voigt viscous stress (zero damping)
# ============================================================================
metadata = {"quadrature_degree": 5}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

d = len(u)
I_tensor = ufl.Identity(d)
F_grad = ufl.variable(I_tensor + ufl.grad(u))
C_tensor = F_grad.T * F_grad

v_current = (u - u_old) / dt
a_current = (v_current - v_old) / dt

trFtF = ufl.tr(C_tensor)
FFtF  = F_grad * F_grad.T * F_grad
lambda_factor = lmbda_svk * (0.5 * trFtF - 1.5)
P_svk = lambda_factor * F_grad + mu_svk * (FFtF - F_grad)

F_rate = ufl.grad(u - u_old) / dt
E_rate = 0.5 * (F_grad.T * F_rate + F_rate.T * F_grad)
P_vis  = lmbda_damp * ufl.tr(E_rate) * F_grad + 2.0 * eta_damp * (F_grad * E_rate)

P_total = P_svk + P_vis

F_form = (rho * ufl.inner(a_current, v_test) * dx +
          ufl.inner(ufl.grad(v_test), P_total) * dx)

# ============================================================================
# SOLVER SETUP
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

# ============================================================================
# TIME STEPPING LOOP WITH TIMING
# ============================================================================
u_old.x.array[:] = 0.0
u_old.x.scatter_forward()
v_old.x.array[:] = 0.0
v_old.x.scatter_forward()

start_time = time.perf_counter()

for n in range(n_steps):
    # Release force at step 1000
    if n == 1000:
        f_ext_vector.set(0.0)
        f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                  mode=PETSc.ScatterMode.FORWARD)

    problem.solve()
    converged = problem.solver.getConvergedReason()
    assert converged > 0, f"Newton solver did not converge at step {n} (reason {converged})."
    u.x.scatter_forward()

    v_new = (u.x.array - u_old.x.array) / dt_val

    u_old.x.array[:] = u.x.array[:]
    u_old.x.scatter_forward()
    v_old.x.array[:] = v_new[:]
    v_old.x.scatter_forward()

end_time = time.perf_counter()
elapsed_time = end_time - start_time

if rank == 0:
    print(f"Solver execution time (s): {elapsed_time:.6f}")
    print(f"Average time per step (ms): {(elapsed_time / n_steps) * 1000:.3f}")
    print(f"RTF: {elapsed_time / (n_steps * dt_val):.6f}")
