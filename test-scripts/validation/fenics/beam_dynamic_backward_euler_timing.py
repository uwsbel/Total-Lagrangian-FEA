"""
Timing analysis script for nonlinear 3D beam dynamic analysis using Backward Euler.
Minimal version with only solver execution and timing.
"""
import os
import time
import numpy as np
import ufl

from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, assemble_residual
from petsc4py import PETSc
from tetgen_mesh_loader import load_tetgen_mesh_from_files

# ============================================================================
# GEOMETRY AND MESH SETUP
# ============================================================================
# Resolution selection: 0 (RES_0), 2 (RES_2), or 4 (RES_4)
RES = 2
print(f"Running with resolution: {RES}")

# Construct mesh file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir))
mesh_dir = os.path.join(project_root, "data", "meshes", "T10", "resolution")

node_file = os.path.join(mesh_dir, f"beam_3x2x1_res{RES}.1.node")
ele_file = os.path.join(mesh_dir, f"beam_3x2x1_res{RES}.1.ele")

# Load TetGen mesh
domain, x_tetgen = load_tetgen_mesh_from_files(node_file, ele_file)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))

# Beam dimensions (for boundary conditions)
L = 3.0   # Length (x)
W = 2.0   # Width (y)
H = 1.0   # Height (z)
tol = 1e-6

# ============================================================================
# BOUNDARY CONDITIONS - Fix x=0 face
# ============================================================================
def fixed_boundary(x):
    return np.isclose(x[0], 0.0, atol=1e-6)

boundary_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)
u_zero = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
bc_fixed = fem.dirichletbc(u_zero, boundary_dofs, V)

# ============================================================================
# APPLIED LOADS - Distribute 250000N at x=3 face in -z direction
# ============================================================================
dof_coords = V.tabulate_dof_coordinates()
force_dofs = []
for i, coord in enumerate(dof_coords):
    if abs(coord[0] - L) < tol:
        force_dofs.append(i)

total_force = 250000.0
num_force_nodes = len(force_dofs)
force_per_node = total_force / num_force_nodes if num_force_nodes > 0 else 0.0

# Create a global PETSc vector for the external force
f_temp = fem.Function(V)
f_temp.x.array[:] = 0.0

dofmap = V.dofmap
block_size = dofmap.index_map_bs

for node_idx in force_dofs:
    # Set Z-component (index 2 in the block) - negative for -z direction
    f_temp.x.array[node_idx * block_size + 2] = -force_per_node

# Move data to the PETSc vector
f_ext_vector = f_temp.x.petsc_vec.copy()
f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# ============================================================================
# MATERIAL MODEL AND KINEMATICS
# ============================================================================
E = default_scalar_type(7.0e8)
nu = default_scalar_type(0.33)
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
rho = fem.Constant(domain, 2700.0)

v = ufl.TestFunction(V)
u = fem.Function(V)
u_old = fem.Function(V)
v_old = fem.Function(V)

d = len(u)
I = ufl.Identity(d)
F = I + ufl.grad(u)
C = F.T * F
trFtF = ufl.tr(C)
FFt = F * F.T
FFtF = FFt * F

lambda_factor = lmbda * (0.5 * trFtF - 1.5)
P = lambda_factor * F + mu * (FFtF - F)

# ============================================================================
# TIME INTEGRATION SETUP (Backward Euler method)
# ============================================================================
dt = 1e-3
print(f"Time step: {dt}")
n_steps = 50
print(f"Number of steps: {n_steps}")
t_final = n_steps * dt
print(f"Total simulation time: {t_final}")

# ============================================================================
# VARIATIONAL FORM (Backward Euler)
# ============================================================================
metadata = {"quadrature_degree": 5}
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

v_current = (u - u_old) / dt
a_current = (v_current - v_old) / dt

# Backward Euler variational form:
# External force is handled separately via PointLoadProblem
F_form = (rho * ufl.inner(a_current, v) * dx +
          ufl.inner(ufl.grad(v), P) * dx)

# ============================================================================
# SOLVER SETUP
# ============================================================================
class PointLoadProblem(NonlinearProblem):
    def __init__(self, F, u, f_ext_vector, bcs=None, **kwargs):
        super().__init__(F, u, bcs=bcs, **kwargs)
        self.f_ext_vector = f_ext_vector
        self._bcs = bcs if bcs is not None else []
        
        # Set up custom residual function callback for SNES
        def residual_callback(snes, x, b, ctx=None):
            # First, assemble the standard residual (Mass + Stiffness) into b
            assemble_residual(self.u, self.F, self.J, self._bcs, snes, x, b)
            
            # Then subtract the external force vector (b = F_int - F_ext)
            b.axpy(-1.0, self.f_ext_vector)
        
        # Set the custom residual function
        self.solver.setFunction(residual_callback, self.b)

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

# ============================================================================
# TIME STEPPING LOOP WITH TIMING
# ============================================================================
u_old.x.array[:] = 0.0
v_old.x.array[:] = 0.0

# Start timing
start_time = time.perf_counter()

# Time stepping loop
for n in range(n_steps):
    problem.solve()
    converged = problem.solver.getConvergedReason()
    assert converged > 0, f"Newton solver did not converge (reason {converged})."
    u.x.scatter_forward()
    
    v_new = (u.x.array - u_old.x.array) / dt
    
    u_old.x.array[:] = u.x.array[:]
    v_old.x.array[:] = v_new[:]

# End timing
end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Solver execution time: {elapsed_time:.6f} seconds ({elapsed_time * 1000:.3f} ms)")
print(f"Average time per step: {elapsed_time / n_steps:.6f} seconds ({(elapsed_time / n_steps) * 1000:.3f} ms)")

