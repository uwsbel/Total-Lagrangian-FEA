"""
Timing analysis script for nonlinear 3D beam dynamic analysis using Backward Euler.
Minimal version with only solver execution and timing.
"""
import os
import time
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, assemble_residual
from petsc4py import PETSc
from tetgen_mesh_loader import load_tetgen_mesh_from_files

rank = MPI.COMM_WORLD.rank

if rank == 0:
    print(f"Running with {MPI.COMM_WORLD.size} MPI ranks")

# ============================================================================
# GEOMETRY AND MESH SETUP
# ============================================================================
# Resolution selection: 0, 2, 4, 8, 16
RES = 4

# Construct mesh file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
# Project root is two levels up from test-scripts/, so go up three from here.
project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
mesh_dir = os.path.join(project_root, "data", "meshes", "T10", "resolution")

node_file = os.path.join(mesh_dir, f"beam_3x2x1_res{RES}.1.node")
ele_file = os.path.join(mesh_dir, f"beam_3x2x1_res{RES}.1.ele")

# Load TetGen mesh
domain, _ = load_tetgen_mesh_from_files(node_file, ele_file)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))

# Beam dimensions (for boundary conditions)
# L = 3.0; W = 2.0; H = 1.0;
L = 3.0   # Length (x)

# ============================================================================
# BOUNDARY CONDITIONS - Fix x=0 face
# ============================================================================
def fixed_boundary(x):
    return np.isclose(x[0], 0.0, atol=1e-6)

boundary_dofs = fem.locate_dofs_geometrical(V, fixed_boundary)
u_zero = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
bc_fixed = fem.dirichletbc(u_zero, boundary_dofs, V)

# ============================================================================
# APPLIED LOADS - Distribute 5000 N at x=3 face in +x direction
# ============================================================================
# Find local DOFs at x=L boundary (only owned, not ghosts)
dof_coords = V.tabulate_dof_coordinates()
dofmap = V.dofmap
num_owned_dofs = dofmap.index_map.size_local

force_dofs = []
for i, coord in enumerate(dof_coords):
    if i < num_owned_dofs and abs(coord[0] - L) < 1e-6:
        force_dofs.append(i)

# Compute GLOBAL total number of force nodes using MPI reduction
local_num_force_nodes = len(force_dofs)
global_num_force_nodes = domain.comm.allreduce(local_num_force_nodes, op=MPI.SUM)

# Calculate force per node based on GLOBAL count
total_force = 5000.0
force_per_node = total_force / global_num_force_nodes if global_num_force_nodes > 0 else 0.0

# Create a global PETSc vector for the external force
f_temp = fem.Function(V)
f_temp.x.array[:] = 0.0

block_size = dofmap.index_map_bs

# Apply force to local DOFs
for node_idx in force_dofs:
    # Set X-component (index 0 in the block) for +x direction
    f_temp.x.array[node_idx * block_size + 0] = force_per_node

# Move data to the PETSc vector
f_ext_vector = f_temp.x.petsc_vec.copy()
f_ext_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

if rank == 0:
    print(f"Load applied at x=3:")
    print(f"  Total force: {total_force} N (+x direction)")

# ============================================================================
# MATERIAL MODEL AND KINEMATICS (SVK or Mooney-Rivlin)
# ============================================================================
# Base material properties
E_val = 7.0e8
nu_val = 0.33
rho = fem.Constant(domain, 2700.0)

E = default_scalar_type(E_val)
nu = default_scalar_type(nu_val)

# Select material model: "SVK" or "MOONEY_RIVLIN"
MATERIAL_MODEL = "SVK"

v = ufl.TestFunction(V)
u = fem.Function(V)
u_old = fem.Function(V)
v_old = fem.Function(V)

# Kinematics
d = len(u)
I = ufl.Identity(d)
F = ufl.variable(I + ufl.grad(u))

if MATERIAL_MODEL == "SVK":
    # St. Venant-Kirchhoff model
    mu_svk = fem.Constant(domain, E / (2 * (1 + nu)))
    lmbda_svk = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

    C = F.T * F
    trFtF = ufl.tr(C)
    FFt = F * F.T
    FFtF = FFt * F

    lambda_factor = lmbda_svk * (0.5 * trFtF - 1.5)
    P = lambda_factor * F + mu_svk * (FFtF - F)
else:   
    # Compressible Mooney-Rivlin model
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    K_val = E_val / (3.0 * (1.0 - 2.0 * nu_val))

    c1_val = 0.30 * mu_val
    c2_val = 0.20 * mu_val
    kappa_val = 1.5 * K_val

    C1 = fem.Constant(domain, default_scalar_type(c1_val))
    C2 = fem.Constant(domain, default_scalar_type(c2_val))
    kappa = fem.Constant(domain, default_scalar_type(kappa_val))

    C = F.T * F
    I1 = ufl.tr(C)
    C2_tens = C * C
    trC2 = ufl.tr(C2_tens)
    I2 = 0.5 * (I1**2 - trC2)

    J = ufl.det(F)
    I1_bar = J**(-2.0 / 3.0) * I1
    I2_bar = J**(-4.0 / 3.0) * I2

    # Strain energy density and first Piola-Kirchhoff stress
    psi = C1 * (I1_bar - 3.0) + C2 * (I2_bar - 3.0) + 0.5 * kappa * (J - 1.0) ** 2
    P = ufl.diff(psi, F)

# ============================================================================
# TIME INTEGRATION SETUP (Backward Euler method)
# ============================================================================
dt = 1e-3
n_steps = 50
t_final = n_steps * dt

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
        "snes_atol": 1e-4,
        "snes_rtol": 1e-4,
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
u_old.x.scatter_forward()  # Sync ghost zones for parallel correctness

v_old.x.array[:] = 0.0
v_old.x.scatter_forward()  # Sync ghost zones for parallel correctness

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
    u_old.x.scatter_forward()  # Sync ghost zones for parallel correctness
    
    v_old.x.array[:] = v_new[:]
    v_old.x.scatter_forward()  # Sync ghost zones for parallel correctness

# End timing
end_time = time.perf_counter()
elapsed_time = end_time - start_time

if rank == 0:
    print(f"Solver execution time (s): {elapsed_time:.6f}")
    print(f"Average time per step (ms): {(elapsed_time / n_steps) * 1000:.3f}")

