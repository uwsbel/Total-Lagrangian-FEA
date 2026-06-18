# TL-FEA

**GPU-accelerated Total-Lagrangian finite element framework for flexible multibody dynamics.**

This code accompanies the following publications:

- **Part I — Formulation:** Zhenhao Zhou, Ganesh Arivoli, Dan Negrut. *A Total Lagrangian Finite Element Framework for Multibody Dynamics: Part I — Formulation.* [arXiv:2602.17002](https://arxiv.org/abs/2602.17002)
- **Part II — GPU Implementation:** Zhenhao Zhou, Ruochun Zhang, Ganesh Arivoli, Dan Negrut. *A Total Lagrangian Finite Element Framework for Multibody Dynamics: Part II — GPU Implementation and Numerical Experiments.* [arXiv:2604.10357](https://arxiv.org/abs/2604.10357)

---

## Capabilities

**Element types**
- `FEAT10` — 10-node quadratic tetrahedra (T10), 5-point Keast quadrature
- `ANCF3243` — ANCF beam elements, 3×2×2 Gauss–Legendre quadrature
- `ANCF3443` — ANCF shell elements, 4×4×3 Gauss–Legendre quadrature

**Material models**
- Saint-Venant Kirchhoff (SVK)
- Mooney-Rivlin hyperelastic
- Kelvin-Voigt viscous damping (layered on any material)

**Solvers**
- `Newton` — second-order implicit Newton method; sparse Hessian factorized with NVIDIA cuDSS
- `AdamW` — first-order adaptive gradient method
- `Nesterov` — accelerated first-order method with momentum restart

All solvers operate within an augmented Lagrangian outer loop for constraint handling.

**Engineering joints** (bilateral constraints)
- Revolute, spherical, welded (fixed), cylindrical

**Contact / collision**
- [DEM-Engine](https://github.com/projectchrono/DEM-Engine) mesh-mesh contact — triangle-soup collision with bin-based spatial partitioning, produces per-node contact forces

**Multi-element problems**
- Mix element types and constitutive models across bodies in a single simulation (e.g., SVK solid + Mooney-Rivlin foam with shared contact)

---

## Build

### Prerequisites

| Dependency | Notes |
|---|---|
| CUDA Toolkit | Tested on CUDA 12.x |
| Bazel 9 | Build system |
| Eigen 3.4 | Fetched automatically via Bzlmod |
| cuDSS | Required for Newton solver; must be available in the system CUDA library path |
| cuBLAS | Ships with CUDA Toolkit |
| DEM-Engine | Git submodule — initialize before building |

### Setup

```bash
# Clone with submodules (DEM-Engine)
git clone --recurse-submodules https://github.com/uwsbel/Total-Lagrangian-FEA.git
cd Total-Lagrangian-FEA
```

### Compile and run a demo

```bash
# Build all targets (default CUDA archs: sm_75, sm_86)
bazel build //...

# Target a specific GPU architecture (e.g., RTX 50xx Blackwell)
bazel build //... --config=sm120

# Run the bunny mesh deformation demo (Newton solver, FEAT10 elements)
bazel run //lib_bin/mesh_deform:test_feat10_bunny_newton

# Run the double-pendulum revolute-joint demo
bazel run //lib_bin/engineering_joint:test_feat10_double_pendulum_revolute

# Run unit tests
bazel test //lib_utest/...
```

---

## Publications

If you use TL-FEA in your work, please cite:

```bibtex
@article{zhou2026tlfea1,
  title   = {A Total Lagrangian Finite Element Framework for Multibody Dynamics: Part {I} -- Formulation},
  author  = {Zhou, Zhenhao and Arivoli, Ganesh and Negrut, Dan},
  journal = {arXiv preprint arXiv:2602.17002},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.17002}
}

@article{zhou2026tlfea2,
  title   = {A Total Lagrangian Finite Element Framework for Multibody Dynamics: Part {II} -- {GPU} Implementation and Numerical Experiments},
  author  = {Zhou, Zhenhao and Zhang, Ruochun and Arivoli, Ganesh and Negrut, Dan},
  journal = {arXiv preprint arXiv:2604.10357},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.10357}
}
```
