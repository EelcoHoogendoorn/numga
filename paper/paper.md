---
title: 'numga: Geometric algebra and typed extensors in NumPy, JAX, PyTorch and WebGPU'
tags:
  - Python
  - geometric algebra
  - Clifford algebra
  - extensors
  - multilinear algebra
  - discrete differential geometry
  - physics simulation
authors:
  - name: Eelco Hoogendoorn
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 15 November 2026
bibliography: paper.bib
---

# Summary

`numga` is an open-source Python package for numerical computation with geometric algebra and extensors on NumPy [@Harris:2020], JAX [@Bradbury:2018], PyTorch [@Ansel:2024] and WebGPU. Geometric algebra places vectors, oriented areas, volumes, rotations, rigid motions and spacetime fields inside one graded algebra whose products extend across dimensions and metric signatures [@Hestenes:1984; @Gunn:2019]. Extensors generalize matrices and tensors to linear and multilinear maps between subspaces of a geometric algebra [@Hestenes:1984].

In `numga`, algebraic elements and maps share one typed array representation. Each subspace type stands for the identity map on that subspace, so the geometric, outer, regressive, inner and sandwich products that combine elements also build linear operators, bilinear energy forms and higher-arity constitutive maps. Those maps compose, invert, solve linear systems and compute eigendecompositions while keeping subspace layouts and orientations attached to every input and output axis. Together with dense batch and field arrays, `numga` supplies simplicial topologies, sparse block operators for coupled discrete models, and a compiler that translates typed Python functions on extensors into WebGPU Shading Language (`WGSL`).

# Statement of need

Geometric algebra streamlines rigid kinematics, ray optics, electromagnetism and relativity at the level of individual elements, where a motor moves a line or a bivector holds an electromagnetic field. Most models in continuum physics, estimation and mechanics, however, turn on maps between subspaces of different grades and orientations, including rigid-body inertia, anisotropic stiffness, optical refraction, camera projection, estimation covariance, spacetime curvature and coupled irreversible transport. When a library represents multivectors alone, users have to unpack coordinate arrays, build matrices by hand, introduce index conventions such as Voigt tables, or write untyped tensor contractions whenever a linear solve, an eigenvalue problem or an assembled operator on a simplicial topology enters the problem.

Degenerate directions add a second requirement. Projective geometric algebra adjoins a basis vector squaring to zero so that translations and rotations combine into a single motor [@Gunn:2019], and adjoining null directions for extensive thermodynamic quantities places coupled potentials, drives and fluxes in one algebra with space and time. Because the metric inner product vanishes along null directions, duals and transposes defined through a metric do not exist on those subspaces. Tracking intrinsic and extrinsic orientation on every subspace [@Dorst:2024] provides a metric-free complementary pairing via the regressive product (`&`) and a metric-free dual map (`.adjugate()`) beside the metric inner product and `.adjoint()`. Attaching both subspace layout and orientation to every axis of a map catches incompatible pairings at construction time and supports linear algebra uniformly across Euclidean, Minkowski, projective and extended signatures.

`numga` serves researchers, engineers and educators in computational physics, mechanics, vision, computer graphics and applied mathematics who want executable models written directly in geometric algebra from compact interactive notebooks up to compiled GPU simulations.

# State of the field

Existing geometric algebra packages in Python focus on multivector arithmetic rather than typed multilinear maps. `galgebra` [@galgebra:2020] implements symbolic geometric algebra and calculus on SymPy for algebraic manipulation rather than array-oriented numerical workloads. `clifford` [@clifford:2021] provides numerical multivector arrays over NumPy accelerated with Numba, storing full `2^n` multivector arrays without narrowed subspace types, JAX or PyTorch execution, or map types. `kingdon` [@Roelfs:2025] generates sparse blade-level Python code dynamically over NumPy, PyTorch and SymPy inputs and integrates interactive notebook views via `ganja.js` [@DeKeninck:2020]. Its multivectors map active basis-blade keys to coefficient arrays indexed by an integer bitmask, accelerating element-wise expressions without representing multilinear maps as values, solving linear systems or generalized eigenproblems across subspaces, assembling sparse block maps over simplicial topologies, or targeting WebGPU shaders.

In broader scientific software, finite-element libraries provide typed low-rank vector tensors specialized to continuum variational forms, whereas array libraries such as `einops` and `xarray` label array axes without encoding algebraic subspaces, orientations or geometric products. Supporting extensors requires map signatures, orientation propagation, exact integer operator tables and site-coupled field axes at the core of the array representation rather than as a wrapper around a multivector class, motivating `numga` as an independent library.

# Software design

Every value in `numga` is an `Extensor` holding a backend `Context`, an immutable `GAType` and a dense coefficient array. An element of the algebra has arity zero (`[batch..., O]`), a unary linear map has arity one (`Output <- Input`, `[batch..., O, I]`), and an `n`-argument map has layout `[batch..., O, I_1, ..., I_n]`. On each axis, `GAType` records an ordered `SubSpace` of basis blades, an `Orientation` (`+` extrinsic or `-` intrinsic), optional site counts for coupled fields, and certified algebraic traits such as `Versor` or `ReverseProductOne` used to dispatch operations such as `.inverse()` and `.exp()` at compile time.

Because any `GAType` acts as the identity map on its subspace, linear and multilinear maps are constructed directly from algebraic products. In 3D projective geometric algebra, building the coupled translational and rotational inertia of weighted mass points, turning that map with a rigid motor, and solving for the motion rate produced by an applied wrench takes four lines:

```python
inertia = (masses & masses.commutator(Bivector)).sum(axis=0)  # [] Antibivector- <- Bivector+
moved_inertia = placement >> inertia(placement << Bivector)   # [] Antibivector- <- Bivector+
rate = moved_inertia.solve(forque)                            # [] Bivector+
kinetic_energy = (forque & rate) / 2                          # [] Scalar+
```

Expressions containing a `GAType` evaluate first in an `ExactContext` whose `SymbolicKernel` caches exact `int8` structure constants (`-1`, `0`, `+1`) for each product and type signature. Binding concrete values compiles and caches a `BindingPlan` that verifies orientation rules, aligns subspaces without silent lossy projection, narrows output grades when repeated versor arguments form a sandwich (`motor >> Point`), and lowers evaluation to a single `matmul` or `einsum` contraction on the backend array. Exact pairings invert in integer arithmetic, compiling `.adjugate()` to a signed permutation without invoking a floating-point solver.

Numerical execution runs through `NumpyContext`, `JaxContext` (supporting `jax.jit`, `jax.vmap`, `jax.lax.scan` and automatic differentiation) and `TorchContext` (CPU and CUDA tensors with autograd). Calling `.field()` marks a trailing batch axis as coupled sites inside `GAType`, separating a single state distributed over vertices, edges or simplices of a `Topology` from independent batch runs. Dense field maps and sparse block operators (`Sparse` and `SparseExtensor`, backed by SciPy [@Virtanen:2020] and native JAX sparse and block solvers) compose, take adjoints and adjugates, and solve coupled linear and eigenvalue systems directly on typed fields. For rendering and browser deployment, `numga.wgsl` compiles Python functions over extensors into static `WGSL` structs and shaders run through `wgpu` in `numga.render` or exported as standalone HTML applications.

# Research impact statement

The `numga` repository includes 184 worked example packages (202 executable Jupyter notebooks in total) covered by automated tests across rigid and deformable mechanics, thin-walled beams, discrete surface geometry, vortex and magnetohydrodynamic flows, electrostatics, piezoelectricity, crystal optics, quantum spin and lattice models, multi-view reconstruction, general relativity and coupled irreversible thermodynamics. Seven companion articles and an illustrated 36-example gallery document both the software idioms and two mathematical formulations developed with the package: discrete differential geometric algebra on simplicial topologies and coupled equilibrium and transport thermodynamics in geometric algebras extended with null extensive-quantity directions. Standalone WebGPU pages compiled via `numga.wgsl`, including interactive geodesic and polarization ray tracing around a Kerr black hole and inside a gravitational wave, run directly in modern browsers without a Python backend.

# AI usage disclosure

Generative AI tools (Anthropic Claude and Google Gemini) were used under the author's direction for code refactoring, test authoring, documentation editing and drafting portions of this paper. All software architecture, type rules, mathematical formulations, physical tests and final prose were specified, reviewed and verified against automated checks by the author.

# References
