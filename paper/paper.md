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
    orcid: 0009-0007-9884-7208
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 16 November 2026
bibliography: paper.bib
---

# Summary

`numga` is an open-source Python library for numerical computing with geometric algebra and extensors on NumPy [@Harris:2020], JAX [@Bradbury:2018], PyTorch [@Ansel:2024] and WebGPU. Geometric algebra places elements of every grade, such as points, planes, rotations and spacetime fields, in one graded algebra whose products work across dimensions and metric signatures [@Hestenes:1984; @Gunn:2019]. Extensors are the multilinear maps between multivector spaces [@Hestenes:1984], generalizing matrices and tensors from vectors to every grade of the algebra.

In `numga`, geometric relationships are values alongside the objects they relate. A body's inertia, a crystal's stiffness or a pinhole camera is an extensor that moves with its geometry, composes with other maps, and enters linear solves and eigenvalue problems without unpacking coordinates into matrices. Dense batch arrays, coupled fields on simplicial topologies, sparse block operators and compiled WebGPU shaders share the same typed representation.

# Statement of need

Geometric algebra is commonly used for individual elements and their motions, where a rotor turns a direction or a bivector holds an electromagnetic field. Most models in mechanics, estimation and field physics, however, turn on maps between spaces: a body's inertia takes its rate of motion to its momentum, an anisotropic stiffness takes strain to stress, and a camera projects scene points to image points while its dual pulls pixel uncertainty back into sight cones. When a library represents multivectors alone, those relationships fall back to hand-built coordinate matrices, Voigt index tables or untyped array contractions whenever a linear solve, an eigenvalue problem or an assembled operator on a topology enters.

Degenerate directions sharpen that need. Projective geometric algebra extends space with a null direction so translations and rotations combine into a single motor [@Gunn:2019], and adding null directions for extensive thermodynamic quantities places coupled potentials, drives and fluxes in one algebra with space and time. Because the metric vanishes along null directions, metric duals and inner-product transposes break down on those subspaces. Tracking intrinsic and extrinsic orientation on every subspace [@Dorst:2024] supplies a metric-free pairing through the regressive product (`&`) and a metric-free dual map (`.adjugate()`) alongside the metric inner product and `.adjoint()`. A camera has no inverse, for instance, yet its adjugate pulls an image quadric back into a world sight cone with no Jacobian derived by hand. `numga` is written for researchers and engineers in physics, mechanics, graphics and vision who want to express and compute with such models directly in the algebra.

# State of the field

Existing geometric algebra packages in Python represent multivector arithmetic rather than maps between multivector spaces. `galgebra` [@galgebra:2020] provides symbolic geometric algebra and calculus on SymPy. `clifford` [@clifford:2021] accelerates dense `2^n` multivector arrays over NumPy with Numba. `kingdon` [@Roelfs:2025] generates sparse blade-level Python code dynamically over NumPy, PyTorch and SymPy with interactive `ganja.js` notebook views [@DeKeninck:2020], storing active blades by bitmask. `numga` extends this ecosystem by representing linear and multilinear maps as values, solving linear systems and generalized eigenproblems across oriented subspaces, assembling sparse operators over simplicial topologies, and compiling extensor functions to WebGPU.

Outside geometric algebra, orthonormal Euclidean coordinates hide the distinction between a space and its dual, while numerical array stacks erased types from contractions for speed. Finite-element frameworks provide typed low-rank tensors tied to variational forms, and labelled-axis libraries such as `einops` name array dimensions without recording what space an axis belongs to or which spaces can pair. Typing extensors by subspace and orientation places ordinary vector and covector tensors inside the same representation as bivector twists, rotors and null subspaces.

# Software design

Every value in `numga` is an `Extensor` pairing a `GAType` with a backend array of shape `[batch..., Output, Input_1, ..., Input_n]`, where elements of the algebra have no input axes. On each axis, `GAType` records an ordered subspace of basis blades and an intrinsic (`-`) or extrinsic (`+`) orientation. Lowercase names denote values and capitalized `GAType` objects denote spaces in the same algebra, so `v ^ w` is one bivector and `v ^ Vector` is the linear map from vectors to bivectors.

In 3D projective geometric algebra, a rigid body's translational and rotational inertia is a single duality map from motion rates (`Bivector+`) to momenta and forques (`Antibivector-`), built from its point masses and moved in space like any other geometric object:

```python
inertia = ((masses * points) & points.commutator(Bivector)).sum(axis=0)  # [] Antibivector- <- Bivector+
moved_inertia = placement >> inertia(placement << Bivector)              # [] Antibivector- <- Bivector+
rate = moved_inertia.solve(forque)                                       # [] Bivector+
kinetic_energy = (rate & forque) / 2                                     # [] Scalar+
```

Typing individual small tensors in Python is slow, so `numga` resolves types once per pair of `GAType` signatures and executes one compiled contraction across an entire batch or field. Algebraic expressions build exact integer tables (`-1`, `0`, `+1`) before any numerical array arrives, check orientation rules, narrow output grades on versor sandwiches such as `placement >> Point`, and lower evaluation to a single `matmul` or `einsum` call on NumPy, JAX (`jit`, `vmap`, `scan`, `grad`) or PyTorch. Exact pairings invert in integer arithmetic, compiling `.adjugate()` to a signed permutation with no floating-point solve.

Marking an axis with `.field()` distinguishes coupled sites on a simplicial `Topology` from independent batch runs. Sparse block operators backed by SciPy [@Virtanen:2020] and JAX compose, take adjoints and adjugates, and solve linear and generalized eigenvalue systems directly on typed fields. For rendering and browser deployment, `numga.wgsl` translates Python functions on extensors into static `WGSL` shaders run via `wgpu` or exported as standalone HTML pages.

# Research impact statement

The `numga` repository includes over 180 tested example packages (202 executable Jupyter notebooks), from rolling contact, camera alignment and active liquid crystals to dendritic solidification, magnetic dynamos and black-hole optics (\autoref{fig:overview}). Seven companion articles and a 36-example visual gallery use the library to develop two mathematical formulations alongside established models: discrete differential geometric algebra from the simplex gradient on a topology, and coupled equilibrium and transport thermodynamics in algebras that extend space and time with null directions for extensive quantities. Standalone WebGPU applications compiled with `numga.wgsl`, including interactive geodesic and polarization ray tracing around a Kerr black hole and inside a gravitational wave, run directly in modern browsers without a Python backend.

![From left to right, each pinhole camera's adjugate pulls pixel uncertainty back into translucent sight cones that intersect at orange landmark splats above the camera pose ellipses. Second, an active liquid crystal evolves a traceless alignment map on a planar topology, tinting regions of weaker alignment around copper +1/2 and dark-blue -1/2 defects. Third, a WebGPU spacetime shader traces light rays from an accretion disc and starfield around a spinning Kerr black hole. On the right, a seed of ice grows into a six-armed dendrite in undercooled water as latent heat glows into the liquid around the tips, modelled in an algebra that extends space with null directions for ice and heat.\label{fig:overview}](joss_overview.png)

# AI usage disclosure

Generative AI tools (Anthropic Claude and Google Gemini) were used under the author's direction for code refactoring, test authoring, documentation editing and drafting portions of this paper. All software architecture, type rules, mathematical formulations, physical tests and final prose were specified, reviewed and verified against automated checks by the author.

# References
