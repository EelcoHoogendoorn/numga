---
title: 'numga: Geometric algebra and typed extensors in NumPy, JAX, PyTorch and WebGPU'
tags:
  - Python
  - geometric algebra
  - Clifford algebra
  - extensors
  - multilinear algebra
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

`numga` is an open-source Python library for geometric algebra and extensors, designed from the ground up around NumPy-style array libraries (NumPy [@Harris:2020], JAX [@Bradbury:2018] and PyTorch [@Ansel:2024]) with shader compilation to WebGPU. Geometric algebra places elements of every grade, such as points, planes, rotations and spacetime fields, in one graded algebra whose products work across dimensions and metric signatures [@Hestenes:1984; @DorstFontijneMann:2007; @Gunn:2019]. Extensors are the multilinear maps between multivector subspaces [@Hestenes:1984], where they play across every grade the role that matrices and tensors play on vectors.

`numga` maps each algebraic subspace directly onto a typed array axis, so linear and multilinear operators are values alongside the geometric elements they relate. A body's inertia, a crystal's stiffness or a pinhole camera moves with its geometry, composes with other maps, takes duals, and enters linear solves and eigenvalue problems directly in the algebra. Vectorized array batches on NumPy, JAX and PyTorch and compiled WebGPU shaders share the same typed representation.

# Statement of need

Geometric algebra unifies individual elements and their motions across grades [@DorstFontijneMann:2007]. In projective geometric algebra, one motor combines rotation and translation, one bivector holds angular and linear motion rates, and one forque combines torque and force [@Gunn:2011; @Gunn:2019; @DorstDeKeninck:2022]. Stepping rigid-body dynamics forward, however, requires a body's inertia, a linear map from motion rates to momenta [@Gunn:2011; @DorstDeKeninck:2022]. In a library that represents multivectors alone, such maps remain opaque functions or revert to hand-built coordinate matrices and index tables whenever an operator is composed, moved in space, solved or decomposed into eigenmodes.

That need extends across physics, optics and estimation wherever models use linear and multilinear maps between multivector subspaces of any grade. Optical elements map rays to rays and move with the bodies that carry them, while pose alignment maximizes a quadratic form over the rotor subalgebra through its leading eigenvector. Boosted media map spacetime bivector fields to excitations, and higher-arity piezoelectric or curvature operators take three or four multivector arguments and specialize by partial application as arguments are supplied.

Projective geometric algebra also shows why maps and pairings cannot rely on a nondegenerate metric. Because its null direction makes the metric vanish on translations and forces, metric duals and inner-product transposes break down. The regressive product (`&`) pairs complementary subspaces against the space's unit volume without a metric, so it evaluates power and kinetic energy directly and defines a metric-free adjugate map where no metric adjoint exists. A pinhole camera has no inverse, for example, yet its adjugate pulls an image quadric back into a world sight cone without a hand-derived Jacobian.

Keeping physical energies positive under reflection then requires tracking orientation on every subspace [@Dorst:2024]. The grade-involuted sandwich reflection commutes with the outer product while reversing the space's unit volume. Applying it uniformly to both factors of `rate & momentum`, with or without an overall minus sign, therefore negates their regressive pairing in every dimension and turns a mirrored spinning body's kinetic energy negative. Planes and motion rates are oriented in the space around them (extrinsic), whereas mass points, momenta and forques are oriented within each element itself (intrinsic). Reflecting the two kinds with opposite signs keeps mirrored kinetic energies positive and scalar observables mirror-invariant. `numga` is written for researchers and engineers in physics, mechanics, graphics and vision who want to express and compute with such models directly in the algebra.

# State of the field

Existing geometric algebra packages in Python are organized around multivector arithmetic rather than typed multidimensional array axes. `galgebra` [@galgebra:2020] provides symbolic geometric algebra and calculus on SymPy. `clifford` [@clifford:2021] accelerates dense `2^n` multivector arrays over NumPy with Numba and includes matrix-backed unary linear transformations between blade spaces. `kingdon` [@Roelfs:2025] generates sparse blade-level Python expressions dynamically over NumPy, PyTorch and SymPy with interactive `ganja.js` notebook views [@DeKeninck:2020] and stores active blades by bitmask. Extending any of these architectures to arbitrary-arity maps, oriented subspaces and native batch solvers would require replacing its core value representation, because multivector-first layouts do not map arbitrary oriented subspaces onto backend array axes. `numga` builds its type system from the ground up around NumPy-style multidimensional arrays and `einsum`-contracted multiplication tables. It treats elements and maps of any arity uniformly, tracks intrinsic and extrinsic orientation, exposes linear solvers and generalized eigensolvers across NumPy, JAX and PyTorch, and compiles static shaders to WebGPU.

Outside geometric algebra, orthonormal Euclidean coordinates hide the distinction between a space and its dual, while numerical array libraries operate on untyped coordinate axes. Finite-element frameworks provide typed low-rank tensors tied to variational forms, and labelled-axis libraries such as `einops` name array dimensions without recording what space an axis belongs to or which spaces can pair. Typing array axes by subspace and orientation places vector and dual-vector tensors inside the same representation as bivector rates, motors and null subspaces.

# Software design

Every value in `numga` is an `Extensor` that pairs a `GAType` with a backend array of shape `[batch..., Output, Input_1, ..., Input_n]`. Elements of the algebra have arity zero, linear operators have arity one, and multilinear maps carry one trailing axis per argument. Each `GAType` axis names an ordered subspace of basis blades (a single grade, an even rotor subalgebra or a general subspace) together with an intrinsic (`-`) or extrinsic (`+`) orientation. Because algebraic arity maps directly onto array axes, batch broadcasting, reductions and linear algebra carry over from NumPy, JAX and PyTorch. Lowercase names denote values and capitalized `GAType` objects denote identity maps on their subspaces, so `v ^ w` is a bivector, `v ^ Vector` is a linear map from vectors to bivectors, and supplying part of a multilinear map's arguments returns the partially applied map on the remaining subspaces.

In 3D projective geometric algebra (`x+ y+ z+ w0`), rigid motion rates are extrinsic bivectors (`Bivector+`), while forques and momenta are intrinsic antibivectors (`Antibivector-`) [@Gunn:2011; @DorstDeKeninck:2022; @Dorst:2024]. A rigid body's coupled translational and rotational inertia is a single linear map `Antibivector- <- Bivector+`, built from its weighted mass points (`Antivector-`), moved in space by a `motor`, and solved from rest directly in the algebra:

```python
inertia = (masses * points).regressive(
    points.commutator(Bivector)
).sum(axis=0)                                # [] Antibivector- <- Bivector+
moved = motor >> inertia(motor << Bivector)  # [] Antibivector- <- Bivector+
acceleration = moved.solve(forque)           # [] Bivector+
kinetic_energy = (rate & moved(rate)) / 2    # [] Scalar+
```

Reflecting the body with `mirror >> x` transforms the extrinsic `rate` and the intrinsic momentum `moved(rate)` with opposite reflection signs so `kinetic_energy` stays positive (`Scalar+`), whereas an orientation-blind sandwich yields a negative energy and mismatched orientations are rejected statically. Every linear map `f` carries a metric-free `.adjugate()` satisfying `u & f(p) == f.adjugate()(u) & p` across dual orientations, even when `f` is rectangular or the algebra's metric is degenerate; when the metric on the input subspace is nondegenerate, `f` also carries a metric `.adjoint()`. Symmetric map pairs such as an elastic stiffness and an inertia enter generalized eigenvalue solves directly on their subspaces.

Algebraic products between subspaces are represented as exact integer multiplication tables (`-1`, `0`, `+1`) over their active basis blades. Compound expressions over `GAType` signatures compose those tables ahead of time through integer `einsum` contractions, enforce orientation rules, and narrow output grades on versor sandwiches such as `motor >> Point` before any numerical array arrives. Concrete data then evaluates through backend `einsum` and `matmul` contractions across whole batches on NumPy, JAX (`jit`, `vmap`, `scan`, `grad`) and PyTorch, while exact pairings invert in integer arithmetic to compile `.adjugate()` as a signed permutation with no floating-point solve.

Where states are coupled across sites rather than independent across a batch, `.field()` extends the same typed operators to sparse block linear and eigenvalue solvers via SciPy [@Virtanen:2020] and JAX. For interactive simulation and rendering, `numga.wgsl` translates typed Python extensor functions into static `WGSL` compute and fragment shaders that run via `wgpu` or export as standalone browser pages.

# Research impact statement

The `numga` repository includes over 180 tested example packages (202 executable Jupyter notebooks), from rolling contact, camera alignment and active liquid crystals to dendritic solidification, magnetic dynamos and black-hole optics (\autoref{fig:overview}). Tutorials, reference guides and a 36-example visual gallery document the library, while eight companion articles in the repository use `numga` to investigate extensor formulations of orientation, the regressive product, discrete operators and coupled thermodynamics. Standalone WebGPU applications compiled with `numga.wgsl`, including interactive geodesic and polarization ray tracing around a Kerr black hole and inside a gravitational wave, run directly in modern browsers without a Python backend.

![On the left, each pinhole camera's adjugate pulls pixel uncertainty back into translucent sight cones that intersect at orange landmark splats above the camera pose ellipses. Second, an active liquid crystal evolves a traceless alignment map on a planar topology, tinting regions of weaker alignment around copper +1/2 and dark-blue -1/2 defects. Third, a WebGPU spacetime shader traces light rays from an accretion disc and starfield around a spinning Kerr black hole. On the right, a seed of ice grows into a six-armed dendrite in undercooled water as latent heat glows into the liquid around the tips, modelled in an algebra that extends space with null directions for ice and heat.\label{fig:overview}](joss_overview.png)

# AI usage disclosure

Generative AI tools (Anthropic Claude and Google Gemini) were used under the author's direction for code refactoring, test authoring, documentation editing and drafting portions of this paper. The author specified all software architecture, type rules and mathematical formulations, verified code and physical invariants against automated tests, and reviewed and edited all prose.

# References
