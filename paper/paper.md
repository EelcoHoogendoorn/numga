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

<!--
Editorial brief (owner feedback, Nov 15–16 2026)

- Core architectural thesis: `numga` is designed from the ground up around NumPy-style multidimensional array libraries (NumPy, JAX, PyTorch) with the explicit goal of making extensors (linear and multilinear maps between subspaces) natural to express and compute with. Mapping each algebraic subspace (`GAType`) directly onto an array axis lets elements and maps share one representation, builds compound expressions as `einsum`-contracted integer multiplication tables (`-1`, `0`, `+1`) evaluated via a single `einsum` or `matmul` contraction, and exposes native array solvers and eigensolvers directly on typed subspaces.
- Motivating and illustrative backbone, plus breadth of multilinear maps: Dorst and De Keninck's PGA rigid-body mechanics (`@DorstDeKeninck:2022`) motivated `numga` directly and anchors both `Statement of need` and `Software design`. Keep the distinction sharp: multivector PGA already handles motors, motion rates and `forques` as elements, whereas the **inertia** requires a map on multivectors; from that concrete anchor, explicitly show the broader scope opened up by multilinear maps across arbitrary grades, subalgebras (such as rotors) and arities (optical ray maps moved by motors, quadratic alignment forms on rotors, boosted bivector media, higher-arity piezoelectric and curvature tensors with partial application, generalized eigenproblems). Treat `forque` as an established term without clumsy parenthetical glosses.
- Causal order for `&`/`.adjugate()` and orientation (never backwards, and never blamed on a 3D grade coincidence):
  - Degenerate metrics (PGA's null direction) and singular maps (cameras) motivate metric-free pairing (`&`) and metric-free dual maps (`.adjugate()`) alongside `.adjoint()`.
  - Keeping physical energies **positive** under reflection motivates intrinsic (`-`) and extrinsic (`+`) orientation (`@Dorst:2024`) in **every** dimension and at **every** grade: because the grade-involuted sandwich reflection `mirror * x.involute() * mirror.inverse()` commutes with `^` while reversing the unit volume, applying it uniformly to both factors of `rate & momentum` (with or without an overall minus sign) negates their regressive pairing in every dimension and turns a mirrored spinning body's kinetic energy negative. Reflecting extrinsic (`+`) and intrinsic (`-`) subspaces with opposite signs keeps mirrored kinetic energies positive (`Scalar+`) and observables mirror-invariant. Keep explicit illustrations of `Antibivector-`, `Bivector+`, `forque`, `acceleration = moved_inertia.solve(forque)` and `kinetic_energy = (rate & moved_inertia(rate)) / 2`.
- Conceptual explanation takes precedence over implementation details, though both belong:
  - Lead with what extensors, subspace types (`GAType` as identity map), `&`/`.adjugate()` and orientation mean and solve conceptually.
  - Keep compilation and execution mechanics (`einsum`-contracted integer multiplication tables, versor grade narrowing, single `einsum`/`matmul` evaluation, signed-permutation `.adjugate()`, `numga.wgsl` static shader compilation) in a clear supporting paragraph in `Software design`.
- Keep orthogonal research programmes out of the core motivation and claims (light hat-tips only):
  - Do not litigate the finer points of the broader complementarity ansatz, and do not pitch thermodynamic null-quantity directions in `Statement of need`.
  - In `Research impact statement`, do not claim that the companion articles "develop two mathematical formulations"; give a brief hat-tip to the tutorials, gallery and eight companion articles instead.
  - Keep the four-panel figure `joss_overview.png` and its caption hat-tips (camera adjugate cones, planar-topology liquid crystal, WebGPU Kerr ray tracer, dendritic ice growth with null directions for ice and heat).
  - Treat sparse operators over simplicial complexes (`Topology`) as downstream/tangential rather than a headline pillar: omit from `tags`, `Summary`, `Statement of need` and `State of the field`, keeping only a brief mention of `.field()` sparse solvers in `Software design` and the panel-2 caption hat-tip.
- Follow `WRITING.md` strictly throughout (plain indicative sentences, no Claudisms, no noun inventories, no internal class-name lists).
-->

# Summary

`numga` is an open-source Python library for geometric algebra and extensors, designed from the ground up around NumPy-style array libraries to run on NumPy [@Harris:2020], JAX [@Bradbury:2018], PyTorch [@Ansel:2024] and WebGPU. Geometric algebra places elements of every grade, such as points, planes, rotations and spacetime fields, in one graded algebra whose products work across dimensions and metric signatures [@Hestenes:1984; @Gunn:2019]. Extensors are the multilinear maps between multivector spaces [@Hestenes:1984], generalizing matrices and tensors from vectors to every grade of the algebra.

Representing each algebraic subspace directly as a typed array axis makes linear and multilinear maps first-class values alongside the geometric objects they relate. A body's inertia, a crystal's stiffness or a pinhole camera moves with its geometry, composes with other maps, takes metric and metric-free duals, and enters linear solves and eigenvalue problems without unpacking coordinates into matrices. Vectorized array batches on NumPy, JAX and PyTorch and compiled WebGPU shaders share the same typed representation.

# Statement of need

Geometric algebra unifies individual elements and their motions across grades. In projective geometric algebra, one motor combines rotation and translation, one bivector holds angular and linear motion rates, and one forque combines torque and force [@Gunn:2019; @DorstDeKeninck:2022]. Stepping rigid-body dynamics forward, however, requires the body's inertia, a linear map from motion rates to momenta [@DorstDeKeninck:2022]. More broadly, models across physics, optics and estimation turn on linear and multilinear maps whose inputs and outputs are multivector subspaces of any grade: optical elements map rays to rays and move with the bodies that carry them, pose alignment maximizes a quadratic form directly over the space of rotors through its leading eigenvector, boosted media map spacetime bivector fields to excitations, and higher-arity piezoelectric or curvature tensors couple three or four multivector arguments and specialize by partial application as arguments are supplied. When a library represents multivectors alone, those maps remain opaque functions or fall back to hand-built coordinate matrices and index tables whenever an operator is composed, moved in space, inverted in a linear solve or decomposed into eigenmodes.

Projective geometric algebra also exposes why maps and pairings cannot rely on a nondegenerate metric alone. Because its null direction makes the metric vanish on translations and forces, metric duals and inner-product transposes break down. Pairing against the space's unit volume through the regressive product (`&`) evaluates power and kinetic energy without consulting a metric, and reading a map across `&` defines a metric-free dual map (`.adjugate()`) alongside the metric inner product and `.adjoint()`. A pinhole camera has no inverse, for instance, yet its adjugate pulls an image quadric back into a world sight cone with no Jacobian derived by hand.

Keeping physical energies positive under reflection then requires tracking orientation on every subspace [@Dorst:2024]. Because the grade-involuted sandwich reflection commutes with the outer product while reversing the space's unit volume, applying it uniformly to both factors of `rate & momentum`, with or without an overall minus sign, negates their regressive pairing in every dimension and turns a mirrored spinning body's kinetic energy negative. Distinguishing extrinsic orientation, a sense in the surrounding space carried by planes and motion rates, from intrinsic orientation, a sense within the element carried by mass points, momenta and forques, reflects the two kinds with opposite signs so mirrored kinetic energies stay positive and scalar observables remain mirror-invariant. `numga` is written for researchers and engineers in physics, mechanics, graphics and vision who want to express and compute with such models directly in the algebra.

# State of the field

Existing geometric algebra packages in Python are organized around multivector arithmetic rather than array-backed maps between subspaces. `galgebra` [@galgebra:2020] provides symbolic geometric algebra and calculus on SymPy. `clifford` [@clifford:2021] accelerates dense `2^n` multivector arrays over NumPy with Numba. `kingdon` [@Roelfs:2025] generates sparse blade-level Python expressions dynamically over NumPy, PyTorch and SymPy with interactive `ganja.js` notebook views [@DeKeninck:2020], storing active blades by bitmask. Because none of these representations maps arbitrary oriented subspaces onto native array axes, linear and multilinear operators cannot be composed, transposed or solved directly through the underlying array library. `numga` addresses this gap by building its type system around multidimensional arrays and `einsum`-contracted multiplication tables, supporting extensors, intrinsic and extrinsic orientations, linear solvers and generalized eigensolvers across NumPy, JAX and PyTorch, and static shader compilation to WebGPU.

Outside geometric algebra, orthonormal Euclidean coordinates hide the distinction between a space and its dual, while numerical array stacks erase types from contractions for speed. Finite-element frameworks provide typed low-rank tensors tied to variational forms, and labelled-axis libraries such as `einops` name array dimensions without recording what space an axis belongs to or which spaces can pair. Typing array axes by subspace and orientation places vector and dual-vector tensors inside the same representation as bivector rates, motors and null subspaces.

# Software design

Every value in `numga` is an `Extensor` pairing a `GAType` with a backend array of shape `[batch..., Output, Input_1, ..., Input_n]`, where elements of the algebra have arity zero, linear operators have arity one, and bilinear or higher-order forms carry one axis per argument. Each `GAType` axis names an ordered subspace of basis blades, whether a single grade, an even rotor subalgebra or a mixed subspace, together with an intrinsic (`-`) or extrinsic (`+`) orientation, aligning algebraic arity directly with array dimensions so batch broadcasting, reductions and linear algebra carry over from NumPy, JAX and PyTorch. Lowercase names denote values and capitalized `GAType` objects denote identity maps on their spaces, so `v ^ w` is one bivector, `v ^ Vector` is a linear map from vectors to bivectors, and supplying part of a multilinear map's arguments returns the partially applied map on the remaining spaces.

In 3D projective geometric algebra (`x+ y+ z+ w0`), rigid motion rates are extrinsic bivectors (`Bivector+`), while forques and momenta are intrinsic antibivectors (`Antibivector-`) [@DorstDeKeninck:2022; @Dorst:2024]. A rigid body's coupled translational and rotational inertia is a single linear map `Antibivector- <- Bivector+`, built from its weighted mass points (`Antivector-`), moved in space by a motor `placement`, and solved directly in the algebra:

```python
inertia = ((masses * points) & points.commutator(Bivector)).sum(axis=0)  # [] Antibivector- <- Bivector+
moved_inertia = placement >> inertia(placement << Bivector)              # [] Antibivector- <- Bivector+
acceleration = moved_inertia.solve(forque)                               # [] Bivector+
kinetic_energy = (rate & moved_inertia(rate)) / 2                        # [] Scalar+
```

Reflecting the body with `mirror >> x` flips the extrinsic `rate` and transforms the intrinsic momentum `moved_inertia(rate)` with the opposite sign so `kinetic_energy` stays positive (`Scalar+`), whereas an orientation-blind sandwich yields a negative energy and mismatched orientations are rejected statically. Any linear map `f` likewise carries both a metric `.adjoint()` and a metric-free `.adjugate()` satisfying `u & f(p) == f.adjugate()(u) & p`, swapping input and output subspaces across their dual orientations even when `f` is rectangular or the metric is degenerate, and symmetric map pairs such as a stiffness and an inertia enter generalized eigenvalue solves directly on their subspaces.

Algebraic products between subspaces are represented as exact integer multiplication tables (`-1`, `0`, `+1`) over their active basis blades. Compound expressions over `GAType` signatures compose those tables ahead of time via integer `einsum` contractions, checking orientation rules and narrowing output grades on versor sandwiches such as `placement >> Point` before any numerical array arrives. Evaluating the compiled expression on concrete data then reduces to a single `einsum` or `matmul` contraction across an entire batch on NumPy, JAX (`jit`, `vmap`, `scan`, `grad`) or PyTorch, while exact pairings invert in integer arithmetic to compile `.adjugate()` as a signed permutation with no floating-point solve. Where states are coupled across sites rather than independent across a batch, `.field()` extends the same typed operators to sparse block linear and eigenvalue solvers via SciPy [@Virtanen:2020] and JAX, and `numga.wgsl` translates Python extensor functions into static `WGSL` compute and rendering shaders run via `wgpu` or exported as standalone HTML pages.

# Research impact statement

The `numga` repository includes over 180 tested example packages (202 executable Jupyter notebooks), from rolling contact, camera alignment and active liquid crystals to dendritic solidification, magnetic dynamos and black-hole optics (\autoref{fig:overview}). Tutorials, reference guides, a 36-example visual gallery and eight companion articles accompany the codebase, covering extensor modelling, orientation and the regressive product alongside applications to discrete operators and coupled thermodynamics. Standalone WebGPU applications compiled with `numga.wgsl`, including interactive geodesic and polarization ray tracing around a Kerr black hole and inside a gravitational wave, run directly in modern browsers without a Python backend.

![From left to right, each pinhole camera's adjugate pulls pixel uncertainty back into translucent sight cones that intersect at orange landmark splats above the camera pose ellipses. Second, an active liquid crystal evolves a traceless alignment map on a planar topology, tinting regions of weaker alignment around copper +1/2 and dark-blue -1/2 defects. Third, a WebGPU spacetime shader traces light rays from an accretion disc and starfield around a spinning Kerr black hole. On the right, a seed of ice grows into a six-armed dendrite in undercooled water as latent heat glows into the liquid around the tips, modelled in an algebra that extends space with null directions for ice and heat.\label{fig:overview}](joss_overview.png)

# AI usage disclosure

Generative AI tools (Anthropic Claude and Google Gemini) were used under the author's direction for code refactoring, test authoring, documentation editing and drafting portions of this paper. All software architecture, type rules, mathematical formulations, physical tests and final prose were specified, reviewed and verified against automated checks by the author.

# References
