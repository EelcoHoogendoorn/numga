# Documentation

This directory holds the tutorials, user guides and reference pages for numga. For the conceptual motivation behind orientation, the regressive product, complementarity, discrete differential geometric algebra and coupled thermodynamics, see the [articles](../articles/README.md); for runnable models across geometry and physics, see the [examples](../examples/README.md).

## Overview and tutorials

* [An introduction to extensors](extensors.md): an illustrated tour of extensors across geometry, estimation, mechanics, optics, electromagnetism, relativity, quantum physics and thermodynamics.
* [Geometric algebra tutorial](tutorial/geometric-algebra/00-introduction.md): a step-by-step primer for readers new to geometric algebra.
* [Extensors tutorial](tutorial/extensors/00-introduction.md): a step-by-step primer building multilinear maps, adjoints, adjugates and solvers from algebraic products.

## Guides and reference

* [Extensor syntax and methods](extensor_syntax.md): algebras, subspaces, `GAType`, partial application, composition, linear algebra solvers and backend contexts.
* [Orientation in numga](orientation.md): how intrinsic (`-`) and extrinsic (`+`) orientations are typed, propagated and statically checked.
* [Physical units in numga](units.md): opting in to unit checking, units on the blades of values and the entries of maps, and the rule each operation follows.
* [Fields and field maps](fields.md): site axes on extensors, pointwise and coupled maps, and `Sparse` operators.
* [Simplicial topologies](topology.md): combinatorial simplicial complexes and their incidence, boundary, coboundary and dual operators.
* [WebGPU shaders and rendering](rendering.md): compiling `Extensor` kernels to WGSL compute shaders and rendering 2D and 3D geometry on the GPU.
* [Conventions](conventions.md): the identities defining products, duals, norms, maps, forms and solvers.
* [Internals](internals.md): type layout, symbolic compilation, method dispatch, backends and custom extensor representations.
