# Extensors, linear algebra, tensors and forms

Extensors, linear algebra, tensor algebra and the exterior algebra of differential forms describe overlapping territory, and the same questions come up whenever they meet: is an extensor just a tensor, is a map just a matrix, and where did the transpose go. This document answers them at the level of what each framework is, without derivations.

## What each one is

**Linear algebra** studies vectors and the linear maps between them. In a basis, a map is a matrix, and most of its operations, the transpose, the trace, the determinant, are defined on that matrix.

**Tensor algebra** extends linear algebra to multilinear maps: functions of several vectors and covectors, linear in each. The metric is one such tensor, a symmetric form, and it is applied by hand to turn vectors into covectors and back, by raising and lowering indices.

**Exterior algebra**, the algebra of differential forms, is graded: its elements have parts of every grade from zero to the dimension, multiplied by the wedge product, which needs no metric. Forms are written on the dual space, as functions of vectors that change sign when two arguments swap, and differential forms are fields of them. The algebra allows sums of different grades, though practice rarely uses them. It has no geometric product: the wedge only raises grade, so a vector has no inverse and there are no rotors.

**Extensors** are the multilinear maps of geometric algebra: functions whose axes take multivectors of any type, and whose outputs are multivectors. They are written with the products of the algebra, with an open axis where a factor is left out:

```python
a ^ Vector                         # Bivector <- Vector
Vector | Vector                    # Scalar <- (Vector, Vector), the metric
(Vector | Vector) + (Vector ^ Vector) - Vector * Vector   # zero: the geometric product, both axes open
```

## How they fit together

- A matrix is an extensor with one vector input and a vector output, `Vector <- Vector`. Linear algebra is the part of the extensors restricted to vectors and the inner product: inputs and outputs of grade one, and the scalar product as the only product that pairs them.
- A tensor on vectors and covectors is an extensor whose inputs take vectors, with a covector input written through one of the two pairings below. Conversely, every extensor is a tensor on the larger space of multivectors. In finite dimensions, the two frameworks have the same reach.
- The tensor product is the product of extensors with independent axes, built from the algebra's own products. A dyad `a * (b | Vector)` is the tensor product of `a` with the covector of `b`, its input paired through the inner product, or through the regressive product where no metric is wanted. Sums of such products give every tensor, as sums of dyads give every matrix.
- Exterior algebra is the outer half of the geometric product, contained in geometric algebra as plainly as the algebra of the scalar product is. A form of grade k is an extensor from k-vectors to scalars, written with the regressive product `&` where no metric is wanted. Tensor practice mostly emulates grades above one through constructions such as the Levi-Civita symbol, which writes a plane as its normal vector; the forms community keeps every grade and stops before the geometric product. What geometric algebra adds is that product of the two halves, and with it mixed grades that mean something: rotors, spinors, the inverse of a vector.

Extensors are therefore the tensor algebra of geometric algebra rather than of plain vectors. Linear algebra is the subset of extensors restricted to vectors and the inner product; extensors are linear algebra over every grade, with every product of the algebra as their vocabulary.

## What differs

The reach is the same; the differences are in where things live and what each makes easy.

**Where the metric lives.** Tensor algebra keeps the metric as a separate object and applies it by index placement. Geometric algebra puts it in the products. Extensors keep two pairings: the inner product `|`, which uses the metric and pairs a space with itself, and the regressive product `&`, which needs no metric and pairs a space with its complement. A covector axis can be written with either. Where the metric is degenerate, as in projective geometry, only the complement pairs every element, much as tensor calculus falls back on the volume form there.

**What an axis is.** An index is up or down. An axis has a type: a vector, a bivector, a rotor, a point, a plane. The type decides which operations mean something:

- There is no transpose, because a transpose identifies a space with its dual through the coefficients. In its place a map has an adjoint, which moves it across the inner product and needs the metric, and an adjugate, which moves it across the regressive product and needs none. In an orthonormal Euclidean basis, both have the coefficients of the transpose; in spacetime and in projective geometry they differ.
- A map has eigenvalues only where its output has the type of its input; a map between different spaces has singular values; a form has eigenvalues only against another form. A matrix does not record which of these applies, and a type does.
- A map and a form with the same coefficients are different objects, and their conversion is a solve against a pairing: a signed permutation of the coefficients, with nothing computed.

**Native ground.** Geometric algebra is native to antisymmetric objects, where index notation needs extra machinery: the geometric product of vectors, which matrix physics rebuilds as Pauli and Dirac matrices; areas and normals under a deformation, the outermorphism `F(a) ^ F(b)` in place of $\det(F)\, F^{-T}$; the electromagnetic field as one bivector; rigid motions as twists and forques with one commutator; curvature as a symmetric map on bivectors. Index notation encodes antisymmetric objects as vectors through the Levi-Civita symbol in three dimensions, and the encoding shows: the inertia tensor's subtracted trace, $r^2 \delta_{ij} - r_i r_j$, is the cost of writing a map on planes of rotation as a map on their normals, and has no counterpart in other dimensions. Index notation is native to symmetric objects. A symmetric product $a \odot b$ is not an element of the algebra; in extensors it is a form with open axes, `a * (b | Vector) + b * (a | Vector)`, and symmetry is a property of the coefficients rather than of the type. In practice the gap is small: forms built by pairing an expression with itself, such as second moments, covariances and the normal forms of least squares, are symmetric by construction.

## What linear algebra does not see

Linear algebra is the theory of a vector space: it sees the space, and not the algebra on it. It knows one bilinear structure, an inner product, or none. Everything else the geometry needs enters as a special construction, a fixed table of signed ones with its own formula and conventions, specific to a dimension, which someone has to know, write down and keep consistent. Each of them is a product of the algebra:

| In linear algebra | In extensors |
| --- | --- |
| the Levi-Civita symbol | `Vector ^ Vector` |
| the cross-product matrix of $a$ | `(a ^ Vector).dual()` |
| the determinant | the outermorphism on the pseudoscalar |
| the cofactor matrix, as in transforming normals | the outermorphism on bivectors |
| the swap matrix that pairs twists with wrenches | the regressive product `&` |
| the Pauli and Dirac matrices | multiplication with an open axis, on spinors |

At the level of coefficients nothing is hidden. Every operation on extensors is a contraction of coefficient arrays against such a table, which is a Euclidean dot product on coefficients, and the reach of the two frameworks is the same. The difference is where the tables come from. In linear algebra they are inputs, supplied by hand. In extensors they follow from the products, and the right contraction is the one that writing the geometric expression gives. The coefficients cannot tell which table a contraction used, as a matrix cannot tell whether it is a map or a form; the products and the types can.

## Linear algebra as a method

That a matrix is a map from vectors to vectors is the dry part. The interesting part is that the methods come along: Gauss-Newton, Schur complements, conjugate gradients, eigenproblems and least squares run on geometric types from start to finish, with nothing flattened into coordinates. From the extensor point of view the boundary between numerical linear algebra and geometry dissolves.

The reach is the same, and anything done this way can be done with matrices. What changes is which form is convenient, and with typed axes and no transpose, the form a problem wants is the one easiest to write. Several habitual forms of current practice, in applied fields and in physics, turn out to be the ones the types do not offer:

- Hybrid force and motion control took orthogonal complements of twists and wrenches under a dot product of six coordinates; Duffy showed in 1990 that this mixes units and depends on the frame. A twist and a forque are different types here, and the only pairing on offer between them is the meaningful one.
- Graphics transforms normals by the inverse transpose, which fails for singular transforms and flips under reflections; the adjugate, which treats a normal as the bivector it is, is the default here.
- Continuum mechanics converts between its stress measures by factors of $F$, $F^{-T}$ and $J$; with stress as a map from oriented areas to forces, they are compositions with the deformation's map on areas.
- A stiffness from rates to momenta is not symmetric in coordinates, and the habit is to symmetrize it or form normal equations, squaring the condition number; joined with an open twist, it is the symmetric form it wants to be.
- The standard rotation fit, Kabsch's SVD, needs a determinant correction because it can return a reflection; posed as an eigenproblem on rotors, the answer is a rotor by type.
- The constitutive relations of a moving medium, and an inertia between body and space frames, are maps moved between frames; each is a composition with the rotation or boost on either side, with the type saying which side is which.

The first-hand account is numga's own multiview example. It was first written with a coefficient transpose and the normal equations $J^T J$, as Gauss-Newton is taught. Removing the transpose from the library left no way to write that, and what remained was better: the cone value as the cost, its curvature a cone joined with the motion, no residual metric to choose, and a Schur complement solved as a form against a form. [Extensors and common practice](common_practice.md) works through each case with its references and code, including the encodings of tensor practice, such as Voigt notation and the inertia tensor, and separately lists the cases, such as composing boosts or blending rigid motions, that are benefits of geometric algebra itself rather than of extensors.

## Object or map

Whether a tensor is an object, defined by a universal property, or a multilinear map, is a question about which construction to call the definition. In finite dimensions the two are the same up to canonical isomorphism. An extensor is both at once: a value that is stored, added, batched and transformed, and a function that is called by filling its axes. A multivector is the case with no axes to fill.

## Closure

The deepest difference between the frameworks is not in any single operation but in closure: whether everything a calculation needs can be written inside the framework, or whether some of it has to be brought in from outside. Geometric algebra is closed in this sense for the objects of geometry and physics. Points, lines, planes, rotations, rigid motions, fields and spinors are all elements of one algebra, and every operation on them is one of its products. Extensors carry that closure over to the maps between those objects: an inertia, a stiffness, a material, a covariance or a camera is built from the same products and solved with the same operations as the geometry it acts on.

Each of the other frameworks keeps part of this and gives up the rest, and what it gives up comes back in as a construction supplied by hand. Linear algebra keeps vectors and the inner product, and brings in the antisymmetric side through the Levi-Civita symbol, cross-product matrices and determinant formulas. Exterior algebra keeps the outer product and every grade, and brings in the metric through the Hodge star; without the geometric product, it has no rotations among its elements. Tensor algebra keeps maps with many axes, each taking vectors, and brings in rotations as matrices and spinors through the Pauli and Dirac matrices. Each of these imports is a fixed table, specific to a dimension, that someone has to know and keep consistent, as the table in [what linear algebra does not see](#what-linear-algebra-does-not-see) lists.

The matrix is where all of these imports meet. Once one part of a calculation has been recast in matrices, it is easiest to recast all of it, and that is what practice does: to compute an inertia, a stiffness or a covariance numerically, the problem is first rewritten in matrix form and then assembled. Numerical software written in tensor notation exists, in finite element mechanics and in the tensor networks of many-body physics, but it works within the restriction to vectors, and its geometric constructions are still supplied by hand.

Closure is what lets one language carry all of numga's examples, from rigid bodies, cameras and estimation to materials, light, spacetime and spin, without leaving the algebra. Neither half achieves it alone. Geometric algebra without extensors closes the objects but not the maps, and a geometric algebra program falls back on a matrix at its first inertia or covariance. Maps built on the inner product alone, or on the outer product alone, lose part of the objects they act on. Together, the full geometric product and maps with open axes make a language in which nothing has to be brought in from outside. This is a claim about coherence, not about reach: every calculation here can also be done with matrices, but done that way it is a collection of separate constructions rather than one language.

## In numga

numga implements extensors as typed coefficient arrays with the products of geometric algebra as their operations. The [extensor overview](extensors.md) shows what this buys in the examples, the [extensor tutorial](tutorial/extensors/01-products-as-maps.md) builds the ideas from a single open axis, and the [geometric algebra tutorial](tutorial/geometric-algebra/01-the-scalar-product.md) builds the algebra from a single product.
