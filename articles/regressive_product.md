# The regressive product

This document sets out a broader perspective on the regressive product in geometric algebra and the role it can play across geometry, calculus and physics. The product itself is the standard regressive product of Grassmann and projective geometric algebra, where it is traditionally used to join points or intersect subspaces. Once elements carry intrinsic and extrinsic orientation, that same product also pairs complementary quantities into scalars, contracts fields, forms discrete gradients on simplices, and builds linear maps and their metric-free duals.

Like the outer product, the regressive product uses no metric. Where the outer product builds elements up in grade, the regressive product pairs elements against the space's unit volume, yielding invariant scalars at complementary grades, joining points and directions, and contracting fields partially even when basis directions square to zero. In models that give their quantities orientations and pair them at complementary grades, the regressive product becomes the operation used most often, as [an operation count](#appendix-operations-in-numga-examples) across [`numga` examples](../examples/README.md) shows.

## Complementary grades and the unit volume

A plane in space divides what lies in front of it from what lies behind it, so its sign chooses a side in the space around it. A weighted point holds its sign in its weight, within the point itself. A mirror that contains the point leaves it as it is, while swapping the sides of an oblique plane. Following [Why orientation](orientation.md) [[1]](#references), an element carries either an extrinsic orientation (`+`), oriented in the space around it, or an intrinsic orientation (`-`), oriented within itself.

In projective three-space, an extrinsic plane has grade one and an intrinsic point has grade three, one below the four-dimensional top space, so `plane ^ point` fills a four-volume. Both `plane ^ point` and the space's intrinsic unit volume `antiscalar` live in that one-dimensional top space, so the first is a scalar multiple of the second, and that scalar is their regressive product:

```python
offset = plane & point                   # [] Scalar+
plane ^ point == offset * antiscalar     # [] Antiscalar-
```

Both `plane ^ point` and the unit volume have the same intrinsic top-grade type. Under an orientation-aware reflection `mirror >> x`, both transform by the same rule and stay unchanged, so their scalar ratio `offset` stays unchanged in a mirror. Neither the outer product nor the metric contractions refer to the unit volume, and none of them pairs an extrinsic element with an intrinsic one into a mirror-even scalar.

Because the unit volume multiplies `offset` on the right rather than dividing `plane ^ point`, this relation needs no inverse of the unit volume and no lengths along the axes. In projective space (`x+ y+ z+ w0`), Galilean spacetime and thermodynamic state spaces, null basis vectors make the unit volume square to zero, yet `plane ^ point` is still a scalar multiple of it. Putting the extrinsic factor first gives complementary basis blades (metric-free right complements) a positive sign in even and odd dimensions alike, whereas swapping the order in even dimensions negates odd-grade pairings (`point & plane == -(plane & point)`).

A plane wave advances its phase from crest to crest, and the phase gained across a step counts how many wavefronts that step crosses. A model can represent the wavevector as the extrinsic hyperplane of its wavefronts and an atomic step as an intrinsic displacement between points, as in a [crystal diffraction](../examples/optics/crystal_diffraction/crystal_diffraction.ipynb) example. Their pairing `wavevector & displacement` then gives the phase directly.

## Joins of points and directions

While the outer product cuts space down by intersecting hyperplanes, the regressive product builds geometry up from points, as in projective geometric algebra [[2, §2.2]](#references), [[3, Ch. 3]](#references), [[4, §6]](#references). Two points join into the line from the first to the second, a third point joins with that line into the plane of their triangle, and in three-space a fourth point closes the tetrahedron into its signed volume:

```python
edge = a & b                             # [] Antibivector-
winding = a & b & c                      # [] Antitrivector-
volume = a & b & c & d                   # [] Scalar-
```

Each join in that chain stays intrinsic. The line is oriented along itself from one point to the next, the triangle's plane is oriented by the order of its corners round the triangle rather than by a positive side outside it, and in three-space `a & b & c & d` is an intrinsic scalar that changes sign in a mirror, as a [surface normals](../examples/orientation/normals/normals.ipynb) example shows.

Joins and meets work together whenever a spanned element meets a plane. The line joined from a light source to a scene point meets a ground plane at the shadow point on the ground, as in a [scenegraph](../examples/geometry/scenegraph/scenegraph.ipynb) example ([extensor gallery](../docs/extensors.md#12-a-scenegraph-a-robot-arm-and-a-camera-as-one-map)):

```python
ray = light & point                      # [] Antibivector-
shadow = ray ^ ground                    # [] Antivector-
```

Points of weight zero represent directions, and they join the same way: `point & direction` is the line through a point along a direction, and `direction & time` is the patch swept out along time. Incidence constructions such as `shadow = ray ^ ground` determine a point up to scale, while quantitative joins such as `edge = a & b` and `volume = a & b & c & d` scale with the weights of their points. Euclidean models fix unit weight with the ideal plane (`w & a == 1`) so `(a & b).norm()` gives the distance between the two points (while `(a & b).antinorm()` gives the line's moment about the origin), and elliptic and hyperbolic Cayley–Klein models normalize points with the metric [[5, §9.2.3]](#references).

## Partial pairing and the determinant

When the grades of two elements add up to more than the dimension of the space, the regressive product pairs part of one with the other and leaves what remains. Two planes meet in an extrinsic line `u ^ v`. Paired with one intrinsic point, that line tests each plane at the point in turn and leaves a single plane:

```python
pencil = (u ^ v) & p                     # [] Vector+
pencil == (u & p) * v - (v & p) * u      # [] Vector+
pencil & p == 0                          # [] Scalar+
```

Because `pencil` is a combination of the two planes, it passes through their meeting line `u ^ v`. Any point joined with itself is zero (`p & p == 0`), so `(u ^ v) & (p & p) == 0`, which means the point lies on `pencil` as well. On three wedged planes, the same graded expansion gives `(u ^ v ^ w) & p == (u & p) * (v ^ w) - (v & p) * (u ^ w) + (w & p) * (u ^ v)`.

Pairing `pencil` with a second point ties this partial contraction to the point join through associativity:

```python
line = p & q                                              # [] Antibivector-
((u ^ v) & p) & q == (u ^ v) & line                       # [] Scalar+
(u ^ v) & line == (u & p) * (v & q) - (v & p) * (u & q)   # [] Scalar+
```

By associativity, `p & q` is the intrinsic line whose pairing with any two wedged planes `u ^ v` equals the two-by-two determinant of their values at the two points, and repeating the step gives the `k`-by-`k` determinant for `k` planes and `k` points in any dimension. Scalar pairing, partial contraction and the point join are one associative operation, and reflections and rigid motions distribute over it (`versor >> (a & b) == (versor >> a) & (versor >> b)`).

## Simplex gradients

On a simplex, dividing boundary joins by the volume join gives a discrete derivative without a metric ([Discrete differential geometric algebra](ddga.md)). At a corner of a simplex, the simplex gradient is the hyperplane that pairs to one with that corner and to zero with the corners opposite. On a tetrahedron with unit-weight corners, dividing the opposite facet join `b & c & d` by the volume join `(b & c & d) & a` builds that corner gradient in closed form, as in a [gravitational waves](../examples/relativity/gravitational_waves/waves.ipynb) example ([gallery](https://numga.org/#gravitational-waves)):

```python
facet = b & c & d                        # [] Antitrivector-
gradient_a = facet / (facet & a)         # [] Vector+
(gradient_a ^ gradient_b) & (a & b) == 1 # [] Scalar+
```

Because numerator and denominator are both intrinsic, their quotient `gradient_a` is an extrinsic hyperplane, and its pairing with `b - a` along any edge of the tetrahedron gives the difference between the values at the edge's two ends. By the determinant identity above, wedging two corner gradients gives a bivector pairing to one with their edge join `a & b` and to zero with every other edge, extending the same construction directly to higher-degree simplex gradients.

## Linear maps and dual pullbacks

Naming the space of an argument by its type builds a linear map directly from a product ([Extensors through multiple lenses](multiple_lenses.md)). In projective three-space, pairing any point with a fixed plane reads a scalar, multiplying a fixed output point by that readout gives a rank-one dyad, and joining any direction or point with a fixed point `p` gives a linear join map. Reading a linear map backwards across the regressive pairing defines its adjugate, keeping the extrinsic factor first on both sides so that signs remain independent of dimension.

For a map preserving orientation, such as a pinhole camera projecting scene points onto an image plane, its adjugate pulls image lines back into planes through the pinhole without needing an inverse camera. Pulling an image conic polarity back through the camera and its adjugate forms a sight cone in space, as in a [multiview reconstruction](../examples/estimation/multiview/multiview_reconstruction.ipynb) example. For the partial-join map with a fixed intrinsic point, associativity and anticommutativity of the regressive product make its adjugate the partial contraction with that same point ([extensor gallery](../docs/extensors.md#1-multi-view-reconstruction-and-camera-alignment)):

```python
plane & camera(point) == camera.adjugate()(plane) & point   # [] Scalar+
cone = camera.adjugate()(conic(camera))                     # [] Vector+ <- Antivector-
(Antivector & p).adjugate() == -(Bivector & p)              # [] Vector+ <- Bivector+
```

For a duality map between complementary orientations, such as `cone` mapping intrinsic points to extrinsic planes or a mobility mapping extrinsic vectors to intrinsic antivectors, keeping the extrinsic factor first on both sides gives `cone(a) & b == cone.adjugate()(b) & a` and `u & mobility(v) == v & mobility.adjugate()(u)`. Symmetry without a metric then reads `map == map.adjugate()` in even and odd dimensions alike. Differentiating the extrinsic-first quadratic energy `(a & map(a)) / 2` (for extrinsic inputs) or `(map(a) & a) / 2` (for intrinsic inputs) gives `((map + map.adjugate()) / 2)(a)`, which reduces to `map(a)` when `map` equals its adjugate, and both pullback `T.adjugate()(map(T))` and pushforward `T(map(T.adjugate()))` preserve that symmetry across any linear map `T`.

## Mechanics and electrodynamics

A mechanical model can represent a body's turning rate as an extrinsic bivector at grade two, and a moving mass's angular momentum as the intrinsic join `point & velocity` two grades below the top space ([Complementarity](complementarity.md#appendix-representations)). Half the pairing of the rate with the momentum is then the kinetic energy, as in [tennis racket](../examples/mechanics/tennis_racket/tennis_racket.ipynb) and [energy in a mirror](../examples/orientation/mirror_energy/mirror_energy.ipynb) examples ([gallery](https://numga.org/#tennis-racket)):

```python
velocity = point.commutator(rate)        # [] Antivector-
momentum = point & velocity              # [] Antibivector-
energy = (rate & momentum) / 2           # [] Scalar+
```

In three dimensions (`x+ y+ z+`), the rate has grade two while the momentum has grade one, so their inner product `rate | momentum` is a vector rather than a scalar. Because grade two and grade `n - 2` always add to `n`, `(rate & momentum) / 2` has the same form in the plane, in space and in four dimensions, and stays positive in a mirror. While that pairing uses no metric, the metric and the normalization of the mass point enter in `point.commutator(rate)` when building the momentum, so the same rate bivector carries different unit-speed weights and kinetic energies in Euclidean, elliptic and hyperbolic spaces [[2, §6.1]](#references), [[5, §§9.2–9.3]](#references).

In spacetime (`x+ y+ z+ t-`), electrodynamics can model the electromagnetic field as an extrinsic bivector and the electric four-current as an intrinsic antivector. Joining a wire step with an observer's time direction forms the worldsheet patch `direction & time`, and pairing the field with that patch reads the voltage per unit length along the wire. By the partial-join adjugate identity above, pulling the field back across the map that sweeps displacements with the four-current gives the Lorentz four-force `-(field & current)`, which has the same algebraic form as `(u ^ v) & p` ([dynamo](https://numga.org/#dynamo), [superconductor](https://numga.org/#superconductor)):

```python
voltage_slope = field & (direction & time)   # [] Scalar+
lorentz = -(field & current)                 # [] Vector+
lorentz & current == 0                       # [] Scalar+
lagrangian = (field & medium(field)) / 2     # [] Scalar+
```

Paired with a spatial velocity, the Lorentz four-force gives the mechanical power done on the charges; paired with `-time`, it gives the electrical power transferred to matter, which becomes Joule heating in a resistive conductor. Because `current & current == 0`, pairing the force with the four-current gives `-(field & (current & current)) == 0`, the same identity `pencil & p == 0` that placed the point on `(u ^ v) & p`. For a moving charge, that identity equates the electrical power delivered by the field with the mechanical power done along its motion before any spacetime metric or medium enters. The metric and material enter separately in the constitutive map `excitation = medium(field)` and its quadratic Lagrangian `(field & medium(field)) / 2`, as in a [constitutive media](../examples/electromagnetism/constitutive/constitutive.ipynb) example. The same partial-join adjugate identity carries directly into coupled transport when thermodynamic directions are adjoined to spacetime ([Geometry and quantities](thermodynamics.md)).

Across these models, a mirror-even scalar reading pairs an extrinsic factor first with an intrinsic one, either directly or through a duality map between complementary orientations:

```python
scalar = extrinsic & intrinsic                # [] Scalar+
scalar = extrinsic & duality(extrinsic)       # [] Scalar+
scalar = duality(intrinsic) & intrinsic       # [] Scalar+
```

[Complementarity](complementarity.md) develops those linear and quadratic forms across mechanics, elasticity, optics, electrodynamics, estimation, quantum measurement and thermodynamics.

## Conclusion

At complementary grades, the regressive product compares `a ^ b` with the space's intrinsic unit volume. Because both share the same intrinsic top-grade transformation under an orientation-aware reflection and the unit volume is never divided by, `a & b` gives a mirror-even scalar without a metric even when basis directions square to zero. By associativity, the same operation joins points into lines, surfaces, volumes and simplex gradients, and contracts wedged planes partially against points.

With elements typed by orientation, that single operation serves geometry and physics alike. The plane `(u ^ v) & p` through a line and a point has the same algebraic form as the Lorentz force `-(field & current)`, both expressing the adjugate of the partial join with a fixed intrinsic element, and the point lying on that plane is the same identity `p & p == 0` that ties electrical power to mechanical power on a current. Simplex gradients, rank-one maps and extrinsic-first adjugates follow from the regressive product in the same way, leaving metrics and material constants inside the duality maps where a model needs them.

## Related work

Grassmann's exterior algebra introduced both the progressive wedge product and the regressive product, and Grassmann–Cayley and projective geometric algebras developed the regressive product for joining points and intersecting subspaces. On degenerate metrics, alternative definitions including pseudoscalar dualization $I^{-1}(IY \wedge IX)$, the shuffle product and Poincaré duality are compared in [[4, §6]](#references).

In exterior calculus, the graded interior contraction $i_v(\omega \wedge \eta) = i_v\omega \wedge \eta + (-1)^p \omega \wedge i_v\eta$, the evaluation bracket $\langle \cdot, \cdot \rangle$ and the dual multivector join $\vee$ are defined on separate vector and form spaces alongside untwisted and twisted differential forms for outer and inner orientation [[6, §§1–2]](#references). On simplicial complexes, finite element exterior calculus [[7, §§1.3, 5]](#references) and discrete exterior calculus [[8, §6]](#references) likewise separate discrete coboundary forms from dual-cell Hodge stars. Within a single oriented geometric algebra, the regressive product carries the evaluation bracket at complementary grades, the interior contraction when an extrinsic element is paired partially with an intrinsic one, and the point join on intrinsic elements.

In projective rigid-body mechanics, a null basis vector makes the inner product vanish on translations. Euclidean dual-Grassmann treatments formulate kinetic energy through a dual pairing between velocity axes and momentum spears, $E = -S(\Omega \wedge \Pi)$ [[2, §6]](#references), and Cayley–Klein generalizations across Euclidean, elliptic and hyperbolic spaces write kinetic energy $E = -\Omega \vee \Pi$ and power $\dot{E} = -\Omega \vee \Delta$ through the metric-free regressive pairing [[5, Ch. 9]](#references). Single-algebra plane-based treatments write point–plane contact ($A_i \vee f < 0$), kinetic energy ($T = \frac{1}{2} B_b \vee I_b[B_b]$) and power ($\Pi = B \vee F$) in one unoriented algebra, while converting to outer-product or reciprocal-pseudoscalar forms when differentiating Lagrangians [[9, §§2.6–2.9]](#references).

Outside projective mechanics, physics in geometric algebra has mostly worked in nondegenerate Euclidean and Minkowski metrics, writing the Lorentz force density as the metric inner product $F \cdot J$ [[10, §VII.C]](#references). Premetric electrodynamics in exterior calculus instead writes the covectorial Lorentz force density from the untwisted 2-form $F$ and twisted 3-form $J$ using an auxiliary frame $e_\alpha$ as $f_\alpha = (e_\alpha \rfloor F) \wedge J$ [[11, §3]](#references).

## Companion documents

* [Why orientation](orientation.md): why geometric constructions and positive physical energies require intrinsic and extrinsic orientation.
* [Extensors through multiple lenses](multiple_lenses.md): how multilinear maps and rank-one dyads arise from algebraic products and transform under motions.
* [Complementarity](complementarity.md): how `extrinsic & intrinsic`, `extrinsic & duality(extrinsic)` and `duality(intrinsic) & intrinsic` unify dual pairs across geometry, mechanics, estimation and physics.
* [Discrete differential geometric algebra](ddga.md): calculus and physics on simplicial topologies built from the simplex gradient.
* [Geometry and quantities](thermodynamics.md): equilibrium and transport thermodynamics in algebras extended with null directions for extensive quantities.

## Appendix: operations in `numga` examples

Across 202 [`numga` examples](../examples/README.md), each row below counts uses of one geometric algebra operation and the number of examples in which it appears. Scalar multiplication, sparse topology operators and rank-one dyads are excluded from the bare geometric product:

| Operation | Uses | Examples |
| --- | ---: | ---: |
| Regressive product | 2249 | 180 |
| Versor sandwich | 877 | 143 |
| Bare geometric product | 544 | 109 |
| Outer product | 328 | 98 |
| Commutator and anticommutator | 207 | 79 |
| Inner product | 199 | 68 |
| Scalar product | 63 | 29 |
| Metric dot product | 31 | 14 |

## References

1. [*Projective Duality Encodes Complementary Orientations in Geometric Algebras*](https://pure.uva.nl/ws/files/189200508/Projective_duality_encodes_complementary_orientations_in_geometric_algebras.pdf) (Leo Dorst, *Mathematical Methods in the Applied Sciences*, 2024).
2. [*On the Homogeneous Model of Euclidean Geometry*](https://arxiv.org/pdf/1101.4542) (Charles Gunn, in *Guide to Geometric Algebra in Practice*, Leo Dorst and Joan Lasenby, eds., Springer, 2011; extended preprint of AGACSE 2010).
3. [*A Guided Tour to the Plane-Based Geometric Algebra PGA*](https://bivector.net/PGA4CS.pdf) (Leo Dorst and Steven De Keninck, 2022).
4. [*A Bit Better: Variants of Duality in Geometric Algebras with Degenerate Metrics*](https://arxiv.org/pdf/2206.02459) (Charles Gunn, 2022).
5. [*Geometry, Kinematics, and Rigid Body Mechanics in Cayley–Klein Geometries*](https://depositonce.tu-berlin.de/bitstreams/5a02e6fc-d9c3-4e6f-a5ab-f4a970e81209/download) (Charles Gunn, Doctoral Dissertation, Technische Universität Berlin, 2011; DOI `10.14279/depositonce-3058`).
6. [*Applied Differential Geometry (A compendium)*](https://people.math.ethz.ch/~hiptmair/Courses/CEM/BOS02c.pdf) (Alain Bossavit, 2002).
7. [*Finite element exterior calculus: from Hodge theory to numerical stability*](https://arxiv.org/pdf/0906.4325) (Douglas N. Arnold, Richard S. Falk and Ragnar Winther, *Bulletin of the American Mathematical Society*, 2010).
8. [*Discrete Exterior Calculus*](https://arxiv.org/pdf/math/0508341) (Mathieu Desbrun, Anil N. Hirani, Melvin Leok and Jerrold E. Marsden, 2005).
9. [*May the Forque Be with You: Dynamics in PGA*](https://bivector.net/PGAdyn.pdf) (Leo Dorst and Steven De Keninck, 2022).
10. [*Spacetime Physics with Geometric Algebra*](https://davidhestenes.net/geocalc/pdf/SpacetimePhysics.pdf) (David Hestenes, *American Journal of Physics*, 2003).
11. [*Recent developments in premetric classical electrodynamics*](https://arxiv.org/pdf/physics/0610221) (Friedrich W. Hehl, Yakov Itin and Yuri N. Obukhov, 2006).
