# The regressive product

This document sets out a broader perspective on the regressive product in geometric algebra and the role it can play across geometry, calculus and physics. The product itself is the standard regressive product of Grassmann and projective geometric algebra, where it is traditionally used to join points or intersect subspaces. Once elements carry [orientation](orientation.md), that same product also pairs complementary quantities into scalars, contracts fields, forms discrete gradients on simplices, and builds linear maps and their metric-free duals.

Like the outer product `^`, the regressive product `&` uses no metric. Where `^` builds elements up in grade, `&` pairs elements against the space's unit volume, yielding invariant scalars at complementary grades, joining points and directions, and contracting fields partially even when basis directions square to zero. In models that give their quantities orientations and pair them by [complementarity](complementarity.md), `&` becomes the product used most often, as [the count across the `numga` examples](#appendix-operations-across-the-numga-examples) shows.

## Complementary grades and the unit volume

A plane in space divides what lies in front of it from what lies behind it, so its sign chooses a side in the space around it. A weighted point holds its sign in its weight, within the point itself. A mirror that contains the point leaves it as it is, while swapping the sides of an oblique plane. The terms ***extrinsic orientation*** and ***intrinsic orientation*** here mean an orientation in the space around an element (`+`) and an orientation within the element itself (`-`), following [[1]](#references).

In projective three-space, an extrinsic plane has grade one and an intrinsic point has grade three, one below the four-dimensional top space, so `plane ^ point` fills a four-volume. Both `plane ^ point` and `antiscalar`, the space's intrinsic unit volume, live in that one-dimensional top space, so the first is a scalar multiple of the second, and that scalar is their regressive product:

```python
offset = plane & point                   # [] Scalar+
plane ^ point == offset * antiscalar     # [] Antiscalar-
```

Both `plane ^ point` and `antiscalar` have the same intrinsic top-grade type. Under an orientation-aware reflection `>>`, both transform by the same rule and stay unchanged, so their scalar ratio `offset` stays unchanged in a mirror. Neither the outer product nor the metric contractions refer to the unit volume, and none of them pairs an extrinsic element with an intrinsic one into a mirror-even scalar.

Because `antiscalar` multiplies `offset` on the right rather than dividing `plane ^ point`, this relation needs no inverse of `antiscalar` and no lengths along the axes. In projective space (`x+ y+ z+ w0`), Galilean spacetime and thermodynamic state spaces, null basis vectors make `antiscalar` square to zero, yet `plane ^ point` is still a multiple of `antiscalar`. Putting the extrinsic factor first gives matching basis pairs a positive sign in even and odd dimensions alike.

A plane wave advances its phase from crest to crest, and the phase gained across a step counts how many wavefronts that step crosses. A model can represent the wavevector as the extrinsic hyperplane of its wavefronts and an atomic step as an intrinsic displacement between points, as in the [crystal diffraction](../examples/optics/crystal_diffraction/crystal_diffraction.ipynb) example. Their pairing `wavevector & displacement` then gives the phase directly.

## Joins of points and directions

While the outer product `^` cuts space down by intersecting hyperplanes, the regressive product `&` builds geometry up from points, as in plane-based projective geometric algebra [[2, Ch. 3]](#references), [[3, §6]](#references). Two points join into the line from the first to the second, a third point joins with that line into the plane of their triangle, and in three-space a fourth point closes the tetrahedron into its signed volume:

```python
edge = a & b                             # [] Antibivector-
winding = a & b & c                      # [] Antitrivector-
volume = a & b & c & d                   # [] Scalar-
```

Each join in that chain stays intrinsic. The line is oriented along itself from one point to the next, the triangle's plane is oriented by the order of its corners round the triangle rather than by a positive side outside it, and in three-space `a & b & c & d` is an intrinsic scalar that changes sign in a mirror, as the [surface normals](../examples/orientation/normals/normals.ipynb) example shows.

Joins and meets work together whenever a spanned element meets a plane. The line joined from a light source to a scene point meets a ground plane at the shadow point on the ground, as in the [scenegraph](../examples/geometry/scenegraph/scenegraph.ipynb) example:

```python
ray = light & point                      # [] Antibivector-
shadow = ray ^ ground                    # [] Antivector-
```

Points of weight zero represent directions, and they join the same way: `point & direction` is the line through a point along a direction, and `direction & time` is the patch swept out along time.

## Partial pairing and the determinant

When the grades of two elements add up to more than the dimension of the space, `&` pairs part of one with the other and leaves what remains. Two planes meet in an extrinsic line `u ^ v`. Paired with one intrinsic point, that line tests each plane at the point in turn and leaves a single plane:

```python
pencil = (u ^ v) & p                     # [] Vector+
pencil == (u & p) * v - (v & p) * u      # [] Vector+
pencil & p == 0                          # [] Scalar+
```

Because `pencil` is a combination of the two planes, it passes through their meeting line `u ^ v`. Any point joined with itself is zero (`p & p == 0`), so `(u ^ v) & (p & p) == 0`, which means the point lies on `pencil` as well.

A second point ties this partial pairing to the point join of the previous section through associativity:

```python
line = p & q                                              # [] Antibivector-
((u ^ v) & p) & q == (u ^ v) & line                       # [] Scalar+
(u ^ v) & line == (u & p) * (v & q) - (v & p) * (u & q)   # [] Scalar+
```

By associativity, `p & q` is the intrinsic line whose pairing with any two wedged planes `u ^ v` equals the two-by-two determinant of their values at the two points, and repeating the step gives the `k`-by-`k` determinant for `k` planes and `k` points in any dimension. Scalar pairing, partial contraction and the point join are one associative operation, and reflections and rigid motions distribute over it (`versor >> (a & b) == (versor >> a) & (versor >> b)`).

## The simplex gradient

On a simplex, dividing boundary joins by the volume join gives a discrete derivative without a metric ([Discrete differential geometric algebra](ddga.md)). The term ***simplex gradient*** here means the hyperplane at a corner of a simplex that pairs to `1` with that corner and to `0` with the corners opposite. On a tetrahedron with unit-weight corners, dividing the opposite facet join `b & c & d` by the volume join `(b & c & d) & a` builds the simplex gradient in closed form, as in the [Whitney forms](../examples/surfaces/whitney_forms/whitney_forms.ipynb) example:

```python
facet = b & c & d                        # [] Antitrivector-
gradient_a = facet / (facet & a)         # [] Vector+
```

Because numerator and denominator are both intrinsic, their quotient `gradient_a` is an extrinsic hyperplane. Its pairing with the displacement `b - a` along any edge of the tetrahedron gives the difference between the values at the edge's two ends.

## Linear maps and the adjugate

Naming the space of an argument by its type builds a linear map directly from a product ([Extensors through multiple lenses](multiple_lenses.md)). For a fixed intrinsic patch, `Bivector & swept_plane` reads a scalar from any bivector, and multiplying an output element by that readout gives a rank-one map:

```python
swept_plane = direction & time           # [] Antibivector-
readout = Bivector & swept_plane         # [] Scalar+ <- Bivector+
dielectric = swept_plane * readout       # [] Antibivector- <- Bivector+
```

A crystal's dielectric response is a sum of such maps over its axes.

Reading a linear map backwards across `&` defines its adjugate `f.adjugate()`, the map for which applying `f` on the right gives the same scalar as applying `f.adjugate()` on the left:

```python
b & f(a) == f.adjugate()(b) & a          # [] Scalar+
cone = camera.adjugate()(conic(camera))  # [] Vector+ <- Antivector-
```

A pinhole `camera` projects three-space onto an image plane, collapsing depth, so it has no inverse. Its adjugate is still defined and maps image lines back to planes through the pinhole, so `camera.adjugate()(conic(camera))` pulls an image conic back into a sight cone in space, as in the [multiview reconstruction](../examples/estimation/multiview/multiview_reconstruction.ipynb) example. A map equal to its own adjugate (`material == material.adjugate()`) is symmetric across the pairing: differentiating `(a & map(a)) / 2` with respect to `a` gives `((map + map.adjugate()) / 2)(a)`, which reduces to `map(a)` when `map` equals its adjugate.

## Mechanics and electrodynamics

A mechanical model can represent a body's turning rate as an extrinsic bivector at grade two, and a moving mass's angular momentum as the intrinsic join `point & velocity` two grades below the top space ([Complementarity](complementarity.md#appendix-representations)). Half the pairing of the rate with the momentum is then the kinetic energy, as in the [tennis racket](../examples/mechanics/tennis_racket/tennis_racket.ipynb) and [energy in a mirror](../examples/orientation/mirror_energy/mirror_energy.ipynb) examples:

```python
velocity = point.commutator(rate)        # [] Antivector-
momentum = point & velocity              # [] Antibivector-
energy = (rate & momentum) / 2           # [] Scalar+
```

In three dimensions (`x+ y+ z+`), the rate has grade two while the momentum has grade one, so their inner product `rate | momentum` is a vector rather than a scalar. Because grade two and grade `n - 2` always add to `n`, `(rate & momentum) / 2` has the same form in the plane, in space and in four dimensions, and stays positive in a mirror.

In spacetime (`x+ y+ z+ t-`), electrodynamics can model the electromagnetic field as an extrinsic bivector and the electric four-current as an intrinsic antivector. Joining a wire step with an observer's time vector forms the worldsheet patch `direction & time`, and pairing the field with that patch reads the voltage per unit length along the wire. Pairing the field partially with the four-current has the form `(u ^ v) & p` from earlier and gives the Lorentz four-force:

```python
voltage_slope = field & (direction & time)   # [] Scalar+
lorentz = -(field & current)                 # [] Vector+
lorentz & current == 0                       # [] Scalar+
```

Paired with a spatial displacement, `lorentz` gives the mechanical work done on the charges; paired with `-time`, it gives the electrical power transferred to matter, which becomes Joule heating in a resistive conductor. Because `current & current == 0`, pairing `lorentz` with `current` gives `-(field & (current & current)) == 0`, the same identity `pencil & p == 0` that placed the point on `(u ^ v) & p`. For a moving charge, that identity equates the electrical power delivered by the field with the spatial work done along its motion before any spacetime metric or medium enters; those enter separately in `excitation = medium(field)`, as in the [constitutive media](../examples/electromagnetism/constitutive/constitutive.ipynb) example.

Across these models, a mirror-even scalar reading takes one of two forms:

```python
linear_reading = extrinsic & intrinsic       # [] Scalar+
quadratic_reading = a & duality(a)           # [] Scalar+
```

The first pairs an extrinsic measurement or drive with an intrinsic state or flux. The term ***duality map*** here means a linear map from one orientation to its complement, such as an inertia or a medium response, so the second form `a & duality(a)` again pairs an extrinsic element with an intrinsic one. [Complementarity](complementarity.md) develops those two forms across mechanics, elasticity, optics, electrodynamics, estimation, quantum measurement and thermodynamics.

## Conclusion

At complementary grades, the regressive product compares `a ^ b` with the space's intrinsic unit volume `antiscalar`. Because both share the same intrinsic top-grade transformation under `>>` and `antiscalar` is never divided by, `a & b` gives a mirror-even scalar without a metric even when basis directions square to zero. By associativity, the same operation joins points into lines, surfaces, volumes and simplex gradients, and contracts wedged planes partially against points.

With elements typed by orientation, that single operation serves geometry and physics alike. The plane `(u ^ v) & p` through a line and a point has the same algebraic form as the Lorentz force `-(field & current)`, and the point lying on that plane is the same identity `p & p == 0` that ties electrical power to mechanical work on a current. Simplex gradients, rank-one maps and `.adjugate()` follow from `&` in the same way, leaving metrics and material constants inside the duality maps where a model needs them.

## Related work

Grassmann's exterior algebra introduced both the progressive wedge product and the regressive product, and Grassmann–Cayley and projective geometric algebras developed the regressive product for joining points and intersecting subspaces. On degenerate metrics, alternative definitions including pseudoscalar dualization $I^{-1}(IY \wedge IX)$, the shuffle product and Poincaré duality are compared in [[3, §6]](#references).

In exterior calculus, the graded interior contraction $i_v(\omega \wedge \eta) = i_v\omega \wedge \eta + (-1)^p \omega \wedge i_v\eta$, the evaluation bracket $\langle \cdot, \cdot \rangle$ and the dual multivector join $\vee$ are defined on separate vector and form spaces alongside untwisted and twisted differential forms for outer and inner orientation [[4, §§1–2]](#references). On simplicial complexes, finite element exterior calculus [[5, §§1.3, 5]](#references) and discrete exterior calculus [[6, §6]](#references) likewise separate discrete coboundary forms from dual-cell Hodge stars. Within a single oriented geometric algebra, `&` carries the evaluation bracket at complementary grades, the interior contraction when an extrinsic element is paired partially with an intrinsic one, and the point join on intrinsic elements.

In projective rigid-body mechanics, where a null basis vector makes the inner product vanish on translations, point–plane contact ($A_i \vee f < 0$), kinetic energy ($T = \frac{1}{2} B_b \vee I_b[B_b]$), power ($\Pi = B \vee F$) and hinge constraints ($L \vee F_L = 0$) have been written with the regressive product in an unoriented projective algebra, alongside outer-product forms such as $T I = \frac{1}{2} B_b \wedge I_b[B_b]$ and reciprocal-pseudoscalar scalar products $\langle \frac{1}{2} B I_r I[B] \rangle$ for differentiating Lagrangians [[7, §§2.6–2.9]](#references).

Outside projective mechanics, physics in geometric algebra has mostly worked in nondegenerate Euclidean and Minkowski metrics, writing contractions such as the Lorentz force density $J \cdot F$ with the metric inner product [[8, §VII.C]](#references). Premetric electrodynamics in exterior calculus instead writes the Lorentz force density from the untwisted 2-form $F$ and twisted 3-form $J$ using an auxiliary frame $e_\alpha$ as $f_\alpha = (e_\alpha \rfloor F) \wedge J$ and treats the force law and field energy–momentum balance separately [[9, §§3, 5]](#references).

## Companion documents

* [Why orientation](orientation.md): why geometric constructions and positive physical energies require intrinsic and extrinsic orientation.
* [Extensors through multiple lenses](multiple_lenses.md): how multilinear maps and rank-one dyads arise from algebraic products and transform under motions.
* [Complementarity](complementarity.md): how `extrinsic & intrinsic` and `a & duality(a)` unify dual pairs across geometry, mechanics, estimation and physics.
* [Discrete differential geometric algebra](ddga.md): calculus and physics on simplicial topologies built from the simplex gradient.
* [Geometry and quantities](thermodynamics.md): equilibrium and transport thermodynamics in algebras extended with null directions for extensive quantities.

## Appendix: operations across the `numga` examples

Across the 202 `numga` examples, each row below counts uses of one geometric algebra operation and the number of examples in which it appears. Scalar multiplication, sparse topology operators and rank-one dyads are excluded from the bare geometric product:

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
2. [*A Guided Tour to the Plane-Based Geometric Algebra PGA*](https://bivector.net/PGA4CS.pdf) (Leo Dorst and Steven De Keninck, 2022).
3. [*A Bit Better: Variants of Duality in Geometric Algebras with Degenerate Metrics*](https://arxiv.org/pdf/2206.02459) (Charles Gunn, 2022).
4. [*Applied Differential Geometry (A compendium)*](https://people.math.ethz.ch/~hiptmair/Courses/CEM/BOS02c.pdf) (Alain Bossavit, 2002).
5. [*Finite element exterior calculus: from Hodge theory to numerical stability*](https://arxiv.org/pdf/0906.4325) (Douglas N. Arnold, Richard S. Falk and Ragnar Winther, *Bulletin of the American Mathematical Society*, 2010).
6. [*Discrete Exterior Calculus*](https://arxiv.org/pdf/math/0508341) (Mathieu Desbrun, Anil N. Hirani, Melvin Leok and Jerrold E. Marsden, 2005).
7. [*May the Forque Be with You: Dynamics in PGA*](https://bivector.net/PGAdyn.pdf) (Leo Dorst and Steven De Keninck, 2022).
8. [*Spacetime Physics with Geometric Algebra*](https://davidhestenes.net/geocalc/pdf/SpacetimePhysics.pdf) (David Hestenes, *American Journal of Physics*, 2003).
9. [*Recent developments in premetric classical electrodynamics*](https://arxiv.org/pdf/physics/0610221) (Friedrich W. Hehl, Yakov Itin and Yuri N. Obukhov, 2006).
