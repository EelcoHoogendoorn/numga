# The regressive product

This document sets out the regressive product in geometric algebra and the role it can play across geometry, calculus and physics. Traditionally treated in projective geometry as a way to join points or intersect subspaces, the operation is developed here as a general metric-free pairing across orientations. It can evaluate complementary quantities into invariant scalars, contract fields without a metric, build closed-form discrete gradients, and construct linear maps and their transposes.

What sets the regressive product `&` apart from products built out of the geometric product is that it compares the outer product of two elements against the space's [oriented](orientation.md) unit volume, `a ^ b == (a & b) * antiscalar`, rather than consulting a metric. Because the spanned volume `a ^ b` and the unit volume `antiscalar` reverse together in a mirror and share one top-dimensional space even when basis directions have zero length, their ratio `a & b` stays unchanged under reflections and works across every metric signature. When quantities are modelled with orientation and paired by [complementarity](complementarity.md) across a wide range of disciplines, `&` becomes the most frequently used product in the algebra, as [the count across the `numga` examples](#appendix-operations-across-the-numga-examples) illustrates.

## Measuring by volume across orientations

A plane in space divides what lies in front of it from what lies behind it, carrying a sense in the surrounding space. A weighted point carries its sense within itself, through its weight, so a mirror containing the point leaves it unchanged while swapping the sides of an oblique plane. The terms ***intrinsic orientation*** and ***extrinsic orientation*** here mean a sense carried within an element itself (`-`) and a sense carried in the space around it (`+`), as introduced in [*Projective Duality Encodes Complementary Orientations in Geometric Algebras*](https://pure.uva.nl/ws/files/189200508/Projective_duality_encodes_complementary_orientations_in_geometric_algebras.pdf).

In projective three-space, an extrinsic plane sits one grade above scalars and an intrinsic point sits one grade below the four-dimensional top space, so their outer product `plane ^ point` fills a four-volume. Comparing that volume against `antiscalar`, the space's intrinsic unit volume, defines their regressive product:

```python
offset = plane & point                   # [] Scalar+
plane ^ point == offset * antiscalar     # [] Antiscalar-
```

Reflecting the scene in a mirror reverses the handedness of `plane ^ point` and reverses the unit volume `antiscalar` along with it, leaving their scalar ratio `offset` untouched. None of the products derived from the geometric product alone (`*`, `^`, `|`, `commutator`) refers to the unit volume, so none of them pairs an extrinsic quantity with an intrinsic one into a mirror-even scalar.

Because `antiscalar` multiplies `offset` on the right, the definition compares two volumes directly without dividing by `antiscalar` and without asking how long any axis is. In projective space (`x+ y+ z+ w0`), Galilean spacetime and thermodynamic state spaces, null basis vectors make `antiscalar` square to zero so it has no inverse, yet `plane ^ point` remains a clean scalar multiple of `antiscalar`. Writing the extrinsic factor first keeps the same positive sign on matching basis pairs in even and odd ambient dimensions alike.

Replacing a single plane by a stack of equally spaced parallel wavefronts (`wavevector`) turns that geometric offset into wave phase. For an atomic `displacement` step between two points, `wavevector & displacement` counts how many wavefronts the step crosses without using a metric, so stretching the crystal and pulling its wavefront planes back preserves every interference peak ([crystal diffraction](../examples/optics/crystal_diffraction/crystal_diffraction.ipynb)).

## Joining points into lines, surfaces and volumes

While the outer product `^` cuts space down by intersecting hyperplanes, the regressive product `&` builds geometry up from points, as developed in plane-based projective geometric algebra ([*A Guided Tour to the Plane-Based Geometric Algebra PGA*](https://bivector.net/PGA4CS.pdf), Ch. 3; [*A Bit Better: Variants of Duality in Geometric Algebras with Degenerate Metrics*](https://arxiv.org/pdf/2206.02459), §6). Joining two points `a` and `b` in projective three-space spans the directed line from `a` to `b`, joining a third point `c` spans the oriented plane of their triangle, and joining a fourth point `d` closes the tetrahedron into its signed volume:

```python
edge = a & b                             # [] Antibivector-
winding = a & b & c                      # [] Vector-
volume = a & b & c & d                   # [] Scalar-
```

Every join in that chain stays intrinsic (`-`). The line `edge` is oriented along itself from `a` to `b`, the triangle plane `winding` is oriented by its corners going round within the surface rather than by a chosen positive side in the space outside, and `volume` reverses sign when the tetrahedron is reflected in a mirror ([surface normals](../examples/orientation/normals/normals.ipynb)). Dividing two elements of the same grade and orientation forms the rigid motion between them directly: in [a pose from a point, a line and a plane](../examples/geometry/three_point_pose/three_point_pose.ipynb), matching a marker point `a`, a joined edge `a & b` and a joined marker plane `a & b & c` in turn recovers a rigid pose as the product of three motor square roots `(target / source).square_root()`.

Dividing a simplex's facing facet join by its full volume join turns those same geometric spans into discrete differentiation ([Discrete differential geometric algebra](ddga.md)). On a tetrahedron with unit-weight corners `a`, `b`, `c` and `d`, joining the three corners opposite `a` builds `facet`, and dividing by `facet & a` yields the ***simplex gradient*** at `a` ([Whitney forms](../examples/surfaces/whitney_forms/whitney_forms.ipynb)):

```python
facet = b & c & d                        # [] Vector-
gradient_a = facet / (facet & a)         # [] Vector+
```

Because both numerator and denominator are intrinsic, their ratio is an extrinsic hyperplane (`Vector+`) that reads `1` at `a` and `0` at `b`, `c` and `d`, so pairing `gradient_a` with any edge step returns the difference of corner values along that edge without a metric or a matrix inverse.

Joins `&` and meets `^` also compose in a single expression whenever spanned elements are cut by planes. Joining a `light` source with a scene `point` builds the sight `ray` between them, and wedging that ray with a `ground` plane meets them at the `shadow` point where the ray strikes the ground ([scenegraph](../examples/geometry/scenegraph/scenegraph.ipynb), [perspective projection](../examples/geometry/projection/projection.ipynb)):

```python
ray = light & point                      # [] Antibivector-
shadow = ray ^ ground                    # [] Antivector-
```

Replacing the second point in a join by an ideal `direction` (a point of weight zero) spans the line through `point` along `direction` (`point & direction`), and joining two directions sweeps one along the other into an oriented area or worldsheet patch (`direction & time`).

## Partial pairing and the determinant

When two elements have more than enough grade to fill the space, `&` pairs part of one against the other and leaves the remainder. Two planes `u` and `v` meet in an extrinsic line `u ^ v`. Supplying a single intrinsic point `p` tests each plane against `p` in turn and leaves one plane behind:

```python
pencil = (u ^ v) & p                     # [] Vector+
pencil == (u & p) * v - (v & p) * u      # [] Vector+
pencil & p == 0                          # [] Scalar+
```

Because `pencil` is a linear combination of `u` and `v`, it contains their meeting line `u ^ v`. Pairing `pencil` with `p` a second time gives `(u ^ v) & (p & p) == 0` because any point joined with itself vanishes (`p & p == 0`), confirming that `pencil` is the unique plane through the line `u ^ v` and the point `p`.

Supplying a second point `q` and requiring `&` to be associative shows why partial pairing and the point join of the previous section are two sides of one product:

```python
line = p & q                                              # [] Antibivector-
((u ^ v) & p) & q == (u ^ v) & line                       # [] Scalar+
(u ^ v) & line == (u & p) * (v & q) - (v & p) * (u & q)   # [] Scalar+
```

Associativity identifies the joined line `p & q` as the intrinsic element whose pairing with any pair of planes `u ^ v` is the two-by-two determinant of their readings on `p` and `q`, and induction gives the `k`-by-`k` determinant for `k` wedged planes paired with `k` joined points in every dimension. Scalar pairing, metric-free contraction and the geometric join are therefore one associative operation across grades, over which every normalized versor transformation distributes (`versor >> (a & b) == (versor >> a) & (versor >> b)`), projective translations included.

## Building linear maps and transposes without a metric

Pairing `Bivector`, the space of bivectors, with a fixed patch `swept_plane` constructs a linear scalar form on bivectors, and multiplying an output element by that form constructs a rank-one linear map without coordinates, a basis transpose or an inner product ([Extensors through multiple lenses](multiple_lenses.md)):

```python
swept_plane = direction & time                       # [] Antibivector-
readout = Bivector & swept_plane                     # [] Scalar+ <- Bivector+
dielectric = swept_plane * readout                   # [] Antibivector- <- Bivector+
```

Summing such dyads over a crystal's axes builds its dielectric response, just as joining each mass point with its velocity over the space of rates sums into a body's inertia.

Reading a linear map `f` backwards across `&` defines its metric-free adjugate `f.adjugate()`, so that applying `f` on the right pairs to the same scalar as applying `f.adjugate()` on the left:

```python
b & f(a) == f.adjugate()(b) & a                      # [] Scalar+
operator = gradient.adjugate()(material(gradient))   # (Antiscalar- <- Scalar+)[vertices <- vertices]
```

Pulling a material map back through the closed-form simplex gradient assembles discrete Laplacians, stiffnesses and wave operators in one expression ([Discrete differential geometric algebra](ddga.md)), while pushing pose covariance forward onto points takes `shift(covariance(shift.adjugate()))`. Symmetry `material == material.adjugate()` expresses elastic reciprocity and thermodynamic Maxwell relations ([Geometry and quantities](thermodynamics.md)), and differentiating a quadratic form `(a & map(a)) / 2` stays directly in `&` through `(map + map.adjugate()) / 2`, without converting to a metric scalar product.

## Physics from joins, pairings and contractions

With those geometric and linear-map constructions in hand, physical observables and conservation laws take the exact forms already built above, isolating lengths, angles and material properties inside linear maps between complementary spaces.

With each quantity's space, grade and orientation chosen from how it is built and transforms ([representations](complementarity.md#appendix-representations)), a mass at `point` moving with `velocity` sweeps out the point–direction join `momentum = point & velocity`, sitting two grades below the top space in any dimension. A body's rotational `rate` is a plane of rotation at grade 2, like `u ^ v` above, so pairing `rate` with `momentum` is the complementary bivector–line pairing across any dimension, as simulated in [the tennis racket theorem](../examples/mechanics/tennis_racket/tennis_racket.ipynb) and [energy in a mirror](../examples/orientation/mirror_energy/mirror_energy.ipynb):

```python
velocity = point.commutator(rate)        # [] Antivector-
momentum = point & velocity              # [] Antibivector-
energy = (rate & momentum) / 2           # [] Scalar+
```

In three-dimensional Euclidean space (`x+ y+ z+`), `rate` has grade 2 while `momentum` has grade 1, so the metric inner product `rate | momentum` produces a vector rather than a scalar. Because grade 2 from the plane end and grade `n - 2` from the point end always fill the `n`-dimensional volume together, `(rate & momentum) / 2` gives the kinetic energy uniformly in two, three and four dimensions and stays positive in a mirror.

In spacetime (`x+ y+ z+ t-`), electrodynamics follows the same pattern of swept patches and partial pairings. Joining a spatial wire step `direction` with an observer's `time` vector sweeps out the worldsheet patch `direction & time`, and pairing the electromagnetic `field` bivector with that patch reads the voltage per unit length along `direction`. Contracting that `field` bivector with an electric four-`current` antivector is the partial pairing `(u ^ v) & p` above, yielding the Lorentz four-force covector with no Minkowski metric involved:

```python
voltage_slope = field & (direction & time)   # [] Scalar+
lorentz = -(field & current)                 # [] Vector+
lorentz & current == 0                       # [] Scalar+
```

Pairing `lorentz` with a spatial step gives the electric and magnetic force along it, pairing with `-time` gives the Joule heating rate, and electromagnetic work–energy balance (`lorentz & current == -(field & (current & current)) == 0`) is the same identity `pencil & p == 0` that placed `p` on the plane `(u ^ v) & p`. The spacetime metric or crystal permittivity enters once, in the constitutive map `excitation = medium(field)` that forms the Lagrangian density `(field & excitation) / 2` ([constitutive media](../examples/electromagnetism/constitutive/constitutive.ipynb)), while parity-violating interactions join two intrinsic quantities directly (`spin & direction` in [the Wu experiment](../examples/orientation/wu_experiment/wu_experiment.ipynb)) into a handed volume ratio (`Scalar-`) like the tetrahedron volume `a & b & c & d`.

Across these examples, every ordinary scalar observable takes one of two forms:

```python
linear_reading = extrinsic & intrinsic       # [] Scalar+
quadratic_reading = a & duality(a)           # [] Scalar+
```

A linear reading pairs an extrinsic measurement or drive directly with an intrinsic state or flux, while a quadratic energy, Lagrangian or dissipation pairs `a` with its image under a map `duality` into its oppositely oriented complement, such as `rate & inertia(rate)` or `field & medium(field)`. Because `&` uses no metric, keeps one sign across dimensions, stays unchanged in a mirror, and transfers linear maps through `.adjugate()`, [Complementarity](complementarity.md) develops those two forms as a unifying principle across mechanics, elasticity, optics, electrodynamics, estimation, quantum measurement and [thermodynamics](thermodynamics.md), extending unchanged to convex state spaces (`effect & state`) and null state spaces (`drive & mobility(drive)`) where no inner product exists.

## Conclusion

Comparing `a ^ b` against the space's intrinsic unit volume `antiscalar` gives geometric algebra a single metric-free operation across orientations. Because `a ^ b` and `antiscalar` reverse together in a mirror and need no metric inverse, `a & b` pairs extrinsic planes with intrinsic points into mirror-even scalars, joins points and directions into lines, surfaces, volumes and closed-form simplex gradients, and contracts plane meets partially against points by associativity.

Every physical role of `&` follows directly from those geometric constructions: `point & velocity` paired with `rate` gives kinetic energy across dimensions where equal-grade inner products fail, `field & (direction & time)` and `-(field & current)` read voltages and Lorentz forces with work–energy balance `-(field & (current & current)) == 0` matching `((u ^ v) & p) & p == 0`, and `plane & point` reads probabilities and thermodynamic dissipation where no metric exists. Together with rank-one dyads and the metric-free adjugate built on the same pairing, `&` carries the incidence, kinematic and conservation structure of geometry and physics, leaving metrics and media to enter explicitly through duality maps.

## Related work

Grassmann's exterior algebra introduced both the progressive wedge product and the regressive product, and Grassmann–Cayley and projective geometric algebras developed the regressive product for joining points and intersecting subspaces; [*A Bit Better: Variants of Duality in Geometric Algebras with Degenerate Metrics*](https://arxiv.org/pdf/2206.02459) (§6) compares pseudoscalar dualization $I^{-1}(IY \wedge IX)$, the shuffle product and Poincaré duality when the metric is degenerate. In exterior calculus, [*Applied Differential Geometry (A compendium)*](https://people.math.ethz.ch/~hiptmair/Courses/CEM/BOS02c.pdf) (§§1–2, Eq. 1.5) defines the graded interior contraction $i_v(\omega \wedge \eta) = i_v\omega \wedge \eta + (-1)^p \omega \wedge i_v\eta$, the evaluation bracket $\langle \cdot, \cdot \rangle$ and the dual multivector join $\vee$ across separate vector and form spaces, alongside untwisted and twisted differential forms for outer and inner orientation, while [*Finite element exterior calculus: from Hodge theory to numerical stability*](https://arxiv.org/pdf/0906.4325) (§§1.3 and 5) and [*Discrete Exterior Calculus*](https://arxiv.org/pdf/math/0508341) (§6, Definition 6.1) separate discrete coboundary forms from dual-cell Hodge stars on simplicial complexes. Within a single oriented geometric algebra, `&` carries the duality bracket at complementary grades, the interior contraction on `+ & -` above complementary grades, and the intrinsic join on `- & -`.

In projective rigid-body mechanics, where a null basis vector makes the inner product vanish on translations, [*May the Forque Be with You: Dynamics in PGA*](https://bivector.net/PGAdyn.pdf) (§§2.6–2.9, pp. 47–55, citing Gunn's thesis) expresses point–plane contact ($A_i \vee f < 0$), kinetic energy ($T = \frac{1}{2} B_b \vee I_b[B_b]$), power ($\Pi = B \vee F$) and hinge constraints ($L \vee F_L = 0$) in an unoriented projective algebra, while preferring the outer-product form $T I = \frac{1}{2} B_b \wedge I_b[B_b]$ and rewriting $L = \frac{1}{2} B \vee I[B]$ as a reciprocal-pseudoscalar scalar product $\langle \frac{1}{2} B I_r I[B] \rangle$ to differentiate Lagrangians. Outside projective mechanics, physics in geometric algebra has worked primarily in nondegenerate Euclidean and Minkowski signatures where an invertible metric is available, writing pairings and contractions such as the Lorentz force density $J \cdot F$ with the metric inner product in [*Spacetime Physics with Geometric Algebra*](https://davidhestenes.net/geocalc/pdf/SpacetimePhysics.pdf) (§VII.C, Eq. 169), whereas premetric exterior calculus in [*Recent developments in premetric classical electrodynamics*](https://arxiv.org/pdf/physics/0610221) (§3, Eq. 8) writes the covectorial Lorentz force density from the untwisted 2-form $F$ and twisted 3-form $J$ through an auxiliary frame $e_\alpha$ in $f_\alpha = (e_\alpha \rfloor F) \wedge J$.

## Further reading

* [Why orientation](orientation.md): why geometric constructions and physical observables require intrinsic and extrinsic orientation.
* [Extensors through multiple lenses](multiple_lenses.md): how multilinear maps and rank-one dyads arise from partial products and transform under linear maps.
* [Complementarity](complementarity.md): how `extrinsic & intrinsic` and `a & duality(a)` unify dual pairs across fourteen branches of geometry, mechanics and physics.
* [Discrete differential geometric algebra](ddga.md): calculus and physics on simplicial topologies built from the closed-form simplex gradient.
* [Orientation in numga](../docs/orientation.md) and [Conventions](../docs/conventions.md): the defining identities and static orientation rules in `numga`.

## Appendix: operations across the `numga` examples

Across the 202 `numga` examples, each row counts uses of one geometric algebra operation and the number of examples in which it appears; scalar scaling, sparse topology couplings and rank-one dyads (`value * scalar_form`) are excluded from the bare geometric product:

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
