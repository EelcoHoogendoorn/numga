# Physical units

This document examines how physical units behave in a geometric algebra whose elements combine coefficients of different units. A plane in projective geometric algebra combines a normal without a unit with an offset in metres, the rate of a rigid body combines an angular velocity with a linear velocity, and an electromagnetic field combines an electric field with a magnetic field.

Following an approach developed by valneq for plane-based projective geometric algebra [[7]](#references), physical units are taken here to sit on the coefficients of an element, with none on the algebra or its basis vectors. A single condition on products then decides which expressions are consistent, and the units of an element decide which transformations may act on it. [Physical units in numga](../docs/units.md) describes how numga checks these units.

## Units on coefficients

In plane-based projective three-space (`x+ y+ z+ w0`), a plane whose normal has unit length has coefficients without a unit along x, y and z, and an offset in metres along w:

```python
plane = normal + offset * w              # [] Vector+, {x/y/z: 1, w: m}
```

The term ***unit pattern*** here means the unit of each coefficient of an element. Every element of projective three-space has a part on the blades without w and a part on the blades with w. The part without w does not depend on the choice of origin, and the part with w is its moment about the origin, with coefficients that carry one more length. A model of rigid-body mechanics can represent a body's rate as an [extrinsic](orientation.md) bivector, and its momentum and the forque on it as intrinsic antibivectors ([Complementarity](complementarity.md#appendix-representations)). The table below gives the unit patterns of six elements.

| Element | Type | Blades without w | Blades with w |
| --- | --- | --- | --- |
| Plane of unit normal | `Vector+` | normal, 1 | offset, m |
| Point of unit weight | `Antivector-` | weight, 1 | position, m |
| Generator of a motor | `Bivector+` | angle, 1 | shift, m |
| Rate | `Bivector+` | angular velocity, 1/s | velocity at the origin, m/s |
| Momentum | `Antibivector-` | linear momentum, kg·m/s | angular momentum, kg·m²/s |
| Forque | `Antibivector-` | force, N | torque, N·m |

A move of the origin changes the part with w and leaves the other part as it is. The velocity at the origin changes by the angular velocity times the displacement between the two origins, and both terms are in metres per second. The split of a rate into an angular and a linear velocity therefore depends on the origin, while its unit pattern is the same at every origin.

Because the algebra holds no unit, elements in different units of length can share one projective space. A point in a scene has metres on its blades with w, and a point on a screen has pixels there. A map from points in the scene to points on the screen has entries in pixels per metre where it sends coordinates to coordinates and entries without a unit where it sends a weight to a weight.

## Products

Every product of the algebra combines pairs of input blades into output blades by a table of numbers. That holds for the geometric, outer, inner and [regressive](regressive_product.md) products, for commutators and for sandwiches with a rotor. Each term multiplies the units of its two input blades. A product is consistent in its units when every term that lands on one output blade has the same unit, and that condition is the only rule this document needs.

A plane of unit normal paired with a point of unit weight pairs the normal with the coordinates and the offset with the weight, and both terms are in metres. Points `a`, `b`, `c` and `d` of unit weight in metres give joins, a volume and a [simplex gradient](ddga.md) whose unit patterns follow from the same rule:

```python
distance = plane & a                             # [] Scalar+, m
edge = a & b                                     # [] Antibivector-, {yz/zx/xy: m, wx/wy/wz: m²}
facet = b & c & d                                # [] Antitrivector-, {x/y/z: m², w: m³}
volume = a & b & c & d                           # [] Antiscalar-, m³
gradient_a = -facet / volume                     # [] Vector+, {x/y/z: 1/m, w: 1}
```

The unit pattern of a line, a part without w and a part with one more length, is assumed nowhere in these lines. It follows from the units of the points and the rule for products. The simplex gradient has the pattern of a plane divided by a length, and its pairing with a point is a number without a unit, as the weight of a point is.

In rigid-body mechanics, `rate & momentum` pairs each blade with its complement, so the angular velocity multiplies the angular momentum and the linear velocity multiplies the linear momentum, and every term is in joules. A rate paired with itself gives the pitch of its screw:

```python
energy = (rate & momentum) / 2                   # [] Scalar+, J
power = rate & forque                            # [] Scalar+, W
pitch = (rate & rate) / (2 * rate.dot(rate))     # [] Scalar-, m
```

`rate.dot(rate)` pairs the blades without w and reads the squared angular velocity alone. `rate & rate` pairs blades without w with blades with w and gives twice the angular velocity times the velocity along the axis, so the quotient is a length. It is an intrinsic scalar, because it pairs two extrinsic elements, and it changes sign in a mirror as the handedness of a screw does.

## Transformations

A sandwich with a versor adds multiples of some coefficients of an element to others. It is consistent where the units of the versor's coefficients bridge the units of the coefficients that it mixes, so the two unit patterns decide whether a transformation can act on an element.

A translation along x is the exponential of a generator on the blade w∧x, which squares to zero, so its series stops after the first order:

```python
translation = (shift * wx / 2).exp()             # [] Even+, {1: 1, wx: m}
boost = (velocity * wx / 2).exp()                # [] Even+, {1: 1, wx: m/s}
```

The translation adds the weight of a point, without a unit, times a shift in metres to the coordinates of the point, which are in metres, so it acts on points in metres. A Galilean boost has the same blades with metres per second in place of metres. It acts on velocities and not on positions, and both live in the same projective algebra.

A rotation in the plane x∧y is the exponential of a generator on a blade that squares to minus one. Every even power of the generator lands on the scalar blade beside the first term of the series, so the angle has no unit. The rotation adds coefficients along x to coefficients along y, so it acts only on elements whose coefficients along x and y share a unit. That is the reason that lengths along x, y and z share a unit, and it is why a rotation cannot turn metres along x into pixels along y. A general motor combines a rotation without a unit and a translation in metres, and it acts on every element in the table of the first section.

## The complement and duality maps

The complement `.dual()` maps each blade to the blade of the basis vectors that it lacks, and it moves the unit of each coefficient with it. The complement of a point of unit weight in metres has metres on its normal and no unit on its offset, which is not the unit pattern of a plane. Normalized to a unit normal, its offset is an inverse length where a plane has a length. The complement is the polarity in a sphere of one unit of radius, so it hides a squared length that it does not name. Written as a [duality map](complementarity.md) with its radius, a polarity gives a plane the unit pattern of a plane:

```python
# Point stands for the identity map on points.
polarity = x * (x & Point) + y * (y & Point) + z * (z & Point) - radius ** 2 * w * (w & Point)   # [] Vector+ <- Antivector-, {x/y/z: 1, w: m²}
polar = polarity(point)                                                                           # [] Vector+, {x/y/z: m, w: m²}
```

A linear map has a unit on each entry, the unit of its output blade over the unit of its input blade. An inertia, the duality map from rates to momenta, has kilogram square metres from angular velocity to angular momentum, kilograms from velocity to linear momentum, and kilogram metres on the entries that couple the two. Because each of these units is a ratio, one inertia acts on a rate and on its rate of change. A map can be inverted, or a system with it solved, when each connected block of its entries factors as the units of its outputs over the units of its inputs, as the inertia's entries do.

Mechanical eigenproblems, as in [tennis racket](../examples/mechanics/tennis_racket/tennis_racket.ipynb), [vibration modes](../examples/mechanics/modes/modes.ipynb) and [rattleback](../examples/mechanics/rattleback/rattleback.ipynb) examples, show where the complement goes wrong. An eigenproblem of the inertia alone needs a map from rates to rates, and the complement turns the inertia into one with kilogram square metres and kilograms on its diagonal, so its eigenvalues depend on the unit of length. The generalized eigenproblem of a stiffness against an inertia takes no complement, and it gives squared frequencies in 1/s² in every unit of length:

```python
stiffness_form = Bivector & stiffness                       # [] Scalar+ <- (Bivector+, Bivector+)
inertia_form = Bivector & inertia                           # [] Scalar+ <- (Bivector+, Bivector+)
omega_squared, modes = stiffness_form.eigh(inertia_form)    # [6] Scalar+, 1/s²
```

## Squares of basis vectors

A basis vector with a nonzero square acts as a conversion constant between the units of the blades that a rotation across it mixes. In an elliptic or a hyperbolic space (`w+ x+ y+ z+` or `w- x+ y+ z+`), w squares to plus or minus one, a rotation in w∧x turns lengths into angles, and lengths are counted in units of the radius of curvature. Flat space is the limit of an infinite radius, in which w squares to zero, and the transformations that mix the two parts of an element are translations, whose generators carry the ratio of the two units.

In spacetime (`x+ y+ z+ t-`), the equations of electromagnetism that use no metric hold in SI units with no factor of the speed of light. A wavevector has 1/s along t and 1/m along space, a field has volts per metre on its electric planes and teslas on its magnetic planes, and an excitation has coulombs per square metre and amperes per metre:

```python
faraday_gauss = wavevector ^ field               # [] Trivector+, {xyt/yzt/zxt: V/m², xyz: T/m}
ampere_gauss = wavevector ^ excitation           # [] Antivector-, {xyz: C/m³, yzt/xzt/xyt: A/m²}
lagrangian = -(field & excitation) / 2           # [] Scalar+, J/m³
excitation = medium(field)                       # [] Antibivector-, {yz/zx/xy: C/m², xt/yt/zt: A/m}
```

A medium at rest has permittivity on its electric entries and inverse permeability on its magnetic ones, two blocks with units of their own, and its inverse maps excitations back to fields. The speed of light appears where a Lorentz boost acts, as in a moving dielectric in an [electromagnetic media](../examples/electromagnetism/constitutive/constitutive.ipynb) example. A boost is a rotation in the plane z∧t, which squares to one, so its rapidity has no unit. The sandwich with it adds electric coefficients to magnetic ones and magnetic to electric, so a field that a boost acts on has one unit on both kinds of plane, with the electric field divided by the speed of light beside the magnetic field in teslas. A wavevector that a boost acts on likewise has its frequency divided by the speed of light. Sending the speed of light to infinity sends the square of t to zero, and boosts become Galilean.

Algebras that join space or spacetime with null basis vectors for thermodynamic [quantities](thermodynamics.md) keep the units of those quantities apart, because no rotation turns a quantity into a length. In a [rubber band](../examples/thermodynamics/rubber_band/rubber_band.ipynb) example (`x+ y+ s0`), a change of state has metres on its spatial blades and joules per kelvin on its entropy blade. The potentials have newtons on their spatial blades and kelvins on the entropy blade, so every term of `potentials & change` is in joules. In a [magnetotransport](../examples/thermodynamics/magnetotransport/magnetotransport.ipynb) example (`x+ y+ t- w0 s0`), a carrier's parcel has charge along time and entropy along s, and its flux paired with the electric and thermal drive gives a power:

```python
# Spatial stands for the identity map on spatial directions.
advection = Spatial & parcel                     # [] Antibivector- <- Antivector-, {C, J/K}
flux = advection(velocity)                       # [] Antibivector-, {C·m/s, J·m/(K·s)}
power = drive & flux                             # [] Scalar+, W
```

## Extensivity

Thermodynamic variables also divide by how they scale when a uniform system is doubled. An extensive variable such as entropy, volume or energy doubles with the system, and an intensive variable such as temperature or pressure does not. For conjugate pairs this division matches orientation, since states are intrinsic and potentials extrinsic, but it follows other rules. Orientation is a sign that closes under every product. Extensivity is a degree of scaling, and ordinary operations leave its two values. A density has degree zero, a thermodynamic stiffness that maps a change of state to a change of potentials has degree minus one, and a product of two extensive values has degree two.

A degree of scaling is an exponent, so extensivity can be counted as one more dimension, `extent`. An extensive quantity carries it to the first power and an intensive one does not. `potentials & change` then comes out extensive, as an energy should, and a sum of an intensive and an extensive quantity is inconsistent. Orientation keeps the sign under reflections, and extensivity sits with the units.

## Conclusion

Units on the coefficients of elements, with none on the algebra, are consistent when the terms that land on one coefficient have one unit. In projective space the pattern of a part without w and a part with one more length is not assumed but follows from that condition, and one projective algebra holds metres beside pixels and translations beside Galilean boosts.

The same condition decides which transformations may act on an element. A rotation in a plane with a nonzero square has a generator without a unit and acts only on coefficients that share a unit, which is what makes the square a conversion constant, between the axes of space, between space and time in a boost, and between length and angle in a curved space. A null basis vector has no such constant, so the units of the quantities in a joined algebra stay apart, and extensivity joins them as one more dimension.

## Related work

The convention that units sit on the coefficients rather than on the basis vectors of projective geometric algebra is worked out in two dimensions by valneq [[7]](#references). In that work, elements whose Euclidean part carries no unit and whose ideal part carries a length are closed under sums, geometric products, exponentials and sandwiches [[7, §§3.1, 3.3]](#references), the join of two points introduces a global factor of length [[7, §§2.2–2.3, 3.2]](#references), the time derivative of a point lands on the ideal part in length per time [[7, §3.4]](#references), and the Hodge dual of a point returns to the same pattern only after multiplying its ideal part by a squared length [[7, §3.2]](#references). This document adopts that coefficient-level view across three-dimensional mechanics, spacetime and coupled thermodynamics, identifies the squared length in the dual as the radius of a polarity, and extends the condition to arbitrary per-blade unit patterns and linear maps.

In projective rigid-body mechanics, the units of planes, points, joins, rates, momenta, forces, inertia, energy and power are set out with the dimension of an inverse length given to the null basis vector [[5, Ch. 1 §§1.1–1.3, Ch. 2 §§2.1–2.7]](#references). That wording and the one used here describe the same coefficients. A dimension on the basis vector fixes one unit of length for the whole algebra, while units on coefficients let points in metres and points in pixels, or translations and Galilean boosts, share one algebra.

In robotics literature, motion and force inhabit dual spaces paired into mechanical power [[1, Ch. 2 §5]](#references), and spatial-vector tutorials emphasize that no inner product is defined on motion coordinates alone [[2]](#references). In the terms of this document, `rate & momentum` pairs motion with force into energy and power, `rate.dot(rate)` reads the angular velocity alone, and `rate & rate` reads the pitch of a screw.

Euclidean dual-Grassmann treatments write kinetic energy through a dual pairing between velocity axes and momentum spears [[3, §6]](#references), Cayley–Klein generalizations across Euclidean, elliptic and hyperbolic spaces write kinetic energy and power through the regressive pairing [[4, Ch. 9]](#references), and plane-based dynamics writes them in one algebra [[5, §§2.6–2.9]](#references). Definitions of the regressive product on degenerate metrics through dualization, the shuffle product and Poincaré duality are compared in [[6, §6]](#references).

## Companion documents

* [Why orientation](orientation.md): intrinsic and extrinsic orientation from mirror reflections and positive observables.
* [The regressive product](regressive_product.md): the metric-free definition against the unit volume, joins of points, simplex gradients and adjugates.
* [Complementarity](complementarity.md): complementary pairings and duality maps across geometry, mechanics, estimation and state spaces.
* [Discrete differential geometric algebra](ddga.md): calculus and physics on simplicial topologies built from the simplex gradient.
* [Geometry and quantities](thermodynamics.md): spacetime joined with null thermodynamic directions for equilibrium and coupled transport.
* [Physical units in numga](../docs/units.md): how numga checks units on the blades of values and the entries of maps.

## References

1. [*A Mathematical Introduction to Robotic Manipulation*](https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-complete.pdf) (Richard M. Murray, Zexiang Li and S. Shankar Sastry, CRC Press, 1994).
2. [*A Beginner's Guide to 6-D Vectors (Part 1)*](https://bleyer.org/files/A%20Beginner%27s%20Guide%20to%206-D%20Vectors%20-%20Feathersone%20%28IEEE%2C%202010%29.pdf) (Roy Featherstone, *IEEE Robotics & Automation Magazine*, 2010).
3. [*On the Homogeneous Model of Euclidean Geometry*](https://arxiv.org/pdf/1101.4542) (Charles Gunn, in *Guide to Geometric Algebra in Practice*, Leo Dorst and Joan Lasenby, eds., Springer, 2011; extended preprint of AGACSE 2010).
4. [*Geometry, Kinematics, and Rigid Body Mechanics in Cayley-Klein Geometries*](https://depositonce.tu-berlin.de/bitstreams/5a02e6fc-d9c3-4e6f-a5ab-f4a970e81209/download) (Charles Gunn, Doctoral Dissertation, Technische Universität Berlin, 2011; DOI `10.14279/depositonce-3058`).
5. [*May the Forque Be with You: Dynamics in PGA*](https://bivector.net/PGAdyn.pdf) (Leo Dorst and Steven De Keninck, 2022).
6. [*A Bit Better: Variants of Duality in Geometric Algebras with Degenerate Metrics*](https://arxiv.org/pdf/2206.02459) (Charles Gunn, 2022).
7. *PGA and the problem of length scale* (valneq, v2, July 2025, shared on the [bivector.net](https://bivector.net/) Discord).
