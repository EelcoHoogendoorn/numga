# Why orientation

This document motivates orientation in geometric algebra from two requirements, one geometric and one physical, and sets out the rules they imply. Nothing in the algebra's products asks for orientation on its own; the need comes from how elements transform and how physical quantities are read.

The geometric requirement is that transformations of space commute with constructions: transforming the ingredients of a construction and building from them gives the same result as building first and transforming the result, across every product of the algebra. The physical requirement is that mirroring a system together with its measuring apparatus leaves every scalar reading unchanged, such as kinetic energy, mechanical power or wave phase, while reversing handed quantities such as signed volumes.

## What a physical mirror does

Two pictures show what a physical mirror does to a line, and everything after them builds on these two. Here points are oriented by their weights and planes by their positive sides.

<p align="center"><img src="../plots/pairs_in_mirror.png" alt="Two points on a line lying in a mirror, and two planes crossing in that line, before and after the mirror" width="353" /></p>

*Figure: a vertical mirror, the pale sheet, and a line lying in it. Top: a red point and a blue point on the line. Bottom: a red plane and a blue plane crossing in the line, each at half a right angle to the mirror. Left: the scene. Right: its mirror image.*

In the top row, the two points lie in the mirror and keep their positions and weights. In the bottom row, each plane's mirror image lies where the other plane was, so the mirror swaps them. In both rows the line stays where it is.

The same line can be built from either pair: as the join of the two points, or as the meet of the two planes. If reflecting is to commute with building, the join of the unmoved points is the join itself, so a reflection must leave the join unchanged. The meet of the swapped planes is the meet built in the opposite order, and because the meet is antisymmetric, changing sign when its two planes are exchanged, a reflection must negate the meet. The same line, built from two points or from two planes, must come out of a reflection unchanged in one case and negated in the other.

## Orientation

The join can be viewed as a direction along the line, from the one point to the other, lying within the line itself. A mirror that contains the line leaves everything within the line alone, so that direction stays as it was.

The meet can be viewed as a rotation around the line, from the one plane to the other, lying in the space around the line. A mirror that contains the line flips the space around it side for side, so that rotation is reversed.

The terms ***intrinsic orientation*** and ***extrinsic orientation*** here mean an orientation within an element itself (`-`) and an orientation in the space around it (`+`), following [[1]](#references). An element's orientation records which of the two it carries.

## The unoriented sandwich

In projective three-space, where vectors are planes and antivectors are points, the outer product of two planes is their meet and the regressive product of two points is their join. Both lines have grade two and the same coefficients up to scale.

Reflecting every element with the grade-involuted sandwich `mirror * x.involute() * mirror.inverse()` swaps `red` and `blue` and commutes with the geometric and outer products, so it negates the meet `red ^ blue` as it should. Because it commutes with the outer product while reversing the space's unit volume, it anti-commutes with the regressive product in every dimension and at every grade, turning around the joined line `ends[0] & ends[1]` that lies in the mirror:

<p align="center"><img src="../plots/line_in_mirror.png" alt="Two points in a mirror and the line through them: the line, its mirror image, and the sandwich product's result" width="603" /></p>

*Figure: two points lying in a mirror, the pale sheet, and the line through them, with an arrow for its intrinsic orientation. Left: the line. Middle: its mirror image, with its direction unchanged. Right: the result of reflecting with the unoriented sandwich; the line is turned around.*

```python
# The mirror is the plane x = 0; two planes cross in the z axis, and two points lie on it.
mirror = mv.x                                          # [] Vector+
red, blue = mv.x + mv.y, mv.y - mv.x                   # [] Vector+, [] Vector+
ends = mv.point([[0, 0, -0.5], [0, 0, 0.5]])           # [points] Antivector-

def reflect(x):
    return mirror * x.involute() * mirror.inverse()

# Minus the meet, matching blue ^ red, the meet of the mirrored planes.
reflect(red ^ blue)                                    # [] Bivector+
# Minus the join, turned around instead of unchanged.
reflect(ends[0] & ends[1])                             # [] Antibivector-
```

Putting an overall minus sign in front, `-mirror * x.involute() * mirror.inverse()`, leaves intrinsic points in the mirror and their joins unchanged, while leaving the mirror plane itself unflipped, negating planes perpendicular to the mirror, and leaving `red ^ blue` unflipped instead of negated. Applied uniformly to every element, no single sandwich rule has the geometric action of a mirror.

## An orientation-aware reflection

An orientation-aware reflection `mirror >> x` negates the grade-involuted sandwich for intrinsic elements and leaves it unchanged for extrinsic ones, reading the orientation from the element's type. Reflection then commutes with the join just as it commutes with every other product:

```python
# The join, unchanged.
mirror >> (ends[0] & ends[1])                          # [] Antibivector-
# Minus the meet.
mirror >> (red ^ blue)                                 # [] Bivector+
```

## Observables

Mirroring a system together with its measuring apparatus must also preserve every physical reading. Kinetic energy gives a direct test, because a spinning body and its mirror image turn opposite ways while carrying the same positive kinetic energy:

<p align="center"><img src="../plots/mirror_energy.gif" alt="A body of three masses turning, and its mirror image turning the other way with the same kinetic energy" width="600" /></p>

*Figure: a body of three masses turns, and its mirror image turns the other way. Colours identify matching masses, marker sizes show the masses, and arrows show their velocities.*

The body's rate of turning is an extrinsic line oriented around its axis, while the mass points are intrinsic. Joining each mass point with its velocity under that rate gives an intrinsic angular momentum oriented along its line, and the kinetic energy is half the regressive pairing of the rate with the momentum. Under the orientation-aware reflection, the mirrored body has the same positive kinetic energy. Under the uniform sandwich `reflect`, both factors preserve their outer product while the unit volume reverses, and adding an overall minus sign negates both factors and cancels in their pairing, so either uniform rule changes the sign of `rate & momentum` in every dimension and turns the mirrored body's kinetic energy negative:

```python
points = mv.point([[1, 0, 0], [0, 1, 0.5], [-0.5, 0, 1]])  # [points] Antivector-
masses = np.array([1, 1.5, 0.5])                           # [points] Scalar+
rate = mv.xy + 0.3 * mv.yz                                 # [] Bivector+
velocities = points.commutator(rate)                       # [points] Antivector-
momentum = (points & (velocities * masses)).sum()          # [] Antibivector-

# 1.494.
(rate & momentum) / 2                                      # [] Scalar+
# 1.494: the same energy.
((mirror >> rate) & (mirror >> momentum)) / 2              # [] Scalar+
# -1.494: negative energy.
(reflect(rate) & reflect(momentum)) / 2                    # [] Scalar+
```

An [energy in a mirror](../examples/orientation/mirror_energy/mirror_energy.ipynb) example animates the body, and a [mechanical work](../examples/orientation/mechanical_work/mechanical_work.ipynb) example shows the same contrast for the power a push delivers to a crank.

Every such reading pairs an extrinsic value with an intrinsic one by the regressive product, such as `rate & momentum` above, `field & spin`, or `force & displacement`. As [Complementarity](complementarity.md) develops, that pairing uses no metric and remains unchanged across dimensions, signatures and general linear maps as well as in a mirror.

## Every element has an orientation

The line is one grade among the rest. Every element of the algebra can carry either orientation, intrinsic or extrinsic, and a mirror treats the two differently at every grade. A triangle shows the distinction on planes:

<p align="center"><img src="../plots/normal_orientation.png" alt="A triangle's positive side, and the order of its corners, before and after a vertical mirror" width="603" /></p>

*Figure: a triangle lying in a horizontal plane, and its mirror image in a vertical mirror. Left: the plane's positive side, which stays above both triangles. Right: the order of the corners, which the mirror reverses.*

The plane equation of the triangle's plane orients it by which side is positive in the space outside it, an extrinsic orientation, and the vertical mirror keeps the positive side up. Joining the triangle's three corners in order gives the same plane oriented by its corners going round within it, an intrinsic orientation, and in the mirror image the corners go round the other way. A [surface normals](../examples/orientation/normals/normals.ipynb) example draws both:

```python
corners = mv.point([[0, 0, 0], [1, 0, 0], [0, 1, 0]])      # [corners] Antivector-
# The plane z = 0, positive above.
side = mv.z                                                # [] Vector+
# The same plane, its corners going round.
winding = corners[0] & corners[1] & corners[2]             # [] Antitrivector-

# +z: the positive side still points up.
mirror >> side                                             # [] Vector+
# -z: the corners go round the other way.
mirror >> winding                                          # [] Antitrivector-
```

In a plane, a point can likewise be oriented by its weight, as a mass is, or by a turn around it, as a pivot is. In the conformal algebra of space [[2, §1.1]](#references), a [circle constructed twice](../examples/orientation/conformal_circle/conformal_circle.ipynb) example shows the same intrinsic and extrinsic split for a circle joined from points versus met from spheres.

## From the plane to space

Stepping up a dimension from a plane into three-space separates the two orientations by how an element behaves when a new axis is added. In the projective plane (`w0 x+ y+`, dimension 3), a mirror line `x` has grade one, a turn `xy` about the origin has grade two, and the origin point itself has that same grade-two blade `xy` with intrinsic orientation.

Embedding that sheet into projective three-space (`w0 x+ y+ z+`, dimension 4) adds an orthogonal axis `z` and splits the two kinds cleanly. An extrinsic element imposes constraints on its surroundings (fixed codimension), so adding `z` leaves both its grade and its algebraic expression untouched while its locus extrudes along the new axis: the mirror line `x` becomes the vertical mirror plane `x`, a wavefront line becomes a wavefront plane, and the turning rate `xy` stays a grade-two bivector whose pivot point extrudes into a rotation axis. Restricting back to the sheet recovers the original element with no choice of sign along `z`.

An intrinsic element carries a fixed dimension of its own rather than extruding along the added axis. A point mass on the sheet remains a point mass in the room, a momentum line remains a line, and a triangle patch remains a patch, so its grade rises by one with every added dimension (grade n − 1 for points, n − 2 for lines, n − 3 for patches). Keeping the two-dimensional point's grade-two blade `xy` fixed in three-space instead turns it into the vertical z-line, one dimension up:

```python
# plane_mv in the projective plane, space_mv in projective three-space.
mirror_2d = plane_mv.x                                     # [] Vector+ (grade 1: x)
mirror_3d = space_mv.x                                     # [] Vector+ (grade 1: x)
pivot = plane_mv.xy                                        # [] Bivector+ (grade 2: xy)
axis = space_mv.xy                                         # [] Bivector+ (grade 2: xy)

point_2d = plane_mv.point([0, 0])                          # [] Antivector- (grade 2: xy)
point_3d = space_mv.point([0, 0, 0])                       # [] Antivector- (grade 3: xyz)
z_line = space_mv.point([0, 0, 0]) & space_mv.point([0, 0, 1])  # [] Antibivector- (grade 2: xy)
```

Constructing an extrinsic point in three-space requires meeting that axis with a third plane, and the chosen side of that plane becomes part of the orientation:

```python
extrinsic_point = axis ^ space_mv.z                        # [] Trivector+
opposite_point = axis ^ (-space_mv.z)                      # [] Trivector+
```

A turn in the sheet does not choose which side of the sheet is positive. Choosing the opposite side reverses the extrinsic point's orientation while keeping its location and the turn in the sheet unchanged, because in three-space an extrinsic point carries a handedness of its surroundings. An intrinsic point needs no choice along the added direction because its orientation stays within the point itself.

Viewing both constructions across dimensions also explains the combinations that at first look exotic, such as an intrinsic plane, an intrinsic bivector or an extrinsic antivector. They are simply what appears where the two ladders cross the same grade in a given dimension. Outer products of mirrors climb up from grade 0 at fixed grade while their loci extrude along each added axis; joins of points step down from grade n at fixed geometric dimension while their grades rise by one with n. Each cell below gives the grade and geometric locus of one construction as the ambient projective dimension grows:

| Construction | Orientation | Plane (n = 3) | Three-space (n = 4) | Four-space (n = 5) |
| --- | --- | --- | --- | --- |
| Mirror `u` | extrinsic (`+`) | grade 1, line | grade 1, plane | grade 1, hyperplane |
| Meet `u ^ v` | extrinsic (`+`) | grade 2, pivot point | grade 2, axis line | grade 2, plane |
| Meet `u ^ v ^ w` | extrinsic (`+`) | grade 3, pseudoscalar | grade 3, extrinsic point | grade 3, line |
| Point `p` | intrinsic (`-`) | grade 2, point | grade 3, point | grade 4, point |
| Join `p & q` | intrinsic (`-`) | grade 1, line | grade 2, line | grade 3, line |
| Join `p & q & r` | intrinsic (`-`) | grade 0, signed area | grade 1, triangle plane | grade 2, triangle patch |

In three-space, an extrinsic antivector (`u ^ v ^ w`, grade 3) is a point cut out by three mirrors, and stepping into four-space extrudes it into a line; an intrinsic bivector (`p & q`, grade 2) is a line or momentum spear joined from two points, and an intrinsic plane (`p & q & r`, grade 1) is the triangle winding `winding` above, both climbing one grade higher in four-space without changing their geometric dimension. Opposite crossings still pair into mirror-even extrinsic scalars: in three-space, an extrinsic point cut out by three planes pairs with an intrinsic triangle plane to read the point's signed offset from the triangle's plane:

```python
# Extrinsic point (grade 3+) paired with the intrinsic triangle winding (grade 1-) above.
above = axis ^ (space_mv.z - 3 * space_mv.w)               # [] Trivector+
# -3: unchanged under mirror >> on both factors.
above & winding                                            # [] Scalar+
```

## Orientation across products

Once the orientations of the inputs are chosen, requiring reflections to commute with every product fixes the orientation of every result. Because the unoriented sandwich commutes with the outer product, the orientation sign of the geometric and outer products is the product of the factors' signs. Because the unoriented sandwich negates the regressive product, commuting with a reflection reverses that sign rule for the regressive product: two intrinsic elements join into an intrinsic one, and pairing an extrinsic element with an intrinsic one produces a mirror-even extrinsic scalar. Taking the metric-free complement swaps what lies within an element with what lies around it, exchanging the two orientations.

The ordering of a pairing also matters across dimensions. In the three-dimensional projective plane, `plane & point` and `point & plane` agree in sign, whereas in four-dimensional projective three-space swapping an extrinsic vector and an intrinsic antivector negates the result (`point & plane == -(plane & point)`). Putting the extrinsic factor first keeps complementary basis pairings positive in even and odd dimensions alike.

## Generators and carried objects

Those product sign rules divide the algebra into two distinct algebraic roles. Because the product of two extrinsic elements stays extrinsic, the extrinsic side forms a closed subalgebra under the geometric product. Multiplying mirror hyperplanes builds versors (rotations, translations and boosts), and extrinsic bivectors (turning rates, twists, boosts and electromagnetic fields) close under the commutator to form the Lie algebra of continuous transformations. Every power of an extrinsic bivector stays extrinsic as well, so its exponential `rate.exp()` has one consistent orientation at every order and generates a continuous motion.

The intrinsic side does not close under the geometric product. Raising an intrinsic element to successive powers alternates its orientation between extrinsic on even powers and intrinsic on odd powers, so an intrinsic bivector cannot be exponentiated, and the commutator of two intrinsic bivectors lands on the extrinsic side rather than closing on itself. Instead, intrinsic elements close under the regressive product (points joined into lines, patches and volumes, together with extensive quantities such as mass and entropy) and remain intrinsic under the action of extrinsic transformations:

```python
motion = rate.exp()                                        # [] Motor+
bracket = rate.commutator(other_rate)                      # [] Bivector+
moved = motion >> point                                    # [] Antivector-
velocity = point.commutator(rate)                          # [] Antivector-
momentum = point & velocity                                # [] Antibivector-
crossed = momentum.commutator(other_momentum)              # [] Bivector+ (leaves intrinsic side)
```

Orientation also rules out expressions whose terms transform in incompatible ways. In the spinning body above, `rate` and `momentum` carry opposite orientations, so `rate + momentum` has no single transformation law under reflection and `momentum.exp()` mixes orientations across its series. Metric norms stay within one orientation at a time as well, measuring intrinsic directions through their duals so a spatial step's squared length stays positive in projective and spacetime algebras alike, while projective points carry both a norm (their weight) and an antinorm (the length of a direction).

In matrix notation a plane's side transforms with the inverse transpose of a linear map and a triangle's area normal with its cofactor matrix, which differs by the determinant of the map, $-1$ for a mirror. In exterior calculus the two kinds correspond to outer and inner orientation, represented by untwisted and twisted differential forms [[3, §§1–2]](#references), and in projective geometry the two kinds of line correspond to Study's spears and axes [[4, §2.2]](#references), [[5, §8.6.1]](#references).

## The rules

Together with two conventions, that expressions keep the same form and signs across dimensions and signatures and that each operation uses a single symbol wherever the requirements allow, the geometric and physical requirements leave six rules.

1. Types follow orientation rather than grade, so an intrinsic plane joined from three points obeys the intrinsic rules as a point does, while an extrinsic point met from three planes obeys the extrinsic ones.
2. A reflection acts by orientation, sending an extrinsic element to `m * x.involute() * m.inverse()` and an intrinsic one to the negative of that, while a rotation (an even number of reflections) carries no orientation-dependent sign.
3. Scalar readings pair an extrinsic element with an intrinsic one, placing the extrinsic factor first so that no sign depends on the dimension.
4. A metric measures one orientation at a time, measuring intrinsic directions through their duals so that spatial squared lengths stay positive in every signature, while opposite orientations pair through the regressive product rather than a metric dot product.
5. The primitive products (geometric, outer, regressive and commutator) act on coefficients as the algebra defines them, and their results' orientations follow from their inputs.
6. An intrinsic element carries both a norm, such as the weight of a projective point, and an antinorm, such as the length of a direction.

## Companion documents

* [Extensors through multiple lenses](multiple_lenses.md): how multilinear maps arise from partial products, transform under motions and assemble from dyads.
* [The regressive product](regressive_product.md): how the regressive product pairs orientations, joins points, contracts fields and builds linear maps and their adjugates.
* [Complementarity](complementarity.md): how complementary pairings and physical representations extend across geometry, mechanics and physics.
* [Orientation in numga](../docs/orientation.md): the orientation rules for every constructor and operation in `numga`.

## References

1. [*Projective Duality Encodes Complementary Orientations in Geometric Algebras*](https://pure.uva.nl/ws/files/189200508/Projective_duality_encodes_complementary_orientations_in_geometric_algebras.pdf) (Leo Dorst, *Mathematical Methods in the Applied Sciences*, 2024).
2. [*Geometric Algebra for Computer Science: An Object-Oriented Approach to Geometry*](https://geometricalgebra.org/downloads/ga4cs_chapter1.pdf) (Leo Dorst, Daniel Fontijne and Stephen Mann, Morgan Kaufmann, 2007; open Chapter 1).
3. [*Applied Differential Geometry (A compendium)*](https://people.math.ethz.ch/~hiptmair/Courses/CEM/BOS02c.pdf) (Alain Bossavit, 2002).
4. [*On the Homogeneous Model of Euclidean Geometry*](https://arxiv.org/pdf/1101.4542) (Charles Gunn, in *Guide to Geometric Algebra in Practice*, Leo Dorst and Joan Lasenby, eds., Springer, 2011; extended preprint of AGACSE 2010).
5. [*A Bit Better: Variants of Duality in Geometric Algebras with Degenerate Metrics*](https://arxiv.org/pdf/2206.02459) (Charles Gunn, 2022).
