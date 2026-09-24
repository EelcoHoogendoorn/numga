# Extensors, advanced

An extensor carries no metric in its coefficients. The metric is in the products of the
algebra, and duality between spaces is given by the regressive product. Several operations
that are a single reflex in matrix algebra are therefore several different things here. Some
are a product with an open slot. Some require a metric to be named. The transpose is not an
operation on extensors at all. For comparison, in matrix algebra one would say that the
transpose, the trace, the squared norm of a residual and the reciprocal of a basis all
identify a vector space with its dual through the coefficients, and that in an orthonormal
basis the identification is invisible.

The sections below treat these in order: the absence of a matrix product and of a transpose; the metric and the
complement; the relation between maps and forms; quadrics; inverses; least squares; the
conversion between second moments and inertia; traces, with the Ricci contraction as the
example; homogeneous unknowns; covariance and information; batch axes and slots; outermorphisms. The
expressions are taken from the [examples](../examples/README.md).

## 1. There is no matrix product

Extensors compose. A call fills an input slot with anything of that slot's type, a value or
another map, and a map's output feeds the next map's input when the types agree:

```python
world = motor >> inertia(motor << Bivector)              # AntiBivector <- Bivector: the inertia of a placed body
cone = on_planes(disc(projection))                       # Plane <- Point: a pixel's disc pulled back into the scene
```

What there is not is an operator like `a @ b`, which contracts two arrays by index position
whatever the indices stand for. Composition only joins an output to an input of the same
type. Every other pairing of two slots, the ones matrix code writes as `x.T @ y`, `A.T @ A`
or a sum of squares, is a product of the algebra, and the product says which pairing it is:

```python
incidence = Plane & Point                                # Scalar <- (Plane, Point): no metric involved
metric = Vector | Vector                                 # Scalar <- (Vector, Vector): the algebra's metric
misfit = residual.reverse().scalar_product(residual)     # a squared norm, in the metric it is measured by
```

Neither is there a tensor product. Arity grows only by leaving a slot open in an expression,
and it falls only by binding a slot or by a product of the algebra. A dyad is written as a
product with an open slot, `a * (b & Point)`, not as a ⊗ b, and a contraction chosen by index,
the other half of `einsum`, has no counterpart.

Underneath, all of these are contractions of coefficient arrays. The difference is in what can be
written: every contraction has a meaning in the algebra and a type, and a contraction by index
alone cannot be expressed. The transpose is the first casualty.

## 2. There is no transpose

Extensors have no transpose. What follows are the correspondences: for each task that would
call for a transpose in matrix algebra, the expression that does it on extensors. The pairings
these expressions use are the subject of sections 3 and 4. For comparison, in matrix algebra
one would write the transpose as swapping rows and columns, which identifies a space with its
dual by equal coefficient index; that is the metric of a Euclidean orthonormal basis and of no
other.

**Pulling a quadric back through a map.** In the [multiview example](../examples/geometry/multiview/core.py) a camera is a map from
scene points to sensor points, `projection: Point <- Point`. It has no inverse: every point on
a sight ray lands on the same pixel. A pixel measurement is a quadric on sensor points,
`disc`, and the reconstruction needs that quadric on scene points, scoring a scene point by how
far its image falls from the pixel. Feeding the camera into the quadric handles its input;
its output, a polar plane on the sensor, has to be carried back to the scene as well, and
that needs a map on planes induced by the map on points. Solving the incidence pairing against
the camera gives that map, and it exists although the camera has no inverse:

```python
on_planes = (Plane & Point).solve(Plane & projection)     # Plane <- Plane: on_planes(l) & p == l & projection(p)
cone = on_planes(disc(projection))                        # Plane <- Point: the pixel's disc as a cone in the scene
```

A quadric kept as a form needs no induced map: the camera goes into both slots,
`form(projection, projection)`.

**The curvature and gradient of a cost.** The same example then aligns the cameras. The cost
is the value of each cone at the reconstructed point, `cones(local_points) & local_points`, and
a small change of a camera's pose moves its points by `motion`, a map from twists to points.
A quadric cost is its own square, so its curvature over the pose is the quadric with the motion
in both slots, and its gradient is the point's polar joined with the motion. When the residual
is a scalar to begin with, as in the [epipolar example](../examples/geometry/epipolar/core.py) where it is the wedge of two lines read
as a number, the curvature is the product of its Jacobian form `j` with itself.

```python
curvature = (cones(motion) & motion).sum(axis=0)          # Scalar <- (Twist, Twist)
gradient = (cones(local_points) & motion).sum(axis=0)     # Scalar <- Twist
curvature = (j * j).sum(axis=0)                           # Scalar <- (Twist, Twist), j a Scalar <- Twist
```

Where a residual is a multivector and a norm on it is part of the problem, the norm is a
choice and is written as a form on the open residual. The [registration example](../examples/geometry/registration/core.py) fits a motor
to point correspondences through a residual that is linear in the motor, `target * Motor -
Motor * source`, and scores it by the sum of its squared coefficients. In PGA that is the bulk
norm plus the weight norm, the second taken through the complement, because the scalar
product alone is blind to the translational part:

```python
bulk = residual.reverse().scalar_product(residual)                  # Scalar <- (Motor, Motor)
weight = residual.dual().reverse().scalar_product(residual.dual())
misfit = (bulk + weight).sum(axis=0)
```

**Moving a covariance.** The [Kalman example](../examples/geometry/kalman/core.py) tracks a pose with an uncertainty `sigma`, the
covariance of a small twist perturbing the estimate. A covariance takes a linear readout of
that twist, which is a line, to the twist correlated with it, so it is a map `Twist <- Line`,
and when the estimate advances by a motor `step` the covariance moves like any map: pull the
readout back through the step, push the twist forward.

```python
sigma = step << sigma(step >> Line) + Q                     # Twist <- Line
```

**A direction from a gradient.** The [ray tracer](../examples/geometry/cyclides/core.py) needs a surface normal for shading. The
derivative of a surface's quadric along a ray, `derivative`, is a linear form on directions,
`Scalar <- Direction`; the normal is the direction obtained by solving the metric form on
directions against it.

```python
normal = (Direction | Direction).solve(derivative)          # Direction
```

**Symmetrizing.** A form built from an open expression paired with itself is symmetric as it
stands. The rotor fit in the registration example aligns source vectors to target vectors by
the form `target.scalar_product(Rotor >> source)`, with the rotor open on both sides of the
sandwich; that form is symmetric, and its top eigenvector is the fit.

```python
alignment = target.scalar_product(Rotor >> source).sum(axis=0)   # Scalar <- (Rotor, Rotor), symmetric
```

**Inverting an orthogonal map.** The inverse. In the ray tracer `screen` is the camera's frame
as a map on directions, orthonormal, and the camera-frame up direction is its inverse applied
to the world's.

```python
up = screen.inverse()(mv.z)
```

This is the one place a transposition of coefficients does happen, underneath. A map whose
type carries the trait `CoefficientOrthogonal`, such as a rotor sandwich in a Euclidean
algebra, dispatches `inverse()` to a transposition of its coefficients, because for such a map
the two coincide. The trait is a promise the type system tracks; a Lorentz boost does not
carry it, and its inverse is computed as an inverse.

**Moving a map or a form to another frame.** A sight cone in the [multiview example](../examples/geometry/multiview/core.py) is built
in its camera's frame and is summed with the other cameras' cones in the world's. As a map it
moves by pulling its input back through the pose and pushing its output forward; as a form it
has no output to push, and the pose goes into both slots.

```python
world_map = motor >> local_map(motor << Point)              # Plane <- Point
world_form = local_form(motor << Point, motor << Point)     # Scalar <- (Point, Point)
```

The pushed output is a plane, and the sandwich moves a plane as it moves a point: the plane
map induced by the pose, `on_planes(motor << Point)`, is `motor >> Plane`. For comparison, in
matrix algebra one would store the quadric as a symmetric matrix Q and the pose as a matrix P
on point coordinates, and move the quadric by congruence, P^{-T} Q P^{-1}. The transpose there
is the pullback of the form's second slot, and the inverse transpose is how a matrix on points
is made to act on planes. The sandwich is both.

In coordinates each of these is a permutation of the same numbers, with signs. The permutation
and the signs come from the structure constants of the product, which blades meet to a scalar
and with what sign, and not from index labels.

## 3. The metric and the complement

The products of the algebra split into two groups. The geometric product, the inner product
`|`, the commutator, the sandwich, the norm and the inverse of a multivector use the metric.
The wedge `^`, the complement `dual`, the regressive product `&`, addition, composition of maps
and the trace of a map do not. The second group is the exterior algebra inside the geometric algebra. A
calculation written entirely in it never uses the metric, and it is the group that carries
incidence in a projective algebra, where the metric is degenerate.

The metric is in the products and not in the coefficients, and it is available as an object
when one is wanted: the inner product with two open slots is the metric as a form. With it,
moving between a vector and the covector it defines is an ordinary solve.

```python
metric = Vector | Vector                    # Scalar <- (Vector, Vector): the metric as a form
covector = v | Vector                       # Scalar <- Vector: v acting through the metric
vector = metric.solve(covector)             # Vector: v again
polar = (Plane & Point).solve(form)         # Plane <- Point: the same form, paired without the metric
```

The last line differs from the one before it only in the pairing. Solving against the metric
keeps the type; solving against the regressive pairing identifies a space with its complement
and changes the type. Contractions split the same way. `trace(slot)` pairs an output with an
input by matching blades and does not use the metric. A contraction between two inputs
requires one of the two pairings, and the pairing has to be written.

A slot is typed by its subspace, and the dual of a subspace is either the subspace itself,
under `|`, or its complement, under `&`. In a Euclidean orthonormal algebra the two coincide
up to signs. In spacetime the signs differ. In a projective algebra the metric identification
does not exist for the null direction, and only the complement remains.

For comparison, in tensor notation one would carry the same distinctions by index position:
a slot is typed as vector or covector by whether its index is up or down; the metric is a
separate symmetric form with two lower indices; raising and lowering apply it explicitly; a
contraction pairs one upper index with one lower. In exterior algebra one would drop the
metric altogether: the wedge, the complement and the regressive product are defined without
it. In geometric algebra the metric is in the product, `(a * b + b * a) / 2 == a | b`, so a
covector is a vector acting through `|`, and raising and lowering happen without notation.
Numga keeps both pictures, by product rather than by index.

## 4. Maps and forms

A map `B <- A` and a form `Scalar <- (A*, A)` hold the same numbers with one slot on the other
side of the arrow. The two directions of conversion are not symmetric.

Map to form is a pairing. Join the output with an open slot of the dual type and the map
becomes a bilinear form, as the [modes example](../examples/mechanics/modes/core.py) does with its
stiffness:

```python
energy = Twist & stiffness                  # Scalar <- (Twist, Twist), from Forque <- Twist
```

Form to map is a solve of that pairing. The pairing form is solved against the given form,
with the unknown in the pairing's first slot:

```python
polarity = (Plane & Point).solve(form)      # Plane <- Point, from Scalar <- (Point, Point)
```

Two pairings are available, and they differ in whether the metric is involved. The regressive
product pairs a space with its complement, so the output of the resulting map is in the dual
space: a plane for a point, a forque for a twist. The inner product `|` uses the metric and
keeps the output in the same space:

```python
ricci_map = (Vector | Vector).solve(ricci_form)   # Vector <- Vector, the form with one index raised
```

Numga does not track index placement; it tracks slot types, and the pairing that is solved
determines the type of the output.

Which to use follows from the operations each supports without conversion. Maps compose,
invert, and transport by sandwich. Forms add, are differentiated as costs, are solved against
linear forms, and are the object of eigenproblems, including the generalized eigenproblem
between two forms, `potential.eigh(kinetic)`, which uses no metric because both sides are
forms. A form's eigenproblem on its own is relative to its slot's metric, the inner product of
`S` with its reverse: `alignment.eigh()` measures rotors, and `misfit.eig()` measures a motor by
its rotor part alone, sending the translations to infinity. In tensor notation one would write a map as a (1,1) tensor and a form as a (0,2)
tensor; the difference is one pairing.

The distinction matters more than its size suggests. In coordinates a map and a form are the
same square array, and nothing in the array says which one it is. Here the type says it, and
the eigenproblems follow the type. A map has eigenvalues only when its output is the same kind
of thing as its input: a twist that maps to a multiple of itself means nothing when twists come
out as forques. A form has none on its own terms; its eigenproblem is posed against a second
form, `potential.eigh(kinetic)`, and has real eigenvalues because both forms are symmetric.

Flattening the distinction produces a familiar pathology. A rigid body's stiffness assembled as
a map from twists to forques comes out almost symmetric: symmetric in one basis, not quite in
another, for no reason anyone can name. The usual remedies treat the symptom. Averaging with
the transpose discards information, and the normal equations are exactly symmetric but square
the condition number. What is missing is one regressive product, pairing the returned forque
with a twist. Each spring's dyad then contributes `(a & line) * (b & line)`, symmetric in `a`
and `b` by construction. The asymmetry was never in the physics; it was a map standing in for
a form, and a typed algebra does not let one pass for the other.

## 5. Quadrics

The [multiview example](../examples/geometry/multiview/core.py) holds a pixel's precision disc and
its sight cone as polarity maps, `Plane <- Point`, the map that sends a point to its polar
plane. Section 8 holds a mass cloud's second moment as a dual quadric, `Point <- Plane`, the
pole of a plane, which is the inverse of the polarity when the quadric is nondegenerate. Either
one paired with an open slot is the form, `Scalar <- (Point, Point)`, which is what a cost or
an eigenproblem takes. The conversions are those of section 4:

```python
form = quadric & Point                      # Scalar <- (Point, Point)
quadric = (Plane & Point).solve(form)       # Plane <- Point
pole = quadric.inverse()                    # Point <- Plane
value = quadric(p) & p                      # the quadric at a point: its polar joined with the point
```

The regressive product of a plane and a point commutes in two dimensions and anticommutes in
three:

```python
plane & point == point & plane              # PGA2D
plane & point == -(point & plane)           # PGA3D
```

The value of a quadric built as a dyad is the product of two such pairings, one inside the
dyad and one at the evaluation. When both are taken in the same order the product is a square,
and the sign never appears; when the orders differ the two pairings do not cancel in three
dimensions:

```python
disc = normal * (normal & Point)            # value  disc(q) & p == (normal & q) * (normal & p): balanced
disc = normal * (Point & normal)            # value  disc(q) & p == (q & normal) * (normal & p): a sign in 3D
```

Plane-first everywhere is the simplest way to keep every pairing in a file in the same order.

Which to keep depends on the use. Maps to compose, invert, or move by sandwich; forms to
sum, to differentiate as a cost, or to solve. The [multiview example](../examples/geometry/multiview/core.py) keeps its sight cones as
polarity maps, moves them with `pose >> cone(pose << Point)`, sums them, and converts to a form
where a form is required.

## 6. Inverses

A multivector is inverted under the geometric product, a map under composition, and a form is
solved against a linear form. In each case the inverse is the element that composes to the
identity in its own product, and the same method name serves all three.

A batch of vectors is not a frame; it is a set of independent copies.

```python
basis = mv.vector(np.eye(4))                # [4] Vector: the coordinate vectors, as a batch
inverses = basis.inverse()                  # [4] Vector: each vector inverted on its own
```

The second line equals the reciprocal frame for an orthonormal basis and not otherwise. A
reciprocal frame is the inverse of the frame as a map from coefficients to vectors, and the
library has no coefficient-space slot to hold that map. Where a reciprocal is required it
comes from solving the pairing; in most cases a frame summed against its reciprocal is the
coordinate spelling of a trace or of an identity map, which have frame-free forms.

## 7. Least squares

`solve` and `lstsq` dispatch on the shape of the problem. There are four cases.

Nullary: both sides are multivectors, and the value has a trailing batch axis. The unknowns
are the coefficients of a linear combination along that axis:

```python
coefficients = vectors.lstsq(target)         # [n] Scalar: sum(coefficients * vectors) ≈ target
```

Unary: a map against a right-hand side, solved exactly or with the pseudoinverse's cutoff.
Input slots of the right-hand side are kept as input slots of the solution. Triangulation in
the [multiview example](../examples/geometry/multiview/core.py) is this case, a polarity map solved against the plane at
infinity:

```python
points = (fused + w * (w & Point)).solve(w)                # Point, the vertex of a fused cone
```

Forms: `form(x, ·) = linear(·)`, with `x` in the form's first slot. A right-hand side with
leading slots yields a map on them. A Schur complement and an induced map are both written
this way:

```python
step = curvature.solve(-gradient)                          # Twist, from Scalar <- (Twist, Twist) and Scalar <- Twist
response = h_pt[:, None].solve(h_cross)                    # Direction <- Twist, from Scalar <- (Twist, Direction)
```

Tensor: a construction with several open slots, solved for an unknown multilinear part. The
right-hand side's inputs must match one subsequence of the construction's inputs, and the
unmatched slots become the solution's signature. Recovering a point cloud's second moment
from its inertia is this case; section 8 shows it.

## 8. Second moments and inertia

A mass cloud has a second-moment quadric and an inertia map. They hold the same information:
the moment is a dual quadric on planes, the inertia a map from twists to forques.

```python
moment = (points * (Plane & points) * masses).sum(axis=0)              # Point <- Plane
inertia = ((points & points.commutator(Bivector)) * masses).sum(axis=0) # Forque <- Twist
```

Moment to inertia goes through principal points. Diagonalize the plane metric against the
moment as a form on planes. The generalized eigensolve normalizes the eigenplanes to unit
moment, so the moment's images of those planes are four principal points whose momentum
dyads sum directly to the inertia of the cloud:

```python
values, planes = (Plane | Plane).eigh(Plane & moment)                   # [4] Scalar, [4] Plane
principal = moment(planes)                                              # [4] Point
inertia = (principal & principal.commutator(Bivector)).sum(axis=0)      # Forque <- Twist
```

Inertia to moment, as in the [inertia example](../examples/mechanics/inertia.py), is the tensor
solve of section 7. Write the construction that turns a
second moment into an inertia with every slot open, and solve it for the unknown map:

```python
construction = Point & Plane.dual().commutator(Bivector)                # Forque <- (Point, Plane, Twist)
moment = construction.lstsq(inertia)                                    # Point <- Plane
```

The two are also related through a trace, but that spelling requires the Euclidean metric and
a frame, so it is not the geometric one; for comparison, in matrix notation one would write it
as S = ½ tr(J) 1 − J, with S the spatial second-moment matrix and J the inertia tensor. The
trace the inertia map does have is its own:

```python
inertia.trace()                             # 0: momentum has no component along its own screw
```

Its kinetic energy form, `Twist & inertia`, has a trace only against a metric on twists, and
the twist metric of PGA is singular: the translations carry no unit of their own.

## 9. Traces

`trace(slot)` contracts a map's output against one of its inputs by matching blades and drops
that slot. It does not use the metric: an output blade and the same input blade are dual to
each other by construction. Slots are numbered in order of appearance in the expression.

A contraction between two inputs is different: either a metric pairs them or the complement
does, and one of the two has to be written. The Ricci contraction shows both. The curvature of
a plane gravitational wave in the [curvature example](../examples/relativity/curvature/core.py) is
`plus`, a map on bivectors. Wedge an open vector into it,
contract with another open vector, and trace the output against the wedge slot:

```python
ricci = Vector.commutator(plus(Vector.wedge(Vector))).trace(slot=1)     # Scalar <- (Vector, Vector)
```

The inner product with the open vector is the one place the metric enters. The trace is a
blade-matching contraction. No frame and no reciprocal basis appear, and the result is a form,
which is what Ricci is. The first Bianchi identity reads the same way, with the third vector
left open so that the cyclic sum is a map that must vanish. A sum over a basis against its
reciprocal is the coordinate spelling of this trace.

A form's own `trace()` pairs its two slots through the slot's metric: it is the trace of the
form with one slot raised, `(S | S).solve(form)` on vectors, and it exists only where that
metric is invertible.

## 10. Homogeneous unknowns

Motors, points and planes are homogeneous: a scalar multiple is the same geometric object, so
their coordinates contain a direction that is not a degree of freedom. A solve that treats such
an object's coordinates as unknowns will use that direction, and the result is either a step
that changes nothing or one that absorbs everything. The rule is that a homogeneous object is
never updated in its own coordinates. It is updated by an element of its tangent space, which
is not homogeneous, and the object is reconstituted from it.

For a motor the tangent space is the twists, and the update is an exponential:

```python
motors = motors * (step * 0.5).exp()                        # step: Twist, the Newton step of section 7
```

For a point it is the directions, the ideal points, and the update is a sum. A Newton step
over points is therefore taken over directions. Taken over full points instead, the solve
moves the points along their scale, and in the Schur complement of the [multiview example](../examples/geometry/multiview/core.py) it
absorbs the whole camera step: the complement is exactly zero and the cameras appear
unobservable.

```python
moved = motors << Direction                                 # Direction <- Direction: a world displacement into a camera
h_pt = (cones(moved) & moved).sum(axis=1)                   # Scalar <- (Direction, Direction), full rank
```

The one place a homogeneous object is itself the unknown is a linear solve for it, such as
triangulation, where the fused cone's vertex is sought as a point. There the scale direction is
pinned with a dyad on the weight, which turns the vertex into the pole of the plane at
infinity without changing the spatial gradient, and the result is normalized afterwards:

```python
points = (fused + w * (w & Point)).solve(w).normalized()
```

The global gauge of a problem, such as the frame of a camera rig or the scale of a two-camera
rig, is the same fact one level up: directions in the unknowns that change nothing. Anchoring
removes them from the solve; it does not make them observable.

## 11. Information and covariance

The curvature of a cost over twists is a form, `Scalar <- (Twist, Twist)`, and it is the
information on the pose. Its inverse on readouts is the covariance. A covariance is a map from
a readout to the twist correlated with it, and a linear readout of a twist is a line, so the
covariance is `Twist <- Line`. It moves like any map, as in the [Kalman example](../examples/geometry/kalman/core.py):
pull the readout through the step, push the twist back:

```python
sigma = step << sigma(step >> Line) + Q                     # prediction
gain = sigma((sigma + R).inverse())                         # Twist <- Twist
```

Readouts go through the pairing. The variance of the position along a line is that line's
readout, pulled back through the position Jacobian, paired with the covariance of itself:

```python
readout = (Line & Twist).solve(Line & shift)                # Line <- Line
position = readout & sigma(readout)                         # Scalar <- (Line, Line)
```

Sampling diagonalizes the readout form: each eigen-readout's twist, scaled by its standard
deviation, carries one unit normal draw. A Schur complement, as in the [multiview example](../examples/geometry/multiview/core.py), is marginalization: the reduced
curvature over the cameras is the information on their poses with the points integrated out.

## 12. Batch axes and slots

A batch axis indexes independent copies of an expression; a slot is an argument. The two look
alike in a coefficient array, and a frame is where they get confused: a basis stored as a
batch, with a reciprocal basis stored as another, is a slot that has been evaluated on each
basis vector and summed. The Ricci contraction of the [curvature example](../examples/relativity/curvature/core.py),
written both ways, makes this concrete.

```python
basis = mv.vector(np.eye(4))                            # [4] Vector: four copies, one per coordinate vector
ribbons = basis.wedge(Vector)                           # [4] Bivector <- Vector: a batch of maps, one slot each
sheets = Vector.wedge(Vector)                           # [] Bivector <- (Vector, Vector): one map, two slots

# the frame spelling: evaluate one slot on the basis, pair with the reciprocal, sum the batch
ricci_frame = plus(basis.wedge(Vector)).commutator(basis.inverse()).sum(axis=0)     # Vector <- Vector

# the slot spelling: keep both slots open, contract them with a trace, raise with the metric
ricci_form = Vector.commutator(plus(Vector.wedge(Vector))).trace(slot=1)            # Scalar <- (Vector, Vector)
ricci = (Vector | Vector).solve(ricci_form)                                         # Vector <- Vector, equal to ricci_frame
```

The batch in the first spelling stands in for the slot that the trace contracts in the second,
and the reciprocal basis stands in for the metric that the solve applies. The frame spelling
also relies on `basis.inverse()` being the reciprocal frame, which holds for an orthonormal
basis and not otherwise. Anything summed over a basis should be checked for a frame-free
spelling first: a trace, a bare type standing for the identity, or a solve of the pairing.

Under the hood, numga's default dense backend evaluates slot contractions over dense maps akin
to a matrix product. This is strictly an implementation detail: extensor semantics are
coordinate-free subspace contractions, and alternative execution strategies—such as sparse
kernel contraction or symbolic unrolling—follow the exact same algebraic rules without changing
the interface.

## 13. Outermorphisms

A map on vectors extends to every grade by mapping each factor of a product. The product is
the one that multiplies the map's own space: `^` for vectors, `&` for antivectors. So a vector
map raises grade and a point map lowers it, with the same construction and no metric:

```python
t.outermorphism(Bivector)(a ^ b) == t(a) ^ t(b)         # t: Vector <- Vector
T.outermorphism(Line)(p & q) == T(p) & T(q)             # T: Point <- Point, in PGA3D
T.outermorphism(Plane)(p & q & r) == T(p) & T(q) & T(r)
t.outermorphism(Pseudoscalar)(I) == t.det() * I         # the top grade is the determinant
t(s).outermorphism(Bivector) == t.outermorphism(Bivector)(s.outermorphism(Bivector))
```

A point map moves lines and planes this way even when it is singular, such as a camera's
central projection, where no inverse exists to pull planes back through. Building the extension
costs a product per basis blade of the grade; applying it is a single map call.

This is the extension operator of A. M. Moya, V. V. Fernández and W. A. Rodrigues Jr.,
[Extensors in Geometric Algebras](https://arxiv.org/abs/math/0501558).
