# Extensor Syntax & Methods Reference

In numga, the `Extensor` abstraction unifies multivectors and linear/multilinear maps under a single value class:
* **Arity 0**: Multivectors (concrete geometric elements).
* **Arity 1**: Unary linear maps (`Output <- Input`).
* **Arity n**: Multilinear maps (bilinear forms, energy functionals, and maps with more slots).

---

## 1. The Core Viewpoint: Multivectors Waiting for Arguments

A GA expression with open slots reads as an operation on its **output type**. An extensor behaves as a multivector of its output type with one or more arguments still to be supplied.

When constructing a projection:
```python
# Reads: join light with an open point (Line), meet with ground plane (Point):
shadow = (light & Point) ^ ground                     # [] Point <- Point

# The result behaves as a Point, waiting for an object vertex to be supplied:
vertex = mv.point([1.0, 0.0, 2.0])                   # [] Point
projected_vertex = shadow(vertex)                    # [] Point
```

Because every GA operation treats an open extensor by its output type, several stages chain without index bookkeeping:

```python
# 1. Wedge with k produces a field Bivector (waiting for a spatial polarization a):
field = k.wedge(Spatial)                             # [n_speeds] Bivector <- Spatial

# 2. Material response transforms field Bivector into excitation Bivector:
excitation = medium(field)                           # [n_speeds] Bivector <- Spatial

# 3. Commutator with k dots the excitation Bivector with k, yielding a spacetime Vector:
wave_map = k.commutator(excitation)                  # [n_speeds] Vector <- Spatial
```
Each step reads as geometric algebra on the output type, and numga composes the linear maps.

---

## 2. Construction via Open Slots & GATypes

Passing an unbound GAType (`Vector`, `Point`, `Twist`, `Plane`, etc.) to any GA operation constructs a compiled extensor with open input slots.

### Open Slot Expressions
```python
# Unary linear map (Arity 1): Forque <- Point
line_map = mv.x & Point

# Binary multilinear map (Arity 2): Line <- (Point, Point)
join_map = Point & Point

# Lie bracket velocity map: Point <- Twist
velocity_map = body.commutator(Twist)
```

### Identity Maps and Dyads
A bare type is the identity map on its subspace, so projectors and their complements are plain arithmetic:
```python
electric = Bivector.commutator(t).wedge(t)           # [] Bivector <- Bivector (projector)
magnetic = Bivector - electric                       # [] Bivector <- Bivector (complement)
```
A multivector times a linear form is a rank-one dyad. Sums of dyads build stiffness, curvature, sensor precision and quadric error metrics:
```python
spring = lines * (Twist & lines) * spring_constants  # [n_springs] Forque <- Twist
plus = nx * (nx | Bivector) - ny * (ny | Bivector)   # [] Bivector <- Bivector
```

### Direct Numerical Construction
When a transformation is defined by numerical coefficients (e.g. principal axes scaling):
```python
# Anisotropic scaling along x, y, z axes:
PointMap = PGA3D.gatype((Point, Point))              # Point <- Point
scale = ctx.extensor(PointMap, np.diag([sx, sy, sz, 1.0]))
```

### Structural Axis Order
Extensor axes are **output-first**, after any leading batch dimensions:
```text
Arity 0 (Multivector):       [batch..., Output]
Arity 1 (Unary Linear Map):  [batch..., Output, Input]
Arity 2 (Bilinear Form):     [batch..., Output, Input1, Input2]
```
For an arity-1 extensor, the trailing kernel axes correspond directly to output and input blade dimensions: `kernel[..., out, in]`.

---

## 3. Binding & Composition Syntax

### Concrete Binding (`__call__`)
Calling an extensor with concrete multivector operands (arity 0) binds its logical input slots:

```python
# 1. Unary linear map:
shadow = (light & Point) ^ ground                    # Point <- Point
vertex = mv.point([1.0, 0.0, 2.0])                  # Point (Arity 0)

# Full binding yields a concrete multivector (Arity 0):
projected_vertex = shadow(vertex)                    # Point

# 2. Binary multilinear map:
join_map = Point & Point                             # Line <- (Point, Point)
origin = mv.point([0.0, 0.0, 0.0])                   # Point
target = mv.point([1.0, 2.0, 3.0])                   # Point

# Full binding: supplying both inputs yields a Line
line = join_map(origin, target)                      # Line

# Partial binding: supplying one argument yields an Arity-1 extensor
ray_from_origin = join_map(origin)                   # Line <- Point
ray = ray_from_origin(target)                        # Line

# The slot's own type in place of an argument leaves that slot open:
ray_to_target = join_map(Point, target)              # Line <- Point
```

### Argument Lifting (Binding Maps into Slots)
Lifting is binding a map, an extensor with open slots of its own, into an open argument slot. Instead of eliminating the slot, numga contracts the intermediate GAType and inherits the argument's input axes: with `M` a `C <- B` and `T` a `B <- A`, `M(T)` is a `C <- A`, feeding the output of `T` into the input of `M`.

In matrix notation this reads as the product $C = AB$, which feeds the output of $B$ into the input of $A$.

```python
# 1. Chaining & Pipeline Lifting:
# 'medium' expects a Bivector input; 'field' is an extensor (Bivector <- Spatial)
field = k.wedge(Spatial)                             # [n_speeds] Bivector <- Spatial
medium = crystal_medium(eps_x=2.25, eps_y=1.5, ...) # Bivector <- Bivector

# Binding 'field' (arity 1) into 'medium' lifts the material response over the Spatial argument:
excitation = medium(field)                           # [n_speeds] Bivector <- Spatial

# 2. Observer Decomposition Lifting:
# 3D spatial permittivity law:
permittivity = permittivity_tensor(eps_x, eps_y, eps_z) # Vector <- Vector
# Observer extractor maps 6D spacetime bivectors to 3D electric vectors:
electric_extractor = B.commutator(observer)             # Vector <- Bivector

# Passing 'electric_extractor' into 'permittivity' lifts the 3D map into spacetime:
spacetime_d = permittivity(electric_extractor)          # Vector <- Bivector

# 3. Composition Across Kinematic Chains:
# Collapsing camera projection with forward kinematics:
# world_to_pixel : Point <- Point,  bodies_to_world : [5] Point <- Point
local_to_pixel = world_to_pixel(bodies_to_world)        # [5] Point <- Point

# 4. Multilinear Slot Lifting:
# join_map expects (Point, Point); trajectory is an arity-1 extensor (Point <- Time)
join_map = Point & Point                                # Line <- (Point, Point)
moving_ray = join_map(trajectory)                       # Line <- (Time, Point)
```

### Frame Transport & Motor Transformations
Because all GA operations act on an extensor's **output type**, applying a motor sandwich directly transforms the output space:

```python
# Transforms the output space from local to world (WorldOut <- LocalIn):
world_emitter = motor >> local_emitter               # WorldOut <- LocalIn
```
The sandwich `motor >> T` transforms only the output space. That is all that is needed when a local generator or sensor model should emit elements in the world frame.

If an extensor is an endomorphism or physical law whose **input space** must also be expressed in the new frame (e.g. transporting a spatial stiffness, inertia, or quadric metric from local to world coordinates), the input is explicitly pulled back via the inverse sandwich:

```python
# 1. Transforming only the input (LocalIn <- WorldIn):
pulled_T = local_T(motor << In)                      # LocalOut <- WorldIn

# 2. Transforming both output and input (WorldOut <- WorldIn):
world_T = motor >> local_T(motor << In)              # WorldOut <- WorldIn

# A quadric as a polarity map (Plane <- Point) moves like any map:
world_cone = pose >> local_cone(pose << Point)       # Plane <- Point
# The same quadric as a form (Scalar <- (Point, Point)) moves by feeding the frame change into both slots:
world_form = local_form(pose << Point, pose << Point)
```
Incidence of a plane with a point is written plane first, `plane & point`, and a quadric's value is `quadric(p) & p`. The order is a convention: the regressive product of a plane and a point changes sign with the dimension, so keep the plane on the left, in dyads too (`normal * (normal & Point)`). A polarity map becomes a form by `quadric & Point`, and a form becomes a polarity map by solving the pairing, `(Plane & Point).solve(form)`.

A map on points induces a map on planes through the same pairing, without any transpose. It satisfies `induced(l) & p == l & T(p)` for every plane and point, even when `T` is singular, and carries a quadric's polar planes back through `T`:
```python
on_planes = (Plane & Point).solve(Plane & projection)   # Plane <- Plane
cone = on_planes(disc(projection))                     # Plane <- Point
```

### Batching and Broadcasting
Batch axes lead the structural axes and are not slots: they index independent copies of an expression. `stack` creates a batch axis, `.sum(axis)` removes one, and indexing follows NumPy:
```python
waves = stack((plus_wave, cross_wave, plus_wave + cross_wave), axis=1)   # [n_time, 3] Bivector <- Bivector
stiffness = spring_stiffness.sum(axis=0)                                  # [] Forque <- Twist
```
Calling a batched map on a batched argument broadcasts the batch axes against each other:
```python
acceleration = response[:, :, None](reference)      # [n_time, 3, n_beads] Vector
```
A frame summed against its reciprocal is the coordinate spelling of a trace; write the trace instead (see `.trace(slot)` below).

---

## 4. Extension Methods by Structural Type

### Unary Linear Maps (`Output <- Input`)
Operate directly on linear transformations while preserving input/output GATypes:

* **`.solve(rhs)`**: Solves `T(x) == rhs` for `x`: the inverse of composing into the map's input, so `T.solve(T(x)) == x`. A right-hand side with inputs of its own keeps them, so `T.solve(T(Y)) == Y` for a map `Y` too.
  ```python
  step = stiffness.solve(force)                      # Twist: the displacement the force causes
  ```
* **`.lstsq(rhs)`**: Least-squares solve for over- or under-determined linear systems. On a construction with several input slots it unbinds every slot the right-hand side does not match, all at once, and returns the unknown map on them:
  ```python
  moment = (Point & Plane.dual().commutator(Bivector)).lstsq(inertia)   # Point <- Plane, from Forque <- (Point, Plane, Twist)
  ```
* **`.inverse()`**: Inverse of the map under composition. On a multivector batch the same method is the geometric-product inverse of each element; a batch of vectors is not a frame, so this is not a reciprocal frame.
* **`.pinv(rcond=1e-4)`**: Moore-Penrose pseudoinverse (e.g. converting Gauss-Newton curvature into posterior pose covariance).
  ```python
  pose_covariance = curvature.pinv()                 # Twist <- Twist
  ```
* **`.det()`**: Determinant of a square endomorphism (`Space <- Space`).
* **`.trace(slot=0)`**: Contracts the output against one input slot by matching blades and drops that slot; it does not use the metric. The slot must be the output's own space: a slot spanning only part of the output is refused, since tracing it would choose a complement by blade label. Slots are numbered in order of appearance in the expression. On `Space <- Space` this is the matrix trace; on a multilinear map it lowers the arity by one:
  ```python
  ricci = Vector.commutator(R(Vector.wedge(Vector))).trace(slot=1)   # [] Scalar <- (Vector, Vector)
  ```
* **`.svd()`**: Singular value decomposition returning `[U, s, Vh]`.
  * `s`: singular values array.
  * `Vh[-1]`: physical polarization / nullspace eigenmode.
* **`.svdvals()`**: Only the singular values, over all batch dimensions, as in a resonance scan:
  ```python
  resonance_curves = wave_map.svdvals()[..., -1]     # Smallest singular value per speed
  ```
* **`.eig()` / `.eigvals()`**: Spectrum of a general (non-symmetric) endomorphism; eigenvalues are complex `[n] Scalar`.
  ```python
  eigenvalues = plus.eigvals()                       # [6] Scalar (complex; all zero for a nilpotent map)
  ```
* **`.cholesky()`**: Cholesky factorization of a positive-definite extensor.

  In matrix notation it reads as the lower-triangular factor $L$ of $L L^\top$.
* **`.decompose_polar()`**: Decomposes an extensor into unitary rotation and symmetric stretch components.

---

### Bilinear & Quadratic Forms (`Scalar <- (Space, Space)`)
Represent metrics, potential/kinetic energy functionals, quadrics, and alignment objectives:

A form has two covector slots, so its eigenvalues, determinant and trace exist relative to a metric form. Without one, the metric is the slot's own: the inner product of `S` with its reverse, which is `V | V` on vectors and positive on bivectors and Euclidean points. Its kind follows from the slot type: an identity metric (Euclidean points, rotors) keeps the plain solver, other metrics use the generalized pencil.

* **`.eigh()`**: Symmetric/Hermitian eigensolve against the slot's metric, which must be positive definite; a singular or indefinite slot metric (PGA points and motors, spacetime vectors) is refused with a `TypeError`. Returns eigenvalues `[n] Scalar` and eigenvectors `[n] Space`.
  ```python
  values, rotors = alignment.eigh()                  # Scalar <- (Rotor, Rotor): the rotor metric is the identity
  ```
* **`.eig()` / `.eigvals()`**: General eigensolve against the slot's metric. A singular metric sends the modes it does not measure to infinity; the eigenpairs of a symmetric pencil are real, and `.real()` keeps them so:
  ```python
  values, motors = misfit.eig()                      # Scalar <- (Motor, Motor): translations at infinity
  motor = motors[values.real().argmin()].real().normalized()
  ```
* **`.eigh(metric)`**: **Generalized Hermitian eigensolve**, one form against another.
  Solves the generalized eigenvalue problem directly between two bilinear energy forms, without inverting the inertia:
  ```python
  pe_form = Twist & stiffness                        # Scalar <- (Twist, Twist)
  ke_form = Twist & inertia                          # Scalar <- (Twist, Twist)
  values, modes = pe_form.eigh(ke_form)              # values: [3] Scalar, modes: [3] Twist
  ```

  In matrix notation it reads as $K v = \lambda M v$, solved without forming the asymmetric product $M^{-1}K$.
* **`.eigvalsh()`** / **`.eigvalsh(metric)`**: Only the eigenvalues, against the slot's metric or a given one.
* **`.det()`** / **`.det(metric)`**: Determinant of the form relative to the metric; the slot's metric must be invertible.

  In matrix notation it reads as $\det(G^{-1} A)$, with $G$ the metric and $A$ the form.
* **`.trace()`**: Trace of the form with one slot raised by the slot's metric, `(S | S).solve(form).trace()` for vectors; the metric must be invertible.
* A form has no singular values: its coefficient matrix changes with the basis. To see that a form vanishes, evaluate it: `ricci(a, b)`.
* **`.solve(linear)`** / **`.lstsq(linear, rcond)`**: Solve `form(x, ·) == linear(·)` for `x` in the form's first slot, where `linear` is `Scalar <- Space`: the inverse of binding that slot, so `form.solve(form(x)) == x`. Only the first slot is solved for. A right-hand side with leading slots yields a map on them, which is how a Schur complement or an induced map is written:
  ```python
  points = (splats + (w & Point) * (w & Point)).solve(w & Point)   # [n] Point: the fused cone's vertex
  response = h_pt.lstsq(h_cross, rcond=1e-4)                      # Point <- Twist, from Scalar <- (Twist, Point)
  ```
* **Squares without a metric**: the curvature of a quadric cost is the Jacobian's polar joined with the Jacobian, `cones(motion) & motion`; a scalar residual's normal equations are the product of its Jacobian form with itself, `(j * j).sum(axis=0)`.

---

### Multilinear Maps (`Output <- (In1, ..., Inn)`)
Arity is not limited to forms; any output type is allowed and slots are numbered in order of appearance:
```python
A = Vector.commutator(R(Vector.wedge(Vector)))       # [] Vector <- (Vector, Vector, Vector)
```
Partial calls fill slots in order, the slot's own type in place of an argument leaves it open, `.trace(slot)` lowers the arity by one, and batch axes broadcast as for unary maps.

---

### Nullary Extensors (Arity 0: Multivectors & Scalars)

Nullary extensors represent concrete multivectors and scalars (carrying no open input slots). Methods on nullary extensors transform or inspect concrete geometric values rather than contracting slots:

* **Scalars (`Scalar`)**: Eigensolvers (`.eigh()`), singular values of maps (`.svdvals()`), and quadratic forms return scalar extensors. Mathematical operations, reductions, and conversions chain directly as methods without unwrapping:
  ```python
  # Eigensolve post-processing: clamp, root, and frequency conversion:
  frequencies = eigenvalues.clip(0, np.inf).square_root() / (2 * np.pi)

  # Batch reductions and backend array extraction for plotting:
  best_speed = speeds[singular_values[..., -1].argmin()]
  raw_array = frequencies.to_array()     # Returns backend array matching batch shape
  ```
  Standard elementwise operations (`.abs()`, `.sin()`, `.cos()`, `.isnan()`, `.isfinite()`) execute over batch axes while preserving extensor typing.

* **`.real()`**: The real part of every coefficient, in a real context, for any extensor. It is for results known to be real that a general eigensolve returns as complex, such as the eigenpairs of a symmetric pencil.

* **Bivectors & Motors (`Bivector`, `Motor`, `Rotor`)**: Lie algebra generators and versors:
  * `bivector.exp()` / `motor.log()`: Lie exponential and logarithm between velocity generators and finite motors.
  * `motor.motor_split()`: Decomposes a motor into translator and rotor components (`motor_translator()`, `motor_rotor()`).
  * `motor.normalized()` / `mv.norm()`: Gauge normalization, to `motor * motor.reverse() == 1`, and Study/geometric norms.

* **`.inverse()`**: On an arity-0 multivector, `.inverse()` is the Clifford / geometric product inverse, `x * x.inverse() == 1`. On an arity-1 extensor, it is the operator inverse under map composition: `T.inverse()(T)` is the identity map.
