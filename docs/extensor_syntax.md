# Extensor syntax and methods

This document is the reference for building, calling and solving extensors in numga. Multivectors, linear maps and multilinear maps share a single representation, distinguished by how many input spaces they take:

* **Arity 0**: multivectors and scalars (`Output`).
* **Arity 1**: unary linear maps (`Output <- Input`).
* **Arity n**: multilinear maps and forms (`Output <- (Input1, ..., Inputn)`).

Every algebraic operation acts on an extensor's output type and keeps its inputs, so maps are built with the same products as multivectors. Capitalized names such as `Point` and `Potential` are types acting as identity maps on their spaces, while lowercase names are concrete values. [Conventions](conventions.md) states the defining identities, and [Orientation in numga](orientation.md) covers `+` and `-` orientation propagation.

```python
# Sending each point along its ray from the light to the ground plane builds a projection map.
shadow = (light & Point) ^ ground                            # [] Point- <- Point-
vertex = mv.point([1.0, 0.0, 2.0])                           # [] Point-
projected = shadow(vertex)                                   # [] Point-
```

Chaining products and map calls composes stages without index bookkeeping:

```python
# Potential to electromagnetic field, through the medium to excitation, and wedged with k to current residual.
field = k ^ Potential                                        # [speeds] Bivector+ <- Vector+
excitation = medium(field)                                   # [speeds] Antibivector- <- Vector+
wave = k ^ excitation                                        # [speeds] Antivector- <- Vector+
```

## Maps from products and GATypes

### Products as maps

A `GAType` (`Vector`, `Point`, `Twist`, `Plane`) is the identity map on its space. Products containing one or more `GAType` operands build linear or multilinear maps:

```python
ray_from_origin = origin & Point                             # [] Line- <- Point-
join_map = Point & Point                                     # [] Line- <- (Point-, Point-)
velocity_map = body.commutator(Twist)                        # [] Point- <- Twist+
```

### Identity maps and dyads

Because a bare `GAType` is the identity map on its subspace, projectors and their complements are written as arithmetic on types, `Bivector.reverse()` is the reversion map on bivectors, and derived types live under `.derive`:

```python
# Spacetime observer split of bivectors into electric and magnetic parts.
electric = Bivector.commutator(t) ^ t                        # [] Bivector+ <- Bivector+
magnetic = Bivector - electric                               # [] Bivector+ <- Bivector+
```

Multiplying a multivector by a scalar-valued linear form builds a rank-one dyad. Sums of dyads build stiffness, inertia, constitutive media and quadric error metrics:

```python
spring = lines * (Twist & lines) * spring_constants          # [springs] Forque- <- Twist+
plus = nx * (nx | Bivector) - ny * (ny | Bivector)           # [] Bivector+ <- Bivector+
```

### Numerical construction and axis order

When a map is given directly by numerical coefficients in a basis, `ctx.extensor(gatype, array)` wraps the coefficient array with its map type:

```python
PointMap = PGA3D.gatype((Point, Point))                      # Point- <- Point-
scale = ctx.extensor(PointMap, np.diag([sx, sy, sz, 1.0]))   # [] Point- <- Point-
```

Structural axes follow output-first order after any leading batch axes:

```text
Arity 0 (multivector):       [batch..., Output]
Arity 1 (unary map):         [batch..., Output, Input]
Arity 2 (bilinear map/form): [batch..., Output, Input1, Input2]
```

For an arity-1 extensor, the trailing kernel axes match the output and input blade dimensions `kernel[..., out, in]`.

## Binding, composition, frames and batches

### Calling and partial binding

Calling an extensor on multivectors binds its inputs in order. Supplying fewer arguments than the arity binds the leading inputs and returns a map on the remaining inputs, while passing a `GAType` passes its identity map or restricts that input's domain to a subtype:

```python
join_map = Point & Point                                     # [] Line- <- (Point-, Point-)
origin = mv.point([0.0, 0.0, 0.0])                           # [] Point-
target = mv.point([1.0, 2.0, 3.0])                           # [] Point-

line = join_map(origin, target)                              # [] Line-
ray_from_origin = join_map(origin)                           # [] Line- <- Point-
# Passing Point as the first argument binds only the second input.
ray_to_target = join_map(Point, target)                      # [] Line- <- Point-
# Passing a subtype restricts the map's input domain.
electric_response = medium(Electric)                         # [] Antibivector- <- Electric+
```

### Composition

Calling a map `M` (`C <- B`) on another map `T` (`B <- A`) composes them into `M(T)` (`C <- A`), feeding `T`'s output into `M`'s input:

```python
# Material response composed with the potential-to-field map.
field = k ^ Potential                                        # [speeds] Bivector+ <- Vector+
excitation = medium(field)                                   # [speeds] Antibivector- <- Vector+

# Camera projection composed across a batch of body placements.
local_to_pixel = world_to_pixel(bodies_to_world)             # [bodies] Point- <- Point-

# A trajectory (Point- <- Time+) composed into the first input of join_map.
moving_ray = join_map(trajectory)                            # [] Line- <- (Time+, Point-)
```

In matrix notation this reads as the product $C = AB$, feeding the output of $B$ into the input of $A$.

### Frame transport

Because algebraic operations act on an extensor's output, `motor >> T` transforms only the output space, while `T(motor << In)` pulls back the input space:

```python
world_emitter = motor >> local_emitter                       # [] WorldOut <- LocalIn
pulled_T = local_T(motor << In)                              # [] LocalOut <- WorldIn
world_T = motor >> local_T(motor << In)                      # [] WorldOut <- WorldIn

# A quadric as a polarity map transforms on both output and input; as a form it transforms on both inputs.
world_cone = pose >> local_cone(pose << Point)               # [] Plane+ <- Point-
world_form = local_form(pose << Point, pose << Point)        # [] Scalar+ <- (Point-, Point-)
```

An extensor built from transformed elements equals the extensor built in the reference frame and transported by `motor >> T(motor << In)`.

### Batching and broadcasting

Batch axes lead the structural axes and index independent copies of an expression. `stack` creates a batch axis, `.sum(axis)` reduces one, indexing follows NumPy, and calling a batched map on a batched argument broadcasts their batch shapes:

```python
waves = stack((plus_wave, cross_wave, plus_wave + cross_wave), axis=1)  # [times, 3] Bivector+ <- Bivector+
stiffness = spring_stiffness.sum(axis=0)                                # [] Forque- <- Twist+
acceleration = response[:, :, None](reference)                          # [times, 3, beads] Vector+
```

When sites are coupled rather than independent (such as surface vertices or chain links), a field axis (`Type[sites]`, toggled by `.field()` and `.batch()`) puts site connectivity inside the type so dense or sparse field maps (`Out[rows] <- In[cols]`) sum over paired sites when called or solved. See [Fields and field maps](fields.md).

## Extension methods

### Multivectors and scalars

* **Elementwise scalar math and predicates**: `.abs()`, `.clip()`, `.square_root()`, `.inverse_square_root()`, `.exp()`, `.log()`, trigonometric and hyperbolic functions (`.sin()`, `.cos()`, `.tan()`, `.arcsin()`, `.arccos()`, `.arctan()`, `.sinh()`, `.cosh()`, `.tanh()`, `.arcsinh()`, `.arccosh()`, `.arctanh()`), and predicates (`.isnan()`, `.isfinite()`, `.isinf()`) chain directly on `Scalar` extensors. Predicates and comparisons (`<`, `<=`, `>`, `>=`) return backend boolean arrays directly.
* **Scalar reductions and array readout**: `.min()`, `.max()`, `.argmin()`, `.argmax()`, `.argsort()` and `.to_array()` operate on `Scalar` extensors, with index reductions returning backend integer arrays directly:
  ```python
  frequencies = eigenvalues.clip(0, np.inf).square_root() / (2 * np.pi)  # [modes] Scalar+
  best_speed = speeds[residuals.argmin()]                                # [] float
  raw_array = frequencies.to_array()                                     # [modes] ndarray
  ```
* **`.real()`**: takes the real part of every coefficient in a real context, on an extensor of any arity, for results of `.eig()` known to be real.
* **Bivectors, rotors and motors**:
  * `bivector.exp()` / `motor.log()`: exponential and logarithm between generators and versors, with low-order approximations `.exp_linear()`, `.exp_linear_normalized()`, `.exp_quadratic()` / `.exp_cayley()`, `.log_linear()`, `.log_linear_normalized()`, `.log_quadratic()` and `.log_pade()`.
  * `bivector.exp_derivative()`: right-trivialized derivative of `exp` at `bivector`, a map `Bivector+ <- Bivector+` satisfying `(b + db * h).exp() == b.exp() * (1 + b.exp_derivative()(db) * h)` to first order in `h`; calling `.solve()` on it differentiates `log`.
  * `rotor.square_root()` / `rotor.geometric_mean(other)`: principal rotor or Study square root, and normalized bisector `(rotor + other).normalized()`.
  * `bivector.decompose_polar()` / `bivector.decompose_invariant()`: factors `b == direction * scale` with Study `scale`, and splits a bivector whose square is a Study number into commuting simple bivectors `(b_plus, b_minus)`.
  * `motor.motor_translator()`, `motor.motor_rotor()`, `motor.motor_split(origin)`: splits a motor into translation and rotation about the canonical origin, or about a supplied `origin` antivector (`translator, rotor = motor.motor_split(origin)`).
* **`.dot(other)`**: symmetric metric pairing between two operands of the same orientation (`Scalar+`), using the signature on extrinsic values and dualizing intrinsic values (`a.dual().dot(b.dual())`). On types it builds the metric form:
  ```python
  length_form = Point.dot(Point)                             # [] Scalar+ <- (Point-, Point-)
  ```
* **`.norm()`, `.normalized()`, `.antinorm()`, `.study_norm()`**: `.norm()` takes the root of `x * ~x` (giving a projective point's weight, so `.normalized()` sets unit weight or unit versor normalization `v * ~v == 1`), `.antinorm()` takes the root of `x.dot(x)` (giving a direction's length and a projective point's bulk), and `.study_norm()` computes the norm of a Study number:
  ```python
  length = (head - tail).antinorm()                          # [] Scalar+
  ```
* **`.inverse()` and `.power(n)`**: geometric-product inverse (`x * x.inverse() == 1`, with explicit formulas `.inverse_shirokov()`, `.inverse_factor()`, `.inverse_hitzer()`) and integer power `x ** n`.

### Unary maps

* **`.solve(rhs)`**: solves `T(x) == rhs` for `x`, the inverse of composing into `T`'s input (`T.solve(T(x)) == x`). Any inputs on `rhs` are preserved on the solution (`T.solve(T(Y)) == Y`):
  ```python
  step = stiffness.solve(forque)                             # [] Twist+
  ```
* **`.lstsq(rhs, rcond=1e-15)`**: minimum-coefficient-norm least-squares solve for singular, overdetermined or underdetermined maps. On a multilinear construction it unbinds every input not matched by `rhs` at once and returns the unknown map on them:
  ```python
  # Recovers Point- <- Plane+ from a construction of type Antibivector- <- (Point-, Plane+, Bivector+).
  moment = Point.regressive(Plane.dual().commutator(Bivector)).lstsq(inertia)  # [] Point- <- Plane+
  ```
* **`.inverse()`**: composition inverse of a map between spaces of equal dimension, swapping input and output types (`T.inverse()(T)` is the identity on `T`'s input). On a batch of multivectors the same method computes each element's geometric-product inverse, not a reciprocal frame.
* **`.pinv(rcond=1e-15)`**: Moore–Penrose pseudoinverse, swapping input and output spaces and discarding singular values at or below `rcond` times the largest:
  ```python
  compliance = stiffness.pinv()                              # [] Twist+ <- Forque-
  ```
* **`.adjugate()`**: the map on complementary spaces carried over by the regressive product, with both pairings read extrinsic first (`T.adjugate()(c) & x == c & T(x)` between intrinsic spaces; [Conventions](conventions.md#maps)). It needs no metric and exists for singular maps:
  ```python
  # From Point- <- Twist+ to Line- <- Plane+ in PGA3D.
  readout = shift.adjugate()                                 # [] Line- <- Plane+
  ```
* **`.adjoint()`**: the metric adjoint of Hestenes and Sobczyk, from output space to input space, satisfying `T.adjoint()(b).scalar_product(a) == b.scalar_product(T(a))`. Requires an invertible scalar-product pairing on the input space:
  ```python
  undo = (rotor >> Vector).adjoint()                         # [] Vector+ <- Vector+
  ```
* **`.outermorphism(Type)`**: extends a map between vectors or antivectors to wedge or regressive products of grade `Type` ([Conventions](conventions.md#maps)):
  ```python
  areas = deformation.outermorphism(Patch)                   # [frames] Patch- <- Patch-
  ```
* **`.det()`**: determinant of an endomorphism (`Space <- Space`).
* **`.trace()` / `.trace(first, second)`**: pairs two axes through the regressive product (extrinsic factor first) and drops both. Axis `0` is the output and axes `1..n` are the inputs in order; axis indices may be omitted when exactly two non-scalar axes exist. On `Space <- Space` this equals the matrix trace:
  ```python
  trace = f.trace()                                          # [] Scalar+
  # Equivalent trace after pairing output with Antivector- into a form.
  from_form = (f & Antivector).trace()                       # [] Scalar+
  ricci = Vector.commutator(R(Vector ^ Vector)).trace(0, 2)  # [] Scalar+ <- (Vector+, Vector+)
  incidence_trace = (Plane & g(Point)).trace()               # [] Scalar+
  ```
* **`.contract()` / `.contract(first, second)`**: pairs two axes through the inner product, requiring an invertible metric on the paired input space. On `Space <- Space` the metric and its inverse cancel, matching `.trace()`; contracting products of `Vector` with a vector map gives its divergence and curl:
  ```python
  divergence = (Vector | f(Vector)).contract()               # [] Scalar+
  curl = (Vector ^ f(Vector)).contract(1, 2)                 # [] Bivector+
  ricci = ((Vector ^ Vector) | R(Vector ^ Vector)).contract(1, 3)  # [] Scalar+ <- (Vector+, Vector+)
  ```
* **`.svd()` / `.svdvals()`**: singular value decomposition `(left_vectors, singular_values, right_vectors)` in descending singular-value order, where `left_vectors` are output-space extensors and `right_vectors` are input-space extensors (`V`, not `Vh`), or singular values `[..., k] Scalar+` alone:
  ```python
  residuals = wave.svdvals()[..., -1]                        # [speeds] Scalar+
  _, _, potentials = wave.svd()                              # [speeds, 4] Vector+
  mode = potentials[..., -1]                                 # [speeds] Vector+
  ```
* **`.eig()` / `.eigvals()`** and **`.eigh()` / `.eigvalsh()`**: general or symmetric spectrum of an endomorphism (`Space <- Space`). `.eig()` and `.eigvals()` return complex `[..., n] Scalar+` eigenvalues and `[..., n] Space` eigenvectors:
  ```python
  eigenvalues = (rotor >> Vector).eigvals()                  # [3] Scalar+
  ```
* **`.cholesky()`**: lower-triangular Cholesky factor of a positive-definite endomorphism or square form.

### Bilinear and quadratic forms

A form `Scalar+ <- (Space, Space)` has two inputs, so its spectrum, determinant and contraction are defined relative to a metric form on `Space` (defaulting to `Space.dot(Space)` when omitted).

* **`.eigh()` / `.eigvalsh()`**: symmetric eigensolve against `Space`'s own metric. A positive-definite metric gives one mode per blade; a semidefinite one (such as motors in `PGA3D`, whose metric measures only the rotor blades) gives one finite mode per measured blade with the remaining components chosen to make the form stationary; an indefinite axis metric raises `TypeError`:
  ```python
  values, rotors = alignment.eigh()                          # [4] Scalar+, [4] Rotor+
  values, motors = misfit.eigh()                             # [4] Scalar+, [4] Motor+
  motor = motors[values.argmin()].normalized()               # [] Motor+
  ```
* **`.eigh(metric)` / `.eigvalsh(metric)`**: generalized symmetric eigensolve of one form against a positive-definite `metric` form, without inverting `metric`:
  ```python
  elastic = Twist & stiffness(Twist)                         # [] Scalar+ <- (Twist+, Twist+)
  kinetic = Twist & inertia(Twist)                           # [] Scalar+ <- (Twist+, Twist+)
  values, modes = elastic.eigh(kinetic)                      # [6] Scalar+, [6] Twist+
  ```

  In matrix notation this reads as $K v = \lambda M v$, solved without forming $M^{-1} K$.
* **`.eig()` / `.eigvals()`**: general eigensolve against `Space`'s metric or a supplied `metric` form; directions unmeasured by a singular metric go to infinity.
* **`.det()` / `.det(metric)`**: determinant of the form relative to `Space`'s metric or an invertible `metric`.

  In matrix notation this reads as $\det(G^{-1} A)$ for metric $G$ and form $A$.
* **`.contract()` and `.trace()`**: `.contract()` raises one input with `Space`'s invertible metric and traces; `.trace()` pairs two inputs of complementary spaces through `&`. A form has no singular values because its coefficient matrix depends on basis scaling; evaluate `form(a, b)` to test whether it vanishes.
* **`.solve(linear)` / `.lstsq(linear, rcond=1e-15)` / `.lstsq(linear, metric, rcond=1e-15)`**: solves `form(x, ·) == linear(..., ·)` for `x` in the form's first input (`form.solve(form(x)) == x`), preserving any leading inputs on `linear`. Supplying a positive-definite `metric` form diagonalizes `form` against `metric`, drops modes with curvature ratio at most `rcond`, and returns the minimum-metric solution:
  ```python
  points = (splats + (w & Point) * (w & Point)).solve(w & Point)  # [splats] Point-
  response = h_pt.lstsq(h_cross, rcond=1e-4)                      # [] Point- <- Twist+
  step = curvature.lstsq(descent, energy, rcond=1e-4)             # [] Twist+
  ```
* **Normal forms without a metric**: pairing a polarity map `cones` (`Plane+ <- Point-`) with its motion Jacobian builds curvature `cones(motion) & motion` directly, and squaring a scalar residual's Jacobian form `j` (`Scalar+ <- Space`) builds `(j * j).sum(axis=0)`.

### Multilinear maps

Multilinear maps (`Output <- (In1, ..., Inn)`) allow any output type, bind leading inputs on partial calls, restrict input domains when passed subtypes, pair chosen axes under `.trace(i, j)` and `.contract(i, j)`, and broadcast batch dimensions:

```python
tidal = Vector.commutator(R(Vector ^ Vector))                # [] Vector+ <- (Vector+, Vector+, Vector+)
```
