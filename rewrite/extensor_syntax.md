# Extensor Syntax & Methods Reference

In numga, the `Extensor` abstraction unifies multivectors and linear/multilinear maps under a single value class:
* **Arity 0**: Multivectors (concrete geometric elements).
* **Arity 1**: Unary linear maps (`Output <- Input`, represented as matrices).
* **Arity $n$**: Multilinear maps / higher-arity tensors (bilinear forms, energy functionals).

---

## 1. The Core Viewpoint: Multivectors Waiting for Arguments

Any normal GA expression involving open slots reads as if operating directly on its **output type**. Conceptually, an extensor *is* a multivector of its output type—it is simply waiting for one or more arguments to be bound at a later time.

When constructing a projection:
```python
# Reads: join light with an open point (Line), meet with ground plane (Point):
shadow = (light & Point) ^ ground                     # [] Point <- Point

# The result behaves as a Point, waiting for an object vertex to be supplied:
projected_vertex = shadow(vertex)                    # [] Point
```

Because every GA operation treats an open extensor by its output type, multi-stage physical, optical, and kinematic pipelines chain naturally without matrix bookkeeping:

```python
# 1. Wedge with k produces a field Bivector (waiting for a spatial polarization a):
field = k.wedge(Spatial)                             # [n_speeds] Bivector <- Spatial

# 2. Material response transforms field Bivector into excitation Bivector:
excitation = medium(field)                           # [n_speeds] Bivector <- Spatial

# 3. Commutator with k dots the excitation Bivector with k, yielding a spacetime Vector:
wave_map = k.commutator(excitation)                  # [n_speeds] Vector <- Spatial
```
At every step, the expression reads as standard geometric algebra on the output type, while numga compiles the multi-stage linear transformations under the hood.

---

## 2. Construction & Argument Lifting

Passing an unbound subspace/GAType (`Vector`, `Point`, `Twist`, `Plane`, etc.) to any GA operation lifts that operation into a compiled extensor.

### Operator Lifting
```python
# Unary linear map (Arity 1): Wrench <- Point
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
spring = lines * (Twist & lines) * spring_constants  # [n_springs] Wrench <- Twist
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
Extensor axes are strictly **output-first**, trailing any leading batch dimensions:
```text
Arity 0 (Multivector):       [batch..., Output]
Arity 1 (Unary Linear Map):  [batch..., Output, Input]
Arity 2 (Bilinear Form):     [batch..., Output, Input1, Input2]
```
For an arity-1 extensor, the coefficient array is a conventional matrix: `kernel[..., out, in]`.

---

## 3. Binding & Composition Syntax

### Calling Syntax (`__call__`)
Calling an extensor binds its logical input slots:

```python
# Full binding on Arity 1: yields a multivector (Arity 0)
p_projected = line_map(p)

# Full binding on Arity 2: binds both inputs
line = join_map(p1, p2)

# Partial binding on Arity 2: supplying one argument yields an Arity-1 extensor
ray_from_origin = join_map(origin)                   # Line <- Point
ray = ray_from_origin(target)                        # Line
```

### Functional Composition
Passing one extensor as the input argument to another contracts the intermediate subspace, multiplying their matrix representations under the hood:

```python
# If A is [B <- A] and B is [C <- B], composing them as B(A) yields [C <- A]:
to_pixel = viewport(camera(world_to_cam))            # Point <- Point

# Collapsing forward kinematics with camera projection:
local_to_pixel = world_to_pixel(bodies_to_world)     # [5] Point <- Point
```

### Frame Transport (Sandwiching Operators)
Transforming a concrete multivector uses a single rotor sandwich: `motor >> x`.
Transforming a linear operator requires pulling the input from world to local and pushing the output from local to world:

```python
# Transforming a linear map T: Out <- In by a motor/rotor:
world_T = motor >> local_T(motor << In)              # Out <- In

# Sight cone quadrics (Plane <- Point) transformed into world frame:
world_cone = pose >> local_cone(pose << Point)       # Plane <- Point
```

### Batching and Broadcasting
Batch axes lead the structural axes and are not slots: they index independent copies of an expression. `stack` creates a batch axis, `.sum(axis)` removes one, and indexing follows NumPy:
```python
waves = stack((plus_wave, cross_wave, plus_wave + cross_wave), axis=1)   # [n_time, 3] Bivector <- Bivector
stiffness = spring_stiffness.sum(axis=0)                                  # [] Wrench <- Twist
```
Calling a batched map on a batched argument broadcasts the batch axes against each other:
```python
acceleration = response[:, :, None](reference)      # [n_time, 3, n_beads] Vector
```
A frame summed against its reciprocal is the coordinate spelling of a trace; write the trace instead (see `.trace(slot)` below).

---

## 4. Extension Methods by Structural Type

### Unary Linear Maps (`Output <- Input`)
Operate directly on linear transformations while preserving input/output blade subspace types:

* **`.solve(rhs)`**: Solves the linear equation $T(x) = y$ directly for $x$, returning a typed multivector.
  ```python
  step = stiffness.solve(force)                      # Solves for displacement twist
  ```
* **`.lstsq(rhs)`**: Least-squares solve for over- or under-determined linear systems.
* **`.inverse()`**: Inverse of the map under composition. On a multivector batch the same method is the geometric-product inverse of each element; a batch of vectors is not a frame, so this is not a reciprocal frame.
* **`.pinv(rcond=1e-4)`**: Moore-Penrose pseudoinverse (e.g. converting Gauss-Newton curvature into posterior pose covariance).
  ```python
  pose_covariance = curvature.pinv()                 # Twist <- Twist
  ```
* **`.transpose()`**: Coefficient transpose with the input and output types swapped, `kernel[..., in, out]`. This is the operator adjoint only in a Euclidean orthonormal blade basis; for a Lorentz boost it is not the inverse. The covariant use is the pullback through duality, which satisfies `pullback(l) & p == l & T(p)`:
  ```python
  pullback = projection.transpose()(Plane.dual()).dual_inverse()   # Plane <- Plane, dual map of Point <- Point
  ```
* **`.det()`**: Determinant of a square endomorphism (`Space <- Space`).
* **`.trace(slot=0)`**: Contracts the output against one input slot by matching blades and drops that slot; the slot's subspace must lie within the output subspace, and the metric is never consulted. Slots are numbered in order of appearance in the expression. On `Space <- Space` this is the matrix trace; on a multilinear map it lowers the arity by one:
  ```python
  ricci = Vector.commutator(R(Vector.wedge(Vector))).trace(slot=1)   # [] Scalar <- (Vector, Vector)
  ```
* **`.svd()`**: Singular value decomposition returning `[U, s, Vh]`.
  * `s`: singular values array.
  * `Vh[-1]`: physical polarization / nullspace eigenmode.
* **`.svdvals()`**: Evaluates only the singular values across all broadcast batch dimensions (ideal for resonance scans).
  ```python
  resonance_curves = wave_map.svdvals()[..., -1]     # Smallest singular value per speed
  ```
* **`.eig()` / `.eigvals()`**: Spectrum of a general (non-symmetric) endomorphism; eigenvalues are complex `[n] Scalar`.
  ```python
  eigenvalues = plus.eigvals()                       # [6] Scalar (complex; all zero for a nilpotent map)
  ```
* **`.cholesky()`**: Cholesky factorization $L L^T$ of a positive-definite linear operator.
* **`.decompose_polar()`**: Decomposes an operator into unitary rotation and symmetric stretch components.

---

### Bilinear & Quadratic Forms (`Scalar <- (Space, Space)`)
Represent metrics, potential/kinetic energy functionals, quadrics, and alignment objectives:

* **`.eigh()`**: Symmetric/Hermitian eigensolve on a single quadratic form ($Q v = \lambda v$).
  Returns eigenvalues `[n] Scalar` and eigenvectors `[n] Space`.
* **`.eigh(metric)`**: **Generalized Hermitian eigensolve** $K v = \lambda M v$.
  Solves the generalized eigenvalue problem directly between two bilinear energy forms without inverting inertia or forming asymmetric coordinate products $M^{-1}K$:
  ```python
  pe_form = Twist & stiffness                        # Scalar <- (Twist, Twist)
  ke_form = Twist & inertia                          # Scalar <- (Twist, Twist)
  values, modes = pe_form.eigh(ke_form)              # values: [3] Scalar, modes: [3] Twist
  ```
* **`.eigvalsh()`**: Evaluates only the real eigenvalues of the symmetric form.
* **`.svdvals()`**: Singular values of the form's coefficient matrix; a vanishing form has all zeros.
  ```python
  ricci.svdvals()                                    # [4] Scalar
  ```
* **`.transpose()`**: Transposes the arguments: $B^T(x, y) = B(y, x)$.
* **Symmetrization**: `(form + form.transpose()) * 0.5` extracts the self-adjoint symmetric part.

---

### Multilinear Maps (`Output <- (In1, ..., Inn)`)
Arity is not limited to forms; any output type is allowed and slots are numbered in order of appearance:
```python
A = Vector.commutator(R(Vector.wedge(Vector)))       # [] Vector <- (Vector, Vector, Vector)
```
Partial calls fill slots in order, `.trace(slot)` lowers the arity by one, and batch axes broadcast as for unary maps.

---

### Scalars (`Scalar`)
Represent eigenvalues, energy values, physical parameters, and norms:

* **`.square_root()`**: Coordinate-free square root of scalar expressions:
  ```python
  frequencies = values.clip(0, np.inf).square_root() / (2 * np.pi)  # [n] Scalar (Hz)
  ```
* **`.clip(min, max)`**: Elementwise numerical clamping over batch dimensions (e.g. clamping small negative eigenvalues from numerical noise to zero).
* **`.argmin(axis)` / `.argmax(axis)`**: Index reduction across batch axes (e.g. locating resonant phase speeds).
* **`.to_array()`**: Converts scalar extensor batches to a standard NumPy/backend array for tabular printout or external consumers.
* **Arithmetic & Math**: Standard elementwise operations: `.exp()`, `.log()`, `.sin()`, `.cos()`, `.abs()`.

---

### Lie Algebra & Motors (`Bivector`, `Motor`, `Rotor`)
* **`bivector.exp()`**: Lie group exponential map (converts angular/twist velocity generators to finite rotors/motors).
  ```python
  motor = (mv.xw * (dx * 0.5) + mv.yz * (dtheta * 0.5)).exp()
  ```
* **`motor.log()`**: Lie group logarithm (extracts the infinitesimal generator twist).
* **`motor.motor_split()`**: Decomposes a PGA motor into pure rotation and translation components.
* **`motor.motor_rotor()` / `motor.motor_translator()`**: Extracts the rotational rotor or translational translator directly.
* **`motor >> x`**: Double-sided rotor conjugation sandwich product ($M x \widetilde{M}$).
* **`motor.normalized()`**: Normalizes motor coefficients to unit gauge ($M \widetilde{M} = 1$).
