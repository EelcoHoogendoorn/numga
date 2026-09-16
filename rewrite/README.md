# Numga rewrite

An isolated, deliberately non-backwards-compatible implementation. The sibling
legacy package is source reference only; these tests do not run it.

## Array execution

```python
from numga import NumpyContext
from numga.backend.jax import JaxContext
from numga.algebras import PGA3D

numpy = NumpyContext(PGA3D, execution="sparse")
jax = JaxContext(PGA3D, execution="sparse")
```

`execution="dense"` remains the default: cached einsum contractions, with
`matmul` for unary matrix/vector application and matrix composition.
`"sparse"` ports the original nonzero-term strategy for exact operator kernels.
It groups terms once, folds signed layout conversions into those terms, and
supports full binding, partial binding, and composition with open extensors.
NumPy uses fused per-term array products; JAX unrolls products and sums for JIT.

Computed numeric maps remain dense, including maps produced by sparse partial
binding. Explicitly lowering an exact operator to an array also selects dense
execution for that operator: no arrays are scanned to rediscover sparsity.
The choice belongs to the context, not GAType, and survives JAX pytree round
trips. Contexts with different execution policies must not be mixed implicitly.
Sparse is opt-in, not an automatic claim of better performance; the original
tradeoff between term count, batch size, and JAX compile time still applies.

Computed extension results use the trusted zero-copy constructor. Their
result GATypes and trait refinements are cached; public input construction
remains separate.

### Other array operations

`map_kernel` forwards to an arbitrary callable without duplicating its API:

```python
selected = x.map_kernel(np.take, indices, axis=0, preserve_traits=True)
total = x.map_kernel(np.sum, axis=0)
```

The callable receives the raw kernel and must return storage compatible with
the same context. It must not mutate the input or change the trailing structural
axes' size, order, or meaning; batch axes may change freely. The result is
wrapped directly, without validation, coercion, or copying (NumPy storage is
marked read-only). Explicit traits are dropped by default; `preserve_traits=True`
is the caller's trusted assertion, not inference about the callable. Use JAX
array functions inside JAX tracing.

## Core extensions

The defaults register on `Extensor`, using the same method descriptors available
to user code. Results carry their own GAType; there is no `returns=` registration
argument or interpreted Python return annotation.

Applicability is stated in registrations, not a second set of type branches
inside the implementation. A warm call extracts the GAType key, performs one
dictionary lookup, and invokes the selected function. Validation, predicates,
and specificity analysis run only on cache misses. This is tested for ordinary
Python execution, not just JAX tracing.

The governing rule is no repeated execution cost for decisions derivable from
GAType statics. Extension bodies execute the mathematics; argument annotations
document the interface without defensive runtime validation. Unsupported
arguments or coefficient operations fail naturally in the operations used.

In examples with a fixed algebra, annotate geometric values with the narrowest
applicable named GAType, such as `Vector`, `Point`, `Rotor`, or a specific map
type. Batch dimensions do not change that type. Use `Extensor` for interfaces
that must remain polymorphic over GATypes or algebras; a concrete GAType belongs
to one algebra. State additional geometric preconditions in the docstring when
the type does not express them.

Construct example inputs in consistent layouts so geometric expressions compose
without intermediate subspace casts. Explicit layout conversions belong at
numerical or plotting boundaries when those need specific coordinates.

```python
import numpy as np
from numga import Algebra, NumpyContext

algebra = Algebra("x+y+z+w0")
mv = NumpyContext(algebra).multivector
x, y, z, w = mv.vector(np.eye(4))

generator = 0.4 * x.wedge(y) + 0.7 * z.wedge(w)
motor = generator.exp()      # carries Versor + ReverseProductOne
inverse = motor.inverse()    # selects reversal, without a norm calculation
recovered = motor.log()      # requires/trusts the unit-versor facts
repaired = motor.normalized()  # explicitly recomputes and corrects drift
```

All of these are nullary Extensors, with normal leading-axis broadcasting.
Unary Extensors also support `inverse()` under composition; their input and
output layouts swap. Higher-arity inversion is not defined by these defaults.

| Method | Default implementation |
| --- | --- |
| `norm_squared()` | Measured reverse product, with exactly symmetrized kernels. Known scalar products have scalar output, but known units are never replaced by a constant. |
| `norm()` | Scalar absolute value; otherwise the measured reverse product's scalar or Study square root. |
| `normalized()` | Measured inverse-root rescaling; establishes reverse-product-one and, where justified, Versor. |
| `square_root()` / `inverse_square_root()` | Original scalar, Study, and unit-rotor formulas. |
| `study_norm()` / `study_norm_squared()` | Original scalar-negation self-product and its square root. |
| Nullary `inverse()` | Unit-product shortcuts, original recursive grade-transform reductions and narrow 5D formulas; the existing solve remains a fallback. |
| Unary `inverse()` | Transpose for coefficient-orthogonal maps; otherwise a square-matrix inverse. No pseudoinverse. |
| `exp()` | Scalar exp; nilpotent shortcut or original quadratic exp followed by repeated squaring. |
| `log()` | Scalar log; original square-root bisection/quadratic log for unit even versors, plus a separate non-unit overload retaining log-scale. |

Unit even-versor logs return grade 2; non-unit even-versor logs retain grades
0 and 2, including the scalar log-scale. Both require the declared Versor fact.
Unit log trusts the input: the normalization of `m+1` in each square-root step
is part of the original root formula, not defensive input repair. Non-unit log
measures the positive scalar scale, applies unit log to the scaled rotor, and
retains the scalar logarithm.

Output selection follows the original API: `value.select[2]` requests the full
grade-2 layout, filling absent blades with zeros; `value.restrict[2]` intersects
the existing static support. Grade tuples and named constructors work too,
such as `.restrict[0, 2]` and `.select.scalar()`. Restriction never inspects
coefficients and cannot recover narrower support once intermediate products
have widened it.

`normalized()` evaluates the full structural reverse product where a Study root
is supported, repairing nonscalar as well as radial drift. On wider carriers,
a certified scalar reverse product still permits scalar rescaling. Unproved,
more general multivector square roots are explicitly unsupported.

## Optional specializations

Generic methods work for PGA3 without any extra import, including the example
above. They use the original quadratic/bisection formulas. The invented analytic
overloads have been removed. Optional coefficient-specific optimizations from
the original library remain to be ported; they must dispatch on their exact
layout and use the original direct-coefficient formulas.

## Numerical contracts and limits

- Explicit norm measurements and normalization always compute. Consumers such
  as unit inverse and geometric log trust their declared input facts and do not
  defensively normalize. User-supplied traits are assertions, not runtime tests.
- Real normalization requires an invertible positive-root branch. Negative
  scalar reverse products are not converted to absolute values and then called
  unit. Invalid numerical inputs retain normal backend NaN/Inf/error behavior;
  structurally impossible operations fail before coefficient execution.
- This exp/log implementation targets real floating coefficients. Complex
  numerical paths are deferred; preserving the distinction between reversion
  and coefficient conjugation does not require implementing them now.
- Generic `exp(n=15)` and `log(n=15)` use the original fixed bisection depth.
  The caller can choose a different depth; no norm-based or dtype-based
  adaptation is inserted.
- Generic unit log uses the original scalar/Study square-root domain. It does
  not invent roots of exactly `-1`, alternate branches, or general 6D+ roots.
  Non-unit log additionally requires a positive scalar reverse product.
- The general inverse is a dense-solve fallback, not a large-algebra optimization.
  NumPy and JAX use their own linear algebra; exact contexts retain rational
  algebraic inverses and norm-squared measurements. Extension methods do not impose a
  blanket dtype guard or silently switch contexts; the operations used determine
  which coefficient contexts can execute them.

The core implementations use `context.xp`. A user can override a method with a
full-GAType predicate and return a typed value directly; see
[`test_core_backend.py`](tests/test_core_backend.py). Backend-specific registry
qualifiers are not implemented yet. No coefficient-dependent dispatch is used.
The public arithmetic vocabulary includes scalar offsets, division, explicit
output casts, self-products, and trusted immutable `with_traits(...)` results;
extension implementations need not reimplement those through raw arrays.
A raw array in `+`, `-`, `*` or `/` is a batch of scalars of the Extensor's own
context with exactly the array's shape, so `mv.w * np.linspace(-1, 1, 9)` is nine lines
and a `(2, 1)` mask makes a `(2, 1)` batch; the geometric
operators still take only Extensors, GATypes and SubSpaces. Explicit constructors never add an
axis: `mv.scalar(weights)` needs `weights[:, None]`, and a raw array's shape in
arithmetic is exactly its batch shape.

## Examples

The [example index](examples/README.md) groups demonstrations into geometry,
quadrics, mechanics, relativity, and electromagnetism, with runnable commands.
Supporting code stays beside each example; tests live under `tests/examples/`
and generated figures under `plots/`.

### Planar stiffness

[`examples/mechanics/stiffness.py`](examples/mechanics/stiffness.py) constructs stiffness from spring
lines and open rigid motions, then combines it with inertia to find normal modes.
The [figure](plots/stiffness.png) and [animation](plots/stiffness.gif) compare two
vertical springs with an added angled spring. They show free and resisted small
motions, with spring colours indicating extension and compression.

From `rewrite/`, with NumPy, SciPy, Matplotlib and Pillow installed:

```sh
PYTHONPATH=src:. python -m examples.mechanics.stiffness --animate
```

### Curvature and gravitational waves

[`examples/relativity/curvature.py`](examples/relativity/curvature.py) builds vacuum plane-wave curvature
from null-bivector dyads. The map is nonzero but squares to zero; binding an
observer produces a tidal map with nonzero eigenvalues. The
[figure](plots/curvature.png) shows the spacetime-plane mapping and three detector
responses; the [animation](plots/curvature.gif) follows plus, cross and circular
wave packets. Particle motion is integrated from curvature in the weak-wave
approximation and magnified for display.

```sh
PYTHONPATH=src:. python -m examples.relativity.curvature
```

To also export the animation:

```sh
PYTHONPATH=src:. python -c 'from examples.relativity.curvature import main; main(animation_path="plots/curvature.gif")'
```

## Tests

For opt-in performance measurements, run from `rewrite/`:

```sh
python -m pytest benchmarks/test_runtime.py -q -s
```

This covers small and batched 3×3 map application, rotor products, and PGA3
inertia construction, with dense/sparse NumPy and JAX execution. A raw NumPy
matrix application gives an overhead baseline. Setup and JAX compilation are
excluded; matrix batches of 1, 10, 100, and 1024 report overhead percentages.
Each timed JAX invocation waits for completion. Results are best-of-three
warm timings in microseconds per call, using each backend's default dtype
(NumPy float64, JAX float32). There are no machine-dependent speed assertions.
Use `-k numpy` or `-k jax` to select a backend. These benchmarks are outside the
default test suite and require no benchmark plugin.

Run `python -m pytest` from this directory with NumPy and pytest installed;
JAX tests additionally need JAX. Pytest selects `rewrite/src`, not the legacy
package. Useful public-surface examples live in
[`test_end_to_end.py`](tests/test_end_to_end.py),
[`test_core_logexp.py`](tests/test_core_logexp.py), and
[`test_core_norms.py`](tests/test_core_norms.py).

[`test_pga3_sandwich.py`](tests/test_pga3_sandwich.py) shows one rigid motion as
distinct plane, line, and point maps: typed composition rejects incompatible
maps, while wedge/intersection and regressive/join commute with the motion.
It checks these relationships as complete open Extensors as well as concrete geometry.
The Euclidean mirror example keeps oriented normals as tangent bivectors: the
same reflection law acts on directions and their wedges, without a separate
normal-handling rule.

Signed-layout examples live in
[`test_layouts.py`](tests/subspace/test_layouts.py). `algebra.subspace("yz zx xy")`
specifies exact coordinates; `from numga.algebras import PGA3D` supplies the
preferred PGA default ordering. Use `select_subspace(target)` for explicit
conversion and `restrict_subspace(target)` to filter without reordering.
GAType equality is layout-exact; `<=` compares mathematical support and traits.

The end-to-end propagation examples pass, including negation, identity-map
application, and reversal before binding. Reversed self-product relations
require known nonzero input self-products; this is static inference, not a
numerical check. The direct PGA2 inertia example in
[`test_geometric_integration.py`](tests/test_geometric_integration.py) constructs
per-point unary maps, sums them, and applies the inertia to batched rates.
[`test_pga3_inertia.py`](tests/test_pga3_inertia.py) does the same for six unit
masses in 3D PGA, checking energy and invariance under signed point relayout.
