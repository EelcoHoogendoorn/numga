# Internals

Every algebraic operation in `numga`, a product of two multivectors as much as `Rotor >> Vector`, is built from types as an exact integer table before any numerical array arrives, and evaluated by contracting (or unrolling) that table on backend arrays. This document covers the `Extensor` representation, how exact tables and binding plans compile into backend calls, how contexts, fields and sparse maps execute, and how to add extension methods or custom `Extensor` subclasses. It pairs with [Extensors](extensors.md) and [Extensor syntax and methods](extensor_syntax.md).

```python
import numpy as np
from numga import Algebra, NumpyContext

ga = Algebra("x+y+")
Q, V = ga.subspace.even(), ga.subspace.vector()
Rotor, Vector = ga.gatype.rotor(), ga.gatype.vector()       # Rotor+, Vector+

# The 2D rotor-vector sandwich r * v * ~r has three inputs before binding: left rotor, vector, right rotor.
sandwich = Rotor >> Vector                                  # Vector+ <- (Rotor+, Vector+, Rotor+)
assert sandwich.arity == 3 and sandwich.axes == (V, Q, V, Q)
print(sandwich.kernel.values)
print(sandwich.formula())
```

```text
[[[[ 1  0]
   [ 0  1]]

  [[ 0 -1]
   [ 1  0]]]


 [[[ 0 -1]
   [ 1  0]]

  [[-1  0]
   [ 0 -1]]]]
out[x] = a0[1] * a1[x] * a2[1] + a0[1] * a1[y] * a2[xy] - a0[xy] * a1[x] * a2[xy] + a0[xy] * a1[y] * a2[1]
out[y] = - a0[1] * a1[x] * a2[xy] + a0[1] * a1[y] * a2[1] - a0[xy] * a1[x] * a2[1] - a0[xy] * a1[y] * a2[xy]
```

Binding a concrete rotor into both rotor inputs produces the `2 x 2` rotation map on vectors, doing the rotor's share of the arithmetic once before applying it across a batch:

```python
mv = NumpyContext(ga).multivector
r = mv.rotor([np.cos(0.3), np.sin(0.3)])                    # [] Rotor+
rotation = r >> Vector                                      # [] Vector+ <- Vector+
points = mv.vector(np.random.default_rng(0).normal(size=(10_000, 2)))  # [10000] Vector+
assert np.allclose(rotation(points).kernel, (r >> points).kernel)
print(np.round(rotation.kernel, 8))
```

```text
[[ 0.82533561  0.56464247]
 [-0.56464247  0.82533561]]
```

## Extensor, GAType and kernel layout

Every `Extensor` stores three attributes (`_context`, `_gatype`, `_kernel`): a backend `Context`, a structural `GAType`, and a coefficient kernel.

A `SubSpace` is an ordered tuple of blade bitmasks paired with per-blade sign bits, canonically ordered by grade and then mask; a blade's position in that tuple is its coefficient index. A `GAType` describes a whole extensor across all its axes, output first:

* `subspaces`: one `SubSpace` per axis (`(Output,)` for arity `0`, `(Output, Input1, ..., Inputn)` for maps).
* `orientations`: one `Orientation` (`extrinsic` `+` or `intrinsic` `-`) per axis, distinguishing geometric kinds that share the same blades ([Orientation in numga](orientation.md)).
* `fields`: `(axis, sites)` pairs marking axes that range over coupled sites as well as blades ([Fields and field maps](fields.md)).
* `traits`: certified algebraic facts on the type, such as `Versor`, `ReverseProductOne`, `Blade` or `CoefficientOrthogonal`.

In the kernel, leading batch axes come first, followed by one site axis per entry in `gatype.fields` (in axis order), followed by one blade axis per entry in `gatype.subspaces` (output first):

```text
nullary      [batch..., O]
unary map    [batch..., O, I]
n-ary map    [batch..., O, I1, ..., In]
field map    [batch..., sites..., O, I1, ..., In]
```

`shape` and `ndim` count batch axes only, while `structural_shape` holds `(sites..., O, I1, ..., In)`. Indexing, `reshape`, `sum`, `stack` and `at[...].set` address batch axes and cannot slice into `structural_shape`. Nothing in the array layout carries a metric or raised/lowered index positions: applying a site-free unary map to a value contracts `"...oi,...i->...o"` (`matmul` for unary calls and compositions, `einsum` for higher arities).

Traits on constructed values (`mv.rotor(values)`) are trusted assertions rather than runtime numerical checks, and methods dispatch on them directly (`mv.rotor(values).inverse()` returns the reverse without testing coefficients). Values of unknown provenance enter as plain types (`mv.even(values)`) and acquire certified traits through operations that establish them (`mv.even(values).normalized()`).

## Exact tables, binding plans and compiled calls

Every primitive operation table, geometric product, wedge, regressive product, inner product, reverse, involute, dual and subspace cast, is an `Extensor` in the algebra's `ExactContext` (`ga.exact`), built once per combination of input `GAType`s on `ga.operator` and cached. Its `SymbolicKernel` stores an `int8` array whose entries on basis-blade products are `-1`, `0` or `+1`. Orientation rules (such as the reflection sign on intrinsic elements in `>>` and pairing/sum checks) enter when `ga.operator` constructs the table, so every composed extensor transforms equivariantly without basis-dependent conventions.

Higher-arity tables are built by binding exact tables together. In `Algebra("x+y+")`, `Rotor >> Vector` composes `Rotor * Vector`, `Vector * Rotor` and `Rotor.reverse()`:

```python
reverse = Rotor.reverse()                                   # Rotor+ <- Rotor+
left = Rotor * Vector                                       # Vector+ <- (Rotor+, Vector+)
right = Vector * Rotor                                      # Vector+ <- (Vector+, Rotor+)
# Bind left into right's first input (Vector+) and reverse into right's second input (Rotor+).
composed = right.bind({0: left, 1: reverse})                # Vector+ <- (Rotor+, Vector+, Rotor+)
assert composed.kernel == sandwich.kernel

# In floating-point coefficients, those two exact binds are two tensor contractions.
right_reverse = np.einsum("ijk,kl->ijl", right.kernel.materialize(), reverse.kernel.materialize())
table = np.einsum("ijk,klm->ijlm", left.kernel.materialize(), right_reverse)
assert np.array_equal(table, sandwich.kernel.materialize())
```

An exact map's `.solve()` never calls a numerical linear solver: its integer inverse is computed once, cached on its `SymbolicKernel`, and bound into the right-hand side as a single constant contraction (raising `LinAlgError` if singular or non-integer). Because regressive pairings of complementary blades and orthonormal metric pairings invert to signed permutations, `T.adjugate()` and `(Vector | Vector).solve(...)` reduce to one contraction against a signed permutation.

`bind` is the single composition primitive, and everything about a bind except the arithmetic is decided from `GAType`s alone in a cached `BindingPlan`:

* **Axis alignment**: per bound input, three lossless relations are applied implicitly: identical subspace, identical blades in another order or sign convention (via a signed permutation), and subset of blades (embedded with structural zeros). Dropping blades or partial overlaps are refused unless written explicitly with `.cast(...)`.
* **Equality groups and narrowing**: passing the same Python object to two inputs (as `r >> v` passes `r` to both versor inputs) records an equality group, allowing `BindingPlan` to narrow the output subspace before contraction (keeping only the passenger's grades under a certified versor sandwich) and propagate stronger traits (`r * r` stays a certified rotor; `r >> Vector` acquires `CoefficientOrthogonal`, so `rotation.inverse()` transposes). Numerical zeros in a float array never narrow a type.

In `numga/expression.py`, each binary `Product` and `VersorAction` separates type construction from value execution: if any operand is a `GAType` (`Rotor * v`, `Rotor >> Vector`), it builds on `ga.operator` in `ga.exact` and calls `.bind(...)` for concrete operands; if both operands are values (`a * b`, `r >> v`), it dispatches through a `staged=True` `ExtensionMethod` to a compiled `application` closure (`numga/extensor/application.py`) keyed on `(target_type, target_context, operand_types, equality_groups)`. Warm calls run that closure directly and wrap the resulting array in the precomputed `GAType`.

## Contexts and backends

An extensor's `context` determines where coefficients live and how contractions run:

```python
ga.exact                                    # ExactContext: int8 SymbolicKernel, no batch axes
NumpyContext(ga, dtype=np.float64)          # NumPy ndarrays
JaxContext(ga, dtype=np.float32)            # JAX arrays, traceable under jit and vmap
TorchContext(ga, dtype=np.float32)          # PyTorch tensors on CPU or CUDA, with autograd
```

Expressions containing bare `GAType`s start in `ga.exact`. Meeting the first array-backed operand promotes the expression to that operand's context, materializing the `SymbolicKernel` in the target `dtype` once and caching it on the symbolic kernel for the life of the process.

Backend array operations go through `context.xp`, a NumPy-shaped namespace supplied directly by NumPy and JAX and adapted by `TorchNamespace` in `numga.backend.torch` (translating spellings such as `axis` to `dim` and `concatenate` to `cat`, uploading exact tables once per device and dtype, and mapping `torch.linalg.LinAlgError` to NumPy's `LinAlgError`). Generalized eigenproblems with a singular metric (such as forms on `PGA3D` motors) require a QZ solver and run on NumPy via SciPy.

In `JaxContext`, `Extensor` is registered as a JAX pytree with `_kernel` as its single leaf and `(gatype, context.key)` as static metadata, rebuilding the context from `context.key` on exit so mutable caches are never hashed by the tracer. Inside `jax.jit`, the exact multiplication table is a compile-time constant rather than a traced input:

```python
import jax
from numga.backend.jax import JaxContext

jr = JaxContext(ga, dtype=np.float32).multivector.rotor([np.cos(0.3), np.sin(0.3)])
jaxpr = str(jax.make_jaxpr(lambda r: r >> Vector)(jr))
assert "f32[2,2,2,2]" in jaxpr and "dot_general" in jaxpr
```

Only the `f32[2]` rotor is traced as an argument; the `f32[2,2,2,2]` sandwich table enters as a constant folded by XLA across the two `dot_general` contractions.

A context can also embed values from another algebra sharing generator names and squares via `context.include_from_algebra(value)`: extrinsic axes keep their blade names, intrinsic axes keep their complements, and coefficients are reordered and signed via a cached inclusion plan (used by the GPU renderers to bring spatial or spacetime values into their projective camera algebra).

## Fields, SparseExtensor and performance

When a dense extensor carries field axes (`gatype.fields`), `bind_fields` (`numga/extensor/fields.py`) treats site axes as labelled trailing batch axes: binding a field into a blade-only axis acts site by site, while binding into a matching field axis contracts the paired site label inside the backend's `einsum`/`matmul` call without materializing per-site-pair intermediates.

`SparseExtensor` and `Sparse` (`numga/sparse/`) store `cells` of shape `[..., couplings]` alongside `rows`, `columns` and `shape == (output_sites, input_sites)` (summing duplicate `(row, column)` entries at construction). Cells may be typed `Extensor` batches or algebra-free NumPy number arrays that promote to scalars when they meet an algebra. Sparse linear solves and eigensolves assemble block-CSR matrices over `(sites * blades)` on the NumPy/SciPy path and cache each map's sparse factorization in `self.factors` on the first `solve`.

For site-free extensors, multiplication tables are stored dense, which stays compact up to five dimensions: one of the largest tables in common use is the 5D CGA versor-trivector sandwich (`(10, 16, 10, 16)`, `25,600` `int8` entries, `1,600` nonzero terms after versor-input symmetrization). Array contexts also accept `execution="sparse"`, which unrolls only the nonzero polynomial terms (as inspected via `sandwich.to_python()`):

```python
print(sandwich.to_python())
```

```text
def apply(a0, a1, a2):
    return [
        a0[0] * a1[0] * a2[0] + a0[0] * a1[1] * a2[1] - a0[1] * a1[0] * a2[1] + a0[1] * a1[1] * a2[0],
        - a0[0] * a1[0] * a2[1] + a0[0] * a1[1] * a2[0] - a0[1] * a1[0] * a2[0] - a0[1] * a1[1] * a2[1],
    ]
```

Dense contraction is the default and has identical semantics. Under NumPy, dense contraction wins on every expression because one BLAS/`einsum` call replaces many array traversals; under JAX, `execution="sparse"` can win on tiny expressions while losing on larger ones such as the CGA sandwich ([benchmarks/motor_map.py](../benchmarks/motor_map.py)).

Warm calls pay a fixed Python dispatch lookup plus the array arithmetic, independent of batch size, so batching values amortizes the dispatch cost completely. While a hand-tuned `PGA3D` versor-point sandwich can match a `4 x 4` matrix multiply on single elements ([look-ma-no-matrices](#ref-look-ma)), building the linear map `versor >> Grade` (`10 x 10 = 100` multiply-adds vs `1,600` direct terms in 5D CGA) wins once applied across a batch:

| Scenario | Faster path (`benchmarks/motor_map.py`) |
| --- | --- |
| 1 CGA trivector, JAX | Direct `versor >> x` sandwich |
| 16,384 CGA trivectors, JAX | Prebuilt `(versor >> Trivector)(x)` map |
| 16,384 CGA trivectors, NumPy | Prebuilt `(versor >> Trivector)(x)` map, by a wider margin |

## Extending numga

Methods such as `inverse`, `solve`, `trace`, `eigh` and `exp` are `ExtensionMethod` descriptors on `Extensor` (`numga/extension.py`), dispatching across operand `GAType`s. First-call resolution evaluates patterns and caches the chosen callable by the exact tuple of operand `GAType`s, so subsequent calls cost a single dictionary lookup.

`ExtensionMethod` supports two kinds of `.register(...)` patterns, plus staging and orientation declarations:

* **Declarative registrations** match on algebraic meaning: a concrete `GAType` matches any subtype in that algebra (in any blade layout, carrying at least its traits), while a `GATypePattern` or `Trait` matches across all algebras. More specific registrations win and are registered before the general fallbacks they refine:
  ```python
  @Extensor.exp.register(PGA3D.gatype.bivector())       # any bivector subtype/layout in PGA3D
  @Extensor.inverse.register(GATypePattern.map())       # any unary map in any algebra
  ```
* **Predicate registrations** take a callable on the full `GAType`s and are tried in registration order *before* declarative matches (`position=0` inserts at the front). Layout-specific closed forms in `numga.extensions.optimized` use predicates that compare `subspaces` (blade order and signs included), falling through to generic implementations on other layouts without conversion overhead:
  ```python
  from numga.algebras import PGA3D
  from numga.extensor import Extensor

  @Extensor.exp.register(
      lambda t: t.algebra.description == PGA3D.description
      and t.subspaces == (t.algebra.subspace("yz zx xy wx wy wz"),),
      position=0,
  )
  def exp_pga3_fast(b: Extensor, *, n: int = 15) -> Extensor: ...

  @Extensor.exp.register(lambda t: t.is_bivector and t.is_nilpotent)
  def exp_nilpotent(b: Extensor, *, n: int = 8) -> Extensor: ...
  ```
* **Staging, fields and orientation**: `register(..., staged=True)` invokes the registered function once per `GAType` signature to return a specialized runtime callable; `fields=True` opts into `GAType`s carrying field axes; and `ExtensionMethod(name, takes=..., input_orientation=..., output_orientation=...)` enforces mathematical orientation rules across all implementations of a method (`input_orientation=Orientation.extrinsic` on `exp` and `square_root`; `output_orientation=Orientation.extrinsic` on `norm` and `cos`; computed naturally when omitted).

`inverse` illustrates layered dispatch: unit versors invert by reversion, `CoefficientOrthogonal` maps by transposition, scalars by reciprocal, square maps by matrix inverse, 5D bivectors and trivectors by closed form, and general multivectors by self-product cancellation or subalgebra linear solve. Every numeric escape (`linalg` calls or direct kernel indexing) stays contained inside its registration and returns a typed `Extensor`.

Dense arrays are not required by `Extensor`'s contract either. `PrincipalInertiaPGA3` in `numga.extensions.optimized` stores a principal-frame rigid-body inertia (`Antibivector- <- Bivector+` in `PGA3D`) as four numbers, one mass and three principal moments, while behaving as an `Extensor` everywhere:

```python
from numga.extensions.optimized import principal_inertia_pga3

mv3 = NumpyContext(PGA3D).multivector
inertia = principal_inertia_pga3(
    mv3.scalar([2.0]), mv3.bivector([3.0, 4.0, 5.0, 0.0, 0.0, 0.0]),
)                                                           # [] Antibivector- <- Bivector+
rate = mv3.bivector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])         # [] Bivector+
# Applying and inverting work directly on the four stored numbers (6 multiplies, 4 reciprocals).
momentum = inertia(rate)                                    # [] Antibivector-
inverse_inertia = inertia.inverse()                         # [] Bivector+ <- Antibivector-
motor = (rate * 0.2).exp()                                  # [] Rotor+
transported = motor >> inertia(motor << PGA3D.gatype.bivector())  # [] Antibivector- <- Bivector+
```

A custom subclass overrides `__call__` and whichever methods it specializes, computes `shape` from its stored arrays, rebuilds the dense `_kernel` on demand for generic operations, and overrides `_from_prepared_kernel` so derived results return ordinary `Extensor`s.

## References

* <a id="ref-look-ma"></a>**[look-ma-no-matrices]** Steven De Keninck, *Look Ma, No Matrices!* [Link](https://enkimute.github.io/LookMaNoMatrices/)
