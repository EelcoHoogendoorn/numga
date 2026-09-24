# Internals

Every numga operation, a product of two multivectors as much as `Rotor >> Vector`, is built as
an exact table before any value is supplied. This document shows that table, where its numbers
live, and what runs when values arrive. It is for a reader of the [guide](extensors.md) who
wants to see the arrays behind an expression, and for anyone about to extend the library with a
method or a backend. The design record that preceded the implementation is
[`extensor_design.md`](../../extensor_design.md) at the repository root; where the two differ, this
document describes what runs.

## 1. Implementation

The implementation is fairly simple. Every operation is first constructed as an exact
multiplication table, and higher-arity operations are composed from those tables by
contraction. When bound with multivector arguments, the tables are executed as dense
contractions in the array backend; unrolling over only the nonzero terms of the expression is
available as an alternative execution policy, and section 6 says when it pays.

Seeing as how numga leans into the multi-staged compilation model of being a Python library
that sets out to trace jit-compiled JAX functions, it is easy to delegate the construction of
the operations to a preprocessing step, which is cached, and the execution of the operations
to be jit-compiled by JAX. This allows for good JAX performance in a relatively simple library.
The NumPy backend runs the same tables eagerly, with the same caching: the tables, the binding
plans and the compiled application closures are built once per signature, and a warm call is
a few array operations over the batch. Its per-call overhead is in line with NumPy's own
per-operation overhead, and dense contraction of the tables is by far the most efficient way
NumPy can evaluate an expression such as a rotor sandwich. Without a separate compilation
stage to fuse arithmetic, that is as fast as a geometric algebra library in numpy gets. 

## 2. Example code

For illustration, we here work out how the ternary rotor-vector sandwich comes to be, in the
smallest algebra that has one.

```python
import numpy as np
from numga import NumpyContext
from numga.algebra import Algebra

ga = Algebra("x+y+")
# the subspace of even grade multivectors, isomorphic to the complex numbers, and of vectors
Q = ga.subspace.even()
V = ga.subspace.vector()
# the type of rotors: even grade, certified as a unit versor
Rotor = ga.gatype.rotor()
Vector = ga.gatype.vector()

# Lets examine how the Cl2 rotor-vector sandwich comes to be
#  output = r * v * r.reverse()
sandwich = Rotor >> Vector
# the two rotor slots are distinct inputs; only supplying the same value to both makes a rotation
assert sandwich.arity == 3
assert sandwich.axes == (V, Q, V, Q)
# the kernel is exact: a table of integers, output axis first
assert sandwich.kernel.shape == (2, 2, 2, 2)
print(sandwich.kernel.values)
[[[[ 1  0]
   [ 0  1]]
  [[ 0 -1]
   [ 1  0]]]
 [[[ 0 -1]
   [ 1  0]]
  [[-1  0]
   [ 0 -1]]]]

# the same table, read out as one polynomial per output coefficient
print(sandwich.formula())
out[x] = a0[1] * a1[x] * a2[1] + a0[1] * a1[y] * a2[xy] - a0[xy] * a1[x] * a2[xy] + a0[xy] * a1[y] * a2[1]
out[y] = - a0[1] * a1[x] * a2[xy] + a0[1] * a1[y] * a2[1] - a0[xy] * a1[x] * a2[1] - a0[xy] * a1[y] * a2[xy]

reverse = ga.operator.reverse(Q)
# reversing a rotor is just negating its bivector part
# this follows the standard GA logic of counting the number of swaps required to map the reversed basis vectors to their original positions
# We here resist the urge to prematurely optimize this representation; but rather lean into the general form of a linear map to represent the reverse at this stage, so as to keep the code for combining it with other extensors simple.
print(reverse.kernel.to_object_array().astype(int))
[[ 1  0]
 [ 0 -1]]

# left hand side of the sandwich;
# the product of Rotor and Vector produces another Vector
# This multiplication table is again constructed using the standard GA logic;
# eliminating repeating terms that contract via the metric; and then mapping to a standardized ordering of basis blades
# We may symbolically deduce in this step that the output space is V;
# these being the only nonzero terms that emerge from the multiplication table
left = Rotor * Vector
assert left.axes == (V, Q, V)
print(left.kernel.to_object_array().astype(int))
[[[ 1  0]
  [ 0  1]]
 [[ 0  1]
  [-1  0]]]

# right hand side of the sandwich; (Rotor * Vector) * Rotor.reverse()
right = Vector * Rotor
assert right.axes == (V, V, Q)

# we can bind the reverse extensor to the Rotor input slot of the right side product,
# and the left side product to its Vector input slot, to get the combined extensor.
# Slots are numbered over inputs, and a bound map splices its own inputs in place of the slot.
composed = right.bind({0: left, 1: reverse})
assert composed.axes == (V, Q, V, Q)
# Since we are merely explicitly retracing the same steps numga takes under the hood,
# we obtain the same result as we get by letting numga handle the binding of arguments
assert composed.kernel == sandwich.kernel

# in coefficients, those two binds are two contractions; materialize() converts an exact table to floats
right_reverse = np.einsum('ijk,kl->ijl', right.kernel.materialize(), reverse.kernel.materialize())
table = np.einsum('ijk,klm->ijlm', left.kernel.materialize(), right_reverse)
assert np.array_equal(table, sandwich.kernel.materialize())

# The below is what happens when binding a specific rotor to the sandwich extensor.
# The rotor goes into both rotor slots at once, and what remains is
# 'the rotor sandwich in matrix form'
mv = NumpyContext(ga).multivector
r = mv.rotor([np.cos(0.3), np.sin(0.3)])
rotation = r >> Vector
assert rotation.axes == (V, V)
print(rotation.kernel)
[[ 0.82533561  0.56464247]
 [-0.56464247  0.82533561]]
assert np.allclose(rotation.kernel, np.einsum('ijkl,j,l->ik', sandwich.kernel.materialize(), r.kernel, r.kernel))

# applying the map to a batch of vectors is the same as sandwiching them directly;
# the map does the rotor's share of the work once
points = mv.vector(np.random.default_rng(0).normal(size=(1_000_000, 2)))
assert np.allclose(rotation(points).kernel, (r >> points).kernel)
```

Under JAX, the (2, 2, 2, 2) sandwich table is not an argument of the jitted function. It is a
compile-time constant for the purpose of this tracing context: it does not depend on the
numerical values of the rotor, only on its type, and it is computed once and cached.

```python
import jax
from numga.backend.jax import JaxContext

r = JaxContext(ga).multivector.rotor([np.cos(0.3), np.sin(0.3)])
print(jax.make_jaxpr(lambda r: r >> Vector)(r))
{ lambda a:f32[2,2,2,2]; b:f32[2]. let
    c:f32[2,2,2,2] = device_put[...] a
    d:f32[2,2,2] = dot_general[dimension_numbers=(([3], [0]), ([], []))] c b
    e:f32[2,2] = dot_general[dimension_numbers=(([1], [0]), ([], []))] d b
  in (e,) }
```

Only the rotor `b` is traced; the table `a` is a constant that the compiler folds as it sees
fit. Since the table is a constant, the same function can also be written out over its nonzero
terms, which is what `to_python()` prints:

```python
print(sandwich.to_python())
def apply(a0, a1, a2):
    return [
        a0[0] * a1[0] * a2[0] + a0[0] * a1[1] * a2[1] - a0[1] * a1[0] * a2[1] + a0[1] * a1[1] * a2[0],
        - a0[0] * a1[0] * a2[1] + a0[0] * a1[1] * a2[0] - a0[1] * a1[0] * a2[0] - a0[1] * a1[1] * a2[1],
    ]

def rotation_matrix(r):
    # after simple term rewriting we may expect of the XLA compiler toolchain,
    # binding r into both rotor slots ends up as code similar to the following
    d = r[0] * r[0] - r[1] * r[1]
    od = 2 * r[0] * r[1]
    return [[d, od], [-od, d]]
```

## 3. Kernel layout

```text
nullary   [batch..., O]
unary     [batch..., O, I]
n-ary     [batch..., O, I1, ..., In]
```

The trailing axes are the structural axes, one per subspace in `gatype.subspaces`, output
first. `shape` and `ndim` describe the batch axes only, `structural_shape` the rest, and
`kernel[..., out, in]` indexes a unary map like a matrix. Indexing, `reshape`, `sum`, `stack`
and `at[...].set` address batch axes and cannot reach a structural axis.

A `SubSpace` is an ordered tuple of blade bit masks with an orientation sign per blade, in a
canonical order by grade and then mask. The position of a blade in that tuple is its
coefficient index. Two subspaces with the same blades in another layout, or with other
orientation signs, are different axes, and a binding between them goes through a signed
permutation matrix that the plan of section 5 supplies.

Applying a unary map to a value is one contraction of the input axis against the value's
structural axis, with the batch axes broadcasting:

```text
"...oi,...i->...o"
```

That string is how the NumPy backend runs the map. It is not what the map is. The map is the
geometric expression with a slot left open; the kernel is that expression's coefficient table
in the blade bases of its slots; the einsum, a matmul, or an XLA dot are ways of evaluating the
table, chosen by the backend. The dense executor uses matmul for a unary map applied to a
value or to another unary map, and einsum otherwise. Nothing in the layout carries a metric or
an index position, for the reasons given in [`extensor_advanced.md`](extensor_advanced.md).

## 4. Stages and contexts

Every extensor has a `context`, and the context decides where the coefficients live and what a
contraction does:

```python
ga.exact                                    # ExactContext: integer coefficients, no batch axes
NumpyContext(ga, dtype=np.float64)          # arrays
JaxContext(ga, dtype=np.float32)            # arrays that can be traced
TorchContext(ga, torch.float32, "cuda")     # tensors on a device, with autograd
```

The exact context is owned by the algebra. Every operation table, the geometric product of
two types, the wedge, the reverse, the dual, the casts between layouts, is an extensor in it,
built once per combination of types from the algebra's blade product table and cached on
`ga.operator`. A `SymbolicKernel` is an int8 array: the coefficient of a product of basis
blades is always -1, 0 or +1, and composing products, as the sandwich of section 2 does,
keeps them integer. Scaling an exact expression by a float follows NumPy and makes its kernel
floating.

An expression with a bare type in it is built in the exact context. The first array-valued
operand promotes the whole expression to that operand's context: the exact table is
materialized in the context's dtype and the contraction runs there. The materialized array is
cached on the symbolic kernel, so a table is converted once per dtype for the life of the
process.

An `Extensor` in a `JaxContext` is a pytree with its kernel as the one leaf and its type and
context key as static metadata. It crosses `jit` and `vmap` and comes back typed; a context is
rebuilt from its key on the way out, so no mutable cache is ever hashed by the tracer.

The library calls its array backend through `context.xp`, a namespace with NumPy's names and
signatures. NumPy and JAX supply one directly. For PyTorch, `numga.backend.torch` translates
the few spellings that differ, `axis` to `dim` and `concatenate` to `cat` among them, and
raises NumPy's `LinAlgError` for a singular matrix. Library code is written once against NumPy
conventions. A torch Extensor is a tensor with a type, so autograd, `torch.vmap` and
`torch.compile` see ordinary tensor operations; exact tables are uploaded to the device once
per dtype. Generalized eigenproblems whose metric is singular, as for forms on PGA points,
need a QZ solver and so run only in NumPy with SciPy.

## 5. Binding plans and types

`bind` is the one primitive. A call binds its operands to the leading slots, the slot's own type in
place of an operand leaves it open, and a full call runs on the compiled path. The operators on types
and values are binds of operation tables:

```python
sandwich.bind({0: r, 2: r})                 # partial: Vector <- Vector, what r >> Vector does
sandwich(r, v, r)                           # full: Vector, what r >> v does
Rotor * v                                   # ga.operator.geometric_product(Rotor, v.gatype).bind({1: v})
```

Everything about a bind except the arithmetic is decided from types alone, once, and cached.
The `BindingPlan` for a target type and its operand types records, per bound slot, how the
operand's output axis maps onto the slot's axis. Three relations are lossless and are applied
implicitly: the same subspace, the same blades in another layout, or a subset of the blades,
which embeds with structural zeros. A projection onto fewer blades, or a partial overlap, is
refused and has to be written as `cast`. The plan can also narrow the output before anything
is contracted: when a certified versor is bound into both slots of a sandwich, the output keeps
only the passenger's grades. Zeros in a floating-point array never narrow anything; only types
do.

The result's type comes from the same plan. Its subspaces are the spliced axes. Its traits are
what the operation's relations and the operands' certified traits prove: the geometric product
carries the relation that a product of two unit-reverse values has unit reverse, so `r * r` is
again a certified rotor, and one certified rotor in both slots of the sandwich yields a
`CoefficientOrthogonal` map, which is what lets `rotation.inverse()` transpose instead of
invert. Supplying the same object to two slots is recorded as an equality group; two rotors
that merely happen to be equal are not identified. Traits on a constructed value, as in
`mv.rotor(...)`, are trusted assertions, and construction never tests coefficients against
them. Methods then act on the trait, not on the numbers: `mv.rotor(values).inverse()` is the
reverse whatever the values are. Coefficients of unknown provenance enter as their plain type,
`mv.even(values)`, and earn the trait by a method that establishes it, `mv.even(values).normalized()`.

## 6. Storage, sparsity and performance

A call compiles once per signature and then only executes. The signature is the target's type
and context, each operand's type and context, and the equality groups. For it, the library
resolves the contraction strings, the output restriction, the coordinate conversions and the
materialization of an exact target, and stores a closure on the array context. A warm call runs
that closure: array operations, and a wrap of the result in its precomputed type. The cost of a
call is therefore a fixed dispatch overhead plus the arithmetic, and the overhead does not
shrink with the batch. An unbatched loop over a thousand points pays it a thousand times for
arithmetic that is a fraction of it, which is why the examples batch everything.

The multiplication tables are stored dense. That keeps the implementation simple, and in
modest dimensional algebras, below six dimensions, it has not been a problem. One of the worst
case constructions one might encounter in practice is the sandwich of a versor with a
trivector in five dimensions, a CGA point pair. The versor has 16 components and the trivector
10; prior to binding any arguments to the ternary operation, the table has shape
(10, 16, 10, 16), or 25,600 entries. Such an object is only constructed outside the scope of an
innermost jit operation, and is thus a compile-time constant. Fetching it from RAM takes a
fraction of a microsecond, and dense contractions over it are fast enough. One can also see
how such dense logic can result in compilation times that get out of hand in more exotic,
higher dimensional algebras.

That sandwich has 1600 nonzero terms, after the symmetrization over its two versor slots that
the sandwich construction performs. The array contexts accept `execution="sparse"`, which
unrolls those terms instead of contracting the table; the [benchmarks](../benchmarks/motor_map.py)
show it winning for small expressions under JAX and losing for this one. In NumPy it loses by
a wide margin for any expression: each unrolled term is a separate array traversal, whereas
the dense contraction is one call into BLAS or einsum. Dense contraction is the default and the
two policies have identical semantics. Note that in the Cl(3,0,1) case, performing a well
optimized versor-point sandwich directly can be [competitive](#ref-look-ma) with a 4x4 matrix
multiplication. In higher dimensions a conversion to matrix form pays off quickly when there is
more than one object to transform, as we can see from the 1600 nonzero terms in the sandwich
expression versus a mere 10x10 = 100 fused multiply-adds in the matrix form. The same
benchmarks measure that trade, construction included:

| Scenario | Result |
| --- | --- |
| One CGA trivector, JAX | The direct sandwich beats building the map and applying it. |
| 16,384 trivectors, JAX | Building the map and applying it beats the direct sandwich. |
| 16,384 trivectors, NumPy | The same, by a wider margin, with no compiler to fuse the direct form. |

Each scenario's docstring records the measured times on one machine.

With regards to the cost of the extensor syntax itself: there is a conditional in each
operator invocation that differentiates concrete arguments, to be bound, from bare types, over
which an operation is to be constructed. From the perspective of a jit-compiled expression,
which is the setting numga concerns itself with, that conditional is resolved at trace time.
Using the eager NumPy backend it is paid per call, along with the cached plan lookup above, as
a fixed cost of the same order as NumPy's own per-operation overhead. It does not change how
NumPy code is written: batch the data, as one would anyway, and the array work dominates.

## 7. Extension methods

`inverse`, `solve`, `trace`, `eigh`, `exp` and the rest are `ExtensionMethod` descriptors on
`Extensor`. Each holds a dispatch table over the GATypes of its operands. There are two kinds
of registration, and they differ in what they can see.

A declarative registration matches on meaning. A GAType matches every operand whose support
lies inside its own, in any blade layout, carrying at least its traits, in that one algebra. A
`GATypePattern` states only an arity and traits, and matches in every algebra. Among the
declarative matches the most specific wins, and a specialization must be registered before the
more general registration it refines:

```python
@Extensor.exp.register(ga.gatype.bivector())         # yz zx xy xw yw zw, xy xz yz ..., or only yz zx xy
@Extensor.inverse.register(GATypePattern.map())      # any map in any algebra
```

A predicate registration is a function of the complete GATypes, and sees everything the
declarative rules abstract away: the algebra, the exact layout, the types of derived products.
Predicates are tried before every declarative registration, in the order they were registered;
`position=0` puts one first. Code written against specific coefficient indices is registered this
way, with a predicate that compares subspaces, which includes blade order and signs:

```python
PGA3 = AlgebraDescription(("x", "y", "z", "w"), (1, 1, 1, 0))
@Extensor.exp.register(lambda t: t.algebra.description == PGA3
                       and t.subspaces == (t.algebra.subspace("yz zx xy xw yw zw"),), position=0)
@Extensor.exp.register(lambda t: t <= t.algebra.subspace.bivector() and t.squared.is_empty)
```

The first is how the opt-in closed forms in `numga.extensions.optimized` bind to one layout;
another layout of the same bivectors falls through to the generic method, not into a
conversion. At runtime a call is one dictionary lookup on its operands' GATypes; predicates and
patterns are evaluated only the first time a type is seen.

`inverse` is the fullest table and shows the pattern. A certified unit versor inverts by its
reverse; an orthogonal map by transposition; a scalar by its reciprocal; a square map by a
matrix inverse; a bivector or trivector in five dimensions by a closed form; and a general
multivector by dividing out a self-product, choosing the involution and the number of steps
that its type needs to reach a scalar, falling back to a linear solve in the subalgebra its
blades generate. `solve` dispatches on the shape of the problem: a map against a value, a form
against a linear form, two values, or a multilinear construction against a map, as
[`extensor_advanced.md`](extensor_advanced.md) describes from the outside.

An implementation takes extensors and returns an extensor with a type it declares. The numeric
escape, a call into `linalg` or a formula over kernel columns, lives inside the registration
and nowhere else.

Dense coefficients and contraction are one way to store and apply an extensor, not part of its
contract. `PrincipalInertiaPGA3` in `numga.extensions.optimized` is a rigid-body inertia stored
as four numbers, a mass and three moments, and it is an `Extensor` of type
`Bivector <- Bivector` like any other:

```python
inertia = principal_inertia_pga3(mass, moments)
inertia(rate)                   # six products on the stored numbers
inertia.inverse()               # four reciprocals, not a 6x6 inverse
motor >> inertia                # anything else sees the dense map, and returns an ordinary Extensor
```

A subclass does this by overriding `__call__` and whichever methods it can do better, providing
`shape` from its own numbers and `_kernel` as the dense map built for each use, and overriding
`_from_prepared_kernel` so that derived results are plain extensors.

## References

* <a id="ref-look-ma"></a>**[look-ma-no-matrices]** *Look Ma, No Matrices!* [Link](https://enkimute.github.io/LookMaNoMatrices/)
