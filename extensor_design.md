# Extensor Runtime and Binding

Status: working design for the non-backwards-compatible rewrite.

This note records the current plan for the `Extensor` abstraction. It complements
[`extensors.md`](rewrite/docs/extensors.md), which motivates extensors conceptually, and
[`gatype_design.md`](gatype_design.md), which defines their type model. The older
[`numga2_design.md`](numga2_design.md) remains untouched and is not yet reconciled
with this design.

## 1. Scope

The rewrite makes one value abstraction cover both multivectors and partially
bound multilinear maps, in exact symbolic and backend representations alike:

```text
arity 0    multivector
arity 1    linear map / matrix
arity n    n-argument multilinear map
```

The central decisions are:

- `Extensor` replaces the parallel concrete `MultiVector` and
  `ConcreteOperator` hierarchies and the separate exact `Operator` value.
- `Extensor` has immutable value semantics at every arity.
- Exact symbolic kernels live in an `ExactContext`; backend kernels live in
  their backend contexts, without changing value class.
- `OperatorFactory` remains an operation factory, not an extensor value.
- Every `Extensor` has one whole-object `GAType`.
- Structural axes are output-first.
- Binding backend data is eager; ExactContext binding remains exact.
- A bare `SubSpace` in a typed expression position implicitly lifts to an
  unrefined nullary `GAType` and denotes an unbound value position.
- `bind` is the primitive for partial application and composition.
- `__call__` remains useful as thin full-bind syntax.
- Repeated arguments are declared explicitly by named constructions.
- Batch behavior is shared by nullary and positive-arity extensors.

This is also a representation unification: exact operation tensors and backend
values are represented by the same Python class. Their contexts, not their
classes, distinguish execution stages.

## 2. Core representation

The abstract runtime representation is:

```text
Extensor:
    context: Context
    gatype:  GAType
    kernel:  context-owned coefficient object

    axes            = gatype.subspaces
    output_subspace = axes[0]
    subspace        = output_subspace
    input_subspaces = axes[1:]
    arity           = len(input_subspaces)
```

`GAType` describes the current extensor as a whole. It is not merely the type of
the output axis. Its structure is defined in
[`gatype_design.md`](gatype_design.md):

```text
GAType:
    subspaces: tuple[SubSpace, ...]
    traits:    canonical TraitSet
```

There is no second `output_gatype`. Every extensor already has exactly one
whole-object `GAType`. `gatype.subspaces[0]` is distinguished structurally as
the output carrier, while relational traits describe how binding affects the
type of the next extensor.

Successive bindings produce successive whole-object types:

```text
F                   GAType((O, I1, I2), map_traits)
F.bind(x)           GAType((O, I2), specialized_map_traits)
F.bind(x).bind(y)   GAType((O,), result_traits)
```

The final bind follows exactly the same rule as every preceding bind. It happens
to produce a `GAType` with one subspace and therefore arity zero; no output type
is projected, created, or substituted as a special case.

Because every arity has the same distinguished output carrier, `.subspace` can
remain a universal convenience alias for `.output_subspace`:

```text
e.subspace == e.output_subspace == e.gatype.subspaces[0]
```

This property returns a structural `SubSpace`, never another `GAType`.

### Value construction namespace

Each coefficient context owns a `multivector` namespace for the common
arity-zero case:

```python
points = context.multivector.vector(coefficients)
rotor = context.multivector.rotor(coefficients)
custom = context.multivector(gatype_or_subspace, coefficients)
```

This is only a construction facade. It always returns the same `Extensor`
class, delegates coefficient coercion and shape validation to
`Context.extensor`, and rejects positive-arity `GAType`s. The latter remain
constructible through the general `context.extensor(gatype, coefficients)`
primitive.

Named constructors resolve through `algebra.gatype`, rather than directly
through `algebra.subspace`. Consequently `multivector.even(...)` constructs an
unrefined even value while `multivector.rotor(...)` preserves the certified
rotor traits. When coefficients are omitted, the value is scalar one projected
into the requested support: this is one for scalar-containing spaces and zero
for spaces such as vectors. Because that omitted value is known exactly at
construction time, its unit or zero facts refine the resulting GAType, and an
explicitly contradictory trait is rejected. Supplied coefficients and explicit
GAType traits remain trusted assertions; ordinary construction never
numerically tests whether supplied coefficients satisfy those traits.

### Kernel layout

All structural dimensions trail the batch dimensions:

```text
kernel.shape
    == batch_shape
     + tuple(len(subspace) for subspace in gatype.subspaces)
```

The structural layout is output-first:

```text
nullary:  [batch..., O]
unary:    [batch..., O, I]
n-ary:    [batch..., O, I1, ..., In]
```

This makes a unary extensor a conventional matrix and permits direct application
with an einsum such as:

```text
"...oi,...i->...o"
```

The public storage name remains open. `kernel` is precise for all arities;
`data` is more familiar for values. The implementation should choose one
canonical attribute rather than reproduce the present `values`/`data`/`kernel`
vocabulary.

### Batch shape

`shape` describes only batch dimensions:

```text
extensor.shape == batch_shape
extensor.ndim  == len(batch_shape)
```

Structural dimensions are described by `gatype.subspaces`, not mixed into the
public collection shape.

Every participating `.shape` follows the backend's ordinary broadcasting
rules in pointwise operations and binding:

```text
result.shape = broadcast_shapes(*(operand.shape for operand in operands))
```

This rule is independent of extensor arity. Structural axes are matched and
contracted through `GAType` and `BindingPlan`; they never enter batch
broadcasting. An exact Extensor has `.shape == ()`, so after lowering its pure
kernel broadcasts as a batch constant over every backend operand.

Addition has a separate structural rule. Same-arity Extensors from one algebra
are zero-embedded into the union of each pair of corresponding SubSpaces before
their kernels are added:

```text
(A0, A1, ..., An) + (B0, B1, ..., Bn)
    -> (A0 union B0, A1 union B1, ..., An union Bn)
```

The embedded kernels then broadcast over their complete leading `.shape`s.
`SubSpace.union` is the semantic primitive; `A + B` may remain its concise
surface spelling. Trait inference for addition is conservative and separate
from both structural union and batch broadcasting.

## 3. Immutable semantics

An `Extensor` never changes its coefficients, context, `GAType`, or batch shape.
Every operation constructs a new extensor:

```text
operation:
    Extensor[A] x Extensor[B] -> Extensor[infer(operation, A, B)]
```

This applies equally to:

- arithmetic;
- binding and composition;
- indexing and functional indexed updates;
- reshape, broadcast, concatenate, stack, and reductions;
- `assume` and `forget` type refinements.

There is therefore no trait-invalidation lifecycle. A newly constructed result
receives exactly the traits justified by its construction; its inputs remain
unchanged.

An arbitrary coefficient-replacement operation must not silently retain the old
traits. Prefer an explicit constructor from raw coefficients. `assume(...)` and
`forget(...)` may share storage only when that storage cannot be mutated through
either value.

Immutable semantics require the implementation to break or prevent mutable
aliases. Constructors must copy, take exclusive ownership of, or freeze
caller-owned NumPy and Torch buffers, and the public API must not expose a
mutable handle to certified coefficients. The backend-specific enforcement
mechanism remains open; the immutability guarantee does not.

### Equality is policy-specific

`Extensor` deliberately has no generic coefficient-level `equals` method.
Exact-kernel equality, floating-point closeness, batched elementwise comparison,
and reduction to one boolean are different operations. Callers and tests must
choose those semantics explicitly: exact Extensors may compare their
`SymbolicKernel`s, while a backend test supplies that backend's comparison
operation and tolerances. Structural type comparison remains the separate,
exact comparison of `GAType` values.

## 4. Context as the staging boundary

There is exactly one extensor value class. Its context owns the coefficient
representation and determines the execution stage:

```text
Extensor(ExactContext, gatype, exact_symbolic_kernel)
Extensor(NumpyContext, gatype, numpy_array)
Extensor(JaxContext, gatype, jax_array)
```

An Extensor in `ExactContext` is algebra-bound, immutable, exactly composable,
and normally batch-free. `ExactContext` owns rational preparation, contraction,
and lowering. An Extensor in a backend context carries that backend's dtype,
device, and batch representation and is ready for eager execution or tracing.
Changing stage changes context and kernel representation, not value class or
`GAType`.

`OperatorFactory` remains the algebra's operation-construction and caching
surface. It returns operation Extensors—normally in `ExactContext` when all
positions are typed or exact—but it is not itself a tensor value or a staging
class.

### Staging rules

```text
exact Extensor composed with exact Extensor       -> exact Extensor
construction containing only typed positions      -> exact Extensor
exact Extensor bound to a backend Extensor         -> backend Extensor
backend Extensor bound to exact/backend Extensors  -> backend Extensor
direct backend coefficient construction           -> backend Extensor
```

Exact-to-exact composition uses exact arithmetic. As soon as backend data
participates, exact operands are lowered into the selected compatible backend
context, contraction is eager, and the result remains in that context. Two
non-exact contexts must be explicitly compatible; selection never depends on
mapping or argument iteration order.

## 5. Typed expression positions and construction syntax

A `SubSpace` in a typed expression position implicitly lifts to an unrefined
nullary type:

```text
lift(S) = GAType((S,), EMPTY_TRAITS)
```

`OperatorFactory` interprets that type as an unbound value position while
constructing an operation:

```text
V * V          -> exact Extensor, arity 2
V * V * V      -> exact Extensor, arity 3
x * V          -> Extensor in x.context, arity 1
x * y          -> Extensor in the selected compatible context, arity 0
m.sandwich(P)  -> Extensor in m.context, arity 1
```

The syntax is surface sugar over factory construction and `bind`; it should not
introduce a second execution path.

The basic rule is:

```text
only typed positions and exact Extensors participate -> ExactContext Extensor
at least one backend Extensor participates           -> backend Extensor
```

A lifted type position is a construction specification, not an Extensor with a
kernel that can be contracted by `bind`. The lift therefore introduces no
phantom coefficients and no second value abstraction.

A bare subspace lift describes only structural support and has
`EMPTY_TRAITS`. A refined `GAType` can be supplied directly when a typed
position requires known value facts. A positive-arity argument cannot satisfy
a nullary value requirement merely by having a whole-map trait; it needs a
certified universal output guarantee. Per-axis annotated subspaces must not
reappear implicitly.

## 6. Binding

`bind` contracts the output axis of one extensor against an input axis of
another. It covers partial application, full application, and tensor
composition.

### Slot convention

The recommended public convention numbers only logical input slots:

```text
logical input slot i <-> physical structural axis i + 1
```

Thus output axis zero is never a valid public bind target. This avoids the
off-by-one ambiguity in the older output-first proposal.

For:

```text
F.axes = (O, A, B)
G.axes = (A, C, D)
```

binding `G` into input slot zero of `F` produces:

```text
F.bind({0: G}).axes = (O, C, D, B)
```

The bound extensor's remaining inputs are spliced into the position of the
contracted slot. Binding a nullary extensor splices no inputs and simply removes
the slot.

### Compatibility

For target input subspace `I` and bound extensor output subspace `X`:

```text
X ⊆ I
```

is required. Missing components are interpreted as structural zeros through the
same widening/select machinery used today.

The initial rewrite imposes structural slot compatibility only. A future
refined slot requirement on a nullary argument may additionally require that
argument's `GAType` to entail specific traits. For a positive-arity argument,
the corresponding check would require a certified universal output guarantee,
not an unrelated whole-map trait.

Binding compatibility is therefore not the same operation as comparing the
complete types of target and argument: their arities generally differ, and the
target consumes only the argument's structural output plus any explicitly
defined output guarantee.

### Result construction

Binding performs the following steps:

1. validate slot indices, algebra, structural containment, traits, and contexts;
2. determine the result axes and their order;
3. specialize the target's and argument's relational traits;
4. construct and intern the resulting whole-object `GAType`;
5. contract the relevant kernel axes;
6. broadcast concrete batch dimensions on the left;
7. return a new `Extensor` in the context selected by the staging rules.

Structural output support is deduced from exact symbolic structure. Accidental
zeros in floating-point runtime data never narrow a `SubSpace` or `GAType`.

Backend eager semantics require the returned `Extensor` to be materialized.
ExactContext binding instead constructs an exact Extensor. Neither requires a
sequence of oversized dense intermediates: one bind operation may plan and fuse
its internal contractions before producing the result.

### Multi-slot binding

A multi-slot bind is atomic:

```text
F.bind({0: x, 2: y})
```

The indices refer to the original input slots, not to a successively shrinking
intermediate. Mapping iteration order must not affect the result.

Result input ordering is defined by walking the original target inputs from
left to right. Each unbound target input is retained; each bound input is
replaced in place by that operand's remaining inputs, in their existing order.
The execution plan is responsible for any contraction and permutation needed
to realize that canonical order.

The implementation and tests must additionally define:

- whether simultaneous and sequential binding are equivalent when no diagonal
  relation is involved;
- how batch broadcasting failures are reported;
- how duplicate or already-bound slot indices are rejected.

### Calling and composition

`bind` is the only primitive implementation. Convenient syntax delegates to it:

```text
F(x, y)       == F.bind({0: x, 1: y})  # full bind
F.compose(G)  == F.bind({slot: G})      # named convenience where useful
```

`__call__` should remain full-application sugar because unary map calls are
pervasive and readable. Partial calls should not create a subtly different path;
partial application remains explicit through `bind`.

## 7. Repeated arguments

The unrestricted structural sandwich operator can be defined as:

```text
S(left, passenger, right) = left * passenger * reverse(right)
```

The reversal is precomposed into the right kernel slot, so both motor inputs
accept values from the same structural carrier. On the diagonal:

```text
S(m, passenger, m) = m * passenger * reverse(m)
```

The unrestricted ternary kernel still accepts distinct `left` and `right`
values. A future extensional diagonal rule may state what can be proved when
specified input slots receive the same certified nullary value. Its concrete
representation is intentionally not named yet.

Repeated arguments and symmetric slots are different concepts:

```text
diagonal evaluation          a relation among bound arguments
symmetric slots              a property of the multilinear kernel
```

Symmetrizing a kernel is valid for diagonal evaluation, where the relevant
slots receive the same value. The unrestricted kernel remains unchanged;
symmetrization may be used internally by the execution plan for that diagonal
bind. A symmetrized kernel must not be exposed as the unrestricted ternary
operation unless its polarized semantics are made explicit.

For the initial rewrite:

- `m.sandwich(P)` performs the repeated motor binding atomically;
- simultaneous binding records same-nullary equality groups, but no diagonal
  trait rule consumes them yet;
- an unrestricted left/passenger/right operation retains unrestricted
  semantics;
- separately bound values do not become equal merely because they happen to
  have the same `GAType`;
- trait inference may use observed same-object binding, never coincidental
  numerical equality.

Binding the same positive-arity extensor into two slots does not identify that
extensor's open arguments. Its input axes are spliced twice as independent
variables, preserving multilinearity. An expression such as `S(G(c), p, G(c))`
that identifies those variables is polynomial rather than multilinear in `c`
and requires a separate expression or diagonalization layer; it is deferred.

A later slice may represent reusable diagonal implications as structured,
extensional traits. Its name and representation remain deferred; it will be a
fact about the resulting extensor, not expression history.

## 8. Traits during binding

The details of traits, entailment, and dynamic dispatch live in
[`gatype_design.md`](gatype_design.md). At the extensor layer, the essential rule
is conservative construction:

```text
traits(result)
    = facts proved by structural deduction
    ∪ facts proved by the operation's relational traits
    ∪ facts proved by the bound argument GATypes
```

The first implemented conditional relations are concrete:

```text
G = geometric_product(A, B)
G.gatype.traits includes ProductRelation(ReverseProduct, ONE, slots=(0, 1))
G.gatype.traits includes VersorProduct(slots=(0, 1))

G.bind(unit_a)          -> ProductRelation(ReverseProduct, ONE, slots=(0,))
G.bind(unit_a, unit_b)  -> ReverseProductOne
```

Positive-arity operands substitute their own relations; `BindingPlan` remaps
their input slots into the residual extensor. If any referenced operand lacks a
certified scalar reverse product or a substitutable relation, the conclusion is
erased. Generic `Maps`-style relations remain deferred.

Trait transfer is property-specific. There is no generic rule that an operation
preserves the passenger `GAType` verbatim.

## 9. Deferred sandwich typing target

The structural sandwich exists, including atomic repeated binding. Its
`Isometry` and passenger-preservation inference does not. The examples below
describe the intended later result, not current emitted traits.

The structural sandwich operation has axes:

```text
(P, M, P, M)  # output, left motor, passenger, right motor
```

The two motor inputs are distinct in the unrestricted ternary operation, whose
right-input kernel already includes reversal. Its diagonal trait states the
consequences of binding one nullary motor value into both motor slots.

Given:

```text
m.gatype = GAType((Even,), {Versor, ReverseProductOne})
```

the named construction performs that diagonal bind and produces a unary
extensor:

```text
A = m.sandwich(P)
A.gatype.subspaces == (P, P)
A.gatype.traits includes Isometry(...)
```

The isometry is now an intrinsic, extensional fact about `A`; the map need not
remember that it originated in a sandwich. Applying `A` specializes its result
traits according to the argument:

```text
A(generic_p) -> generic nullary P
A(null_p)    -> nullary P with ReverseProductZero
A(unit_p)    -> nullary P with ReverseProductOne
```

Only facts justified by the map's certified traits transfer. In particular,
`Isometry` must not silently imply `InvertibleMap` in a degenerate metric.

## 10. Extensor arithmetic and collection behavior

Nullary and positive-arity extensors share collection behavior. Every operation
acts on leading batch dimensions while preserving or recomputing the structural
type.

Required operations include:

- batch indexing and slicing;
- functional indexed update;
- reshape and broadcast of batch dimensions;
- stack and concatenate;
- batch `sum` and `mean`;
- unary negation and scalar multiplication;
- addition and subtraction of compatible extensors.

`__getitem__`, `reshape`, and reductions address batch axes only. Structural
kernel axes should not be reachable accidentally through ordinary collection
syntax.

All members of a batch share one `GAType`. A certified trait therefore applies
to every member of the batch. Combining heterogeneous batches produces the
least sound common refinement representable by the trait system.

Examples:

```text
index/reshape/broadcast
    construct a new extensor with the same semantic traits

concatenate
    retains only facts guaranteed for every input batch

sum/mean/addition
    infer a new type; unit, blade, or isometry do not survive by default
```

This is result-type inference, not mutation or trait invalidation.

`*` is geometric-product construction at every arity. For positive-arity
operands it multiplies their outputs and uses `bind` to splice their residual
inputs into the new expression in operand order. Thus `V * V * V` constructs a
ternary Extensor, while supplying values reduces its arity through the same
binding path.

## 11. Context and backend execution

A `Context` supplies:

- the coefficient representation and array/exact execution implementation;
- dtype and device policy where the representation has them;
- an exact, dense, einsum, or sparse-unrolled execution strategy;
- materialized-kernel and execution-plan caches;
- construction of extensors in that context.

Exact operation Extensors use the algebra's `ExactContext` and can be shared by
backend contexts. Lowering converts an exact Extensor's symbolic kernel into a
backend representation without changing its Python class, structural meaning,
or certified `GAType`.

Dense and sparse execution are alternative plans for the same bind operation;
they must have identical public semantics. Dense canonical symbolic storage is
the initial design. Sparse canonical storage remains an independent performance
question.

### JAX and Torch

JAX pytree flattening must be explicit and stable:

```text
dynamic leaf: backend kernel
static metadata: GAType plus an immutable context/policy key
```

A mutable context cache should not itself become hashed static metadata.
Accessing cached properties must not change the pytree structure.

Automatic differentiation creates new mathematical values. Tangents and
cotangents must not inherit coefficient-sensitive primal traits such as
`ReverseProductOne` merely because JAX reconstructs the same container shape.
Ordinary pytree reconstruction cannot determine that a value is a tangent, so
the JAX integration needs either a distinct tangent representation, suitable
custom differentiation rules, or another explicit trait-erasure boundary.

Torch follows the same semantic contract for eager execution and autograd.
Whether Torch compilation is a supported goal should be stated separately.

The existing pure-Python backend must either conform to the shared backend tests
or be explicitly removed from the supported 2.0 surface.

## 12. Caches

Cache keys reflect the staging layers:

```text
GAType flyweight:
    canonical (subspaces, traits)

exact operation/kernel cache:
    operation + structural subspaces

typed exact Extensor cache:
    symbolic kernel + whole GAType

dispatch cache:
    operation + complete actual GATypes

execution-plan cache:
    selected implementation + symbolic kernel
    + bound-slot/operand signature + result permutation
    + backend + dtype + device + strategy
    + batch rank/layout when compilation depends on them
```

Traits may select a semantic overload without forcing duplication of identical
structural multiplication tables. Concrete value-specific extensors are never
stored in global algebra or type caches.

Whole-`GAType` overload dispatch selects backend-independent mathematical
implementations. Backend-, layout-, and device-specific choices belong to
execution-plan selection. Mutating an overload registry invalidates its exact
resolution cache.

The exact rational representation remains open. An integer numerator array plus
a shared denominator is the leading compact option; Python `Fraction` arrays are
better suited to a correctness-first reference implementation than to canonical
production storage.

## 13. Public surface sketch

The minimum shared surface is approximately:

```text
e.context
e.gatype
e.axes
e.arity
e.output_subspace
e.subspace             # universal alias for output_subspace
e.input_subspaces
e.shape                 # batch shape
e.dtype

e.bind(...)
e(...)                  # full-bind sugar
e[index]                # batch indexing
e.at[...]               # functional update
e.reshape(...)
e.broadcast_to(...)
e.sum(...)
e.mean(...)
```

There is no arity-dependent meaning for `.subspace`: it universally aliases the
structural `.output_subspace`. It never means an output `GAType`.

Arity-dependent operations should be named carefully. In particular, Clifford
inverse of a nullary value and linear inverse of a unary map are distinct
operations even if arity-based dispatch could technically put both behind
`.inverse()`. `linear_inverse()` is the clearer provisional name for the latter.

## 14. Errors and diagnostics

The implementation should report type and staging mistakes before handing an
invalid contraction to a backend. Errors should include:

- algebra or context mismatch;
- invalid bind slot;
- output subspace not contained in the target input subspace;
- missing required traits;
- incompatible batch broadcast;
- unsupported arity for an operation;
- ambiguous whole-`GAType` dynamic dispatch;
- attempts to mutate an extensor or its certified metadata.

Diagnostics should print the relevant axis tuple and traits, not only backend
array shapes.

## 15. Migration phases

The implementation should be staged so each high-risk change has an independent
correctness boundary.

### Phase 0: baseline

- checkpoint or otherwise isolate the existing dirty worktree;
- prefer direct expected results over opaque captured fixtures;
- replace selected print-only and disabled cases with direct native assertions;
- encode explicit expected behavior for representative operations and examples;
- separate performance benchmarks from correctness tests.

### Phase 1: exact symbolic kernels

- introduce the rational kernel representation and `ExactContext` storage and
  execution primitives, exercised directly until the Extensor shell lands;
- do not introduce an interim exact tensor value class;
- make squeeze and symbolic cancellation exact;
- test fractional symmetry coefficients and exact composition;
- benchmark construction, storage, and backend materialization.

### Phase 2: output-first layout

- flip symbolic and backend kernels in isolation;
- use unequal input and output dimensions so accidental transposes cannot pass;
- cover bind, compose, squeeze, inverse, dense execution, and sparse execution.

### Phase 3: whole-object GAType shell

- introduce the interned `GAType(subspaces, traits)` with initially minimal
  traits;
- mirror declared lowercase SubSpace constructors on `GATypeFactory` as
  empty-trait nullary lifts, while reserving explicit semantic constructors
  such as `gatype.rotor()` for refined types;
- introduce the one output-first `Extensor` class with `ExactContext` and a
  complete `GAType`;
- make `OperatorFactory` return those exact Extensors;
- establish comparison, hashing, and canonicalization laws;
- keep `SubSpace` structural.

### Phase 4: NumPy Extensor and bind

- merge concrete multivector/operator behavior into the immutable `Extensor`;
- implement implicit SubSpace lifting, typed expressions, eager bind,
  higher-arity composition, and full-call sugar;
- define batch indexing, functional updates, reductions, and concatenation;
- migrate the Levi-Civita, complex product, inertia, and sandwich examples.

### Phase 5: traits and dispatch

Status: the first marker vocabulary, implication/refinement, whole-GAType
dispatch, focused product relations, and default unit/orthogonal inverse
overloads are implemented. Broader map, diagonal, and audit relations remain
deferred.

- implement the first trait vocabulary and implication closure;
- specialize traits through bind and composition;
- dispatch on complete GATypes;
- reject shadowing registration order and diagnose incomparable overlaps;
- add optional audit checks without inspecting runtime zeros for ordinary type
  inference.

### Phase 6: JAX

- implement explicit stable pytree behavior;
- test eager execution, `jit`, `vmap`, gradients, and compilation reuse;
- verify that transformed values receive sound traits;
- run a deterministic headless physics integration test.

### Phase 7: Torch and backend policy

- port the shared backend conformance suite to Torch;
- test autograd and any promised compilation path;
- decide whether the pure-Python backend is ported or retired.

### Phase 8: release surface

- define the public import facade and optional backend dependencies;
- convert documentation snippets into executable examples;
- publish a breaking-change migration guide;
- remove obsolete implementation paths only after replacement coverage exists.

## 16. Behavioral contracts and tests

The shared conformance suite should cover:

### Representation

- constructor shape invariants at arities zero, one, and greater than one;
- output-first layout with unequal structural dimensions;
- immutable operations and stable input objects;
- consistent `axes`, `arity`, `shape`, and `output_subspace` properties;
- implicit `SubSpace` lifting gives `V * V`, `x * V`, and `x * y` arities two,
  one, and zero without constructing phantom values.

### Binding

- nullary, partial, full, and higher-arity binding;
- input-splice order;
- mapping-order independence for multi-bind;
- sequential versus simultaneous bind where equivalent;
- repeated-argument diagonal binding;
- structural subset widening and clear incompatibility errors;
- context checks and left-broadcast batch behavior.

### Staging

- ExactContext-Extensor composition remains exact;
- the first backend value lowers exact Extensors into its compatible context
  without changing value class;
- factory-produced exact values and backend-produced values have exact Python
  class identity;
- materialized kernels agree with direct GA evaluation;
- dense and sparse plans return identical results.

### Collections

- batch indexing, reshape, broadcast, stack, concatenate, sum, and mean;
- traits shared soundly across batched values;
- structural axes inaccessible through accidental batch indexing.

### Integration

- Levi-Civita/cross-product matrix construction;
- complex-number multiplication as a unary extensor;
- summed inertia maps and their linear inverse;
- future: unit-motor sandwich producing a certified isometry;
- NumPy/JAX/Torch agreement within each backend's numeric tolerance.

## 17. Removed implementation distinctions

Once replacement behavior is covered, the rewrite removes:

- parallel `AbstractMultiVector` and `AbstractConcreteOperator` hierarchies;
- duplicated backend collection logic for multivectors and concrete operators;
- distinct implementation paths for `partial`, `fuse`, `__call__`, and named
  `*_map` helpers;
- the backend-specific `isinstance(SubSpace)` call hack;
- floating-point epsilon-based symbolic squeeze after exact kernels land.

`__call__` may remain as syntax, but delegates to `bind`. Named operations such
as `sandwich` may remain as semantic factories while their dedicated concrete
map classes or execution paths disappear.

The separate symbolic `Operator` value is removed. `OperatorFactory` deliberately
remains as a factory whose results are Extensors.

## 18. Open questions

The following details are intentionally unresolved:

- the public coefficient attribute name: `kernel`, `data`, or another single
  term;
- the final public slot-indexing syntax for `bind`;
- the exact representation and specialization API for additional extensional
  diagonal traits;
- whether refined input requirements live inside `TraitSet` or an adjacent
  output-contract system, after the initial structurally typed version;
- whether nullary and unary inverse share one name;
- the concrete context-compatibility and immutable context-key rules;
- the backend-specific mechanism that enforces deep immutability for NumPy and
  Torch buffers;
- the JAX tangent/cotangent representation and trait-erasure boundary;
- exact rational kernel storage;
- supported status of the pure-Python and compiled Torch backends.

These questions do not change the core separation:

```text
SubSpace   structural support
GAType     whole-extensor semantic type
Context    coefficient representation and execution stage
Extensor   the immutable value class in every context and at every arity
OperatorFactory   operation construction, not a value class
bind       the common contraction/composition primitive
```
