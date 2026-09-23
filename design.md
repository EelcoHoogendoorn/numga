# Integrated Numga 2.0 Rewrite Plan

## Scope: three changes, not a numerical-library reinvention

This rewrite has exactly three objectives, in priority order:

1. Unify multivectors and operators as Extensors.
2. Add traits and whole-extensor GAType inference and dispatch.
3. Allow configurable SubSpace layouts beyond lexical blade ordering.

Everything else is a port of existing Numga functionality, not permission to
invent new requirements. This scope boundary takes precedence over broader or
aspirational suggestions elsewhere in the design documents.

- Read and deliberately reuse the original source. Preserve its established
  mathematical algorithms, numerical behavior, and sensible support/layout
  choices. Do not substitute newly invented approximations for verified math.
- No unrelated features, speculative edge cases, or general-purpose machinery
  without an explicit user request. Supporting the three changes above is the
  reason to alter architecture; the rewrite itself is not a reason to expand scope.
- No repeated runtime work for facts derivable from GAType statics. Resolve
  structural decisions during construction or cached dispatch, not inside
  numerical implementations.
- Warm NumPy calls matter independently of JAX compilation. Cache binding
  plans, coordinate relationships, contraction choices, and the resulting
  flyweight GAType from the incoming types (plus repeated-operand evidence
  where a law needs it). Propagating input traits is cache-miss work, not work
  repeated for each coefficient array.
- Full application caches the executable recipe, not just its constituent
  plans. Warm concrete unary calls look up a flat context/GAType key, run the
  selected array contraction, and attach the cached result type. They do not
  normalize bindings, detect repeated operands, resolve contexts, or hash
  nested binding plans. General partial binding remains a separate path.
- Computed results attach the cached GAType and context to their array directly:
  no repeated public-input validation, dtype coercion, or array copy. NumPy
  results remain read-only. Public construction still owns external arrays
  and checks their structural shape; internal execution trusts the static plan.
- Let array operations handle batch broadcasting and ordinary argument errors.
  Dense contractions use ellipses; ordinary unary matrix composition selects
  backend `matmul` once. Do not add runtime assertions of planner invariants.
- Static queries and result-trait inference belong on GAType as cached,
  named properties. Extension modules contain dispatch declarations and the
  mathematical implementations, not local blade-mask analysis or type helpers.
- Use named mathematical methods in extensions, not string-selected transforms
  or wrappers around existing methods. Let whole-GAType queries enforce their
  arity; do not repeat arity guards already implied by those queries or traits.
- No faux defensive programming: no gratuitous argument/type checks, numerical
  trait validation, or silent input repair. Use type annotations, declared
  preconditions, and the underlying operations' natural failure behavior.
- Respect numerical intent. Consumers trust declared input traits; they do not
  silently normalize inputs. Explicit norm measurements and normalization
  always execute, even when traits already claim unit input.
- Keep the original library untouched as source reference. No legacy-package
  imports, subprocess comparisons, or compatibility scaffolding in the rewrite.

No backwards compatibility is required. That permits API and representation
changes; it does not authorize replacing proven mathematics or discarding
useful existing behavior unrelated to these three objectives.

### Signed-layout implementation boundary

The first layout slice retains canonical bit-mask products and folds input and
output orientation signs into generated kernels. SubSpace/GAType identities and
caches include exact layouts; semantic GAType `<=` and ordinary dispatch ignore
layout. Exact-axis predicates select coefficient-specific overrides. Binding
inserts cached signed conversions when necessary; it never reinterprets buffers.
See [subspace_design.md](subspace_design.md) for the implemented spelling,
default-factory, restriction, and PGA3D conventions.

Autodiff considerations elsewhere in these plans are optional future work, not
a requirement or acceptance gate for the three rewrite objectives.

### Array execution port

NumPy and JAX contexts now accept `execution="dense"` (default) or `"sparse"`.
Sparse execution reuses the original nonzero-term approach on exact kernels;
term grouping, open-axis splicing, and signed coordinate conversions are cached.
Computed numeric maps still execute densely, without coefficient scanning.
Execution policy belongs to the context key, not to GAType or its traits.
Both policies return the same Extensors; JAX tracing retains the selected policy.

Core extension results now use the trusted constructor. Trait refinement,
transpose result types, and exact output restriction are cached. The original
mathematical formulas and explicit normalization behavior are unchanged.

## Type annotations are a library-wide baseline

All code under `rewrite/src/numga/` carries basic parameter and return
annotations, including private helpers, properties, operators, and nested
functions. Omit redundant `self`/`cls` annotations; lambdas get their types
from the surrounding callable contract. New code follows this convention
without needing a separate request for each method.

- Use concrete domain types such as `Extensor`, `GAType`, `SubSpace`, `Algebra`,
  and `Context`; expose the actual result types of factory accessors and
  expression methods so they remain useful in the IDE.
- Type collections and callables where their contents or results matter.
  Use `object` for deliberately unrestricted inputs that the implementation
  inspects; reserve `Any` for genuinely dynamic backend arrays/namespaces and
  user-defined extension results. Do not use either to hide known domain types.
- Use postponed annotations and `TYPE_CHECKING` imports where needed to avoid
  import cycles and keep optional backends optional. Remain compatible with
  the rewrite's declared Python version.
- Annotations document contracts; they do not add runtime validation, defensive
  checks, numerical work, or new dispatch mechanisms. Keep them readable rather
  than introducing speculative typing or shape frameworks.

The existing rewrite library has received this annotation sweep. Future edits
must maintain it. The original library remains untouched reference material.

Status: working integration plan for the non-backwards-compatible rewrite.

This document coordinates the three detailed designs:

- [`extensor_design.md`](extensor_design.md): runtime representation, staging,
  binding, and collection behavior;
- [`gatype_design.md`](gatype_design.md): whole-extensor typing, traits,
  inference, and dispatch; and
- [`subspace_design.md`](subspace_design.md): exact coefficient layouts,
  oriented blades, casts, and algebra defaults.

[`extensors.md`](rewrite/docs/extensors.md) remains the motivating user-facing essay.
[`numga2_design.md`](numga2_design.md) is an earlier snapshot and is now
historical where it conflicts with the three focused documents or this
integration plan.

This document is authoritative for implementation order and integration seams.
The focused documents remain authoritative for the detailed semantics of their
individual concerns.

## 1. Priorities and conclusion

The development priorities are:

1. unify exact and backend multivectors and operators as `Extensor`;
2. add whole-object `GAType`, traits, inference, and dispatch; and
3. add ordered, oriented SubSpaces and configurable defaults.

There is deliberately **no backwards-compatibility objective**. The rewrite
does not preserve old classes, method spellings, import paths, serialization,
or extension registration APIs unless a particular idea is independently
chosen again on its merits.

The recommended strategy is neither a monolithic rewrite nor three naive,
independent rewrites.

> Establish a small final-form architectural spine, then implement the three
> concerns incrementally in priority order.

The spine consists of:

- output-first structural axes;
- immutable `Extensor` semantics;
- one `Extensor` value class, with exact symbolic storage owned by an
  `ExactContext`;
- an `OperatorFactory` that constructs operations but is not a value class;
- an interned `GAType(subspaces, EMPTY_TRAITS)` shell;
- one centralized `BindingPlan`;
- one centralized `AxisTransform`/conversion interface; and
- separate caches for symbolic structure, types, dispatch, and backend plans.

The GAType shell is not an early implementation of the trait system. The axis
conversion interface is not an early implementation of signed SubSpaces. They
are small permanent seams that prevent the Extensor implementation from being
rewritten when those later concerns arrive.

The guiding phrase is:

> Stage the risks; integrate the seams.

### 1.1 Parallel development tree

The rewrite is built beside the working implementation rather than through
incremental demolition of it. The recommended development topology is:

```text
numga/                      working 1.x source retained as reference material

rewrite/
    pyproject.toml          isolated build, dependency, and test configuration
    src/numga/              final package name, entirely separate source root
        algebra/
        subspace/
        gatype/
        operator/
        extensor/
        backend/
    tests/                  self-contained native 2.0 tests
```

This is the development topology. Phase 5 performs one atomic repository-level
promotion after the rewrite is independently releasable.

Using the final `numga` import name inside an isolated source root avoids a
disposable `numga2 -> numga` rename. It also means the old and new packages
cannot be imported accidentally into the same interpreter.

Production code and tests under `rewrite/` must not import implementation
modules from the root `numga/`. The old source is readable reference material,
not a runtime or test dependency, compatibility layer, or base-class library
for the rewrite. Retained behavior is expressed directly as native 2.0 tests
with explicit expected results.

All rewrite tooling runs with `rewrite/` as its project root, and an import
origin assertion verifies that tests loaded `rewrite/src/numga`, not the sibling
legacy package.

## 2. Why the concerns cannot be completely independent

The final ownership relation is:

```text
SubSpace
    exact interpretation of one coefficient axis

GAType
    ordered tuple of SubSpace axes + whole-extensor traits

Context
    coefficient representation and execution policy

Extensor
    Context + GAType + context-owned kernel

OperatorFactory
    constructs operation Extensors; it is not itself a tensor value
```

Therefore `Extensor` logically depends on `GAType`, and `GAType` logically
depends on `SubSpace`. That does not mean their full feature sets must be
implemented bottom-up.

The implementation can deepen each layer over time:

| Epoch | SubSpace capability | GAType capability | Extensor capability |
|---|---|---|---|
| Extensor | rewrite's initial unsigned canonical-mask layout behind a stable axis API | interned axes with empty traits | complete core semantics and NumPy reference runtime |
| Traits | unchanged | traits, inference, and dispatch | unchanged except richer inferred types |
| Layouts | ordered masks, signs, casts, custom defaults | exact layouts plus layout-neutral facts | unchanged except richer conversion plans |

This is the central incremental strategy. Lower layers begin thin but already
have their final ownership boundaries.

## 3. Final architectural spine

### 3.1 Extensor

`Extensor` is the one context-bound value at every arity and every execution
stage:

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

Structural axes are output-first:

```text
kernel.shape
    == batch_shape
     + tuple(len(axis) for axis in gatype.subspaces)
```

The complete `batch_shape`, exposed as `extensor.shape`, participates in normal
backend broadcasting for arithmetic and binding. Structural axes do not.
Exact-context kernels have empty batch shape and therefore broadcast as
constants after lowering into a backend context.

Addition first zero-embeds same-arity operands into the per-axis unions of their
SubSpaces, then applies ordinary broadcasting to their leading `.shape`s.
Structural union and batch broadcasting are deliberately independent steps.

The new runtime must use this layout from its first implementation. Building a
new output-last Extensor and transposing it later would spread disposable axis
logic through binding, indexing, inversion, batching, and every backend.

`Extensor` has immutable value semantics before certified traits exist. This is
foundational: retrofitting immutability after traits would create a period in
which type facts could be invalidated by aliased mutation.

It does not define a universal coefficient-level `equals` operation. Exact
symbolic equality and backend numerical closeness have different policies;
tests and callers select an explicit kernel comparison and, for floating-point
backends, explicit tolerances. `GAType` equality remains exact and structural.

### 3.2 ExactContext and OperatorFactory

There is no separate symbolic tensor value class. Exact and backend-stage
tensors are both `Extensor` instances; the context is the staging boundary:

```text
ExactContext Extensor
    gatype: GAType
    kernel: exact symbolic kernel

backend Context Extensor
    gatype: GAType
    kernel: backend array
```

`ExactContext` is bound to an algebra and owns exact rational coefficient
preparation, exact contraction, and lowering into a backend context. Exact
composition therefore remains symbolic without changing Python value class.
When backend data participates, exact operands are lowered into the selected
compatible backend context and execution remains there.

`OperatorFactory` remains the algebra's operation-construction and caching
surface. It returns `Extensor` values, normally in `ExactContext` when all
positions are type-only or exact; it is not an `Operator` value or a second
staging hierarchy.

`SymbolicKernel` provides the permanent exact construction, composition,
squeeze, and materialization representation used by `ExactContext`. Its initial
storage may favor correctness over compactness, but it must not use floating
epsilon semantics; storage can later be optimized without changing `Extensor`.
The correctness-first storage remains a valid reference implementation rather
than disposable scaffolding.

### 3.3 Minimal GAType shell

Every `Extensor`, including one in `ExactContext`, stores a whole-object GAType
from day one:

```text
GAType(
    subspaces=(Output, Input1, ..., InputN),
    traits=EMPTY_TRAITS,
)
```

`axes` is always derived from `gatype.subspaces`; it is not separately stored as
primary metadata. Likewise, `.subspace` remains a convenience alias for the
output SubSpace and is not a second type field.

Initially:

- `TraitSet` has only its canonical empty value;
- structural binding splices SubSpaces and returns another empty-trait GAType;
- type-inference hooks exist but prove no value-level facts; and
- GAType interning already uses the final `(subspaces, traits)` shape.

Later trait work populates these hooks. It does not migrate Extensor from one
metadata model to another.

A `SubSpace` implicitly lifts in a typed expression position to its unrefined
nullary type:

```text
S  -> GAType((S,), EMPTY_TRAITS)

V * V  -> exact Extensor, arity 2
V * V * V -> exact Extensor, arity 3
x * V  -> Extensor, arity 1
x * y  -> Extensor, arity 0
```

This lift supplies the type of an unbound value position to `OperatorFactory`;
it does not manufacture an Extensor or phantom coefficients. A `GAType` can be
used directly when a refined typed position is required.

### 3.4 AxisTransform

No new binding or execution code should compare raw `.blades`, manufacture
padding directly, or assume a bit mask is a coefficient index. It asks one
planner for the relationship between the actual output axis and the required
input axis:

```text
AxisTransform.plan(source, target) ->
    exact
    relayout
    embed
    project
    reframe
    incompatible
```

The meanings are:

- `exact`: identical coefficient interpretation;
- `relayout`: equal support, lossless permutation/sign conversion;
- `embed`: narrower support into a wider input, with structural zero fill;
- `project`: wider support into narrower support, losing information;
- `reframe`: arbitrary support change, combining projection and zero fill; and
- `incompatible`: no permitted structural conversion.

Binding permits `exact`, `relayout`, and sound zero-fill `embed`. Projection is
never implicit. The public explicit `cast(target_subspace)` accepts every
compatible target layout: it may relayout, embed, project, or reframe. Project
and reframe can lose values and therefore use conservative trait-transport
rules. `incompatible` is reserved for axes from algebra/frame systems for which
no coordinate correspondence has been defined.

Within one algebra and basis frame, every target SubSpace is compatible,
including a disjoint target whose explicit cast is the zero value. Cross-frame
casts require a separately registered coordinate correspondence.

During the Extensor epoch, the rewrite's initial unsigned SubSpaces have one
canonical-mask layout, so `relayout` collapses to `exact`. During the
signed-SubSpace epoch, the same interface begins returning signed permutations.
The binding algorithm itself does not change.

An AxisTransform is executable metadata, not merely an enum. Depending on its
kind it carries the coordinate indices, signs/scales, and zero-fill information
needed to perform the conversion, or denotes an exact unary conversion
operator. Backend code consumes this plan rather than reconstructing padding or
selection itself.

The following stable support API also lands during the Extensor epoch, backed by
the rewrite's initial canonical masks:

```text
subspace.support_key
subspace.same_support(other)
subspace.support_is_subset_of(other)
```

The signed-SubSpace epoch changes exact layout representation, not the meaning
of these support queries.

### 3.5 BindingPlan

`bind` delegates planning before it performs contraction:

```text
BindingPlan:
    original target input-slot indices
    target and operand GATypes
    AxisTransform for every contracted axis
    result SubSpaces and residual-axis order
    operand-input splice order
    batch broadcast plan
    same-object argument partition
```

Same-object groups are recorded but are not yet consumed by trait inference;
they are reserved for a later repeated-argument sandwich rule. The implemented
Geometric-product relation substitution instead uses structural input splices
and certified GATypes, never object identity.

Only repeated **nullary** operands constitute diagonal equality evidence.
Binding the same positive-arity Extensor into two slots splices two independent
copies of its residual inputs; object identity does not identify those new
inputs.

The plan is deterministic and independent of mapping iteration order. Logical
input slot `i` maps to physical structural axis `i + 1`.

The plan contains no concrete operand arrays or object references. Same-object
evidence is reduced to a small equality partition such as `(0, 2)`. After the
structural plan is built:

```text
result_gatype = TypeRules.bind(binding_plan)
execution_plan = context.plan(binding_plan, result_gatype, backend_metadata)
```

This avoids a cycle in which BindingPlan both contains and is needed to compute
the result GAType.

Two hashable, value-free signatures key reusable work:

```text
TypeBindingSignature
    target and operand GATypes
    + bound slots and AxisTransforms
    + same-object equality partition
    + residual and splice order

ExecutionSignature
    TypeBindingSignature
    + immutable context/policy key
    + backend, dtype, device, static batch shapes/broadcast strategy
```

The type-level signature is sufficient for TypeRules. The execution signature
is used for lowered plans. Concrete Extensors are passed to execution and are
never retained by a global cache.

### 3.6 Type-inference hooks

Every constructor and operation delegates result typing to pure metadata
functions even while traits are empty:

```text
TypeRules.operation(op, operand_gatypes, result_subspaces)
TypeRules.bind(binding_plan)
TypeRules.collection(op, operation_parameters, operand_gatypes)
TypeRules.transform(gatype, transforms_by_axis)
```

The initial shell returned structurally correct GATypes and erased algebraic
and binding traits. The implemented narrow trait slice now attaches
family-specific `ProductRelation`s and `VersorProduct` to geometric product and substitutes
them through binding and composition. Pure batch selection, reshape, broadcast, stack, and
concatenation retain identical explicit traits; unrelated operations,
reductions, arbitrary updates, and axis transforms still erase facts they
cannot prove. Numeric contraction and collection code remain unchanged.

### 3.7 Cache boundaries

The following caches remain separate from the start:

```text
SubSpace flyweight
    exact coordinate-axis representation; strong pool owned by the algebra's
    SubSpace factory

GAType flyweight
    exact SubSpaces + canonical traits; strong pool owned by the algebra's
    GAType factory

symbolic kernel cache
    canonical algebra/operation parameters + support keys

layout-realized symbolic kernel cache
    canonical kernel + exact ordered/oriented axes

type-inference cache
    operations: rule + complete input GATypes + result SubSpaces
    binding: rule + TypeBindingSignature

dispatch cache
    operation + complete actual GATypes + registry generation

backend execution-plan cache
    selected implementation + ExecutionSignature + exact result layout
```

This permits many refined GATypes to share a structural kernel while preventing
a kernel or execution plan from being reinterpreted under the wrong axis
layout.

The `Extensor` allocation path does not own a process-global value cache or
override allocation to intern instances. Class-level `ExtensionMethod`
descriptors deliberately own overload registrations and their resolution
caches; they never cache concrete Extensor values. Structural metadata and
exact kernels retain structural equality and hashing; factory identity is a
safe fast path, not a correctness requirement.

The support-keyed cache has a strict invariant: its kernels are always
expressed in one deterministic canonical bit-blade basis, never in the exact
axes of whichever request populated the cache first. Materialization follows:

```text
operation + supports
    -> build/cache canonical-support kernel
    -> realize requested order and signs
    -> build/cache backend execution plan
```

The layout-realization step is the only place where request-specific coordinate
positions enter a cached generated symbolic kernel.

## 4. What should and should not land together

### 4.1 The Extensor foundation is one atomic cluster

These pieces should be introduced together:

- output-first new runtime layout;
- immutable Extensor;
- ExactContext and backend Context staging within that one value class;
- implicit SubSpace-to-nullary-GAType lifting in typed expression positions;
- the empty-trait GAType shell;
- BindingPlan and AxisTransform;
- OperatorFactory as a factory rather than a value boundary; and
- constructor shape validation.

Separating these would cause immediate representation churn in the highest
priority implementation.

### 4.2 Trait semantics begin with one complete vertical slice

The trait infrastructure and its first certified behavior belong together after
Extensor behavior is stable:

- canonical structured TraitSet values;
- entailment and contradiction validation;
- one structural and operation-specific inference path;
- the focused self-product `ProductRelation` and `VersorProduct` geometric-product
  relations;
- specialization of those relations through bind and composition;
- whole-GAType dispatch for that path;
- overlap/declaration-order diagnostics; and
- dispatch-cache invalidation.

Landing isolated trait flags before inference and dispatch understand them would
create apparently precise but unsound types. Once the first end-to-end slice is
sound, the trait vocabulary and transfer rules can expand incrementally rather
than arriving in one enormous switch.

### 4.3 Signed SubSpaces are a later atomic cluster

These belong together after trait behavior is stable:

- ordered canonical masks plus orientation signs;
- exact layout-sensitive equality and flyweight keys;
- explicit spelling that never reorders entries;
- one explicit cast surface for relayout, embedding, projection, and arbitrary
  support-changing reframing;
- sign-aware operator construction;
- exact-layout cache keys;
- the overridable default-layout constructor;
- the single-string default-layout format; and
- the importable configured `PGA3D`.

Enabling sign-bearing SubSpaces before every primitive operator and conversion
path has become sign-aware would silently compute incorrect coefficients. This
cluster should therefore be developed incrementally behind tests but switched
on atomically.

### 4.4 Work that need not block these priorities

The following are useful but orthogonal workstreams:

- compact exact-rational symbolic-kernel storage;
- sparse canonical kernel storage;
- execution-plan performance tuning;
- packaging and optional-dependency cleanup;
- non-diagonal metrics; and
- arbitrary non-blade coordinate bases.

Exact symbolic semantics are part of the Phase 1 Extensor/ExactContext spine.
Only the choice and optimization of a compact rational storage format is
orthogonal; a slower correctness-first implementation behind `SymbolicKernel`
is acceptable for the first working unified Extensor.

## 5. Implementation roadmap

Every phase ends in an executable checkpoint. Later phases enrich stable
objects rather than replacing them.

### Phase 0: establish an isolated baseline

Before changing representation:

- scaffold `rewrite/` with its own package, environment, and test collection;
- declare the 2.0 backend support matrix, including whether the current
  pure-Python backend is retained or retired;
- translate only selected behavior into direct native assertions instead of
  repairing or wrapping the legacy suite wholesale;
- write explicit expected results for representative products, partial
  binding, sandwich, inertia, and collection operations;
- use unequal structural axis lengths to expose accidental transposes; and
- distinguish semantic algebra-value comparisons from raw coefficient-layout
  comparisons.

Any retained raw-coordinate expectation must name its layout explicitly so a
later PGA3D default change cannot silently alter what the test means.

### Phase 1: build the Extensor spine as a NumPy vertical slice

Implement:

- one native output-first `Extensor` class used by both `ExactContext` and the
  NumPy context;
- a context-owned `multivector` construction namespace as a nullary-only
  facade over the general `Context.extensor` constructor;
- `ExactContext` and `OperatorFactory`, with exact operation construction
  returning Extensors rather than a separate symbolic value;
- the permanent `SymbolicKernel` interface with correctness-first exact rational
  storage and exact composition/squeeze;
- interned `GAType(subspaces, EMPTY_TRAITS)`;
- output-first immutable `Extensor`;
- typed-position normalization from `SubSpace` to
  `GAType((S,), EMPTY_TRAITS)`;
- the common batch/structural shape invariant;
- stable `support_key`, `same_support`, and support-containment queries on the
  rewrite's initial unsigned canonical-mask SubSpace;
- `AxisTransform` with `exact`, zero-fill `embed`, and `incompatible`;
- deterministic `BindingPlan`;
- nullary, unary, and higher-arity eager bind;
- full-call sugar over bind;
- one NumPy execution path;
- the type-inference hooks in their structural, empty-trait form; and
- only the low-level algebra routines and symbolic factories needed by this
  acceptance slice, copied or rewritten directly into their final APIs.

The first acceptance slice should demonstrate:

- an ordinary multivector and a partially bound matrix are the same Python
  runtime class;
- exact operation tensors and their NumPy-lowered values are that same Python
  runtime class;
- binding a nullary value reduces arity without a special nullary case;
- binding a positive-arity value splices its inputs in place;
- a cross-product/Levi-Civita example;
- complex multiplication as a unary extensor;
- an inertia map; and
- a motor sandwich map.

The inertia and sandwich examples may use existing named symbolic factories
as specifications, but their required mathematics is ported into the native
output-first factory rather than reached through an adapter. They prove the new
runtime representation and ordinary binding; general atomic multi-bind and
diagonal evidence arrive in Phase 2.

This is the first product milestone and directly serves the highest priority.

### Phase 2: complete unified Extensor behavior

Complete the concern before deepening the type system:

- complete typed expression syntax around that lift (`V * V`, `x * V`, and
  `x * y`);
- atomic multi-bind and mapping-order independence;
- declared repeated-argument binding;
- batch indexing, reshape, broadcast, stack, concatenate, sum, and mean;
- functional updates under immutable semantics;
- context compatibility and backend lowering;
- a backend-neutral execution interface;
- dense NumPy plan conformance; and
- at least one tracing-backend smoke path, provisionally JAX, proving the
  abstraction is not NumPy-specific.

The tracing smoke is an architectural gate even if that backend is not selected
for the released support matrix. Full parity for every selected backend and
execution strategy can proceed in a parallel lane and is a release gate, not a
prerequisite for beginning the trait epoch. Where JAX is present, static pytree
metadata is the GAType plus an immutable context/policy key, never a mutable
Context or cache object.

Only essential backend execution primitives are ported here. Specialized inverse,
normalization, logarithm/exponential, and decomposition overloads should not be
fully rewritten against a temporary structural dispatcher; they are ported
once during the GAType phase.

Further behavior is added by selective ports. A copied routine must either be
context-free and already compatible with the final invariants, or be translated
once at the copy boundary into output-first axes and the final metadata model.
Whole legacy modules are not copied merely to expose one useful function.

At the end of this phase:

- `rewrite/src/numga` has no runtime import or dependency on the root package;
- unsupported legacy features are simply absent from the 2.0 surface;
- every exposed operation uses the final Extensor/OperatorFactory path; and
- the root source remains unchanged as reference material.

### Phase 3: implement GAType traits and dispatch

Status: marker traits, implication/refinement, dispatch, shared self-product
propagation, and default reverse-unit/conjugate-unit/orthogonal inverse
overloads are implemented. The shared families are reverse and Clifford
conjugate products, not an unqualified notion of norm. Generic fact storage is
separate from explicitly installed mathematical propagation laws; see
[the self-product contract](gatype_design.md#42-shared-self-product-facts-explicit-mathematical-laws).
Repeated-argument versor sandwiches now narrow their output by grade before
execution and propagate passenger reverse-product and versor facts. Unit
Euclidean sandwiches with matching input/output support also produce
coefficient-orthogonal maps; composition retains that fact and inverse dispatch
uses transpose. This is a specific sandwich law, not general map inference.

The core library must supply generic, reasonably efficient norms,
normalization, logarithms, exponentials, and inverses for its supported domains.
These are default extension implementations, not work delegated to end users.
End users can register more specialized implementations for exact layouts and
backends through the same extension surface. Layout predicates already receive
the whole GAType; backend selection needs separate static execution metadata
and must not be smuggled into coefficient-dependent predicates or GAType facts.
The current descriptor has no backend qualifier. Output facts are carried by
the actual returned Extensor's GAType, not by a separate registration clause or
interpreted return annotation. The first default-method port now supplies
scalar/Study norms and normalization, nullary and unary inverses, and the
original quadratic/bisection bivector-exp/local-versor-log algorithms.
The invented analytic overloads were removed, not retained as optional ports.
Domain and precision limits are documented in
[`rewrite/README.md`](rewrite/README.md); this is not a claim of full legacy
coverage or large-algebra performance parity.

The working behavioral inventory is
[`rewrite/tests/test_end_to_end.py`](rewrite/tests/test_end_to_end.py). Design
propagation backwards from that modest set: structural and normalized value
facts; exp/log/norm guarantees; products and equivalent staging; grade-preserving
versor sandwiches; map composition/inversion and passenger facts; and collection
preservation versus reduction. In particular, a 5D general-even sandwich can
produce grades 1 and 5, while a known versor must construct a grade-1 codomain
before application. Low-dimensional closure alone does not establish that rule.
These examples specify behavior, not a mandatory relation language. Extend the
core or trait vocabulary only when an accepted example needs it. Autodiff trait
handling is optional later work and does not drive this phase.

The sandwich law requires the same nullary Extensor in both outer slots of one
binding, with a known `Versor` trait. Equal GATypes alone are insufficient.
It preserves passenger grades, not necessarily the passenger's exact blade
support. For `y = m * x * reverse(m)` and scalar passenger reverse product,
`y * reverse(y) = (m * reverse(m))**2 * (x * reverse(x))`: a unit sandwicher
preserves unit facts, while a scaled versor preserves nonzero facts without
claiming unit norm. These relations remain available on the resulting unary
map and specialize when a passenger is bound.

General sandwiches remain unrestricted. Explicit output projections do not
receive the unrestricted sandwich certificate; PGA and indefinite-metric
actions do not receive coefficient-orthogonality merely from unit versor facts.
Equality evidence across separate outer-slot bindings and passenger-relation
propagation through transpose are not implemented by this slice.

Explicit norm measurements and normalization always compute, even for known
unit inputs: the caller controls numerical drift. Propagated traits instead
let consumers trust their declared preconditions without defensive input
normalization. Drift examples in the inventory distinguish these two cases;
see [the extension contract](gatype_design.md#93-explicit-normalization-and-trusted-preconditions).

Populate the already-present type layer by completing one end-to-end proof and
dispatch path first:

- canonical immutable TraitSet flyweights;
- implication closure and invalid-combination checks;
- the first conservative value facts, including `ReverseProductOne`;
- structural implications through SubSpace support;
- operation, bind, composition, and transform TypeRules needed by the slice;
- conditional self-product and versor-closure geometric-product inference;
- one specialization selected by whole-GAType multiple dispatch;
- duplicate/declaration-order errors and incomparable-overlap warnings; and
- optional audit checks outside compiled execution.

After that slice is sound, expand the vocabulary and rules incrementally:

- add the value and map facts required by the accepted examples; `Isometry`
  remains a candidate representation for the map guarantees;
- design the separate repeated-argument sandwich implication only when its
  concrete output facts are needed;
- add collection rules and property-specific cast/transform laws;
- add `assume` and `forget`; and
- port the remaining specialized implementations to whole-GAType dispatch.

Each specialized extension implementation is ported once, directly to
whole-GAType dispatch. Before this phase, an extension that depends on certified
traits remains unavailable in the rewrite; it is not reached through a
compatibility bridge and is not copied into a temporary SubspaceDispatch
registry.

Any supported automatic-differentiation backend needs an explicit trait-erasure
or custom-rule boundary. In JAX, tangent values cannot inherit
coefficient-sensitive primal traits merely because they reuse the same
container metadata.

This phase must remain conservative. Unknown is not false, and no rule inspects
ordinary floating-point zeros to manufacture static facts.

### Phase 4: implement ordered, oriented SubSpaces

Land the complete signed-layout cluster:

- retain canonical integer bit blades;
- add immutable per-coordinate orientation signs;
- intern on ordered `(mask, sign)` entries;
- extend exact representation identity to ordered signs while preserving the
  stable equal-support and support-containment API introduced in Phase 1;
- split default support construction from explicit exact-layout construction;
- parse explicit blade words into masks plus permutation parity;
- extend AxisTransform with same-support signed `relayout`;
- complete explicit `cast(target_subspace)` for any compatible target support,
  using gather, sign correction, dropping, and zero fill as needed;
- fold all input/output signs into symbolic kernels;
- make selection and coordinate lookup layout-aware;
- make caches include exact layouts where required;
- accept one complete default layout as a string;
- make the default constructor overridable by a configured algebra; and
- export `PGA3D` with the agreed Hodge-friendly default.

Because Extensor binding already consumes AxisTransforms, it needs no new
control path. Because trait inference already routes casts through
`TypeRules.transform`, relayout uses its invariant-preserving path without
changing operation-specific trait rules.

Signed layouts must not flow through code that still assumes lexical
coefficient positions. Before enabling signed layouts at all:

- every primitive symbolic factory used by the new runtime emits the final
  canonical-support form and passes through layout realization;
- the rewrite contains no route into a root-package factory; and
- every supported coefficient-position-sensitive extension is ported, or
  routed to a layout-independent fallback, or explicitly rejects nondefault
  layouts.

Before enabling the new defaults, also settle:

- the result layout of `union` and other SubSpace combinations;
- the output layout of operations applied to explicitly laid-out inputs;
- whether narrower named relayout/project/embed helpers supplement the general
  explicit `cast`; and
- the exact grammar of the default-layout string.

The PGA3D activation gate includes:

```text
grade 0:  1
grade 1:  x, y, z, w
grade 2:  yz, zx, xy, xw, yw, zw
grade 3:  yzw, zxw, xyw, zyx
grade 4:  xyzw
```

The vector/trivector and nondegenerate/degenerate-bivector Hodge maps must have
the intended identity coefficient maps. Rotor and motor order then follow by
ordinary grade-first filtering; they receive no special ordering code.

### Phase 5: harden the standalone rewrite and publish

- complete execution-plan parity for every backend and strategy selected in the
  Phase 0 support matrix;
- verify that every supported extension uses GAType dispatch and that
  SubspaceDispatch does not exist in the rewrite;
- verify that binding and casting never bypass AxisTransform; symbolic kernel
  construction uses the centralized SubSpace/frame lookup API;
- optimize exact-rational storage and execution performance where release
  measurements justify it;
- run deterministic physics smoke tests on supported backends;
- build and test the rewrite wheel in an environment where the root repository
  and legacy package are unavailable;
- define the public import facade and optional dependency matrix;
- perform one atomic repository cutover: rehome the retained 1.x project intact
  under `reference/v1/`, promote the rewrite project metadata and source root,
  and point root CI, documentation, IDE configuration, and `pip install .` at
  2.0;
- convert design examples into executable documentation; and
- publish 2.0 API and intentionally-breaking-change notes.

The cutover is a single project-layout operation after the standalone wheel has
passed its gates, not a piecemeal merge into or teardown of 1.x. The reference
source remains intact under `reference/v1/`.

## 6. Representation identity, semantic refinement, and conversion

Ordered SubSpaces require a distinction that the present GAType document does
not yet make.

### Exact representation identity

The flyweight and cache identity is:

```text
GAType.representation_key
    = (exact ordered/oriented SubSpaces, canonical traits)
```

Changing an axis order or sign produces another exact GAType because the same
array coefficients mean something different.

### Semantic facts

Structural theorems and most trait reasoning use:

```text
GAType.semantic_key
    = (axis support keys, canonical traits)
```

Two alternate layouts may therefore be distinct represented types but carry
the same known mathematical facts. Here canonical trait parameters are
layout-neutral, or have first been rebased to their semantic support objects;
raw coordinate positions never enter this key.

### Conversion compatibility

Whether one concrete value can bind into a slot is answered by AxisTransform,
not by equality and not by semantic refinement alone. Resolution returns an
implementation together with required conversions:

```text
resolve(operation, actual_gatypes)
    -> (implementation, axis_transforms)
```

Every registration declares how each layout is matched:

```text
layout_match = semantic
    match support and facts; implementation is parameterized by actual layouts

layout_match = exact
    require the registered ordered/oriented SubSpaces exactly

layout_match = convert_to
    match support and facts, then convert to the registration's declared layouts
```

`semantic` is the normal mode for generated algebra operations. `exact` is for
hand-written kernels or external libraries whose array convention is part of
their contract. `convert_to` is for a fixed preferred convention when an
explicit lossless conversion is acceptable. It can request `relayout` or
zero-fill `embed`, but never an implicit lossy projection.

For example, PGA3D can keep its Hodge-friendly default while a Hamilton
interoperability overload either requires an exact quaternion layout or uses
`convert_to` to reach one.

Resolution is true most-specific resolution, not a blind first-match scan:

1. collect every matching registration;
2. discard any candidate dominated by a semantically more-specific candidate;
3. among semantically equivalent candidates, prefer an exact/native match,
   then a layout-parametric `semantic` implementation, then a `convert_to`
   match that requires conversion; and
4. use declaration order only for unresolved equivalent or incomparable
   candidates.

Semantic specificity means narrower accepted support and stronger required
traits, using the product order for multiple dispatched operands.

Declaration order remains meaningful as readable fallback precedence, so the
registry enforces specific-to-general declaration order even though the
resolver itself is specificity-aware. An equivalent declaration is a duplicate
error; a strictly more-general declaration placed before a later specialization
is an ordering error; and an unacknowledged incomparable overlap is a warning.
Coverage checks include layout mode: for the same facts, an unconstrained
semantic pattern covers an exact-layout pattern, while an exact-layout pattern
covers only that representation.

If `<` and `>` remain overloaded, they should mean strict semantic refinement.
Alternate layouts with identical support and traits are then neither `<` nor
`>` one another. The current formula

```text
A < B iff A <= B and A != B
```

cannot be retained with exact representation equality and layout-neutral
semantic comparison. The named methods for exact equality, semantic
refinement, and conversion should be settled before assigning Python operators.

### Trait transport across axis transforms

`TypeRules.transform` owns trait transport; AxisTransform only describes the
coordinate operation. For a positive-arity GAType it receives a mapping from
structural axis positions to transforms, so a cast can re-express one or several
axes without ambiguity.

- A lossless same-support relayout preserves basis-invariant mathematical
  traits. Any trait parameters that name represented axes are rewritten to the
  corresponding destination axes.
- Embedding, projection, and reframing use property-specific laws. A trait is
  retained only when its definition proves that transport; otherwise it becomes
  unknown.
- Layout-local execution facts such as contiguous coordinates, identity signs,
  or availability of a particular external kernel are not GAType traits. They
  belong to AxisTransform or the selected backend execution plan.

This keeps mathematical knowledge on the Extensor while preventing a
coordinate-layout optimization from masquerading as a theorem.

Trait parameters such as the forms named by `Isometry` must therefore either be
layout-neutral semantic objects or define how they are rebased by these
transforms.

## 7. Reference boundary and selective ports

There are no runtime or test-time migration bridges. The legacy tree may be
read while deciding what behavior survives, but the rewrite suite imports and
executes only rewrite code. Each retained behavior becomes an ordinary,
self-contained test with an explicit mathematical result.

Code crosses the source boundary only as a reviewed port:

| Legacy material | Rewrite treatment |
|---|---|
| pure bit operations and algebra descriptions | candidate for direct copy after dependency and property-test review |
| current SubSpace machinery | semantic reference; implement behind the final support/AxisTransform API |
| symbolic operator factories | port the formula into `OperatorFactory`, immediately producing canonical-support, output-first Extensors in `ExactContext` |
| backend contraction primitives | port individually behind BindingPlan's backend protocol |
| optimized extensions | port only when useful, directly to whole-GAType dispatch |
| examples and physics code | acceptance scenarios, not library dependencies |
| selected tests and examples | restate useful cases as native rewrite assertions; do not port harness machinery |

Every port is a copy into a new file or a clean reimplementation. It never
moves, renames, edits, or deletes its legacy source.

Each port records its old commit/path, new destination, whether it was copied,
adapted, or reimplemented, why it is valid under the new invariants, and the
test that covers it. A small `rewrite/PORTING.md` ledger is sufficient.

No port begins as a compatibility wrapper. If an old feature has not earned a
direct implementation on the final surface, it remains unsupported.

## 8. Code that should not be written

To keep the incremental approach from becoming serial rewrites:

- do not build Extensor with bare stored `axes` and add GAType later;
- do not introduce a separate exact `Operator` value class beside `Extensor`;
- do not implement binding independently in NumPy, JAX, and Torch;
- do not import root-package implementation modules from rewrite production
  code;
- do not create old-name aliases, delegators, wrappers, or shared legacy base
  classes;
- do not copy an entire module when only one context-free algorithm survives
  the new invariants;
- do not let new Extensor code inspect `.blades`;
- do not encode subset widening directly in einsum construction;
- do not port every optimized extension first to SubSpace dispatch and later to
  GAType dispatch;
- do not modify or dismantle the legacy concrete hierarchies as part of rewrite
  work;
- do not let coefficient-level fixtures depend on an implicit default layout;
- do not expose sign-bearing SubSpaces before kernels and casts honor signs; and
- do not combine the output-axis flip, new bind semantics, trait deduction, and
  PGA sign changes in one undifferentiated change.

The target is zero substantive disposable runtime or test infrastructure. The
porting ledger is documentation, not an executable compatibility layer.

## 9. Validation and completion gates

### Isolation gate

- the rewrite installs and imports from `rewrite/src/numga` in a clean
  environment with the root package unavailable;
- its wheel contains no root-package files or dependencies;
- the test configuration asserts that `numga` resolves inside `rewrite/src`;
- rewrite tests contain no imports, subprocesses, or adapters for the legacy
  package; and
- rewrite work does not modify files under the root `numga/`.

### Selected-semantics gate

- every deliberately retained behavior has a direct native test or an explicit
  reason why it is intentionally absent;
- expected values state their coordinate layout whenever raw coefficient order
  matters; and
- absence of an unselected 1.x feature is not a failure—only the declared 2.0
  feature set is tested.

### Symbolic-kernel gate

- symbolic coefficients are exact rationals from the first native Extensor in
  `ExactContext`;
- composition and squeeze use exact zero/cancellation tests, never epsilon
  heuristics;
- the storage implementation is hidden behind `SymbolicKernel`; and
- backend materialization is the sole rational-to-backend numeric conversion.

### Extensor gate

- one runtime class represents arity zero, one, and higher in both exact and
  backend contexts;
- typed-expression lifting gives `V * V`, `x * V`, and `x * y` arities two,
  one, and zero without constructing phantom Extensor values;
- output-first invariants use unequal axis lengths;
- all bind laws and residual-axis ordering pass;
- batch behavior is complete on NumPy and passes the selected second-backend
  smoke path;
- inputs remain unchanged after every operation; and
- representative physics operations satisfy direct 2.0 acceptance tests.

### Trait gate

- TraitSet interning and entailment laws pass;
- every transfer rule is sound for all values admitted by its inputs;
- geometric-product self-product relations survive partial and nested binding,
  and collapse to justified nullary facts after full binding;
- atomic and sequential geometric-product binding produce the same canonical
  relation and flyweight type;
- atomic repeated-versor sandwich binding establishes grade preservation and
  passenger relations, with coefficient-orthogonal inverse dispatch where valid;
- exact-layout, misordered, overlapping, and disjoint dispatch patterns behave
  as specified;
- kernels are shared across trait-only refinements; and
- where automatic differentiation is supported, gradients do not inherit
  unsound primal traits.

### SubSpace gate

- explicit spelling preserves order and orientation;
- same-support casts round-trip as exact signed permutations;
- support-changing casts have specified drop/zero-fill behavior for every
  source/target relationship;
- equivalent calculations agree after decoding into canonical blade values;
- all primitive kernels account for every input and output sign;
- exact layout is present in every materialized-kernel and plan key;
- generic defaults remain deterministic;
- every PGA3D grade order matches its specification; and
- Hodge, rotor, motor, and external-layout interoperability tests pass.

### Publication gate

- the standalone rewrite wheel passes without the root project on disk;
- after the atomic cutover, root build, test, documentation, IDE, and install
  entrypoints all resolve to 2.0;
- the rehomed 1.x reference artifact remains reproducible and runnable; and
- no production dependency points from 2.0 into `reference/v1/`.

There is no legacy deletion gate. The old implementation remains intact;
completion is measured against the explicit 2.0 feature and backend matrix, not
against how much legacy source has been removed or wrapped.

Before release, every backend and strategy in the declared support matrix must
pass the same Extensor conformance suite; this broader parity gate does not
block beginning the trait phase.

## 10. Current document reconciliation

The focused documents intentionally predate some cross-cutting decisions. They
should eventually be reconciled as follows:

- `extensor_design.md` keeps the final Extensor and binding semantics, but this
  document supersedes its in-place migration, compatibility, and legacy-removal
  guidance;
- `gatype_design.md` must describe SubSpace as an exact coordinate axis with a
  derived support, not merely an unordered blade set;
- `gatype_design.md` must separate exact type identity, semantic refinement,
  and conversion compatibility before retaining its partial-order claims;
- `gatype_design.md` must replace blind first-match dispatch with the
  most-specific resolution and declaration-order diagnostics specified here;
- `subspace_design.md` should incorporate the agreed single-string default
  layout constructor and the explicit any-target cast surface; its migration
  notes are subordinate to the parallel-tree boundary here;
- `numga2_design.md` should be treated as historical where it uses an
  output-axis-only type, unsound early tag implications, or defers accepted
  SubSpace work; and
- `todo.md` may eventually link to the settled documents instead of duplicating
  partial proposals.

These edits are documentation cleanup, not prerequisites for the first
Extensor vertical slice. Until reconciliation, this document defines how the
three focused designs fit together.

## 11. Integration decisions still required

The following decisions have limited blast radius when made at their indicated
phase:

Before the Extensor vertical slice:

- the single public coefficient attribute name;
- the correctness-first exact `SymbolicKernel` storage representation;
- concrete context compatibility and immutable context identity;
- the exact constructor ownership boundary between Context and Extensor;
- the minimum backend protocol used by BindingPlan;
- whether public indexing and reductions address batch axes only or expose raw
  kernel axes; and
- the 2.0 backend support matrix.

Before expanding the trait epoch:

- the next trait vocabulary beyond the implemented marker and self-product
  facts;
- the representation of additional relations; the two product relations are
  settled narrowly, while generic map and repeated-argument sandwich rules
  remain open;
- the named versus operator syntax for refinement;
- the exact semantic meaning of GAType comparison operators; and
- when an autodiff backend is selected, its tangent/cotangent trait boundary.

Before the signed-SubSpace epoch:

- generic lexical-order definition;
- union/intersection/difference result-layout policy;
- operation output-layout policy;
- explicit spelling and default-layout-string grammar; and
- names and trait behavior for relayout, embedding, projection, and reframing.

None of these open details requires a big-bang implementation. Each has a clear
owner, phase, and test boundary.

## 12. Final recommendation

Build the unified Extensor first, but build it on its permanent metadata and
binding seams:

```text
Extensor first:
    full runtime behavior

GAType now:
    only the final shell and inference hook

SubSpace now:
    only opaque exact axes, support queries, and conversion planning

GAType later:
    traits, proofs, relational rules, and dispatch

SubSpace later:
    ordering, orientation, casts, defaults, and PGA3D
```

This uses deliberate one-way ports, but no runtime or test-time compatibility
scaffolding and no piecemeal legacy teardown. The highest-priority feature
becomes usable early, while the later type and layout systems can deepen it
without changing its representation or execution model.
