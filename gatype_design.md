# GAType and Extensor Typing

Status: working design for the non-backwards-compatible rewrite.

Layout implementation clarification: exact equality, flyweights, and cache keys
include each axis's ordered masks and signs. `<=`/`refines` and generic dispatch
compare mathematical support and traits, independently of layout. This is a
preorder: distinct layouts can mutually refine each other. Coefficient-specific
overloads use exact-axis predicates; binding performs signed coordinate
conversion rather than treating semantic inclusion as array compatibility.
See [subspace_design.md](subspace_design.md) for the implemented conventions.

This note records the type model developed while reviewing
[`extensors.md`](rewrite/docs/extensors.md) and [`numga2_design.md`](numga2_design.md). It is a
new planning document: the older documents have deliberately not been rewritten
or treated as reconciled yet.

## 1. Core model

Every extensor has one type describing the entire extensor, at every arity:

```text
GAType (interned flyweight):
    subspaces: tuple[SubSpace, ...]  # output first
    traits:    canonical TraitSet   # facts about the whole extensor

Extensor (immutable semantics):
    context: Context
    gatype:  GAType
    kernel:  context-owned coefficient object

    axes  = gatype.subspaces
    arity = len(gatype.subspaces) - 1

ExactContext
    owns exact rational symbolic kernels for Extensor

backend Context
    owns materialized backend arrays for the same Extensor class

OperatorFactory
    constructs operation Extensors; it is not a value class
```

There is one `Extensor` value class across exact and backend stages. Moving an
exact value into a backend context changes its coefficient representation, but
not its Python class or `GAType`.

`gatype` is not an alias for the output axis. The output subspace is
`gatype.subspaces[0]`; the complete `GAType` includes every input axis and every
certified property of the extensor.

The array layout is output-first:

```text
kernel.shape == batch_shape + tuple(len(s) for s in gatype.subspaces)
```

A nullary extensor has one subspace and is an ordinary multivector. A unary
extensor has two subspaces and is a matrix. Higher-arity extensors follow the
same representation.

## 2. Interpretation

Conceptually,

```text
GAType((S0, S1, ..., Sn), traits)
```

describes a refined set of tensors in the carrier

```text
S0 ⊗ S1* ⊗ ... ⊗ Sn*.
```

The subspaces define structural support. The traits restrict which tensors in
that carrier are described by the type.

Examples:

```text
# Nullary: a value
GAType((Even,), {Versor, ReverseProductOne})

# Unary: a map
GAType((Point, Point), {Isometry(point_form), Outermorphism})

# Binary: a product
GAType(
    (Product, Left, Right),
    {
        ProductRelation(ReverseProduct, ProductResult.ONE, slots=(0, 1)),
        ProductRelation(CliffordConjugateProduct, ProductResult.ONE, slots=(0, 1)),
        VersorProduct(slots=(0, 1)),
    },
)
```

The binary fact is conditional relation metadata, not an unconditional
`ReverseProductScalar` claim about every output.

Traits are predicates on the whole extensor. Their valid vocabulary and meaning
depend on arity.

## 3. SubSpace remains structural

`SubSpace` needs no annotation system beyond its existing structural
properties. It remains the algebra-specific set of supported blades and the
operations derived from that set.

Structural properties may imply traits for inhabitants without storing those
traits on the `SubSpace`:

```text
effective_traits(gatype)
    = explicit_traits(gatype)
    ∪ structurally_implied_traits(gatype.subspaces)
    ∪ logical_trait_closure
```

For example, a future `Blade` trait could be implied by grade-1 support for a
nullary extensor. Scalar support already implies that the reverse product is
scalar-valued; it does not imply that a particular scalar is nonzero,
invertible, or unit.

A rotor is consequently a refined nullary type, not a decorated subspace:

```text
Rotor = GAType((Even,), {Versor, ReverseProductOne})
```

If two application-level concepts have identical structural support but
different nominal meanings, that nominal distinction belongs in a separate
application-level facility, not in `SubSpace` identity.

## 4. Trait vocabulary and entailment

Traits are immutable, hashable, structured predicates rather than an
open-ended set of strings. The implemented first vocabulary is:

```text
Nullary/value traits:
    Versor
    ReverseProductScalar
    ReverseProductZero
    ReverseProductNonzero
    ReverseProductOne
    CliffordConjugateProductScalar
    CliffordConjugateProductZero
    CliffordConjugateProductNonzero
    CliffordConjugateProductOne

Unary/map traits:
    CoefficientOrthogonal

Higher-arity traits:
    ProductRelation(self_product, factor, slots)
    VersorProduct(slots)
```

`Blade`, `Isometry`, `Outermorphism`, `InvertibleMap`, slot symmetry, and generic
`Maps`/`Requires` are possible vocabulary, not requirements for a universal
inference framework. An internal `Sandwich` certificate identifies the
unrestricted trilinear sandwich. Binding the same known versor to both outer
slots establishes grade preservation and specializes into the existing
`ProductRelation` and `VersorProduct` vocabulary. Unit Euclidean actions on
matching input/output support additionally establish `CoefficientOrthogonal`.
This uses static types and same-object binding evidence, not coefficient tests.

The reverse-product facts form an implication lattice rather than one vague
numeric tag:

```text
ReverseProductOne
    => ReverseProductNonzero
    => ReverseProductScalar

ReverseProductZero
    => ReverseProductScalar
```

Under the proposed definition of a versor as a product of invertible vectors,
`Versor` also entails `ReverseProductNonzero`. In particular, `Blade * Blade =>
Versor` is not generally sound in a degenerate algebra unless sufficient
nondegeneracy facts are known.

Absence of a trait means unknown, never false. Negative facts are deferred until
a concrete need justifies their additional logic.

### 4.1 The first relational traits

The first implemented relations are deliberately narrower than a generic
`Maps` language. Define the one-sided reverse product

```text
N(x) = x * reverse(x)
```

Every geometric-product extensor carries

```text
ProductRelation(ReverseProduct, factor=ONE, slots=(0, 1))
ProductRelation(CliffordConjugateProduct, factor=ONE, slots=(0, 1))
VersorProduct(slots=(0, 1))
```

For the reverse-product family this states a conditional law: when every referenced input has a certified
scalar reverse product, the output category is `factor` times the referenced
input categories. The finite abstract categories are `ONE`, `ZERO`, `NONZERO`,
and `SCALAR`; they describe known facts and never store a runtime scalar.
For equal slot tuples their factors use the same entailment lattice as nullary
facts, so a `ONE` relation refines and dispatches through `NONZERO` and
`SCALAR`, while a `ZERO` relation refines `SCALAR`.

Binding consumes a certified nullary fact or substitutes the corresponding
relation of a positive-arity operand. Slots are then remapped into the result.
For example:

```text
product relation                         ONE * N(0) * N(1)
bind NONZERO value into slot 0           NONZERO * N(0)
bind ONE value into the remaining slot   ReverseProductNonzero
compose (a * b) * c                      ONE * N(0) * N(1) * N(2)
```

Repeated slots are powers and are not deduplicated. A `ZERO` factor does not
short-circuit the scalar precondition on any remaining slot: `zero * unknown`
is still unknown because product multiplicativity has not been established.

This rule follows from

```text
N(a * b) = a * N(b) * reverse(a)
```

and therefore becomes `N(a) * N(b)` only under the recorded scalar
precondition. It is attached by the geometric-product factory, not inferred by
examining a symbolic or numerical kernel.

`VersorProduct` records the independent closure law that a product of versors
is a versor. It uses the same slot substitution during composition.
Consequently, an expression built from unit-rotor values recovers the canonical
Rotor GAType and immediately enables the default unit inverse specialization:

```python
product = (Rotor * Rotor * Rotor)(a, b, c)
assert product.gatype is algebra.gatype.rotor()
product.inverse()  # dispatches to reverse()
```

The same one-sided definition makes reverse propagation asymmetric:

```text
N(reverse(x)) = reverse(x) * x
```

`ReverseProductOne` and `ReverseProductNonzero` transport through reverse, as
does `Versor`; merely `ReverseProductScalar` or `ReverseProductZero` does not.
Those latter facts would require a stronger two-sided definition. The initial
implementation therefore has no relational reverse or sandwich rule.

Operation rules must justify the facts they propagate. A trait need not supply
a universal transfer rule for every operation. The relevant contracts are:

- the arities and axes for which it is meaningful;
- the other traits it entails;
- how it specializes under binding;
- how it behaves under composition;
- whether it can be transported when axes are narrowed or widened.

### 4.2 Shared self-product facts, explicit mathematical laws

The implementation now separates a small generic fact container from the
mathematical rules of each family:

```python
ReverseProduct = SelfProduct("reverse")
CliffordConjugateProduct = SelfProduct("clifford_conjugate")

ReverseProductOne = ProductFact(ReverseProduct, ProductResult.ONE)
CliffordConjugateProductOne = ProductFact(
    CliffordConjugateProduct, ProductResult.ONE,
)
```

`SelfProduct(transform, product="geometric_product")` identifies the full
expression `product(x, transform(x))`. It is immutable metadata, not an
executable callback or a request to infer a theorem. `ProductResult` classifies
the full result as scalar, zero, nonzero scalar, or scalar one. It says nothing
about a scalar projection, magnitude, reality, or positivity. For another
product, `ONE` must not silently mean a different identity element.

`ProductFact` and `ProductRelation` share family-local implication and
compatibility rules. Different families can coexist, even with different
result categories. The first pass allows one relation slot signature per
family. Binding substitutes only matching families and keeps their scalar
guards separate. `VersorProduct` remains an independent closure fact.

Geometric-product multiplicativity is explicitly installed for reversion and
Clifford conjugation, which satisfy `T(a*b) = T(b)*T(a)`. It is not granted to
grade involution or arbitrary grade-sign transformations just because
`T(T(x)) == x`. A full `x*T(x) == 1` fact can nevertheless justify an inverse
implementation without granting that propagation law.

`TraitSet` no longer branches on the reverse-product family. Traits supply
`implied_traits()`, `validate_peers(...)`, and a canonical `sort_key`; the
container computes closure, checks compatibility, and removes redundancy.
Mutually implying aliases retain a deterministic representative. Custom
structured traits store immutable parameters; subclasses with additional
state provide their own pickling. Unknown facts remain usable for dispatch
but are not automatically preserved by numerical operations.

The concrete example is
[`test_self_product_end_to_end.py`](rewrite/tests/test_self_product_end_to_end.py):
Clifford-conjugate-unit paravectors compose through partial binding and select
the default conjugation-based inverse, without asserting a versor or unit
reverse-product guarantee. The two inverse specializations explicitly use
declaration order when both promises hold.

Complex floating coefficients already work in the dense contexts, but this
does not define a general complex-algebra/adjoint API. Reversion and Clifford
conjugation are coefficient-linear; neither conjugates coefficients. Tests
cover this distinction and a complex rotor whose reverse product is one but
whose coefficient magnitude is not one. Conjugate-linear execution, Hermitian
adjoints, dualized self-products, and additional product identities are
deferred. No implicit normalization or coefficient-dependent dispatch is added.

A trait is an extensional claim, not provenance. An isometric matrix behaves as
an isometry regardless of whether it arose from a sandwich, composition, a
checked constructor, or an explicit assumption.

## 5. Flyweight semantics

`GAType` is interned per `Algebra`, alongside `SubSpace`. Construction proceeds
through a `GATypeFactory` that:

1. verifies that every axis belongs to the same algebra;
2. validates each trait against the arity and axes;
3. canonicalizes trait parameters and ordering;
4. normalizes logically equivalent trait sets;
5. interns on `(subspaces, normalized_traits)`.

The pool belongs to that per-algebra factory, not to the `GAType` class.
`GAType` itself is an ordinary immutable structural value; its constructor has
no global registry or custom allocation behavior. The canonical production
path is `algebra.gatype(subspaces, traits)`, whose strong pool retains canonical
instances for the lifetime of that algebra.

For example, these must resolve to the same flyweight:

```text
{ReverseProductOne}
{ReverseProductOne, ReverseProductNonzero, ReverseProductScalar}
```

How a fact was established is deliberately absent from the interning key. A
checked unit versor and an assumed unit versor have the same `GAType`. Optional
audit evidence is diagnostic information, not type identity.

Structural equality and hashing remain the correctness contract. Flyweight
identity is a valid fast path for objects returned by the same algebra factory,
but accidentally bypassing the factory must not change semantic behavior.

A bare subspace implicitly lifts to the unrefined nullary type whenever it
appears in a typed expression position:

```text
lift(S) = GAType((S,), EMPTY_TRAITS)
```

The lift is a type normalization, not a value conversion: it creates neither an
Extensor nor coefficients. An operation factory uses it to describe an unbound
nullary argument, giving the uniform expression idioms:

```text
V * V  -> exact Extensor, arity 2
V * V * V -> exact Extensor, arity 3
x * V  -> Extensor, arity 1
x * y  -> Extensor, arity 0
```

A complete `GAType` remains usable directly when the typed position needs
traits beyond `EMPTY_TRAITS`.

### 5.1 Construction namespace

The algebra-owned GAType factory mirrors the public constructors declared by
its SubSpace factory. A mirrored constructor lifts the resulting structural
axis and adds no traits:

```python
algebra.gatype.even()
    == algebra.gatype(algebra.subspace.even(), EMPTY_TRAITS)

algebra.gatype.k_vector(2)
    == algebra.gatype(algebra.subspace.k_vector(2), EMPTY_TRAITS)
```

This forwarding is lowercase and case-sensitive, following the existing
factory API. It applies to nullary types only; arbitrary-arity construction
continues to use `algebra.gatype((output, *inputs), traits)`.

Semantic GAType constructors are explicit methods in the same namespace and
take precedence over structural forwarding. In particular:

```python
algebra.gatype.even()
    == GAType((Even,), EMPTY_TRAITS)

algebra.gatype.rotor()
    == GAType((Even,), {Versor, ReverseProductOne})
```

`rotor()` certifies a refined type; it is not another spelling for an Even
SubSpace. Supplying arbitrary coefficients together with a refined GAType is a
trusted low-level assertion. Checked value constructors can be added during the
trait epoch without changing this namespace.

The rewrite now implements implication closure, refinement comparison,
whole-GAType dispatch, and the narrow geometric-product propagation slice.
Checked constructors and the broader relational vocabulary remain later work.

## 6. Refinement and subset relations

`GAType` denotes a set of possible extensors, so its comparison operators follow
set direction. For two types of the same arity:

```text
A <= B iff
    every A.subspaces[i] ⊆ B.subspaces[i]
    and A.traits entails every trait in B.traits

A < B iff A <= B and A != B
A > B iff B < A
```

Narrower structural support and stronger known facts both make a type smaller:

```text
GAType((Even,), {Versor, ReverseProductOne})
    < GAType((Multivector,), {Versor})
    < GAType((Multivector,), {})
```

The order is partial. For example, a generic even value and a generic unit
multivector are incomparable. Both `<` and `>` return false, as with sets.

Named operations should accompany the operators:

```text
A.refines(B)            # A <= B
A.strictly_refines(B)   # A < B
A.overlaps(B)
A.entails(trait)
```

A `SubSpace` appearing at this comparison boundary is shorthand for its plain
empty-trait nullary GAType:

```python
b <= algebra.subspace.bivector()
# exactly the same refinement question as
b <= algebra.gatype.bivector()
```

The same lift is used by `refines`, `strictly_refines`, `overlaps`, and the
reflected `<`, `<=`, `>`, and `>=` comparisons. It does not weaken algebra or
arity checks: a foreign-algebra SubSpace and a SubSpace compared with a
positive-arity GAType do not refine one another. Equality remains structural
object equality—`GAType == SubSpace` is not made true—and no coefficients or
Extensor values are constructed. This is only the existing bare-SubSpace type
lift applied consistently at the semantic comparison boundary.

The richer methods are required for diagnostics because `<` alone cannot
distinguish disjoint, overlapping, and presently unprovable relationships.

This relation orders tensor descriptions. It is not callable-function
subtyping; if higher-order callable substitution is introduced, the
contravariance of input domains must be modeled separately.

Relational traits require special care when axes differ. For example, an
isometry on `Point -> Point` does not become an isometry on `Multivector ->
Multivector` merely because `Point ⊆ Multivector`. The conservative default is
that such a trait transports only when its definition explicitly supplies the
required restriction or extension law.

## 7. Binding

Suppose:

```text
F.gatype.subspaces = (Output, Input1, Input2)
x.gatype.subspaces = (X,)
```

Binding `x` into the first input checks:

```text
X ⊆ Input1
```

Future operations may declare input requirements. The current relations do not
make such requirements part of call compatibility: they simply withhold a
conclusion unless the bound argument's `GAType` proves the required fact.

Binding then constructs a new extensor and a new canonical `GAType`:

1. contract the bound kernel axis;
2. remove the bound input subspace;
3. specialize relational traits using the argument's known traits;
4. promote satisfied conclusions to traits of the result or residual map;
5. retain only claims justified for the new extensor;
6. intern the resulting type.

No existing extensor changes. There is no trait-invalidation lifecycle:

```text
operation: Extensor[A] x Extensor[B] -> Extensor[infer(operation, A, B)]
```

Functional indexing, arithmetic, binding, composition, `assume`, and `forget`
all return new extensors. A constructor from arbitrary coefficients supplies
only the traits it can justify or that the caller explicitly assumes.

Repeated-argument information and tensor symmetry are distinct:

```text
SameArgumentSlots((1, 3))  # how a binding operation supplies arguments
SymmetricSlots((1, 3))     # a property of the extensor itself
```

## 8. Deferred sandwich typing target

The structural sandwich operation and atomic repeated binding are implemented,
but no `Isometry`, generic passenger-preservation, or diagonal trait is emitted
yet. The following sketch is a future target; its trait names are illustrative.

Let a motor value have:

```text
m.gatype = GAType((Even,), {Versor, ReverseProductOne})
```

Binding that same value into both motor positions of the sandwich operation can
prove that the residual unary extensor is:

```text
A = m.sandwich(Point)

A.gatype = GAType(
    (Point, Point),
    {
        Isometry(point_form, point_form),
        Outermorphism,
        # passenger-preservation relations are deferred
    },
)
```

Applying `A` specializes those relational traits:

```text
A(generic_point) -> generic point
A(null_point)    -> point with ReverseProductZero
A(unit_point)    -> point with ReverseProductOne
```

The result does not preserve the passenger `GAType` verbatim. Each fact is
preserved only when a certified map trait justifies it. This matters when future
traits are not invariant under an isometry or outermorphism.

Once `Isometry` is attached to `A`, it is intrinsic to that matrix; its sandwich
origin is irrelevant. In a degenerate metric, `Isometry` alone must not imply
`InvertibleMap`.

## 9. Whole-GAType dynamic dispatch

Overloaded implementations receive the complete `GAType` of every dispatched
operand. There are two ordinary declarative registration forms.

An algebra-independent `GATypePattern` records an Extensor arity and the
minimum facts required by an implementation, while deliberately leaving its
concrete axes unconstrained:

```python
GATypePattern.value(Versor, ReverseProductOne)
GATypePattern.map(CoefficientOrthogonal)
GATypePattern.nary(2, AlternatingSlots((1, 2)))
```

The wildcard axes belong only to the dispatch pattern. Every actual Extensor
still has a complete, concrete, layout-bearing `GAType`. A `Trait` or
`TraitSet` whose valid arities have exactly one common value is shorthand for
the corresponding `GATypePattern`, so common registrations remain terse:

```python
@Extensor.inverse.register(ReverseProductOne)
def inverse_unit(value):
    return value.reverse()
```

This one implementation applies to matching values from every algebra. It
would be incorrect to manufacture its pattern through one algebra's
`gatype.full()` factory: that would silently make a semantic implementation
local to that particular algebra object.

A concrete `GAType` registration remains available deliberately. It describes
the maximum structural carrier and minimum facts accepted by an
algebra-local implementation.

An actual type matches a concrete registered pattern when:

```text
actual.arity == pattern.arity
every actual.subspaces[i] ⊆ pattern.subspaces[i]
actual entails pattern.traits
```

Extra known facts never prevent a match. Trait implication, rather than literal
set inclusion, participates in matching: `ReverseProductOne` satisfies a
`ReverseProductNonzero` requirement.

An actual type matches an algebra-independent pattern when its arity agrees and
its effective traits entail the pattern traits. Generic structural conditions
belong in axis-parameterized facts derived from the concrete SubSpaces, rather
than in deferred recipes such as "call this algebra's `vector()` factory".
That keeps pluggable SubSpace construction and semantic matching separate.
Concrete GAType patterns continue to provide per-axis support-subset matching
when an implementation really is algebra-local.

Resolution collects every matching registration and discards each candidate
dominated by a semantically more-specific match. Declaration order is used only
to break a tie between incomparable maximal candidates. Declarations are still
required to read from most specific to most general, and the registry validates
each declaration against earlier declarations:

- an equivalent earlier pattern is a duplicate and is an error;
- an earlier, more-general pattern would shadow the new specialization and is
  an error;
- an earlier, more-specific pattern is correctly ordered;
- incomparable but overlapping patterns use declaration order as the
  deterministic tie-breaker and produce a warning by default;
- disjoint patterns require no diagnostic.

For example, this is invalid:

```text
1. GAType((Multivector,), {})
2. GAType((Even,), {Versor, ReverseProductOne})
```

The declarations are in the wrong conceptual order even though the resolver
could discover the later specialization. The reverse order is valid.

These patterns are incomparable but may overlap:

```text
GAType((Even,), {})
GAType((Multivector,), {ReverseProductOne})
```

An even unit value matches both. Declaration order chooses the implementation,
and the dispatcher warns unless precedence is explicitly acknowledged. An
intersection overload for `GAType((Even,), {ReverseProductOne})` removes the
ambiguity.

For multiple dispatch arguments, pattern refinement uses the product order: a
registration is at least as specific in every argument and strictly more
specific in at least one. This proof applies to generic `GATypePattern` and
concrete `GAType` registrations. Opaque callable predicates cannot support the
proof and therefore use the explicitly separate priority tier below.

After resolution, the implementation is cached by the exact tuple of interned
actual `GAType` objects.

### 9.1 Opaque low-level registrations

Some hand-written kernels depend on facts that are intentionally below the
portable semantic type layer: a precise signed coefficient layout, basis
naming, an algebra signature, or some combination of these. For this case only,
`register` accepts one callable condition:

```python
@Extensor.exp.register(
    lambda t: (
        t.algebra.description.to_compact_string() == "x+y+z+w0"
        and t.subspaces == (expected_bivector_layout,)
    )
)
def exp_pga3_bivector(value):
    ...
```

The predicate receives the complete `GAType` values, never Extensors,
coefficients, or runtime numeric data. It can therefore test exact
representation while deliberately ignoring additional certified traits. This
is preferable to a concrete GAType pattern for fixed-coordinate code: concrete
patterns use support-subset matching, so a narrower actual carrier may also
match them.

Opaque predicates cannot participate honestly in refinement or overlap proofs.
They form an explicit low-level tier which is presumed more specific than every
declarative registration. Matching predicates are tried in list order;
`position=` can insert one relative to the other predicates. Declarative
registrations still use semantic specificity and declaration order only for
incomparable maximal matches.

Backend identity is not a GAType fact. This predicate can select the algebra
and exact coordinate representation used by an optimized formula, but it
cannot safely select a NumPy-only implementation in the unified Extensor
model. Backend-only kernels need a separate execution-strategy/backend
dispatch boundary; the type predicate must not pretend to certify a backend.

### 9.2 Extensor extension methods

Extension methods are a consumer of whole-`GAType` dispatch, not a second
dispatch mechanism. They use an explicitly installed descriptor on the unified
`Extensor` type, retaining the direct declaration and call syntax:

```python
@Extensor.inverse.register(ReverseProductOne)
def inverse_unit_versor(value):
    return value.reverse()

@Extensor.inverse.register(CoefficientOrthogonal)
def inverse_orthogonal_map(value):
    return value.transpose()
```

The default descriptor and these implementations are installed by the
rewrite. It binds itself on instance access, so both a nullary value and a
unary map can use `value.inverse()`. It owns one cross-algebra registry.
The first shared self-product pass also installs a
`CliffordConjugateProductOne` inverse using `value.clifford_conjugate()`.
Algebra-independent patterns apply everywhere, while a concrete GAType entry
matches only its owning algebra. Installing the descriptor mutates the Extensor
type deliberately.

The number of dispatched Extensor operands is inferred from the first
registration and fixed for that method across all algebras. It is independent
of each operand type's own extensor arity. Multiple dispatch therefore has the
same compact form:

```python
Extensor.apply_pair = ExtensionMethod("apply_pair")

@Extensor.apply_pair.register(left_type, right_type)
def apply_pair(left, right):
    ...
```

Declarative patterns retain specificity, overlap, and declaration-order
diagnostics. The explicitly opaque callable form follows the separate priority
rules above.

The transpose specialization requires a narrow `CoefficientOrthogonal` trait,
whose contract is that the represented coefficient matrix has transpose as its
inverse. A general geometric `Isometry` does not imply this in arbitrary
signature or coordinates; its inverse is the appropriate metric adjoint, and
degenerate forms may not establish invertibility at all.

An extension intended for arbitrary future Algebra instances uses a generic
arity/trait pattern once. Concrete factories are involved only for an
intentional algebra-local specialization.

### 9.3 Explicit normalization and trusted preconditions

Traits do not override an explicit request to measure or normalize coefficients.
`value.norm()`, `value.norm_squared()`, and `value.normalized()` perform their
numerical work even when the input carries `ReverseProductOne`. In particular,
`.normalized()` must not become an identity operation on a known-unit input:
the caller may be deliberately correcting floating-point drift. It returns a
new value with the established unit facts; the input remains unchanged.

A consumer requiring unit input declares that precondition and trusts it.
For example, a unit-rotor logarithm must not silently normalize its input,
and the unit inverse uses reversal without defensive norm computation.
Propagating unit facts through products and sandwiches lets results enter
these consumers directly; it does not erase explicit normalization calls.
Numerical drift and the placement of corrective normalization remain under
the caller's control.

If the required facts are absent, dispatch may select a genuine general-input
implementation or fail explicitly. A normalize-then-call variant can be
explicitly requested, but is not an automatic repair fallback. This contract
does not forbid numerical work intrinsic to the selected algorithm, such as
measuring a bivector magnitude inside a logarithm; it forbids defensive repair
of input whose required properties are already promised.

## 10. Cache boundaries

Type refinement must not fragment structural kernels unnecessarily:

```text
GAType flyweight:
    canonical (subspaces, traits)

symbolic kernel cache:
    operation + subspaces

overload-resolution cache:
    operation + complete actual GATypes

backend execution-plan cache:
    selected operation/kernel + backend + dtype + device + strategy
```

The same symbolic kernel may therefore support many refined `GAType`s. No
value-specific extensor is stored in a global type or kernel cache.

## 11. Sources of certified traits

A trait may enter a `GAType` through only a small number of auditable paths:

1. a universal theorem derived from structural support;
2. a trusted constructor or named operation;
3. a binding, composition, or implication rule over already-certified traits;
4. an explicit assumption by the caller;
5. optionally, a checked constructor or audit assertion.

Ordinary floating-point zeros are not inspected to narrow a type. This keeps
typing stable across eager execution, batching, and tracing.

For a batched extensor, a trait is a promise about every member of the batch
unless a future trait explicitly defines another quantification.

## 12. Initial behavioral checks

The first implementation should demonstrate at least these properties:

- a `SubSpace` in typed expression position normalizes to the interned nullary
  `GAType((S,), EMPTY_TRAITS)` and the three product idioms have arities two,
  one, and zero;
- logically equivalent trait descriptions intern to the same `GAType`;
- a unit rotor refines both a generic versor and a generic multivector;
- incomparable types remain incomparable;
- dispatch matches subsets of structural support and subsets of required facts;
- a specialized registration placed behind a general one is rejected;
- incomparable overlapping registrations warn and use declaration order;
- every geometric-product extensor carries independent conditional reverse-
  and Clifford-conjugate-product relations;
- partial binding specializes that relation, while full binding promotes it to
  the strongest justified nullary reverse-product fact;
- nested products substitute and remap their relations identically under
  atomic and sequential binding;
- products of unit versors recover the canonical Rotor GAType and thereby
  select the default reversal-based inverse implementation;
- an unknown reverse product blocks the conclusion even after a zero-norm
  factor has been bound;
- reverse retains unit and nonzero reverse-product facts but not merely scalar
  or zero one-sided facts;
- all operations construct new extensors and leave their inputs unchanged.

## 13. Open questions

The central model is coherent, but these details remain open:

- the next trait vocabulary and names;
- whether input requirements belong inside `TraitSet` or in an adjacent
  application contract while remaining part of type identity;
- the representation of broader relations needed for sandwich passenger facts;
- which relational traits safely transport across axis restriction;
- whether incomparable-overlap diagnostics are warnings by default and errors in
  a strict or CI mode;
- whether the opaque predicate tier eventually needs named priority bands in
  addition to its current list order and `position=` insertion;
- how deeply immutability is enforced for backends with mutable array objects.

These questions refine the model; they do not require returning to annotated
subspaces or output-axis-only GATypes.
