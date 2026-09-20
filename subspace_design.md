# Ordered SubSpaces and Blade Layouts

Status: first signed-layout implementation is in `rewrite/`.

## Current implementation

```python
from numga.algebras import PGA3D
from numga import NumpyContext

spaces = PGA3D.subspace
mv = NumpyContext(PGA3D).multivector
cyclic = spaces("yz zx xy")
lexical = spaces("xy xz yz")

rotation = mv(cyclic, [1, 2, 3])
rotation.select_subspace(lexical)  # coefficients [3, -2, 1]
rotation.dual()                   # xw, yw, zw; coefficients [1, 2, 3]
```

- `from_blades` (also the factory call syntax) preserves explicit order/signs.
  Token tuples work for multi-character generator names. `from_layout` accepts
  ordered masks and signs directly.
- `from_masks` selects support using the algebra's default constructor.
  Plain algebras retain the existing grade-major, increasing-mask default in
  this pass. `PGA3D` binds the oriented defaults below through the injectable
  `Algebra(..., subspace_factory=...)` seam. A factory accepts its default full
  layout as one string; there is no profile registry.
- Restriction/intersection preserve the source layout. Unions and newly
  inferred operation outputs use the algebra default for their support.
- `==`, hashing, flyweights, and operator caches include exact order/signs.
  GAType `<=` means mathematical support/trait inclusion, ignoring layout;
  it is a **preorder** on concrete layouts. Mutual inclusion does not mean
  equal types. Strict `<` excludes mutually equivalent mathematical types.
- Generic dispatch uses that semantic relation. A coefficient-specific
  overload uses an exact predicate such as `g.subspaces == (cyclic,)`.
  Dispatch does not reinterpret or automatically cast its arguments. Operators
  constructed for a new layout get their own cached signed kernels; binding
  to an existing operator performs the necessary coordinate conversion.
- Same-support conversion retains the extensor's mathematical traits;
  projection does not. Scalar nonlinear overloads separately handle storage
  against `-1`, leaving ordinary scalar execution unchanged.

These decisions supersede the open choices in the original discussion below.

`PGA2D` is also available from `numga.algebras`, with signature `x+y+w0`
and default spelling `1 x y w yw wx xy xyw`. Its line/vector coordinates
`(x, y, w)` dualize to point/bivector coordinates `(yw, wx, xy)` unchanged;
affine points therefore read as `(x, y, 1)`. Unlike PGA3 odd-grade duality,
both directions here are coefficient identities.

This note records the planned `SubSpace` representation and its relationship to
array layout. It complements [`gatype_design.md`](gatype_design.md) and
[`extensor_design.md`](extensor_design.md). Those documents have deliberately
not been rewritten or treated as reconciled yet.

## 1. Scope and central decision

The existing integer bit-blade machinery remains the low-level representation
of canonical basis blades. The rewrite does not require replacing it with a
string- or object-based blade algebra.

What changes is the meaning of `SubSpace`. A `SubSpace` describes an actual
coefficient axis, including:

- which canonical basis blades are present;
- their order along the array axis; and
- the orientation sign of every presented basis blade.

The defining invariant is:

> Two SubSpaces are equal only when they assign coefficients to algebra
> elements in exactly the same way.

Equivalently, for a SubSpace `S` with coefficient vector `c`,

```text
interpret_S(c) = sum_i c[i] * S.signs[i] * E[S.blades[i]],
```

where `E[mask]` is the canonical bit-blade associated with `mask`.

Anything that changes this interpretation, including blade order or blade
orientation, produces a different `SubSpace`. Dtype, device, batch shape, and
dense versus sparse storage do not change the `SubSpace`; they do not change
the meaning of its coordinates.

## 2. Representation

The conceptual representation is:

```text
SubSpace (interned flyweight):
    algebra: Algebra
    blades:  immutable uint[N]  # ordered canonical bit masks
    signs:   immutable int8[N]  # each +1 or -1
```

The invariants are:

```text
len(blades) == len(signs)
all(sign in {-1, +1} for sign in signs)
all canonical masks in blades are unique
```

The empty SubSpace is valid, and canonical mask `0` remains the scalar basis
blade. The algebraic zero produced by a repeated generator is not a basis blade
and cannot occur as a coordinate entry.

The flyweight key contains the algebra identity and the complete ordered
`(mask, sign)` sequence. Input arrays must be copied or frozen so later mutation
cannot invalidate hashing, equality, dispatch, or operator caches.

As with `GAType`, the strong flyweight pool belongs to the SubSpace factory
owned by an algebra. It is not a process-global cache embedded in the
`SubSpace` class. The value class retains structural equality and hashing, so
factory identity is an optimization and canonicalization guarantee rather than
the basis of correctness.

Human-readable spelling may be retained for display or interchange, but it
does not independently determine identity after normalization. Two spellings
that produce the same ordered masks and signs describe the same coefficient
interpretation and therefore intern to the same `SubSpace`.

## 3. Canonical masks and the sign overlay

The algebra declares an order for its generating vectors. A canonical bit-blade
is their exterior product in increasing generator-index order. Existing bit
operations continue to provide:

- blade identity;
- grade;
- complement;
- involutions; and
- the canonical geometric-product table.

An explicitly spelled blade is reduced to this canonical mask by sorting its
generators. The parity of that permutation becomes its orientation sign.

For generators ordered `x < y < z < w`:

```text
xy   -> (mask(xy), +1)
yx   -> (mask(xy), -1)
zx   -> (mask(xz), -1)
zyx  -> (mask(xyz), -1)
```

Thus the explicit layout

```text
(yz, zx, xy)
```

has the internal representation

```text
blades = (mask(yz), mask(xz), mask(xy))
signs  = (+1,       -1,       +1)
```

The signs are not presentation metadata. They are part of the basis and affect
the interpretation of every coefficient.

`xy` and `yx` may consequently be distinct one-dimensional SubSpaces, but they
cannot both occur as independent coordinates in one SubSpace: both have the
same canonical mask and are linearly dependent. A layout containing duplicate
canonical masks is rejected.

A spelled basis blade must contain distinct generators. A repeated generator
would describe the zero exterior blade and cannot form a coordinate basis.
Multi-character generator names should be accepted through unambiguous token
tuples or delimiters rather than relying solely on greedy character parsing.

Only the `+1/-1` orientation overlay is required for this rewrite. Arbitrarily
scaled or non-blade coordinate bases are outside the initial scope.

## 4. Exact layout versus structural support

Because `SubSpace` is coordinate-bearing, three different relationships must
remain distinct:

```text
S == T
    Exact equality of algebra, blade order, and signs.

S.same_support(T)
    The canonical blade-mask sets are equal.

S.support_is_subset_of(T)
    Every canonical mask in S also occurs in T.
```

For an oriented blade basis, equal support is equivalent to spanning the same
mathematical blade subspace. In particular:

```text
SubSpace((xy,)) != SubSpace((yx,))
SubSpace((xy,)).same_support(SubSpace((yx,)))
```

Set-like structural reasoning operates on canonical-mask support, not on the
ordered `(mask, sign)` entries. Exact array compatibility operates on the full
SubSpace.

Support containment should not be confused with exact type refinement. Two
different layouts of the same support contain one another structurally but are
not interchangeable as raw arrays. Treating support containment as the only
`SubSpace` ordering would produce a preorder rather than an antisymmetric
partial order. Layout conversion must remain a separate relation.

## 5. Default and explicit construction

There are two intentionally different construction paths.

### Default construction

Ordinary structural constructors select support and apply the one default
ordering bound to the algebra:

```python
ga.subspace.vector()
ga.subspace.bivector()
ga.subspace.even_grade()
```

An alternative generic default would use grade-major ordering and lexicographic ordering of
canonical generator-index tuples within each grade. For generators declared as
`x < y < z < w`, that proposed generic order is:

```text
1
x, y, z, w
xy, xz, xw, yz, yw, zw
xyz, xyw, xzw, yzw
xyzw
```

This definition does not sort arbitrary display strings. It is also distinct
from simply sorting integer masks: tuple-lexicographic and increasing-mask
orders differ in four or more dimensions. This alternative remains a suggestion;
the implementation preserves the existing grade/mask default, covered by tests.

The default SubSpace constructor is an overridable algebra construction seam.
It may supply another ordered and oriented full basis. Every ordinary default
SubSpace is then obtained by filtering that full ordering by structural
support:

```text
default_subspace(support)
    = tuple(entry for entry in algebra.default_basis
            if entry.mask in support)
```

This one rule applies to vectors, bivectors, even-grade elements, rotors,
motors, and every other structurally selected multivector space. Rotor and motor
layouts are not special cases.

### Explicit construction

Explicit blade spelling preserves the requested order and orientation exactly:

```python
# Illustrative syntax; final API spelling remains to be selected.
ga.subspace.yz_zx_xy
ga.subspace.from_layout(("yz", "zx", "xy"))
```

The implementation normalizes each individual word to `(mask, sign)` but does
not reorder the resulting entries. Explicit construction bypasses the default
ordering policy and then uses the same flyweight interning path as default
construction.

An API named `from_blades` must not ambiguously mean both "apply the default
order" and "preserve this exact layout". Those construction paths need distinct
names or otherwise unmistakable syntax.

Filtering or restricting an already constructed SubSpace should preserve the
relative order and signs of the surviving coordinates. It must not silently
return them to the algebra default.

## 6. Configured, importable algebras

The default-constructor seam supports importable named algebras with useful
conventions already bound. Conceptually:

```python
generic = Algebra("x+y+z+w0")
# Generic grade-major/lexical default.

from numga.algebras import PGA3D
ctx = NumpyContext(PGA3D)
# Same bit-blade algebraic machinery, with the PGA3D default constructor bound.
```

The exact API may inject a SubSpace factory class, a constructor callable, or
an immutable configured algebra specification. It does not require a registry
of selectable layout profiles.

The default affects construction only. A resulting `SubSpace` records its
actual ordered masks and signs; it carries no extra `"PGA3D default"`,
`"lexical"`, or other provenance annotation.

## 7. The 3D PGA default

For the current generator convention

```text
x^2 = y^2 = z^2 = +1
w^2 = 0
I = xyzw
```

and the current public right-Hodge dual convention, the proposed default basis
is:

```text
grade 0:  1
grade 1:  x, y, z, w
grade 2:  yz, zx, xy, xw, yw, zw
grade 3:  yzw, zxw, xyw, zyx
grade 4:  xyzw
```

The non-canonical spellings encode real signs:

```text
zx   = -xz
zxw  = -xzw
zyx  = -xyz
```

In masks and signs, the nontrivial portions are therefore:

```text
grade 2 masks:  yz, xz, xy, xw, yw, zw
grade 2 signs:  +,  -,  +,  +,  +,  +

grade 3 masks:  yzw, xzw, xyw, xyz
grade 3 signs:  +,   -,   +,   -
```

This ordering makes the important dual maps coefficient identities:

```text
x  -> yzw
y  -> zxw
z  -> xyw
w  -> zyx

yz -> xw
zx -> yw
xy -> zw
```

Thus vector-to-trivector duality is represented by an identity matrix. The
nondegenerate-to-degenerate bivector dual is also an identity matrix. With the
full bivector order

```text
(yz, zx, xy, xw, yw, zw),
```

the right-Hodge dual is the simple block swap

```text
[[0, I],
 [I, 0]].
```

The trivector-to-vector Hodge map is `-I`, as expected from the square of the
chosen Hodge dual on odd grades.

### Quaternion and motor consequences

The rotor and motor orders follow by ordinary grade ordering and support
filtering:

```text
rotor:  1, yz, zx, xy
motor:  1, yz, zx, xy, xw, yw, zw, xyzw
```

The rotor therefore uses the same component positions as a common scalar-first
quaternion tuple:

```text
(real, x, y, z).
```

This is a consequence of the default nondegenerate-bivector ordering, not a
rotor-specific rule. Direct Hamilton-quaternion interoperability also depends
on the chosen multiplication and active/passive action convention: the final
interop contract may require orientation signs even though it requires no
component permutation. That convention must be fixed and tested explicitly.
The motor likewise inherits the algebra's ordinary even-grade order rather than
receiving a special layout.

If the public dual convention or pseudoscalar orientation changes, the signs in
this table must be reconsidered. They are chosen for Numga's current
right-Hodge semantics.

## 8. Relayout and casting

An explicit cast between two layouts of the same support is always available:

```python
target_value = value.cast(target_subspace)
```

This operation preserves the represented algebra element. It is not a raw
buffer reinterpretation.

Let source coordinate `i` and target coordinate `j` have the same canonical
mask. Then:

```text
out[j] = in[i] * source.signs[i] / target.signs[j].
```

Because signs are `+1/-1`, division is equivalent to multiplication. The full
conversion is a signed permutation and is exactly invertible:

```text
cast(T <- S) followed by cast(S <- T) = identity.
```

In the extensor model, a cast is an ordinary unary operator/extensor with
output-first axes:

```text
Cast[T <- S].gatype.subspaces == (T, S)
```

It represents the mathematical identity map expressed in two coordinate
systems. Consequently, a pure same-support relayout preserves all valid value
traits.

Binding and overload dispatch may insert this cast implicitly when supports
match but layouts differ. "Implicit" means that a conversion node is selected
and accounted for; it never means silently treating one layout's buffer as
another. The signed permutation can often be folded into an adjacent operator
kernel or fused by a compiler.

Embedding into a larger support and projection onto a smaller support are not
mere relayouts. Whether they share the public word `cast` remains an open API
decision; internally they must remain distinct from an invertible
same-support conversion. In particular, projection need not preserve traits
such as `Unit` or `Versor`.

## 9. Operator construction

Canonical bit-blade products remain unchanged. Suppose the bit algebra gives:

```text
E[a] * E[b] = p * E[c].
```

For oriented input coordinates

```text
A = s_a * E[a]
B = s_b * E[b]
```

and oriented output coordinate

```text
C = s_c * E[c],
```

the coefficient placed in the layout-specific product kernel is:

```text
p * s_a * s_b / s_c.
```

With `+1/-1` signs this is `p * s_a * s_b * s_c`. Higher-arity kernels apply
the same input-sign product and output-sign correction.

All operator axes contain exact layout-bearing SubSpaces. Selection and output
lookup must therefore map canonical masks to relative coordinate positions;
they cannot use a bit mask as though it were the coordinate index.

Strings are involved only while constructing or displaying a SubSpace. Backend
NumPy, JAX, Torch, or other arrays contain ordinary numeric coefficients, and
materialized kernels contain numeric indices and coefficients. There is no
string or object-array work inside compiled execution.

## 10. GAType, binding, and dispatch

A `GAType` contains exact SubSpace objects:

```text
GAType(
    subspaces=(Output, Input1, ..., InputN),
    traits=...,
)
```

Its flyweight identity consequently includes every axis's exact blade order and
signs. That is necessary because the extensor kernel dimensions are interpreted
using those layouts.

The type system must distinguish:

- exact layout identity;
- same-support conversion compatibility;
- structural support containment; and
- trait entailment.

Generic dispatch matches mathematical support and traits independently of layout.
Exact-layout predicates are the opt-in coefficient-specific specialization;
their existing predicate precedence applies. Distinct layouts remain distinct
cache keys even when they select the same implementation.

Support containment alone must never authorize raw contraction of differently
ordered axes. Binding must either prove exact layout equality or insert the
appropriate conversion before contraction.

GAType refinement deliberately expresses semantic inclusion: two unequal,
same-support and same-fact layouts refine one another. This is a preorder,
not exact layout equality. Binding compatibility is independently determined
by the signed coordinate conversion, never by treating `<=` as raw-array
compatibility.

The signed-permutation rule settles only equal-support layout differences.
Binding across strict support containment additionally requires an embedding,
restriction, or projection policy. That is a separate conversion from relayout
and must be reconciled with the binding rules in `gatype_design.md` and
`extensor_design.md`.

## 11. Caching and performance

Layout-specific SubSpaces, GATypes, operators, and materialized execution plans
must include the ordered sign-bearing axes in their cache keys. A kernel built
for `(xy, xz, yz)` cannot be reused directly for `(yz, zx, xy)`.

The canonical bit-blade product table may still be cached independently of
layout. A layout-specific kernel can be generated by permuting its axes and
folding in the orientation signs. This is an optimization rather than a
requirement for the first implementation.

The expected runtime cost is negligible after lowering:

- signs are folded into operator coefficients;
- order becomes ordinary tensor-axis indexing;
- strings never reach a backend trace; and
- casts become signed gathers/permutations and may be fused away.

A materialized cast is not literally free, but there is no persistent dynamic
metadata overhead in an already compiled contraction. The main practical cost
of supporting many layouts is additional construction and cache entries, not
per-coefficient string processing.

## 12. Ordering behavior of SubSpace operations

`SubSpace.union` is the structural meaning of adding two SubSpaces, with `+` as
optional surface sugar. The result contains every blade support present in
either operand and is returned through the algebra's SubSpace factory.

The following behavior is settled:

- default namespace constructors filter the algebra's default ordering;
- explicit layout constructors preserve their input order and signs;
- restriction or filtering of an existing layout preserves survivor order and
  signs; and
- support comparison ignores order and signs.

The implemented result-layout policy for `union` uses the algebra-default
layout for the resulting support. Alternatives considered were:

- return the algebra-default layout for the resulting support; or
- preserve the left layout and append right-only blades in right order.

`intersection` and restriction use stable left-order filtering, retaining signs.

Generated products and duals select the default layout for their inferred
output support. Grade-sign transforms preserve the input layout. Explicit
output selection requests its exact target layout.

## 13. Implementation outline

The initial migration can remain focused:

1. Extend `SubSpace` with an immutable sign array and include it in its
   flyweight key.
2. Split default support construction from exact layout construction.
3. Parse each explicit blade word into a canonical mask plus permutation sign.
4. Remove unconditional sorting from the explicit-layout path.
5. Make restriction and coordinate lookup order- and sign-aware.
6. Fold input and output signs into all operator kernels.
7. Implement same-support casts as signed permutation extensors.
8. Make the default SubSpace constructor injectable by a configured algebra.
9. Add an importable `PGA3D` configuration with the ordering in this document.
10. Update GAType matching and binding to distinguish equality, conversion, and
    support containment.

Important current assumptions to audit include:

- `SubSpaceFactory.order_blades` and `from_blades`, which currently force one
  ordering;
- set-based union, intersection, and difference;
- slicing paths that reconstruct and reorder a SubSpace;
- relative-index lookup and selection matrices;
- product output construction and squeeze;
- optimized routines that unpack fixed lexical coefficient positions; and
- every cache key containing a SubSpace or GAType.

The existing bit-level product, grade, reverse, and complement algorithms do
not need to be replaced.

## 14. Validation requirements

At minimum, the rewrite needs tests for:

- permutation parity for `xy`, `yx`, `zx`, and `zyx`;
- rejection of repeated generators and duplicate canonical masks;
- order- and sign-sensitive equality and flyweight interning;
- `same_support` and support-containment laws;
- preservation of explicit order through filtering;
- exact signed-permutation cast matrices and cast round trips;
- implicit relayout producing the same result as an explicit cast;
- numerical equality of products, reverse, and dual across alternate layouts;
- generic default-order determinism;
- every exact 3D PGA grade ordering listed above;
- identity-form vector/trivector and bivector dual maps;
- rotor and motor layouts arising through ordinary support filtering;
- quaternion interoperability under the selected action convention;
- backend agreement; and
- absence of strings and object blade identifiers in materialized kernels.

## 15. Deliberate non-goals

This design does not require:

- replacing canonical integer bit blades;
- a pluggable general blade-algebra implementation;
- runtime-selectable named layout profiles;
- storing default-constructor provenance in a SubSpace;
- arbitrary linear combinations as coordinate basis elements; or
- non-diagonal metrics.

Those capabilities may be considered independently later. The present design
solves arbitrary blade ordering and orientation, explicit interoperability
layouts, and useful algebra-specific defaults while retaining the current fast
bit-blade core.
