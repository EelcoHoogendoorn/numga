# numga 2.0 — extensor unification

Design for the compat-breaking 2.0 rewrite. Companion to [`extensors.md`](rewrite/docs/extensors.md)
(user-facing motivation); this doc is the implementation model. Two themes:

1. **Merge the value types** — `MultiVector` and `ConcreteOperator` become one
   arity-parameterised `Extensor`, constructible inline (`v ^ V`).
2. **GAType** — the type layer: `(subspace, tags)` replaces bare `SubSpace` throughout,
   carrying value-level promises (`unit`, `null`, `blade`, `versor`) through deduction
   and dispatch.

## 0. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Merge `MultiVector` + `ConcreteOperator` into one `Extensor`; arity is a property, nullary = multivector | that is where the real duplication lives: per-backend `shape`/`at`/`__getitem__`/`concatenate`/`sum`/broadcasting, and `partial` vs `__call__` vs `*_map` |
| 2 | The symbolic `Operator` stays a separate, context-free class | it is the compile-time staging layer; what 2.0 removes is the boundary's *rigidity* — bind and compose freely cross it |
| 3 | Binding is **eager**, always | no identified case where laziness finds zeros that named ops + late binding of hole-expressions don't; eager keeps effective matrices materialized — printable, serializable, batch-ready |
| 4 | Inline construction via **holes**: a subspace in argument position is an unbound slot — `v ^ V` | the syntax goal of the rewrite; any concrete argument ⇒ `Extensor`, all holes ⇒ `Operator` |
| 5 | Repeated-argument operators (`sandwich` & co) remain factory methods | the repetition must be *declared* to be exploited; an inline chain cannot see slot 0 == slot 2 |
| 6 | `GAType` = interned `(subspace, tags)` replaces `SubSpace` throughout the type layer: axes, deduction, dispatch keys | value-level refinement rides the exact machinery subspaces use today |
| 7 | Kernels constructed & cached keyed by the **subspace part only**; tags transfer via cheap per-op tables at bind | wedge of a unit vector and of a generic vector are the same kernel — full-gatype keys would fragment the cache |
| 8 | Operator axes **output-first**: `kernel[batch..., O, I1..In]` (`todo.md:35`) | a unary extensor is then a standard matrix (`'ij,...j->...i'` — the layout `extensors.md`'s affine example assumes); arity-0 layout unchanged from 1.0 |
| 9 | Symbolic kernels carry exact **rational** coefficients | `symmetry` permutation-averaging creates fractions; float storage is why `squeeze` needs `eps = 1e-6` (`operator.py:215`); rational restores exact `== 0` |
| 10 | Symbolic operators are **context-free**; extensors inherit context from bound values | kills `ConcreteOperator` = "operator + context pointer"; contexts hold per-backend execution plans and caches |
| 11 | Norm tag is a class `{unknown, invertible, unit, null}`, not a number | numeric norm doesn't survive `+`; the class covers every identified fast path |
| 12 | Kernel storage stays dense; byte-string-blades deferred | orthogonal to all of the above (§8) |

## 1. GAType — the type layer

```
GAType (interned flyweight, as SubSpace is today):
    subspace: SubSpace      # the blade set — SubSpace survives as the inner component
    tags:     Tags          # value-level promises
```

Tags, v1 (fields shaped so later refinements slot in):

- `norm ∈ {unknown, invertible, unit, null}` — the class of `x·~x`.
- `blade: bool` — simple; a wedge of vectors; squares to a scalar.
- `versor: bool` — product of invertible vectors. Later refinement: the reflection
  *count* k ("n-reflection"), which distinguishes 2-reflection (simple motor →
  closed-form log) from 4-reflection (general motor → decompose path) — the split
  `logexp.py` currently makes via subspace heuristics.

Mechanics:

- **Deduction lifts unchanged in shape.** 1.0's `SubSpace.product(other) -> SubSpace`
  (cached op + squeeze) becomes `GAType.product(other) -> GAType`: subspace part computed
  exactly as today, tag part by transfer table (§4).
- **Canonicalization at interning**: subspace-implied tags are baked in when the
  flyweight is constructed (grade-1 ⇒ blade; bivector in ≤3d ⇒ blade; scalar ⇒
  everything), so equal meanings are the same object and dispatch keys are canonical.
- **Subsumption**: `(s₁,t₁) ⊑ (s₂,t₂)` iff `s₁ ⊆ s₂` and `t₁ ⊇ t₂` — smaller space, more
  promises = more specific. One relation governs both binding validity and dispatch
  specificity.
- **Named gatypes** as context-level constructors: `Rotor = GAType(even_grade, {unit})`,
  `Versor = GAType(even_grade, {versor})`, … (namespace details open, §7). Typed holes
  follow: a factory slot may *demand* `GAType(even_grade, {unit})`.
- Tags are **promises**, not runtime-checked (audit mode excepted, §4). They enter via
  constructors (`normalized() → unit`, `v ∧ w → blade`), canonicalization, or explicit
  `x.assume(...)` / `x.forget()`.

## 2. Extensor and Operator — the value layer

```
Extensor:                       # merged 1.0 MultiVector + ConcreteOperator
    context: Context            # backend + execution policy + plan cache
    axes:    Tuple[GAType]      # axes[0] = output, axes[1:] = open input slots
    kernel:  backend array      # [batch..., len(axes[0]), len(axes[1]), ...]

    arity  = len(axes) - 1      # 0 = multivector; layout identical to 1.0 values
    gatype = axes[0]

Operator:                       # 1.0 Operator, kept separate — the compile-time layer
    axes:    Tuple[GAType]      # input tags act as slot requirements (usually empty)
    kernel:  rational numpy     # exact; context-free; factory-cached by subspace parts
```

- The class boundary *is* the staging boundary (compile-time vs runtime data). 2.0 makes
  it freely crossable: `Operator ∘ Operator → Operator` (rational fuse + exact squeeze);
  binding values into an `Operator` → `Extensor` (eager contraction, context inherited
  from the values); fusing an `Operator` onto an fp `Extensor` → `Extensor`, deduction
  treating the extensor as a **generic element of its axes-gatype** — sound because its
  axes are tight by construction (rational squeeze ran before contraction), and
  value-specific fp zeros are runtime zeros: never narrowed on, invisible under jit
  anyway.
- **Contexts.** Symbolic operators live in the algebra, shared across backends (1.0
  already splits `algebra.operator` vs `context.operator`); their rational-numpy kernel
  is the metaprogramming substrate, not a backend choice. A context = backend + execution
  policy (dense einsum vs sparse unroll, jit) + a cache of execution plans and
  backend-dtype kernels, keyed by symbolic op. Mixing contexts in one bind is an error.
- **jax pytree**: aux = `(axes, context)`, leaf = kernel.

## 3. Syntax: holes and `bind`

**The hole rule.** A subspace (or gatype) in argument position is an unbound slot:

```python
V ^ V          # Operator, arity 2  — the Levi-Civita table
v ^ V          # Extensor, arity 1  — cross-product matrix of v
v ^ w          # Extensor, arity 0  — a value
m.sandwich(P)  # Extensor, arity 1  — factory: slots 0,2 declared identical
```

Any concrete argument ⇒ eager `Extensor`; all holes ⇒ symbolic `Operator`. This replaces
the `isinstance(inp, SubSpace)` hack at `numpy/operator.py:117` — the same feature,
promoted from hack to design.

**`bind`, the one operation.** `y.bind({i: x})` contracts `x`'s output axis against slot
`i`; multi-slot dicts bind in one call. Subsumes 1.0's `fuse` / `bind` / `partial` /
`__call__` / `sandwich_map` / `project_map` / `inertia_map`. Validity: `x.gatype ⊑
y.axes[i]` (implicit-zero widening via `select`, as today). Batch axes broadcast left.

**Repeated arguments stay factory-declared.** `m.sandwich(P)` knows slots 0 and 2 hold
the same motor: symmetrized rational kernel, plus the isometry transfer rule (§4). The
inline spelling `m * P * ~m` works but cannot see the repetition — looser kernel, no
isometry rule.

**Where you bind is where you commit.** Tight types come from named ops, or from
hole-expressions composed symbolically and bound *late* (a multi-slot bind of
`Q * P * ~Q` at the end also hands the symmetry information over — both slots filled with
the same object in one call). Eagerly-bound concrete chains (`q * v * ~q`) come out loose
— `V ⊕ trivector` with ε-garbage in the mathematically-zero components — exactly as in
1.0. Known, bounded, user-controlled.

## 4. Tag transfer and dispatch

Transfer rules attach to **named factory ops** (a raw kernel doesn't know it's a
sandwich; the factory does). Output tags are recomputed at bind time from the actual
argument tags; the kernel is reused from the subspace-keyed cache (decision 7).

| op | transfer |
|----|----------|
| `product` | blade·blade → versor; versor·versor → versor. Norm, **gated on both versor**: unit·unit → unit; invertible·invertible → invertible; anything·null → null |
| `reverse` / `involute` / `conjugate` | preserve all tags |
| `sandwich` by **unit** versor | passenger gatype preserved **verbatim** — isometry: a unit point stays a unit point through a motor |
| `sandwich` by invertible versor | preserves blade/versor; norm keeps invertible/null, drops unit |
| `wedge` | blade ∧ blade → blade; norm dropped |
| `+` | subspace part: union (as 1.0); tags dropped (norm isn't additive; blade + blade isn't a blade) |
| `normalized` | sets unit |
| `dual` | preserves blade; norm → unknown in v1 (metric-dependent) |

The versor gate keeps norm transfer sound: only versors have scalar-valued — hence
multiplicative — norms.

**Dispatch**: `SubspaceDispatch` (`dynamic_dispatch.py:53`) keys on `gatype` instead of
`subspace`; registration-order predicates as today, fast paths registered first:

```python
@mv.inverse.register(lambda t: t.tags.norm == 'unit')
def unit_inverse(x): return x.reverse()          # motors: one sign flip vs Hitzer/Shirokov

@mv.exp.register(lambda t: t.tags.blade)
def blade_exp(b): ...                            # closed-form cos/sinc, no decompose

@mv.inverse.register(lambda t: t.tags.norm == 'null')
def null_inverse(x): raise ZeroDivisionError     # at trace time, not NaN at runtime
```

**Soundness**: a transferred tag must hold for *every* value consistent with the input
tags — in doubt, drop it. Optional **audit mode** for tests: numerically verify tags at
construction (`allclose(norm², 1)`, `allclose(x∧x, 0)`); off under jit.

## 5. What 2.0 deletes

- `AbstractMultiVector` + `AbstractConcreteOperator` as parallel hierarchies, and their
  per-backend duplication (`numpy/multivector.py` vs `numpy/operator.py`, ditto
  jax/torch).
- `partial` vs `fuse` vs `__call__` vs `*_map` as distinct code paths → `bind`.
- the `isinstance(SubSpace)` branch in operator `__call__` → the hole rule.
- `squeeze`'s `eps` → rational kernels.
- `sandwich_map` / `project_map` / `inertia_map` helpers → partial binds of factory ops
  (`m.sandwich(P)`).

## 6. Phasing

0. **Rational symbolic kernels** — standalone, 1.x-compatible. Exact `squeeze`.
1. **GAType shell + Extensor merge + `bind` + holes**, numpy backend. `GAType` lands as a
   thin all-empty-tags interned wrapper so `axes: Tuple[GAType]` from day one; tag
   *semantics* arrive in phase 3. Port the extension modules (`inverse`, `norms`,
   `logexp`, …) and tests.
2. **jax/torch backends** — pytree aux = `(axes, context)`, leaf = kernel; physics
   examples as the integration test.
3. **Tags** — transfer tables, canonicalization rules, gatype dispatch, audit mode, the
   three fast paths above.
4. **Deferred**: byte-string-blades / basis permutations (`todo.md:26-29`);
   reflection-count refinement of the versor tag; provenance-based symmetry detection for
   hand-written chains.

## 7. Open questions

- Tags v1 field shape: `versor` as bool, or reflection count from the start?
- Which subspace⇒tag canonicalization implications to bake in (grade-1 ⇒ blade, ≤3d
  bivector ⇒ blade, scalar ⇒ all; others?).
- Namespace for named gatypes: does `ctx.subspace` grow tags, or a separate `ctx.type`?
- Typed holes (factory slots *demanding* e.g. unit) in v1, or later?
- Whether `*` on `Operator` means composition or stays unoverloaded (explicit `.bind`).
- Rational representation: `fractions.Fraction` object arrays vs int
  numerator/denominator pair — decide in phase 0.

## 8. Considered and rejected (do not reopen without new evidence)

- **Deferred/lazy binding; one-level recipes/provenance** — no identified use case over
  named ops + late binding of hole-expressions; and eager binding is what keeps effective
  matrices materialized (print, serialize, batch-apply). Both schemes are strictly
  type-*tightening*, so either can be retrofitted without breaking semantics if a real
  use case ever appears.
- **Merging the symbolic `Operator` into `Extensor`** — interleaving needs the
  compile/runtime boundary to be *crossable*, not gone; as a class split it carries its
  weight (context-free, rational, factory-cached).
- **Sparse canonical kernel storage / structural masks** — the axes carry all structure
  that deduction, dispatch, and tags consume; storage is an orthogonal perf choice
  (`extensors.md §Sparsity`).
- **Unbounded expression history** — grows without bound in loops (`r ← r * step`).
- **`SubSpace` as a kind of extensor** — it is the inner component of `GAType`; the hole
  rule gives the ergonomics without the ontology.
