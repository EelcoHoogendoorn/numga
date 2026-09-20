# Rewrite review, 2026-09-15

Assessment of `rewrite/` against the design docs and the 1.x package. Every claim marked
*verified* was reproduced by running code from `rewrite/` with `PYTHONPATH=src`.
Respond inline; the numbered items are meant to be answered individually.

Actionable pass, 2026-09-18: status notes appear below the owner’s comments.
Validation: 488 tests passed; the existing Agg/plt.show warning remains. NumPy
and JAX rigid-body stepping also ran successfully. Torch and anti-products remain pending.

Test run: `373 passed, 1 warning in 58.73s`. No skips, no xfails.

Logistics: the whole `rewrite/` tree is untracked in git (`?? rewrite/`). Roughly 21k
lines of source, tests and examples with no commit behind them.

Line counts:

| tree | lines |
|---|---|
| legacy `numga/` (non-test) | 7387 |
| `rewrite/src` | 7608 |
| `rewrite/tests` | 7293 |
| `rewrite/examples` | 6250 |

---

## 1. Design goals

### 1.1 Met (verified)

| goal | evidence |
|---|---|
| One `Extensor` for every arity | `extensor/extensor.py:26`; no parallel hierarchy anywhere in `src/` |
| Hole syntax | `V ^ V` arity 2, `v ^ V` arity 1; Levi-Civita and inertia examples from `extensors.md` reproduce exactly (`expression.py:117-157`) |
| Eager `bind` as the one primitive | `extensor.py:220`; no `partial`, `fuse`, `sandwich_map`, `project_map`, `inertia_map` |
| Exact rational kernels, exact squeeze | `SymbolicKernel` stores `Fraction` tuples (`operator/kernel.py:40-73`); zero occurrences of `eps` / `1e-` in `src/` |
| Output-first axes | `np.einsum('ij,...j->...i', S.kernel, pts.kernel)` equals `S(pts)` bit for bit |
| Factory sandwich tight, inline chain loose | PGA3D: `m.sandwich(P)` output support 4; `m * P * m.reverse()` support 8 |
| GAType replaces SubSpace in axes / deduction / dispatch keys | `gatype/gatype.py:39`, `dispatch.py:239` |
| Trait propagation | `(m*m)` carries `{ReverseProductOne, Versor}`; `(R*R*R)(a,b,c).gatype is rotor()`; unit-versor sandwich keeps passenger grade and unit fact (`propagation.py:179-268`) |
| Trait dispatch | `m.inverse()` is bit-identical to `m.reverse()` for a unit versor (`extensions/inverse.py:28`) |
| Subsumption governs bind validity and dispatch specificity | `gatype.py:341-422`, `binding.py:76-152`; wide-into-narrow raises, narrow-into-wide widens |
| Canonicalisation at interning | `traits.py:547-579`; a PGA3 vector's effective traits are `{CliffordConjugateProductScalar, ReverseProductScalar}` with explicit traits empty |
| JAX pytree: leaf = kernel, aux = (gatype, context key) | `backend/jax.py:110-129`; jit over `m.sandwich(P)(pt)` traces |
| Signed / user-ordered layouts | `algebra.subspace("yz zx xy")` vs `"yz xz xy"`: same support, different signs, `==` False |
| Dense and sparse execution agree | `backend/dense.py`, `backend/sparse.py` |
| Hot path is one dict lookup | `dispatch.py:236-252`; `test_dispatch_hot_path.py` asserts it |
| No legacy imports from `rewrite/` | checked |

Warm NumPy timings, PGA3D motor `m` and point `p`, 2000 calls each:

| op | legacy | rewrite |
|---|---|---|
| `m >> p` | 52 µs | 31 µs |
| `m * p` | 34 µs | 17 µs |
| `m.inverse()` | 160 µs | 7 µs |
| motor log | 5.8 ms | 3.7 ms |
| batched `sandwich(P)` on (1000,4) | — | 15.4 µs (raw einsum 10.6 µs) |

The inverse number is the trait system paying for itself.

### 1.2 Drifted from the docs

The code follows `design.md` and `gatype_design.md`. `numga2_design.md` is the stale one,
and it still reads as normative with a "do not reopen" §8.

1. **Separate symbolic `Operator` class was merged after all.** `numga2_design.md` §0 d2, d10
   and §8 say keep it separate. `design.md:1088` and `extensor_design.md:884` say remove it.
   Code: no `Operator` class; staging boundary is `ExactContext` (`backend/exact.py:24`).
2. **Tag vocabulary.** Doc: `norm ∈ {unknown, invertible, unit, null}`, `blade: bool`,
   `versor: bool`. Code: `ProductFact(SelfProduct, ProductResult)` per family plus
   `ProductRelation`, `VersorProduct`, markers `Versor`, `CoefficientOrthogonal`, `Sandwich`,
   `Identity` (`gatype/traits.py`). No `Blade` trait exists at all; the §4 example
   `@mv.exp.register(lambda t: t.tags.blade)` cannot be written. `gatype_design.md` §4 is
   the accurate spec.
3. **Kernel cache key.** Decision 7: key on subspace part only. Code: `_geometric_product`,
   `_wedge`, `_sandwich`, `_symmetric_product` are `lru_cache`d on full `GAType`
   (`operator/factory.py:86,167,213,323`). A unit rotor and a generic even element build
   separate identical kernels.
4. **Sandwich mechanism.** Doc: symmetrised rational kernel with exact zeros. Code:
   unrestricted ternary kernel (`factory.py:347-373`) plus the mod-4 grade theorem applied
   at bind (`propagation.py:192-209`). Same tightness, different machine. Fine, but say so.
NOTE: yeah lets port the original logic

> Status (2026-09-18): Implemented: sandwich kernels now symmetrize the repeated sandwicher slots with exact rational arithmetic and squeeze the resulting exact zeros. Bind-time grade restriction now only supplies the additional versor guarantee.
5. **Stale in the other direction.** `PORTING.md` "not ported yet" and `extensor_design.md`
   §9 / `:465` say sandwich isometry inference, signed subspaces, sparse execution, log/exp
   and equality-group consumption do not exist. All are implemented and tested
   (`propagation.py:166,185` consumes equality groups).
6. **Machinery not in the decisions table**, each with tests and a consumer: sparse
   execution policy, `map_kernel`, `at[].set`, the `Identity` trait and its bind rule.

NOTE: yeah lets carry over at.set. also, sparse execution needs to be ported.

> Status (2026-09-18): Verified already present: `at[].set` works on batched extensors; `execution="sparse"` unrolls exact nonzero terms for NumPy/JAX, including partial binding and inserted open slots. Computed numerical maps use dense application. Existing sparse and indexed-update tests cover these paths.

Recommendation: one consolidating doc pass, mark `numga2_design.md` superseded. Not
incremental patches.

### 1.3 Not met

7. **`null_inverse` raising at trace time.** No `ReverseProductZero` registration in
   `extensions/inverse.py`. *Verified:* inverting the degenerate basis vector `w` returns
   `[nan nan nan nan]` with a runtime divide warning from `backend/context.py:102`. This
   is the failure mode the design said it would eliminate. `ProductResult.ZERO` is the one
   trait category with no consumer.
NOTE: sure fix that

> Status (2026-09-18): Implemented: `ReverseProductZero` selects an inverse that raises before numerical execution, including during JAX tracing. Structurally empty values follow the same rule.
8. **Audit mode** (§4): absent. `design.md:48` arguably retires it; no replacement.
9. **`forget()`** (§1): absent. `with_traits` is `assume`; the only way to strip traits is
   `map_kernel(..., preserve_traits=False)` or `gatype.structural`.
10. **Typed holes**: a refined GAType is accepted as a hole and propagated, but nothing can
    *demand* one; a slot cannot reject an unrefined operand.
11. **Named gatypes**: only `rotor` (`gatype/factory.py:110`). Every example defines its own
    `Point`, `Line`, `Motor`, `Camera`.

### 1.4 Bugs (all reproduced)

12. **JAX iteration never terminates.** `Extensor` has `__getitem__` with no bounds check and
    no `__iter__`; NumPy raises `IndexError`, JAX clamps. `iter()` on a 4-row JAX Extensor
    yields 6+ items with item 5 equal to item 3. The README idiom
    `x, y, z, w = mv.vector(np.eye(4))` fails under `JaxContext` with "too many values to
    unpack". Fix: explicit `__iter__` or a length check in `_batch_kernel_index`.
NOTE: yeah needs fixing

> Status (2026-09-18): Implemented: explicit iteration over the first batch dimension, independent of backend indexing behavior.
13. **Float leaks into `ExactContext`.** `ExactContext.prepare_scalar` accepts floats
    (`backend/exact.py:86`) while `SymbolicKernel._as_fraction` rejects them
    (`kernel.py:23`). `(V*V) * 0.1` stays exact with a 55-bit denominator; the exact `!= 0`
    squeeze is then meaningless downstream. The two policies should agree.
NOTE: fix that

> Status (2026-09-18): Implemented: exact scalar arithmetic accepts integers and rationals, matching `SymbolicKernel`; floats must enter through a numerical context. Affected EM examples now construct numerical GA scalars for those inputs.
14. **Context leak.** `Context.is_compatible_with` (`backend/context.py:120`),
    `_binding_context` (`extensor.py:644`) and `application()` (`application.py:27`) are
    unbounded `lru_cache`s keyed on context identity; `Context` has no `__hash__`. Every
    context ever constructed is retained. 72 `lru_cache` decorators on instance methods in
    total (14 property pairs in `gatype.py`); `cached_property` would be faster and leak
    free where the receiver is interned.
FIX IT

> Status (2026-09-18): numerical application caches are context-owned; bound multivector constructors are cached on their factory; compatibility/context selection no longer have global identity-keyed caches. GAType cached properties are instance-local. GC regression checks cover temporary compatible contexts after application. Structural algebra/type caches remain shared.

15. **Test accommodation in production code.** `gatype.py:520-526` falls back to bare
    `GAType((value,))` to keep "directly constructed test/dialect SubSpaces" usable,
    bypassing the flyweight pool so `is` comparisons fail. `design.md:48` forbids this.
FIX IT

> Status (2026-09-18): removed the bare-GAType test fallback; SubSpaces are lifted through their owning algebra’s canonical factory.

16. Minor: `lru_cache` on `GATypeFactory.__getattr__` caches attribute misses; private
    `_nullary_identity_groups`, `_binding_steps`, `_executor` imported across modules.
ALSO FIX IT

> Status (2026-09-18): forwarded GAType constructors are cached on the factory, not on global `__getattr__`; the three cross-module binding/executor helpers now have explicit public internal names. Attribute errors were not cached by functools in the first place.

17. **Inner product wrapped around for lower left grade (fixed while building the sketches).**
    `OperatorFactory._product` computed grades as `uint8`, so `abs(l - r)` in the `inner`
    rule was 255 for a vector on the left of a bivector and `x | xy` came out empty; only
    `B | v` and `point | plane` were tested. Fixed by casting the grade arrays to `int` in
    `operator/factory.py`. Worth a test for `v | B` on every algebra.
UHM WHAT? trying to use minimal sized integer math for large algebras. just fix the abs with a cast dont ape the intetional subspace size minimation

> Status (2026-09-18): Implemented: grade arrays remain uint8; only the inner-product subtraction uses signed int16 before abs.

### 1.5 Ergonomics regressions (decisions needed)

17. **`~x` for reverse is gone.** No `__invert__` on `Extensor`. Legacy has it
    (`multivector.py:269`).
FIX THIS

> Status (2026-09-18): `~value` calls reverse.

18. **`x << y` changed meaning.** Legacy: reverse sandwich. Rewrite: `inverse_sandwich`
    (`extensor.py:565`). *Verified:* `(m << p)` equals `m.inverse_sandwich(p)`, not
    `m.reverse_sandwich(p)`, for a non-unit `m`. Silent semantic change for ported code.
19. **`ctx.subspace` does not exist.** `extensors.md` spells `ctx.subspace.vector.dual`; the
    real spelling is `ctx.algebra.subspace.vector().dual()`.
NOTE: forwarding to the context makes sense for brevity. however more importantly construction should happen via gatype preferentially; subspace should be viewed more as an internal detail the end user rarely interacts with directly.

> Status (2026-09-18): Implemented: `ctx.gatype` and `ctx.subspace` forward to the algebra; the rigid-body construction now uses `ctx.gatype`.
20. **`NumpyContext('x+y+z+')` no longer accepted**; requires an `Algebra`.
FIX IT

> Status (2026-09-18): NumPy and JAX contexts accept algebra-description strings as well as Algebra objects.

21. **The flagship example regressed.** `examples/mechanics/rigid_body/core.py` was ported
    to `.sandwich(...)`, `.regressive(...)`, `.reverse().sandwich(g).restrict_subspace(...)`
    and lost all its comments, versus legacy `motors >> anchors`, `a & b`, `motor << gravity`.
    The README's "57 lines" pitch no longer holds and the example does not use the syntax the
    rewrite exists to enable. Re-port it to the operator form. The `restrict_subspace` patch
    on the reverse sandwich also deserves a look: `<<` should be tight by the same theorem
    as `>>`.
NOTE yes, this all looks like shit porting; but frankly the example itself needs some love i think. stylistically out of date; more clean math/plumbing seperation would be nice. inertia, integrations and geometric constraint projection are the center pieces; other crap should move out of the way

> Status (2026-09-18): Implemented: inertia construction and motor integration are visible in `Body`; constraints use `>>`, `<<`, and `&` without the old grade-selection patches. State allocation and JAX registration live in `base.py`.

### 1.6 Complexity / YAGNI

Looked specifically for machinery without a consumer. Found essentially none:
`gatype/pattern.py`, `propagation.py`, `binding.py` equality groups, `flyweight.py`,
`algebra/self_product.py`, and every trait class are load-bearing. Genuinely speculative
surface is small: `SelfProduct.product` is parameterised but only `geometric_product` is
ever used; `AxisTransformKind.REFRAME` exists only for an error message;
`Trait.valid_arities=None` serves two classes and forces an escape in `dispatch.py:326`.
No `TODO`, `HACK`, `FIXME`, `type: ignore`, bare `except` anywhere in `src/`. Comments
state the mathematics, not the mechanics. This is not a speculative codebase; if anything
it is under-documented relative to how much it proves.

---

## 2. Functionality missing from 1.x

Ranked by how much a legacy user would feel it. 1 to 5 I would port. 6 to 8 are your call.
9 is deliberately dropped and fine.

1. **`extension/decompose.py` entirely absent.** `decompose_invariant` (simple and
   bisimple), `decompose_polar`, `motor_translator`, `motor_rotor`, `motor_split` (three
   variants), `operator.euclidian_factorization`. Core PGA workhorses; general-motor log
   depends on them.
PORT IT

> Status (2026-09-18): Implemented in `extensions/decompose.py`: polar/invariant bivector decompositions, canonical motor rotation/translation factors, and splitting about a supplied origin. The general origin path covers blade and mixed-coordinate origins, with a canonical Euclidean shortcut.
2. **Torch backend gone.** `numga/backend/torch/` has no counterpart; `run_chain_torch.py`
   cannot run. README advertises torch.
KEEP PENDING
3. **Anti-products and contractions.** No `anti_*` family, no `antify`, no
   `left/right_contraction_product`, `left/right_interior_product`, `left_hodge`,
   `left/right_complement`, `commutator_anti` / `anti_commutator` family. Cheap factory
   formulas (`legacy factory.py:144-201, 273-391`); PGA dialect unusable without them.
i think this is fine? also pending
4. **Named exp/log/root variants.** `exp_linear`, `exp_linear_normalized`, `exp_quadratic` /
   `exp_cayley`, `motor_log_linear(_normalized)`, `motor_log_quadratic`, working-tree
   `motor_log_pade` and `motor_sqrt_denman_beaver`, `motor_geometric_mean`. Integrators call
   these by name. `optimized.py` closed forms for 3D PGA also absent; the rewrite README
   says they "remain to be ported".
motor log becomes specialzied dispatch on regular log with a trait. same for the sqrt? also sure lets port the exp variants. also, optimized yup.

> Status (2026-09-18): Implemented: regular `log`/`square_root` already dispatch on rotor/versor facts. Added linear, normalized-linear, Cayley/quadratic and explicit bisection exponentials; corresponding log approximations, odd-series log, Denman–Beavers root, and motor mean. `extensions.optimized.register()` opts into PGA3 closed forms with signed-layout conversion and a small-angle screw limit.
5. **Post-rewrite legacy work never crossed over** (uncommitted, `git diff --stat numga/`
   is +1618/−207): `SubSpace.inverse_subspace_estimate`, subspace-restricted Shirokov
   inverse, `solve(rhs)`, `precompute_sparse_tensor` and `JaxSparseOperator.partial`,
   `Operator.equals`, the `project(l, r)` symmetry operator, `motor_scalar_square_root`,
   relaxed `decompose_polar` predicate, `numerical_normalize.py` (Newton on the sandwich
   map), and the new conformal-elliptical renderer plus tests. None in `rewrite/`.
6. **Alternative inverse algorithms.** Shirokov, Hitzer / `inverse_factor` family,
   `inverse_la`, `solve`. Rewrite has exactly one general path (`inverse_geometric`).
   Rewrite adds `inverse_orthogonal`, `inverse_linear`, `inverse_nonsquare`.
solve and inverse la essentially get a native extensor spelling now; would be good to make a test for that. Shirokov, Hitzer, port both

> Status (2026-09-18): Implemented and tested: `(value * Full).solve(rhs)` and its unary inverse spelling; `inverse_shirokov`, `inverse_factor`, and `inverse_hitzer`. The legacy Hitzer factor path is dispatched only where its reverse/conjugate product provably reduces to a scalar; the existing recursive inverse handles the broader grade-negation cases.
7. **Subspace predicate and constructor surface.** `grades`, `is_blade`, `is_degenerate`,
   `is_subalgebra`, `minimal_exponential`, the n-simple family, `symmetric_alt_product`,
   `complement`, `difference`, `reject`, `__contains__`, `slice_subspace`; factory
   `translator`, `rotor`, `reflection`, `k_reflection`, `self_reverse`, `mod4`,
   `odd_grade`, `multivector`, `basis`, `blade`, `order_blades`. Patterns replace part of
   this; user dispatch rules against `s.inside.*` need full rewriting.
some of this has been superceded by ga type comparisons? have not missed the rest but port where it cleans up code.

> Status (2026-09-18): Reviewed: these ports use GAType comparisons and existing structural support operations. No additional legacy predicate/constructor aliases were needed.
8. **Operator pretty-printing and the Python codegen backend.** `unary_operator_str`,
   `binary_operator_str`, `test_pinning.py`, `PythonCodegenOperator`. The "show me the
   expanded formula" workflow has no replacement. Rational kernels would make it nicer than
   before; cheap to bring back as a printer over `SymbolicKernel`.
YES, port

> Status (2026-09-18): Implemented: exact extensors expose `formula()` for blade-labelled equations and `to_python()` for standalone coefficient functions at any arity. Rational coefficients remain rational in the generated source; no second operator hierarchy or generated-file tree.
9. **Deliberately dropped, fine:** `upcast` and array-coerced arithmetic, `copy`, the
   `*_map` helpers, `transform` / `reverse_transform`, `degenerate()` / `nondegenerate()`,
   elementwise `sin/cos/sqrt/abs/nan_to_num` on multivectors, `take_along_axis`,
   `rearrange`, `repeat`, `flatten`, the context math namespace (replaced by `context.xp`),
   per-backend operator class hierarchy and the einsum execution path.

Lost examples: `conformal.py` CGA scaffolding, `ga_sparse.py` and `spin_transformations.py`
on top of it, `tennis_racket_theorem_jax.py`, `render_1/2/3.py`, integrators
`explicit_lie_newmark_rev`, `new3`, `momentum_world`.

Lost test areas: `test_decompose`, `test_optimized`, `test_numerical_normalize`,
`test_pinning`, torch benchmark, Shirokov / Hitzer / LA comparisons in `test_inverse`.

Rewrite-only gains, for balance: first-class `bind` and holes, trait inference and
declarative dispatch, exact rational contexts, signed layouts, backend-independent
`stack/concatenate/sum/mean/reshape/broadcast_to/at`, unary `inverse`, `transpose`,
`trace`, named algebras (`PGA2D`, `PGA3D`, `Spherical3D`), and the relativity /
electromagnetism / quadrics / fitting / registration / projection examples.

---

## 3. Example ideas

The current set is strong on physics and relativity. What it lacks is anything that
differentiates through an extensor, and anything where the map itself is the unknown.
Each idea with what it exercises.

1. **Robot arm Jacobian.** Forward kinematics as a product of exponentials. The Jacobian
   from joint rates to end-effector twist is a unary extensor built by binding; inverse
   kinematics via `.inverse()` (or least squares on the kernel) of that map. Reciprocal
   wrench and twist screw systems fall out as the kernel of an extensor. Exercises:
   composition of sandwich maps, arity-1 inverse, the adjoint `m.sandwich(B)`.
2. **Kalman filter on motors.** Pose covariance is a unary extensor on bivectors,
   transported by `m.sandwich(B)` and summed with process noise; measurement update via
   the map inverse. Cleanest demonstration that "the inertia tensor has a PGA
   representation" generalises to every rank-2 object over the Lie algebra. Uncertainty
   ellipsoids animate well.
3. **Fit a camera or an inertia by gradient.** Take the projection example and recover the
   rig motor from image points with `jax.grad` through the extensor pytree; or learn an
   inertia extensor from observed trajectories. The library's stated reason to exist is
   JAX, and no example differentiates yet.
4. **Paraxial optics.** ABCD matrices are unary extensors on lines. A lens is a map on
   lines, an optical system is a composition, and the quadrics module already provides
   curved mirrors. Same pitch as the affine-matrix section, in a domain where people already
   think in matrices.
5. **Group averaging / symmetry-adapted tensors.** Sum `g.sandwich(V)` over a finite point
   group to get a projector onto invariant tensors; apply it to stiffness and inertia to
   derive which components a crystal symmetry allows. Directly showcases extensors that
   broadcast and sum. Kaleidoscope orbits as the picture.
6. **Motor skinning versus matrix skinning side by side.** Blend motors, blend their
   sandwich maps, show the artifacts of each. The two representations are the same library
   object, which is the thesis of `extensors.md`.
7. **Hyperbolic kaleidoscope.** Reflection groups on the Cayley-Klein disk, orbit of a
   fundamental domain. Visually striking and cheap given `cayley_klein.py`.
8. **Kustaanheimo-Stiefel regularisation of the Kepler problem.** Spinor exp and log heavy;
   a good stress test for the log path at large angles.
9. **Linear line complexes.** A bivector as a linear functional on lines; null systems and
   the reciprocal screw system as the kernel of `L | Line`. Small, but a clean
   "extensor of arity 1 on a non-vector space" example.
