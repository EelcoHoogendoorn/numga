# Numga 2 porting ledger

This ledger records the deliberately selected ideas that cross from the 1.x
reference tree into the isolated rewrite.  Production code in `rewrite/` never
imports the reference implementation.

The source revision is the repository `HEAD` plus the working-tree content that
was present when each entry was reviewed. The ledger records human-reviewed,
one-way ports; it is not an executable bridge to the legacy package.

| Reference material | Rewrite destination | Treatment | Why it survives | Coverage |
|---|---|---|---|---|
| `numga/algebra/description.py`, `numga/algebra/bitops.py`, `numga/algebra/algebra.py` | `src/numga/algebra/` | Reimplemented and narrowed | Canonical bit blades and diagonal-metric products are context-free and remain the low-level algebra model. Mutable caches and vectorized-array-only APIs were not copied. | `tests/test_algebra.py` |
| `numga/subspace/subspace.py`, `numga/subspace/factory.py` | `src/numga/subspace/` | Reimplemented behind the final axis boundary | Phase 1 needs structural support, interning, and stable support queries. Legacy array-position assumptions and the broad dynamic factory surface were not copied. | `tests/subspace/test_subspace.py` |
| Whole-object typing discussion in the new design documents | `src/numga/gatype/` | New implementation with focused product propagation | `GAType` now carries canonical marker facts plus reverse-norm and versor-closure relations for geometric product. Binding substitutes certified nullary facts and nested positive-arity relations without inspecting coefficients. | `tests/gatype/test_gatype.py`, `tests/gatype/test_propagation.py` |
| `numga/operator/operator.py`, selected formulas from `numga/operator/factory.py` | `src/numga/operator/` | Reimplemented output-first with exact rational storage | Symbolic staging and algebraic factory formulas remain useful; output-last storage, epsilon simplification, and compatibility APIs do not. | `tests/operator/` |
| Partial binding ideas in `numga/operator/operator.py` and backend operator implementations | `src/numga/binding.py`, `src/numga/backend/` | Reimplemented around one value-free plan | A single deterministic planner owns slot numbering, structural conversions, and input splicing; trait substitution consumes the same explicit result-slot splices while backends only execute the plan. Equality groups are recorded but not yet used for diagonal traits. | `tests/test_binding.py`, `tests/test_operator_extensor.py`, `tests/gatype/test_propagation.py` |
| Concrete multivector/operator behavior | `src/numga/extensor/`, `src/numga/multivector/` | Replaced by one native class plus a nullary construction facade | One immutable runtime class now represents nullary values and positive-arity maps. `context.multivector` is only a typed constructor namespace; the legacy class hierarchy is reference material only. | `tests/test_multivector_factory.py`, `tests/extensor/` |
| `numga/dynamic_dispatch.py`, `numga/multivector/extension/` | `src/numga/gatype/dispatch.py`, `src/numga/extension.py` | Replaced underneath the familiar type-method registration shape | `ExtensionMethod` descriptors live on `Extensor`; portable overloads use algebra-independent arity/trait patterns and concrete GATypes retain algebra-local support matching. An explicit GAType-predicate tier survives only for exact-layout low-level kernels. Declarative resolution is most-specific, while declaration order remains the diagnosed tie-breaker for incomparable overlap. | `tests/gatype/test_dispatch.py`, `tests/test_extensions.py` |
| JAX array-namespace and pytree concepts | `src/numga/backend/jax.py` | Reimplemented as one optional policy | The shared dense binding path traces without a JAX-specific Extensor or operator hierarchy. Only the kernel is dynamic pytree data; `GAType` and the immutable context key are static metadata. | `tests/test_jax_backend.py` |
| Representative legacy tests and examples | focused rewrite tests | Semantics restated case by case | Selected behavior becomes direct 2.0 assertions; legacy harness internals and print-only tests are not copied. | `tests/test_geometric_integration.py` |

## Deliberately not ported yet

- logarithm/exponential, decomposition, normalization, and remaining inverse
  implementations stay selectively deferred; the unit-value and
  coefficient-orthogonal inverse overloads are native rather than bulk-ported;
- sandwich `Isometry`/passenger-preservation inference and a general relation
  language; the implemented relation is deliberately limited to reverse norms
  of geometric products;
- signed and user-ordered SubSpaces, until the full layout-aware kernel and cast
  cluster can land atomically;
- legacy aliases, import paths, wrappers, serialization formats, and extension
  registration machinery;
- sparse and unrolled execution strategies, pending measurements against the
  dense reference path; and
- backend-specific feature parity beyond the explicitly selected 2.0 support
  matrix.
