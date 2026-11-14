"""Bilinear forms (`Scalar <- (S, S)`) pair two covector inputs and carry no intrinsic spectrum of
their own. Eigenvalues and determinants of a form are always defined relative to a second form
(`partner`) on the same axes (`form(v) == eigenvalue * partner(v)`), such as a kinetic-energy or mass
form in mechanics or a metric form when measuring geometric lengths. Omitting `partner` supplies the
second axis's canonical metric form `S.scalar_product(S.reverse())`, which equals `V | V` on vectors
and measures projective bulk norm on `PGA` elements. Summing a form's two axes against that metric is
`contract`; `trace` pairs complementary spaces without a metric and refuses a form on one space.

Partner classification governs solver selection. When `partner` has exact symbolic coefficients (as
any type-derived metric form does), its matrix is classified once and cached along with the indices of
measured and unmeasured blades: an identity partner delegates to the unary map solver, a
positive-definite partner runs the generalized pencil, an indefinite or singular partner permits
`eig` (and `det` when invertible) while refusing `eigh`, and a runtime numeric `partner` dispatches to
the generalized pencil directly. Generalized form eigenproblems on the NumPy backend use SciPy
(`numga[linalg]`).

Positive-semidefinite exact partners arise naturally in `PGA`, where ideal translation or horizon
blades carry zero metric weight. On such a partner, `eigh` and `eigvalsh` eliminate the null blades
by a Schur complement that makes `form` stationary along the unmeasured subspace for each mode (for
instance, pairing each principal rotation axis with its natural screw pitch or centre), leaving a
positive-definite generalized eigenproblem over the measured blades alone. Form eigenproblems and
relative determinants are developed in `docs/conventions.md`, `docs/frameworks.md`, and
`docs/tutorial/extensors/11-eigenvalues-and-singular-values.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np

from numga.backend.context import binding_context
from numga.extensor import Extensor
from numga.gatype import GATypeError, MetricKind
from numga.gatype.metric import classify

from numga.extensions._linalg import _eigenpairs, _scalars


def _form_map(form: Extensor) -> Extensor:
    form = form.cast(form.algebra.subspace.scalar())
    return Extensor._from_prepared_kernel(
        form.context, form.algebra.gatype.oriented(form.input_subspaces, form.gatype.input_orientations),
        form._kernel[..., 0, :, :],
    )


def _on_second_axis(form: Extensor, partner: Extensor) -> Extensor:
    identity = form.algebra.operator.identity(form.gatype.derive.axis(2))
    return _form_map(partner)(identity).cast(form.axes[2])


def _pencil(form: Extensor, partner: Extensor) -> tuple[Extensor, Extensor]:
    context = binding_context(form.context, (partner.context,))
    return _form_map(context.lower(form)).cast(form.axes[2]), _on_second_axis(form, context.lower(partner))


@lru_cache(maxsize=None)
def _classified(shape: tuple[int, int], coefficients: bytes) -> tuple[MetricKind, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.frombuffer(coefficients).reshape(shape)
    blind = ~matrix.any(axis=0)
    measured, unmeasured = np.flatnonzero(~blind), np.flatnonzero(blind)
    return classify(matrix), measured, unmeasured, np.argsort(np.concatenate([measured, unmeasured]))


def _partner_kind(form: Extensor, partner: Extensor) -> tuple[MetricKind, np.ndarray, np.ndarray, np.ndarray]:
    if not partner.context.is_exact:
        return MetricKind.POSITIVE, np.array([]), np.array([]), np.array([])
    # Exact symbolic partners carry no batch axes and have real coefficients even in complex contexts.
    lowered = binding_context(form.context, (partner.context,)).lower(partner)
    coefficients = np.real(np.asarray(_on_second_axis(form, lowered)._kernel)).astype(np.float64)
    matrix = np.ascontiguousarray(coefficients.reshape(coefficients.shape[-2:]))
    return _classified(matrix.shape, matrix.tobytes())


def _reduced(form: Extensor, partner: Extensor) -> tuple:
    left, right = _pencil(form, partner)
    _, measured, unmeasured, order = _partner_kind(form, partner)
    xp = left.context.xp

    def block(matrix: Any, rows: np.ndarray, columns: np.ndarray) -> Any:
        return matrix[..., rows, :][..., :, columns]

    # Eliminate blades to which partner is blind by solving for stationary unmeasured components.
    form_kernel, metric = left._kernel, right._kernel
    coupling = -xp.linalg.solve(block(form_kernel, unmeasured, unmeasured), block(form_kernel, unmeasured, measured))
    reduced = block(form_kernel, measured, measured) + block(form_kernel, measured, unmeasured) @ coupling
    return left, reduced, block(metric, measured, measured), coupling, order


def _refuse(method: str, needs: str, kind: MetricKind) -> GATypeError:
    return GATypeError(f"{method} of a form needs {needs} metric form; the one it is measured against is {kind.value}")


# --- without a partner: the axis's own metric ---------------------------------------------
def _against_own_metric(method: str) -> Callable:
    def solve(form: Extensor) -> Any:
        return getattr(Extensor, method)(form, form.gatype.derive.axis(2).metric.default)
    return solve


for _method in ("eig", "eigvals", "eigh", "eigvalsh", "det"):
    getattr(Extensor, _method).register(lambda gatype: gatype.form.is_square)(_against_own_metric(_method))


# --- against a partner -------------------------------------------------------------------
@Extensor.eig.register(lambda t, partner: t.form.matches(partner))
def eig_form(form: Extensor, partner: Extensor) -> tuple[Extensor, Extensor]:
    """The generalized eigendecomposition solves the pencil `form(v) == eigenvalue * partner(v)` with
    modes on the trailing batch axis `[..., modes]`, admitting infinite eigenvalues when `partner` is
    singular."""
    left, right = _pencil(form, partner)
    if _partner_kind(form, partner)[0] is MetricKind.IDENTITY:
        return left.eig()
    return _eigenpairs(left, *left.context.generalized_eig(left._kernel, right._kernel))


@Extensor.eigvals.register(lambda t, partner: t.form.matches(partner))
def eigvals_form(form: Extensor, partner: Extensor) -> Extensor:
    """Computing generalized eigenvalues returns the spectrum `[..., modes] Scalar` of the pencil
    `(form, partner)` without eigenvectors."""
    left, right = _pencil(form, partner)
    if _partner_kind(form, partner)[0] is MetricKind.IDENTITY:
        return left.eigvals()
    return _scalars(left, left.context.generalized_eigvals(left._kernel, right._kernel))


@Extensor.eigh.register(lambda t, partner: t.form.matches(partner))
def eigh_form(form: Extensor, partner: Extensor) -> tuple[Extensor, Extensor]:
    """The Hermitian eigendecomposition of a form returns ascending eigenvalues `[..., modes] Scalar`
    and `partner`-orthonormal modes `[..., modes]`; against a positive-semidefinite `partner`, Schur
    reduction yields one finite mode per measured blade with stationary components on the null blades."""
    kind = _partner_kind(form, partner)[0]
    if kind in (MetricKind.INDEFINITE, MetricKind.SINGULAR):
        raise _refuse("eigh", "a positive-semidefinite", kind)
    if kind is MetricKind.SEMIDEFINITE:
        left, reduced, metric, coupling, order = _reduced(form, partner)
        values, measured = left.context.generalized_eigh(reduced, metric)
        vectors = left.context.xp.concatenate([measured, coupling @ measured], axis=-2)[..., order, :]
        return _eigenpairs(left, values, vectors)
    left, right = _pencil(form, partner)
    if kind is MetricKind.IDENTITY:
        return left.eigh()
    return _eigenpairs(left, *left.context.generalized_eigh(left._kernel, right._kernel))


@Extensor.eigvalsh.register(lambda t, partner: t.form.matches(partner))
def eigvalsh_form(form: Extensor, partner: Extensor) -> Extensor:
    """Computing Hermitian form eigenvalues returns ascending finite eigenvalues `[..., modes] Scalar`
    relative to a positive-definite or positive-semidefinite `partner`."""
    kind = _partner_kind(form, partner)[0]
    if kind in (MetricKind.INDEFINITE, MetricKind.SINGULAR):
        raise _refuse("eigvalsh", "a positive-semidefinite", kind)
    if kind is MetricKind.SEMIDEFINITE:
        left, reduced, metric, _, _ = _reduced(form, partner)
        return _scalars(left, left.context.generalized_eigvalsh(reduced, metric))
    left, right = _pencil(form, partner)
    if kind is MetricKind.IDENTITY:
        return left.eigvalsh()
    return _scalars(left, left.context.generalized_eigvalsh(left._kernel, right._kernel))


@Extensor.det.register(lambda t, partner: t.form.matches(partner))
def det_form(form: Extensor, partner: Extensor) -> Extensor:
    """The relative determinant returns the basis-independent ratio `det(form) / det(partner)` against
    an invertible `partner` form."""
    kind = _partner_kind(form, partner)[0]
    if kind in (MetricKind.SEMIDEFINITE, MetricKind.SINGULAR):
        raise _refuse("det", "an invertible", kind)
    left, right = _pencil(form, partner)
    if kind is MetricKind.IDENTITY:
        return left.det()
    return left.det() / right.det()


# --- coefficient factor ------------------------------------------------------------------
@Extensor.cholesky.register(lambda t: t.form.is_square)
def cholesky_form(form: Extensor) -> Extensor:
    """The coefficient Cholesky factorization returns the lower triangular unary map `L` in the second
    axis's blade layout."""
    return _form_map(form).cholesky()
