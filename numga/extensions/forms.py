"""Spectra of forms: eigenproblems and determinant against a second form.

A form has two covector axes and no spectrum of its own: its eigenvalues and determinant are
relative to a second form on the same axes, the pencil's partner: a physical form such as a mass or
an energy, or a metric form where the question is about lengths. Left out, the partner is the axis's
own metric, the inner product of S with its reverse, which is V | V on vectors and, on a projective
element, the condition that makes it a unit one. A partner built from types alone, the axis's own
metric included, has exact coefficients, and its kind picks the solver: against an
identity the plain solver, against a positive form the generalized pencil, against a semidefinite
one, blind to some blades as PGA's ideal ones, the pencil on the blades it measures, and against an
indefinite or singular one only the general eigenproblem. A numeric partner takes the generalized
pencil. Pairing a form's two axes is `contract`; `trace` pairs axes without a metric and refuses a
form. Generalized form eigenproblems use SciPy on the NumPy backend (the linalg extra).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np

from numga.backend.context import binding_context
from numga.extensor import Extensor
from numga.gatype import GATypeError, MetricKind
from numga.gatype.metric import classify

from numga.extensions._linalg import _eigenpairs, _scalars


def _form_map(value: Extensor) -> Extensor:
    value = value.cast(value.algebra.subspace.scalar())
    return Extensor._from_prepared_kernel(
        value.context, value.algebra.gatype.oriented(value.input_subspaces, value.gatype.input_orientations),
        value._kernel[..., 0, :, :],
    )


def _on_second_axis(value: Extensor, partner: Extensor) -> Extensor:
    """The partner as a map, in the layout of the form's second axis."""
    identity = value.algebra.operator.identity(value.gatype.derive.axis(2))
    return _form_map(partner)(identity).cast(value.axes[2])


def _pencil(value: Extensor, partner: Extensor) -> tuple[Extensor, Extensor]:
    context = binding_context(value.context, (partner.context,))
    return _form_map(context.lower(value)).cast(value.axes[2]), _on_second_axis(value, context.lower(partner))


@lru_cache(maxsize=None)
def _classified(shape: tuple[int, int], coefficients: bytes) -> tuple[MetricKind, np.ndarray, np.ndarray, np.ndarray]:
    """The kind of an exact partner, and the positions of the blades it measures, of those it is
    blind to, and the order that puts the two back into the axis's layout."""
    matrix = np.frombuffer(coefficients).reshape(shape)
    blind = ~matrix.any(axis=0)
    measured, unmeasured = np.flatnonzero(~blind), np.flatnonzero(blind)
    return classify(matrix), measured, unmeasured, np.argsort(np.concatenate([measured, unmeasured]))


def _partner_kind(value: Extensor, partner: Extensor) -> tuple[MetricKind, np.ndarray, np.ndarray, np.ndarray]:
    """What the partner is like: classified from exact coefficients, or positive for a numeric partner,
    which takes the generalized pencil."""
    if not partner.context.is_exact:
        return MetricKind.POSITIVE, np.array([]), np.array([]), np.array([])
    # Exact coefficients carry no batch: one matrix, in the layout the pencil uses.
    lowered = binding_context(value.context, (partner.context,)).lower(partner)
    # Exact coefficients are real, also when lowered into a complex context.
    coefficients = np.real(np.asarray(_on_second_axis(value, lowered)._kernel)).astype(np.float64)
    matrix = np.ascontiguousarray(coefficients.reshape(coefficients.shape[-2:]))
    return _classified(matrix.shape, matrix.tobytes())


def _reduced(value: Extensor, partner: Extensor):
    """The pencil on the blades a semidefinite partner measures, with the blind blades eliminated.

    For each mode the blind components are those that make the form stationary, a Schur
    complement: in PGA, the best translation for each rotation. Returns the form as a map, the
    reduced form and partner, the map from measured to blind components, and the blade order.
    """
    left, right = _pencil(value, partner)
    _, measured, unmeasured, order = _partner_kind(value, partner)
    xp = left.context.xp

    def block(matrix, rows, columns):
        return matrix[..., rows, :][..., :, columns]

    form, metric = left._kernel, right._kernel
    coupling = -xp.linalg.solve(block(form, unmeasured, unmeasured), block(form, unmeasured, measured))
    reduced = block(form, measured, measured) + block(form, measured, unmeasured) @ coupling
    return left, reduced, block(metric, measured, measured), coupling, order


def _refuse(method: str, needs: str, kind: MetricKind) -> GATypeError:
    return GATypeError(f"{method} of a form needs {needs} metric form; the one it is measured against is {kind.value}")


# --- without a partner: the axis's own metric ---------------------------------------------
def _against_own_metric(method: str) -> Callable[[Extensor], Extensor]:
    def solve(value: Extensor) -> Extensor:
        return getattr(Extensor, method)(value, value.gatype.derive.axis(2).metric.default)
    return solve


for _method in ("eig", "eigvals", "eigh", "eigvalsh", "det"):
    getattr(Extensor, _method).register(lambda gatype: gatype.form.is_square)(_against_own_metric(_method))


# --- against a partner -------------------------------------------------------------------
@Extensor.eig.register(lambda value, partner: value.form.matches(partner))
def eig_form(value: Extensor, partner: Extensor) -> tuple[Extensor, Extensor]:
    """Generalized form eigenpairs, including infinite eigenvalues against a singular partner.

    The NumPy backend uses SciPy (the linalg extra); mode selection belongs to the caller.
    """
    left, right = _pencil(value, partner)
    if _partner_kind(value, partner)[0] is MetricKind.IDENTITY:
        return left.eig()
    return _eigenpairs(left, *left.context.generalized_eig(left._kernel, right._kernel))


@Extensor.eigvals.register(lambda value, partner: value.form.matches(partner))
def eigvals_form(value: Extensor, partner: Extensor) -> Extensor:
    left, right = _pencil(value, partner)
    if _partner_kind(value, partner)[0] is MetricKind.IDENTITY:
        return left.eigvals()
    return _scalars(left, left.context.generalized_eigvals(left._kernel, right._kernel))


@Extensor.eigh.register(lambda value, partner: value.form.matches(partner))
def eigh_form(value: Extensor, partner: Extensor) -> tuple[Extensor, Extensor]:
    """Hermitian form eigenpairs, normalized in a positive or semidefinite partner; against a
    semidefinite one, one finite mode per blade it measures, with the blind components chosen to make
    the form stationary."""
    kind = _partner_kind(value, partner)[0]
    if kind in (MetricKind.INDEFINITE, MetricKind.SINGULAR):
        raise _refuse("eigh", "a positive-semidefinite", kind)
    if kind is MetricKind.SEMIDEFINITE:
        left, reduced, metric, coupling, order = _reduced(value, partner)
        values, measured = left.context.generalized_eigh(reduced, metric)
        vectors = left.context.xp.concatenate([measured, coupling @ measured], axis=-2)[..., order, :]
        return _eigenpairs(left, values, vectors)
    left, right = _pencil(value, partner)
    if kind is MetricKind.IDENTITY:
        return left.eigh()
    return _eigenpairs(left, *left.context.generalized_eigh(left._kernel, right._kernel))


@Extensor.eigvalsh.register(lambda value, partner: value.form.matches(partner))
def eigvalsh_form(value: Extensor, partner: Extensor) -> Extensor:
    kind = _partner_kind(value, partner)[0]
    if kind in (MetricKind.INDEFINITE, MetricKind.SINGULAR):
        raise _refuse("eigvalsh", "a positive-semidefinite", kind)
    if kind is MetricKind.SEMIDEFINITE:
        left, reduced, metric, _, _ = _reduced(value, partner)
        return _scalars(left, left.context.generalized_eigvalsh(reduced, metric))
    left, right = _pencil(value, partner)
    if kind is MetricKind.IDENTITY:
        return left.eigvalsh()
    return _scalars(left, left.context.generalized_eigvalsh(left._kernel, right._kernel))


@Extensor.det.register(lambda value, partner: value.form.matches(partner))
def det_form(value: Extensor, partner: Extensor) -> Extensor:
    """Determinant of the form relative to an invertible partner: det(partner^-1 value), frame
    independent."""
    kind = _partner_kind(value, partner)[0]
    if kind in (MetricKind.SEMIDEFINITE, MetricKind.SINGULAR):
        raise _refuse("det", "an invertible", kind)
    left, right = _pencil(value, partner)
    if kind is MetricKind.IDENTITY:
        return left.det()
    return left.det() / right.det()


# --- coefficient factor ------------------------------------------------------------------
@Extensor.cholesky.register(lambda value: value.form.is_square)
def cholesky_form(value: Extensor) -> Extensor:
    """Lower unary coefficient factor, with both axes in the second axis layout."""
    return _form_map(value).cholesky()
