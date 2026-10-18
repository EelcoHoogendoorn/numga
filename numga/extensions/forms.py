"""Spectra of forms: eigenproblems and determinant relative to a metric form.

A form has two covector slots and no spectrum of its own: its eigenvalues and
determinant are relative to a metric form, as is pairing its two slots, which is
`contract`; `trace` pairs slots without a metric and refuses a form. Given explicitly, the metric
selects the generalized pencil. Left out, it is the slot's own metric, the inner
product of S with its reverse, which is V | V on vectors. Its kind follows from
the slot type, so dispatch picks the implementation once per type: an identity
keeps the plain solver, other metrics use the pencil, and a metric that cannot
support the operation is refused. Generalized form eigenproblems use SciPy on the
NumPy backend (the optional linalg extra).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np

from numga.backend.context import binding_context
from numga.extensor import Extensor
from numga.gatype import GAType, MetricKind
from numga.subspace import SubSpace

from numga.extensions._linalg import _eigenpairs, _scalars


def _form_map(value: Extensor) -> Extensor:
    value = value.cast(value.algebra.subspace.scalar())
    return Extensor._from_prepared_kernel(
        value.context, value.algebra.gatype.oriented(value.input_subspaces, value.gatype.input_orientations),
        value._kernel[..., 0, :, :],
    )


def _pencil(value: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    context = binding_context(value.context, (metric.context,))
    left = _form_map(context.lower(value)).cast(value.axes[2])
    right = _form_map(context.lower(metric))
    right = right(value.algebra.operator.identity(value.gatype.derive.axis(2))).cast(value.axes[2])
    return left, right


# --- the slot's own metric ---------------------------------------------------------------
@lru_cache(maxsize=None)
def _measured_blades(slot: GAType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions of the blades the slot's metric measures and of those it is blind to, and the
    order that puts the two back into the slot's layout."""
    matrix = slot.metric.default._kernel.materialize(np.float64)[0]
    blind = ~matrix.any(axis=0)
    measured, unmeasured = np.flatnonzero(~blind), np.flatnonzero(blind)
    return measured, unmeasured, np.argsort(np.concatenate([measured, unmeasured]))


def _metric_is(*kinds: MetricKind) -> Callable[[GAType], bool]:
    """Dispatch predicate: a square form whose slot metric is of one of these kinds."""
    return lambda gatype: gatype.form.is_square and gatype.derive.axis(2).metric.kind in kinds


def _plain(value: Extensor) -> Extensor:
    """The form as a map, for a slot metric that is the identity."""
    return _form_map(value).cast(value.axes[2])


def _default_pencil(value: Extensor) -> tuple[Extensor, Extensor]:
    return _pencil(value, value.gatype.derive.axis(2).metric.default)


def _refusal(method: str, needs: str) -> Callable[[Extensor], Extensor]:
    def refuse(value: Extensor) -> Extensor:
        raise TypeError(
            f"{method} of a form needs {needs} metric; the metric of {value.axes[2]} is "
            f"{value.gatype.derive.axis(2).metric.kind.value}: pass an explicit metric form"
        )
    return refuse


# --- against the slot's metric: identity -------------------------------------------------
@Extensor.eig.register(_metric_is(MetricKind.IDENTITY))
def eig_form_identity(value: Extensor) -> tuple[Extensor, Extensor]:
    return _plain(value).eig()


@Extensor.eigvals.register(_metric_is(MetricKind.IDENTITY))
def eigvals_form_identity(value: Extensor) -> Extensor:
    return _plain(value).eigvals()


@Extensor.eigh.register(_metric_is(MetricKind.IDENTITY))
def eigh_form_identity(value: Extensor) -> tuple[Extensor, Extensor]:
    return _plain(value).eigh()


@Extensor.eigvalsh.register(_metric_is(MetricKind.IDENTITY))
def eigvalsh_form_identity(value: Extensor) -> Extensor:
    return _plain(value).eigvalsh()


@Extensor.det.register(_metric_is(MetricKind.IDENTITY))
def det_form_identity(value: Extensor) -> Extensor:
    return _plain(value).det()


# --- against the slot's metric: the pencil -----------------------------------------------
@Extensor.eig.register(_metric_is(MetricKind.POSITIVE, MetricKind.INDEFINITE, MetricKind.SEMIDEFINITE, MetricKind.SINGULAR))
def eig_form_pencil(value: Extensor) -> tuple[Extensor, Extensor]:
    """Eigenpairs against the slot's metric; a singular metric gives infinite modes."""
    left, right = _default_pencil(value)
    return _eigenpairs(left, *left.context.generalized_eig(left._kernel, right._kernel))


@Extensor.eigvals.register(_metric_is(MetricKind.POSITIVE, MetricKind.INDEFINITE, MetricKind.SEMIDEFINITE, MetricKind.SINGULAR))
def eigvals_form_pencil(value: Extensor) -> Extensor:
    left, right = _default_pencil(value)
    return _scalars(left, left.context.generalized_eigvals(left._kernel, right._kernel))


@Extensor.eigh.register(_metric_is(MetricKind.POSITIVE))
def eigh_form_pencil(value: Extensor) -> tuple[Extensor, Extensor]:
    """Hermitian eigenpairs, orthonormal in the slot's metric, which must be positive definite."""
    left, right = _default_pencil(value)
    return _eigenpairs(left, *left.context.generalized_eigh(left._kernel, right._kernel))


@Extensor.eigvalsh.register(_metric_is(MetricKind.POSITIVE))
def eigvalsh_form_pencil(value: Extensor) -> Extensor:
    left, right = _default_pencil(value)
    return _scalars(left, left.context.generalized_eigvalsh(left._kernel, right._kernel))


@Extensor.det.register(_metric_is(MetricKind.POSITIVE, MetricKind.INDEFINITE))
def det_form_pencil(value: Extensor) -> Extensor:
    """Determinant relative to the slot's metric, which must be invertible."""
    left, right = _default_pencil(value)
    return left.det() / right.det()


# --- against the slot's metric: semidefinite ---------------------------------------------
def _reduced_pencil(value: Extensor):
    """The pencil on the blades the metric measures, with the blind blades eliminated.

    For each mode the blind components are those that make the form stationary, a Schur
    complement: in PGA, the best translation for each rotation. Returns the form as a map, the
    reduced form and metric, the map from measured to blind components, and the blade order.
    """
    left, right = _default_pencil(value)
    measured, unmeasured, order = _measured_blades(value.gatype.derive.axis(2))
    xp = left.context.xp

    def block(matrix, rows, columns):
        return matrix[..., rows, :][..., :, columns]

    form, metric = left._kernel, right._kernel
    coupling = -xp.linalg.solve(block(form, unmeasured, unmeasured), block(form, unmeasured, measured))
    reduced = block(form, measured, measured) + block(form, measured, unmeasured) @ coupling
    return left, reduced, block(metric, measured, measured), coupling, order


@Extensor.eigh.register(_metric_is(MetricKind.SEMIDEFINITE))
def eigh_form_semidefinite(value: Extensor) -> tuple[Extensor, Extensor]:
    """Hermitian eigenpairs against a semidefinite slot metric: one finite mode per blade the
    metric measures, orthonormal in it, with the blind components chosen to make the form
    stationary."""
    left, reduced, metric, coupling, order = _reduced_pencil(value)
    values, measured = left.context.generalized_eigh(reduced, metric)
    vectors = left.context.xp.concatenate([measured, coupling @ measured], axis=-2)[..., order, :]
    return _eigenpairs(left, values, vectors)


@Extensor.eigvalsh.register(_metric_is(MetricKind.SEMIDEFINITE))
def eigvalsh_form_semidefinite(value: Extensor) -> Extensor:
    left, reduced, metric, _, _ = _reduced_pencil(value)
    return _scalars(left, left.context.generalized_eigvalsh(reduced, metric))


# --- against the slot's metric: refused --------------------------------------------------
Extensor.eigh.register(_metric_is(MetricKind.INDEFINITE, MetricKind.SINGULAR))(_refusal("eigh", "a positive-semidefinite"))
Extensor.eigvalsh.register(_metric_is(MetricKind.INDEFINITE, MetricKind.SINGULAR))(_refusal("eigvalsh", "a positive-semidefinite"))
Extensor.det.register(_metric_is(MetricKind.SEMIDEFINITE, MetricKind.SINGULAR))(_refusal("det", "an invertible"))


# --- against an explicit metric ----------------------------------------------------------
@Extensor.eig.register(lambda value, metric: value.form.matches(metric))
def eig_form_metric(value: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    """Generalized form eigenpairs, including infinite eigenvalues for singular metrics.

    The NumPy backend uses SciPy (the linalg extra); mode selection belongs to the caller.
    """
    left, right = _pencil(value, metric)
    return _eigenpairs(left, *left.context.generalized_eig(left._kernel, right._kernel))


@Extensor.eigh.register(lambda value, metric: value.form.matches(metric))
def eigh_form_metric(value: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    """Generalized Hermitian form eigenpairs, normalized in a positive-definite metric."""
    left, right = _pencil(value, metric)
    return _eigenpairs(left, *left.context.generalized_eigh(left._kernel, right._kernel))


@Extensor.eigvals.register(lambda value, metric: value.form.matches(metric))
def eigvals_form_metric(value: Extensor, metric: Extensor) -> Extensor:
    left, right = _pencil(value, metric)
    return _scalars(left, left.context.generalized_eigvals(left._kernel, right._kernel))


@Extensor.eigvalsh.register(lambda value, metric: value.form.matches(metric))
def eigvalsh_form_metric(value: Extensor, metric: Extensor) -> Extensor:
    left, right = _pencil(value, metric)
    return _scalars(left, left.context.generalized_eigvalsh(left._kernel, right._kernel))


@Extensor.det.register(lambda value, metric: value.form.matches(metric))
def det_form_metric(value: Extensor, metric: Extensor) -> Extensor:
    """Determinant of the form relative to a metric form: det(metric^-1 value), frame independent."""
    left, right = _pencil(value, metric)
    return left.det() / right.det()


# --- coefficient factor ------------------------------------------------------------------
@Extensor.cholesky.register(lambda value: value.form.is_square)
def cholesky_form(value: Extensor) -> Extensor:
    """Lower unary coefficient factor, with both axes in the second slot layout."""
    return _form_map(value).cholesky()
