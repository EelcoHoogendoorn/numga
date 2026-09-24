"""Spectra of forms: eigenproblems, determinant and trace relative to a metric form.

A form has two covector slots and no spectrum of its own: its eigenvalues,
determinant and trace are relative to a metric form. Given explicitly, the metric
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
from numga.gatype import GAType
from numga.subspace import SubSpace

from numga.extensions._linalg import _eigenpairs, _is_square_form, _matching_forms, _scalars


def _form_map(value: Extensor) -> Extensor:
    value = value.cast(value.algebra.subspace.scalar())
    return Extensor._from_prepared_kernel(
        value.context, value.algebra.gatype(value.input_subspaces), value._kernel[..., 0, :, :],
    )


def _pencil(value: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    context = binding_context(value.context, (metric.context,))
    left = _form_map(context.lower(value)).cast(value.axes[2])
    right = _form_map(context.lower(metric))
    right = right(value.algebra.operator.identity(value.axes[2])).cast(value.axes[2])
    return left, right


# --- the slot's own metric ---------------------------------------------------------------
@lru_cache(maxsize=None)
def _default_metric(slot: SubSpace) -> Extensor:
    """The slot's metric as a symbolic form: the inner product of S with its reverse.

    On vectors this is V | V. On other grades the reverse removes the sign of the blade's
    square, so bivectors and Euclidean points get a positive metric where S | S is negative.
    """
    algebra = slot.algebra
    return algebra.operator.symmetric_reverse_product(slot).cast(algebra.subspace.scalar())


@lru_cache(maxsize=None)
def _default_metric_kind(slot: SubSpace) -> str:
    """Classify the default metric from its exact coefficients: a property of the slot type.

    'identity' solves as a plain eigenproblem; 'positive' needs the generalized pencil;
    'singular' and 'indefinite' admit only the general eigenproblem.
    """
    matrix = _default_metric(slot)._kernel.materialize(np.float64)[0]
    if np.array_equal(matrix, np.eye(len(slot))):
        return "identity"
    spectrum = np.linalg.eigvalsh(matrix)
    if np.any(np.abs(spectrum) < 1e-12):
        return "singular"
    return "positive" if np.all(spectrum > 0) else "indefinite"


def _metric_is(*kinds: str) -> Callable[[GAType], bool]:
    """Dispatch predicate: a square form whose slot metric is of one of these kinds."""
    return lambda gatype: _is_square_form(gatype) and _default_metric_kind(gatype.subspaces[2]) in kinds


def _plain(value: Extensor) -> Extensor:
    """The form as a map, for a slot metric that is the identity."""
    return _form_map(value).cast(value.axes[2])


def _default_pencil(value: Extensor) -> tuple[Extensor, Extensor]:
    return _pencil(value, _default_metric(value.axes[2]))


def _refusal(method: str, needs: str) -> Callable[[Extensor], Extensor]:
    def refuse(value: Extensor) -> Extensor:
        raise TypeError(
            f"{method} of a form needs {needs} metric; the metric of {value.axes[2]} is "
            f"{_default_metric_kind(value.axes[2])}: pass an explicit metric form"
        )
    return refuse


# --- against the slot's metric: identity -------------------------------------------------
@Extensor.eig.register(_metric_is("identity"))
def eig_form_identity(value: Extensor) -> tuple[Extensor, Extensor]:
    return _plain(value).eig()


@Extensor.eigvals.register(_metric_is("identity"))
def eigvals_form_identity(value: Extensor) -> Extensor:
    return _plain(value).eigvals()


@Extensor.eigh.register(_metric_is("identity"))
def eigh_form_identity(value: Extensor) -> tuple[Extensor, Extensor]:
    return _plain(value).eigh()


@Extensor.eigvalsh.register(_metric_is("identity"))
def eigvalsh_form_identity(value: Extensor) -> Extensor:
    return _plain(value).eigvalsh()


@Extensor.det.register(_metric_is("identity"))
def det_form_identity(value: Extensor) -> Extensor:
    return _plain(value).det()


@Extensor.trace.register(_metric_is("identity"), position=0)
def trace_form_identity(value: Extensor) -> Extensor:
    return _plain(value).trace()


# --- against the slot's metric: the pencil -----------------------------------------------
@Extensor.eig.register(_metric_is("positive", "indefinite", "singular"))
def eig_form_pencil(value: Extensor) -> tuple[Extensor, Extensor]:
    """Eigenpairs against the slot's metric; a singular metric gives infinite modes."""
    left, right = _default_pencil(value)
    return _eigenpairs(left, *left.context.generalized_eig(left._kernel, right._kernel))


@Extensor.eigvals.register(_metric_is("positive", "indefinite", "singular"))
def eigvals_form_pencil(value: Extensor) -> Extensor:
    left, right = _default_pencil(value)
    return _scalars(left, left.context.generalized_eigvals(left._kernel, right._kernel))


@Extensor.eigh.register(_metric_is("positive"))
def eigh_form_pencil(value: Extensor) -> tuple[Extensor, Extensor]:
    """Hermitian eigenpairs, orthonormal in the slot's metric, which must be positive definite."""
    left, right = _default_pencil(value)
    return _eigenpairs(left, *left.context.generalized_eigh(left._kernel, right._kernel))


@Extensor.eigvalsh.register(_metric_is("positive"))
def eigvalsh_form_pencil(value: Extensor) -> Extensor:
    left, right = _default_pencil(value)
    return _scalars(left, left.context.generalized_eigvalsh(left._kernel, right._kernel))


@Extensor.det.register(_metric_is("positive", "indefinite"))
def det_form_pencil(value: Extensor) -> Extensor:
    """Determinant relative to the slot's metric, which must be invertible."""
    left, right = _default_pencil(value)
    return left.det() / right.det()


@Extensor.trace.register(_metric_is("positive", "indefinite"), position=0)
def trace_form_pencil(value: Extensor) -> Extensor:
    """Trace of the form raised by the slot's metric, which must be invertible."""
    left, right = _default_pencil(value)
    return right.solve(left).trace()


# --- against the slot's metric: refused --------------------------------------------------
Extensor.eigh.register(_metric_is("indefinite", "singular"))(_refusal("eigh", "a positive-definite"))
Extensor.eigvalsh.register(_metric_is("indefinite", "singular"))(_refusal("eigvalsh", "a positive-definite"))
Extensor.det.register(_metric_is("singular"))(_refusal("det", "an invertible"))
Extensor.trace.register(_metric_is("singular"), position=0)(_refusal("trace", "an invertible"))


# --- against an explicit metric ----------------------------------------------------------
@Extensor.eig.register(_matching_forms)
def eig_form_metric(value: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    """Generalized form eigenpairs, including infinite eigenvalues for singular metrics.

    The NumPy backend uses SciPy (the linalg extra); mode selection belongs to the caller.
    """
    left, right = _pencil(value, metric)
    return _eigenpairs(left, *left.context.generalized_eig(left._kernel, right._kernel))


@Extensor.eigh.register(_matching_forms)
def eigh_form_metric(value: Extensor, metric: Extensor) -> tuple[Extensor, Extensor]:
    """Generalized Hermitian form eigenpairs, normalized in a positive-definite metric."""
    left, right = _pencil(value, metric)
    return _eigenpairs(left, *left.context.generalized_eigh(left._kernel, right._kernel))


@Extensor.eigvals.register(_matching_forms)
def eigvals_form_metric(value: Extensor, metric: Extensor) -> Extensor:
    left, right = _pencil(value, metric)
    return _scalars(left, left.context.generalized_eigvals(left._kernel, right._kernel))


@Extensor.eigvalsh.register(_matching_forms)
def eigvalsh_form_metric(value: Extensor, metric: Extensor) -> Extensor:
    left, right = _pencil(value, metric)
    return _scalars(left, left.context.generalized_eigvalsh(left._kernel, right._kernel))


@Extensor.det.register(_matching_forms)
def det_form_metric(value: Extensor, metric: Extensor) -> Extensor:
    """Determinant of the form relative to a metric form: det(metric^-1 value), frame independent."""
    left, right = _pencil(value, metric)
    return left.det() / right.det()


# --- coefficient factor ------------------------------------------------------------------
@Extensor.cholesky.register(_is_square_form)
def cholesky_form(value: Extensor) -> Extensor:
    """Lower unary coefficient factor, with both axes in the second slot layout."""
    return _form_map(value).cholesky()
