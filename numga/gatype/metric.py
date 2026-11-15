"""Every nullary `GAType` induces a canonical quadratic metric form from the scalar part of the
symmetric reverse product `0.5 * (x * ~x + ~x * x)`. On grade-1 vectors this coincides with the
inner product `V | V`, while on higher grades the reverse cancels the grade-dependent sign of blade
squares so Euclidean bivectors and points acquire a positive-definite metric even where `S | S` is
negative.

Materializing the exact coefficient matrix of this form classifies the space into one of five
`MetricKind` regimes, determining which eigensolvers and metric factorizations apply to forms and
operators over that space. Identity metrics reduce generalized eigenproblems to standard Hermitian
ones, positive-definite metrics admit Cholesky pencils, positive-semidefinite metrics (such as
projective spaces with ideal null blades) restrict Hermitian eigensolvers to the measured subspace,
and singular or indefinite metrics (such as Minkowski spacetime) require general non-Hermitian
solvers.
"""

from __future__ import annotations

from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numga.extensor import Extensor

    from .gatype import GAType


class MetricKind(Enum):
    """Exact eigenvalue regimes of a symmetric metric matrix select the solver strategy for forms
    over a space."""

    IDENTITY = "identity"
    POSITIVE = "positive"
    SEMIDEFINITE = "semidefinite"
    SINGULAR = "singular"
    INDEFINITE = "indefinite"


class Metric:
    """A nullary type's metric namespace caches its canonical symmetric-reverse form and spectral
    classification."""

    __slots__ = ("_gatype", "__dict__")

    def __init__(self, gatype: GAType) -> None:
        self._gatype = gatype

    @cached_property
    def default(self) -> Extensor:
        """The scalar part of the symmetric reverse product defines the space's canonical bilinear
        metric form."""
        algebra = self._gatype.algebra
        return algebra.operator.symmetric_reverse_product(self._gatype).cast(algebra.subspace.scalar())

    @cached_property
    def kind(self) -> MetricKind:
        """Exact coefficient eigenvalues classify the space's default metric into a `MetricKind`."""
        return classify(self.default._kernel.materialize(np.float64)[0])


def classify(metric_matrix: np.ndarray) -> MetricKind:
    """Eigenvalues of a symmetric metric matrix distinguish identity, positive-definite,
    positive-semidefinite, singular indefinite, and nonsingular indefinite forms."""
    if np.array_equal(metric_matrix, np.eye(len(metric_matrix))):
        return MetricKind.IDENTITY
    spectrum = np.linalg.eigvalsh(metric_matrix)
    if np.any(np.abs(spectrum) < 1e-12):
        return MetricKind.SEMIDEFINITE if np.all(spectrum > -1e-12) else MetricKind.SINGULAR
    return MetricKind.POSITIVE if np.all(spectrum > 0) else MetricKind.INDEFINITE
