"""What a axis's own metric is like, which decides how forms on that axis can be solved."""

from __future__ import annotations

from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numga.extensor import Extensor

    from .gatype import GAType


class MetricKind(Enum):
    """The default metric of a axis, classified from its exact coefficients.

    An identity metric solves as a plain eigenproblem; a positive one needs the generalized pencil;
    a semidefinite one, positive but blind to some blades as PGA's ideal ones, admits a Hermitian
    eigenproblem on the blades it measures; singular and indefinite ones admit only the general
    eigenproblem.
    """

    IDENTITY = "identity"
    POSITIVE = "positive"
    SEMIDEFINITE = "semidefinite"
    SINGULAR = "singular"
    INDEFINITE = "indefinite"


class Metric:
    """The metric of a nullary type, read as a axis: `axis.metric.default` and `axis.metric.kind`."""

    __slots__ = ("_gatype", "__dict__")

    def __init__(self, gatype: GAType) -> None:
        self._gatype = gatype

    @cached_property
    def default(self) -> Extensor:
        """This axis's metric as a symbolic form: the inner product of S with its reverse.

        On vectors this is V | V. On other grades the reverse removes the sign of the blade's
        square, so bivectors and Euclidean points get a positive metric where S | S is negative.
        """
        algebra = self._gatype.algebra
        return algebra.operator.symmetric_reverse_product(self._gatype).cast(algebra.subspace.scalar())

    @cached_property
    def kind(self) -> MetricKind:
        """What this axis's default metric is like, from its exact coefficients."""
        matrix = self.default._kernel.materialize(np.float64)[0]
        if np.array_equal(matrix, np.eye(len(self._gatype.output_subspace))):
            return MetricKind.IDENTITY
        spectrum = np.linalg.eigvalsh(matrix)
        if np.any(np.abs(spectrum) < 1e-12):
            return MetricKind.SEMIDEFINITE if np.all(spectrum > -1e-12) else MetricKind.SINGULAR
        return MetricKind.POSITIVE if np.all(spectrum > 0) else MetricKind.INDEFINITE
