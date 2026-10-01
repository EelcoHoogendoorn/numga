"""Frequency doubling from an electric-dipole crystal.

Two electric field bivectors drive a polarization source through a bilinear extensor.
Each directed bond spans an electric plane with the observer's time direction.
Coherent growth retains two real bivector quadratures of the temporal oscillation.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA

ga = STA
mv = NumpyContext(ga).multivector
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()
Response = ga.gatype((Bivector, Bivector, Bivector))

# Electric planes across a beam along a face diagonal of the crystal.
HORIZONTAL = (mv.ty - mv.tx) / np.sqrt(2)
VERTICAL = mv.tz
LONGITUDINAL = (mv.tx + mv.ty) / np.sqrt(2)
ELECTRIC = (Bivector - (mv.t >> Bivector)) / 2
TRANSVERSE = ELECTRIC - LONGITUDINAL * (LONGITUDINAL | Bivector)
# Four directed bond planes, with unit response coefficients.
BONDS = mv(ga.subspace("tx ty tz"), [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
STRENGTH = 3 * np.sqrt(3) / 4


# --- math -----------------------------------------------------------------------------
def response(bonds: Bivector) -> Response:
    """Each bond responds to the product of the two electric fields along it."""
    return STRENGTH * (bonds * (bonds | Bivector) * (bonds | Bivector)).sum(axis=-1)


def turned(crystal: Response, rotation: Rotor) -> Response:
    """Pull both fields into the crystal and turn its response forward."""
    return rotation >> crystal(rotation << Bivector, rotation << Bivector)


def doubled(crystal: Response, pump: Bivector) -> Bivector:
    """The transverse polarization source at twice the pump frequency."""
    return 0.5 * TRANSVERSE(crystal(pump, pump))


def growth(source: Bivector, orientation: np.ndarray, mismatch: np.ndarray, depths: np.ndarray) -> Bivector:
    """Accumulate source fields with their temporal phase; final axes are slices and quadratures."""
    thickness = depths[1] - depths[0]
    phase = mismatch * depths
    quadratures = stack((source[..., None] * np.cos(phase), source[..., None] * np.sin(phase)), axis=-1)
    return (quadratures * orientation[..., None] * thickness).cumsum(axis=-2)
