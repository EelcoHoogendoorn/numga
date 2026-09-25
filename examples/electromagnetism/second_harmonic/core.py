"""A crystal that doubles the frequency of light: its response as a bilinear extensor.

A polar bond responds along itself to the product of the two fields along it, so four bonds pointing
to the corners of a tetrahedron give the crystal's response, `Vector <- (Vector, Vector)`: two
electric fields in, one polarization out. Turning the crystal turns the output and pulls both inputs
back; binding a pump into one input leaves a linear map on the other. A real pump `E * cos(t)`
drives `response(E, E) * cos(t) ** 2`, half of it static and half oscillating at twice the
frequency, so the doubled-frequency amplitude is `0.5 * response(E, E)`; only the part transverse to
the beam radiates forward.

Each slice of the crystal launches doubled-frequency light, and along the crystal the driving
polarization and that light slip out of phase. Inverting the bonds reverses the response, so a
crystal whose orientation flips every time the slip reaches half a turn keeps adding up where a
uniform one cancels: quasi-phase-matching. The phase is carried by the pseudoscalar, which squares
to minus one and commutes with every vector.

In Cartesian tensor notation the response reads as the susceptibility tensor of rank three, and
turning it as three rotation matrices contracted with it; in complex-amplitude notation the growth
reads as the integral of the source times the exponential of the mismatch times the depth. The bond
model follows Hardhienata et al., Bond Model and Group Theory of Second Harmonic Generation in
GaAs(001), https://arxiv.org/abs/1408.1185.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D

ga = VGA3D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Rotor = ga.gatype.rotor()
Phasor = ga.gatype(ga.subspace("1 xyz"))
Response = ga.gatype((Vector, Vector, Vector))                                # Vector <- (Vector, Vector)

# A beam along a face diagonal of the crystal, with horizontal and vertical across it.
HORIZONTAL = (mv.y - mv.x) / np.sqrt(2)                                       # [] Vector
VERTICAL = mv.z                                                                # [] Vector
PROPAGATION = (mv.x + mv.y) / np.sqrt(2)                                      # [] Vector
# The part of a polarization across the beam: only it radiates forward.
TRANSVERSE = Vector - PROPAGATION * (PROPAGATION | Vector)                    # [] Vector <- Vector
# The four directed bonds of a tetrahedral crystal, with the strength that makes the response's
# coefficients one.
BONDS = mv.vector(np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3))   # [bonds] Vector
STRENGTH = 3 * np.sqrt(3) / 4


# --- math -----------------------------------------------------------------------------
def response(bonds: Vector) -> Response:
    """The polarization two fields drive: each bond responds along itself to the product of the
    two fields along it."""
    return STRENGTH * (bonds * (bonds | Vector) * (bonds | Vector)).sum(axis=-1)   # [...] Vector <- (Vector, Vector)


def turned(crystal: Response, rotation: Rotor) -> Response:
    """The response of the crystal turned by a rotor: both fields pulled back into the crystal,
    the polarization turned forward."""
    return rotation >> crystal(rotation << Vector, rotation << Vector)          # [...] Vector <- (Vector, Vector)


def doubled(crystal: Response, pump: Vector) -> Vector:
    """The doubled-frequency polarization across the beam that a pump drives."""
    return 0.5 * TRANSVERSE(crystal(pump, pump))                              # [...] Vector


def growth(orientation: np.ndarray, mismatch: np.ndarray, depths: np.ndarray) -> Phasor:
    """The amplitude of the doubled-frequency light after each slice of a crystal, on the last axis,
    in units of the source of a unit length of crystal: the running sum of each slice's orientation,
    plus one or minus one, turned by the phase it has slipped at its depth."""
    thickness = depths[1] - depths[0]
    slip = (mv.xyz * (mismatch * depths)).exp()                               # [..., slices] Phasor
    return (slip * orientation * thickness).cumsum(axis=-1)                    # [..., slices] Phasor
