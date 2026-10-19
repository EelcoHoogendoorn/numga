"""Second-harmonic generation: a crystal whose response is quadratic in the electric field doubles the
frequency of the light passing through it.

The electric field is a bivector, extrinsic: it is read by how much it pushes across a plane. Directions,
the observer's in time and each bond's, are antivectors, intrinsic like all geometry, and their join is
the plane a bond sweeps as the observer moves through time, an antibivector. A bond's share of a field is
the field paired with that plane, `field & bond`, and the crystal's polarization, intrinsic like the
excitation it is part of, is a sum of bond planes weighted by the square of that pairing: a map from two
fields to an antibivector. The doubled light grown through the crystal is tracked as the cosine and sine
amplitudes of one polarization.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA

ga = STA
mv = NumpyContext(ga).multivector
# Fields are bivectors, extrinsic; directions are antivectors, intrinsic, and polarizations, the planes
# directions sweep in time, antibivectors.
Field = ga.gatype.bivector()
Direction = ga.gatype.antivector()
Polarization = ga.gatype.antibivector()
Spatial = ga.gatype.vector().derive.within("x y z")
Rotor = ga.gatype.rotor()
Response = ga.gatype((Polarization, Field, Field))      # Polarization <- (Field, Field)

# The observer's direction in time.
VELOCITY = mv.t.dual()
# Electric planes across a beam along a face diagonal of the crystal, and the planes the directions they
# push along sweep in time: the polarizations along them, each paired to one with its field.
HORIZONTAL_FIELD = (mv.ty - mv.tx) / np.sqrt(2)
VERTICAL_FIELD = mv.tz
HORIZONTAL_POLARIZATION = ((mv.y - mv.x) / np.sqrt(2)).dual() & VELOCITY
VERTICAL_POLARIZATION = mv.z.dual() & VELOCITY
# The polarization across the beam, which light carries: each transverse polarization, weighted by the
# pairing of its field with a polarization.
TRANSVERSE = (stack((HORIZONTAL_POLARIZATION, VERTICAL_POLARIZATION))
              * (stack((HORIZONTAL_FIELD, VERTICAL_FIELD)) & Polarization)).sum(axis=0)   # Polarization <- Polarization
# The four bonds' directions, and the planes they sweep in time, with unit response coefficients.
BOND_DIRECTIONS = mv(Spatial, [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]).dual() / np.sqrt(3)
BONDS = BOND_DIRECTIONS & VELOCITY
BOND_STRENGTH = 3 * np.sqrt(3) / 4


# --- math -----------------------------------------------------------------------------
def bond_response(bonds: Polarization) -> Response:
    """Each bond responds along its plane to the product of the two fields' pairings with that plane."""
    return BOND_STRENGTH * (bonds * (Field & bonds) * (Field & bonds)).sum(axis=-1)


def turned_response(crystal: Response, rotation: Rotor) -> Response:
    """Pull both fields into the crystal and turn its response forward."""
    return rotation >> crystal(rotation << Field, rotation << Field)


def second_harmonic(crystal: Response, pump: Field) -> Polarization:
    """The transverse polarization source at twice the pump frequency."""
    return 0.5 * TRANSVERSE(crystal(pump, pump))


def growth(source: Polarization, bond_orientation: np.ndarray, mismatch: np.ndarray, depths: np.ndarray) -> Field:
    """Accumulate source fields with their temporal phase; final axes are slices and quadratures."""
    thickness = depths[1] - depths[0]
    phase = mismatch * depths
    quadratures = stack((source[..., None] * np.cos(phase), source[..., None] * np.sin(phase)), axis=-1)
    return (quadratures * bond_orientation[..., None] * thickness).cumsum(axis=-2)
