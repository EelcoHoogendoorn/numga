"""Second-harmonic generation models a crystal whose electric response is quadratic in the field,
doubling the frequency of light passing through it.

The spacetime algebra holds the electric field as an extrinsic bivector read through the plane a
direction sweeps in time. Each spatial direction and the observer's time direction are intrinsic
antivectors, and their join `bond_directions & time` is the intrinsic antibivector plane the bond
sweeps in time. Vacuum duality `(Direction & time).dual()` maps a spatial direction to the electric
field along it. Pairing a field with a bond plane `Field & bonds` extracts that bond's share of the
field, and summing `bonds * (Field & bonds) * (Field & bonds)` over the four tetrahedral bonds gives
the crystal's bilinear polarization response. Along a long crystal, phase slip between the pump and
the doubled light turns the cosine and sine quadratures of the grown polarization into each other.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA

ga = STA
mv = NumpyContext(ga).multivector
# Fields are extrinsic bivectors; directions are intrinsic antivectors; polarizations are the
# intrinsic antibivector planes that spatial directions sweep in time.
Field = ga.gatype.bivector()                            # Bivector+
Direction = ga.gatype.antivector()                      # Antivector-
Polarization = ga.gatype.antibivector()                 # Antibivector-
Response = ga.gatype((Polarization, Field, Field))      # Antibivector- <- (Bivector+, Bivector+)

# Vacuum duality identifies each spatial direction with the electric field dual to the plane it sweeps in time.
time = -mv.t.dual()                                     # Direction-
field_along = (Direction & time).dual()                 # [] Field+ <- Direction-
# Transverse directions across a beam along x + y, together with their polarizations and unit fields.
axes = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))   # [3] Direction-
horizontal_step = mv.y.dual() - mv.x.dual()             # Direction-
horizontal = horizontal_step / horizontal_step.antinorm()  # Direction-
vertical = mv.z.dual()                                  # Direction-
horizontal_polarization = horizontal & time             # Polarization-
vertical_polarization = vertical & time                 # Polarization-
horizontal_field = field_along(horizontal)              # Field+
vertical_field = field_along(vertical)                  # Field+
# Transverse projection onto the two polarizations carried across the beam.
transverse = (stack((horizontal_polarization, vertical_polarization))
              * (stack((horizontal_field, vertical_field)) & Polarization)).sum(axis=0)   # [] Polarization- <- Polarization-
# Bivector plane generating spatial rotations about the beam axis.
turn_plane = horizontal_field.commutator(vertical_field)                                  # Bivector+
# Four tetrahedral bond directions, their spacetime planes, and the bond strength giving unit bulk coefficients.
bond_steps = (axes * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])).sum(axis=-1)   # [4] Direction-
bond_directions = bond_steps / bond_steps.antinorm()                                      # [4] Direction-
bonds = bond_directions & time                          # [4] Polarization-
bond_strength = 3 * np.sqrt(3) / 4


# --- math -----------------------------------------------------------------------------
def bond_response(bonds: Polarization) -> Response:
    """Sums each bond's quadratic response along its plane (`[4] Polarization-`) into a bulk response
    map (`[] Polarization- <- (Field+, Field+)`)."""
    return bond_strength * (bonds * (Field & bonds) * (Field & bonds)).sum(axis=-1)


crystal = bond_response(bonds)                          # [] Polarization- <- (Field+, Field+)
# Cosine-squared oscillation splits equally between a static offset and the doubled-frequency component.
second_harmonic = 0.5 * transverse(crystal)             # [] Polarization- <- (Field+, Field+)


def growth(source: Polarization, bond_orientation: np.ndarray, mismatch: np.ndarray,
           depths: np.ndarray) -> Polarization:
    """Accumulates the cosine and sine quadratures of `source` across slices (`[..., 2] Polarization-`),
    weighted by each slice's `bond_orientation` and phase slip `mismatch * depths`."""
    thickness = depths[1] - depths[0]
    phase_slip = mismatch * depths
    quadratures = stack((source[..., None] * np.cos(phase_slip), source[..., None] * np.sin(phase_slip)), axis=-1)  # [..., 2] Polarization-
    return (quadratures * bond_orientation[..., None] * thickness).cumsum(axis=-2)
