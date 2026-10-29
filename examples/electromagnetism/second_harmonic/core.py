"""Second-harmonic generation: a crystal whose response is quadratic in the electric field doubles the
frequency of the light passing through it.

The electric field is a bivector, extrinsic: it is read through the plane a direction sweeps in time.
Directions, the observer's in time and each bond's, are antivectors, intrinsic like all geometry, and their
join is the plane a bond sweeps as the observer moves through time, an antibivector. The field along a
direction is the dual of the plane the direction sweeps in time, `(Direction & time).dual()`: the vacuum's
identification of a direction with the field along it. A bond's share of a field is the field paired with
its plane, `field & bond`, and the crystal's polarization, intrinsic like the excitation it is part of, is a
sum of bond planes weighted by the square of that pairing: a map from two fields to an antibivector. The
doubled light grown through the crystal is tracked as the cosine and sine amplitudes of one polarization.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA

ga = STA
mv = NumpyContext(ga).multivector
# Fields are bivectors, extrinsic; directions are antivectors, intrinsic, and polarizations, the planes
# directions sweep in time, antibivectors.
Field = ga.gatype.bivector()                            # Bivector+
Direction = ga.gatype.antivector()                      # Antivector-
Polarization = ga.gatype.antibivector()                 # Antibivector-
Response = ga.gatype((Polarization, Field, Field))      # Polarization- <- (Field+, Field+)

# The observer's direction in time, and the field along a direction: the dual of the plane the direction
# sweeps in time, the vacuum's identification of a direction with the field along it.
time = -mv.t.dual()                                     # [] Direction-
field_along = (Direction & time).dual()                 # [] Field+ <- Direction-
# The directions across a beam along a face diagonal of the crystal, the polarizations along them, the planes
# they sweep in time, and the fields along them, each pairing to one with its polarization.
# The directions of space, along each axis.
axes = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))  # [3] Direction-
horizontal_step = mv.y.dual() - mv.x.dual()             # [] Direction-
horizontal = horizontal_step / horizontal_step.squared().square_root()   # [] Direction-
vertical = mv.z.dual()                                  # [] Direction-
horizontal_polarization = horizontal & time             # [] Polarization-
vertical_polarization = vertical & time                 # [] Polarization-
horizontal_field = field_along(horizontal)              # [] Field+
vertical_field = field_along(vertical)                  # [] Field+
# The polarization across the beam, which light carries: each transverse polarization, weighted by the
# pairing of its field with a polarization.
transverse = (stack((horizontal_polarization, vertical_polarization))
              * (stack((horizontal_field, vertical_field)) & Polarization)).sum(axis=0)   # [] Polarization- <- Polarization-
# The plane of turns about the beam.
turn_plane = horizontal_field.commutator(vertical_field)                                   # [] Bivector+
# The four bonds' directions, the planes they sweep in time, and the bond strength that gives the crystal
# unit response coefficients.
bond_steps = (axes * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])).sum(axis=-1)   # [bonds] Direction-
bond_directions = bond_steps / bond_steps.squared().square_root()                         # [bonds] Direction-
bonds = bond_directions & time                          # [bonds] Polarization-
bond_strength = 3 * np.sqrt(3) / 4


# --- math -----------------------------------------------------------------------------
def bond_response(bonds: Polarization) -> Response:
    """Each bond responds along its plane to the product of the two fields' pairings with that plane."""
    return bond_strength * (bonds * (Field & bonds) * (Field & bonds)).sum(axis=-1)


crystal = bond_response(bonds)                          # [] Polarization- <- (Field+, Field+)
# The polarization at twice the pump's frequency, across the beam: a pump oscillating as a cosine drives the
# response to its square, half of it static and half at twice the frequency.
second_harmonic = 0.5 * transverse(crystal)             # [] Polarization- <- (Field+, Field+)


def growth(source: Polarization, bond_orientation: np.ndarray, mismatch: np.ndarray,
           depths: np.ndarray) -> Polarization:
    """The doubled light after each slice: the sources of the slices up to it, each signed by its bonds'
    orientation and split into the cosine and sine amplitudes of its phase slip; the final axes are slices and
    quadratures."""
    thickness = depths[1] - depths[0]
    phase_slip = mismatch * depths
    quadratures = stack((source[..., None] * np.cos(phase_slip), source[..., None] * np.sin(phase_slip)), axis=-1)
    return (quadratures * bond_orientation[..., None] * thickness).cumsum(axis=-2)
