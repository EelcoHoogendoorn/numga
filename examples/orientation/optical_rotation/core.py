"""Round-trip polarization rotation separates natural optical activity in a chiral medium from the Faraday effect in a magnetic field.

Spacetime geometric algebra models the propagation direction and polarizer axis as intrinsic antivectors, while electric and magnetic fields are extrinsic bivectors. Pairing the electric field with the spacetime plane swept out by a polarizer axis, `field & (axis & time)`, reads the transmitted amplitude. An optically active medium turns the polarization in the spacetime plane `direction & time` swept out by the propagation direction, scaled by an intrinsic rotary power whose sense reverses under reflection, so the return trip unwinds the outgoing turn. A magnetic field turns the polarization in proportion to the magnetic flux `magnetic_field & cross_section` through the beam cross-section, accumulating the same turn on both legs.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA

ga = STA
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                       # Scalar+
# Propagation directions and polarizer axes are intrinsic spacetime antivectors.
Direction = ga.gatype.antivector()                                                # Antivector-
# Mirror planes are extrinsic vectors.
Plane = ga.gatype.vector()                                                        # Vector+
# Electric fields, magnetic fields, and spatial turn rates are extrinsic bivectors.
Field = ga.gatype.bivector()                                                      # Bivector+
TurnRate = ga.gatype.bivector()                                                   # Bivector+
# The beam cross-section is the intrinsic patch spanned by two transverse directions.
CrossSection = ga.gatype.antibivector()                                           # Antibivector-
# Chiral rotary power per unit path length is an intrinsic scalar.
RotaryPower = ga.gatype.antipseudoscalar()                                        # Antipseudoscalar-

# Unit timelike direction of the medium rest frame, pairing to +1 with the time 1-form; `-` compensates `t^2 = -1`.
time = -mv.t.dual()                                                               # [] Direction-
# Unit electric field polarized along a spatial direction, dual to the spacetime plane swept out in time.
polarization_field = (Direction & time).dual()                                    # [] Field+ <- Direction-


# --- math -----------------------------------------------------------------------------
def round_trip_fields(outgoing_rate: TurnRate, returning_rate: TurnRate, incident_field: Field,
                      depths: np.ndarray) -> Field:
    """Propagating `incident_field` over `depths` (`[n_depths]`) at `outgoing_rate` to a far-end reflector
    and back at `returning_rate` traces the outgoing and returning electric fields (`[2, n_depths] Field+`)."""
    outgoing_rotors = (outgoing_rate * (-depths / 2)).exp()                       # [n_depths] Rotor+
    outgoing = outgoing_rotors >> incident_field                                  # [n_depths] Field+
    returning_rotors = (returning_rate * (-depths / 2)).exp()                     # [n_depths] Rotor+
    returning = returning_rotors >> outgoing[-1]                                  # [n_depths] Field+
    return stack([outgoing, returning])                                           # [2, n_depths] Field+


def media_round_trips(direction: Direction, cross_section: CrossSection, rotary_power: RotaryPower,
                      magnetic_field: Field, verdet_constant: float, incident_field: Field,
                      depths: np.ndarray) -> Field:
    """Propagating `incident_field` out and back across `depths` (`[n_depths]`) through a chiral medium
    and a Faraday rotator yields both media's outgoing and returning fields (`[2, 2, n_depths] Field+`)."""
    # Multiplying the intrinsic travel-time plane by the intrinsic rotary power yields an extrinsic spatial turn rate.
    outgoing_sweep, returning_sweep = direction & time, -direction & time         # [] Antibivector-, Antibivector-
    chiral_legs = round_trip_fields(outgoing_sweep * rotary_power, returning_sweep * rotary_power,
                                    incident_field, depths)                       # [2, n_depths] Field+
    # Magnetic flux through the transverse cross-section drives the same spatial turn rate on both legs.
    faraday_rate = cross_section.dual_inverse() * (magnetic_field & cross_section) * verdet_constant  # [] TurnRate+
    faraday_legs = round_trip_fields(faraday_rate, faraday_rate, incident_field, depths)  # [2, n_depths] Field+
    return stack([chiral_legs, faraday_legs])                                     # [2, 2, n_depths] Field+
