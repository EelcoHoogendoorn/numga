"""Light's polarization turned by an optically active medium and by a magnetic field, out and back.

Light travels along a unit direction in spacetime, an antivector, intrinsic like all geometry. Its electric
field is a bivector of spacetime, extrinsic: the field along its polarization, `field_along(polarization)`,
the dual of the plane the polarization sweeps through time. A polarizer reads the field by pairing it with
the plane its axis sweeps, `field & (axis & time)`. An optically active medium, such as quartz or a sugar
solution, turns the field across the path by its rotary power per distance, in the plane the light sweeps
through time, `direction & time`. The turn takes its sense from the direction of travel, so the rotary power
is an intrinsic scalar: the mirror image of the medium turns light the other way. A magnetic field, a plane
of space, turns the light too, the Faraday effect, by the Verdet constant per distance and field, in
proportion to the field's flux through the beam's cross-section, `magnetic_field & cross_section`, the patch
the two directions across the path span; the flux and the cross-section are the same whichever way the light
goes.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA

ga = STA
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                       # Scalar+
# Directions, of travel and of a polarizer's axis, are antivectors, intrinsic.
Direction = ga.gatype.antivector()                                                # Antivector-
# Electric and magnetic fields alike are bivectors, extrinsic.
Field = ga.gatype.bivector()                                                      # Bivector+
TurnRate = ga.gatype.bivector()                                                   # Bivector+
# The beam's cross-section: the patch two directions across the path span, intrinsic.
CrossSection = Field.derive.flipped_orientation                                   # Bivector-
# A turn per distance with a sense of its own: measured against a handedness.
RotaryPower = ga.gatype.antipseudoscalar()                                   # Antipseudoscalar-

# The direction of time of an observer at rest with the media, which pairs to one with `mv.t`; the dual
# carries the minus of t's square.
time = -mv.t.dual()                                                               # [] Direction-
# The field along a direction, the dual of the plane the direction sweeps in time: the vacuum's
# identification of a direction with the field along it, pairing to one with that plane for a unit direction.
field_along = (Direction & time).dual()                                           # [] Field+ <- Direction-


# --- math -----------------------------------------------------------------------------
def out_and_back(out_rate: TurnRate, back_rate: TurnRate, entering: Field, depths: np.ndarray) -> Field:
    """The light's field at each depth on the way out, and on the way back after a reflection at the far
    end, `[legs, depths]`: the light leaves the medium where it entered."""
    out = (out_rate * (-depths / 2)).exp() >> entering                            # [depths] Field+
    back = (back_rate * (-depths / 2)).exp() >> out[-1]                           # [depths] Field+
    return stack([out, back])                                                     # [legs, depths] Field+


def through_media(direction: Direction, cross_section: CrossSection, rotary_power: RotaryPower,
                  magnetic_field: Field, verdet: float, entering: Field, depths: np.ndarray) -> Field:
    """Light out and back through an optically active medium and through a magnetic field,
    `[media, legs, depths]`: the medium turns it by the direction it travels in, the field by the
    cross-section it crosses, the same out and back."""
    # The plane the light sweeps through time, an intrinsic bivector oriented by the travel, lies across the
    # path as a turn; the rotary power's own sense makes the product extrinsic, a rate that exponentiates.
    active = out_and_back((direction & time) * rotary_power, (-direction & time) * rotary_power, entering, depths)
    # The magnetic field's flux through the cross-section, turning the light in the cross-section's plane.
    faraday_rate = cross_section.dual_inverse() * (magnetic_field & cross_section) * verdet   # [] TurnRate+
    faraday = out_and_back(faraday_rate, faraday_rate, entering, depths)
    return stack([active, faraday])                                               # [media, legs, depths] Field+
