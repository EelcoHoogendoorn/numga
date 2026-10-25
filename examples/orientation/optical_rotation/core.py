"""Light's polarization turned by an optically active medium and by a magnetic field, out and back.

Light travels along a unit direction in spacetime, and its polarization is the direction of its
electric field, across the path; both are antivectors, intrinsic like all geometry. An optically active
medium, such as quartz or a sugar solution, turns the polarization about the direction of travel, by its
rotary power per distance, in the plane the light sweeps through time, `direction & time`. The turn takes its sense
from the direction of travel, so the rotary power is an intrinsic scalar: the mirror image of the
medium turns light the other way. A magnetic field turns the polarization too, the Faraday effect, by
the Verdet constant per distance and field, in proportion to the field's flux through the beam's
cross-section, `field & cross_section`, the patch the two directions across the path span; the flux and
the cross-section are the same whichever way the light goes.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, Orientation, stack
from numga.algebras import STA

ga = STA
mv = NumpyContext(ga).multivector
# Directions, of travel and of the electric field across the path, are antivectors, intrinsic.
Direction = ga.gatype.antivector()
Polarization = ga.gatype.antivector()
Mirror = ga.gatype.vector()
Field = ga.gatype.bivector()
TurnRate = ga.gatype.bivector()
# The beam's cross-section: the patch two directions across the path span, intrinsic.
CrossSection = Field.derive.flipped_orientation
# A turn per distance with a sense of its own: measured against a handedness.
RotaryPower = ga.gatype(ga.subspace.scalar(), orientation=Orientation.intrinsic)

# The direction of time of an observer at rest with the media, which pairs to one with `mv.t`; the dual
# carries the minus of t's square.
time = -mv.t.dual()                                                               # [] Antivector


# --- math -----------------------------------------------------------------------------
def active_rate(direction: Direction, rotary_power: RotaryPower) -> TurnRate:
    """The turn per distance of an optically active medium: in the plane the light sweeps through
    time, the join of its direction with time, an intrinsic bivector oriented by the travel, which as a
    turn lies across the path. The rotary power's own sense makes the product extrinsic, a rate that
    exponentiates."""
    return (direction & time) * rotary_power                                      # [...] TurnRate


def faraday_rate(cross_section: CrossSection, field: Field, verdet: float) -> TurnRate:
    """The turn per distance in a magnetic field: the field's flux through the beam's cross-section,
    `field & cross_section`, turning the polarization in the cross-section's plane. Neither depends on
    the direction of travel."""
    return cross_section.dual_inverse() * (field & cross_section) * verdet        # [...] TurnRate


def out_and_back(out_rate: TurnRate, back_rate: TurnRate, start: Polarization, depths: np.ndarray) -> Polarization:
    """The polarization at each depth on the way out, and on the way back after a reflection at the far
    end, `[legs, depths]`: the light leaves the medium where it entered."""
    out = (out_rate * (-depths / 2)).exp() >> start                               # [depths] Polarization
    back = (back_rate * (-depths / 2)).exp() >> out[-1]                           # [depths] Polarization
    return stack([out, back])                                                     # [legs, depths] Polarization


def through_media(direction: Direction, cross_section: CrossSection, rotary_power: RotaryPower, field: Field,
                  verdet: float, start: Polarization, depths: np.ndarray) -> Polarization:
    """Light out and back through an optically active medium and through a magnetic field,
    `[media, legs, depths]`: the medium turns it by the direction it travels in, the field by the
    cross-section it crosses, the same out and back."""
    active = out_and_back(active_rate(direction, rotary_power), active_rate(-direction, rotary_power), start, depths)
    turn = faraday_rate(cross_section, field, verdet)                             # [] TurnRate
    faraday = out_and_back(turn, turn, start, depths)
    return stack([active, faraday])                                               # [media, legs, depths] Polarization
