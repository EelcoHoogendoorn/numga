"""Light's polarization turned by an optically active medium and by a magnetic field, out and back.

Light travels along a unit direction, and its polarization is the direction of its electric field, a
vector across the path. An optically active medium, such as quartz or a sugar solution, turns the
polarization about the direction of travel, by its rotary power per distance. The turn takes its sense
from the direction of travel, so the rotary power is an intrinsic scalar: the mirror image of the
medium turns light the other way. A magnetic field turns the polarization too, the Faraday effect, in
the part of the field's bivector that lies across the path, by the Verdet constant per distance and
field. That part does not depend on which way the light goes.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, Orientation, stack
from numga.algebras import VGA3D

ga = VGA3D
mv = NumpyContext(ga).multivector
Direction = ga.gatype.vector()
Polarization = ga.gatype.vector()
Mirror = ga.gatype.vector()
Field = ga.gatype.bivector()
TurnRate = ga.gatype.bivector()
# A turn per distance with a sense of its own: measured against a handedness.
RotaryPower = ga.gatype(ga.subspace.scalar(), orientation=Orientation.intrinsic)


# --- math -----------------------------------------------------------------------------
def active_rate(direction: Direction, rotary_power: RotaryPower) -> TurnRate:
    """The turn per distance of an optically active medium: in the plane across the path, oriented by
    the direction of travel, `direction.dual()`, an intrinsic bivector. The rotary power's own sense
    makes the product extrinsic, a rate that exponentiates."""
    return direction.dual() * rotary_power                                        # [...] TurnRate


def faraday_rate(direction: Direction, field: Field, verdet: float) -> TurnRate:
    """The turn per distance in a magnetic field: the part of the field's bivector across the path,
    `(field ^ direction) * direction`, the same for either direction of travel."""
    return (field ^ direction) * direction * verdet                               # [...] TurnRate


def out_and_back(out_rate: TurnRate, back_rate: TurnRate, start: Polarization, depths: np.ndarray) -> Polarization:
    """The polarization at each depth on the way out, and on the way back after a reflection at the far
    end, `[legs, depths]`: the light leaves the medium where it entered."""
    out = (out_rate * (-depths / 2)).exp() >> start                               # [depths] Polarization
    back = (back_rate * (-depths / 2)).exp() >> out[-1]                           # [depths] Polarization
    return stack([out, back])                                                     # [legs, depths] Polarization


def through_media(direction: Direction, rotary_power: RotaryPower, field: Field, verdet: float,
                  start: Polarization, depths: np.ndarray) -> Polarization:
    """Light out and back through an optically active medium and through a magnetic field,
    `[media, legs, depths]`."""
    active = out_and_back(active_rate(direction, rotary_power), active_rate(-direction, rotary_power), start, depths)
    faraday = out_and_back(faraday_rate(direction, field, verdet), faraday_rate(-direction, field, verdet), start, depths)
    return stack([active, faraday])                                               # [media, legs, depths] Polarization
