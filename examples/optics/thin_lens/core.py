"""Ideal thin lenses, prisms, and flat mirrors act as global linear maps on projective ray lines in the plane.

Each ray is modelled as an antibivector line joined from two points, and an ideal thin lens of focal length `focal` in the home plane `home` acts linearly on lines as `Line - (Line & origin) * (home / focal)`. Conjugating that map by an element's placement motor positions and tilts the element anywhere in the plane, and composing element maps yields a single linear operator for the entire optical train. Because rays are global geometric lines rather than local height-slope pairs on reference planes, free space between elements requires no propagation operator, and tilted or off-axis elements compose with zero coordinate re-referencing.
"""

from __future__ import annotations

from functools import reduce
from itertools import accumulate

from numga import NumpyContext, stack
from numga.algebras import PGA2D

ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
# Rays are joins of points, and element planes are vectors.
Line = ga.gatype.antibivector()                                                 # Antibivector-
Plane = ga.gatype.vector()                                                      # Vector+
Point = ga.gatype.antivector()                                                  # Antivector-
# Two lines meet in an extrinsic bivector crossing.
Crossing = ga.gatype.bivector()                                                 # Bivector+
LineMap = ga.gatype((Line, Line))                                               # Line- <- Line-
Motor = ga.gatype.rotor()                                                       # Rotor+

origin = mv.point([0, 0])                                                       # [] Point-
# Every optical element is defined in the home plane x == 0.
home = mv.x                                                                     # [] Plane+


# --- math -----------------------------------------------------------------------------
def thin_lens_map(plane: Plane, focal: float) -> LineMap:
    """Translating the home-plane shear `Line - (Line & origin) * (home / focal)` onto vertical `plane` forms the thin-lens ray map (`[] Line- <- Line-`)."""
    shear = Line - (Line & origin) * (home / focal)                             # [] Line- <- Line-
    shift = (mv.wx * (plane & origin) * 0.5).exp()                              # [] Rotor+
    return shift >> shear(shift << Line)


# Four optical elements defined in the home plane: converging lens, thin prism, diverging lens, and flat mirror.
train_elements = (
    Line - (Line & origin) * (home / 1.0),                                      # [] Line- <- Line-
    Line - (Line & mv.direction([0, 1])) * home * 0.15,                         # [] Line- <- Line-
    Line - (Line & origin) * (home / -2.0),                                     # [] Line- <- Line-
    home.normalized() >> Line,                                                  # [] Line- <- Line-
)


def optical_train_trace(
    rays: Line, motors: tuple[Motor, ...], elements: tuple[LineMap, ...],
) -> tuple[Plane, Line, LineMap]:
    """Conjugating `elements` (`n_elements` of `[] Line- <- Line-`) by `motors` traces `rays` (`[n_rays] Line-`) into element planes (`[n_elements] Plane+`), per-stage ray bundles (`[n_elements + 1, n_rays] Line-`), and the composed train map (`[] Line- <- Line-`)."""
    placed = [motor >> element(motor << Line) for motor, element in zip(motors, elements)]
    legs = stack(list(accumulate(placed, lambda bundle, element: element(bundle), initial=rays)))  # [..., ...] Line-
    composed = reduce(lambda so_far, element: element(so_far), placed, Line)    # [] Line- <- Line-
    return stack([motor >> home for motor in motors]), legs, composed
