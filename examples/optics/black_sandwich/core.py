"""A half-wave plate between crossed polarizers: a nonzero optical map that squares to zero.

The electric field of the light is a bivector of spacetime, extrinsic, and the axes of the polarizers
and plates are directions, antivectors. Each axis sweeps a plane through the observer's time, an
antibivector, and a field's component along the axis is its pairing with that plane,
`field & (axis & time)`. The field along a direction is the dual of that plane,
`field_along = (Direction & time).dual()`. Every component maps fields to fields.
"""

from __future__ import annotations

from itertools import accumulate

from numga import Algebra, NumpyContext, stack

# Spacetime, space positive and time negative.
ga = Algebra("x+y+z+t-")
mv = NumpyContext(ga).multivector
# The electric field is a bivector, extrinsic; the components' axes are directions, antivectors.
Field = ga.gatype.bivector()                                           # Bivector+
Direction = ga.gatype.antivector()                                     # Antivector-
OpticalMap = ga.gatype((Field, Field))                                 # Field+ <- Field+

# The direction in time of an observer at rest with the apparatus, which pairs to one with `mv.t`; the
# dual carries the minus of t's square.
time = -mv.t.dual()                                                    # [] Direction-
# The field along a direction, the dual of the plane the direction sweeps in time: the vacuum's
# identification of a direction with the field along it, pairing to one with that plane for a unit direction.
field_along = (Direction & time).dual()                                # [] Field+ <- Direction-
# The field along one direction, weighted by the field's pairing with the plane another sweeps in time;
# bound to a unit axis twice, the polarizer that transmits the field's component along the axis.
polarizer = field_along * ((Direction & time) & Field)                 # [] Field+ <- (Direction-, Direction-, Field+)


# --- math
def through_pair(entrance: OpticalMap, plate: OpticalMap, exit: OpticalMap, incident: Field) -> Field:
    """The field before and after each component of two identical stacks."""
    components = (entrance, plate, exit) * 2
    fields = accumulate(components, lambda field, component: component(field), initial=incident)
    return stack(tuple(fields), axis=-1)                               # [..., stages] Field+
