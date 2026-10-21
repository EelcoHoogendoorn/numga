"""A half-wave plate between crossed polarizers: a nonzero optical map that squares to zero.

The electric field of the light is a bivector of spacetime, extrinsic, and the axes of the polarizers
and plates are directions, antivectors. Each axis sweeps a plane through the observer's time, an
antibivector, and a field's component along the axis is its pairing with that plane, `field & sweep`.
Every component maps fields to fields.
"""

from __future__ import annotations

from itertools import accumulate

from numga import Algebra, NumpyContext, stack

# Spacetime, space positive and time negative.
ga = Algebra("x+y+z+t-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# The electric field is a bivector, extrinsic; the components' axes are directions, antivectors.
Field = ga.gatype.bivector()
Direction = ga.gatype.antivector()
OpticalMap = ga.gatype((Field, Field))

# The direction in time of an observer at rest with the apparatus, which pairs to one with `mv.t`; the
# dual carries the minus of t's square.
time = -mv.t.dual()


# --- math
def field_along(axis: Direction) -> Field:
    """The unit field along an axis: the one that pairs to one with the plane the axis sweeps in time."""
    sweep = axis & time                                                    # [...] Antibivector
    field = sweep.dual_inverse()                                           # [...] Field
    return field / (field & sweep)


def polarizer(axis: Direction) -> OpticalMap:
    """Transmit the field's component along a unit axis."""
    return field_along(axis) * (Field & (axis & time))                     # [...] Field <- Field


def half_wave(fast_axis: Direction) -> OpticalMap:
    """Keep the fast component and reverse the perpendicular one after a half-period delay."""
    return 2 * polarizer(fast_axis) - Field                                # [...] Field <- Field


def through_pair(entrance: OpticalMap, plate: OpticalMap, exit: OpticalMap, incident: Field) -> Field:
    """The field before and after each component of two identical stacks."""
    components = (entrance, plate, exit) * 2
    fields = accumulate(components, lambda field, component: component(field), initial=incident)
    return stack(tuple(fields), axis=-1)                                   # [..., stages] Field
