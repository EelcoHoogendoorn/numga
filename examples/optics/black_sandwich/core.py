"""A half-wave plate between crossed polarizers forms a nonzero optical map whose square vanishes.

Spacetime geometric algebra models the electric field of light as an extrinsic bivector and the axes of polarizers and wave plates as intrinsic antivectors. Joining an axis with the rest-frame time direction sweeps out an intrinsic spacetime plane, and pairing the field with `axis & time` reads its transmitted amplitude along that axis. Mapping that amplitude back into the dual field `field_along` defines a linear `Field+ <- Field+` extensor for every polarizer and wave plate, so two identical stacks compose by direct extensor evaluation.
"""

from __future__ import annotations

from itertools import accumulate

from numga import Algebra, NumpyContext, stack

# Spacetime `x+y+z+t-`, space positive and time negative.
ga = Algebra("x+y+z+t-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
# Electric fields are extrinsic bivectors; component axes are intrinsic antivectors.
Field = ga.gatype.bivector()                                                   # Bivector+
Direction = ga.gatype.antivector()                                             # Antivector-
OpticalMap = ga.gatype((Field, Field))                                         # Field+ <- Field+

# Unit timelike direction of the apparatus rest frame, pairing to +1 with the time 1-form; `-` compensates `t^2 = -1`.
time = -mv.t.dual()                                                            # [] Direction-
# Unit electric field polarized along a spatial direction, dual to the spacetime plane swept out in time.
field_along = (Direction & time).dual()                                        # [] Field+ <- Direction-
# Rank-one optical projection onto one axis weighted by the field's pairing with a second axis's worldsheet.
polarizer = field_along * ((Direction & time) & Field)                         # [] Field+ <- (Direction-, Direction-, Field+)


# --- math -----------------------------------------------------------------------------
def stack_stage_fields(entrance: OpticalMap, plate: OpticalMap, exit_polarizer: OpticalMap,
                       incident_field: Field) -> Field:
    """Propagating `incident_field` (`[...] Field+`) sequentially through two identical
    `(entrance, plate, exit_polarizer)` stacks collects the electric field before and after every
    component (`[..., 7] Field+`)."""
    components = (entrance, plate, exit_polarizer) * 2
    stage_fields = tuple(accumulate(components, lambda field, component: component(field), initial=incident_field))
    batch_shape = stage_fields[-1].shape
    return stack([field.broadcast_to(batch_shape) for field in stage_fields], axis=-1)  # [..., 7] Field+
