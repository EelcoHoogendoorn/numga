"""Radar ranging reconstructs a moving reflector's trajectory from one clock's echo times.

Projective spacetime algebra `x+t-w0` describes one space dimension and one time dimension.
The signs of x and t supply the Minkowski metric; w supplies the homogeneous coordinate,
so boosts and translations act on events and line equations by the same motor sandwich.
Events are intrinsic antivectors, and worldlines are extrinsic line equations. Joining an
event to a light direction and intersecting a worldline gives a linear map on homogeneous
events; composing the two legs maps emissions to receptions.
"""

from __future__ import annotations

from numga import Algebra, NumpyContext

ga = Algebra("x+t-w0")
mv = NumpyContext(ga).multivector
# Events are intrinsic homogeneous points: a nonzero common scale leaves their location unchanged.
Event = ga.gatype.antivector()                             # Antivector-
# An extrinsic line equation pairs to zero with events on a worldline or simultaneous slice.
Slice = ga.gatype.vector()                                 # Vector+
Scalar = ga.gatype.scalar()                                # Scalar+
EventMap = ga.gatype((Event, Event))                       # Event- <- Event-

# The signed blade complement gives an origin with homogeneous weight mv.w & origin == 1.
origin = mv.w.dual()                                       # [] Event-
# Weight-zero events are ideal points: spatial and temporal directions.
across = mv.x.dual()                                       # [] Event-
time = -mv.t.dual()                                        # [] Event-
# Joining these directions to the origin gives light rays whose line squares are zero.
# Every ideal point itself squares to zero, including across and time.
outgoing = time + across                                   # [] Event-
incoming = time - across                                   # [] Event-


def echo_maps(receiver: Slice, reflector: Slice, outward: Event, inward: Event,
              reference: Event) -> tuple[EventMap, EventMap]:
    """The reflection and echo maps for straight timelike worldlines and fixed light directions."""
    # Join an event to the outgoing ideal direction, then intersect the reflector's line equation.
    reflection = (Event & outward) ^ reflector
    returning = (Event & inward) ^ receiver
    # Fixed directions and straight worldlines make output weight a constant times input weight.
    # A unit-weight reference fixes that one scale for the whole map, preserving its linearity.
    reflection = reflection / (mv.w & reflection(reference))
    returning = returning / (mv.w & returning(reference))
    echo = returning(reflection)
    return reflection, echo


def reconstruct(emitted: Event, received: Event, clock: Slice, reference: Event,
                tangent: Event, spatial: Event) -> tuple[Event, Event, Scalar, Scalar]:
    """Locate reflections using an inertial observer's clock and the outgoing spatial direction."""
    radar_time = (clock & (received + emitted)) / 2
    radar_range = (clock & (received - emitted)) / 2
    simultaneous = reference + tangent * radar_time
    located = simultaneous + spatial * radar_range
    return located, simultaneous, radar_time, radar_range


def infer_motion(emitted: Event, received: Event, clock: Slice) -> tuple[Scalar, Scalar, Scalar]:
    """Infer echo stretch, relative velocity, and range at clock zero from two echoes."""
    # An inertial reflector makes the arrival time an affine function of emission time.
    stretch = (clock & (received[-1] - received[0])) / (clock & (emitted[-1] - emitted[0]))
    # Radar range changes with the difference of the two intervals; radar time with their sum.
    velocity = (stretch - 1) / (stretch + 1)
    first_time = (clock & (received[0] + emitted[0])) / 2
    first_range = (clock & (received[0] - emitted[0])) / 2
    initial_range = first_range - velocity * first_time
    return stretch, velocity, initial_range
