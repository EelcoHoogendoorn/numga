"""Radar ranging reconstructs a moving reflector's trajectory from one clock's echo times.

Projective spacetime models one spatial dimension and one temporal dimension together with a degenerate weight axis, so boosts and spacetime translations act on events, worldlines, and clocks by the same motor sandwich. Events are intrinsic homogeneous points (`Event-`), and straight worldlines or simultaneous slices are extrinsic planes (`Plane+`). Joining an event to an ideal light direction with `&` and intersecting a worldline with `^` forms a linear map (`Event- <- Event-`) on homogeneous events, and composing the outbound and return legs maps emissions directly to receptions.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("x+t-w0")
mv = NumpyContext(ga).multivector
# Events are intrinsic homogeneous points: a nonzero common scale leaves their location unchanged.
Event = ga.gatype.antivector()                             # Antivector-
# An extrinsic plane pairs to zero with events on a worldline or simultaneous slice.
Plane = ga.gatype.vector()                                 # Vector+
Scalar = ga.gatype.scalar()                                # Scalar+
EventMap = ga.gatype((Event, Event))                       # Event- <- Event-

# The signed blade complement gives an origin with homogeneous weight mv.w & origin == 1.
origin = mv.w.dual()                                       # Event-
# Weight-zero events are ideal points: spatial and temporal directions.
across = mv.x.dual()                                       # Event-
time = -mv.t.dual()                                        # Event-
# Joining these directions to the origin gives light rays whose line squares are zero.
# Every ideal point itself squares to zero, including across and time.
outgoing = time + across                                   # Event-
incoming = time - across                                   # Event-


# --- math -----------------------------------------------------------------------------
def echo_maps(receiver: Plane, reflector: Plane, outward: Event, inward: Event,
              reference: Event) -> tuple[EventMap, EventMap]:
    """The reflection and round-trip echo maps (`Event- <- Event-`) follow straight timelike worldlines along fixed null directions."""
    # Join an event to the outgoing ideal direction, then intersect the reflector's worldline.
    reflection = (Event & outward) ^ reflector             # [] Event- <- Event-
    returning = (Event & inward) ^ receiver                # [] Event- <- Event-
    # Fixed directions and straight worldlines make output weight a constant times input weight.
    # A unit-weight reference fixes that one scale for the whole map, preserving its linearity.
    reflection = reflection / (mv.w & reflection(reference))  # [] Event- <- Event-
    returning = returning / (mv.w & returning(reference))     # [] Event- <- Event-
    echo = returning(reflection)                              # [] Event- <- Event-
    return reflection, echo


def radar_assignment(emitted: Event, received: Event, clock: Plane, reference: Event,
                     tangent: Event, spatial: Event) -> tuple[Event, Event, Scalar, Scalar]:
    """Einstein's midpoint convention assigns each reflection its event, simultaneous clock event, radar time, and radar range (`[...] Event-, Event-, Scalar+, Scalar+`)."""
    radar_time = (clock & (received + emitted)) / 2        # [...] Scalar+
    radar_range = (clock & (received - emitted)) / 2       # [...] Scalar+
    simultaneous = reference + tangent * radar_time        # [...] Event-
    located = simultaneous + spatial * radar_range         # [...] Event-
    return located, simultaneous, radar_time, radar_range


def reflector_motion(emitted: Event, received: Event, clock: Plane) -> tuple[Scalar, Scalar, Scalar]:
    """Two or more echoes on an inertial clock determine the echo stretch, relative velocity, and range at clock zero (`[] Scalar+, Scalar+, Scalar+`)."""
    # An inertial reflector makes the arrival time an affine function of emission time.
    stretch = (clock & (received[-1] - received[0])) / (clock & (emitted[-1] - emitted[0]))  # [] Scalar+
    # Radar range changes with the difference of the two intervals; radar time with their sum.
    velocity = (stretch - 1) / (stretch + 1)               # [] Scalar+
    first_time = (clock & (received[0] + emitted[0])) / 2  # [] Scalar+
    first_range = (clock & (received[0] - emitted[0])) / 2  # [] Scalar+
    initial_range = first_range - velocity * first_time    # [] Scalar+
    return stretch, velocity, initial_range


def boosted_frames(worldlines: Event, emitted: Event, reflected: Event, received: Event,
                   final_rapidity: float, spatial_shift: float, time_shift: float,
                   n_frames: int) -> Iterator[tuple[Event, Event, Event, Event, Event, Event, Scalar, Scalar, Plane]]:
    """Yields the worldlines, pulse events, radar reconstruction, and clock plane across `n_frames` Poincare transformations."""
    midpoint_time = (mv.t & (emitted[0] + received[-1])) / 2              # [] Scalar+
    centre = (-mv.tw * (midpoint_time / 2)).exp()                         # [] Motor+
    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    fractions = (1 - np.cos(phases)) / 2                                  # [n_frames]
    for fraction in fractions:
        boost = (mv.xt * (final_rapidity * fraction / 2)).exp()           # [] Motor+
        # The translation generator squares to zero, so its exponential is one plus the generator.
        translation = ((mv.xw * spatial_shift - mv.tw * time_shift) * (fraction / 2)).exp()  # [] Motor+
        # Conjugating the boost by centre anchors it at the midpoint event.
        motor = translation * centre * boost * ~centre                    # [] Motor+
        # The clock's unit and zero reading move with the experiment.
        clock = motor >> mv.t                                             # [] Plane+
        sent, returned = motor >> emitted, motor >> received              # [...] Event-, Event-
        hit = motor >> reflected                                          # [...] Event-
        moved_worldlines = motor >> worldlines                            # [2, 2] Event-
        moved_origin = motor >> origin                                    # [] Event-
        moved_time, moved_across = motor >> time, motor >> across         # [] Event-, Event-
        located, simultaneous, radar_time, radar_range = radar_assignment(
            sent, returned, clock, moved_origin, moved_time, moved_across)
        yield moved_worldlines, sent, hit, returned, located, simultaneous, radar_time, radar_range, clock
