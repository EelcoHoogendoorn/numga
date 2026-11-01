"""Gravitational waves of the round three-sphere seen on the sphere below its Hopf circles.

The round three-sphere is a bundle of great circles, its Hopf circles, each as far from every other all along it. Read
in the isoclinic frame at any of its points, a circle's direction is one and the same unit step of the origin: a point
of the sphere of such steps, the base, below the whole circle. A wave that is the same all along the circles is seen on
the base without loss. Read along a circle, its strain is how much the circle stretches; read across it, turned a
quarter about the base point, it is the strain of the base itself, which a small triangle of atoms there feels. The
three-sphere's slowest wave, its squash, is such a wave.
"""

from __future__ import annotations

import numpy as np

from examples.relativity.motor_connection import scenarios as rolling

three_sphere = rolling.three_sphere
Point, Step, Rate, Motor, Scalar = (three_sphere.Point, three_sphere.Step, three_sphere.Rate, three_sphere.Motor,
                                    three_sphere.Scalar)


# --- math -----------------------------------------------------------------------------
def base_points(places: Point, circling: Rate) -> Point:
    """The base point below each place, `Point-`: the direction its circle runs in under the rate `circling`, read in
    the isoclinic frame at the place, a unit step of the origin, the same all along the circle."""
    along = places.commutator(circling)                                        # Point-
    return (three_sphere.isoclinic(places).inverse() >> along).cast(Step).normalized()


def lifts(bases: Point, circling: Rate) -> Point:
    """A point on the circle above each base point, `Point-`: the origin moved square to its own base point and the
    base point, by half the angle between them, which the Hopf map doubles."""
    below_origin = base_points(three_sphere.origin, circling)                  # [] Point-
    # The plane the two base points turn in, times the pseudoscalar: the rate that moves the origin square to both.
    moving = ((three_sphere.polar(bases) ^ three_sphere.polar(below_origin)) * three_sphere.mv.xyzw).normalized()
    angles = (three_sphere.polar(bases) & below_origin).arccos()               # Scalar+
    return (moving * (-angles / 4)).exp() >> three_sphere.origin


def round_circles(bases: Point, turn: Motor, n_turns: int, n_samples: int) -> Point:
    """So many points evenly round the circle above each base point, `[samples, bases] Point-`: the lifts carried by
    powers of the turn along the circles, `n_turns` of which close them."""
    circling = turn.log()                                                       # [] Rate+
    shares = np.arange(n_samples) * n_turns / n_samples                         # [samples]
    return (circling * shares[:, None]).exp() >> lifts(bases, circling)[None]


def stretches(waves: three_sphere.Waves, mesh: three_sphere.Mesh, vertices: Point, bases: Point, directions: Point,
              turn: Motor, n_turns: int, n_samples: int, time: float) -> Scalar:
    """How much lengths stretch along steps of the origin `directions` in the frames, at the circles above the base
    points, averaged round each, at a time, `Scalar+[bases]`: half the strain's reading of the step on itself."""
    places = round_circles(bases, turn, n_turns, n_samples)                    # [samples, bases] Point-
    faces = mesh.picking(vertices)(places.reshape(-1))                         # [samples * bases]
    strain, _, _, _ = waves.fields(faces, places.reshape(-1), time)            # [samples * bases] Vector+ <- Antivector-
    along = directions[None].broadcast_to(places.shape).reshape(-1)            # [samples * bases] Antivector-
    readings = (strain(along) & along) / (2 * along.dot(along))                 # [samples * bases] Scalar+
    return readings.reshape(places.shape).sum(axis=0) / n_samples


def across(bases: Point, directions: Point) -> Point:
    """Steps along the base, `directions` at the base points, turned a quarter about them, `Point-`: the steps across
    the circles that a base direction is the image of."""
    return three_sphere.turning(bases).commutator(directions).cast(Step)
