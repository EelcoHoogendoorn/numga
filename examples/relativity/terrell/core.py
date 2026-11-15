"""Spacetime light-cone geometry models Terrell rotation of relativistic passing bodies.

Light reaching a camera at one instant leaves farther parts of a moving body earlier than nearer parts. Transforming the camera's past light cone into the body's rest frame reduces each relativistic photograph to a stationary perspective view rotated backward by aberration while preserving circular sphere outlines on the sky.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Antivector = ga.gatype.antivector()                 # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Rotor = ga.gatype.rotor()                           # Rotor+

time = -mv.t.dual()                                 # [] Antivector-
space = stack([mv.x, mv.y, mv.z]).dual()            # [3] Antivector-
travel, line_of_sight, up = space


def boost_rotors(speeds: np.ndarray) -> Rotor:
    """Hyperbolic turns in the travel-time plane map the body's rest frame to moving frames (`[...] Rotor+`)."""
    return ((travel & time).dual() * (np.arctanh(speeds) / 2)).exp()            # [...] Rotor+


def sighting_events(distance: float, angles: np.ndarray) -> Antivector:
    """Past-light-cone rays back from the camera reach the track at `distance` along each sighting angle (`[...] Antivector-`)."""
    turn = (mv.xy * (-angles / 2)).exp()                                        # [...] Rotor+
    sight = turn >> travel                                                      # [...] Antivector-
    return (sight - time) * (distance / (mv.y & sight))                         # [...] Antivector-


def rest_camera(frame: Rotor, centre: Antivector) -> Antivector:
    """Boosting the emission event `centre` into the body's frame gives the camera's spatial offset (`[...] Antivector-`)."""
    camera = -(frame << centre)                                                 # [...] Antivector-
    return camera - (mv.t & camera) * time                                      # [...] Antivector-


def camera_positions(frame: Rotor, distance: float, angles: np.ndarray) -> Iterator[Antivector]:
    """Yields the camera's spatial position in each body frame (`[...] Antivector-`) across the sequence of sighting `angles`."""
    for angle in angles:
        yield rest_camera(frame, sighting_events(distance, angle))


def rest_rays(camera: Antivector, points: Antivector) -> Antivector:
    """Null rays in the body's rest frame run straight from `points` to `camera` (`[...] Antivector-`)."""
    offsets = camera - points                                                   # [...] Antivector-
    return offsets + time * offsets.antinorm()                                  # [...] Antivector-


def grazing_outline(camera: Antivector, centre: Antivector, radius: float, angles: np.ndarray) -> Antivector:
    """Rays from `camera` graze a sphere around `centre` along a spatial circle (`[...] Antivector-`)."""
    offset = camera - centre                                                    # [...] Antivector-
    distance = offset.antinorm()                                                # [...] Scalar+
    # Build the grazing ring around `-line_of_sight` and rotate it onto the direction from the sphere to the camera.
    alignment = ((offset / distance) * (-line_of_sight).inverse()).square_root()  # [...] Rotor+
    nearest = (radius / distance) * -line_of_sight + (1 - (radius / distance).squared()).square_root() * up
    ring_rotors = (mv.zx * (angles / 2)).exp()                                  # [...] Rotor+
    ring = ring_rotors >> nearest[..., None]                                    # [...] Antivector-
    return centre + radius * (alignment[..., None] >> ring)                     # [...] Antivector-
