"""Terrell rotation: the photograph of a body passing a camera at nearly the speed of light, in the
spacetime algebra.

A body moving at nearly the speed of light is contracted along its motion, yet no photograph shows the
contraction. A photograph records the light that reaches the camera at one instant, and light from the
far parts of a passing body set out earlier than light from its near parts, when the body was further
back. The delay and the contraction together turn the picture instead of squashing it: a passing cube
is seen turned, showing the face that trails behind it, and a sphere still shows a circular outline.
Terrell and Penrose pointed this out in 1959, more than half a century after the contraction itself.

Events, and the rays of light between them, are modelled as antivectors, directions in spacetime; a
ray is null, `ray.squared() == 0`, and the directions of the camera's sky are the rays arriving at it.
In the body's own frame nothing moves: there the camera, at `frame << event`, takes an ordinary
photograph of a body at rest, each ray running straight from a point of the body to the camera, and
the boost `frame >> rays` carries the photograph to the camera's frame. A boost is linear, so rays in
one hyperplane through the camera, where `plane & ray == 0`, stay in one, and those rays are a circle
on the sky: the boost maps circles on the sky to circles. The outline of a sphere is such a circle in
its own frame, so it is one in every frame, and the whole photograph is the one taken at rest from
where the camera is in the body's frame: turned, not squashed.

In the vector notation of special relativity, a line of sight at angle $\\theta$ from the body's
motion in the camera's frame is at $\\theta'$ in the body's, with
$\\cos\\theta' = \\frac{\\cos\\theta + \\beta}{1 + \\beta\\cos\\theta}$: seen at right angles to its
motion, the body is photographed from $\\cos\\theta' = \\beta$, turned by $\\arcsin\\beta$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Events, and the rays of light between them, are antivectors: directions in spacetime, intrinsic.
Antivector = ga.gatype.antivector()
Rotor = ga.gatype.rotor()

# One unit along the camera's time, and unit directions of space, which square to minus one.
time = -mv.t.dual()                                          # [] Antivector
space = stack([mv.x, mv.y, mv.z]).dual()                     # [3] Antivector
# The direction the body travels, the camera's line of sight to its track, and up.
travel, line_of_sight, up = space


# --- math -----------------------------------------------------------------------------
def boost(speeds: np.ndarray) -> Rotor:
    """The body's frame at each speed `[...]`, a fraction of the speed of light along its motion: a turn
    in the plane of the motion and time."""
    return ((travel & time).dual() * (np.arctanh(speeds) / 2)).exp()            # [...] Rotor


def sighting(distance: float, angles: np.ndarray) -> Antivector:
    """The events where the body's centre sends the light the camera, at the origin, receives along a line of sight
    at each angle from the body's motion `[...]`, on a track at the given distance: on the camera's past
    light cone, the ray back along the line of sight, scaled to reach the track."""
    sight = (mv.xy * (angles / 2)).exp() >> travel                              # [...] Antivector
    return (sight - time) * (distance / (mv.y & sight))                         # [...] Antivector


def rest_camera(frame: Rotor, centre: Antivector) -> Antivector:
    """Where the camera is in the body's frame when it receives the light the body's centre sent from
    the event `centre`: the camera's event relative to it, seen from the body, without its time."""
    camera = -(frame << centre)                                                  # [...] Antivector
    return camera - (mv.t & camera) * time                                      # [...] Antivector


def rays(camera: Antivector, points: Antivector) -> Antivector:
    """The rays from points of a body at rest to the camera at rest beside it, in their frame: straight
    lines in space, taking as long as they are long."""
    offsets = camera - points                                                   # [...] Antivector
    # Directions of space square to minus their length squared.
    return offsets + time * (-offsets.squared()).square_root()                  # [...] Antivector


def facing(camera: Antivector, points: Antivector, normals: Antivector) -> Scalar:
    """How far the camera is out along the outward normal at each point of a body: the point is seen
    where this is positive."""
    return normals | (points - camera)                                          # [...] Scalar


def outline(camera: Antivector, centre: Antivector, radius: float, angles: np.ndarray) -> Antivector:
    """The outline of a sphere as the camera sees it, at each angle around it `[..., angles]`: where the
    rays from the camera graze the sphere, a circle on the sphere square to the camera's direction."""
    offset = camera - centre                                                    # [...] Antivector
    distance = (-offset.squared()).square_root()                                # [...] Scalar
    # The grazing points ring the line from the sphere's centre to the camera. The ring is built around the
    # direction to a camera straight in front, and turned onto the camera's, which is never straight behind.
    turn = (-((offset / distance) * -line_of_sight)).square_root()              # [...] Rotor
    nearest = (radius / distance) * -line_of_sight + (1 - (radius / distance).squared()).square_root() * up
    circle = (mv.zx * (angles / 2)).exp() >> nearest[..., None]                # [..., angles] Antivector
    return centre + radius * (turn[..., None] >> circle)                        # [..., angles] Antivector
