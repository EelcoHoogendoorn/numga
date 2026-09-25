"""Projective cameras, shadows and epipolar geometry from supplied geometric data.

The camera is an expression with a hole, `(centre & Point) ^ screen`: join the point
with the centre into a ray, and meet the ray with the screen. Every question below is
the same expression with a different slot open.

This module is the mathematics: types, constructors, and the expressions with holes.
The scenarios choose the scene and the reference frame.

In the matrix notation of projective vision the camera reads as a four-by-four projection
matrix applied at the end of a pipeline, the spotlight and sunlight maps as shadow matrices,
and the correspondence form as the fundamental matrix.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Line = ga.gatype.antibivector()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Pseudoscalar = ga.gatype.pseudoscalar()
Camera = ga.gatype((Point, Point))
LineCamera = ga.gatype((Line, Line))
ShadowTrail = ga.gatype((Point, Point))
Correspondence = ga.gatype((Pseudoscalar, Point, Point))

# Which corner pairs of `cube` are joined by an edge.
CUBE_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def point(coords: np.ndarray) -> Point:
    """Finite points at (..., 3) coordinates: the dual of the homogeneous vector."""
    return (mv("x y z", coords) + mv.w).dual()


def direction(coords: np.ndarray) -> Point:
    """Ideal points: the directions (..., 3), the dual of a weightless vector."""
    return mv("x y z", coords).dual()


def circle(n: int) -> Point:
    """Construct n points around the unit circle in the xy plane, centred on the origin."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return point(np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1))


def cube(size: float) -> Point:
    """Construct the eight corner points of an axis-aligned cube centred on the origin."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    return point(corners * (size / 2.0))


# ---------------------------------------------------------------------------
# Shadows: leave the point open
# ---------------------------------------------------------------------------
def shadows(body: Point, ground: Plane, point_light: Point, sun: Point,
            corner: Point, light_path: Point):
    """Cast a body's shadows, then reopen the light slot to sweep one corner's shadow."""
    # Join the point with the light into a ray, then meet the ray with the ground. With the
    # point slot open this is a linear map, Point <- Point. Nothing in the expression cares
    # whether the light is a finite point or a direction at infinity, so the sun's
    # orthographic shadow is the same line of code.
    spotlight: Camera = (point_light & Point) ^ ground
    sunlight: Camera = (sun & Point) ^ ground

    # Bind a different slot and you ask a different question. Fix one corner and leave
    # the light open: the corner's shadow is now a linear map of the light position, so a
    # whole path of light positions binds in one call.
    shadow_of_corner: ShadowTrail = (Point & corner) ^ ground

    return spotlight(body), sunlight(body), shadow_of_corner(light_path)


# ---------------------------------------------------------------------------
# Cameras and epipolar geometry: one rig, moved as a map
# ---------------------------------------------------------------------------
def stereo(subject: Point, centre: Point, screen: Plane,
           rig_1: Motor, rig_2: Motor):
    """Move the supplied centre and screen into two poses and relate their images."""
    # The same join-then-meet with a line in the open slot is the camera for lines.
    camera: Camera = (centre & Point) ^ screen
    line_camera: LineCamera = (centre & Line) ^ screen

    # The camera moves as a map: pull world points back into the rig frame, push image
    # points forward. The line camera moves the same way and stays consistent with the
    # point camera: the image of a join is the join of the images.
    camera_1 = rig_1 >> camera(rig_1 << Point)
    camera_2 = rig_2 >> camera(rig_2 << Point)
    line_camera_2 = rig_2 >> line_camera(rig_2 << Line)
    centre_1 = rig_1 >> centre
    centre_2 = rig_2 >> centre

    image_1 = camera_1(subject)
    image_2 = camera_2(subject)

    # The epipole is the image of the other centre. The epipolar line of an image point is
    # the image of its ray, which the line camera gives directly.
    epipole_2 = camera_2(centre_1)
    epipolar_lines_2 = line_camera_2(centre_1 & image_1)

    # Two image points correspond iff their rays meet, and two lines meet iff their wedge
    # vanishes. Leave both point slots open and that sentence is a bilinear form on image
    # points.
    correspondence: Correspondence = (centre_1 & Point) ^ (centre_2 & Point)

    return image_1, image_2, epipole_2, epipolar_lines_2, correspondence
