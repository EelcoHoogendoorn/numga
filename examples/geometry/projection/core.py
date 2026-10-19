r"""Projective cameras, shadow casting, and two-view epipolar geometry all evaluate the same join-and-meet expression with different slots left open.

The pinhole camera is an expression with an open slot, `(centre & Point) ^ screen`, joining a world point with the optical centre into a sight ray and meeting that ray with the image plane. Leaving the point open casts shadows from a point light or a directional sun; leaving the light open sweeps the shadow trail of a fixed corner; and moving the camera map by two pose motors relates their images through the epipoles and epipolar lines.

In 3D Projective Geometric Algebra (`PGA3D`, signature `x+y+z+w0`), planes are vectors, lines are bivectors, points are antivectors (trivectors), and rigid motions are even-grade motors.

In the matrix notation of projective vision the camera reads as a $3 \times 4$ projection matrix, the spotlight and sunlight maps as $4 \times 4$ shadow projection matrices, and the two-view correspondence form as the fundamental matrix.
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


def circle(n: int) -> Point:
    """Construct n points around the unit circle in the xy plane, centred on the origin."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return (mv.xy * (-t / 2)).exp() >> mv.point([1, 0, 0])


def cube(size: float) -> Point:
    """Construct the eight corner points of an axis-aligned cube centred on the origin."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    return mv.point(corners * (size / 2.0))


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

    # Binding a different slot asks a different question. Fix one corner and leave
    # the light open: the corner's shadow is a linear map of the light position, so a
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
