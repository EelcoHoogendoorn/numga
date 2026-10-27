r"""Projective cameras, shadow casting, and two-view epipolar geometry all evaluate the same join-and-meet expression, as maps on different arguments.

The pinhole camera is a map on points, `(centre & Point) ^ screen`, joining a world point with the optical centre into a sight ray and meeting that ray with the image plane. As a map on a body's points it casts shadows from a point light or a directional sun; as a map on the light, `(Point & corner) ^ ground`, it sweeps the shadow trail of a fixed corner; and moved by two pose motors, the camera relates their images through the epipoles and epipolar lines.

In the matrix notation of projective vision the camera reads as a $3 \times 4$ projection matrix, the spotlight and sunlight maps as $4 \times 4$ shadow projection matrices, and the two-view correspondence form as the fundamental matrix.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                  # Antivector-
Line = ga.gatype.antibivector()                 # Antibivector-
Plane = ga.gatype.vector()                      # Vector+
Motor = ga.gatype.rotor()                       # Rotor+
Pseudoscalar = ga.gatype.pseudoscalar()         # Pseudoscalar+
Correspondence = ga.gatype((Pseudoscalar, Point, Point))


# --- math -----------------------------------------------------------------------------
def shadows(
    body: Point, ground: Plane, point_light: Point, sun: Point,
    corner: Point, light_path: Point,
) -> tuple[Point, Point, Point]:
    """A body's shadows under a point light and the sun, and the shadow trail of one corner along a light path."""
    # Join the point with the light into a ray, then meet the ray with the ground. As a map on
    # the point this is linear, Point <- Point. Nothing in the expression cares whether the
    # light is a finite point or a direction at infinity, so the sun's orthographic shadow is
    # the same line of code.
    spotlight = (point_light & Point) ^ ground                                  # [] Point- <- Point-
    sunlight = (sun & Point) ^ ground                                           # [] Point- <- Point-

    # The same expression as a map on the light asks a different question: with the corner
    # fixed, its shadow is a linear map of the light's position, so a whole path of light
    # positions is cast in one call.
    shadow_of_corner = (Point & corner) ^ ground                                # [] Point- <- Point-

    return spotlight(body), sunlight(body), shadow_of_corner(light_path)


def stereo(
    subject: Point, centre: Point, screen: Plane,
    rig_1: Motor, rig_2: Motor,
) -> tuple[Point, Point, Point, Line, Correspondence]:
    """The two views of a subject from a camera posed twice, with the second view's epipole, epipolar lines, and two-view correspondence form."""
    # The same join then meet, as a map on lines, is the camera for lines.
    camera = (centre & Point) ^ screen                                          # [] Point- <- Point-
    line_camera = (centre & Line) ^ screen                                      # [] Line- <- Line-

    # The camera moves as a map: pull world points back into the rig frame, push image
    # points forward. The line camera moves the same way and stays consistent with the
    # point camera: the image of a join is the join of the images.
    camera_1 = rig_1 >> camera(rig_1 << Point)                                  # [] Point- <- Point-
    camera_2 = rig_2 >> camera(rig_2 << Point)                                  # [] Point- <- Point-
    line_camera_2 = rig_2 >> line_camera(rig_2 << Line)                         # [] Line- <- Line-
    centre_1 = rig_1 >> centre                                                  # [] Point-
    centre_2 = rig_2 >> centre                                                  # [] Point-

    image_1 = camera_1(subject)                                                 # [corners] Point-
    image_2 = camera_2(subject)                                                 # [corners] Point-

    # The epipole is the image of the other centre. The epipolar line of an image point is
    # the image of its ray, which the line camera gives directly.
    epipole_2 = camera_2(centre_1)                                              # [] Point-
    epipolar_lines_2 = line_camera_2(centre_1 & image_1)                        # [corners] Line-

    # Two image points correspond iff their rays meet, and two lines meet iff their wedge
    # vanishes. As a map on both image points, that sentence is a bilinear form.
    correspondence = (centre_1 & Point) ^ (centre_2 & Point)                    # [] Pseudoscalar+ <- (Point-, Point-)

    return image_1, image_2, epipole_2, epipolar_lines_2, correspondence


# --- plumbing -------------------------------------------------------------------------
# Which corner pairs of `cube` are joined by an edge.
CUBE_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def circle(n_points: int) -> Point:
    """The points of the unit circle in the xy plane, spaced evenly about the origin."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    return (mv.xy * (-angles / 2)).exp() >> mv.point([1, 0, 0])


def cube(size: float) -> Point:
    """The eight corners of an axis-aligned cube of the given side, centred on the origin."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return mv.point(corners * (size / 2.0))
