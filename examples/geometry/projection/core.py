"""Projective cameras, shadow casting, and two-view epipolar geometry all evaluate the same join-and-meet expression as maps on different arguments.

The pinhole camera is a map on points, `(centre & Point) ^ screen`, joining a world point with the optical centre into a sight ray and meeting that ray with the image plane. As a map on a body's points it casts shadows from a point light or a directional sun; as a map on the light, `(Point & corner) ^ ground`, it sweeps the shadow trail of a fixed corner; and moved by two pose motors, the camera relates their images through the epipoles and epipolar lines.
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
Correspondence = ga.gatype((Pseudoscalar, Point, Point))  # Pseudoscalar+ <- (Point-, Point-)


# --- math -----------------------------------------------------------------------------
def circle(n_points: int) -> Point:
    """Evenly spaced rotor turns in the xy plane sweep `[n_points] Point-` along the unit circle about the origin."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    turns = (mv.xy * (-angles / 2)).exp()                                       # [n_points] Motor+
    return turns >> mv.point([1, 0, 0])


def cube(side: float) -> Point:
    """An axis-aligned cube of edge length `side` centred on the origin has eight corners (`[8] Point-`)."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])  # [8, 3]
    return mv.point(corners * (side / 2.0))


def shadows(
    body: Point, ground: Plane, point_light: Point, sun: Point,
    corner: Point, light_path: Point,
) -> tuple[Point, Point, Point]:
    """Joining points with a light and meeting the ground casts `body`'s spotlight and sunlight shadows (`[...] Point-`) and `corner`'s shadow trail along `light_path`."""
    # Join the point with the light into a ray, then meet the ray with the ground; a finite point light
    # and an ideal direction at infinity evaluate the same linear map Point- <- Point-.
    spotlight = (point_light & Point) ^ ground                                  # [] Point- <- Point-
    sunlight = (sun & Point) ^ ground                                           # [] Point- <- Point-

    # Fixing the corner instead makes its shadow a linear map of the light's position, casting an entire
    # path of light positions in one call.
    shadow_of_corner = (Point & corner) ^ ground                                # [] Point- <- Point-

    return spotlight(body), sunlight(body), shadow_of_corner(light_path)


def stereo(
    subject: Point, centre: Point, screen: Plane,
    rig_1: Motor, rig_2: Motor,
) -> tuple[Point, Point, Point, Line, Correspondence]:
    """Two camera poses `rig_1` and `rig_2` image `subject` (`[...] Point-`) and determine the second view's epipole, epipolar lines (`[...] Line-`), and two-view correspondence form."""
    # Joining with the optical centre and meeting the screen gives both the point camera and the line camera.
    camera = (centre & Point) ^ screen                                          # [] Point- <- Point-
    line_camera = (centre & Line) ^ screen                                      # [] Line- <- Line-

    # Conjugating each camera map by its pose motor pulls world elements into the rig frame and pushes
    # screen images out, keeping the image of a join equal to the join of the images.
    camera_1 = rig_1 >> camera(rig_1 << Point)                                  # [] Point- <- Point-
    camera_2 = rig_2 >> camera(rig_2 << Point)                                  # [] Point- <- Point-
    line_camera_2 = rig_2 >> line_camera(rig_2 << Line)                         # [] Line- <- Line-
    centre_1 = rig_1 >> centre                                                  # [] Point-
    centre_2 = rig_2 >> centre                                                  # [] Point-

    image_1 = camera_1(subject)                                                 # [...] Point-
    image_2 = camera_2(subject)                                                 # [...] Point-

    # The epipole images the other optical centre, and the line camera maps each first-view sight ray
    # directly onto its epipolar line on the second screen.
    epipole_2 = camera_2(centre_1)                                              # [] Point-
    epipolar_lines_2 = line_camera_2(centre_1 & image_1)                        # [...] Line-

    # Two image points correspond when their sight rays meet, which makes the wedge of those rays vanish
    # as a bilinear form on both image points.
    correspondence = (centre_1 & Point) ^ (centre_2 & Point)                    # [] Pseudoscalar+ <- (Point-, Point-)

    return image_1, image_2, epipole_2, epipolar_lines_2, correspondence
