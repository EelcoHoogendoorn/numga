"""Projective cameras, shadows and epipolar geometry from supplied geometric data.

The usual pipeline builds a 4x4 projection matrix and applies it at the very end.
Here the camera is an expression with a hole, (centre ∨ point) ∧ screen, and the
matrix is what you get by choosing which hole to leave open. Every question below is
the same expression with a different slot open.

This module is the mathematics alone. It constructs geometry and returns geometry;
the scenario chooses the algebra, backend and reference frame.
"""

from __future__ import annotations

from examples.geometry.projection.scenarios import (
    Camera, Correspondence, Line, LineCamera, Motor, Plane, Point, ShadowTrail,
)


# ---------------------------------------------------------------------------
# Shadows: leave the point open
# ---------------------------------------------------------------------------
def shadows(body: Point, ground: Plane, point_light: Point, sun: Point,
            corner: Point, light_path: Point) -> tuple[Point, Point, Point]:
    """Cast a body's shadows, then reopen the light slot to sweep one corner's shadow."""
    # Join the point with the light into a ray, then meet the ray with the ground. With the
    # point slot open this is a linear map Point -> Point: the shadow matrix. Nothing in the
    # expression cares whether the light is a finite point or a direction at infinity, so
    # the sun's orthographic shadow is the same line of code.
    spotlight: Camera = (point_light & Point) ^ ground
    sunlight: Camera = (sun & Point) ^ ground

    # Bind a different slot and you ask a different question. Fix one corner and leave
    # the light open: the corner's shadow is now a linear map of the light position, so a
    # whole path of light positions binds in one call. A fixed pipeline cannot express this
    # without rebuilding its matrix per light.
    shadow_of_corner: ShadowTrail = (Point & corner) ^ ground

    return spotlight(body), sunlight(body), shadow_of_corner(light_path)


# ---------------------------------------------------------------------------
# Cameras and epipolar geometry: one rig, moved as a map
# ---------------------------------------------------------------------------
def stereo(subject: Point, centre: Point, screen: Plane,
           rig_1: Motor, rig_2: Motor) -> tuple[Point, Point, Point, Line, Correspondence]:
    """Move the supplied centre and screen into two poses and relate their images."""
    # The same join-then-meet with a line in the open slot is the camera for lines.
    camera: Camera = (centre & Point) ^ screen
    line_camera: LineCamera = (centre & Line) ^ screen

    # The camera moves as a map: pull world points back into the rig frame, push image
    # points forward. Because the map is an extensor and not a matrix baked at the end, the
    # line camera moves the same way and stays consistent with the point camera, image of a
    # join = join of the images.
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
    # vanishes. Leave both point slots open and that sentence is a bilinear form whose
    # kernel is the fundamental matrix. There is no matrix to slap on at the end here; the
    # form exists only because the rays were never evaluated.
    correspondence: Correspondence = (centre_1 & Point) ^ (centre_2 & Point)

    return image_1, image_2, epipole_2, epipolar_lines_2, correspondence
