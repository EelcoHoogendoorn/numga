"""Scenes of the lens camera: three still settings, and a zoom and refocus in motion.

Every setting uses the same two-lens train on a grid of points at three depths; a setting
places the rear lens, picks the point to focus on, tilts the sensor, and sets the aperture.
"""

from __future__ import annotations

import numpy as np

from examples.optics.lens_camera.core import (
    Motor, Plane, Point, SensorPlane, SensorPoint, direction, expose, home, mv,
    on_planes, origin, point, section, unit,
)

FOCAL = np.array([1.0, 0.6])                  # focal lengths of the front and the rear lens
FRONT_AT = 1.0                                # the front lens sits at x = 1; the rear lens moves to zoom


# A scene: a grid of points at three depths in front of the camera, and one subject on the far
# layer whose rays through the aperture rim are traced.
SCENE: Point = point(np.stack(np.meshgrid(
    np.array([-3.2, -2.2, -1.6]), np.linspace(-0.8, 0.8, 5), np.linspace(-0.5, 0.5, 4), indexing="ij",
), axis=-1))                                                              # [depth, height, width]
SUBJECT: Point = point(np.array([-3.2, 0.0, -1.0 / 6.0]))


def setting(rear_at: float, focus_at: float, tilt: float, radius: float):
    """Expose the scene with the rear lens at rear_at, focused on the axis at focus_at, the
    sensor tilted by tilt radians, and an aperture of the given radius."""
    placements: Motor = (mv.xw * (np.array([FRONT_AT, rear_at]) / 2)).exp()
    focus: Point = point(np.array([focus_at, 0.0, 0.0]))
    turn: Motor = (mv.xy * (tilt / 2)).exp()
    rim: Point = point(np.array([[0.0, radius, 0.0], [0.0, 0.0, 0.0], [0.0, -radius, 0.0]]))
    return expose(SCENE, SUBJECT, FOCAL, placements, focus, turn, radius, rim)


def stills():
    """Wide and tele at one focus and aperture, and tele with the sensor tilted 25 degrees."""
    wide = setting(1.4, -2.2, 0.0, 0.45)
    tele = setting(1.8, -2.2, 0.0, 0.45)
    tilted = setting(1.8, -2.2, np.radians(25), 0.45)

    # --- checks -------------------------------------------------------------
    collineation, cam, frame, cones, planes, _ = tilted
    place_front: Motor = (mv.xw * (FRONT_AT / 2)).exp()
    sensor, centre = frame >> home, place_front >> origin
    dy, dz = direction(np.eye(3)[1:])
    unit_disc = dy * (dy & Plane) + dz * (dz & Plane) - origin * (origin & Plane)
    pupil = place_front >> (unit_disc * 0.45**2 - origin * (origin & Plane) * (1.0 - 0.45**2))(place_front << Plane)
    # The train on lines is the join of the collineation's images: the lens is a collineation.
    other = point(np.array([[-1.0, 0.3, 0.1], [-1.5, 0.2, -0.4], [-2.0, -0.5, 0.6]]))
    np.testing.assert_allclose(cam(Point, other)(SCENE[0, 0, 0]).kernel,
                               ((collineation(SCENE[0, 0, 0]) & collineation(other)) ^ sensor).kernel, atol=1e-12)
    # The image cone's vertex is the collineation's image of the subject.
    np.testing.assert_allclose((collineation(SCENE) & cones(collineation(SCENE))).kernel, 0.0, atol=1e-9)
    # The sensor's section of the image cone, traced from the chief-ray hit, lies on the conic
    # obtained the short way: the aperture disc, a flat dual quadric, pushed through the
    # pupil-to-sensor collineation of the same point.
    for index in ((0, 0, 0), (2, 4, 3)):
        start = (collineation(SCENE[index]) & collineation(centre)) ^ sensor
        boundary = section(cones[index], start, frame, 48)
        to_sensor = cam(SCENE[index])
        pushed = to_sensor(pupil(on_planes(to_sensor)))
        section_dual = (frame << pushed(frame >> SensorPlane)).cast(SensorPoint.output_subspace)
        hits = (frame << boundary).cast(SensorPoint.output_subspace)
        np.testing.assert_allclose((hits & section_dual.solve(hits)).kernel, 0.0, atol=1e-10)
    # A point at the focus depth images to a point: its section collapses onto the chief-ray hit.
    collineation, cam, frame, cones, planes, _ = wide
    start = (collineation(SCENE[1, 2, 1]) & collineation(centre)) ^ (frame >> home)
    np.testing.assert_allclose(unit(section(cones[1, 2, 1], start, frame, 48)).kernel,
                               unit(start).broadcast_to((48,)).kernel, atol=1e-6)

    return wide, tele, tilted


def motion(frames: int):
    """The rear lens swings to zoom while the focus and the aperture change.

    Yields per frame the rear lens position, the focus distance, the aperture radius and the exposure.
    """
    for t in np.linspace(0.0, 2 * np.pi, frames, endpoint=False):
        rear_at, focus_at, radius = 1.6 + .2 * np.sin(t), 2.4 - .8 * np.cos(t), .3 + .1 * np.sin(2 * t)
        yield rear_at, focus_at, radius, setting(rear_at, -focus_at, 0.0, radius)


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.optics.lens_camera import render

    save_figure(render.draw_stills(stills()), "lens_camera")
    save_animation(render.animate_camera(motion(72), SCENE), "lens_camera", 60)
