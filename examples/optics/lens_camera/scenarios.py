"""Scenes of the lens camera: three still settings, and a zoom and refocus in motion.

Every setting uses the same two-lens train on a grid of points at three depths; a setting
places the rear lens, picks the point to focus on, tilts the sensor, and sets the aperture.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.optics.lens_camera.core import (
    Motor, Plane, Point, SensorPlane, SensorPoint, expose, home, mv,
    origin, point, section, unit,
)

# Focal lengths of the front and the rear lens. The front lens sits at x == 1; the rear lens moves to zoom.
FOCAL = np.array([1.0, 0.6])
FRONT_AT = 1.0
# A scene: a grid of points at three depths in front of the camera, and one subject on the far layer
# whose rays through the aperture rim are traced.
DEPTHS = np.array([-3.2, -2.2, -1.6])
N_HEIGHTS, N_WIDTHS = 5, 4
HALF_HEIGHT, HALF_WIDTH = 0.8, 0.5
SCENE = point(np.stack(np.meshgrid(                                             # Point
    DEPTHS, np.linspace(-HALF_HEIGHT, HALF_HEIGHT, N_HEIGHTS), np.linspace(-HALF_WIDTH, HALF_WIDTH, N_WIDTHS), indexing="ij",
), axis=-1))                                                                   # [depth, height, width] Point
SUBJECT = point(np.array([-3.2, 0.0, -1.0 / 6.0]))                              # Point
# The stills: the rear lens wide and tele, the focus distance, the aperture radius, and the sensor's tilt.
WIDE, TELE = 1.4, 1.8
FOCUS = 2.2
APERTURE = 0.45
TILT = np.radians(25)
# The motion: the rear lens, the focus distance and the aperture radius, each about a middle and by a swing;
# its frames, and the duration of a frame in milliseconds.
REAR_MIDDLE, REAR_SWING = 1.6, 0.2
FOCUS_MIDDLE, FOCUS_SWING = 2.4, 0.8
APERTURE_MIDDLE, APERTURE_SWING = 0.3, 0.1
N_FRAMES = 72
FRAME_DURATION = 60
# The samples traced around each blur section in the checks.
N_SECTION = 48


# --- math -----------------------------------------------------------------------------
def setting(rear_at: float, focus_at: float, tilt: float, radius: float):
    """Expose the scene with the rear lens at rear_at, focused on the axis at focus_at, the
    sensor tilted by tilt radians, and an aperture of the given radius."""
    placements = (mv.xw * (np.array([FRONT_AT, rear_at]) / 2)).exp()            # Motor
    focus = point(np.array([focus_at, 0.0, 0.0]))                               # Point
    turn = (mv.xy * (tilt / 2)).exp()                                           # Motor
    rim = point(np.array([[0.0, radius, 0.0], [0.0, 0.0, 0.0], [0.0, -radius, 0.0]]))  # Point
    return expose(SCENE, SUBJECT, FOCAL, placements, focus, turn, radius, rim)


def stills():
    """Wide and tele at one focus and aperture, and tele with the sensor tilted."""
    wide = setting(WIDE, -FOCUS, 0.0, APERTURE)
    tele = setting(TELE, -FOCUS, 0.0, APERTURE)
    tilted = setting(TELE, -FOCUS, TILT, APERTURE)
    return wide, tele, tilted


def motion(n_frames: int) -> Iterator[tuple]:
    """The rear lens swings to zoom while the focus and the aperture change.

    Yields per frame the rear lens position, the focus distance, the aperture radius and the exposure.
    """
    for t in np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False):
        rear_at = REAR_MIDDLE + REAR_SWING * np.sin(t)
        focus_at = FOCUS_MIDDLE - FOCUS_SWING * np.cos(t)
        radius = APERTURE_MIDDLE + APERTURE_SWING * np.sin(2 * t)
        yield rear_at, focus_at, radius, setting(rear_at, -focus_at, 0.0, radius)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.optics.lens_camera import render

    wide, tele, tilted = stills()
    save_figure(render.draw_stills((wide, tele, tilted)), "lens_camera")
    save_animation(render.animate_camera(motion(N_FRAMES), SCENE), "lens_camera", FRAME_DURATION)

    # checks
    # The train on lines is the join of the collineation's images: the lens is a collineation. The image
    # cone's vertex is the collineation's image of each scene point. The sensor's section of an image cone,
    # traced from the chief-ray hit, lies on the conic obtained the short way: the aperture disc, a flat
    # dual quadric, pushed through the pupil-to-sensor collineation of the same point. A point at the focus
    # depth images to a point: its section collapses onto the chief-ray hit.
    collineation, camera, frame, cones, planes, _ = tilted
    place_front = (mv.xw * (FRONT_AT / 2)).exp()                                # Motor
    sensor, centre = frame >> home, place_front >> origin
    across_y, across_z = mv.y.dual(), mv.z.dual()
    unit_disc = across_y * (across_y & Plane) + across_z * (across_z & Plane) - origin * (origin & Plane)
    pupil = place_front >> (unit_disc * APERTURE**2 - origin * (origin & Plane) * (1.0 - APERTURE**2))(place_front << Plane)
    other = point(np.array([[-1.0, 0.3, 0.1], [-1.5, 0.2, -0.4], [-2.0, -0.5, 0.6]]))
    np.testing.assert_allclose(camera(Point, other)(SCENE[0, 0, 0]).kernel,
                               ((collineation(SCENE[0, 0, 0]) & collineation(other)) ^ sensor).kernel, atol=1e-12)
    np.testing.assert_allclose((collineation(SCENE) & cones(collineation(SCENE))).kernel, 0.0, atol=1e-9)
    for index in ((0, 0, 0), (2, 4, 3)):
        start = (collineation(SCENE[index]) & collineation(centre)) ^ sensor
        boundary = section(cones[index], start, frame, N_SECTION)
        to_sensor = camera(SCENE[index])
        pushed = to_sensor(pupil(to_sensor.adjugate()))
        section_dual = (frame << pushed(frame >> SensorPlane)).cast(SensorPoint.output_subspace)
        hits = (frame << boundary).cast(SensorPoint.output_subspace)
        np.testing.assert_allclose((hits & section_dual.solve(hits)).kernel, 0.0, atol=1e-10)
    collineation, camera, frame, cones, planes, _ = wide
    start = (collineation(SCENE[1, 2, 1]) & collineation(centre)) ^ (frame >> home)
    np.testing.assert_allclose(unit(section(cones[1, 2, 1], start, frame, N_SECTION)).kernel,
                               unit(start).broadcast_to((N_SECTION,)).kernel, atol=1e-6)


if __name__ == "__main__":
    main()
