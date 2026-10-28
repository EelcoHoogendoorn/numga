"""The lens camera is shown in three still settings, and in motion as it zooms and refocuses.

Every setting uses the same two-lens train on a grid of points at three depths; a setting
places the rear lens, picks the point to focus on, tilts the sensor, and sets the aperture.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack

from examples.optics.lens_camera.core import (
    Camera, Motor, Plane, Point, PointMap, Quadric, SensorPlane, SensorPoint, expose, home, mv,
    origin, point, section, unit,
)

# Focal lengths of the front and the rear lens. The front lens sits at x == 1; the rear lens moves to zoom.
focal_lengths = np.array([1.0, 0.6])
front_at = 1.0
# A scene: a grid of points at three depths in front of the camera, and one subject on the far layer
# whose rays through the aperture rim are traced, at the rim's top, its centre and its bottom, in aperture radii.
depths = np.array([-3.2, -2.2, -1.6])
n_heights, n_widths = 5, 4
half_height, half_width = 0.8, 0.5
scene = point(np.stack(np.meshgrid(
    depths, np.linspace(-half_height, half_height, n_heights), np.linspace(-half_width, half_width, n_widths), indexing="ij",
), axis=-1))                                                                    # [depths, heights, widths] Point-
subject = point(np.array([-3.2, 0.0, -1.0 / 6.0]))                              # [] Point-
rim_heights = np.array([1, 0, -1])
# The stills: the rear lens wide and tele, the focus distance, the aperture radius, and the sensor's tilt.
wide_at, tele_at = 1.4, 1.8
focus_distance = 2.2
aperture = 0.45
sensor_tilt = np.radians(25)
# The motion: the rear lens, the focus distance and the aperture radius, each about a middle and by a swing;
# its frames, and the duration of a frame in milliseconds.
rear_middle, rear_swing = 1.6, 0.2
focus_middle, focus_swing = 2.4, 0.8
aperture_middle, aperture_swing = 0.3, 0.1
n_frames = 108
frame_duration_ms = 40
# The samples traced around each blur section in the checks.
n_section_samples = 48


# --- math -----------------------------------------------------------------------------
def setting(rear_at: float, focus_at: float, tilt: float, radius: float) -> tuple[PointMap, Camera, Motor, Quadric, Plane, Point]:
    """The scene exposed with the rear lens at `rear_at`, focused on the axis at `focus_at`, the sensor tilted by
    `tilt` radians, and an aperture of the given radius."""
    placements = (mv.xw * (np.array([front_at, rear_at]) / 2)).exp()            # [lenses] Motor+
    focus = origin + mv.x.dual() * focus_at                                     # [] Point-
    turn = (mv.xy * (tilt / 2)).exp()                                           # [] Motor+
    rim = origin + mv.y.dual() * (radius * rim_heights)                         # [rays] Point-
    return expose(scene, subject, focal_lengths, placements, focus, turn, radius, rim)


def stills() -> tuple[tuple, tuple, tuple]:
    """Wide and tele at one focus and aperture, and tele with the sensor tilted."""
    wide = setting(wide_at, -focus_distance, 0.0, aperture)
    tele = setting(tele_at, -focus_distance, 0.0, aperture)
    tilted = setting(tele_at, -focus_distance, sensor_tilt, aperture)
    return wide, tele, tilted


def motion(n_frames: int) -> Iterator[tuple]:
    """The rear lens swings to zoom while the focus and the aperture change.

    Yields per frame the rear lens position, the focus distance, the aperture radius and the exposure.
    """
    for phase in np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False):
        rear_at = rear_middle + rear_swing * np.sin(phase)
        focus_at = focus_middle - focus_swing * np.cos(phase)
        radius = aperture_middle + aperture_swing * np.sin(2 * phase)
        yield rear_at, focus_at, radius, setting(rear_at, -focus_at, 0.0, radius)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.optics.lens_camera import render

    wide, tele, tilted = stills()
    save_figure(render.draw_stills((wide, tele, tilted)), "lens_camera")
    save_animation(render.animate_camera(motion(n_frames), scene), "lens_camera", frame_duration_ms)

    # checks
    # The train on lines is the join of the collineation's images: the lens is a collineation. The image
    # cone's vertex is the collineation's image of each scene point. The sensor's section of an image cone,
    # traced from the chief-ray hit, lies on the conic obtained the short way: the aperture disc, a flat
    # dual quadric, pushed through the pupil-to-sensor collineation of the same point. A point at the focus
    # depth images to a point: its section collapses onto the chief-ray hit.
    collineation, camera, frame, cones, planes, _ = tilted
    place_front = (mv.xw * (front_at / 2)).exp()                                # [] Motor+
    sensor, centre = frame >> home, place_front >> origin                       # [] Plane+, Point-
    pupil_directions = stack((mv.y.dual(), mv.z.dual()))                        # [2] Point-
    aperture_disc = (pupil_directions * (pupil_directions & Plane)).sum(-1) * aperture**2 - origin * (origin & Plane)
    pupil = place_front >> aperture_disc(place_front << Plane)                  # [] Point- <- Plane+
    other = point(np.array([[-1.0, 0.3, 0.1], [-1.5, 0.2, -0.4], [-2.0, -0.5, 0.6]]))
    np.testing.assert_allclose(camera(Point, other)(scene[0, 0, 0]).kernel,
                               ((collineation(scene[0, 0, 0]) & collineation(other)) ^ sensor).kernel, atol=1e-12)
    np.testing.assert_allclose((collineation(scene) & cones(collineation(scene))).kernel, 0.0, atol=1e-9)
    for index in ((0, 0, 0), (2, 4, 3)):
        start = (collineation(scene[index]) & collineation(centre)) ^ sensor
        boundary = section(cones[index], start, frame, n_section_samples)
        pupil_to_sensor = camera(scene[index])
        blur_disc = pupil_to_sensor(pupil(pupil_to_sensor.adjugate()))
        home_blur_disc = (frame << blur_disc(frame >> SensorPlane)).cast(SensorPoint)
        hits = (frame << boundary).cast(SensorPoint)
        np.testing.assert_allclose((hits & home_blur_disc.solve(hits)).kernel, 0.0, atol=1e-10)
    collineation, camera, frame, cones, planes, _ = wide
    start = (collineation(scene[1, 2, 1]) & collineation(centre)) ^ (frame >> home)
    np.testing.assert_allclose(unit(section(cones[1, 2, 1], start, frame, n_section_samples)).kernel,
                               unit(start).broadcast_to((n_section_samples,)).kernel, atol=1e-6)


if __name__ == "__main__":
    main()
