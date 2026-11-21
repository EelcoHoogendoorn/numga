"""Three still camera exposures (wide, telephoto, and Scheimpflug sensor tilt) and a continuous zoom-and-refocus animation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from examples.optics.lens_camera.core import (
    Camera,
    Motor,
    Plane,
    Point,
    PointMap,
    Quadric,
    Scalar,
    SensorPlane,
    SensorPoint,
    camera_exposure,
    cone_section,
    home,
    mv,
    origin,
    scene_point,
    unit_point,
)
from numga import dim, stack

# Front and rear focal lengths; the front lens stands fixed at x == 1 while the rear lens translates to zoom.
focal_lengths = mv.scalar([[1.0], [0.6]], unit=dim.length)                      # [2] Scalar+, length
front_at = mv.scalar(1.0, unit=dim.length)                                      # [] Scalar+, length
# Three depth layers of scene points and one far-layer subject point with three traced aperture-rim heights.
depths = np.array([-3.2, -2.2, -1.6])
n_depths = len(depths)
n_heights, n_widths = 5, 4
half_height, half_width = 0.8, 0.5
scene = scene_point(
    np.stack(
        np.meshgrid(
            depths,
            np.linspace(-half_height, half_height, n_heights),
            np.linspace(-half_width, half_width, n_widths),
            indexing="ij",
        ),
        axis=-1,
    )
)                                                                               # [n_depths, n_heights, n_widths] Point-, 1 | length
subject = scene_point(np.array([-3.2, 0.0, -1.0 / 6.0]))                        # [] Point-, 1 | length
rim_heights = np.array([1, 0, -1])
# Wide, telephoto, and tilted-sensor still settings.
wide_at, tele_at = mv.scalar(1.4, unit=dim.length), mv.scalar(1.8, unit=dim.length)
focus_distance = mv.scalar(2.2, unit=dim.length)
aperture = mv.scalar(0.45, unit=dim.length)
sensor_tilt = np.radians(25)
# Zoom, focus, and aperture oscillation midpoints and swings across the animation loop.
rear_middle, rear_swing = mv.scalar(1.6, unit=dim.length), mv.scalar(0.2, unit=dim.length)
focus_middle, focus_swing = mv.scalar(2.4, unit=dim.length), mv.scalar(0.8, unit=dim.length)
aperture_middle, aperture_swing = mv.scalar(0.3, unit=dim.length), mv.scalar(0.1, unit=dim.length)
n_frames = 108
frame_duration_ms = 40
n_section_samples = 48


# --- math -----------------------------------------------------------------------------
def camera_setting(
    rear_at: Scalar, focus_at: Scalar, tilt: float, radius: Scalar,
) -> tuple[PointMap, Camera, Motor, Quadric, Plane, Point]:
    """Exposing `scene` with the rear lens at `rear_at`, focus target at `focus_at`, sensor tilt `tilt`, and aperture `radius` evaluates `camera_exposure`."""
    placements = (mv.xw * (stack((front_at, rear_at)) / 2)).exp()               # [2] Rotor+, 1 | length
    focus = origin + mv.x.dual() * focus_at                                     # [] Point-, 1 | length
    turn = (mv.xy * (tilt / 2)).exp()                                           # [] Rotor+
    rim = origin + mv.y.dual() * (radius * rim_heights)                         # [3] Point-, 1 | length
    return camera_exposure(scene, subject, focal_lengths, placements, focus, turn, radius, rim)


def still_exposures() -> tuple[tuple, tuple, tuple]:
    """Wide, telephoto, and tilted-sensor exposures form the three comparison stills."""
    wide = camera_setting(wide_at, -focus_distance, 0.0, aperture)
    tele = camera_setting(tele_at, -focus_distance, 0.0, aperture)
    tilted = camera_setting(tele_at, -focus_distance, sensor_tilt, aperture)
    return wide, tele, tilted


def zoom_states(n_frames: int) -> Iterator[tuple]:
    """Yields `(rear_at, focus_at, radius, exposure)` across `n_frames` phases as the camera zooms, refocuses, and stops down."""
    for phase in np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False):
        rear_at = rear_middle + rear_swing * np.sin(phase)
        focus_at = focus_middle - focus_swing * np.cos(phase)
        radius = aperture_middle + aperture_swing * np.sin(2 * phase)
        yield rear_at, focus_at, radius, camera_setting(rear_at, -focus_at, 0.0, radius)


# --- plumbing -------------------------------------------------------------------------
def save_states(
    states: Iterable, animate: Callable[..., Iterable[np.ndarray]], *arguments, name: str, duration_ms: int,
) -> list:
    """Streams `states` once through `animate(*arguments, states)` into `save_animation` and returns the collected states."""
    from examples.util.animation import save_animation

    kept = []

    def passing() -> Iterator:
        for state in states:
            kept.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms)
    return kept


def main() -> None:
    """Saves the three-still sensor comparison and zoom animation, then checks collineation incidence and blur conics."""
    from examples.optics.lens_camera import render
    from examples.util.animation import save_figure

    wide, tele, tilted = still_exposures()
    save_figure(render.stills_figure((wide, tele, tilted)), "lens_camera")
    recorded = save_states(zoom_states(n_frames), render.camera_animation, scene, name="lens_camera", duration_ms=frame_duration_ms)

    # checks
    # The lens train's line map agrees with the join of its point collineation images, every image cone has its
    # vertex at the collineated scene point, the sensor section of an image cone lies on the pushed-forward
    # aperture dual conic, and a point on the focal plane collapses its blur section to a single point.
    collineation, camera, frame, cones, _, _ = tilted
    place_front = (mv.xw * (front_at / 2)).exp()                                # [] Rotor+
    sensor, centre = frame >> home, place_front >> origin                       # [] Plane+, Point-
    pupil_directions = stack((mv.y.dual(), mv.z.dual()))                        # [2] Point-
    aperture_disc = (pupil_directions * (Plane & pupil_directions)).sum(-1) * aperture**2 - origin * (Plane & origin)  # [] Point- <- Plane+
    pupil = place_front >> aperture_disc(place_front << Plane)                  # [] Point- <- Plane+
    other = scene_point(np.array([[-1.0, 0.3, 0.1], [-1.5, 0.2, -0.4], [-2.0, -0.5, 0.6]]))  # [3] Point-
    camera_hits = unit_point(camera(Point, other)(scene[0, 0, 0]))              # [3] Point-
    joined_hits = unit_point((collineation(scene[0, 0, 0]) & collineation(other)) ^ sensor)  # [3] Point-
    assert bool(((camera_hits & joined_hits).norm() < mv.scalar(1e-11, unit=dim.length)).all())
    assert bool(((cones(collineation(scene)) & collineation(scene)).abs() < mv.scalar(1e-9, unit=dim.length**4)).all())

    for index in ((0, 0, 0), (2, 4, 3)):
        start = (collineation(scene[index]) & collineation(centre)) ^ sensor    # [] Point-
        boundary = cone_section(cones[index], start, frame, n_section_samples)  # [n_section_samples] Point-
        pupil_to_sensor = camera(scene[index])                                  # [] Point- <- Point-
        blur_disc = pupil_to_sensor(pupil(pupil_to_sensor.adjugate()))          # [] Point- <- Plane+
        home_blur_disc = (frame << blur_disc(frame >> SensorPlane)).cast(SensorPoint)  # [] Point- <- Plane+
        hits = (frame << boundary).cast(SensorPoint)                            # [n_section_samples] Point-
        assert bool(((home_blur_disc.solve(hits) & hits).abs() < 1e-9).all())

    wide_collineation, _, wide_frame, wide_cones, _, _ = wide
    focused_hit = (wide_collineation(scene[1, 2, 1]) & wide_collineation(centre)) ^ (wide_frame >> home)  # [] Point-
    focused_section = unit_point(cone_section(wide_cones[1, 2, 1], focused_hit, wide_frame, n_section_samples))  # [n_section_samples] Point-
    assert bool(((focused_section & unit_point(focused_hit)).norm() < mv.scalar(1e-6, unit=dim.length)).all())
    assert len(recorded) == n_frames


if __name__ == "__main__":
    main()
