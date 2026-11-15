"""Scene parameters and animation driver photograph a sphere and cube passing at three relativistic speeds."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from examples.relativity.terrell import core

mv = core.mv
travel, line_of_sight, up = core.space
track_distance = 4.5
body_speeds = np.array([0.0, 0.6, 0.9])
n_frames = 108
sightings = np.linspace(3 * np.pi / 4, np.pi / 4, n_frames)
sphere_radius = 1.0
half_width = 1.0
height = 1.6
meridian_spacing = np.pi / 4
parallel_spacing = np.pi / 6
parallel_reach = np.pi / 3
n_outline = 96
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def camera_stream() -> Iterator[core.Antivector]:
    """Yields the camera's rest-frame position (`[3] Antivector-`) for the three body speeds across all sightings."""
    frames = core.boost_rotors(body_speeds)                                     # [3] Rotor+
    return core.camera_positions(frames, track_distance, sightings)


# --- plumbing -------------------------------------------------------------------------
def photograph_frames(cameras: Iterable[core.Antivector]) -> Iterator[np.ndarray]:
    """Yields multi-lane stereographic photographs from streamed camera positions."""
    from examples.relativity.terrell import render

    frames = core.boost_rotors(body_speeds)                                     # [3] Rotor+
    return render.photographs_animation(
        frames, height * up, sphere_radius, -height * up, half_width,
        meridian_spacing, parallel_spacing, parallel_reach, cameras,
    )


def main() -> None:
    """Streams the Terrell rotation animation once and checks aberration and circular-outline invariants."""
    from examples.util.animation import save_animation

    camera_states: list[core.Antivector] = []

    def recorded_cameras() -> Iterator[core.Antivector]:
        for state in camera_stream():
            camera_states.append(state)
            yield state

    save_animation(photograph_frames(recorded_cameras()), "terrell_rotation", frame_duration_ms)

    # checks
    frames = core.boost_rotors(body_speeds)                                     # [3] Rotor+
    sideways_camera = core.rest_camera(frames, core.sighting_events(track_distance, np.pi / 2))  # [3] Antivector-
    leaning = (mv.x & sideways_camera) / sideways_camera.antinorm()             # [3] Scalar+
    assert ((leaning + body_speeds).abs() < 1e-8).all()

    first_camera = camera_states[0]                                             # [3] Antivector-
    outline_angles = np.linspace(0, 2 * np.pi, n_outline, endpoint=False)
    grazing_points = core.grazing_outline(first_camera, height * up, sphere_radius, outline_angles)  # [3, n_outline] Antivector-
    boosted_rays = frames[:, None] >> core.rest_rays(first_camera[:, None], grazing_points)  # [3, n_outline] Antivector-
    sky_points = boosted_rays / (mv.t & boosted_rays)                           # [3, n_outline] Antivector-
    quarter = n_outline // 4
    four_ray_join = sky_points[:, 0] & sky_points[:, quarter] & sky_points[:, 2 * quarter] & sky_points[:, 3 * quarter - 5]  # [3] Scalar+
    assert (four_ray_join.abs() < 1e-8).all()


if __name__ == "__main__":
    main()
