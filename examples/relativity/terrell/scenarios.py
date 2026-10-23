"""Scenes for Terrell rotation: a sphere above a cube passing a camera, photographed at rest, at six
tenths and at nine tenths of the speed of light, each time its centre is seen along the same line of
sight."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.relativity.terrell import core

mv = core.mv
travel, line_of_sight, up = core.space
# The distance from the camera to the track, the speeds of the cases, and the lines of sight to the
# centre of the pair as it passes, as angles from the motion.
track_distance = 4.5
body_speeds = np.array([0.0, 0.6, 0.9])
n_frames = 72
sightings = np.linspace(3 * np.pi / 4, np.pi / 4, n_frames)
# The sphere's radius and the cube's half-width, and how far above and below the centre of the pair
# their centres are.
sphere_radius = 1.0
half_width = 1.0
height = 1.6
# The angles between the sphere's meridians and between its parallels, and how far its parallels reach from its
# equator; the samples around its outline in the checks; the duration of a frame.
meridian_spacing = np.pi / 4
parallel_spacing = np.pi / 6
parallel_reach = np.pi / 3
n_outline = 96
duration_ms = 60


# --- math -----------------------------------------------------------------------------
def cameras() -> Iterator[core.Antivector]:
    """Per frame, where the camera is in each body's frame `[cases]` when the centre of the pair is seen along that
    frame's line of sight."""
    frame = core.boost(body_speeds)                                                 # [cases] Rotor
    for sighting in sightings:
        yield core.rest_camera(frame, core.sighting(track_distance, sighting))     # [cases] Antivector


def photographs(cameras: Iterator[core.Antivector]) -> list[np.ndarray]:
    """The pair photographed from each of the cameras, one lane per speed."""
    from examples.relativity.terrell import render

    return render.animate_photographs(core.boost(body_speeds), cameras, height * up, sphere_radius, -height * up,
                                      half_width, meridian_spacing, parallel_spacing, parallel_reach)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation

    save_animation(photographs(cameras()), "terrell_rotation", duration_ms)

    # checks
    # Seen at right angles to its motion, the body is photographed from behind the perpendicular, by the
    # speed's fraction of the distance: turned by the arcsine of the speed.
    frame = core.boost(body_speeds)                                                 # [cases] Rotor
    camera = core.rest_camera(frame, core.sighting(track_distance, np.pi / 2))      # [cases] Antivector
    leaning = (mv.x & camera) / camera.squared().square_root()                # [cases] Scalar
    np.testing.assert_allclose(leaning.to_array(), -body_speeds, atol=1e-12)
    # The sphere's outline is a circle on the sky in every frame: its rays lie in one hyperplane through
    # the camera, so the join of any four of them vanishes.
    camera = core.rest_camera(frame, core.sighting(track_distance, sightings[0]))   # [cases] Antivector
    angles = np.linspace(0, 2 * np.pi, n_outline + 1)
    grazing = core.outline(camera, height * up, sphere_radius, angles)             # [cases, outline + 1] Antivector
    outline = frame[:, None] >> core.rays(camera[:, None], grazing)              # [cases, outline + 1] Antivector
    # Each ray scaled to one unit of the camera's time.
    points = outline / (mv.t & outline)                                        # [cases, outline + 1] Antivector
    quarter = n_outline // 4
    joined = points[:, 0] & points[:, quarter] & points[:, 2 * quarter] & points[:, 3 * quarter - 5]   # [cases] Scalar
    np.testing.assert_allclose(joined.to_array(), 0, atol=1e-12)


if __name__ == "__main__":
    main()
