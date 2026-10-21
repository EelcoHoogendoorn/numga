"""Scenes for Terrell rotation: a sphere above a cube passing a camera, photographed at rest, at six
tenths and at nine tenths of the speed of light, each time its centre is seen along the same line of
sight."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate
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
# The samples along each edge of the cube, along each line of the sphere's grid and around its outline;
# the longitudes of the grid's meridians and the latitudes of its parallels; the duration of a frame.
n_edge = 16
n_grid = 64
n_outline = 96
grid_longitudes = np.arange(8) * np.pi / 4
grid_latitudes = np.array([-np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3])
duration_ms = 60


# --- math -----------------------------------------------------------------------------
def cube() -> tuple[core.Antivector, core.Antivector]:
    """The cube's faces at rest: the boundary of each face `[6, 4 * edge]`, and each face's outward
    normal `[6]`."""
    # One edge of the top face, turned by quarter turns about the vertical into the face's boundary.
    edge = up + travel + line_of_sight * np.linspace(-1, 1, n_edge, endpoint=False)  # [edge] Antivector
    square = (mv.xy * (-np.arange(4)[:, None] * np.pi / 4)).exp() >> edge     # [4, edge] Antivector
    # The top face turned onto each face: itself, upside down, and by quarter turns towards the four sides.
    turns = (mv.zx * (-np.array([0, 2, 1, -1, 0, 0]) * np.pi / 4)
             + mv.yz * (-np.array([0, 0, 0, 0, 1, -1]) * np.pi / 4)).exp()     # [6] Rotor
    boundary = turns[:, None] >> square.reshape(4 * n_edge)                    # [6, 4 * edge] Antivector
    return -height * up + half_width * boundary, turns >> up


def globe() -> core.Antivector:
    """The sphere's grid at rest, its meridians and then its parallels `[lines, grid]`."""
    latitudes = np.linspace(-np.pi / 2, np.pi / 2, n_grid)
    longitudes = np.linspace(0, 2 * np.pi, n_grid)
    # A direction at a latitude and a longitude: turned up out of the motion, then about the vertical.
    meridians = (mv.xy * (-grid_longitudes[:, None] / 2)).exp() >> ((mv.zx * (latitudes / 2)).exp() >> travel)
    parallels = (mv.xy * (-longitudes / 2)).exp() >> ((mv.zx * (grid_latitudes[:, None] / 2)).exp() >> travel)
    return height * up + sphere_radius * concatenate([meridians, parallels])          # [lines, grid] Antivector


def frames() -> Iterator[tuple[core.Antivector, core.Scalar, core.Antivector, core.Scalar, core.Antivector]]:
    """Per frame, in every case `[cases]`: the rays from the boundary of each face of the cube and how far
    the face turns towards the camera, the rays from the sphere's grid and how far each point of it
    turns towards the camera, and the rays grazing the sphere."""
    frame = core.boost(body_speeds)                                                 # [cases] Rotor
    faces, normals = cube()                                                    # [6, 4 * edge], [6] Antivector
    grid = globe()                                                             # [lines, grid] Antivector
    face_centres = -height * up + half_width * normals                         # [6] Antivector
    sphere = height * up                                                       # [] Antivector
    angles = np.linspace(0, 2 * np.pi, n_outline + 1)
    for sighting in sightings:
        # Where the camera is in each body's frame when the centre of the pair is seen along this line.
        camera = core.rest_camera(frame, core.sighting(track_distance, sighting))    # [cases] Antivector
        outline = core.outline(camera, sphere, sphere_radius, angles)                 # [cases, outline + 1] Antivector
        yield (frame[:, None, None] >> core.rays(camera[:, None, None], faces),
               core.facing(camera[:, None], face_centres, normals),
               frame[:, None, None] >> core.rays(camera[:, None, None], grid),
               core.facing(camera[:, None, None], grid, grid - sphere),
               frame[:, None] >> core.rays(camera[:, None], outline))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.relativity.terrell import render

    save_animation(render.animate_photographs(frames()), "terrell_rotation", duration_ms)

    # checks
    # Seen at right angles to its motion, the body is photographed from behind the perpendicular, by the
    # speed's fraction of the distance: turned by the arcsine of the speed.
    frame = core.boost(body_speeds)                                                 # [cases] Rotor
    camera = core.rest_camera(frame, core.sighting(track_distance, np.pi / 2))      # [cases] Antivector
    leaning = (mv.x & camera) / camera.squared().square_root()                # [cases] Scalar
    np.testing.assert_allclose(leaning.to_array(), -body_speeds, atol=1e-12)
    # The sphere's outline is a circle on the sky in every frame: its rays lie in one hyperplane through
    # the camera, so the join of any four of them vanishes.
    *_, outline = next(iter(frames()))                                         # [cases, outline + 1] Antivector
    # Each ray scaled to one unit of the camera's time.
    points = outline / (mv.t & outline)                                        # [cases, outline + 1] Antivector
    quarter = n_outline // 4
    joined = points[:, 0] & points[:, quarter] & points[:, 2 * quarter] & points[:, 3 * quarter - 5]   # [cases] Scalar
    np.testing.assert_allclose(joined.to_array(), 0, atol=1e-12)


if __name__ == "__main__":
    main()
