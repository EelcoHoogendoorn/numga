"""Drawing for vortices on the sphere, in the projective picture: every vortex at both of its antipodal points, the pair its bivector stands for. Each sphere is ray traced on the GPU, the points on it as dots with trails behind them that fade into the sphere's colour. What lies on the far side is hidden by the sphere: its antipode on the near side shows it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate
from examples.fluids.sphere_vortices import core
from numga.topology import Topology
from numga.render import Scene, simplices, solids

sm = solids.mv
# One colour per vortex, repeating, and the sphere's colour; the trail's length in steps; the trails' width and the
# dots' radius in pixels, the pixels across a sphere, and the margin around it as a share of its radius.
palette = np.array([to_rgb(colour) for colour in
                    ("#c84b31", "#2d6a9f", "#4c956c", "#e0a458", "#8e7dbe", "#d17a9b", "#5fa8d3", "#7f6a4f")])
sphere_colour = np.array([0.90, 0.91, 0.93])
n_trail_steps = 120
trail_width, dot_radius = 1.6, 2.2
n_pixels, margin = 160, 0.08
# The intrinsic scalar one: a pivot, the meet of great circles that generates the turn about its point, times it is
# that point as an offset from the sphere's centre, every coefficient kept.
intrinsic_one = core.mv.antipseudoscalar([1])                                # [] Scalar-


def animate(histories: list[core.Pivot], view: tuple[float, float], stride: int) -> Iterator[np.ndarray]:
    """Frames of vortices on spheres side by side, one sphere per history, all of one length, seen from (elevation, azimuth) in degrees, the azimuth measured from x, every vortex at both its antipodal points: a frame every `stride` steps, its trails through the last `n_trail_steps` steps, at least one, older parts fading into the sphere. The spheres stand in a row across the view, set up once, all traced in one pass."""
    elevation, azimuth = np.radians(view[0]), np.radians(view[1])
    n_spheres, n_times = len(histories), histories[0].shape[0]
    screen = solids.View(solids.aim(solids.origin, elevation, azimuth + np.pi / 2),
                         np.array([n_spheres, 1.0]) * (1 + margin), n_spheres * n_pixels)
    # The row runs along the camera's own x; each sphere's centre two radii and the margins from the next, the row
    # about the origin.
    row_direction = screen.camera >> sm.x.dual()                             # [] Point-
    centres = solids.origin + row_direction * (2 * (1 + margin) * (np.arange(n_spheres) - (n_spheres - 1) / 2))   # [n_spheres] Point-
    # Each vortex and its antipode, in one colour, on its sphere in the row.
    antipodal_pairs = [concatenate((vortices, -vortices), axis=1).normalized() * intrinsic_one for vortices in histories]
    places = concatenate([centre + solids.context.include_from_algebra(pair) for pair, centre in zip(antipodal_pairs, centres)],
                         axis=1)                                             # [n_times, ...] Point-
    colours = np.concatenate([palette[np.arange(2 * vortices.shape[1]) % vortices.shape[1] % len(palette)]
                              for vortices in histories])                    # [..., 3]
    scene = Scene(screen, np.ones(3), [(solids.ball(centres, 1.0)[:, None], sphere_colour)], [], inks=[],
                  translucent=[])
    for step in range(0, n_times, stride):
        first, last = max(step - n_trail_steps, 0), max(step, 1)
        n_segments = last - first
        # Each trail's segment its vortex's colour, mixed with the sphere's the older its middle is.
        ages = np.clip((np.arange(first, last) + 0.5 - first) / n_segments, 0, 1)[:, None, None]   # [n_segments, 1, 1]
        shades = sphere_colour + (colours - sphere_colour) * ages             # [n_segments, ..., 3]
        # Each vortex's trail a chain of edges down the steps, through the trails' points numbered as they lie.
        trails = places[first:last + 1]                                       # [n_segments + 1, ...] Point-
        indices = np.arange(np.prod(trails.shape)).reshape(trails.shape)
        chains = Topology(indices.size, np.stack([indices[:-1], indices[1:]], axis=-1).reshape(-1, 2))
        tubes = simplices.tubes(trails.reshape(-1), chains, shades.reshape(-1, 3), trail_width, screen,
                                gloss=solids.gloss)
        dots = solids.ball(places[step], dot_radius * screen.pixel)[:, None]
        yield scene.draw([(dots, colours)], inks=[], translucent=[], meshes=[(tubes, 1.0)])
