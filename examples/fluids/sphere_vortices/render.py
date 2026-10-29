"""Drawing for vortices on the sphere, in the projective picture: every vortex at both of its antipodal
points, the pair its bivector stands for. Each sphere is ray traced on the GPU, the points on it as dots with trails
behind them that fade into the sphere's colour. What lies on the far side is hidden by the sphere: its antipode on the
near side shows it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.fluids.sphere_vortices import core
from examples.util.wgsl import solids, strokes
from examples.util.wgsl.scene import Scene

sm = solids.mv
# One colour per vortex, repeating, and the sphere's colour; the trail's length in steps; the trails' width and the
# dots' radius in pixels, the pixels across a sphere, and the margin around it as a share of its radius.
strands = np.array([to_rgb(colour) for colour in
                    ("#c84b31", "#2d6a9f", "#4c956c", "#e0a458", "#8e7dbe", "#d17a9b", "#5fa8d3", "#7f6a4f")])
sphere_colour = np.array([0.90, 0.91, 0.93])
trail = 120
trail_width, dot_radius = 1.6, 2.2
n_pixels, margin = 160, 0.08
# The planes x = 0, y = 0 and z = 0 of the sphere's space, the drawing frame.
axes = stack((core.mv.x, core.mv.y, core.mv.z))                              # [3] Plane+


def gpu_directions(pivots: core.Pivot) -> solids.Point:
    """The GPU library's own directions toward points of the unit sphere: their coordinates, read by pairing the
    points with the drawing frame, are the one hand-off from the sphere's algebra to the renderer's."""
    return sm.direction((axes & pivots[..., None]).to_array())


def animate(histories: list[core.Pivot], view: tuple[float, float], every: int) -> Iterator[np.ndarray]:
    """Frames of vortices on spheres side by side, one sphere per history `[steps, vortices]`, all of one
    length, seen from (elevation, azimuth) in degrees, the azimuth measured from x, every vortex at both its antipodal
    points: a frame every `every` steps, its trails through the last `trail` steps, at least one, older parts fading
    into the sphere. The spheres stand in a row across the view, set up once, all traced in one pass."""
    elevation, azimuth = np.radians(view[0]), np.radians(view[1])
    count = len(histories)
    seen = solids.View(solids.aim(solids.origin, elevation, azimuth + np.pi / 2),
                       np.array([count, 1.0]) * (1 + margin), count * n_pixels)
    # The row runs across the view, along the camera's own x; each sphere's centre two radii and the margins from the
    # next, the row about the origin.
    across = seen.camera >> sm.x.dual()                                       # [] Point-
    centres = solids.origin + across * (2 * (1 + margin) * (np.arange(count) - (count - 1) / 2))   # [spheres] Point-
    # Each vortex and its antipode, in one colour, on its sphere in the row.
    places = concatenate([centre + gpu_directions(concatenate((pivots, -pivots), axis=1).normalized())
                          for pivots, centre in zip(histories, centres)], axis=1)   # [steps, points] Point-
    n_points = [2 * pivots.shape[1] for pivots in histories]
    colours = np.concatenate([strands[np.arange(n) % (n // 2) % len(strands)] for n in n_points])   # [points, 3]
    scene = Scene(seen, np.ones(3), [(solids.ball(centres, 1.0)[:, None], sphere_colour)], [], inks=[],
                  translucent=[])
    for time in range(0, places.shape[0], every):
        first, last = max(time - trail, 0), max(time, 1)
        # Each trail's segment its vortex's colour, mixed with the sphere's the older its middle is.
        age = np.clip((np.arange(first, last) + 0.5 - first) / max(last - first, 1), 0, 1)[:, None, None]   # [segments, 1, 1]
        shades = sphere_colour + (colours - sphere_colour) * age                  # [segments, points, 3]
        drawn = strokes.segments(places[first:last].reshape(-1), places[first + 1:last + 1].reshape(-1),
                                 shades.reshape(-1, 3), trail_width, seen)
        dots = solids.ball(places[time], dot_radius * seen.pixel)[:, None]
        yield scene.draw([(dots, colours)], inks=[], translucent=[], meshes=[(drawn, 1.0)])
