"""Drawing for vortices on the sphere, in the projective picture: every vortex at both of its antipodal
points, the pair its bivector stands for. Each sphere is ray traced on the GPU, the points on it as dots with trails
behind them that fade into the sphere's colour. What lies on the far side is hidden by the sphere: its antipode on the
near side shows it.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb

from examples.fluids.sphere_vortices import core
from examples.wgsl import meshes, solids

sm = solids.mv
# One colour per vortex, repeating, and the sphere's colour; the trail's length in steps; the trails' width and the
# dots' radius in pixels, the points around a trail, the pixels across a sphere, and the margin around it as a share
# of its radius.
STRANDS = np.array([to_rgb(colour) for colour in
                    ("#c84b31", "#2d6a9f", "#4c956c", "#e0a458", "#8e7dbe", "#d17a9b", "#5fa8d3", "#7f6a4f")])
SPHERE_COLOUR = np.array([0.90, 0.91, 0.93])
TRAIL = 240
TRAIL_WIDTH, DOT_RADIUS, N_AROUND = 3.0, 4.0, 6
N_PIXELS, MARGIN = 300, 0.08


def coordinates(pivots: core.Pivot) -> np.ndarray:
    """The unit directions `[..., 3]` of points of the sphere: where they lie on it."""
    parts = pivots.cast(core.ga.subspace("yz zx xy")).kernel
    return parts / np.linalg.norm(parts, axis=-1, keepdims=True)


def trails(places: np.ndarray, pixel: float) -> tuple[solids.Point, np.ndarray, dict]:
    """Every point's path through every step `[steps, points, 3]` as a tube a trail's width across: its vertices, its
    triangles `[points, steps - 1, triangles, 3]`, a run of them per step, and their normals, built once."""
    rings, faces = meshes.tube(np.moveaxis(places, 1, 0), TRAIL_WIDTH / 2 * pixel, N_AROUND)
    return rings, faces.reshape(places.shape[1], places.shape[0] - 1, -1, 3), meshes.smooth(rings, faces)


def animate(histories: list[core.Pivot], view: tuple[float, float], every: int) -> list[np.ndarray]:
    """Frames of vortices on spheres side by side, one sphere per history `[steps, vortices]`, all of one
    length, seen from (elevation, azimuth) in degrees, the azimuth measured from x, every vortex at both its antipodal
    points: a frame every `every` steps, its trails through the last `TRAIL` steps, at least one, older parts fading
    into the sphere. The spheres stand in a row across the view, all traced in one pass."""
    elevation, azimuth = np.radians(view[0]), np.radians(view[1])
    toward = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    across = np.cross([0.0, 0.0, 1.0], toward)
    across /= np.linalg.norm(across)
    count = len(histories)
    # Each sphere's centre along the row, two radii and the margins apart, the row about the origin.
    centres = (2 * (1 + MARGIN) * (np.arange(count) - (count - 1) / 2))[:, None] * across   # [spheres, 3]
    seen = solids.View(solids.aim(solids.origin, elevation, azimuth + np.pi / 2),
                       np.array([count, 1.0]) * (1 + MARGIN), count * N_PIXELS)
    # Each vortex and its antipode, in one colour, on its sphere in the row.
    places = np.concatenate([np.concatenate([coordinates(pivots), -coordinates(pivots)], axis=1) + centre
                             for pivots, centre in zip(histories, centres)], axis=1)   # [steps, points, 3]
    n_points = [2 * pivots.shape[1] for pivots in histories]
    colours = np.concatenate([STRANDS[np.arange(n) % (n // 2) % len(STRANDS)] for n in n_points])   # [points, 3]
    rings, faces, normals = trails(places, seen.pixel)
    balls = [(solids.ball(sm.point(centres), 1.0)[:, None], SPHERE_COLOUR)]

    def frame(time: int) -> np.ndarray:
        first, last = max(time - TRAIL, 0), max(time, 1)
        # Each trail's vertex its vortex's colour, mixed with the sphere's the older it is.
        age = np.clip((np.arange(places.shape[0]) - first) / max(last - first, 1), 0, 1)[None, :, None]   # [1, steps, 1]
        shades = SPHERE_COLOUR + (colours[:, None] - SPHERE_COLOUR) * age        # [points, steps, 3]
        shown = faces[:, first:last].reshape(-1, 3)                               # [faces, 3]
        corners = np.repeat(shades.reshape(-1, 3), N_AROUND, axis=0)[shown]       # [faces, 3, 3]
        corner_normals = {name: corner.reshape(faces.shape[:-1])[:, first:last].reshape(-1) for name, corner in normals.items()}
        drawn = meshes.samples(rings, shown, corners, corner_normals, seen)
        dots = solids.ball(sm.point(places[time]), DOT_RADIUS * seen.pixel)[:, None]
        return solids.draw(balls + [(dots, colours)], seen, np.ones(3), meshes=[(*drawn, 1.0)])

    return [frame(time) for time in range(0, places.shape[0], every)]
