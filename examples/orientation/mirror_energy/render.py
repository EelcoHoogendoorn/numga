"""A mass distribution and its mirror, with tangent velocities on the unit sphere, drawn on the GPU: the sphere
translucent with three great circles inked on it, each mass's path a faint stroke, the masses as balls joined into
their shape, and their velocities as arrows."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.orientation.mirror_energy.core import Point, ga
from examples.util.wgsl import image_figure, solids, strokes
from examples.util.wgsl.scene import Scene

COLOURS = np.array([to_rgb(colour) for colour in ("#df7246", "#238c9b", "#7662a1")])
SPHERE, CIRCLE, SHAPE = np.array(to_rgb("#e6ebf0")), np.array(to_rgb("#aab4bf")), np.array(to_rgb("#75808a"))
# The view, elevation and azimuth in radians, azimuth zero looking along +y; its margin as a share of its larger
# side, and the pixels across each case; how much of what lies behind them the sphere's surfaces and the paths hide;
# in pixels, the widths of the great circles and the paths, the radius of the shape's edges, a unit mass's radius,
# and the arrows' shafts, heads' length and heads' radius; and the drawn length of a unit velocity.
ELEVATION, AZIMUTH = np.radians([23.0, -63.0 + 90.0])
MARGIN, N_PIXELS = 0.04, 300
SPHERE_OPACITY, PATH_OPACITY = 0.12, 0.5
CIRCLE_WIDTH, PATH_WIDTH, SHAPE_RADIUS, MASS_RADIUS = 1.0, 1.67, 0.67, 5.0
SHAFT_RADIUS, HEAD_LENGTH, HEAD_RADIUS = 1.17, 8.3, 3.3
VELOCITY = 0.35


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """x, y and z of points, on the last axis."""
    return points.cast(ga.subspace("yz zx xy")).kernel


def scenes(paths: np.ndarray) -> list[Scene]:
    """Each case's unit sphere with its great circles, and the paths of its masses `[times, cases, masses, 3]`, closed,
    as faint strokes, set up once for the masses to be drawn among, all in one view."""
    corners = np.stack(np.meshgrid(*[[-1.0, 1.0]] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    view = solids.view([solids.mv.point(corners)], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    closed = solids.mv.point(np.concatenate([paths, paths[:1]], axis=0))      # [times + 1, cases, masses] Point
    sphere = solids.ball(solids.origin, 1.0)[None, None]
    # The great circles where the planes x, y and z through the centre cut the sphere, the first translucent part,
    # after the shape, each mass's ball, and the arrows' shafts and heads.
    circles = solids.Ink(paths.shape[2] + 3, solids.half_space(solids.axes), solids.everywhere[None],
                         CIRCLE, CIRCLE_WIDTH, 1.0)
    return [Scene(view, np.ones(3), [],
                  [(strokes.polylines([(closed[:, case, mass], colour) for mass, colour in enumerate(COLOURS)],
                                      PATH_WIDTH, view), PATH_OPACITY)],
                  inks=[circles], translucent=[(sphere, SPHERE, SPHERE_OPACITY)])
            for case in range(paths.shape[1])]


def case_image(scene: Scene, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """One case at one moment: the masses at their positions `[masses, 3]`, joined into their shape, with their
    velocities `[masses, 3]`, among the scene's sphere and paths."""
    view = scene.view
    places = solids.mv.point(positions)                                        # [masses] Point
    # Joining the masses makes the reflected shape visible throughout the turn.
    shape = solids.rods(places, places[np.roll(np.arange(len(positions)), -1)], SHAPE_RADIUS * view.pixel)
    # Each mass a ball whose area goes with its mass.
    balls = [solids.ball(places[mass:mass + 1], MASS_RADIUS * np.sqrt(masses[mass]) * view.pixel)
             for mass in range(len(positions))]
    shafts, heads = solids.arrows(places, solids.mv.point(positions + VELOCITY * velocities), SHAFT_RADIUS * view.pixel,
                                  HEAD_LENGTH * view.pixel, HEAD_RADIUS * view.pixel)
    parts = ([(shape, SHAPE)] + [(ball[:, None], colour) for ball, colour in zip(balls, COLOURS)]
             + [(shafts, COLOURS), (heads, COLOURS)])
    return scene.draw(parts, inks=[], translucent=[], meshes=[])


def frames(positions: Point, velocities: Point, masses: np.ndarray, moments: range) -> Iterator[np.ndarray]:
    """The motion and its mirror side by side, at each of the moments."""
    paths, tangents = coordinates(positions), coordinates(velocities)         # [times, cases, masses, 3] each
    cases = scenes(paths)
    for time in moments:
        yield np.concatenate([case_image(scene, paths[time, case], tangents[time, case], masses)
                              for case, scene in enumerate(cases)], axis=1)


def draw_motion(positions: Point, velocities: Point, masses: np.ndarray) -> plt.Figure:
    """The motion and its mirror at the first moment."""
    return image_figure(next(frames(positions, velocities, masses, range(1))))


def animate_motion(positions: Point, velocities: Point, masses: np.ndarray) -> Iterator[np.ndarray]:
    """The motion and its mirror at every moment."""
    return frames(positions, velocities, masses, range(positions.shape[0]))
