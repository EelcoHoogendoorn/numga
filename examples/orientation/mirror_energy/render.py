"""A mass distribution and its mirror, with tangent velocities on the unit sphere, drawn on the GPU: the sphere
translucent with three great circles inked on it, each mass's path a faint stroke, the masses as balls joined into
their shape, and their velocities as arrows."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack

from examples.orientation.mirror_energy.core import Point
from examples.util.wgsl import image_figure, solids, strokes
from examples.util.wgsl.scene import Scene

mass_colours = np.array([to_rgb(colour) for colour in ("#df7246", "#238c9b", "#7662a1")])
sphere_colour, circle_colour, shape_colour = (np.array(to_rgb(colour)) for colour in ("#e6ebf0", "#aab4bf", "#75808a"))
# The view, elevation and azimuth in radians, azimuth zero looking along +y; its margin as a share of its larger
# side, and the pixels across each case; how much of what lies behind them the sphere's surfaces and the paths hide;
# in pixels, the widths of the great circles and the paths, the radius of the shape's edges, a unit mass's radius,
# and the arrows' shafts, heads' length and heads' radius; and the drawn length of a unit velocity.
elevation, azimuth = np.radians([23.0, -63.0 + 90.0])
margin, n_pixels = 0.04, 300
sphere_opacity, path_opacity = 0.12, 0.5
circle_width, path_width, shape_radius, mass_radius = 1.0, 1.67, 0.67, 5.0
shaft_radius, head_length, head_radius = 1.17, 8.3, 3.3
velocity_length = 0.35


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """x, y and z of points, on the last axis, read by pairing with the planes x, y and z."""
    mv = points.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array()


def gpu_points(points: Point) -> solids.Point:
    """The GPU library's own points at the places of the sphere's points: their coordinates, read by pairing, are the
    one hand-off from this algebra to the renderer's."""
    return solids.mv.point(coordinates(points))


def scenes(paths: Point) -> list[Scene]:
    """Each case's unit sphere with its great circles, and the paths of its masses `[times, cases, masses]`, closed,
    as faint strokes, set up once for the masses to be drawn among, all in one view."""
    corners = np.stack(np.meshgrid(*[[-1.0, 1.0]] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    view = solids.view([solids.mv.point(corners)], 0.0, elevation, azimuth, margin, n_pixels)
    closed = gpu_points(concatenate([paths, paths[:1]]))                       # [times + 1, cases, masses] Point-
    sphere = solids.ball(solids.origin, 1.0)[None, None]
    # The great circles where the planes x, y and z through the centre cut the sphere, the first translucent part,
    # after the shape, each mass's ball, and the arrows' shafts and heads.
    circles = solids.Ink(paths.shape[2] + 3, solids.half_space(solids.axes), solids.everywhere[None],
                         circle_colour, circle_width, 1.0)
    return [Scene(view, np.ones(3), [],
                  [(strokes.polylines([(closed[:, case, mass], colour) for mass, colour in enumerate(mass_colours)],
                                      path_width, view), path_opacity)],
                  inks=[circles], translucent=[(sphere, sphere_colour, sphere_opacity)])
            for case in range(paths.shape[1])]


def case_image(scene: Scene, positions: Point, velocities: Point, masses: np.ndarray) -> np.ndarray:
    """One case at one moment: the masses at their positions `[masses]`, joined into their shape, with their
    velocities `[masses]`, among the scene's sphere and paths."""
    view = scene.view
    places = gpu_points(positions)                                             # [masses] Point-
    # Joining the masses makes the reflected shape visible throughout the turn.
    shape = solids.rods(places, places[np.roll(np.arange(len(positions)), -1)], shape_radius * view.pixel)
    # Each mass a ball whose area goes with its mass.
    balls = [solids.ball(places[mass:mass + 1], mass_radius * np.sqrt(masses[mass]) * view.pixel)
             for mass in range(len(positions))]
    tips = gpu_points(positions + velocity_length * velocities)               # [masses] Point-
    shafts, heads = solids.arrows(places, tips, shaft_radius * view.pixel, head_length * view.pixel,
                                  head_radius * view.pixel)
    parts = ([(shape, shape_colour)] + [(ball[:, None], colour) for ball, colour in zip(balls, mass_colours)]
             + [(shafts, mass_colours), (heads, mass_colours)])
    return scene.draw(parts, inks=[], translucent=[], meshes=[])


def frames(positions: Point, velocities: Point, masses: np.ndarray, moments: range) -> Iterator[np.ndarray]:
    """The motion and its mirror side by side, at each of the moments."""
    cases = scenes(positions)
    for time in moments:
        yield np.concatenate([case_image(scene, positions[time, case], velocities[time, case], masses)
                              for case, scene in enumerate(cases)], axis=1)


def draw_motion(positions: Point, velocities: Point, masses: np.ndarray) -> plt.Figure:
    """The motion and its mirror at the first moment."""
    return image_figure(next(frames(positions, velocities, masses, range(1))))


def animate_motion(positions: Point, velocities: Point, masses: np.ndarray) -> Iterator[np.ndarray]:
    """The motion and its mirror at every moment."""
    return frames(positions, velocities, masses, range(positions.shape[0]))
