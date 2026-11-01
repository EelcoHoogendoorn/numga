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
from examples.util.wgsl import image_figure, simplices, solids
from examples.util.wgsl.scene import Scene

mass_colours = np.array([to_rgb(colour) for colour in ("#df7246", "#238c9b", "#7662a1")])
sphere_colour, circle_colour, shape_colour = (np.array(to_rgb(colour)) for colour in ("#e6ebf0", "#aab4bf", "#75808a"))
# The view, elevation and azimuth in radians, azimuth zero looking along +y, within the mirror plane x, so that the
# reflection shows as the picture flipped left for right; its margin as a share of its larger
# side, and the pixels across each case; how much of what lies behind them the sphere's surfaces and the paths hide;
# in pixels, the widths of the great circles and the paths, the radius of the shape's edges, a unit mass's radius,
# and the arrows' width with the length of the strokes of the chevrons at their tips; and the drawn length of a unit
# velocity.
elevation, azimuth = np.radians([23.0, 0.0])
margin, n_pixels = 0.04, 300
sphere_opacity, path_opacity = 0.12, 0.8
circle_width, path_width, shape_radius, mass_radius = 1.0, 1.67, 0.67, 5.0
arrow_width, chevron_length = 2.34, 9.4
velocity_length = 0.35


# --- plumbing -------------------------------------------------------------------------
def scenes(paths: solids.Point) -> list[Scene]:
    """Each case's unit sphere with its great circles, and the paths of its masses `[times, cases, masses]`, closed,
    as faint strokes, set up once for the masses to be drawn among, all in one view."""
    corners = np.stack(np.meshgrid(*[[-1.0, 1.0]] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    view = solids.view([solids.mv.point(corners)], 0.0, elevation, azimuth, margin, n_pixels)
    sphere = solids.ball(solids.origin, 1.0)[None, None]
    # The great circles where the planes x, y and z through the centre cut the sphere, the first translucent part,
    # after the shape and each mass's ball.
    circles = solids.Ink(paths.shape[2] + 1, solids.half_space(solids.axes), solids.everywhere[None],
                         circle_colour, circle_width, 1.0)
    # Each mass's closed path a loop of edges, each edge in its mass's colour.
    n_times, n_masses = len(paths), paths.shape[2]
    path_loops = simplices.Topology.loops([n_times] * n_masses)
    path_colours = np.repeat(mass_colours, n_times, axis=0)                                 # [edges, 3]
    return [Scene(view, np.ones(3), [],
                  [(simplices.tubes(concatenate([paths[:, case, mass] for mass in range(n_masses)]), path_loops,
                                    path_colours, path_width, view, gloss=solids.gloss), path_opacity)],
                  inks=[circles], translucent=[(sphere, sphere_colour, sphere_opacity)])
            for case in range(paths.shape[1])]


def case_image(scene: Scene, places: solids.Point, velocities: solids.Point, masses: np.ndarray) -> np.ndarray:
    """One case at one moment: the masses at their places `[masses]`, joined into their shape, with their
    velocities `[masses]`, among the scene's sphere and paths."""
    view = scene.view
    # Joining the masses makes the reflected shape visible throughout the turn.
    tails, heads = (simplices.Topology.loops([len(places)]).incidence[1, 0] * places.field()).batch()   # [masses] Point-, Point-
    shape = solids.rods(tails, heads, shape_radius * view.pixel)
    # Each mass a ball whose area goes with its mass.
    balls = [solids.ball(places[mass:mass + 1], mass_radius * np.sqrt(masses[mass]) * view.pixel)
             for mass in range(len(places))]
    # Each velocity an arrow, a tube ending in a chevron.
    tips = places + velocity_length * velocities                                # [masses] Point-
    n_masses = len(places)
    arrows = simplices.tubes(stack((places, tips), axis=-1).reshape(-1), simplices.Topology.chains([2] * n_masses),
                             mass_colours, arrow_width, view, gloss=solids.gloss, chevrons=np.ones(n_masses),
                             chevron=chevron_length)
    parts = [(shape, shape_colour)] + [(ball[:, None], colour) for ball, colour in zip(balls, mass_colours)]
    return scene.draw(parts, inks=[], translucent=[], meshes=[(arrows, 1.0)])


def frames(positions: Point, velocities: Point, masses: np.ndarray, moments: range) -> Iterator[np.ndarray]:
    """The motion and its mirror side by side, at each of the moments."""
    # The projective embedding of the masses' positions and of their velocities, directions at infinity.
    places = solids.origin + solids.context.include_from_algebra(positions)    # [times, cases, masses] Point-
    reaches = solids.context.include_from_algebra(velocities)                  # [times, cases, masses] Point-
    cases = scenes(places)
    for time in moments:
        yield np.concatenate([case_image(scene, places[time, case], reaches[time, case], masses)
                              for case, scene in enumerate(cases)], axis=1)


def draw_motion(positions: Point, velocities: Point, masses: np.ndarray) -> plt.Figure:
    """The motion and its mirror at the first moment."""
    return image_figure(next(frames(positions, velocities, masses, range(1))))


def animate_motion(positions: Point, velocities: Point, masses: np.ndarray) -> Iterator[np.ndarray]:
    """The motion and its mirror at every moment."""
    return frames(positions, velocities, masses, range(positions.shape[0]))
