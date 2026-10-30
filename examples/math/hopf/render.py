"""Drawing for the Hopf fibration: the spins on the sphere beside their fibres in space, each spin and
its fibre in the same colour. Both are drawn on the GPU: the sphere a translucent ball with a few of its
circles inked on it, the spins balls on it, the fibres strokes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_rgb
from matplotlib.image import AxesImage

from numga import concatenate, stack
from examples.util.animation import capture
from examples.math.hopf import core
from examples.util.wgsl import fitted, meshes, show, solids, strokes
from examples.util.wgsl.scene import Scene

sm = solids.mv
# Projected fibres are drawn out to this distance from the origin; the fibres near the projection
# point run off towards infinity.
limit = 2.6
# The sphere's colour and how much each of its surfaces hides, and its circles' colour and how much they hide.
sphere_colour, sphere_opacity = np.full(3, 0.92), 0.13
circle_colour, circle_opacity = np.full(3, 0.72), 0.5
loop_colour, fibre_colour, start_colour = (np.array(to_rgb(colour)) for colour in ("#c0392b", "#999999", "#2e86c1"))
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the half width of the sphere's square
# view, and the margin around the fibres as a share of the larger side.
elevation, azimuth = np.radians(22.0), np.radians(35.0)
sphere_reach, margin = 1.12, 0.12
# The places of space, on the right, and of the small sphere over its left edge, as shares of the figure.
space_place, sphere_place = (0.24, 0.0, 0.76, 1.0), (0.0, 0.2, 0.28, 0.6)
# The view of the swept fibres: the point in space it is centred on, and its half width and height, holding every
# fibre of the spiral from near the south pole to just above the equator.
sweep_middle, sweep_half = np.array([0.15, 0.08, -0.06]), np.array([2.71, 1.71])
# The sphere's parallels at these polar angles and its meridians at these azimuths.
parallel_polars = np.pi * np.arange(1, 6) / 6
meridian_azimuths = np.pi * np.arange(6) / 6
# The pixels across and up the figures of the tori and of the lift, and of the sweep's frames.
figure_pixels, sweep_pixels = (495, 270), (450, 275)
# Widths in pixels: the sphere's circles, the fibres of the tori, the swept fibres and the newest of them, the lifted
# spinor's track and the fibre it starts on; and the radii of the spins' dots and of the track's ends.
circle_width, torus_width, swept_width, newest_width, track_width, lift_fibre_width = 0.8, 1.1, 0.8, 1.9, 1.6, 1.3
dot_radius, end_radius = 1.5, 2.2
# The sweep's smaller sphere: the width of its circles and the radius of its dots, in pixels.
sweep_circle_width, sweep_dot_radius = 0.7, 1.3


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Antivector) -> np.ndarray:
    """Points, spins on the sphere or projected fibres in space, across x, y and z `[..., 3]`: their pairings
    with the planes x, y and z."""
    mv = points.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array()


def colour(spins: core.Antivector) -> np.ndarray:
    """An RGB colour for each spin: its azimuth as the hue, darker towards the south pole."""
    x, y, z = np.moveaxis(coordinates(spins), -1, 0)
    hue = (np.arctan2(y, x) / (2 * np.pi)) % 1.0
    return hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.75), 0.55 + 0.4 * (1 + z) / 2], axis=-1))


def visible(curves: core.Antivector, colours: np.ndarray) -> list[tuple[solids.Point, np.ndarray]]:
    """Polylines `[curves, n]`, each with its colour `[curves, 3]`, cut where they pass beyond the drawing limit:
    the runs of two or more points inside it `[run]`, each with its curve's colour."""
    runs = []
    for curve, rgb in zip(curves, colours):
        inside = curve.norm() <= limit                                         # [n]
        bounds = np.concatenate([[0], np.flatnonzero(np.diff(inside)) + 1, [len(inside)]])
        places = solids.origin + solids.context.include_from_algebra(curve)                     # [n] Point-
        runs += [(places[start:stop], rgb) for start, stop in zip(bounds[:-1], bounds[1:])
                 if inside[start] and stop - start > 1]
    return runs


def space_view(runs: list[tuple[solids.Point, np.ndarray]]) -> solids.View:
    """A view framing every run drawn in space."""
    return solids.view([concatenate([run for run, _ in runs])], 0.0, elevation, azimuth, margin, 1)


def sweep_view() -> solids.View:
    """The view of the swept fibres in space."""
    return solids.View(solids.aim(sm.point(sweep_middle), elevation, azimuth), sweep_half, 1)


def sphere_view() -> solids.View:
    """A square view of the sphere of spins."""
    return solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, sphere_reach), 1)


def sphere_surface(n_parts: int,
                   width: float) -> tuple[list[solids.Ink], list[tuple[solids.Quadric, np.ndarray, float]]]:
    """The sphere of spins, a translucent ball after `n_parts` opaque parts, with its parallels and meridians inked
    on it where their planes cut it, a width in pixels: its inks and its translucent part."""
    ball = [(solids.ball(solids.origin, 1.0)[None, None], sphere_colour, sphere_opacity)]
    # The parallels: the planes through the pole turned down by each polar angle, along the directions x and y; the
    # meridians: the plane x turned about z by each azimuth.
    rims = (sm.zx * (-parallel_polars / 2)).exp() >> (solids.origin + sm.z.dual())   # [parallels] Point-
    parallels = (rims & sm.x.dual() & sm.y.dual()) * solids.flip               # [parallels] Plane+
    meridians = (sm.xy * (-meridian_azimuths / 2)).exp() >> sm.x             # [meridians] Plane+
    planes = concatenate([parallels, meridians])                             # [circles] Plane+
    # The ball is the first translucent part, after the opaque ones.
    circles = solids.Ink(n_parts, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape),
                         circle_colour, width, circle_opacity)
    return [circles], ball


def sphere_image(parts: list[tuple[solids.Quadric, np.ndarray]], lines: list[tuple[meshes.Samples, float]],
                 view: solids.View) -> np.ndarray:
    """The sphere of spins, as `sphere_surface`, among opaque parts and lines, each its samples and opacity,
    `[rows, columns, 3]`."""
    inks, ball = sphere_surface(len(parts), circle_width)
    return solids.draw(parts, view, background, inks, translucent=ball, meshes=lines)


def dots(points: core.Antivector, colours: np.ndarray, radius: float,
         view: solids.View) -> tuple[solids.Quadric, np.ndarray]:
    """A dot at each spin on the sphere `[n]`, in its colour `[n, 3]`, of a radius in pixels, as an opaque part."""
    return solids.ball(solids.origin + solids.context.include_from_algebra(points), radius * view.pixel)[:, None], colours


def segments(runs: list[tuple[solids.Point, np.ndarray]]) -> tuple[solids.Point, solids.Point, np.ndarray]:
    """Polylines, each with its colour, as their segments: the starts and the ends `[segments]`, and the colours
    `[segments, 3]`, none for no polylines."""
    none = sm.point(np.zeros((0, 3)))                                          # [0] Point-
    return (concatenate([none] + [run[:-1] for run, _ in runs]), concatenate([none] + [run[1:] for run, _ in runs]),
            np.concatenate([np.zeros((0, 3))] + [np.broadcast_to(rgb, (len(run) - 1, 3)) for run, rgb in runs]))


def panels(space: solids.View, pixels: tuple[int, int]) -> tuple[plt.Figure, list[plt.Axes], list[solids.View]]:
    """A figure of pixels across and up with space on the right and, over its left edge, a small sphere of spins: the
    figure, the axes of space and of the sphere, and the views of space and of the sphere fitted to them."""
    figure = plt.figure(figsize=np.array(pixels) / 100, dpi=100)
    axes = [figure.add_axes(place) for place in (space_place, sphere_place)]
    return figure, axes, [fitted(view, ax) for view, ax in zip((space, sphere_view()), axes)]


def shown(axes: list[plt.Axes], space_image: np.ndarray, sphere_image: np.ndarray,
          views: list[solids.View]) -> list[AxesImage]:
    """The images of space and of the sphere in the axes `panels` gives: their images, for an animation to set their
    data."""
    return [show(ax, image, view) for ax, image, view in zip(axes, (space_image, sphere_image), views)]


def draw_tori(spins: core.Antivector, projected: core.Antivector) -> plt.Figure:
    """The fibres over circles of spins: each circle's fibres fill a torus, the tori nested, and any two
    fibres linked once."""
    points = spins.reshape(-1)                                                 # [spins] Antivector-
    curves = projected.reshape((-1, projected.shape[-1]))                      # [spins, n] Antivector-
    colours = colour(points)
    runs = visible(curves, colours)
    figure, axes, (space, sphere) = panels(space_view(runs), figure_pixels)
    fibres = strokes.polylines(runs, torus_width, space)
    shown(axes, solids.draw([], space, background, inks=[], translucent=[], meshes=[(fibres, 1.0)]),
          sphere_image([dots(points, colours, dot_radius, sphere)], [], sphere), (space, sphere))
    return figure


def draw_lift(loop: core.Antivector, path: core.Antivector, fibre: core.Antivector) -> plt.Figure:
    """A loop of spins on the sphere, and in space the spinor carried around it: it leaves its
    fibre, drawn grey, and comes back onto it further along."""
    tracks, circles = visible(path[None], loop_colour[None]), visible(fibre[None], fibre_colour[None])
    figure, axes, (space, sphere) = panels(space_view(tracks + circles), figure_pixels)
    lines = [(strokes.polylines(runs, width, space), 1.0)
             for runs, width in ((circles, lift_fibre_width), (tracks, track_width))]
    # The track's start in blue, its end in its own colour.
    track_ends = solids.origin + solids.context.include_from_algebra(path[[0, -1]])            # [2] Point-
    ends = (solids.ball(track_ends, end_radius * space.pixel)[:, None], np.stack([start_colour, loop_colour]))
    on_sphere = strokes.polylines([(solids.origin + solids.context.include_from_algebra(loop), loop_colour)], track_width, sphere)
    shown(axes, solids.draw([ends], space, background, inks=[], translucent=[],
                            meshes=lines), sphere_image([], [(on_sphere, 1.0)], sphere),
          (space, sphere))
    return figure


def animate_sweep(stream: Iterable[tuple[core.Antivector, core.Antivector]]) -> Iterator[np.ndarray]:
    """Fibres added one by one as their spin spirals over the sphere, the newest drawn heavier; one frame
    per spin, in a fixed view that holds the fibres of the spiral."""
    # The figure laid out once, its panels blank until each frame sets their images, and the sphere and its circles
    # set up once, among each frame's dots.
    figure, axes, (space, sphere) = panels(sweep_view(), sweep_pixels)
    space_shown, sphere_shown = shown(axes, np.ones((*space.shape, 3)), np.ones((*sphere.shape, 3)), (space, sphere))
    inks, ball = sphere_surface(1, sweep_circle_width)
    sphere_scene = Scene(sphere, background, [], [], inks=inks, translucent=ball)
    space_scene = Scene(space, background, [], [], inks=[], translucent=[])
    # The spins so far, their colours, and each one's fibre as segments of its runs inside the drawing limit.
    points, colours, fibres = [], [], []
    for spin, projected in stream:
        points.append(spin)                                                    # [] Antivector-
        colours.append(colour(spin))                                           # [3]
        fibres.append(segments(visible(projected[None], colours[-1][None])))
        # Every fibre so far thin, and the newest thick around its thin self.
        starts, ends, shades = zip(*fibres)
        thin = strokes.segments(concatenate(starts), concatenate(ends), np.concatenate(shades), swept_width, space)
        thick = strokes.segments(*fibres[-1], newest_width, space)
        sphere_shown.set_data(sphere_scene.draw([dots(stack(points), np.stack(colours), sweep_dot_radius, sphere)],
                                                inks=[], translucent=[], meshes=[]))
        space_shown.set_data(space_scene.draw([], inks=[], translucent=[], meshes=[(thin, 1.0), (thick, 1.0)]))
        yield capture(figure)
    plt.close(figure)
