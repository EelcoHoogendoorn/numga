"""Drawing for the Hopf fibration: the spins on the sphere beside their fibres in space, each spin and
its fibre in the same colour. Both are drawn on the GPU: the sphere a translucent ball with a few of its
circles inked on it, the spins balls on it, the fibres strokes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_rgb
from matplotlib.image import AxesImage

from numga import concatenate
from examples.util.animation import capture
from examples.math.hopf import core
from examples.util.wgsl import meshes, show, solids, strokes
from examples.util.wgsl.scene import Scene

sm = solids.mv
# Projected fibres are drawn out to this distance from the origin; the fibres near the projection
# point run off towards infinity.
LIMIT = 2.6
# The sphere's colour and how much each of its surfaces hides, and its circles' colour and how much they hide.
SPHERE_COLOUR, SPHERE_OPACITY = np.full(3, 0.92), 0.13
CIRCLE_COLOUR, CIRCLE_OPACITY = np.full(3, 0.72), 0.5
LOOP_COLOUR, FIBRE_COLOUR, START_COLOUR = (np.array(to_rgb(colour)) for colour in ("#c0392b", "#999999", "#2e86c1"))
BACKGROUND = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the half width of the sphere's square
# view, the margin around the fibres as a share of the larger side, and the pixels across the sphere and across space.
ELEVATION, AZIMUTH = np.radians(22.0), np.radians(35.0)
SPHERE_REACH, MARGIN = 1.12, 0.12
SPHERE_PIXELS, SPACE_PIXELS = 320, 840
# The view of the swept fibres: the point in space it is centred on, and its half width and height, holding every
# fibre of the spiral from near the south pole to just above the equator.
SWEEP_MIDDLE, SWEEP_HALF = np.array([0.15, 0.08, -0.06]), np.array([2.71, 1.71])
# The sphere's parallels at these polar angles and its meridians at these azimuths.
PARALLELS = np.pi * np.arange(1, 6) / 6
MERIDIANS = np.pi * np.arange(6) / 6
# Widths in pixels: the sphere's circles, the fibres of the tori, the swept fibres and the newest of them, the lifted
# spinor's track and the fibre it starts on; and the radii of the spins' dots and of the track's ends.
CIRCLE_WIDTH, TORUS_WIDTH, SWEPT_WIDTH, NEWEST_WIDTH, TRACK_WIDTH, LIFT_FIBRE_WIDTH = 1.6, 2.4, 1.8, 4.5, 3.4, 2.8
DOT_RADIUS, END_RADIUS = 3.0, 4.5


# --- plumbing -------------------------------------------------------------------------
def components(bivectors: core.Bivector) -> np.ndarray:
    """The yz, zx and xy components of bivectors, drawn along x, y and z."""
    return bivectors.cast(core.ga.subspace("yz zx xy")).kernel


def colour(spins: core.Bivector) -> np.ndarray:
    """An RGB colour for each spin: its azimuth as the hue, darker towards the south pole."""
    x, y, z = np.moveaxis(components(spins), -1, 0)
    hue = (np.arctan2(y, x) / (2 * np.pi)) % 1.0
    return hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.75), 0.55 + 0.4 * (1 + z) / 2], axis=-1))


def visible(curves: np.ndarray, colours: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Polylines `[curves, n, 3]`, each with its colour `[curves, 3]`, cut where they pass beyond the drawing limit:
    the runs of two or more points inside it, each with its curve's colour."""
    runs = []
    for curve, rgb in zip(curves, colours):
        inside = np.linalg.norm(curve, axis=-1) <= LIMIT
        cuts = np.flatnonzero(np.diff(inside)) + 1
        runs += [(run, rgb) for run, kept in zip(np.split(curve, cuts), np.split(inside, cuts)) if kept[0] and len(run) > 1]
    return runs


def space_view(runs: list[tuple[np.ndarray, np.ndarray]]) -> solids.View:
    """A view framing every run drawn in space."""
    return solids.view([sm.point(np.concatenate([run for run, _ in runs]))], 0.0, ELEVATION, AZIMUTH, MARGIN,
                       SPACE_PIXELS)


def sweep_view() -> solids.View:
    """The view of the swept fibres in space."""
    return solids.View(solids.aim(sm.point(SWEEP_MIDDLE), ELEVATION, AZIMUTH), SWEEP_HALF, SPACE_PIXELS)


def sphere_view() -> solids.View:
    """A square view of the sphere of spins."""
    return solids.View(solids.aim(solids.origin, ELEVATION, AZIMUTH), np.full(2, SPHERE_REACH), SPHERE_PIXELS)


def sphere_surface(n_parts: int) -> tuple[list[solids.Ink], list[tuple[solids.Quadric, np.ndarray, float]]]:
    """The sphere of spins, a translucent ball after `n_parts` opaque parts, with its parallels and meridians inked
    on it where their planes cut it: its inks and its translucent part."""
    ball = [(solids.ball(solids.origin, 1.0)[None, None], SPHERE_COLOUR, SPHERE_OPACITY)]
    # The parallels: the planes across z at the height of each polar angle; the meridians: the plane x turned about
    # z by each azimuth.
    meridians = (sm.xy * (-MERIDIANS / 2)).exp() >> sm.x                     # [meridians] Plane
    planes = concatenate([sm.z - sm.w * np.cos(PARALLELS), meridians])        # [circles] Plane
    # The ball is the first translucent part, after the opaque ones.
    circles = solids.Ink(n_parts, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape),
                         CIRCLE_COLOUR, CIRCLE_WIDTH, CIRCLE_OPACITY)
    return [circles], ball


def sphere_image(parts: list[tuple[solids.Quadric, np.ndarray]], lines: list[tuple[meshes.Samples, float]],
                 view: solids.View) -> np.ndarray:
    """The sphere of spins, as `sphere_surface`, among opaque parts and lines, each its samples and opacity,
    `[rows, columns, 3]`."""
    inks, ball = sphere_surface(len(parts))
    return solids.draw(parts, view, BACKGROUND, inks, translucent=ball, meshes=lines)


def dots(points: np.ndarray, colours: np.ndarray, view: solids.View) -> tuple[solids.Quadric, np.ndarray]:
    """A dot at each spin on the sphere `[n, 3]`, in its colour `[n, 3]`, as an opaque part."""
    return solids.ball(sm.point(points), DOT_RADIUS * view.pixel)[:, None], colours


def segments(runs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polylines, each with its colour, as their segments: the starts, the ends and the colours, `[segments, 3]`
    each, none for no polylines."""
    none = [np.zeros((0, 3))]
    return (np.concatenate(none + [run[:-1] for run, _ in runs]), np.concatenate(none + [run[1:] for run, _ in runs]),
            np.concatenate(none + [np.broadcast_to(rgb, (len(run) - 1, 3)) for run, rgb in runs]))


def panels(figure: plt.Figure, sphere_image: np.ndarray, space_image: np.ndarray,
           space: solids.View) -> list[AxesImage]:
    """Space on the right and, over its left edge, a small sphere of spins: their images, for an animation to set
    their data."""
    return [show(figure.add_axes(place), image, view)
            for place, image, view in (((0.24, 0.0, 0.76, 1.0), space_image, space),
                                       ((0.0, 0.2, 0.28, 0.6), sphere_image, sphere_view()))]


def draw_tori(spins: core.Bivector, projected: core.Bivector) -> plt.Figure:
    """The fibres over circles of spins: each circle's fibres fill a torus, the tori nested, and any two
    fibres linked once."""
    points, curves, colours = components(spins), components(projected), colour(spins)
    points, curves, colours = points.reshape(-1, 3), curves.reshape(-1, curves.shape[-2], 3), colours.reshape(-1, 3)
    runs = visible(curves, colours)
    space, sphere = space_view(runs), sphere_view()
    fibres = strokes.polylines(runs, TORUS_WIDTH, space)
    figure = plt.figure(figsize=(11, 6))
    panels(figure, sphere_image([dots(points, colours, sphere)], [], sphere),
           solids.draw([], space, BACKGROUND, meshes=[(fibres, 1.0)]), space)
    return figure


def draw_lift(loop: core.Bivector, path: core.Bivector, fibre: core.Bivector) -> plt.Figure:
    """A loop of spins on the sphere, and in space the spinor carried around it: it leaves its
    fibre, drawn grey, and comes back onto it further along."""
    track, circle = components(path), components(fibre)
    tracks, circles = visible(track[None], LOOP_COLOUR[None]), visible(circle[None], FIBRE_COLOUR[None])
    space, sphere = space_view(tracks + circles), sphere_view()
    lines = [(strokes.polylines(runs, width, space), 1.0)
             for runs, width in ((circles, LIFT_FIBRE_WIDTH), (tracks, TRACK_WIDTH))]
    # The track's start in blue, its end in its own colour.
    ends = (solids.ball(sm.point(track[[0, -1]]), END_RADIUS * space.pixel)[:, None], np.stack([START_COLOUR, LOOP_COLOUR]))
    on_sphere = strokes.polylines([(components(loop), LOOP_COLOUR)], TRACK_WIDTH, sphere)
    figure = plt.figure(figsize=(11, 6))
    panels(figure, sphere_image([], [(on_sphere, 1.0)], sphere), solids.draw([ends], space, BACKGROUND, meshes=lines),
           space)
    return figure


def animate_sweep(stream: Iterable[tuple[core.Bivector, core.Bivector]]) -> Iterator[np.ndarray]:
    """Fibres added one by one as their spin spirals over the sphere, the newest drawn heavier; one frame
    per spin, in a fixed view that holds the fibres of the spiral."""
    space, sphere = sweep_view(), sphere_view()
    # The sphere and its circles set up once, among each frame's dots, and the figure laid out once, its panels blank
    # until each frame sets their images.
    inks, ball = sphere_surface(1)
    sphere_scene = Scene(sphere, BACKGROUND, [], [], inks=inks, translucent=ball)
    space_scene = Scene(space, BACKGROUND, [], [])
    figure = plt.figure(figsize=(9, 5.5))
    space_shown, sphere_shown = panels(figure, np.ones((1, 1, 3)), np.ones((1, 1, 3)), space)
    # The spins so far, their colours, and each one's fibre as segments of its runs inside the drawing limit.
    points, colours, fibres = [], [], []
    for spin, projected in stream:
        points.append(components(spin))                                        # [3]
        colours.append(colour(spin))                                           # [3]
        fibres.append(segments(visible(components(projected)[None], colours[-1][None])))
        # Every fibre so far thin, and the newest thick around its thin self.
        thin = strokes.segments(*(np.concatenate(part) for part in zip(*fibres)), SWEPT_WIDTH, space)
        thick = strokes.segments(*fibres[-1], NEWEST_WIDTH, space)
        sphere_shown.set_data(sphere_scene.draw([dots(np.stack(points), np.stack(colours), sphere)]))
        space_shown.set_data(space_scene.draw(meshes=[(thin, 1.0), (thick, 1.0)]))
        yield capture(figure)
    plt.close(figure)
