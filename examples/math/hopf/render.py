"""GPU rendering of the Hopf fibration, pairing unit spins on an inset translucent sphere with their projected fibres in space."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_rgb
from matplotlib.image import AxesImage

from examples.math.hopf import core
from examples.util.animation import capture
from examples.util.panels import fitted, show
from numga import concatenate, stack
from numga.render import Scene, simplices, solids
from numga.render.drawing import Draw
from numga.topology import Topology

# Clip projected fibres where they run off toward infinity near the projection pole -1.
limit = 2.6
sphere_colour, sphere_opacity = np.full(3, 0.92), 0.13
circle_colour, circle_opacity = np.full(3, 0.72), 0.5
loop_colour, fibre_colour, start_colour = (np.array(to_rgb(colour)) for colour in ("#c0392b", "#999999", "#2e86c1"))
background = np.ones(3)
elevation, azimuth = np.radians(22.0), np.radians(35.0)
sphere_reach, margin = 1.12, 0.12
space_place, sphere_place = (0.24, 0.0, 0.76, 1.0), (0.0, 0.2, 0.28, 0.6)
sweep_middle, sweep_half = np.array([0.15, 0.08, -0.06]), np.array([2.71, 1.71])
parallel_polars = np.pi * np.arange(1, 6) / 6
meridian_azimuths = np.pi * np.arange(6) / 6
figure_pixels, sweep_pixels = (495, 270), (450, 275)
circle_width, torus_width, swept_width, newest_width, track_width, lift_fibre_width = 0.8, 1.1, 0.8, 1.9, 1.6, 1.3
dot_radius, end_radius = 1.5, 2.2
sweep_circle_width, sweep_dot_radius = 0.7, 1.3


def spin_colours(spins: core.Antivector) -> np.ndarray:
    """Each spin's azimuth and height determine an RGB colour (`[..., 3]`), shaded toward the south pole."""
    basis_mv = spins.context.multivector
    coords = (stack((basis_mv.x, basis_mv.y, basis_mv.z)) & spins[..., None]).to_array()
    hue = (np.arctan2(coords[..., 1], coords[..., 0]) / (2 * np.pi)) % 1.0
    return hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.75), 0.55 + 0.4 * (1 + coords[..., 2]) / 2], axis=-1))


def clipped_runs(curves: core.Antivector, colours: np.ndarray) -> list[tuple[solids.Point, np.ndarray]]:
    """Polylines clipped at `limit` split into interior runs of two or more `Point-` vertices paired with each curve's colour."""
    runs = []
    for curve, rgb in zip(curves, colours):
        inside = curve.antinorm() <= limit
        bounds = np.concatenate([[0], np.flatnonzero(np.diff(inside)) + 1, [len(inside)]])
        places = solids.origin + solids.context.include_from_algebra(curve)    # [...] Point-
        runs += [(places[start:stop], rgb) for start, stop in zip(bounds[:-1], bounds[1:])
                 if inside[start] and stop - start > 1]
    return runs


def space_view(runs: list[tuple[solids.Point, np.ndarray]]) -> solids.View:
    """An elevation-and-azimuth camera view frames all clipped polyline runs in three-dimensional space."""
    return solids.view([concatenate([run for run, _ in runs])], 0.0, elevation, azimuth, margin, 1)


def sweep_view() -> solids.View:
    """A fixed camera view frames every fibre swept across the spiral of spins."""
    return solids.View(solids.aim(solids.mv.point(sweep_middle), elevation, azimuth), sweep_half, 1)


def sphere_view() -> solids.View:
    """A square camera view frames the unit sphere of spins."""
    return solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, sphere_reach), 1)


def sphere_surface(n_parts: int,
                   width: float) -> tuple[list[solids.Ink], list[tuple[solids.Quadric, np.ndarray, float]]]:
    """Five horizontal parallels and six vertical meridians ink a translucent unit sphere placed after `n_parts` opaque parts."""
    ball = [(solids.ball(solids.origin, 1.0)[None, None], sphere_colour, sphere_opacity)]
    # Cut parallels with horizontal planes at tilted pole heights and meridians with azimuth-turned vertical planes.
    parallel_rotors = (solids.mv.zx * (-parallel_polars / 2)).exp()            # [5] Rotor+
    rims = parallel_rotors >> (solids.origin + solids.mv.z.dual())             # [5] Point-
    parallels = solids.mv.z - solids.mv.w * (solids.mv.z & rims)               # [5] Plane+
    meridian_rotors = (solids.mv.xy * (-meridian_azimuths / 2)).exp()          # [6] Rotor+
    meridians = meridian_rotors >> solids.mv.x                                 # [6] Plane+
    planes = concatenate([parallels, meridians])                               # [11] Plane+
    circles = solids.Ink(n_parts, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape),
                         circle_colour, width, circle_opacity)
    return [circles], ball


def sphere_image(parts: list[tuple[solids.Quadric, np.ndarray]], lines: list[tuple[Draw, float]],
                 view: solids.View) -> np.ndarray:
    """The translucent spin sphere renders alongside opaque `parts` and tube `lines`."""
    inks, ball = sphere_surface(len(parts), circle_width)
    return solids.draw(parts, view, background, inks, translucent=ball, meshes=lines)


def spin_dots(points: core.Antivector, colours: np.ndarray, radius: float,
              view: solids.View) -> tuple[solids.Quadric, np.ndarray]:
    """Small opaque sphere quadrics mark `points` (`[...] Antivector-`) with pixel `radius` and RGB `colours`."""
    centers = solids.origin + solids.context.include_from_algebra(points)      # [...] Point-
    return solids.ball(centers, radius * view.pixel)[:, None], colours


def polyline_chains(runs: list[tuple[solids.Point, np.ndarray]]) -> tuple[solids.Point, Topology, np.ndarray]:
    """Coloured polyline `runs` concatenate into one `Point-` batch, a chain `Topology`, and an edge colour array."""
    empty_points = solids.mv.point(np.zeros((0, 3)))                           # [0] Point-
    return (concatenate([empty_points] + [run for run, _ in runs]),
            Topology.chains([0] + [len(run) for run, _ in runs]),
            np.concatenate([np.zeros((0, 3))] + [np.broadcast_to(rgb, (len(run) - 1, 3)) for run, rgb in runs]))


def figure_panels(space: solids.View, pixels: tuple[int, int]) -> tuple[plt.Figure, list[plt.Axes], list[solids.View]]:
    """A two-panel figure places projected three-dimensional space on the right and the inset spin sphere on the left."""
    figure = plt.figure(figsize=np.array(pixels) / 100, dpi=100)
    axes = [figure.add_axes(place) for place in (space_place, sphere_place)]
    return figure, axes, [fitted(view, ax) for view, ax in zip((space, sphere_view()), axes)]


def panel_images(axes: list[plt.Axes], space_raster: np.ndarray, sphere_raster: np.ndarray,
                 views: list[solids.View]) -> list[AxesImage]:
    """Raster images of space and the spin sphere populate `axes` as `AxesImage` artists."""
    return [show(ax, image, view) for ax, image, view in zip(axes, (space_raster, sphere_raster), views)]


def tori_figure(spins: core.Antivector, projected_fibres: core.Antivector) -> plt.Figure:
    """Nested Hopf tori of `projected_fibres` appear beside their latitude circles of `spins` on the inset sphere."""
    points = spins.reshape(-1)                                                 # [...] Antivector-
    curves = projected_fibres.reshape((-1, projected_fibres.shape[-1]))        # [...] Antivector-
    colours = spin_colours(points)
    runs = clipped_runs(curves, colours)
    figure, axes, (space, sphere) = figure_panels(space_view(runs), figure_pixels)
    fibres = simplices.tubes(*polyline_chains(runs), torus_width, space, gloss=solids.gloss)
    panel_images(axes, solids.draw([], space, background, inks=[], translucent=[], meshes=[(fibres, 1.0)]),
                 sphere_image([spin_dots(points, colours, dot_radius, sphere)], [], sphere), (space, sphere))
    return figure


def lift_figure(loop: core.Antivector, lifted_path: core.Antivector, start_fibre: core.Antivector) -> plt.Figure:
    """A closed `loop` of spins on the inset sphere accompanies `lifted_path` departing and returning to `start_fibre`."""
    tracks = clipped_runs(lifted_path[None], loop_colour[None])
    circles = clipped_runs(start_fibre[None], fibre_colour[None])
    figure, axes, (space, sphere) = figure_panels(space_view(tracks + circles), figure_pixels)
    lines = [(simplices.tubes(*polyline_chains(runs), width, space, gloss=solids.gloss), 1.0)
             for runs, width in ((circles, lift_fibre_width), (tracks, track_width))]
    # Mark the initial point on the fibre in blue and the holonomy-shifted endpoint in red.
    track_ends = solids.origin + solids.context.include_from_algebra(lifted_path[[0, -1]])  # [2] Point-
    ends = (solids.ball(track_ends, end_radius * space.pixel)[:, None], np.stack([start_colour, loop_colour]))
    loop_places = solids.origin + solids.context.include_from_algebra(loop)    # [...] Point-
    sphere_loop = simplices.tubes(*polyline_chains([(loop_places, loop_colour)]),
                                  track_width, sphere, gloss=solids.gloss)
    panel_images(axes, solids.draw([ends], space, background, inks=[], translucent=[], meshes=lines),
                 sphere_image([], [(sphere_loop, 1.0)], sphere), (space, sphere))
    return figure


def sweep_animation(stream: Iterable[tuple[core.Antivector, core.Antivector]]) -> Iterator[np.ndarray]:
    """Yields RGB frames accumulating projected fibres one by one as their spin spirals across the sphere."""
    figure, axes, (space, sphere) = figure_panels(sweep_view(), sweep_pixels)
    space_artist, sphere_artist = panel_images(
        axes, np.ones((*space.shape, 3)), np.ones((*sphere.shape, 3)), (space, sphere)
    )
    inks, ball = sphere_surface(1, sweep_circle_width)
    sphere_scene = Scene(sphere, background, [], [], inks=inks, translucent=ball)
    space_scene = Scene(space, background, [], [], inks=[], translucent=[])
    points, colours, fibres = [], [], []
    for spin, projected_fibre in stream:
        points.append(spin)                                                    # [] Antivector-
        colours.append(spin_colours(spin))
        fibres.append(clipped_runs(projected_fibre[None], colours[-1][None]))
        # Draw accumulated fibres thin and highlight the newest fibre with a thicker stroke.
        thin_tubes = simplices.tubes(*polyline_chains([run for runs in fibres for run in runs]), swept_width, space,
                                     gloss=solids.gloss)
        thick_tubes = simplices.tubes(*polyline_chains(fibres[-1]), newest_width, space, gloss=solids.gloss)
        sphere_artist.set_data(sphere_scene.draw([spin_dots(stack(points), np.stack(colours), sweep_dot_radius, sphere)],
                                                 inks=[], translucent=[], meshes=[]))
        space_artist.set_data(space_scene.draw([], inks=[], translucent=[],
                                               meshes=[(thin_tubes, 1.0), (thick_tubes, 1.0)]))
        yield capture(figure)
    plt.close(figure)
