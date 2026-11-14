"""Flavor-sphere trajectories and survival-probability charts for neutrino flavor oscillations.

`solids` renders the translucent flavor sphere with dotted coordinate axes, pure-flavor pole markers,
and Hamiltonian great circles on the GPU, while matplotlib plots the electron and muon flavor
probabilities alongside it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.util.animation import capture
from examples.quantum.neutrino_isospin import core
from examples.util import charts
from numga.topology import Topology
from numga.render import Scene, simplices, solids
from examples.util.panels import fitted, show
from numga.render.drawing import Draw

mv = core.mv
# Multiplying an intrinsic isospin patch by the intrinsic unit volume yields its extrinsic plane, whose dual is the
# intrinsic spatial direction pointing to its pole on the flavor sphere.
unit_volume = mv.antipseudoscalar([1])                                       # [] Antipseudoscalar-
direction_of_isospin = (core.Isospin * unit_volume).dual()                   # [] Direction- <- Isospin-
flavors = stack((core.flavor_e, core.flavor_mu))                             # [2] Isospin-
flavor_directions = solids.context.include_from_algebra(direction_of_isospin(flavors))   # [2] Direction-
flavor_colours = np.array([to_rgb("#0284c7"), to_rgb("#e11d48")])
path_colour, start_colour, end_colour = np.array(to_rgb("#6366f1")), np.array(to_rgb("#10b981")), np.array(to_rgb("#f59e0b"))
plane_colour, axis_colour = np.array(to_rgb("#1e293b")), np.array(to_rgb("#94a3b8"))
sphere_colour, sphere_opacity = np.array(to_rgb("#cbd5e1")), 0.1
grid_colour, grid_opacity = np.array(to_rgb("#94a3b8")), 0.5
elevation, azimuth = np.radians(35.0), np.radians(100.0)
view_limit, n_pixels = 1.3, 300
grid_width, path_width, plane_width = 0.0093, 0.0195, 0.0167
axis_dot_radius, circle_dot_radius, flavor_radius, end_radius = 0.0074, 0.0102, 0.0232, 0.0302
n_axis_dots, n_circle_dots = 41, 90
axis_reach, label_distance = 1.2, 1.3
panels_px = (charts.DESIGN_WIDTH, 360)


# --- plumbing -------------------------------------------------------------------------
def electron_probability(trajectory: core.Isospin) -> np.ndarray:
    """Electron-flavor survival probability along `trajectory` (`[...] Isospin-`) as a NumPy array."""
    electron, _ = core.flavor_probabilities(trajectory)
    return electron.to_array()


def flavor_sphere(start_plane: core.Bivector, end_plane: core.Bivector, ax: plt.Axes
                  ) -> tuple[list, list, list, solids.View]:
    """Static flavor-sphere scene elements: dotted axes, pole markers, dotted initial great circle, translucent
    unit ball, inked coordinate and final Hamiltonian circles, and fitted `solids.View`."""
    view = fitted(solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, view_limit), n_pixels), ax)
    reaches = np.linspace(-axis_reach, axis_reach, n_axis_dots)[:, None]
    axis_dots = solids.origin + solids.axes.dual() * reaches                 # [n_axis_dots, 3] Point-
    # Projecting `flavor_e` into `start_plane` gives a starting point on the initial Hamiltonian's great circle.
    start_unit = start_plane.normalized()                                    # [] Bivector+
    in_plane = core.flavor_e - start_unit.dual() * (start_unit & core.flavor_e)   # [] Isospin-
    angles = np.linspace(0, 2 * np.pi, n_circle_dots, endpoint=False)
    turns = (start_unit * (-angles / 2)).exp()                               # [n_circle_dots] Rotor+
    circle_dots = turns >> in_plane.normalized()                             # [n_circle_dots] Isospin-
    circle_places = solids.origin + solids.context.include_from_algebra(direction_of_isospin(circle_dots))   # [n_circle_dots] Point-
    parts = [(solids.ball(points, radius)[:, None], colour)
             for points, radius, colour in ((axis_dots.reshape(-1), axis_dot_radius, axis_colour),
                                            (circle_places, circle_dot_radius, plane_colour),
                                            (solids.origin + flavor_directions, flavor_radius, flavor_colours))]
    ball = [(solids.ball(solids.origin, 1.0)[None, None], sphere_colour, sphere_opacity)]
    end_circle = solids.context.include_from_algebra(end_plane.dual()[None] * unit_volume)   # [1] Plane+
    circles = [(solids.axes, grid_colour, grid_width / view.pixel, grid_opacity),
               (end_circle, plane_colour, plane_width / view.pixel, 1.0)]
    return parts, ball, circles, view


def circle_inks(circles: list, n_opaque: int) -> list[solids.Ink]:
    """Great-circle ink strokes bound to the translucent sphere after `n_opaque` opaque quadric groups."""
    return [solids.Ink(n_opaque, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape),
                       colour, width, opacity) for planes, colour, width, opacity in circles]


def path_marks(path: core.Isospin, view: solids.View) -> tuple[Draw, tuple[solids.Quadric, np.ndarray]]:
    """Tube stroke and green/orange endpoint balls marking `path` (`[...] Isospin-`) on the unit sphere."""
    places = solids.origin + solids.context.include_from_algebra(direction_of_isospin(path))   # [...] Point-
    stroke = simplices.tubes(places, Topology.chains([len(places)]), path_colour, path_width / view.pixel, view,
                             gloss=solids.gloss)
    ends = (solids.ball(places[np.array([0, -1])], end_radius)[:, None], np.stack([start_colour, end_colour]))
    return stroke, ends


def flavor_labels(ax: plt.Axes, view: solids.View) -> None:
    """Places pole labels just beyond the electron and muon flavor markers in `ax`."""
    for (x, y), name, colour in zip(view.across(solids.origin + flavor_directions * label_distance),
                                    ("\u03bd\u2091", "\u03bd\u03bc"), flavor_colours):
        ax.text(x, y, name, ha="center", va="center", fontweight="bold", color=colour)


def sphere_panel(ax: plt.Axes, sphere: tuple[list, list, list, solids.View], trajectory: core.Isospin) -> None:
    """Renders the flavor sphere and `trajectory` (`[...] Isospin-`) into `ax`."""
    parts, ball, circles, view = sphere
    stroke, ends = path_marks(trajectory, view)
    opaque = parts + [ends]
    show(ax, solids.draw(opaque, view, np.ones(3), circle_inks(circles, len(opaque)), translucent=ball,
                         meshes=[(stroke, 1.0)]), view)
    flavor_labels(ax, view)


def probability_curves(ax: plt.Axes, distances: np.ndarray, electron: np.ndarray) -> tuple[plt.Line2D, plt.Line2D]:
    """Plots electron and muon flavor probabilities against `distances` in `ax`."""
    electron_line, = ax.plot(distances, electron, color=flavor_colours[0])
    muon_line, = ax.plot(distances, 1 - electron, color=flavor_colours[1], linestyle="--")
    for level, style in ((0.5, ":"), (1.0, "-")):
        ax.axhline(level, color="#cbd5e1", linestyle=style, linewidth=charts.FRAME_WIDTH)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set(xticks=[], yticks=[], ylim=(0, 1.05), xlim=(distances[0], distances[-1]), xlabel="distance",
           ylabel="probability")
    return electron_line, muon_line


def panels() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Two-panel figure pairing a bare sphere subplot with a probability chart subplot."""
    figure = plt.figure(figsize=(panels_px[0] / charts.DPI, panels_px[1] / charts.DPI))
    sphere = figure.add_subplot(1, 2, 1)
    sphere.set_axis_off()
    return figure, sphere, figure.add_subplot(1, 2, 2)


@charts.style()
def precession_figure(states: Iterable[core.Isospin], plane: core.Bivector, distances: np.ndarray) -> plt.Figure:
    """Two-panel figure of coherent precession in `plane` alongside oscillating flavor probabilities."""
    trajectory = stack(list(states), axis=0)                                 # [...] Isospin-
    figure, sphere, probabilities = panels()
    probability_curves(probabilities, distances, electron_probability(trajectory))
    figure.tight_layout()
    sphere_panel(sphere, flavor_sphere(plane, plane, sphere), trajectory)
    return figure


@charts.style()
def decoherence_figure(states: Iterable[core.Isospin], plane: core.Bivector, distances: np.ndarray,
                       asymptotic_probability: float) -> plt.Figure:
    """Two-panel figure of inward isospin spiralling and damping toward `asymptotic_probability`."""
    trajectory = stack(list(states), axis=0)                                 # [...] Isospin-
    figure, sphere, probabilities = panels()
    probability_curves(probabilities, distances, electron_probability(trajectory))
    probabilities.axhline(asymptotic_probability, color="#64748b", linestyle="--")
    figure.tight_layout()
    sphere_panel(sphere, flavor_sphere(plane, plane, sphere), trajectory)
    return figure


@charts.style()
def adiabatic_figure(steps: Iterable[tuple[core.Isospin, core.Bivector]], distances: np.ndarray,
                     potentials: np.ndarray) -> plt.Figure:
    """Two-panel figure of adiabatic conversion as the Hamiltonian plane tilts with falling `potentials`."""
    pairs = list(steps)
    trajectory = stack([isospin for isospin, _ in pairs], axis=0)            # [...] Isospin-
    figure, sphere, probabilities = panels()
    probability_curves(probabilities, distances, electron_probability(trajectory))
    probabilities.plot(distances, potentials / potentials.max(), color="#94a3b8", linestyle=":")
    figure.tight_layout()
    sphere_panel(sphere, flavor_sphere(pairs[0][1], pairs[-1][1], sphere), trajectory)
    return figure


def precession_animation(plane: core.Bivector, distances: np.ndarray, n_frames: int,
                         states: Iterable[core.Isospin]) -> Iterator[np.ndarray]:
    """Yields `n_frames` RGB frames growing the flavor-sphere path and survival probabilities as `states`
    streams in."""
    with charts.style():
        figure, sphere, probabilities = panels()
        electron_line, muon_line = probability_curves(probabilities, distances, np.full(len(distances), np.nan))
        figure.tight_layout()
        parts, ball, circles, view = flavor_sphere(plane, plane, sphere)
        scene = Scene(view, np.ones(3), parts, [], inks=circle_inks(circles, len(parts) + 1), translucent=ball)
        image = show(sphere, np.full((*view.shape, 3), 255, dtype=np.uint8), view)
        flavor_labels(sphere, view)
        stream = iter(states)
        path, electron = [], []
        for end in np.linspace(2, len(distances), n_frames, dtype=int):
            for isospin in islice(stream, end - len(path)):
                path.append(isospin)
                electron.append(electron_probability(isospin))
            stroke, ends = path_marks(stack(path), view)
            image.set_data(scene.draw([ends], inks=[], translucent=[], meshes=[(stroke, 1.0)]))
            electron_line.set_data(distances[:end], electron)
            muon_line.set_data(distances[:end], 1 - np.array(electron))
            yield capture(figure)
        plt.close(figure)


def decoherence_animation(plane: core.Bivector, distances: np.ndarray, asymptotic_probability: float, n_frames: int,
                          states: Iterable[core.Isospin]) -> Iterator[np.ndarray]:
    """Yields `n_frames` RGB frames of wavepacket decoherence spiralling inward toward `asymptotic_probability`."""
    with charts.style():
        figure, sphere, probabilities = panels()
        electron_line, muon_line = probability_curves(probabilities, distances, np.full(len(distances), np.nan))
        probabilities.axhline(asymptotic_probability, color="#64748b", linestyle="--")
        figure.tight_layout()
        parts, ball, circles, view = flavor_sphere(plane, plane, sphere)
        scene = Scene(view, np.ones(3), parts, [], inks=circle_inks(circles, len(parts) + 1), translucent=ball)
        image = show(sphere, np.full((*view.shape, 3), 255, dtype=np.uint8), view)
        flavor_labels(sphere, view)
        stream = iter(states)
        path, electron = [], []
        for end in np.linspace(2, len(distances), n_frames, dtype=int):
            for isospin in islice(stream, end - len(path)):
                path.append(isospin)
                electron.append(electron_probability(isospin))
            stroke, ends = path_marks(stack(path), view)
            image.set_data(scene.draw([ends], inks=[], translucent=[], meshes=[(stroke, 1.0)]))
            electron_line.set_data(distances[:end], electron)
            muon_line.set_data(distances[:end], 1 - np.array(electron))
            yield capture(figure)
        plt.close(figure)


def adiabatic_animation(distances: np.ndarray, potentials: np.ndarray, start_plane: core.Bivector,
                        end_plane: core.Bivector, n_frames: int,
                        steps: Iterable[tuple[core.Isospin, core.Bivector]]) -> Iterator[np.ndarray]:
    """Yields `n_frames` RGB frames of solar adiabatic flavor conversion as the local Hamiltonian plane tilts."""
    with charts.style():
        figure, sphere, probabilities = panels()
        electron_line, muon_line = probability_curves(probabilities, distances, np.full(len(distances), np.nan))
        probabilities.plot(distances, potentials / potentials.max(), color="#94a3b8", linestyle=":")
        figure.tight_layout()
        parts, ball, _, view = flavor_sphere(start_plane, end_plane, sphere)
        grid_circles = [(solids.axes, grid_colour, grid_width / view.pixel, grid_opacity)]
        scene = Scene(view, np.ones(3), parts, [], inks=circle_inks(grid_circles, len(parts) + 1), translucent=ball)
        image = show(sphere, np.full((*view.shape, 3), 255, dtype=np.uint8), view)
        flavor_labels(sphere, view)
        stream = iter(steps)
        path, electron = [], []
        current_plane = start_plane
        for end in np.linspace(2, len(distances), n_frames, dtype=int):
            for isospin, current_plane in islice(stream, end - len(path)):
                path.append(isospin)
                electron.append(electron_probability(isospin))
            stroke, ends = path_marks(stack(path), view)
            active_circle = solids.context.include_from_algebra(current_plane.dual()[None] * unit_volume)   # [1] Plane+
            plane_ink = circle_inks([(active_circle, plane_colour, plane_width / view.pixel, 1.0)], len(parts) + 1)
            image.set_data(scene.draw([ends], inks=plane_ink, translucent=[], meshes=[(stroke, 1.0)]))
            electron_line.set_data(distances[:end], electron)
            muon_line.set_data(distances[:end], 1 - np.array(electron))
            yield capture(figure)
        plt.close(figure)
