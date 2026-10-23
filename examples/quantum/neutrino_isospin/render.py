"""Drawing for neutrino flavor oscillations: the isospin vector's path on the flavor sphere with the plane it
turns in, beside the flavor probabilities along the way. The flavor sphere is drawn on the GPU, translucent, its
dotted lines rows of small balls; the probabilities are matplotlib's."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.util.animation import capture
from examples.quantum.neutrino_isospin import core
from examples.util.wgsl import meshes, show, solids

mv, sm = core.mv, solids.mv
# The flavors at the poles, the electron's and the muon's, and their colours.
FLAVORS = stack((core.flavor_e, core.flavor_mu))                             # [2] Vector
flavor_colours = np.array([to_rgb("#0284c7"), to_rgb("#e11d48")])
path_colour, start_colour, end_colour = np.array(to_rgb("#6366f1")), np.array(to_rgb("#10b981")), np.array(to_rgb("#f59e0b"))
plane_colour, axis_colour = np.array(to_rgb("#1e293b")), np.array(to_rgb("#94a3b8"))
sphere_colour, sphere_opacity = np.array(to_rgb("#cbd5e1")), 0.1
grid_colour, grid_opacity = np.array(to_rgb("#94a3b8")), 0.5
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the half width of the square view of
# the sphere, and the pixels across it.
ELEVATION, AZIMUTH = np.radians(35.0), np.radians(100.0)
LIMIT, N_PIXELS = 1.3, 560
# The points around a tube; the widths of the grid's circles, the path and the
# plane's circle, and the radii of the axes' dots, the circle's dots, the flavors' markers and the path's ends, a
# little larger to show over a flavor's, in pixels; the dots along each axis and around the dotted circle; how far
# the axes and the flavors' names reach, in radii of the sphere.
N_AROUND = 8
GRID_WIDTH, PATH_WIDTH, PLANE_WIDTH = 2.0, 4.2, 3.6
AXIS_DOT_RADIUS, CIRCLE_DOT_RADIUS, FLAVOR_RADIUS, END_RADIUS = 1.6, 2.2, 5.0, 6.5
N_AXIS_DOTS, N_CIRCLE_DOTS = 41, 90
AXIS_REACH, LABEL_DISTANCE = 1.2, 1.3


# --- plumbing -------------------------------------------------------------------------
def coordinates(vectors: core.Vector) -> np.ndarray:
    """The x, y and z coordinates of vectors."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def electron_probability(trajectory: core.Vector) -> np.ndarray:
    """The electron flavor's probability along a path of isospin vectors."""
    return (1.0 + coordinates(trajectory)[..., 2]) * 0.5


def flavor_sphere(start_plane: core.Bivector, end_plane: core.Bivector) -> tuple[list, list, list, solids.View]:
    """What stays put on the flavor sphere: its axes dotted, its flavors at the poles, and the plane the vector turns
    in at the start, dotted, as opaque parts; the sphere, translucent; its equator and two meridians, and the plane
    the vector turns in at the end, as great circles to ink on it, each the planes cutting it, their colour, width and
    opacity; and the view."""
    view = solids.View(solids.aim(solids.origin, ELEVATION, AZIMUTH), np.full(2, LIMIT), N_PIXELS)
    along = np.linspace(-AXIS_REACH, AXIS_REACH, N_AXIS_DOTS)[:, None, None] * np.eye(3)    # [dots, 3, 3]
    # The great circle in the plane the vector turns in at the start, square to its dual, every point of it but the
    # last, which closes it onto the first.
    dotted = meshes.circles(np.zeros(3), np.array(1.0), coordinates(start_plane.dual()), N_CIRCLE_DOTS)[:-1]  # [dots, 3]
    parts = [(solids.ball(sm.point(points), radius * view.pixel)[:, None], colour)
             for points, radius, colour in ((along.reshape(-1, 3), AXIS_DOT_RADIUS, axis_colour),
                                            (dotted, CIRCLE_DOT_RADIUS, plane_colour),
                                            (coordinates(FLAVORS), FLAVOR_RADIUS, flavor_colours))]
    ball = [(solids.ball(solids.origin, 1.0)[None, None], sphere_colour, sphere_opacity)]
    # The equator and two meridians, where the planes x, y and z cut the sphere; the plane the vector turns in at the
    # end, through the centre square to its dual.
    end_circle = solids.plane(coordinates(end_plane.dual())[None], np.zeros(1))  # [1] Plane
    circles = [(solids.axes, grid_colour, GRID_WIDTH, grid_opacity), (end_circle, plane_colour, PLANE_WIDTH, 1.0)]
    return parts, ball, circles, view


def draw_sphere(ax, sphere: tuple[list, list, list, solids.View], trajectory: core.Vector) -> None:
    """The flavor sphere and the isospin vector's path over it, from its start, green, to its end, orange, with
    the flavors named at the poles."""
    parts, ball, circles, view = sphere
    path = coordinates(trajectory)                                           # [steps, 3]
    tube = meshes.tube_samples([(path, path_colour)], PATH_WIDTH, view, N_AROUND)
    ends = (solids.ball(sm.point(path[[0, -1]]), END_RADIUS * view.pixel)[:, None], np.stack([start_colour, end_colour]))
    opaque = parts + [ends]
    # The great circles inked on the ball, the first part after the opaque ones.
    inks = [solids.Ink(len(opaque), solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape),
                       colour, width, opacity) for planes, colour, width, opacity in circles]
    show(ax, solids.draw(opaque, view, np.ones(3), inks, translucent=ball, meshes=[(tube, 1.0)]), view)
    # The flavors named just past the poles, where the camera sees them.
    for (x, y), name, colour in zip(view.across(sm.point(LABEL_DISTANCE * coordinates(FLAVORS))),
                                    (r"$\nu_e$", r"$\nu_\mu$"), flavor_colours):
        ax.text(x, y, name, ha="center", va="center", fontsize=13, fontweight="bold", color=colour)


def draw_probabilities(ax, distances: np.ndarray, electron: np.ndarray) -> None:
    """The electron and muon flavors' probabilities along the distance, against zero, a half and one."""
    ax.plot(distances, electron, color="#0284c7", linewidth=2.0, label=r"$P(\nu_e)$")
    ax.plot(distances, 1 - electron, color="#e11d48", linewidth=1.8, linestyle="--", label=r"$P(\nu_\mu)$")
    for level, style in ((0.5, ":"), (1.0, "-")):
        ax.axhline(level, color="#cbd5e1", linestyle=style, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set(xticks=[], yticks=[], ylim=(0, 1.05), xlim=(distances[0], distances[-1]), xlabel="distance",
           ylabel="probability")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)


def panels() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    figure = plt.figure(figsize=(10, 4.5), dpi=120)
    return figure, figure.add_subplot(1, 2, 1), figure.add_subplot(1, 2, 2)


def draw_precession(states: Iterable[core.Vector], plane: core.Bivector, distances: np.ndarray) -> plt.Figure:
    """The isospin vector turning in one plane, and the flavor probabilities oscillating."""
    trajectory = stack(list(states), axis=0)                                 # [steps] Vector
    figure, sphere, probabilities = panels()
    draw_sphere(sphere, flavor_sphere(plane, plane), trajectory)
    draw_probabilities(probabilities, distances, electron_probability(trajectory))
    figure.tight_layout()
    return figure


def draw_decoherence(states: Iterable[core.Vector], plane: core.Bivector, distances: np.ndarray, settled: float) -> plt.Figure:
    """The isospin vector spiralling inward, and the flavor probabilities settling where the dashed line says."""
    trajectory = stack(list(states), axis=0)                                 # [steps] Vector
    figure, sphere, probabilities = panels()
    draw_sphere(sphere, flavor_sphere(plane, plane), trajectory)
    draw_probabilities(probabilities, distances, electron_probability(trajectory))
    probabilities.axhline(settled, color="#64748b", linestyle="--", linewidth=1.2)
    figure.tight_layout()
    return figure


def draw_adiabatic(steps: Iterable[tuple[core.Vector, core.Bivector]], distances: np.ndarray, potentials: np.ndarray) -> plt.Figure:
    """The isospin vector following the plane as it tilts, and the flavor converting as the potential, dotted,
    falls away."""
    pairs = list(steps)
    trajectory = stack([state for state, _ in pairs], axis=0)               # [steps] Vector
    figure, sphere, probabilities = panels()
    draw_sphere(sphere, flavor_sphere(pairs[0][1], pairs[-1][1]), trajectory)
    draw_probabilities(probabilities, distances, electron_probability(trajectory))
    probabilities.plot(distances, potentials / potentials.max(), color="#94a3b8", linestyle=":", linewidth=1.2)
    figure.tight_layout()
    return figure


def animate_precession(states: Iterable[core.Vector], plane: core.Bivector, distances: np.ndarray, n_frames: int) -> list[np.ndarray]:
    """The isospin vector's path and the flavor probabilities growing along the distance, frame by frame."""
    trajectory = stack(list(states), axis=0)                                 # [steps] Vector
    electron = electron_probability(trajectory)
    figure, sphere, probabilities = panels()
    backdrop = flavor_sphere(plane, plane)
    frames = []
    for end in np.linspace(2, len(distances), n_frames, dtype=int):
        sphere.cla()
        probabilities.cla()
        draw_sphere(sphere, backdrop, trajectory[:end])
        draw_probabilities(probabilities, distances[:end], electron[:end])
        probabilities.set_xlim(distances[0], distances[-1])
        frames.append(capture(figure))
    plt.close(figure)
    return frames
