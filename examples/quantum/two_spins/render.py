"""Rendering helpers draw two spins on their spheres of turning planes, Bell bounds, steering, field levels, and the singlet-triplet qubit."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor, stack
from numga.render import Scene, simplices, solids
from numga.render.drawing import Draw
from numga.topology import Topology
from examples.quantum.two_spins import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import fitted, labels, show

first_colour, second_colour, together_colour = "#c0392b", "#2e86c1", "#7d3c98"
four_names = ("singlet", "triplet, zero spin", "triplet, both up", "triplet, both down")
sphere_colour, sphere_opacity = np.array([0.70, 0.76, 0.86]), 0.09
circle_colour, circle_opacity = np.full(3, 0.6), 0.6
label_colour, start_colour = "0.35", np.full(3, 0.1)
elevation, azimuth = np.radians(18.0), np.radians(-55.0 + 90.0)
reach, n_pixels = 1.3, 300
circle_width, path_width, arrow_width = 0.00891, 0.026, 0.0223
dot_radius, landing_radius, start_radius = 0.0186, 0.0334, 0.026
chevron_length = 4.0


class Reading(NamedTuple):
    """Three unit basis elements `[3]` per spin copy and a magnitude label specify how a spin is plotted."""
    first: Extensor
    second: Extensor
    size: str


plane_reading = Reading(
    stack((core.mv.yz, core.mv.zx, core.mv.xy)),
    stack((core.mv.YZ, core.mv.ZX, core.mv.XY)),
    "size of each spin's plane",
)


def coordinates(spins: Extensor, units: Extensor) -> np.ndarray:
    """Dot products against `units` (`[3]`) extract three Euclidean components (`[..., 3]`) from `spins`."""
    return units.dot(spins[..., None]).to_array()


def gpu_points(spins: Extensor, units: Extensor) -> solids.Point:
    """Basis readings along `units` place `spins` as 3D scene points (`[...] Point-`) around the sphere origin."""
    readings = units.dot(spins[..., None])                                     # [..., 3] Scalar+
    return solids.origin + (solids.axes.dual() * solids.context.include_from_algebra(readings)).sum(axis=-1)   # [...] Point-


def sizes(spins: Extensor) -> np.ndarray:
    """The square root of the scalar norm squared gives the Euclidean magnitude (`[...]`) of `spins`."""
    return spins.scalar_norm_squared().square_root().to_array()


def plain_axes(ax: plt.Axes, x_label: str, y_label: str) -> None:
    """Minimal chart spines place the horizontal axis at zero without numeric tick marks."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.set(xticks=[], yticks=[], xlabel=x_label, ylabel=y_label)
    ax.xaxis.set_label_coords(0.5, -0.03)
    ax.xaxis.label.set_va("top")


def bell_bounds(ax: plt.Axes, angles: np.ndarray) -> None:
    """Horizontal reference lines mark the local bound at 2 (dashed) and the Tsirelson bound at `2 * sqrt(2)` (dotted)."""
    ax.axhline(2.0, color="0.5", linestyle="--", linewidth=charts.FRAME_WIDTH)
    ax.axhline(2 * np.sqrt(2), color="0.5", linestyle=":", linewidth=charts.FRAME_WIDTH)
    ax.set(xlim=(angles[0], angles[-1]), ylim=(0.0, 3.0))
    plain_axes(ax, "exchange angle", "largest Bell combination")
    ax.set(yticks=[2.0, 2 * np.sqrt(2)], yticklabels=["2", r"$2\sqrt{2}$"])


def ball(ax: plt.Axes) -> solids.View:
    """A square orthographic camera view frames the unit sphere inside `ax`."""
    return fitted(solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, reach), n_pixels), ax)


def sphere_scene(view: solids.View, n_parts: int) -> Scene:
    """A translucent unit sphere with three inked coordinate great circles follows `n_parts` opaque solids."""
    sphere = [(solids.ball(solids.origin, 1.0)[None, None], sphere_colour, sphere_opacity)]
    circles = solids.Ink(
        n_parts,
        solids.half_space(solids.axes),
        solids.everywhere[None].broadcast_to((3,)),
        circle_colour,
        circle_width / view.pixel,
        circle_opacity,
    )
    return Scene(view, np.ones(3), [], [], inks=[circles], translucent=sphere)


def on_sphere(
    parts: list[tuple[solids.Quadric, np.ndarray]],
    view: solids.View,
    lines: list[tuple[Draw, float]],
) -> np.ndarray:
    """Opaque `parts` and tube `lines` composite inside the translucent unit sphere."""
    return sphere_scene(view, len(parts)).draw(parts, inks=[], translucent=[], meshes=lines)


def arrow(tip: solids.Point, colour: str, view: solids.View) -> Draw:
    """A chevron-tipped 3D tube arrow runs from the sphere origin to `tip`."""
    width = arrow_width / view.pixel
    return simplices.tubes(
        stack((solids.origin, tip)),
        Topology.chains([2]),
        np.array(to_rgb(colour)),
        width,
        view,
        gloss=solids.gloss,
        chevrons=np.ones(1),
        chevron=chevron_length * width,
    )


def dots(centres: solids.Point, radii: np.ndarray) -> solids.Quadric:
    """Spherical quadric markers (`[..., 1] Plane+ <- Point-`) of `radii` sit at `centres`."""
    shapes = solids.ellipsoid(radii[:, None] * np.ones(3))                     # [...] Plane+ <- Point-
    return solids.placed(shapes, solids.translator(centres - solids.origin))[:, None]


def draw_pair(tips: list[tuple[solids.Point, str]]) -> plt.Figure:
    """Two side-by-side unit spheres display one spin arrow per panel."""
    figure = plt.figure(figsize=(6.0, 3.0))
    for column, (tip, colour) in enumerate(tips):
        ax = figure.add_axes((column / 2, 0.0, 0.5, 1.0))
        view = ball(ax)
        show(ax, on_sphere([], view, [(arrow(tip, colour, view), 1.0)]), view)
    return figure


def draw_state(first_spin: Extensor, second_spin: Extensor, reading: Reading) -> plt.Figure:
    """Two side-by-side spheres show `first_spin` on the left and `second_spin` on the right."""
    return draw_pair([
        (gpu_points(first_spin, reading.first), first_colour),
        (gpu_points(second_spin, reading.second), second_colour),
    ])


@charts.style()
def draw_exchange(angles: np.ndarray, bells: Extensor, first_spin: Extensor, reading: Reading) -> plt.Figure:
    """Two panels plot single-spin magnitude and the largest Bell combination across exchange `angles`."""
    figure, (size_ax, bell_ax) = charts.subplots(640, 260, ncols=2, layout="constrained")
    size_ax.plot(angles, sizes(first_spin), color=first_colour)
    size_ax.set(xlim=(angles[0], angles[-1]), ylim=(0.0, 1.05))
    plain_axes(size_ax, "exchange angle", reading.size)
    bell_ax.plot(angles, bells.to_array(), color=together_colour)
    bell_bounds(bell_ax, angles)
    return figure


def steering_frames(
    points: solids.Point,
    colours: np.ndarray,
    targets: solids.Point,
    weights: np.ndarray,
) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames of first-spin measurement points and steered second-spin landing dots."""
    figure = plt.figure(figsize=(7.5, 3.75), dpi=80)
    measured_ax, landed_ax = (figure.add_axes((column / 2, 0.0, 0.5, 1.0)) for column in range(2))
    view = ball(measured_ax)
    show(measured_ax, on_sphere([(solids.ball(points, dot_radius)[:, None], colours)], view, []), view)

    landing = sphere_scene(view, 1)

    def sphere_frame(target: solids.Point, weight: np.ndarray) -> np.ndarray:
        return landing.draw(
            [(dots(target, landing_radius * np.sqrt(weight)), colours)],
            inks=[],
            translucent=[],
            meshes=[],
        )

    image = show(landed_ax, sphere_frame(targets[0], weights[0]), view)
    for target, weight in zip(targets, weights):
        image.set_data(sphere_frame(target, weight))
        yield capture(figure)
    plt.close(figure)


def animate_steering(
    sample_planes: Extensor,
    probability: Extensor,
    steered_planes: Extensor,
    reading: Reading,
) -> Iterator[np.ndarray]:
    """Yields frames showing first-spin measurement settings and the steered second-spin states sized by probability."""
    points = gpu_points(sample_planes, reading.first).reshape(-1)              # [...] Point-
    n_frames, n_samples = len(steered_planes), len(points)
    colours = np.clip((coordinates(sample_planes, reading.first).reshape(n_samples, 3) + 1) / 2, 0.0, 1.0)   # [n_samples, 3]
    targets = gpu_points(steered_planes, reading.second).reshape(n_frames, n_samples)   # [n_frames, n_samples] Point-
    weights = probability.to_array().reshape(n_frames, n_samples)              # [n_frames, n_samples]
    return steering_frames(points, colours, targets, weights)


@charts.style()
def draw_entanglement(angles: np.ndarray, singular_values: Extensor, first_spin: Extensor, reading: Reading) -> plt.Figure:
    """Three curves compare the joint spin map's singular values against the first spin's magnitude over `angles`."""
    values = singular_values.to_array()                                        # [..., 3]
    figure, ax = charts.subplots(480, 300, layout="constrained")
    ax.plot(angles, values[:, 0], color=together_colour, label="largest singular value")
    ax.plot(angles, values[:, 1], color=together_colour, linestyle="--", label="the other two: the concurrence")
    ax.plot(angles, sizes(first_spin), color=first_colour, label=reading.size)
    ax.set(xlim=(angles[0], angles[-1]), ylim=(0.0, 1.05))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
    plain_axes(ax, "exchange angle", "")
    return figure


@charts.style()
def draw_levels(fields: np.ndarray, energies: Extensor, spins: Extensor) -> plt.Figure:
    """Two panels show the four linear eigenstate energy levels and the ground-state spin across `fields`."""
    values = energies.to_array()                                               # [..., 4]
    lowest_index = energies.argmin(axis=-1)                                    # [...]
    figure, (levels_ax, spin_ax) = charts.subplots(680, 320, ncols=2, layout="constrained")
    for column, (name, colour) in enumerate(zip(four_names, (together_colour, "0.55", first_colour, second_colour))):
        levels_ax.plot(fields, values[:, column], color=colour, label=name)
    levels_ax.plot(fields, values.min(axis=-1), color="0.1", linewidth=charts.points(3 * charts.LINE_PX), alpha=0.25)
    levels_ax.set(xlim=(fields[0], fields[-1]))
    figure.legend(*levels_ax.get_legend_handles_labels(), frameon=False, loc="outside lower center", ncol=len(four_names))
    plain_axes(levels_ax, "field", "energy")
    spin_ax.plot(fields, spins.to_array()[lowest_index], color="0.2")
    spin_ax.set(xlim=(fields[0], fields[-1]), ylim=(-0.1, 2.1))
    plain_axes(spin_ax, "field", "total spin, lowest state")
    return figure


@charts.style()
def draw_qubit(
    imbalance: Extensor,
    phase_coordinate: Extensor,
    singlet_balance: Extensor,
    times: np.ndarray,
    singlet_return: Extensor,
    exchange_rates: np.ndarray,
) -> plt.Figure:
    """A 3D sphere and companion chart display precession paths and singlet return probabilities across `exchange_rates`."""
    n_rates, n_times = imbalance.shape
    readings = stack((imbalance, phase_coordinate, singlet_balance), axis=-1)  # [n_rates, n_times, 3] Scalar+
    paths = solids.origin + (solids.axes.dual() * solids.context.include_from_algebra(readings)).sum(axis=-1)   # [n_rates, n_times] Point-
    shades = (together_colour, first_colour, second_colour)
    figure = plt.figure(figsize=(760 / charts.DPI, 320 / charts.DPI))
    ax = figure.add_axes((0.0, 0.0, 0.45, 1.0))
    view = ball(ax)
    tracks = simplices.tubes(
        paths.reshape(-1),
        Topology.chains([n_times] * n_rates),
        np.repeat([to_rgb(colour) for colour in shades[:n_rates]], n_times - 1, axis=0),
        path_width / view.pixel,
        view,
        solids.gloss,
    )
    start = [(solids.ball(paths[0, 0], start_radius)[None, None], start_colour)]
    show(ax, on_sphere(start, view, [(tracks, 1.0)]), view)
    reaches = stack((
        solids.mv.z.dual() * 1.15,
        solids.mv.z.dual() * -1.2,
        solids.mv.x.dual() * 1.25,
        solids.mv.x.dual() * -1.25,
    ))                                                                         # [4] Antivector-
    labels(
        ax,
        view,
        solids.origin + reaches,
        ["singlet", "triplet", "down-up", "up-down"],
        ha="center",
        va="center",
        color=label_colour,
    )
    curves = figure.add_axes((0.53, 0.18, 0.44, 0.7))
    for rate, curve, colour in zip(exchange_rates, singlet_return.to_array(), shades):
        curves.plot(times, curve, color=colour, label=f"exchange rate {rate:g}")
    curves.set(xlim=(times[0], times[-1]), ylim=(-0.05, 1.05))
    curves.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=3)
    plain_axes(curves, "time", "back in the singlet")
    return figure
