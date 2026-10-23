"""The electric and magnetic fields of a moving charge, as the lab sees them."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.image import AxesImage

from numga import stack

from examples.util.animation import capture
from examples.electromagnetism.moving_charge import core

electric_range = (0.05, 50.0)
magnetic_limit = 10.0
density_level = 5.0
density_colour = "#ea580c"
# Magnitudes start at white, as the signed fields do at zero.
strength_colours = "Purples"
wave_zone = 4.0
# The speeding-up animations' window about the grid's middle: its height as a share of the grid's, and its width as a
# share of its height. The charge runs across, so its field gathers into an upright band, and the two panels side by
# side make a square.
sweep_zoom, sweep_aspect = 0.85, 0.5
wave_percentile = 99.0
wave_power = 0.4
charge_colour = "#facc15"
# The radiating charge's frames' dots per inch.
wave_dpi = 75
# The planes the lab's time sweeps along each spatial direction, whose pairings with the field are the lab's
# electric field, and the patch spanned by x and y, whose pairing with the field is the magnetic flux across
# the plane.
electric_planes = core.time & stack((core.mv.x.dual(), core.mv.y.dual(), core.mv.z.dual()))   # [3] Antibivector
across_patch = core.mv.x.dual() & core.mv.y.dual()                                  # [] Antibivector


# --- plumbing -------------------------------------------------------------------------
def coordinates(events: core.Antivector) -> np.ndarray:
    """x and y of each event, on the last axis: its pairings with the slices x and y."""
    return (stack((core.mv.x, core.mv.y)) & events[..., None]).to_array()     # [..., 2]


def extent(events: core.Antivector) -> tuple[float, float, float, float]:
    """The box the grid's cells cover: its samples sit at the cell centres, half a cell inside."""
    corners = coordinates(events)
    half_cell = (corners[..., 1, 1, :] - corners[..., 0, 0, :]) / 2
    low, high = corners[..., 0, 0, :] - half_cell, corners[..., -1, -1, :] + half_cell
    return low[..., 0], high[..., 0], low[..., 1], high[..., 1]


def panel(ax: Axes, events: core.Antivector, values: np.ndarray, colours: str, norm) -> AxesImage:
    box = extent(events)
    image = ax.imshow(values, extent=box, origin="lower", cmap=colours, norm=norm, interpolation="bilinear")
    ax.set(xlim=box[:2], ylim=box[2:], xticks=[], yticks=[])
    ax.set_aspect("equal")
    return image


def windowed(axes: list[Axes], events: core.Antivector, zoom: float, aspect: float) -> None:
    """Each of the axes framed on a window about the middle of the grid's box, `zoom` of its height high and `aspect`
    of that wide."""
    left, right, bottom, top = extent(events)
    middle, half = np.array([left + right, bottom + top]) / 2, zoom * (top - bottom) / 2
    for ax in axes:
        ax.set(xlim=(middle[0] - aspect * half, middle[0] + aspect * half), ylim=(middle[1] - half, middle[1] + half))


def electric_strength(field: core.Bivector) -> np.ndarray:
    """The lab's electric field by its length."""
    return np.linalg.norm((field[..., None] & electric_planes).to_array(), axis=-1)


def magnetic_across(field: core.Bivector) -> np.ndarray:
    """The lab's magnetic field across the plane."""
    return (field & across_patch).to_array()


def density(current: core.Antivector) -> np.ndarray:
    """The charge density: the current crossing the lab's time slice."""
    return (core.mv.t & current).to_array()


def fields(events: core.Antivector, field: core.Bivector, zoom: float = sweep_zoom,
           aspect: float = sweep_aspect) -> plt.Figure:
    """Electric field strength and magnetic field side by side for each case, the cases side by side, each panel an
    upright window, `zoom` of the grid's height high and `aspect` of that wide."""
    strengths, magnetic = electric_strength(field), magnetic_across(field)       # [cases, rows, columns] each
    result = plt.figure(figsize=(7 * len(strengths), 7), layout="constrained")
    axes = result.subplots(1, 2 * len(strengths))
    for pair, strength, across in zip(axes.reshape(-1, 2), strengths, magnetic):
        panel(pair[0], events, strength, strength_colours, LogNorm(*electric_range))
        panel(pair[1], events, across, "RdBu_r", TwoSlopeNorm(0.0, -magnetic_limit, magnetic_limit))
    windowed(list(axes), events, zoom, aspect)
    return result


def densities(events: core.Antivector, current: core.Antivector, zoom: float = sweep_zoom,
              aspect: float = sweep_aspect) -> plt.Figure:
    """The charge density, signed, one panel per case, on one colour scale, each panel an upright window, `zoom` of
    the grid's height high and `aspect` of that wide."""
    values = density(current)                                                   # [cases, rows, columns]
    limit = np.abs(values).max()
    result = plt.figure(figsize=(7 * len(values) / 2, 7), layout="constrained")
    axes = result.subplots(1, len(values), squeeze=False)[0]
    for ax, value in zip(axes, values):
        panel(ax, events, value, "RdBu_r", TwoSlopeNorm(0.0, -limit, limit))
    windowed(list(axes), events, zoom, aspect)
    return result


def sweep(events: core.Antivector, field: core.Bivector, current: core.Antivector,
          zoom: float = sweep_zoom, aspect: float = sweep_aspect) -> Iterator[np.ndarray]:
    """Electric field strength with the charge's outline, and magnetic field, one frame per case, each in an upright
    window, `zoom` of the grid's height high and `aspect` of that wide, side by side in a square. One figure is
    updated in place and drawn once per frame, without supersampling, to keep pace."""
    xy = coordinates(events)
    strengths, magnetic, charge_densities = electric_strength(field), magnetic_across(field), density(current)
    result = plt.figure(figsize=(7, 7), dpi=80, layout="constrained")
    axes = result.subplots(1, 2)
    images = (panel(axes[0], events, strengths[0], strength_colours, LogNorm(*electric_range)),
              panel(axes[1], events, magnetic[0], "RdBu_r", TwoSlopeNorm(0.0, -magnetic_limit, magnetic_limit)))
    windowed(axes, events, zoom, aspect)
    for strength, across, value in zip(strengths, magnetic, charge_densities):
        images[0].set_data(strength)
        images[1].set_data(across)
        outline = axes[0].contour(xy[..., 0], xy[..., 1], np.abs(value), levels=[density_level],
                                  colors=[density_colour], linewidths=1.2)
        result.canvas.draw()
        yield np.asarray(result.canvas.buffer_rgba())[..., :3].copy()
        outline.remove()
    plt.close(result)


def animate(frames: Iterable[tuple[core.Antivector, core.Bivector, core.Antivector]],
            zoom: float = sweep_zoom, aspect: float = sweep_aspect) -> Iterator[np.ndarray]:
    """Electric field strength with the charge's outline, and magnetic field, a frame for each of the lab's views
    over one grid of events, each in an upright window, `zoom` of the grid's height high and `aspect` of that wide,
    side by side in a square: one figure set up for the first, its images' data set for each."""
    states = iter(frames)
    events, field, current = next(states)
    xy = coordinates(events)
    figure = plt.figure(figsize=(7, 7), dpi=80, layout="constrained")
    left, right = figure.subplots(1, 2)
    images = (panel(left, events, electric_strength(field), strength_colours, LogNorm(*electric_range)),
              panel(right, events, magnetic_across(field), "RdBu_r", TwoSlopeNorm(0.0, -magnetic_limit, magnetic_limit)))
    windowed([left, right], events, zoom, aspect)
    for _, field, current in chain([(events, field, current)], states):
        images[0].set_data(electric_strength(field))
        images[1].set_data(magnetic_across(field))
        outline = left.contour(xy[..., 0], xy[..., 1], np.abs(density(current)), levels=[density_level],
                               colors=[density_colour], linewidths=1.2)
        yield capture(figure)
        outline.remove()
    plt.close(figure)


def wave_limit(events: core.Antivector, field: core.Bivector) -> np.ndarray:
    """One colour scale per case for a whole turn: the magnetic field times the distance from the
    orbit's centre, far from the orbit; the sharpest crests of a fast charge's beam saturate."""
    distance = np.linalg.norm(coordinates(events), axis=-1)      # [..., rows, columns]
    far = np.where(distance > wave_zone, np.abs(magnetic_across(field) * distance), np.nan)
    return np.nanpercentile(far, wave_percentile, axis=(-2, -1))                       # [...]


def compressed_waves(events: core.Antivector, field: core.Bivector, limit: np.ndarray) -> np.ndarray:
    """The magnetic field across the orbit's plane times the distance from the orbit's centre, so that
    the outgoing waves keep their strength across the picture; a power below one on the magnitude keeps
    a fast charge's narrow pulses and the field between them."""
    xy = coordinates(events)
    scaled = magnetic_across(field) * np.linalg.norm(xy, axis=-1) / limit[..., None, None]
    return np.sign(scaled) * np.abs(scaled) ** wave_power


def animate_waves(frames: Iterable[tuple[core.Antivector, core.Bivector, core.Antivector]]) -> Iterator[np.ndarray]:
    """The waves over one turn, on the first frame's colour scale, the charge marked: one figure set up for the
    first frame, its image's data and the charge's place set for each."""
    frames = iter(frames)
    first = next(frames)
    events, field, charge = first
    limit = wave_limit(events, field)
    figure = plt.figure(figsize=(5.6, 5.6), dpi=wave_dpi, layout="constrained")
    ax = figure.subplots()
    image = panel(ax, events, compressed_waves(events, field, limit), "RdBu_r", TwoSlopeNorm(0.0, -1.0, 1.0))
    position = coordinates(charge)
    marker = ax.scatter(position[..., 0], position[..., 1], s=18, color=charge_colour, edgecolors="#0f172a",
                        linewidths=0.8, zorder=3)
    for events, field, charge in chain([first], frames):
        image.set_data(compressed_waves(events, field, limit))
        marker.set_offsets(coordinates(charge).reshape(-1, 2))
        yield capture(figure)
    plt.close(figure)


def radiation(events: core.Antivector, field: core.Bivector, charge: core.Antivector) -> plt.Figure:
    """The waves of circling charges side by side, one panel per case, each on its own colour scale,
    with the charge marked."""
    values = compressed_waves(events, field, wave_limit(events, field))           # [cases, rows, columns]
    positions = coordinates(charge).reshape(len(values), 2)   # [cases, 2]
    result = plt.figure(figsize=(11, 11 / len(values) + 0.2), layout="constrained")
    for ax, value, position in zip(result.subplots(1, len(values), squeeze=False)[0], values, positions):
        panel(ax, events, value, "RdBu_r", TwoSlopeNorm(0.0, -1.0, 1.0))
        ax.scatter(*position, s=18, color=charge_colour, edgecolors="#0f172a", linewidths=0.8, zorder=3)
    return result
