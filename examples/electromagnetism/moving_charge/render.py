"""The electric and magnetic fields of a moving charge, as the lab sees them."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from numga import stack

from examples.util.animation import capture
from examples.util.wgsl import pinned
from examples.electromagnetism.moving_charge import core

# The scales of the electric field's strength, of the magnetic field, and of the waves, compressed to within one.
electric_norm = LogNorm(0.05, 50.0)
magnetic_norm = TwoSlopeNorm(0.0, -10.0, 10.0)
wave_norm = TwoSlopeNorm(0.0, -1.0, 1.0)
density_level = 5.0
density_colour = "#ea580c"
# Magnitudes start at white, as the signed fields do at zero.
strength_colours = plt.get_cmap("Purples")
signed_colours = plt.get_cmap("RdBu_r")
wave_zone = 4.0
wave_percentile = 99.0
wave_power = 0.4
charge_colour, charge_rim_colour = "#facc15", "#0f172a"
# The figures' dots per inch, and the pixels between panels side by side.
figure_dpi = 100
panel_gap = 4
# The width of the charge's outline, in pixels; the charge's dot and its rim, as shares of the picture's width.
outline_width = 0.9
charge_size, charge_rim = 0.0105, 0.002


# --- plumbing -------------------------------------------------------------------------
def coordinates(events: core.Antivector) -> np.ndarray:
    """x and y of each event, on the last axis: its pairings with the slices x and y."""
    mv = events.context.multivector
    return (stack((mv.x, mv.y)) & events[..., None]).to_array()               # [..., 2]


def extent(events: core.Antivector) -> tuple[float, float, float, float]:
    """The box the grid's cells cover: its samples sit at the cell centres, half a cell inside."""
    corners = coordinates(events)
    half_cell = (corners[..., 1, 1, :] - corners[..., 0, 0, :]) / 2
    low, high = corners[..., 0, 0, :] - half_cell, corners[..., -1, -1, :] + half_cell
    return low[..., 0], high[..., 0], low[..., 1], high[..., 1]


def electric_strength(field: core.Bivector) -> np.ndarray:
    """The strength of the electric field the lab reads: the size of the field met with the lab's time, the part of
    the field that the planes swept along the lab's time read."""
    mv = field.context.multivector
    time = -mv.t.dual()                                                         # [] Antivector-
    return (field & time).norm().to_array()


def distances(events: core.Antivector) -> np.ndarray:
    """Each event's distance from the orbit's centre in the lab's space: the size of its place, the event less the
    lab's time as far as the lab's slice reads it."""
    mv = events.context.multivector
    time = -mv.t.dual()                                                         # [] Antivector-
    places = events - time * (mv.t & events)                                    # [...] Antivector-
    return places.squared().square_root().to_array()


def magnetic_across(field: core.Bivector) -> np.ndarray:
    """The lab's magnetic field across the plane: the field read on the patch of the plane of x and y."""
    mv = field.context.multivector
    return (field & mv.xy.dual()).to_array()


def density(current: core.Antivector) -> np.ndarray:
    """The charge density: the current crossing the lab's time slice."""
    mv = current.context.multivector
    return (mv.t & current).to_array()


def coloured(values: np.ndarray, colours: Colormap, norm: Normalize) -> np.ndarray:
    """Values `[..., rows, columns]` in colours, `[..., rows, columns, 3]`, the rows flipped so that the grid's first
    row, lowest in y, is at the bottom of the picture."""
    return colours(np.ma.filled(norm(values), 0.0))[..., ::-1, :, :3]


def panels(n_panels: int, pixels: tuple[int, int]) -> tuple[Figure, list[Axes]]:
    """A figure of panels side by side, each with room for exactly its pixels, rows and columns, a gap between them."""
    rows, columns = pixels
    width = n_panels * columns + (n_panels - 1) * panel_gap
    figure = plt.figure(figsize=(width / figure_dpi, rows / figure_dpi), dpi=figure_dpi)
    axes = figure.subplots(1, n_panels, squeeze=False)[0]
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=panel_gap / columns)
    return figure, list(axes)


def panel(ax: Axes, events: core.Antivector, values: np.ndarray, colours: Colormap, norm: Normalize) -> AxesImage:
    """Values on the grid, one pixel per event, held pixel for pixel over the box the grid covers; the image, for an
    animation to set its data."""
    image = pinned(ax, coloured(values, colours, norm), extent(events))
    ax.set_autoscale_on(False)
    return image


def outline(ax: Axes, events: core.Antivector, current: core.Antivector):
    """The charge's outline, where its density is high."""
    xy = coordinates(events)
    return ax.contour(xy[..., 0], xy[..., 1], np.abs(density(current)), levels=[density_level],
                      colors=[density_colour], linewidths=outline_width * 72 / figure_dpi)


def fields(events: core.Antivector, field: core.Bivector) -> Figure:
    """Electric field strength and magnetic field side by side for each case, the cases side by side."""
    strengths, magnetic = electric_strength(field), magnetic_across(field)       # [cases, rows, columns] each
    figure, axes = panels(2 * len(strengths), events.shape[-2:])
    for left, right, strength, across in zip(axes[::2], axes[1::2], strengths, magnetic):
        panel(left, events, strength, strength_colours, electric_norm)
        panel(right, events, across, signed_colours, magnetic_norm)
    return figure


def densities(events: core.Antivector, current: core.Antivector) -> Figure:
    """The charge density, signed, one panel per case, on one colour scale."""
    values = density(current)                                                   # [cases, rows, columns]
    limit = np.abs(values).max()
    figure, axes = panels(len(values), events.shape[-2:])
    for ax, value in zip(axes, values):
        panel(ax, events, value, signed_colours, TwoSlopeNorm(0.0, -limit, limit))
    return figure


def animate(frames: Iterable[tuple[core.Antivector, core.Bivector, core.Antivector]]) -> Iterator[np.ndarray]:
    """Electric field strength with the charge's outline, and magnetic field, side by side, a frame for each of the
    lab's views over one grid of events: one figure set up for the first, its images' data set for each."""
    states = iter(frames)
    events, field, current = next(states)
    figure, (left, right) = panels(2, events.shape[-2:])
    images = (panel(left, events, electric_strength(field), strength_colours, electric_norm),
              panel(right, events, magnetic_across(field), signed_colours, magnetic_norm))
    for _, field, current in chain([(events, field, current)], states):
        images[0].set_data(coloured(electric_strength(field), strength_colours, electric_norm))
        images[1].set_data(coloured(magnetic_across(field), signed_colours, magnetic_norm))
        charge_outline = outline(left, events, current)
        yield capture(figure)
        charge_outline.remove()
    plt.close(figure)


def wave_limit(events: core.Antivector, field: core.Bivector) -> np.ndarray:
    """One colour scale per case for a whole turn: the magnetic field times the distance from the
    orbit's centre, far from the orbit; the sharpest crests of a fast charge's beam saturate."""
    distance = distances(events)                                                 # [..., rows, columns]
    far = np.where(distance > wave_zone, np.abs(magnetic_across(field) * distance), np.nan)
    return np.nanpercentile(far, wave_percentile, axis=(-2, -1))                       # [...]


def compressed_waves(events: core.Antivector, field: core.Bivector, limit: np.ndarray) -> np.ndarray:
    """The magnetic field across the orbit's plane times the distance from the orbit's centre, so that
    the outgoing waves keep their strength across the picture; a power below one on the magnitude keeps
    a fast charge's narrow pulses and the field between them."""
    scaled = magnetic_across(field) * distances(events) / limit[..., None, None]
    return np.sign(scaled) * np.abs(scaled) ** wave_power


def mark_charge(ax: Axes, columns: int, charge: core.Antivector):
    """The charge as a dot with a dark rim, sized with the picture, the given pixels across."""
    position = coordinates(charge).reshape(-1, 2)
    diameter, rim = charge_size * columns * 72 / figure_dpi, charge_rim * columns * 72 / figure_dpi
    return ax.scatter(position[:, 0], position[:, 1], s=diameter**2, color=charge_colour,
                      edgecolors=charge_rim_colour, linewidths=rim, zorder=3)


def averaged(image: np.ndarray, samples: int) -> np.ndarray:
    """A picture sampled `samples` times along each side of its pixels, `[rows * samples, columns * samples, 3]`,
    averaged back to exactly its pixels."""
    rows, columns = image.shape[0] // samples, image.shape[1] // samples
    return image.reshape(rows, samples, columns, samples, 3).mean(axis=(1, 3))


def animate_waves(frames: Iterable[tuple[core.Antivector, core.Bivector, core.Antivector]], samples: int
                  ) -> Iterator[np.ndarray]:
    """The waves over one turn, on the first frame's colour scale, the charge marked: one figure set up for the
    first frame, its image's data and the charge's place set for each. The events sample each pixel `samples` times
    along each side, and the colours are averaged back to the pixel."""
    frames = iter(frames)
    first = next(frames)
    events, field, charge = first
    limit = wave_limit(events, field)
    pixels = (events.shape[-2] // samples, events.shape[-1] // samples)
    figure, (ax,) = panels(1, pixels)
    image = pinned(ax, averaged(coloured(compressed_waves(events, field, limit), signed_colours, wave_norm), samples),
                   extent(events))
    ax.set_autoscale_on(False)
    marker = mark_charge(ax, pixels[1], charge)
    for events, field, charge in chain([first], frames):
        image.set_data(averaged(coloured(compressed_waves(events, field, limit), signed_colours, wave_norm), samples))
        marker.set_offsets(coordinates(charge).reshape(-1, 2))
        yield capture(figure)
    plt.close(figure)


def radiation(events: core.Antivector, field: core.Bivector, charge: core.Antivector) -> Figure:
    """The waves of circling charges side by side, one panel per case, each on its own colour scale,
    with the charge marked."""
    values = compressed_waves(events, field, wave_limit(events, field))           # [cases, rows, columns]
    figure, axes = panels(len(values), events.shape[-2:])
    for case, (ax, value) in enumerate(zip(axes, values)):
        panel(ax, events, value, signed_colours, wave_norm)
        mark_charge(ax, events.shape[-1], charge[case])
    return figure
