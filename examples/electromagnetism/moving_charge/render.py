"""The electric and magnetic fields of a moving charge, as the lab sees them."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.image import AxesImage

from examples.animation import capture
from examples.electromagnetism.moving_charge import core

ELECTRIC_RANGE = (0.05, 50.0)
MAGNETIC_LIMIT = 10.0
DENSITY_LEVEL = 5.0
DENSITY_COLOUR = "#ea580c"
# Magnitudes start at white, as the signed fields do at zero.
STRENGTH_COLOURS = "Purples"
WAVE_ZONE = 4.0
WAVE_PERCENTILE = 99.0
WAVE_POWER = 0.4
CHARGE_COLOUR = "#facc15"
# The radiating charge's frames' dots per inch.
WAVE_DPI = 36


# --- plumbing -------------------------------------------------------------------------
def extent(events: core.Vector) -> tuple[float, float, float, float]:
    """The box the grid's cells cover: its samples sit at the cell centres, half a cell inside."""
    coordinates = events.cast(core.ga.subspace("x y")).kernel
    half_cell = (coordinates[..., 1, 1, :] - coordinates[..., 0, 0, :]) / 2
    low, high = coordinates[..., 0, 0, :] - half_cell, coordinates[..., -1, -1, :] + half_cell
    return low[..., 0], high[..., 0], low[..., 1], high[..., 1]


def panel(ax: Axes, events: core.Vector, values: np.ndarray, colours: str, norm) -> AxesImage:
    box = extent(events)
    image = ax.imshow(values, extent=box, origin="lower", cmap=colours, norm=norm, interpolation="bilinear")
    ax.set(xlim=box[:2], ylim=box[2:], xticks=[], yticks=[])
    ax.set_aspect("equal")
    return image


def electric_strength(field: core.Bivector) -> np.ndarray:
    """The lab's electric field, the field's part along the lab's time, by its length."""
    return np.linalg.norm((field | core.mv.t).cast(core.ga.subspace("x y z")).kernel, axis=-1)


def magnetic_across(field: core.Bivector) -> np.ndarray:
    """The lab's magnetic field, the dual's part along the lab's time, across the plane."""
    return (field.dual() | core.mv.t).cast(core.ga.subspace("z")).kernel[..., 0]


def density(current: core.Vector) -> np.ndarray:
    """The charge density: the current's part along the lab's time."""
    return (current | core.mv.t).kernel[..., 0]


def fields(events: core.Vector, field: core.Bivector) -> plt.Figure:
    """Electric field strength and magnetic field, one row per case."""
    strengths, magnetic = electric_strength(field), magnetic_across(field)       # [cases, rows, columns] each
    width = 11
    left, right, bottom, top = extent(events)
    result = plt.figure(figsize=(width, len(strengths) * width / 2 * (top - bottom) / (right - left) + 0.3), layout="constrained")
    for axes, strength, across in zip(result.subplots(len(strengths), 2, squeeze=False), strengths, magnetic):
        panel(axes[0], events, strength, STRENGTH_COLOURS, LogNorm(*ELECTRIC_RANGE))
        panel(axes[1], events, across, "RdBu_r", TwoSlopeNorm(0.0, -MAGNETIC_LIMIT, MAGNETIC_LIMIT))
    return result


def densities(events: core.Vector, current: core.Vector) -> plt.Figure:
    """The charge density, signed, one panel per case, on one colour scale."""
    values = density(current)                                                   # [cases, rows, columns]
    limit = np.abs(values).max()
    width = 11
    left, right, bottom, top = extent(events)
    result = plt.figure(figsize=(width, width / len(values) * (top - bottom) / (right - left) + 0.3), layout="constrained")
    for ax, value in zip(result.subplots(1, len(values), squeeze=False)[0], values):
        panel(ax, events, value, "RdBu_r", TwoSlopeNorm(0.0, -limit, limit))
    return result


def sweep(events: core.Vector, field: core.Bivector, current: core.Vector) -> list[np.ndarray]:
    """Electric field strength with the charge's outline, and magnetic field, one frame per case. One
    figure is updated in place and drawn once per frame, without supersampling, to keep pace."""
    coordinates = events.cast(core.ga.subspace("x y")).kernel
    strengths, magnetic, charge_densities = electric_strength(field), magnetic_across(field), density(current)
    left, right, bottom, top = extent(events)
    result = plt.figure(figsize=(11, 11 / 2 * (top - bottom) / (right - left) + 0.2), dpi=80, layout="constrained")
    axes = result.subplots(1, 2)
    images = (panel(axes[0], events, strengths[0], STRENGTH_COLOURS, LogNorm(*ELECTRIC_RANGE)),
              panel(axes[1], events, magnetic[0], "RdBu_r", TwoSlopeNorm(0.0, -MAGNETIC_LIMIT, MAGNETIC_LIMIT)))
    pixels = []
    for strength, across, value in zip(strengths, magnetic, charge_densities):
        images[0].set_data(strength)
        images[1].set_data(across)
        outline = axes[0].contour(coordinates[..., 0], coordinates[..., 1], np.abs(value), levels=[DENSITY_LEVEL],
                                  colors=[DENSITY_COLOUR], linewidths=1.2)
        result.canvas.draw()
        pixels.append(np.asarray(result.canvas.buffer_rgba())[..., :3].copy())
        outline.remove()
    plt.close(result)
    return pixels


def frame(events: core.Vector, field: core.Bivector, current: core.Vector) -> np.ndarray:
    coordinates = events.cast(core.ga.subspace("x y")).kernel
    figure = plt.figure(figsize=(11, 3.7), dpi=80, layout="constrained")
    left, right = figure.subplots(1, 2)
    panel(left, events, electric_strength(field), STRENGTH_COLOURS, LogNorm(*ELECTRIC_RANGE))
    left.contour(coordinates[..., 0], coordinates[..., 1], np.abs(density(current)), levels=[DENSITY_LEVEL], colors=[DENSITY_COLOUR], linewidths=1.2)
    panel(right, events, magnetic_across(field), "RdBu_r", TwoSlopeNorm(0.0, -MAGNETIC_LIMIT, MAGNETIC_LIMIT))
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def animate(frames: Iterable[tuple[core.Vector, core.Bivector, core.Vector]]) -> list[np.ndarray]:
    return [frame(*state) for state in frames]


def wave_limit(events: core.Vector, field: core.Bivector) -> np.ndarray:
    """One colour scale per case for a whole turn: the magnetic field times the distance from the
    orbit's centre, far from the orbit; the sharpest crests of a fast charge's beam saturate."""
    distance = np.linalg.norm(events.cast(core.ga.subspace("x y")).kernel, axis=-1)      # [..., rows, columns]
    far = np.where(distance > WAVE_ZONE, np.abs(magnetic_across(field) * distance), np.nan)
    return np.nanpercentile(far, WAVE_PERCENTILE, axis=(-2, -1))                       # [...]


def compressed_waves(events: core.Vector, field: core.Bivector, limit: np.ndarray) -> np.ndarray:
    """The magnetic field across the orbit's plane times the distance from the orbit's centre, so that
    the outgoing waves keep their strength across the picture; a power below one on the magnitude keeps
    a fast charge's narrow pulses and the field between them."""
    coordinates = events.cast(core.ga.subspace("x y")).kernel
    scaled = magnetic_across(field) * np.linalg.norm(coordinates, axis=-1) / limit[..., None, None]
    return np.sign(scaled) * np.abs(scaled) ** WAVE_POWER


def waves(events: core.Vector, field: core.Bivector, charge: core.Vector, limit: np.ndarray) -> np.ndarray:
    position = charge.cast(core.ga.subspace("x y")).kernel
    figure = plt.figure(figsize=(5.6, 5.6), dpi=WAVE_DPI, layout="constrained")
    ax = figure.subplots()
    panel(ax, events, compressed_waves(events, field, limit), "RdBu_r", TwoSlopeNorm(0.0, -1.0, 1.0))
    ax.scatter(position[..., 0], position[..., 1], s=18, color=CHARGE_COLOUR, edgecolors="#0f172a", linewidths=0.8, zorder=3)
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def animate_waves(frames: list[tuple[core.Vector, core.Bivector, core.Vector]]) -> list[np.ndarray]:
    events, field, _ = frames[0]
    limit = wave_limit(events, field)
    return [waves(*state, limit) for state in frames]


def radiation(events: core.Vector, field: core.Bivector, charge: core.Vector) -> plt.Figure:
    """The waves of circling charges side by side, one panel per case, each on its own colour scale,
    with the charge marked."""
    values = compressed_waves(events, field, wave_limit(events, field))           # [cases, rows, columns]
    positions = charge.cast(core.ga.subspace("x y")).kernel.reshape(len(values), 2)   # [cases, 2]
    result = plt.figure(figsize=(11, 11 / len(values) + 0.2), layout="constrained")
    for ax, value, position in zip(result.subplots(1, len(values), squeeze=False)[0], values, positions):
        panel(ax, events, value, "RdBu_r", TwoSlopeNorm(0.0, -1.0, 1.0))
        ax.scatter(*position, s=18, color=CHARGE_COLOUR, edgecolors="#0f172a", linewidths=0.8, zorder=3)
    return result


def inline(frames: list[np.ndarray], duration_ms: int):
    """Frames as a looping GIF to show in a notebook, kept in memory."""
    from io import BytesIO
    from IPython.display import Image as Shown
    from PIL import Image
    images = [Image.fromarray(pixels) for pixels in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return Shown(data=buffer.getvalue(), format="gif")
