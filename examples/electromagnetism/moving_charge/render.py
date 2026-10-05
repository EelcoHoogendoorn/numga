"""The electric and magnetic fields of a moving charge, as the lab sees them."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, TwoSlopeNorm

from examples.animation import capture
from examples.electromagnetism.moving_charge import core

ELECTRIC_RANGE = (0.05, 50.0)
MAGNETIC_LIMIT = 10.0
DENSITY_LEVEL = 5.0
DENSITY_COLOUR = "#22d3ee"
WAVE_ZONE = 4.0
WAVE_PERCENTILE = 99.0
WAVE_POWER = 0.4


# --- plumbing -------------------------------------------------------------------------
def panel(ax: Axes, events: core.Vector, values: np.ndarray, colours: str, norm, title: str) -> tuple[float, ...]:
    coordinates = events.cast(core.ga.subspace("x y")).kernel
    box = (coordinates[0, 0, 0], coordinates[0, -1, 0], coordinates[0, 0, 1], coordinates[-1, 0, 1])
    ax.imshow(values, extent=box, origin="lower", cmap=colours, norm=norm, interpolation="bilinear")
    ax.set(xlim=box[:2], ylim=box[2:], xticks=[], yticks=[], title=title)
    ax.set_aspect("equal")
    return coordinates


def frame(events: core.Vector, field: core.Bivector, current: core.Vector) -> np.ndarray:
    # The lab's electric and magnetic fields: the field's parts along and across the lab's time.
    electric = (field | core.mv.t).cast(core.ga.subspace("x y z")).kernel
    magnetic = (field.dual() | core.mv.t).cast(core.ga.subspace("z")).kernel[..., 0]
    density = (current | core.mv.t).kernel[..., 0]
    figure = plt.figure(figsize=(11, 3.9), dpi=80, layout="constrained")
    left, right = figure.subplots(1, 2)
    coordinates = panel(left, events, np.linalg.norm(electric, axis=-1), "magma", LogNorm(*ELECTRIC_RANGE),
                        "Electric field strength, and the charge")
    left.contour(coordinates[..., 0], coordinates[..., 1], density, levels=[DENSITY_LEVEL], colors=[DENSITY_COLOUR], linewidths=1.2)
    panel(right, events, magnetic, "RdBu_r", TwoSlopeNorm(0.0, -MAGNETIC_LIMIT, MAGNETIC_LIMIT), "Magnetic field across the plane")
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def animate(frames: Iterable[tuple[core.Vector, core.Bivector, core.Vector]]) -> list[np.ndarray]:
    return [frame(*state) for state in frames]


def waves(events: core.Vector, field: core.Bivector, charge: core.Vector, limit: float) -> np.ndarray:
    """The magnetic field across the orbit's plane times the distance from the orbit's centre, so
    that the outgoing waves keep their strength across the picture."""
    coordinates = events.cast(core.ga.subspace("x y")).kernel
    magnetic = (field.dual() | core.mv.t).cast(core.ga.subspace("z")).kernel[..., 0]
    scaled = magnetic * np.linalg.norm(coordinates, axis=-1) / limit
    # A power below one on the magnitude keeps a fast charge's narrow pulses and the field between them.
    compressed = np.sign(scaled) * np.abs(scaled) ** WAVE_POWER
    position = charge.cast(core.ga.subspace("x y")).kernel
    figure = plt.figure(figsize=(5.6, 5.6), dpi=80, layout="constrained")
    ax = figure.subplots()
    panel(ax, events, compressed, "RdBu_r", TwoSlopeNorm(0.0, -1.0, 1.0), "Magnetic field times distance")
    ax.scatter(position[..., 0], position[..., 1], s=18, color="#facc15", edgecolors="#0f172a", linewidths=0.8, zorder=3)
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def animate_waves(frames: list[tuple[core.Vector, core.Bivector, core.Vector]]) -> list[np.ndarray]:
    # One colour scale for the whole turn, from the field far from the orbit in the first frame; the
    # sharpest crests of a fast charge's beam saturate.
    events, field, _ = frames[0]
    coordinates = events.cast(core.ga.subspace("x y")).kernel
    distance = np.linalg.norm(coordinates, axis=-1)
    magnetic = (field.dual() | core.mv.t).cast(core.ga.subspace("z")).kernel[..., 0]
    limit = np.percentile(np.abs(magnetic * distance)[distance > WAVE_ZONE], WAVE_PERCENTILE)
    return [waves(*state, limit) for state in frames]
