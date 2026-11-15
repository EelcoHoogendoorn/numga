"""A hanging rubber band clamped at its top row with a weight at its bottom rows lifts the weight when warmed by a
lamp, alongside an identical band pulled down by hand on the same timeline."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.rubber_band import core
from examples.util import shapes
from numga.topology import Topology

mv = core.mv

# Band geometry and mass density, plus the denser, stiffer weight rows attached at the bottom.
n_across, n_band_rows, density = 4, 11, 0.25
n_weight_rows, weight_density, weight_stiffening = 2, 1.0, 6.0
# Room-temperature shear modulus, Poisson ratio (so the band narrows visibly when stretched), first plane-stress Lamé
# modulus, heat capacity per area and degree, and thermal conductivity.
lame_mu, poisson_ratio = 1.0, 1 / 3
lame_lambda = 2 * lame_mu * poisson_ratio / (1 - poisson_ratio)
heat_capacity, conductivity = 600.0, 6.0
# Room temperature, gravitational acceleration and air drag per unit mass.
room_temperature, gravity, drag = 1.0, 1.0, 0.5
# Peak equilibrium lamp temperature at its half-intensity reach (exaggerated so the lift is plain) and warming time.
lamp_temperature, reach, warming_time = 1.8, 1.0, 20.0
cooling = heat_capacity / warming_time
lamp_heat = 2 * cooling * (lamp_temperature - room_temperature)
# Lamp position beside the middle of the hanging band.
lamp_across, lamp_up = -1.2, -2.4
lamp = lamp_across * core.across + lamp_up * core.up                          # [] Point-
# Time step, unrendered settling steps, lit and dark step counts, frame stride and frame duration.
dt = 0.025
n_settling, n_lit, n_dark, n_stride = 3000, 2000, 2000, 40
frame_duration_ms = 40
# Downward hand-pull schedule on the lamp's timeline: a quick rise, a hold while heat leaks to the room, a quick
# release, and a final hold, together with the colour range and framing drop for the pulled band.
hand_pull = 1.0
n_rise = 160
n_hold = n_lit - n_rise
pull_swing, pull_drop = 6e-4, 0.9
# Per-step schedules for lamp exposure and downward hand pull.
exposure = np.concatenate([np.ones(n_lit), np.zeros(n_dark)])
rise = (1 - np.cos(np.pi * np.arange(n_rise) / n_rise)) / 2
pulling = np.concatenate([rise, np.ones(n_hold), rise[::-1], np.zeros(n_hold)])


# --- plumbing -------------------------------------------------------------------------
def lattice(n_across: int, n_rows: int) -> np.ndarray:
    """Staggered vertex coordinates of a unit-wide strip hanging from the origin in equilateral-height rows, shifting
    alternate interior vertices by half a spacing so the triangulation favours neither diagonal."""
    spacing = 1 / (n_across - 1)
    heights = spacing * np.sqrt(3) / 2 * np.arange(n_rows + 1)
    straight = np.linspace(-0.5, 0.5, n_across)
    shifted = np.concatenate([[-0.5], (straight[:-1] + straight[1:]) / 2, [0.5]])
    rows = [np.stack([(shifted if row % 2 else straight), np.full(len(shifted if row % 2 else straight), -height)],
                     axis=-1) for row, height in enumerate(heights)]
    return np.concatenate(rows)                                               # [V, 2]


# --- scenes ---------------------------------------------------------------------------
def hanging_band() -> tuple[Topology, core.Point, np.ndarray, core.Band]:
    """Triangulated strip, unloaded rest positions, boolean face mask for the hanging weight, and assembled band."""
    coordinates = lattice(n_across, n_band_rows + n_weight_rows)              # [V, 2]
    topology, coordinates = shapes.delaunay(coordinates)
    rest = mv(core.Upright, coordinates).dual().field()                       # Point-[V]
    # Mark the bottom rows as the denser, stiffer weight and clamp the top row at y == 0.
    band_length = n_band_rows * np.sqrt(3) / 2 / (n_across - 1)
    in_weight = coordinates[topology.elements[2]][..., 1].max(axis=-1) < -band_length + 1e-9   # [F]
    held = np.flatnonzero(coordinates[:, 1] == 0)                            # [V_held]
    band = core.band(topology, rest, held, core.constitutive(lame_lambda, lame_mu),
                     mv.scalar(np.where(in_weight, weight_stiffening, 1.0)[:, None]).field(),
                     mv.scalar(np.where(in_weight, weight_density, density)[:, None]).field(),
                     mv.scalar(in_weight[:, None] * 1.0).field(),
                     heat_capacity, conductivity, cooling, room_temperature, gravity, drag, hand_pull)
    return topology, rest, in_weight, band


def settled(band: core.Band, rest: core.Point, n_steps: int) -> core.Point:
    """Equilibrium vertex state after releasing the unloaded band under the weight at room temperature with the lamp off."""
    *_, (points, *_) = core.run(band, rest, lamp, lamp_heat, reach, np.zeros(n_steps), np.zeros(n_steps), dt, n_steps)
    return points


def engine() -> tuple[Topology, np.ndarray, Iterator[tuple[core.Point, core.Scalar, float]]]:
    """Triangulation, weight mask, and state stream for the lamp-driven heat engine starting from the settled band."""
    topology, rest, in_weight, band = hanging_band()
    return topology, in_weight, core.run(band, settled(band, rest, n_settling), lamp, lamp_heat, reach, exposure,
                                         np.zeros_like(exposure), dt, n_stride)


def pull() -> tuple[Topology, np.ndarray, Iterator[tuple[core.Point, core.Scalar, float]]]:
    """Triangulation, weight mask, and state stream for the hand-pulled band with the lamp switched off."""
    topology, rest, in_weight, band = hanging_band()
    return topology, in_weight, core.run(band, settled(band, rest, n_settling), lamp, lamp_heat, reach,
                                         np.zeros_like(pulling), pulling, dt, n_stride)


def engine_and_pull(n_steps: int) -> tuple[Topology, np.ndarray,
                                           Iterator[tuple[tuple[core.Point, core.Scalar, float],
                                                          tuple[core.Point, core.Scalar, float]]]]:
    """Triangulation, weight mask, and paired state stream of the lamp-driven band and the hand-pulled band on a shared
    timeline, both starting from the settled band."""
    topology, rest, in_weight, band = hanging_band()
    start = settled(band, rest, n_steps)
    lamp_run = core.run(band, start, lamp, lamp_heat, reach, exposure, np.zeros_like(exposure), dt, n_stride)
    hand_run = core.run(band, start, lamp, lamp_heat, reach, np.zeros_like(pulling), pulling, dt, n_stride)
    return topology, in_weight, zip(lamp_run, hand_run)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.thermodynamics.rubber_band import render

    topology, in_weight, states = engine_and_pull(n_settling)
    save_animation(render.animate_together(topology, in_weight, lamp, room_temperature, lamp_temperature, pull_swing,
                                           pull_drop, states),
                   "rubber_band_lamp_and_pull", frame_duration_ms)


if __name__ == "__main__":
    main()
