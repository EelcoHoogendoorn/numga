"""A rubber band lifting a weight by heat: the band held by its top row, the weight its lowest rows, and a lamp beside
it switched on and then off; and the same band pulled down by hand, warming as it stretches, on the same timeline."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.rubber_band import core
from examples.util import shapes
from examples.util.topology import Topology

mv = core.mv

# The band: its vertices across, its rows of triangles along, each the height of an equilateral one, and its mass per
# area; the weight: its rows below the band, and its density and stiffness against the band's.
n_across, n_band_rows, density = 4, 11, 0.25
n_weight_rows, weight_density, weight_stiffening = 2, 1.0, 6.0
# The rubber: its shear modulus at room temperature, one, and its Poisson ratio, a moderate one, so that the band
# narrows plainly as it stretches; its first Lamé modulus follows, for a sheet; the heat it holds per area and degree,
# about real rubber's in these units, and its conductivity, so that heat crosses the band in about twice the time the
# band takes to warm.
lame_mu, poisson_ratio = 1.0, 1 / 3
lame_lambda = 2 * lame_mu * poisson_ratio / (1 - poisson_ratio)
heat_capacity, conductivity = 600.0, 6.0
# The room's temperature, gravity's acceleration and the air's drag per mass.
room_temperature, gravity, drag = 1.0, 1.0, 0.5
# The lamp: the temperature it would warm rubber to at its reach, where it shines half as bright as at its own place,
# with no conduction, far beyond what real rubber bears, so that the lift is plain; and the time the band takes to warm
# or cool, its heat capacity over the room's cooling per degree.
lamp_temperature, reach, warming_time = 1.8, 1.0, 20.0
cooling = heat_capacity / warming_time
lamp_heat = 2 * cooling * (lamp_temperature - room_temperature)
# The lamp's place beside the band, across and up, at the height of the band's middle when it hangs.
lamp_across, lamp_up = -1.2, -2.4
lamp = lamp_across * core.across + lamp_up * core.up                          # [] Point-
# The run: its time step; the steps the band settles under the weight with the lamp off, run but not shown; the steps
# the lamp is on and then off, as many each; the steps between frames, and the frames' duration in milliseconds.
dt = 0.025
n_settling, n_lit, n_dark, n_stride = 3000, 2000, 2000, 40
frame_duration_ms = 40
# The pull: a hand pulling the weight down as hard as its weight again, in a quick ramp, held while the band's warmth
# leaks to the room, let go as quickly, and the band left to warm back, on the lamp's timeline: the hand pulls as the
# lamp comes on and lets go as it goes off. The steps of the ramp, in a few of the weight's bounces, and of the holds,
# the rest of the lamp's time on and off; the colours' reach either side of the room's temperature, about a thousandth
# of the lamp's, and how far the hand pulls the weight down, which the picture leaves room for.
hand_pull = 1.0
n_ramp = 160
n_hold = n_lit - n_ramp
pull_swing, pull_drop = 6e-4, 0.9
# Each step's exposure to the lamp, on and then off, and the hand's pull, ramped up, held, let go and the band left.
exposure = np.concatenate([np.ones(n_lit), np.zeros(n_dark)])
ramp = (1 - np.cos(np.pi * np.arange(n_ramp) / n_ramp)) / 2
pulling = np.concatenate([ramp, np.ones(n_hold), ramp[::-1], np.zeros(n_hold)])


# --- plumbing -------------------------------------------------------------------------
def lattice(n_across: int, n_rows: int) -> np.ndarray:
    """Rows of places one unit wide hanging down from the origin `[vertices, 2]`, across and up, these many rows of
    triangles and one more of places, a row every height of an equilateral triangle, every other row shifted by half a
    spacing with the band's edges kept, so that its triangulation is uniform and prefers no diagonal."""
    spacing = 1 / (n_across - 1)
    heights = spacing * np.sqrt(3) / 2 * np.arange(n_rows + 1)
    straight = np.linspace(-0.5, 0.5, n_across)
    shifted = np.concatenate([[-0.5], (straight[:-1] + straight[1:]) / 2, [0.5]])
    rows = [np.stack([(shifted if row % 2 else straight), np.full(len(shifted if row % 2 else straight), -height)],
                     axis=-1) for row, height in enumerate(heights)]
    return np.concatenate(rows)


# --- scenes ---------------------------------------------------------------------------
def hanging_band() -> tuple[Topology, core.Point, np.ndarray, core.Band]:
    """The band at rest: its triangles, the band's above the weight's, its places at rest `Point-[vertices]`, which of
    its triangles are the weight's `[faces]`, and the band and its room."""
    topology, coordinates = shapes.delaunay(lattice(n_across, n_band_rows + n_weight_rows))
    rest = mv(core.Upright, coordinates).dual().field()                       # Point-[vertices]
    # The weight's triangles, those wholly below the band's rows, denser and stiffer; the held top row.
    band_length = n_band_rows * np.sqrt(3) / 2 / (n_across - 1)
    in_weight = coordinates[topology.elements[2]][..., 1].max(axis=-1) < -band_length + 1e-9   # [faces]
    held = np.flatnonzero(coordinates[:, 1] == 0)                            # [held]
    band = core.band(topology, rest, held, core.constitutive(lame_lambda, lame_mu),
                     mv.scalar(np.where(in_weight, weight_stiffening, 1.0)[:, None]).field(),
                     mv.scalar(np.where(in_weight, weight_density, density)[:, None]).field(),
                     mv.scalar(in_weight[:, None] * 1.0).field(),
                     heat_capacity, conductivity, cooling, room_temperature, gravity, drag, hand_pull)
    return topology, rest, in_weight, band


def settled(band: core.Band, rest: core.Point, n_settling: int = n_settling) -> core.Point:
    """The band let go at rest under the weight at the room's temperature, holding no entropy beyond the room's, and
    settled with the lamp off for these many steps: its points `Point-[vertices]`, each its place and the entropy it
    holds."""
    *_, (points, *_) = core.run(band, rest, lamp, lamp_heat, reach, np.zeros(n_settling), np.zeros(n_settling), dt,
                                n_settling)
    return points


def engine() -> tuple[Topology, np.ndarray, Iterator[tuple[core.Point, core.Scalar, float]]]:
    """The engine's run: the band's triangles, which are the weight's, and its states, from the band settled under the
    weight at room temperature, the lamp on and then off."""
    topology, rest, in_weight, band = hanging_band()
    return topology, in_weight, core.run(band, settled(band, rest), lamp, lamp_heat, reach, exposure,
                                         np.zeros_like(exposure), dt, n_stride)


def pull() -> tuple[Topology, np.ndarray, Iterator[tuple[core.Point, core.Scalar, float]]]:
    """The band pulled by hand, with the lamp off: its triangles, which are the weight's, and its states, from the band
    settled under the weight, the hand's pull ramped up, held, let go and the band left."""
    topology, rest, in_weight, band = hanging_band()
    return topology, in_weight, core.run(band, settled(band, rest), lamp, lamp_heat, reach, np.zeros_like(pulling),
                                         pulling, dt, n_stride)


def engine_and_pull(n_settling: int = n_settling) -> tuple[Topology, np.ndarray,
                                                          Iterator[tuple[tuple[core.Point, core.Scalar, float],
                                                                         tuple[core.Point, core.Scalar, float]]]]:
    """The engine's run and the pull's on one timeline: the band's triangles, which are the weight's, and at each frame
    the two runs' states, the lamp's and the hand's, both from the same band settled under the weight at room
    temperature for these many steps, the hand pulling as the lamp comes on and letting go as it goes off."""
    topology, rest, in_weight, band = hanging_band()
    start = settled(band, rest, n_settling)
    lamp_run = core.run(band, start, lamp, lamp_heat, reach, exposure, np.zeros_like(exposure), dt, n_stride)
    hand_run = core.run(band, start, lamp, lamp_heat, reach, np.zeros_like(pulling), pulling, dt, n_stride)
    return topology, in_weight, zip(lamp_run, hand_run)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.thermodynamics.rubber_band import render

    topology, in_weight, states = engine_and_pull()
    save_animation(render.animate_together(topology, in_weight, lamp, room_temperature, lamp_temperature, pull_swing,
                                           pull_drop, states),
                   "rubber_band_lamp_and_pull", frame_duration_ms)


if __name__ == "__main__":
    main()
