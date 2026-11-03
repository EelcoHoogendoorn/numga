"""A rubber band lifting a weight by heat: the band held by its top row, the weight its lowest rows, and a lamp beside
it switched on and then off."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.rubber_band import core
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
# the lamp is on and then off; the steps between frames; the frames' duration in milliseconds, and the colours of their
# palette.
dt = 0.025
n_settling, n_lit, n_dark, n_stride = 3000, 2000, 2000, 40
frame_duration_ms = 40
n_animation_colours = 64
# The pull: a hand pulling the weight down as hard as its weight again, in a quick ramp, held while the band's warmth
# leaks to the room, let go as quickly, and the band left to warm back; the steps of each, the ramp's in a few of the
# weight's bounces, the holds in about a warming time; the colours' reach either side of the room's temperature, and how
# far the hand pulls the weight down, which the picture leaves room for.
hand_pull = 1.0
n_ramp, n_hold = 160, 800
pull_swing, pull_drop = 6e-4, 0.9


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
    coordinates = lattice(n_across, n_band_rows + n_weight_rows)             # [vertices, 2]
    rest = mv(core.Upright, coordinates).dual().field()                       # Point-[vertices]
    topology = Topology.triangulate(rest)
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


def settled(band: core.Band, rest: core.Point, n_vertices: int) -> tuple[core.Point, core.Scalar]:
    """The band let go at rest under the weight at the room's temperature and settled with the lamp off: its places
    `Point-[vertices]` and the entropies they hold `Scalar+[vertices]`."""
    unheld = mv.scalar(np.zeros((n_vertices, 1))).field()                     # Scalar+[vertices]
    *_, (places, entropies, *_) = core.heating(band, rest, unheld, lamp, lamp_heat, reach, np.zeros(n_settling),
                                               np.zeros(n_settling), dt, n_settling)
    return places, entropies


def engine() -> tuple[Topology, np.ndarray, Iterator[tuple[core.Point, core.Scalar, core.Scalar, float, core.Budget]]]:
    """The engine's run: the band's triangles, which are the weight's, and its states, from the band settled under the
    weight at room temperature, the lamp on and then off."""
    topology, rest, in_weight, band = hanging_band()
    exposure = np.concatenate([np.ones(n_lit), np.zeros(n_dark)])
    return topology, in_weight, core.heating(band, *settled(band, rest, topology.n_vertices), lamp, lamp_heat, reach,
                                             exposure, np.zeros_like(exposure), dt, n_stride)


def pull() -> tuple[Topology, np.ndarray, Iterator[tuple[core.Point, core.Scalar, core.Scalar, float, core.Budget]]]:
    """The band pulled by hand, with the lamp off: its triangles, which are the weight's, and its states, from the band
    settled under the weight, the hand's pull ramped up, held, let go and the band left."""
    topology, rest, in_weight, band = hanging_band()
    ramp = (1 - np.cos(np.pi * np.arange(n_ramp) / n_ramp)) / 2
    pulling = np.concatenate([ramp, np.ones(n_hold), ramp[::-1], np.zeros(n_hold)])
    return topology, in_weight, core.heating(band, *settled(band, rest, topology.n_vertices), lamp, lamp_heat, reach,
                                             np.zeros_like(pulling), pulling, dt, n_stride)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.thermodynamics.rubber_band import render

    topology, in_weight, states = engine()
    save_animation(render.animate(topology, in_weight, lamp, room_temperature, lamp_temperature, states),
                   "rubber_band", frame_duration_ms, colors=n_animation_colours)
    topology, in_weight, states = pull()
    save_animation(render.animate_pulled(topology, in_weight, room_temperature, pull_swing, pull_drop, states),
                   "rubber_band_pulled", frame_duration_ms, colors=n_animation_colours)


if __name__ == "__main__":
    main()
