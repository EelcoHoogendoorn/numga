"""A thermocouple bent into a hairpin, a leg of each metal joined across the top, its two feet clamped to one cold
block; from equilibrium, heated at the junction and, apart, driven by a current. Heated, the warmth spreads from the
junction down both legs, and a voltage grows between the feet as it does. Driven by a current, the junction cools,
and the legs conduct warmth up to it from the block. Settled, a unit of heat a second gives as large a voltage as a
unit of current gives a fall of the junction's temperature.

Each metal has two kinds of carrier, its electrons, which hold a charge and an entropy, and the vibrations of its
lattice, which hold entropy and no charge; the electrons of the second metal hold a quarter of the entropy the first
metal's hold. The constants are of order one, in units where the hairpin is one wide, an electron's charge is one and
each kind's conductance is one.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from numga.topology import Topology

from examples.thermodynamics.thermocouple import core
from examples.util import shapes

mv = core.mv

# The two kinds of carrier in each metal, its electrons and the vibrations of its lattice: the charge each holds; the
# entropy each holds, in the first metal and in the second; and its mobility, the velocity a unit force gives it,
# times its density.
charges = np.array([-1, 0])                                                   # [kinds]
entropies = np.array([[2.0, 1.0], [0.5, 1.0]])                                # [metals, kinds]
mobilities = np.array([1.0, 1.0])                                             # [kinds]
# The entropy a unit of area takes up for each unit its temperature rises, and the entropy the cold block takes from a
# unit of area of the second foot a unit of time, for each unit of its temperature.
capacity, clamping = 1.0, 4000.0
# The two drives: the entropy put in at the junction a unit of time, and the current through the hairpin.
heating, current = 1.0, 1.0
# The frames, their duration in milliseconds, and the time between them, so that the hairpin has all but settled by
# the last.
n_frames = 75
frame_duration_ms = 40
time_step = 0.1
# The hairpin: its height in its widths, a leg's width, the height of the bridge across its top, and the spacing of
# its places, which divides all three.
height, leg, bridge, spacing = 2.0, 0.3, 0.3, 0.05
# The pictures' pixels to a width.
resolution = 150


# --- scenes ---------------------------------------------------------------------------
def hairpin(spacing: float) -> tuple[Topology, core.Point, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The hairpin, its places this far apart: its triangles, its vertices' places `Point-[vertices]`, the metal of
    each triangle `[faces]`, the first on the left, the vertices of its feet, the first metal's `[foot]` and the second
    metal's `[foot]`, and the vertices along the top of the junction `[heated]`."""
    columns = np.round(np.arange(-0.5, 0.5 + spacing / 2, spacing), 9)
    rows = np.round(np.arange(0, height + spacing / 2, spacing), 9)
    grid = np.stack(np.meshgrid(columns, rows, indexing="ij"), axis=-1).reshape(-1, 2)   # [places, 2]
    # The gap between the legs, below the bridge, holds no places but those on its edge.
    in_gap = (np.abs(grid[:, 0]) < 0.5 - leg - spacing / 2) & (grid[:, 1] < height - bridge - spacing / 2)
    places = grid[~in_gap]                                                    # [vertices, 2]
    filled, _ = shapes.delaunay(places)
    centres = places[filled.elements[2]].mean(axis=1)                         # [faces, 2]
    across_gap = (np.abs(centres[:, 0]) < 0.5 - leg) & (centres[:, 1] < height - bridge)
    topology = filled.subset(~across_gap, 2)
    metals = (centres[~across_gap, 0] > 0).astype(int)                        # [faces]
    feet = [np.flatnonzero((places[:, 1] == 0) & (side * places[:, 0] > 0)) for side in (-1, 1)]
    heated = np.flatnonzero((places[:, 1] == height) & (np.abs(places[:, 0]) <= spacing))
    # Each place the origin moved across and up, a point of weight one.
    vertices = (core.origin + mv(core.Force, places).dual()).field()          # Point-[vertices]
    return topology, vertices, metals, feet[0], feet[1], heated


def operators(topology: Topology, vertices: core.Point, metals: np.ndarray, clamped: np.ndarray
              ) -> tuple[core.SparseExtensor, core.SparseExtensor]:
    """The hairpin's conductance and its storage of entropy, `(Parcel- <- Potential+)[vertices <- vertices]` each:
    each triangle responding as its metal does, and the cells of the vertices `clamped` `[clamped]` losing entropy to
    the cold block besides, in proportion to their temperature."""
    sizes, gradients = core.simplex_gradients(topology, vertices)
    responses = core.response(charges, entropies, mobilities)[metals].field()   # (Flux- <- Drive+)[faces]
    # The block takes entropy from the clamped cells as a cell takes it up, in proportion to its temperature.
    on_block = np.isin(np.arange(topology.n_vertices), clamped)               # [vertices]
    block = core.storage(topology, sizes, clamping * on_block)
    storing = core.storage(topology, sizes, np.full(topology.n_vertices, capacity))
    return core.conductance(topology, sizes, gradients, responses) + block, storing


def states(topology: Topology, losing: core.SparseExtensor, storing: core.SparseExtensor, high: np.ndarray,
           low: np.ndarray, heated: np.ndarray) -> Iterator[tuple[core.Potential, core.Potential]]:
    """The hairpin's potentials `Potential+[vertices]` at each frame, heated at the vertices `heated` and, apart,
    driven by the current, from equilibrium: its first foot `high` `[foot]` held at its potentials in equilibrium, and
    the current leaving through its second foot `low` `[foot]`, shared evenly between the foot's vertices, as the
    heating is between the heated ones."""
    shares = np.zeros((2, topology.n_vertices))
    shares[0, heated], shares[1, low] = 1 / len(heated), 1 / len(low)
    at_heated, at_low = mv.scalar(shares[..., None]).field()                  # Scalar+[vertices] each
    return zip(core.settling(topology, losing, storing, high, at_heated * heating * core.entropy, time_step, n_frames),
               core.settling(topology, losing, storing, high, at_low * -current * core.time, time_step, n_frames))


def readings(topology: Topology, low: np.ndarray, heated: np.ndarray, states: list
             ) -> tuple[core.Scalar, core.Scalar]:
    """At each of the hairpin's frames `[frames] Scalar+` each: heated, the voltage of its second foot above its
    first, per unit of heating; driven by the current, the fall of the junction's temperature per unit of current.
    Each is the mean over its vertices."""
    warmed = stack([potentials for potentials, _ in states])                 # [frames] Potential+[vertices]
    driven = stack([potentials for _, potentials in states])                 # [frames] Potential+[vertices]
    voltages = (topology.reading(low, 0) * (warmed & core.time)).batch().mean(axis=-1)       # [frames] Scalar+
    cooling = -(topology.reading(heated, 0) * (driven & core.entropy)).batch().mean(axis=-1)   # [frames] Scalar+
    return voltages / heating, cooling / current


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.thermocouple import render

    topology, vertices, metals, high, low, heated = hairpin(spacing)
    save_figure(render.draw_hairpin(topology, vertices, metals, (high, low), resolution), "thermocouple_hairpin")
    losing, storing = operators(topology, vertices, metals, low)
    frames = list(states(topology, losing, storing, high, low, heated))
    save_animation(render.animate(topology, vertices, metals, (high, low), resolution, frames[-1], frames),
                   "thermocouple", frame_duration_ms)
    voltage, cooling = readings(topology, low, heated, frames)
    save_figure(render.draw_readings(time_step * np.arange(n_frames), voltage, cooling), "thermocouple_readings")

    # checks
    # Settled, a unit of heating raises the voltage by as much as a unit of current lowers the junction's temperature;
    # and both have all but settled by the last frame.
    np.testing.assert_allclose(voltage.to_array()[-1], cooling.to_array()[-1], rtol=2e-2)
    assert abs(voltage.to_array()[-1] - voltage.to_array()[-5]) < 2e-2 * abs(voltage.to_array()[-1])


if __name__ == "__main__":
    main()
