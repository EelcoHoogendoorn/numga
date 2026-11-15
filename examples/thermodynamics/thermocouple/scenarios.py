"""Two-metal hairpin geometry, carrier parameters, and transient runs configure the Seebeck and Peltier demonstrations.

Two legs of dissimilar metals meet across a top bridge while both feet rest on a cold reference block. When entropy enters at the top junction, warmth diffuses down both legs and builds an open-circuit voltage between the feet. When electric current leaves through the second foot instead, the junction cools and draws heat up both legs from the block, matching the heated run's foot voltage per unit input at every frame.

Each metal carries mobile electrons with charge and entropy alongside neutral lattice vibrations that carry entropy alone, with electrons in the second metal carrying one quarter of the entropy carried in the first. Dimensionless units set the overall hairpin width, elementary charge, and carrier mobilities to order unity.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from numga import stack
from numga.topology import Topology
from examples.thermodynamics.thermocouple import core
from examples.util import shapes

mv = core.mv

# Carrier charges, per-metal entropies, and mobilities for conduction electrons and lattice vibrations.
charges = np.array([-1, 0])                                                   # [2]
entropies = np.array([[2.0, 1.0], [0.5, 1.0]])                                # [2, 2]
mobilities = np.array([1.0, 1.0])                                             # [2]
capacity, clamping = 1.0, 4000.0
heating, current = 1.0, 1.0
n_frames = 75
frame_duration_ms = 40
time_step = 0.1
height, leg, bridge, spacing = 2.0, 0.3, 0.3, 0.05
resolution = 150


# --- scenes ---------------------------------------------------------------------------
def hairpin(spacing: float) -> tuple[Topology, core.Point, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Delaunay triangulation of the slotted grid at step `spacing` gives the hairpin `Topology`, unit-weight `vertices` (`Point-[V]`), triangle `metals` (`[F]`), both foot vertex arrays, and the top junction vertex indices."""
    columns = np.round(np.arange(-0.5, 0.5 + spacing / 2, spacing), 9)
    rows = np.round(np.arange(0, height + spacing / 2, spacing), 9)
    grid = np.stack(np.meshgrid(columns, rows, indexing="ij"), axis=-1).reshape(-1, 2)
    # Omit interior grid points inside the gap between the two legs below the top bridge.
    gap_points = (np.abs(grid[:, 0]) < 0.5 - leg - spacing / 2) & (grid[:, 1] < height - bridge - spacing / 2)
    places = grid[~gap_points]                                                # [V, 2]
    full_topology = shapes.delaunay(places)[0]
    centres = places[full_topology.elements[2]].mean(axis=1)
    gap_triangles = (np.abs(centres[:, 0]) < 0.5 - leg) & (centres[:, 1] < height - bridge)
    topology = full_topology.subset(~gap_triangles, 2)
    metals = (centres[~gap_triangles, 0] > 0).astype(int)                     # [F]
    feet = [np.flatnonzero((places[:, 1] == 0) & (side * places[:, 0] > 0)) for side in (-1, 1)]
    junction_vertices = np.flatnonzero((places[:, 1] == height) & (np.abs(places[:, 0]) <= spacing))
    vertices = (core.origin + mv(core.Force, places).dual()).field()          # Point-[V]
    return topology, vertices, metals, feet[0], feet[1], junction_vertices


def operators(
    topology: Topology,
    vertices: core.Point,
    metals: np.ndarray,
    clamped_vertices: np.ndarray,
) -> tuple[core.SparseExtensor, core.SparseExtensor]:
    """Triangle responses and dual-cell capacities assemble into the vertex loss and entropy-storage operators (`(Parcel- <- Potential+)[V <- V]`) with thermal clamping on `clamped_vertices`."""
    V, E, F = topology.n_elements
    areas, gradients = core.simplex_gradients(topology, vertices)
    responses = core.response(charges, entropies, mobilities)[metals].field()   # (Flux- <- Drive+)[F]
    clamped_mask = np.isin(np.arange(V), clamped_vertices)                      # [V]
    block_loss = core.storage(topology, areas, clamping * clamped_mask)
    storage_map = core.storage(topology, areas, np.full(V, capacity))
    return core.conductance(topology, areas, gradients, responses) + block_loss, storage_map


def states(
    topology: Topology,
    loss_map: core.SparseExtensor,
    storage_map: core.SparseExtensor,
    first_foot: np.ndarray,
    second_foot: np.ndarray,
    junction_vertices: np.ndarray,
) -> Iterator[tuple[core.Potential, core.Potential]]:
    """Yields paired `(heated, driven)` vertex potential states (`Potential+[V]`) across `n_frames` steps with `first_foot` pinned to equilibrium."""
    V, E, F = topology.n_elements
    shares = np.zeros((2, V))
    shares[0, junction_vertices], shares[1, second_foot] = 1 / len(junction_vertices), 1 / len(second_foot)
    junction_weights, foot_weights = mv.scalar(shares[..., None]).field()       # Scalar+[V], Scalar+[V]
    return zip(
        core.settling_potentials(
            topology, loss_map, storage_map, first_foot, junction_weights * heating * core.entropy, time_step, n_frames
        ),
        core.settling_potentials(
            topology, loss_map, storage_map, first_foot, foot_weights * -current * core.time, time_step, n_frames
        ),
    )


def readings(
    topology: Topology,
    second_foot: np.ndarray,
    junction_vertices: np.ndarray,
    states: Sequence[tuple[core.Potential, core.Potential]],
) -> tuple[core.Scalar, core.Scalar]:
    """Vertex averages on `second_foot` and `junction_vertices` give the foot voltage per unit heating and junction cooling per unit current (`[...] Scalar+`) across `states`."""
    heated_states = stack([potentials for potentials, _ in states])             # [...] Potential+[V]
    driven_states = stack([potentials for _, potentials in states])             # [...] Potential+[V]
    voltages = (topology.reading(second_foot, 0) * (heated_states & core.time)).batch().mean(axis=-1)            # [...] Scalar+
    cooling = -(topology.reading(junction_vertices, 0) * (driven_states & core.entropy)).batch().mean(axis=-1)   # [...] Scalar+
    return voltages / heating, cooling / current


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.thermocouple import render

    topology, vertices, metals, first_foot, second_foot, junction_vertices = hairpin(spacing)
    save_figure(
        render.hairpin_figure(topology, vertices, metals, (first_foot, second_foot), resolution),
        "thermocouple_hairpin",
    )
    loss_map, storage_map = operators(topology, vertices, metals, second_foot)
    frames = list(states(topology, loss_map, storage_map, first_foot, second_foot, junction_vertices))
    save_animation(
        render.animation(topology, vertices, metals, (first_foot, second_foot), resolution, frames[-1], frames),
        "thermocouple",
        frame_duration_ms,
    )
    voltage, cooling = readings(topology, second_foot, junction_vertices, frames)
    save_figure(render.readings_figure(time_step * np.arange(n_frames), voltage, cooling), "thermocouple_readings")

    # checks
    assert (voltage[-1] - cooling[-1]).abs() < 2e-2 * voltage[-1].abs()
    assert (voltage[-1] - voltage[-5]).abs() < 2e-2 * voltage[-1].abs()


if __name__ == "__main__":
    main()
