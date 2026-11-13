"""The magnetic refrigerator's cycle in a crystal: a slab of a salt of electron spins between a bath of one kelvin at
its near end and a load at ten millikelvins at its far end. The magnet raises its field to four tesla with the bath
linked and holds it while the bath takes the warmth; lowers it with both ends open until the ideal salt would reach the
load's temperature; lowers it to a millitesla with the load linked and holds it while the load's heat fills the crystal;
and raises it with both ends open until the ideal salt is back at the bath's temperature. And the salt's equation of
state, and the ideal cycle on the temperature-entropy chart."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from numga.sparse import SparseExtensor
from examples.thermodynamics.paramagnetic_salt import core
from examples.util import shapes
from numga.topology import Topology

mv = core.mv

# The spins: each an electron's, its moment a Bohr magneton, 0.672 kelvin per tesla with Boltzmann's constant one; the
# lattice's heat capacity per spin, a little against the spins' entropy, and its reference temperature, the bath's, at
# which it holds no entropy.
moment, heat_capacity = 0.6717, 0.003
bath_temperature, load_temperature = 1.0, 0.01
reference = bath_temperature
# The magnet's highest field, at the bath's temperature, and its lowest, at the load's; the halvings that find the
# spins' share at the other corners.
highest_field, lowest_field = 4.0, 0.001
n_halvings = 20
# The equation of state: the shares of saturation it is drawn at, up to nearly saturated.
n_shares, largest_share = 120, 0.995
# The lines of equal field on the chart, at the highest field and at the lowest: the temperatures they are drawn
# through, up the chart from where their entropy enters it.
n_line_samples = 80
line_temperatures = np.geomspace([0.45, 5e-4], [1.6, 1.6], n_line_samples, axis=-1)   # [lines, samples]
# The crystal: its length in widths, and its vertices along it and across; its lattice's conductivity, and the
# conductance of a link to a reservoir per length of the crystal's end.
length, n_along, n_across = 2.0, 17, 9
conductivity, link_conductance = 3.0, 100.0
# The spins drawn on the crystal, one at each vertex inside it: the seed of their draws; and the moment times the field
# over the temperature at the samples of the line of the magnet's field on the chart, from the chart's warmest at the
# lowest field to spins all but saturated.
spin_seed = 0
line_ratios = np.geomspace(1e-4, 15.0, n_line_samples)                        # [samples]
# The run's phases, in order: magnetized with the bath linked, held while the bath takes the warmth, demagnetized
# isolated, demagnetized with the load linked, held while the load's heat fills the crystal in, and magnetized
# isolated. The corners each runs between; its reservoir, none, the bath or the load; its frames; the time it takes;
# each frame's time over the one before, the holds slicing their relaxation evenly; and the steps each frame takes.
# The phases in contact ramp the field faster than the crystal's conduction carries the heat, so that the end away from
# the reservoir warms or cools well ahead of the end in contact, and take their frames where the crystal changes along
# its length; those isolated, which change it alike everywhere, take fewer.
phase_starts, phase_ends = np.array([0, 1, 1, 2, 3, 3]), np.array([1, 1, 2, 3, 3, 0])
phase_reservoirs = np.array([1, 1, 0, 2, 2, 0])
n_phase_frames = np.array([22, 20, 16, 20, 12, 8])
phase_times = np.array([0.3, 0.6, 0.02, 0.1, 0.045, 0.02])
phase_growths = np.array([1.0, 1.06, 1.0, 1.0, 1.03, 1.0])
n_phase_steps = np.array([3, 3, 4, 5, 5, 4])
frame_duration_ms = 40


def corners() -> tuple[core.Scalar, core.Scalar]:
    """The ideal cycle's corners on the temperature-entropy chart, their entropies and their temperatures `[corners]
    Scalar+`: at the bath's temperature with the entropy of the lowest field at the load's; at the bath's temperature in
    the highest field; at the load's temperature with the entropy of the highest field at the bath's; and at the load's
    temperature in the lowest field. Between them the salt gives the bath heat, cools isolated, takes heat from the
    load, and warms isolated."""
    temperatures = mv.scalar([[bath_temperature], [load_temperature]])       # [2] Scalar+
    fields = stack((highest_field * core.magnet_plane, lowest_field * core.magnet_plane))   # [2] Magnetic+
    magnetized, demagnetized = mv.s & core.state_at(fields, temperatures, moment, heat_capacity, reference)
    return (stack((demagnetized, magnetized, magnetized, demagnetized)),
            mv.scalar([[bath_temperature], [bath_temperature], [load_temperature], [load_temperature]]))


def corner_fields() -> core.Magnetic:
    """The fields at the ideal cycle's corners `[corners] Magnetic+`."""
    return core.corner_fields(bath_temperature, load_temperature, highest_field, lowest_field, moment, heat_capacity,
                              reference, n_halvings)


def lines() -> tuple[core.State, core.Potential]:
    """The lines of equal field on the chart: the salt's states held at the highest field and at the lowest across the
    chart's temperatures `[lines, samples] State-`, and their potentials."""
    fields = stack((highest_field * core.magnet_plane, lowest_field * core.magnet_plane))[:, None]   # [lines, 1] Magnetic+
    states = core.state_at(fields, mv.scalar(line_temperatures[..., None]), moment, heat_capacity, reference)
    return states, core.potentials(states, moment, heat_capacity, reference)


def equation_of_state() -> tuple[core.State, core.Potential]:
    """The salt's states from no magnetization to nearly saturated `[shares] State-`, at the reference temperature's
    entropy for the lattice, and their potentials."""
    shares = mv.scalar(np.linspace(0, largest_share, n_shares)[:, None])     # [shares] Scalar+
    states = moment * shares * core.spin + core.spin_entropy(shares) * core.entropy   # [shares] State-
    return states, core.potentials(states, moment, heat_capacity, reference)


# --- plumbing -------------------------------------------------------------------------
def slab(n_along: int, n_across: int) -> tuple[Topology, np.ndarray]:
    """The crystal's triangles and its vertices' places along it and across `[vertices, 2]`, a slab one width across
    with these many vertices along it and across it."""
    return shapes.box([np.linspace(0, length, n_along), np.linspace(0, 1, n_across)])


def timeline(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray]:
    """The run's steps, for the crystal at these places `[vertices, 2]`: the corner each step's phase starts at and the
    one it ends at, and how far between them the step's end lies, evenly over the phase's steps `[steps]`; the
    conductance of each vertex's link to its reservoir `[steps, vertices]`, each vertex on an end linking as much of the
    end's length as lies nearer to it than to its neighbours; each step's time `[steps]`; and the steps of each frame
    `[frames]`."""
    along, across = coordinates.T
    shares = np.where((across == 0) | (across == 1), 0.5, 1.0) / (np.unique(across).size - 1)   # [vertices]
    reservoir_links = link_conductance * shares * np.stack([0 * along, along == 0, along == length])   # [reservoirs, vertices]
    n_steps = np.repeat(n_phase_steps, n_phase_frames)                        # [frames]
    phases = np.repeat(np.arange(len(n_phase_frames)), n_phase_frames * n_phase_steps)   # [steps]
    # Each frame's time, growing by the phase's growth from one frame to the next and adding up to the phase's time.
    growths = [phase_growths[phase] ** np.arange(frames) for phase, frames in enumerate(n_phase_frames)]
    frame_times = np.concatenate([phase_times[phase] * growth / growth.sum() for phase, growth in enumerate(growths)])
    # Each step's place in its phase, the end of its share of the phase's steps.
    within = np.concatenate([(np.arange(frames * steps) + 1) / (frames * steps)
                             for frames, steps in zip(n_phase_frames, n_phase_steps)])   # [steps]
    return (phase_starts[phases], phase_ends[phases], within, reservoir_links[phase_reservoirs[phases]],
            np.repeat(frame_times / n_steps, n_steps), n_steps)


def reservoirs(coordinates: np.ndarray) -> np.ndarray:
    """Each vertex's reservoir's temperature `[vertices]`, for the crystal at these places `[vertices, 2]`: the bath's
    on its near half and the load's on its far half, of which only its ends link."""
    return np.where(coordinates[:, 0] < length / 2, bath_temperature, load_temperature)


def spin_draws(n_vertices: int) -> np.ndarray:
    """A draw for the spin drawn at each vertex `[vertices]`, a number between none and one, the same on every call."""
    return np.random.default_rng(spin_seed).random(n_vertices)


# --- scenes ---------------------------------------------------------------------------
def crystal(n_along: int = n_along, n_across: int = n_across) -> tuple[Topology, core.Point, core.Scalar,
                                                                      SparseExtensor, np.ndarray]:
    """The crystal of these many vertices along and across: its triangles, its vertices, places of weight one
    `Point-[vertices]`, each vertex's cell, a third of each triangle about it `Scalar+[vertices]`, the heat each cell
    loses to the conduction `(Antiscalar- <- Scalar+)[vertices <- vertices]`, and its vertices' places
    `[vertices, 2]`."""
    topology, coordinates = slab(n_along, n_across)
    vertices = (mv(core.Plane, coordinates) + mv.w).dual().field()           # Point-[vertices]
    areas, gradients = core.triangles(topology, vertices)
    cells = topology.incidence[2, 0].mean(axis=0).adjugate() * areas         # Scalar+[vertices]
    return topology, vertices, cells, core.conduction(topology, areas, gradients, conductivity), coordinates


def pictured(frames: Iterator[tuple[core.State, core.Potential, core.Magnetic, core.Scalar]], draws: core.Scalar
             ) -> Iterator[tuple[core.State, core.Potential, core.Magnetic, core.Scalar, core.Spin, core.State,
                                 core.Potential]]:
    """Each of these frames with what the picture draws of it besides: one spin of each cell `Spin-[vertices]`, drawn
    from the cell's two populations by these draws `Scalar+[vertices]`, and the line of the magnet's field on the
    chart, the salt's states held at that field from the chart's warmest to spins all but saturated `[samples] State-`,
    and their potentials."""
    for states, salt_potentials, field, links in frames:
        line_temperatures = moment * (field & core.spin) / mv.scalar(line_ratios[:, None])   # [samples] Scalar+
        line_states = core.state_at(field, line_temperatures, moment, heat_capacity, reference)   # [samples] State-
        yield (states, salt_potentials, field, links, core.drawn_spins(states, draws, moment), line_states,
               core.potentials(line_states, moment, heat_capacity, reference))


def refrigerator(n_along: int = n_along, n_across: int = n_across
                 ) -> tuple[Topology, core.Point, np.ndarray,
                            Iterator[tuple[core.State, core.Potential, core.Magnetic, core.Scalar, core.Spin,
                                           core.State, core.Potential]]]:
    """The crystal's run around the cycle, of these many vertices along and across: its triangles, its vertices
    `Point-[vertices]`, the vertices inside it, where its spins are drawn `[sites]`, and at each frame its states,
    potentials, the magnet's field and the links, with a spin of each cell and the line of the magnet's field as
    `pictured` adds them, from the crystal at the bath's temperature throughout in the field of the cycle's first
    corner."""
    topology, vertices, cells, conducting, coordinates = crystal(n_along, n_across)
    starts, ends, within, conductances, durations, n_steps = timeline(coordinates)
    fields = core.magnet(corner_fields(), starts, ends, within)               # [steps] Magnetic+
    links = mv.scalar(conductances[..., None]).field()                       # [steps] Scalar+[vertices]
    held = mv.scalar(reservoirs(coordinates)[:, None]).field()               # Scalar+[vertices]
    # The crystal at the bath's temperature at every vertex, in the field of the cycle's first corner.
    warm = mv.scalar(np.full((len(coordinates), 1), bath_temperature)).field()   # Scalar+[vertices]
    start = core.state_at(corner_fields()[0], warm, moment, heat_capacity, reference)   # State-[vertices]
    draws = mv.scalar(spin_draws(topology.n_vertices)[:, None]).field()      # Scalar+[vertices]
    frames = core.run(cells, conducting, start, fields, links, held, durations, n_steps, moment, heat_capacity,
                      reference)
    return topology, vertices, topology.interior[0], pictured(frames, draws)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.paramagnetic_salt import render

    save_figure(render.draw_equation_of_state(*equation_of_state(), moment), "paramagnetic_salt_state")
    save_figure(render.draw_cycle(*corners(), *lines()), "paramagnetic_salt_ideal_cycle")
    topology, vertices, _, _, _ = crystal()
    save_figure(render.draw_crystal(topology, vertices, topology.interior[0], bath_temperature, load_temperature),
                "paramagnetic_salt_slab")
    topology, vertices, spin_sites, frames = refrigerator()
    save_animation(render.animate_crystal(topology, vertices, spin_sites, bath_temperature, load_temperature,
                                          highest_field, *corners(), *lines(), frames), "paramagnetic_salt_crystal",
                   frame_duration_ms)


if __name__ == "__main__":
    main()
