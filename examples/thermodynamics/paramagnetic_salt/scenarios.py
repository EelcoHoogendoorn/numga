"""A paramagnetic crystal slab cycles between a 1 K bath at its near end and a 10 mK load at its far end.

Six phases step the crystal around the magnetic refrigeration cycle: magnetizing to 4 T with the bath switch closed and
holding while warmth drains to the bath, demagnetizing in isolation toward the load temperature, demagnetizing further
to 1 mT with the load switch closed and holding while heat enters from the load, and magnetizing in isolation back to
the bath temperature. Helper builders also sample the paramagnetic equation of state and the ideal temperature-entropy
cycle."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from numga.sparse import SparseExtensor
from numga.topology import Topology
from examples.thermodynamics.paramagnetic_salt import core
from examples.util import shapes

mv = core.mv

# Bohr-magneton electron spin moment in kelvin per tesla, small constant lattice heat capacity per spin, and bath
# reference temperature where the lattice carries zero entropy.
moment, heat_capacity = 0.6717, 0.003
bath_temperature, load_temperature = 1.0, 0.01
reference = bath_temperature
# Highest magnetic field at the bath temperature, lowest field at the load temperature, and bisection steps for the
# adiabatic corners.
highest_field, lowest_field = 4.0, 0.001
n_halvings = 20
# Magnetization shares sampled along the equation-of-state curve.
n_shares, largest_share = 120, 0.995
# Constant-field reference curves on the temperature-entropy chart.
n_line_samples = 80
line_temperatures = np.geomspace([0.45, 5e-4], [1.6, 1.6], n_line_samples, axis=-1)   # [2, n_line_samples]
# Crystal aspect ratio, grid resolution, lattice thermal conductivity, and end-switch conductance per unit width.
length, n_along, n_across = 2.0, 17, 9
conductivity, link_conductance = 3.0, 100.0
# Deterministic seed for vertex spin thresholds and dimensionless field-to-temperature ratios along the live isofield.
spin_seed = 0
line_ratios = np.geomspace(1e-4, 15.0, n_line_samples)                        # [n_line_samples]
# Six cycle phases: bath magnetization, bath hold, isolated demagnetization, load demagnetization, load hold, and
# isolated magnetization. Contact phases sweep the field faster than thermal diffusion so temperature fronts cross the
# slab, while holds use geometrically growing frame intervals as the slab relaxes.
phase_starts, phase_ends = np.array([0, 1, 1, 2, 3, 3]), np.array([1, 1, 2, 3, 3, 0])
phase_reservoirs = np.array([1, 1, 0, 2, 2, 0])
n_phase_frames = np.array([22, 20, 16, 20, 12, 8])
phase_times = np.array([0.3, 0.6, 0.02, 0.1, 0.045, 0.02])
phase_growths = np.array([1.0, 1.06, 1.0, 1.0, 1.03, 1.0])
n_phase_steps = np.array([3, 3, 4, 5, 5, 4])
n_frames = int(n_phase_frames.sum())
n_steps = int((n_phase_frames * n_phase_steps).sum())
frame_duration_ms = 40


def corners() -> tuple[core.Scalar, core.Scalar]:
    """The ideal rectangular refrigeration cycle spans four corners with entropies and temperatures (`[4] Scalar+`) set
    by the bath and load isotherms."""
    temperatures = mv.scalar([[bath_temperature], [load_temperature]])        # [2] Scalar+
    fields = stack((highest_field * core.magnet_plane, lowest_field * core.magnet_plane))   # [2] Magnetic+
    high_field_entropy, low_field_entropy = mv.s & core.equilibrium_state(fields, temperatures, moment, heat_capacity,
                                                                          reference)
    return (stack((low_field_entropy, high_field_entropy, high_field_entropy, low_field_entropy)),
            mv.scalar([[bath_temperature], [bath_temperature], [load_temperature], [load_temperature]]))


def corner_fields() -> core.Magnetic:
    """Adiabatic entropy matching determines the applied magnetic fields (`[4] Magnetic+`) at the four corners of the
    ideal cycle."""
    return core.corner_fields(bath_temperature, load_temperature, highest_field, lowest_field, moment, heat_capacity,
                              reference, n_halvings)


def lines() -> tuple[core.State, core.Potential]:
    """The highest-field and lowest-field reference curves trace equilibrium states (`[2, n_line_samples] State-`) and
    potentials (`[2, n_line_samples] Potential+`) across `line_temperatures`."""
    fields = stack((highest_field * core.magnet_plane, lowest_field * core.magnet_plane))[:, None]   # [2, 1] Magnetic+
    states = core.equilibrium_state(fields, mv.scalar(line_temperatures[..., None]), moment, heat_capacity, reference)
    return states, core.potentials(states, moment, heat_capacity, reference)


def equation_of_state() -> tuple[core.State, core.Potential]:
    """Zero-lattice-entropy equilibrium states (`[n_shares] State-`) and potentials (`[n_shares] Potential+`) trace the
    magnetization curve from zero up to `largest_share`."""
    shares = mv.scalar(np.linspace(0, largest_share, n_shares)[:, None])      # [n_shares] Scalar+
    states = moment * shares * core.spin + core.spin_entropy(shares) * core.entropy   # [n_shares] State-
    return states, core.potentials(states, moment, heat_capacity, reference)


# --- plumbing -------------------------------------------------------------------------
def slab(n_along: int, n_across: int) -> tuple[Topology, np.ndarray]:
    """A rectangular grid of `n_along` by `n_across` vertices triangulates the crystal slab into a `Topology` with
    vertex coordinates (`[V, 2]`)."""
    return shapes.box([np.linspace(0, length, n_along), np.linspace(0, 1, n_across)])


def timeline(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray]:
    """Six cycle phases schedule corner endpoints, phase fractions (`[n_steps]`), boundary switch conductances
    (`[n_steps, V]`), step durations (`[n_steps]`), and step counts per frame (`[n_frames]`) from `coordinates`
    (`[V, 2]`)."""
    x_coords, y_coords = coordinates.T
    end_weights = np.where((y_coords == 0) | (y_coords == 1), 0.5, 1.0) / (np.unique(y_coords).size - 1)   # [V]
    reservoir_links = link_conductance * end_weights * np.stack([0 * x_coords, x_coords == 0, x_coords == length])   # [3, V]
    steps_per_frame = np.repeat(n_phase_steps, n_phase_frames)                # [n_frames]
    phases = np.repeat(np.arange(len(n_phase_frames)), n_phase_frames * n_phase_steps)   # [n_steps]
    # Frame intervals grow geometrically during relaxation holds while summing to each phase's total duration.
    growths = [phase_growths[phase] ** np.arange(frames) for phase, frames in enumerate(n_phase_frames)]
    frame_times = np.concatenate([phase_times[phase] * growth / growth.sum() for phase, growth in enumerate(growths)])
    phase_progress = np.concatenate([(np.arange(frames * steps) + 1) / (frames * steps)
                                     for frames, steps in zip(n_phase_frames, n_phase_steps)])   # [n_steps]
    return (phase_starts[phases], phase_ends[phases], phase_progress, reservoir_links[phase_reservoirs[phases]],
            np.repeat(frame_times / steps_per_frame, steps_per_frame), steps_per_frame)


def reservoirs(coordinates: np.ndarray) -> np.ndarray:
    """Vertices on the near and far halves of `coordinates` (`[V, 2]`) pair with `bath_temperature` and
    `load_temperature` (`[V]`)."""
    return np.where(coordinates[:, 0] < length / 2, bath_temperature, load_temperature)


def spin_draws(n_vertices: int) -> np.ndarray:
    """Seeded random generation produces deterministic uniform thresholds (`[n_vertices]`) in `[0, 1)` for sampling
    representative vertex spins."""
    return np.random.default_rng(spin_seed).random(n_vertices)


# --- scenes ---------------------------------------------------------------------------
def crystal(n_along: int = n_along, n_across: int = n_across) -> tuple[Topology, core.Point, core.Scalar,
                                                                      SparseExtensor, np.ndarray]:
    """Simplex gradients on the `n_along` by `n_across` slab define unit-weight `vertices` (`Point-[V]`), lumped
    `dual_areas` (`Scalar+[V]`), `lattice_conduction` (`(Antiscalar- <- Scalar+)[V <- V]`), and `coordinates`
    (`[V, 2]`)."""
    topology, coordinates = slab(n_along, n_across)
    vertices = (mv(core.Plane, coordinates) + mv.w).dual().field()            # Point-[V]
    areas, gradients = core.triangles(topology, vertices)
    dual_areas = topology.incidence[2, 0].mean(axis=0).adjugate() * areas     # Scalar+[V]
    lattice_conduction = core.conduction(topology, areas, gradients, conductivity)
    return topology, vertices, dual_areas, lattice_conduction, coordinates


def scene_frames(frames: Iterator[tuple[core.State, core.Potential, core.Magnetic, core.Scalar]], draws: core.Scalar
                 ) -> Iterator[tuple[core.State, core.Potential, core.Magnetic, core.Scalar, core.Spin, core.State,
                                     core.Potential]]:
    """Yields each simulation frame augmented with sampled vertex spins (`Spin-[V]`) and instantaneous constant-field
    states (`[n_line_samples] State-`) and potentials (`[n_line_samples] Potential+`)."""
    for states, salt_potentials, field, links in frames:
        field_temperatures = moment * (field & core.spin) / line_ratios       # [n_line_samples] Scalar+
        field_states = core.equilibrium_state(field, field_temperatures, moment, heat_capacity, reference)   # [n_line_samples] State-
        yield (states, salt_potentials, field, links, core.drawn_spins(states, draws, moment), field_states,
               core.potentials(field_states, moment, heat_capacity, reference))


def refrigerator(n_along: int = n_along, n_across: int = n_across
                 ) -> tuple[Topology, core.Point, np.ndarray,
                            Iterator[tuple[core.State, core.Potential, core.Magnetic, core.Scalar, core.Spin,
                                           core.State, core.Potential]]]:
    """An `n_along` by `n_across` crystal starting at `bath_temperature` in the initial corner field provides the
    `Topology`, `vertices` (`Point-[V]`), interior spin indices (`[V_interior]`), and streaming cycle frames."""
    topology, vertices, dual_areas, lattice_conduction, coordinates = crystal(n_along, n_across)
    V, E, F = topology.n_elements
    from_corners, to_corners, phase_progress, conductances, durations, steps_per_frame = timeline(coordinates)
    fields = core.magnet_fields(corner_fields(), from_corners, to_corners, phase_progress)   # [n_steps] Magnetic+
    links = mv.scalar(conductances[..., None]).field()                        # [n_steps] Scalar+[V]
    reservoir_temperatures = mv.scalar(reservoirs(coordinates)[:, None]).field()   # Scalar+[V]
    initial_temperatures = mv.scalar(np.full((V, 1), bath_temperature)).field()   # Scalar+[V]
    initial_states = core.equilibrium_state(corner_fields()[0], initial_temperatures, moment, heat_capacity,
                                            reference)                        # State-[V]
    draws = mv.scalar(spin_draws(V)[:, None]).field()                         # Scalar+[V]
    frames = core.run(dual_areas, lattice_conduction, initial_states, fields, links, reservoir_temperatures, durations,
                      steps_per_frame, moment, heat_capacity, reference)
    return topology, vertices, topology.interior[0], scene_frames(frames, draws)


def main() -> None:
    from examples.thermodynamics.paramagnetic_salt import render
    from examples.util.animation import save_animation, save_figure

    eos_states, eos_potentials = equation_of_state()
    save_figure(render.draw_equation_of_state(eos_states, eos_potentials, moment), "paramagnetic_salt_state")
    cycle_entropies, cycle_temperatures = corners()
    line_states, line_potentials = lines()
    save_figure(render.draw_cycle(cycle_entropies, cycle_temperatures, line_states, line_potentials),
                "paramagnetic_salt_ideal_cycle")
    topology, vertices, dual_areas, _, _ = crystal()
    save_figure(render.draw_crystal(topology, vertices, topology.interior[0], bath_temperature, load_temperature),
                "paramagnetic_salt_slab")
    topology, vertices, spin_sites, frame_stream = refrigerator()
    frames = list(frame_stream)
    save_animation(render.animate_crystal(topology, vertices, spin_sites, bath_temperature, load_temperature,
                                          highest_field, cycle_entropies, cycle_temperatures, line_states,
                                          line_potentials, frames), "paramagnetic_salt_crystal", frame_duration_ms)

    # checks
    curie_slope = (core.magnet_plane & eos_states[1]) / (
        (eos_potentials[1] & core.spin) / (eos_potentials[1] & core.entropy)
    )
    assert ((curie_slope - moment ** 2).abs() < 1e-3 * moment ** 2).all()
    assert ((cycle_entropies[0] - cycle_entropies[3]).abs() < 1e-9).all()
    assert ((cycle_entropies[1] - cycle_entropies[2]).abs() < 1e-9).all()
    assert ((dual_areas.sites.sum() - length).abs() < 1e-9).all()
    frame_entropies = stack([
        (dual_areas * (mv.s & frame_states)).sites.sum() for frame_states, *_ in frames
    ])                                                                        # [n_frames + 1] Scalar+
    cumulative_frames = np.cumsum(n_phase_frames)
    bath_end, demagnetized_end, load_end = cumulative_frames[1], cumulative_frames[2], cumulative_frames[4]
    assert (frame_entropies[bath_end] < frame_entropies[0]).all()
    assert ((frame_entropies[demagnetized_end] - frame_entropies[bath_end]).abs() < 1e-2 * frame_entropies[bath_end]).all()
    bath_temperatures = (frames[bath_end][1] & core.entropy).batch()
    demagnetized_temperatures = (frames[demagnetized_end][1] & core.entropy).batch()
    load_temperatures = (frames[load_end][1] & core.entropy).batch()
    assert (demagnetized_temperatures < bath_temperatures.min() / 50).all()
    assert (frame_entropies[load_end] > frame_entropies[demagnetized_end]).all()
    assert ((load_temperatures - load_temperature).abs() < 0.1 * load_temperature).all()
    assert ((frame_entropies[-1] - frame_entropies[0]).abs() < 1e-2 * frame_entropies[0]).all()


if __name__ == "__main__":
    main()
