"""A paramagnetic crystal slab cycles between a 1 K bath at its near end and a 10 mK load at its far end.

Six phases step the crystal around the magnetic refrigeration cycle: magnetizing to 4 T with the bath switch closed and
holding while warmth drains to the bath, demagnetizing in isolation toward the load temperature, demagnetizing further
to 1 mT with the load switch closed and holding while heat enters from the load, and magnetizing in isolation back to
the bath temperature. Helper builders also sample the paramagnetic equation of state and the ideal temperature-entropy
cycle."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import dim, stack
from numga.sparse import SparseExtensor
from numga.topology import Topology
from examples.thermodynamics.paramagnetic_salt import core
from examples.util import shapes

mv = core.mv

# Extensive crystal amounts carry dim.extent; intensive potentials and per-spin constants carry none.
moment_dim = dim.boltzmann * dim.temperature / dim.field
conductance_dim = dim.boltzmann * dim.extent / dim.time

# Bohr-magneton electron spin moment, small constant lattice heat capacity per spin, and bath reference temperature.
moment = mv.scalar(0.6717, unit=moment_dim)                                   # [] Scalar+, boltzmann*temperature/field
heat_capacity = mv.scalar(0.003, unit=dim.boltzmann)                          # [] Scalar+, boltzmann
bath_temperature, load_temperature = mv.scalar(1.0, unit=dim.temperature), mv.scalar(0.01, unit=dim.temperature)
reference = bath_temperature
# Highest magnetic field at the bath temperature, lowest field at the load temperature, and bisection steps for the
# adiabatic corners.
highest_field, lowest_field = mv.scalar(4.0, unit=dim.field), mv.scalar(0.001, unit=dim.field)
n_halvings = 20
# Magnetization shares sampled along the equation-of-state curve.
n_shares, largest_share = 120, 0.995
# Constant-field reference curves on the temperature-entropy chart.
n_line_samples = 80
line_temperatures = mv.scalar(np.geomspace([0.45, 5e-4], [1.6, 1.6], n_line_samples, axis=-1)[..., None], unit=dim.temperature)   # [2, n_line_samples] Scalar+, temperature
# Crystal aspect ratio, grid resolution, areal spin density, lattice thermal conductivity, and end-switch conductance.
slab_length, n_along, n_across = 2.0, 17, 9
length = mv.scalar(slab_length, unit=dim.length)                              # [] Scalar+, length
spin_density = mv.scalar(1.0, unit=dim.extent / dim.length**2)                # [] Scalar+, extent/length^2
conductivity = mv.scalar(3.0, unit=conductance_dim)                           # [] Scalar+, boltzmann*extent/time
link_conductance = mv.scalar(100.0, unit=conductance_dim)                     # [] Scalar+, boltzmann*extent/time
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
    """The ideal rectangular refrigeration cycle spans four corners with entropies (`[4] Scalar+, boltzmann*extent`) and temperatures
    (`[4] Scalar+, temperature`) set by the bath and load isotherms for one spin of extent."""
    temperatures = stack((bath_temperature, load_temperature))                # [2] Scalar+, temperature
    fields = stack((highest_field * core.magnet_plane, lowest_field * core.magnet_plane))   # [2] Magnetic+, field
    high_field_entropy, low_field_entropy = mv.s & core.equilibrium_state(
        fields, temperatures, core.one_spin, moment, heat_capacity, reference)
    return (stack((low_field_entropy, high_field_entropy, high_field_entropy, low_field_entropy)),
            stack((bath_temperature, bath_temperature, load_temperature, load_temperature)))


def corner_fields() -> core.Magnetic:
    """Adiabatic entropy matching determines the applied magnetic fields (`[4] Magnetic+, field`) at the four corners
    of the ideal cycle."""
    return core.corner_fields(bath_temperature, load_temperature, highest_field, lowest_field, moment, heat_capacity,
                              reference, n_halvings)


def lines() -> tuple[core.State, core.Potential]:
    """The highest-field and lowest-field reference curves trace single-spin equilibrium states
    (`[2, n_line_samples] State-`) and potentials (`[2, n_line_samples] Potential+`) across `line_temperatures`."""
    fields = stack((highest_field * core.magnet_plane, lowest_field * core.magnet_plane))[:, None]   # [2, 1] Magnetic+, field
    states = core.equilibrium_state(fields, line_temperatures, core.one_spin, moment, heat_capacity, reference)
    return states, core.potentials(states, core.one_spin, moment, heat_capacity, reference)


def equation_of_state() -> tuple[core.State, core.Potential]:
    """Zero-lattice-entropy single-spin equilibrium states (`[n_shares] State-`) and potentials
    (`[n_shares] Potential+`) trace the magnetization curve from zero up to `largest_share`."""
    shares = mv.scalar(np.linspace(0, largest_share, n_shares)[:, None])      # [n_shares] Scalar+
    states = (core.one_spin * moment) * shares * core.spin + (core.one_spin * core.boltzmann) * core.spin_entropy(shares) * core.entropy   # [n_shares] State-
    return states, core.potentials(states, core.one_spin, moment, heat_capacity, reference)


# --- plumbing -------------------------------------------------------------------------
def slab(n_along: int, n_across: int) -> tuple[Topology, np.ndarray]:
    """A rectangular grid of `n_along` by `n_across` vertices triangulates the crystal slab into a `Topology` with
    vertex coordinates (`[V, 2]`)."""
    return shapes.box([np.linspace(0, slab_length, n_along), np.linspace(0, 1, n_across)])


def timeline(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, core.Scalar, core.Scalar,
                                               np.ndarray]:
    """Six cycle phases schedule corner endpoints, phase fractions (`[n_steps]`), boundary switch conductances
    (`[n_steps] Scalar+[V], boltzmann*extent/time`), step durations (`[n_steps] Scalar+, time`), and step counts per frame (`[n_frames]`)
    from `coordinates` (`[V, 2]`)."""
    x_coords, y_coords = coordinates.T
    end_weights = np.where((y_coords == 0) | (y_coords == 1), 0.5, 1.0) / (np.unique(y_coords).size - 1)   # [V]
    reservoir_links = end_weights * np.stack([0 * x_coords, x_coords == 0, x_coords == slab_length])   # [3, V]
    steps_per_frame = np.repeat(n_phase_steps, n_phase_frames)                # [n_frames]
    phases = np.repeat(np.arange(len(n_phase_frames)), n_phase_frames * n_phase_steps)   # [n_steps]
    # Frame intervals grow geometrically during relaxation holds while summing to each phase's total duration.
    growths = [phase_growths[phase] ** np.arange(frames) for phase, frames in enumerate(n_phase_frames)]
    frame_times = np.concatenate([phase_times[phase] * growth / growth.sum() for phase, growth in enumerate(growths)])
    phase_progress = np.concatenate([(np.arange(frames * steps) + 1) / (frames * steps)
                                     for frames, steps in zip(n_phase_frames, n_phase_steps)])   # [n_steps]
    links = (mv.scalar(reservoir_links[phase_reservoirs[phases]][..., None]) * link_conductance).field()   # [n_steps] Scalar+[V], boltzmann*extent/time
    durations = mv.scalar(np.repeat(frame_times / steps_per_frame, steps_per_frame)[:, None], unit=dim.time)   # [n_steps] Scalar+, time
    return phase_starts[phases], phase_ends[phases], phase_progress, links, durations, steps_per_frame


def reservoirs(coordinates: np.ndarray) -> core.Scalar:
    """Vertices on the near and far halves of `coordinates` (`[V, 2]`) pair with `bath_temperature` and
    `load_temperature` (`Scalar+[V], temperature`)."""
    near = mv.scalar((coordinates[:, 0] < slab_length / 2)[:, None]).field()  # Scalar+[V]
    return bath_temperature * near + load_temperature * (1 - near)


def spin_draws(n_vertices: int) -> np.ndarray:
    """Seeded random generation produces deterministic uniform thresholds (`[n_vertices]`) in `[0, 1)` for sampling
    representative vertex spins."""
    return np.random.default_rng(spin_seed).random(n_vertices)


# --- scenes ---------------------------------------------------------------------------
def crystal(n_along: int = n_along, n_across: int = n_across) -> tuple[Topology, core.Point, core.Scalar,
                                                                       SparseExtensor, np.ndarray]:
    """Simplex gradients on the `n_along` by `n_across` slab define unit-weight `vertices` (`Point-[V], 1 | length`), lumped
    `cell_spins` (`Scalar+[V], extent`), `lattice_conduction` (`(Antiscalar- <- Scalar+)[V <- V]`), and `coordinates`
    (`[V, 2]`)."""
    topology, coordinates = slab(n_along, n_across)
    vertices = (mv(core.Plane, coordinates) * dim.length + mv.w).dual().field()   # Point-[V], 1 | length
    areas, gradients = core.triangles(topology, vertices)
    dual_areas = topology.incidence[2, 0].mean(axis=0).adjugate() * areas     # Scalar+[V], length^2
    cell_spins = spin_density * dual_areas                                    # Scalar+[V], extent
    lattice_conduction = core.conduction(topology, areas, gradients, conductivity)
    return topology, vertices, cell_spins, lattice_conduction, coordinates


def scene_frames(frames: Iterator[tuple[core.State, core.Potential, core.Magnetic, core.Scalar]],
                 cell_spins: core.Scalar, draws: core.Scalar
                 ) -> Iterator[tuple[core.State, core.Potential, core.Magnetic, core.Scalar,
                                     core.Spin, core.State, core.Potential]]:
    """Yields each simulation frame augmented with sampled vertex spins (`Spin-[V]`) and instantaneous
    constant-field single-spin states (`[n_line_samples] State-`) and potentials (`[n_line_samples] Potential+`)."""
    for states, salt_potentials, field, links in frames:
        field_temperatures = moment * (field & core.spin) / (core.boltzmann * line_ratios)   # [n_line_samples] Scalar+, temperature
        field_states = core.equilibrium_state(field, field_temperatures, core.one_spin, moment, heat_capacity, reference)   # [n_line_samples] State-
        yield (states, salt_potentials, field, links, core.drawn_spins(states, cell_spins, draws, moment), field_states,
               core.potentials(field_states, core.one_spin, moment, heat_capacity, reference))


def refrigerator(n_along: int = n_along, n_across: int = n_across
                 ) -> tuple[Topology, core.Point, core.Scalar, np.ndarray,
                            Iterator[tuple[core.State, core.Potential, core.Magnetic,
                                           core.Scalar, core.Spin, core.State,
                                           core.Potential]]]:
    """An `n_along` by `n_across` crystal starting at `bath_temperature` in the initial corner field provides the
    `Topology`, `vertices` (`Point-[V], 1 | length`), `cell_spins` (`Scalar+[V], extent`), interior spin indices (`[V_interior]`), and
    streaming cycle frames."""
    topology, vertices, cell_spins, lattice_conduction, coordinates = crystal(n_along, n_across)
    V, E, F = topology.n_elements
    from_corners, to_corners, phase_progress, links, durations, steps_per_frame = timeline(coordinates)
    fields = core.magnet_fields(corner_fields(), from_corners, to_corners, phase_progress)   # [n_steps] Magnetic+, field
    reservoir_temperatures = reservoirs(coordinates)                          # Scalar+[V], temperature
    initial_temperatures = mv.scalar(np.ones((V, 1))).field() * bath_temperature   # Scalar+[V], temperature
    initial_states = core.equilibrium_state(corner_fields()[0], initial_temperatures, cell_spins, moment, heat_capacity,
                                            reference)                        # State-[V]
    draws = mv.scalar(spin_draws(V)[:, None]).field()                         # Scalar+[V]
    frames = core.run(cell_spins, lattice_conduction, initial_states, fields, links, reservoir_temperatures, durations,
                      steps_per_frame, moment, heat_capacity, reference)
    return topology, vertices, cell_spins, topology.interior[0], scene_frames(frames, cell_spins, draws)


def main() -> None:
    from examples.thermodynamics.paramagnetic_salt import render
    from examples.util.animation import save_animation, save_figure

    eos_states, eos_potentials = equation_of_state()
    save_figure(render.draw_equation_of_state(eos_states, eos_potentials, moment), "paramagnetic_salt_state")
    cycle_entropies, cycle_temperatures = corners()
    line_states, line_potentials = lines()
    save_figure(render.draw_cycle(cycle_entropies, cycle_temperatures, line_states, line_potentials),
                "paramagnetic_salt_ideal_cycle")
    topology, vertices, cell_spins, _, _ = crystal()
    save_figure(render.draw_crystal(topology, vertices, topology.interior[0], bath_temperature, load_temperature),
                "paramagnetic_salt_slab")
    topology, vertices, cell_spins, spin_sites, frame_stream = refrigerator()
    frames = list(frame_stream)
    save_animation(render.animate_crystal(topology, vertices, cell_spins, spin_sites, bath_temperature,
                                          load_temperature, highest_field, cycle_entropies, cycle_temperatures,
                                          line_states, line_potentials, frames),
                   "paramagnetic_salt_crystal", frame_duration_ms)

    # checks
    curie_slope = (core.magnet_plane & eos_states[1]) / (
        (eos_potentials[1] & core.spin) / (eos_potentials[1] & core.entropy)
    )
    expected_curie = core.one_spin * moment ** 2 / core.boltzmann
    assert ((curie_slope - expected_curie).abs() < 1e-3 * expected_curie).all()
    entropy_scale = core.one_spin * core.boltzmann
    assert ((cycle_entropies[0] - cycle_entropies[3]).abs() < 1e-9 * entropy_scale).all()
    assert ((cycle_entropies[1] - cycle_entropies[2]).abs() < 1e-9 * entropy_scale).all()
    expected_spins = spin_density * length * mv.scalar(1.0, unit=dim.length)
    assert ((cell_spins.sites.sum() - expected_spins).abs() < 1e-9 * expected_spins).all()
    frame_entropies = stack([
        (mv.s & frame_states).sites.sum() for frame_states, *_ in frames
    ])                                                                        # [n_frames + 1] Scalar+, boltzmann*extent
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
