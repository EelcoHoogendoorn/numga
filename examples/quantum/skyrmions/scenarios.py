"""Scenarios for magnetic skyrmions cover crystallization from smoothed random spins, side-by-side relaxation into spiral, crystal, and ferromagnetic regimes, hysteretic collapse under a field sweep, and skyrmion Hall drift under a spin-polarized current."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.quantum.skyrmions import core
from examples.util import shapes
from numga import Algebra, stack
from numga.topology import Topology

mv = core.mv
Upright = core.ga.gatype.vector().derive.within(Algebra("x+y+"))              # Vector+
n_sites = 48
twist_angle = 0.55
damping, time_step, n_substeps = 0.5, 0.2, 12
frame_duration_ms = 40

# Crystallization uses geometrically growing time steps so early aggregation and late lattice relaxation advance evenly across frames.
crystal_field, n_crystal_frames, first_time_step = 0.4, 140, 0.014
n_smoothing, seed = 1, 2
crystal_time_steps = np.geomspace(first_time_step, time_step, n_crystal_frames)

# Side-by-side regimes compare zero field, the skyrmion-crystal field, and a field above collapse from identical initial spins.
regime_fields = np.array([0.0, crystal_field, 1.0])

# Field sweep lowers the field, raises it past skyrmion collapse, and drops to zero with weak thermal kicks that nucleate stripes.
sweep_turning_frames = np.array([0, 20, 55, 67, 99])
sweep_turning_fields = np.array([0.4, 0.15, 0.95, 0.0, 0.0])
n_sweep_frames = int(sweep_turning_frames[-1]) + 1
sweep_schedule = np.interp(np.arange(n_sweep_frames), sweep_turning_frames, sweep_turning_fields)
sweep_temperature = 0.02

# Current drive holds the crystal still briefly before switching on a rightward spin-polarized drift velocity.
current_direction = mv.x.dual()                                               # [] Point-
drift_speed, n_still_frames, n_driven_frames = 0.2, 10, 90
n_drive_frames = n_still_frames + n_driven_frames
drive_speeds = np.concatenate([np.zeros(n_still_frames), np.full(n_driven_frames, drift_speed)])


# --- plumbing -------------------------------------------------------------------------
def smoothed_noise(topology: Topology, n_rounds: int, noise_seed: int) -> core.Spin:
    """Gaussian random spins (`Spin-[V]`) seeded by `noise_seed` relax across bonds for `n_rounds` Laplacian averaging steps before normalization."""
    V, E, F = topology.n_elements
    rng = np.random.default_rng(noise_seed)
    spins = mv.bivector(rng.normal(size=(V, 3))).dual().field()               # Spin-[V]
    for _ in range(n_rounds):
        spins = spins - (topology.d[0].adjugate() * (topology.d[0] * spins)) / 12
    return spins.normalized()


# --- math -----------------------------------------------------------------------------
def torus_lattice(n_axis_sites: int) -> tuple[Topology, core.Point, core.Point, core.Point, core.Rotor]:
    """A flat-torus triangular lattice of `n_axis_sites` sites per period carries site positions (`Point-[V]`), bond steps (`Point-[E]`), periods (`[2] Point-`), and chiral bond rotors (`Rotor+[E]`)."""
    topology, coordinates, steps, spans = shapes.flat_torus(n_axis_sites, n_axis_sites)
    vertices = mv(Upright, coordinates).dual().field()                        # Point-[V]
    bonds = mv(Upright, steps).dual().field()                                 # Point-[E]
    periods = mv(Upright, spans).dual()                                       # [2] Point-
    twist = twist_angle * mv.xyz.dual()                                       # [] Scalar-
    twists = twist * bonds                                                    # Field+[E]
    carriers = (twists * (-1 / 2)).exp()                                      # Rotor+[E]
    return topology, vertices, bonds, periods, carriers


def trajectory(
    topology: Topology,
    bonds: core.Point,
    carriers: core.Rotor,
    spins: core.Spin,
    field_strengths: np.ndarray,
    speeds: np.ndarray,
    time_steps: np.ndarray,
    temperature: float,
) -> Iterator[core.Spin]:
    """Yields lattice spins (`Spin-[V]`) across frames for scalar schedules of perpendicular `field_strengths`, current `speeds`, and `time_steps`."""
    return core.dynamics(
        topology,
        bonds,
        carriers,
        spins,
        core.film * field_strengths,
        current_direction * speeds,
        time_steps,
        damping,
        temperature,
        n_substeps,
        seed,
    )


def crystallization(topology: Topology, bonds: core.Point, carriers: core.Rotor) -> Iterator[core.Spin]:
    """Yields lattice spins (`Spin-[V]`) as smoothed random noise relaxes into a hexagonal skyrmion crystal."""
    return trajectory(
        topology,
        bonds,
        carriers,
        smoothed_noise(topology, n_smoothing, seed),
        np.full(n_crystal_frames, crystal_field),
        np.zeros(n_crystal_frames),
        crystal_time_steps,
        0.0,
    )


def regime_comparison(topology: Topology, bonds: core.Point, carriers: core.Rotor) -> Iterator[tuple[core.Spin, ...]]:
    """Yields synchronized spin triples (`Spin-[V]`) relaxing from identical noise at zero, crystal, and high fields."""
    initial_spins = smoothed_noise(topology, n_smoothing, seed)               # Spin-[V]
    return zip(*(
        trajectory(
            topology,
            bonds,
            carriers,
            initial_spins,
            np.full(n_crystal_frames, field_strength),
            np.zeros(n_crystal_frames),
            crystal_time_steps,
            0.0,
        )
        for field_strength in regime_fields
    ))


def field_sweep(topology: Topology, bonds: core.Point, carriers: core.Rotor, crystal: core.Spin) -> Iterator[core.Spin]:
    """Yields lattice spins (`Spin-[V]`) starting from `crystal` as the perpendicular field swells, collapses, and warms the texture into stripes."""
    return trajectory(
        topology,
        bonds,
        carriers,
        crystal,
        sweep_schedule,
        np.zeros(n_sweep_frames),
        np.full(n_sweep_frames, time_step),
        sweep_temperature,
    )


def current_drive(topology: Topology, bonds: core.Point, carriers: core.Rotor, crystal: core.Spin) -> Iterator[core.Spin]:
    """Yields lattice spins (`Spin-[V]`) as a spin-polarized current sets `crystal` drifting at the skyrmion Hall angle."""
    return trajectory(
        topology,
        bonds,
        carriers,
        crystal,
        np.full(n_drive_frames, crystal_field),
        drive_speeds,
        np.full(n_drive_frames, time_step),
        0.0,
    )


def transient_preview(time_steps: np.ndarray, noise_seed: int) -> None:
    """A crystallization preview run from `noise_seed` writes its spin animation alongside a transient chart of charge, energy, and step change."""
    from examples.quantum.skyrmions import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, bonds, periods, carriers = torus_lattice(n_sites)
    V, E, F = topology.n_elements
    n_frames = len(time_steps)
    states: list[core.Spin] = []
    stream = (
        states.append(spins) or spins
        for spins in trajectory(
            topology,
            bonds,
            carriers,
            smoothed_noise(topology, n_smoothing, noise_seed),
            np.full(n_frames, crystal_field),
            np.zeros(n_frames),
            time_steps,
            0.0,
        )
    )
    save_animation(
        render.spin_frames(topology, vertices, bonds, periods, stream),
        f"skyrmions_transient_preview_seed_{noise_seed}",
        frame_duration_ms,
        colors=render.n_animation_colours,
    )
    times = n_substeps * np.cumsum(time_steps) - n_substeps * time_steps
    frame_charges = stack([core.charges(topology, spins).sites.sum() for spins in states])   # [n_frames] Scalar+
    frame_energies = stack([core.energy(topology, carriers, crystal_field * core.film, spins) for spins in states]) / V   # [n_frames] Scalar+
    frame_changes = stack([
        (after - before).scalar_norm_squared().sites.mean().square_root()
        for before, after in zip(states[:-1], states[1:])
    ])                                                                        # [n_frames - 1] Scalar+
    save_figure(render.transient_figure(times, frame_charges, frame_energies, frame_changes), "skyrmions_transient")


def main() -> None:
    """The scenario entry point writes the lattice figure and all four skyrmion animations before checking physical invariants on the streamed states."""
    from examples.quantum.skyrmions import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, bonds, periods, carriers = torus_lattice(n_sites)
    save_figure(render.lattice_figure(topology, vertices, bonds, periods), "skyrmions_lattice")

    crystal_states: list[core.Spin] = []
    crystal_stream = (crystal_states.append(spins) or spins for spins in crystallization(topology, bonds, carriers))
    save_animation(
        render.spin_frames(topology, vertices, bonds, periods, crystal_stream),
        "skyrmions_crystallizing",
        frame_duration_ms,
        colors=render.n_animation_colours,
    )
    crystal = crystal_states[-1]                                              # Spin-[V]

    save_animation(
        render.comparison_frames(topology, vertices, bonds, periods, regime_comparison(topology, bonds, carriers)),
        "skyrmions_regimes",
        frame_duration_ms,
        colors=render.n_animation_colours,
    )

    sweep_states: list[core.Spin] = []
    sweep_stream = (sweep_states.append(spins) or spins for spins in field_sweep(topology, bonds, carriers, crystal))
    save_animation(
        render.spin_frames(topology, vertices, bonds, periods, sweep_stream),
        "skyrmions_sweep",
        frame_duration_ms,
        colors=render.n_animation_colours,
    )

    drive_states: list[core.Spin] = []
    drive_stream = (drive_states.append(spins) or spins for spins in current_drive(topology, bonds, carriers, crystal))
    save_animation(
        render.spin_frames(topology, vertices, bonds, periods, drive_stream),
        "skyrmions_driven",
        frame_duration_ms,
        colors=render.n_animation_colours,
    )

    # checks
    # Cold, undriven Gilbert damping decreases the total energy monotonically across crystallization frames.
    crystal_energies = stack([
        core.energy(topology, carriers, crystal_field * core.film, spins) for spins in crystal_states
    ])                                                                        # [n_crystal_frames] Scalar+
    assert bool(((crystal_energies[1:] - crystal_energies[:-1]) <= 1e-10 * crystal_energies[0].abs()).all())
    # Rotor updates preserve unit spin norm across every run.
    for final_spins in (sweep_states[-1], drive_states[-1]):
        assert bool(((final_spins.norm().batch() - 1).abs() < 1e-9).all())
    # Total topological charge equals minus the count of reversed-spin skyrmion cores.
    reversed_faces = (core.film & (topology.incidence[2, 0].mean(axis=0) * crystal)) < 0
    n_skyrmions = topology.subset(np.asarray(reversed_faces), 2).n_components
    assert bool((core.charges(topology, crystal).sites.sum() + n_skyrmions).abs() < 1e-7)
    # Above the collapse field, every skyrmion has unwound into the uniform ferromagnet with zero charge.
    plateau_states = sweep_states[sweep_turning_frames[2]:sweep_turning_frames[3]]
    plateau_charges = stack([core.charges(topology, spins).sites.sum() for spins in plateau_states])   # [...] Scalar+
    assert bool((plateau_charges.abs() < 1e-7).all())
    # Rotor-transported bond exchange equals Heisenberg exchange plus Dzyaloshinskii-Moriya twist plus second-order bond anisotropy.
    tail_spins, head_spins = topology.incidence[1, 0] * crystal               # Spin-[E], Spin-[E]
    twist_planes = ((twist_angle * mv.xyz.dual()) * bonds).normalized()       # Field+[E]
    heisenberg = -np.cos(twist_angle) * (core.exchange(tail_spins) & head_spins).sites.sum()   # [] Scalar+
    dzyaloshinskii_moriya = np.sin(twist_angle) * ((head_spins ^ tail_spins) & twist_planes.dual()).sites.sum()   # [] Scalar+
    anisotropy = -(1 - np.cos(twist_angle)) * ((twist_planes & tail_spins) * (twist_planes & head_spins)).sites.sum()   # [] Scalar+
    zeeman = -(crystal_field * core.film & crystal).sites.sum()               # [] Scalar+
    total_energy = core.energy(topology, carriers, crystal_field * core.film, crystal)   # [] Scalar+
    decomposed_energy = heisenberg + dzyaloshinskii_moriya + anisotropy + zeeman         # [] Scalar+
    assert bool((total_energy - decomposed_energy).abs() < 1e-9 * total_energy.abs())


if __name__ == "__main__":
    main()
