"""Scenes for magnetic skyrmions: a lattice of skyrmions crystallizing from noise; the same noise relaxing beside it at no
field and above the skyrmions' range, into the helical spiral and the ferromagnet; the field lowered, raised until the
skyrmions vanish into the ferromagnet, and lowered again until the ferromagnet gives way to the spiral; and the lattice
driven by a current, drifting at the skyrmion Hall angle."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.quantum.skyrmions import core
from examples.util import shapes
from examples.util.topology import Topology

mv = core.mv
Upright = core.ga.gatype.from_blades("x y")                                   # Vector+
# The lattice's sites along each of its periods.
n_cells = 48
# The bonds' twist per unit length, in radians, its sense the crystal's handedness; the Gilbert damping, the time step,
# and the steps between frames; a frame's duration in milliseconds.
twist_angle = 0.55
damping, dt, stride = 0.5, 0.2, 12
frame_duration_ms = 40
# Crystallization: the applied field in units of the exchange, the frames, and the first frame's time step, which grows
# in proportion from frame to frame up to the time step, so that each frame shows about as much change as the last,
# from the noise's first gathering to the lattice's settling; the rounds of averaging that smooth the noise, and the
# noise's seed, which also seeds the thermal field.
crystal_field, n_crystal_frames, first_time_step = 0.4, 140, 0.014
n_smoothing, seed = 1, 2
# The regimes: the fields the same noise relaxes in beside the crystal's, none and above the skyrmions' collapse.
regime_fields = np.array([0.0, crystal_field, 1.0])
# The sweep: the frames at which the field passes its turning points, and the field there, lowered, raised until the
# skyrmions vanish, lowered to nothing and held; the temperature, in units of the exchange, whose kicks seed the spiral.
sweep_frames = np.array([0, 20, 55, 67, 99])
sweep_fields = np.array([0.4, 0.15, 0.95, 0.0, 0.0])
sweep_temperature = 0.02
# The drive: the current's direction, across the picture, the speed at which its spins drift, in lattice spacings per
# unit of time, and the frames before it is switched on and after.
current_direction = mv.x.dual()                                               # [] Point-
drift_speed, n_still_frames, n_driven_frames = 0.2, 10, 90


# --- plumbing -------------------------------------------------------------------------
def noise(topology: Topology, n_smoothing: int, seed: int) -> core.Spin:
    """Random spins `Spin-[vertices]`, smoothed over a site or two: each spin drawn at random, each then moved toward
    its neighbours' mean a few times by the lattice's Laplacian, the bonds' differences summed back, and turned back to
    unit length."""
    rng = np.random.default_rng(seed)
    spins = mv.bivector(rng.normal(size=(topology.n_vertices, 3))).dual().field()   # Spin-[vertices]
    for _ in range(n_smoothing):
        spins = spins - (~topology.d[0] * (topology.d[0] * spins)) / 12
    return spins.normalized()


# --- math -----------------------------------------------------------------------------
def film() -> tuple[Topology, core.Point, core.Point, core.Point, core.Rotor]:
    """The lattice's topology, its sites `Point-[vertices]`, its bonds `Point-[edges]`, the periods over which it
    repeats `[2] Point-`, and the bonds' rotors `Rotor+[edges]`, each turning a spin about its own bond by the twist."""
    topology, coordinates, steps, spans = shapes.flat_torus(n_cells, n_cells)
    vertices = mv(Upright, coordinates).dual().field()                        # Point-[vertices]
    bonds = mv(Upright, steps).dual().field()                                 # Point-[edges]
    periods = mv(Upright, spans).dual()                                       # [2] Point-
    twists = mv.antipseudoscalar([twist_angle]) * bonds                       # Field+[edges]
    carriers = (twists * (-1 / 2)).exp()                                      # Rotor+[edges]
    return topology, vertices, bonds, periods, carriers


def run(topology: Topology, bonds: core.Point, carriers: core.Rotor, spins: core.Spin, fields: np.ndarray,
        speeds: np.ndarray, time_steps: np.ndarray, temperature: float) -> Iterator[core.Spin]:
    """The spins `Spin-[vertices]` at each frame from these, in the applied field of each frame's strength `[frames]`,
    with the current's spins drifting at each frame's speed `[frames]`, each frame's steps of its time step `[frames]`,
    at the temperature."""
    areas, gradients, cells = core.triangles(topology, bonds)
    return core.dynamics(topology, carriers, areas, gradients, cells, spins, core.film * fields,
                         current_direction * speeds, time_steps, damping, temperature, stride, seed)


def crystallizing(topology: Topology, bonds: core.Point, carriers: core.Rotor) -> Iterator[core.Spin]:
    """The spins `Spin-[vertices]` at each frame, from smoothed noise, settling in the field into a lattice of
    skyrmions."""
    return run(topology, bonds, carriers, noise(topology, n_smoothing, seed), np.full(n_crystal_frames, crystal_field),
               np.zeros(n_crystal_frames), np.geomspace(first_time_step, dt, n_crystal_frames), 0.0)


def regimes(topology: Topology, bonds: core.Point, carriers: core.Rotor) -> Iterator[tuple[core.Spin, ...]]:
    """The spins `Spin-[vertices]` of each of the regimes' fields at each frame, the same noise relaxing in each as the
    crystal does in its own."""
    start = noise(topology, n_smoothing, seed)                                # Spin-[vertices]
    time_steps = np.geomspace(first_time_step, dt, n_crystal_frames)
    return zip(*(run(topology, bonds, carriers, start, np.full(n_crystal_frames, field), np.zeros(n_crystal_frames),
                     time_steps, 0.0) for field in regime_fields))


def sweeping(topology: Topology, bonds: core.Point, carriers: core.Rotor, crystal: core.Spin) -> Iterator[core.Spin]:
    """The spins `Spin-[vertices]` at each frame, from the crystal, as the field is lowered, raised and lowered again,
    warm enough for the ferromagnet to give way to the spiral."""
    fields = np.interp(np.arange(sweep_frames[-1] + 1), sweep_frames, sweep_fields)
    return run(topology, bonds, carriers, crystal, fields, np.zeros(len(fields)), np.full(len(fields), dt),
               sweep_temperature)


def driving(topology: Topology, bonds: core.Point, carriers: core.Rotor, crystal: core.Spin) -> Iterator[core.Spin]:
    """The spins `Spin-[vertices]` at each frame, from the crystal in its field, a current switched on after a few
    frames."""
    speeds = np.concatenate([np.zeros(n_still_frames), np.full(n_driven_frames, drift_speed)])
    return run(topology, bonds, carriers, crystal, np.full(len(speeds), crystal_field), speeds,
               np.full(len(speeds), dt), 0.0)


def transient_preview(time_steps: np.ndarray, noise_seed: int) -> None:
    """A preview of the crystallization's transient, for choosing its frames: the run from the noise of this seed in the
    crystal's field, each frame's steps of its time step `[frames]`, its animation saved, and its charge, its energy and
    the spins' change from frame to frame drawn against time."""
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.skyrmions import render

    topology, vertices, bonds, periods, carriers = film()
    n_frames = len(time_steps)
    states = list(run(topology, bonds, carriers, noise(topology, n_smoothing, noise_seed),
                      np.full(n_frames, crystal_field), np.zeros(n_frames), time_steps, 0.0))
    save_animation(render.animate(topology, vertices, bonds, periods, states),
                   f"skyrmions_transient_preview_seed_{noise_seed}", frame_duration_ms,
                   colors=render.n_animation_colours)
    times = stride * np.cumsum(time_steps) - stride * time_steps
    charges = np.array([core.charges(topology, spins).sites.sum().to_array() for spins in states])
    energies = np.array([core.energy(topology, carriers, crystal_field * core.film, spins).to_array()
                         for spins in states])
    # How far the spins turn from each frame to the next, the root mean square of their change over the sites.
    changes = np.array([np.sqrt((after - before).scalar_norm_squared().sites.mean().to_array())
                        for before, after in zip(states[:-1], states[1:])])
    save_figure(render.draw_transient(times, charges, energies / topology.n_vertices, changes), "skyrmions_transient")


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.skyrmions import render

    topology, vertices, bonds, periods, carriers = film()
    save_figure(render.draw_lattice(topology, vertices, bonds, periods), "skyrmions_lattice")
    crystal_states = list(crystallizing(topology, bonds, carriers))
    save_animation(render.animate(topology, vertices, bonds, periods, crystal_states), "skyrmions_crystallizing",
                   frame_duration_ms, colors=render.n_animation_colours)
    crystal = crystal_states[-1]
    save_animation(render.animate_together(topology, vertices, bonds, periods, regimes(topology, bonds, carriers)),
                   "skyrmions_regimes", frame_duration_ms, colors=render.n_animation_colours)
    sweep_states = list(sweeping(topology, bonds, carriers, crystal))
    save_animation(render.animate(topology, vertices, bonds, periods, sweep_states), "skyrmions_sweep",
                   frame_duration_ms, colors=render.n_animation_colours)
    drive_states = list(driving(topology, bonds, carriers, crystal))
    save_animation(render.animate(topology, vertices, bonds, periods, drive_states), "skyrmions_driven",
                   frame_duration_ms, colors=render.n_animation_colours)

    # checks
    # Damped and undriven, the energy never rises as the lattice crystallizes.
    energies = np.array([core.energy(topology, carriers, crystal_field * core.film, spins).to_array()
                         for spins in crystal_states])
    assert np.all(np.diff(energies) <= 1e-12 * np.abs(energies[0]))
    # Every step turns every spin by a rotor: after all the runs, every spin is still unit.
    np.testing.assert_allclose(drive_states[-1].norm().batch().to_array(), 1.0, atol=1e-11)
    # The crystal's charge is a whole number, one per skyrmion, each a patch of spins pointing against the field.
    pointing_down = (core.film & (topology.incidence[2, 0].mean(axis=0) * crystal)) < 0
    np.testing.assert_allclose(core.charges(topology, crystal).sites.sum().to_array(),
                               -topology.subset(np.asarray(pointing_down), 2).n_components, atol=1e-9)
    # From the field's peak until it is low again, no skyrmion is left: the charge is none.
    ferromagnet = sweep_states[sweep_frames[2]:sweep_frames[3]]
    np.testing.assert_allclose([core.charges(topology, spins).sites.sum().to_array() for spins in ferromagnet], 0.0,
                               atol=1e-9)
    # The exchange carried across a twisted bond is the Heisenberg exchange, the Dzyaloshinskii-Moriya interaction, the
    # plane through the two spins paired with the bond's twist, and an anisotropy along the bond, of second order.
    tails, heads = topology.incidence[1, 0] * crystal                         # Spin-[edges], Spin-[edges]
    twist_planes = (mv.antipseudoscalar([twist_angle]) * bonds).normalized()  # Field+[edges]
    heisenberg = -np.cos(twist_angle) * (core.exchange(tails) & heads).sites.sum()
    dzyaloshinskii_moriya = np.sin(twist_angle) * ((heads ^ tails) & twist_planes.dual()).sites.sum()
    anisotropy = -(1 - np.cos(twist_angle)) * ((twist_planes & tails) * (twist_planes & heads)).sites.sum()
    zeeman = -(crystal_field * core.film & crystal).sites.sum()
    np.testing.assert_allclose(core.energy(topology, carriers, crystal_field * core.film, crystal).to_array(),
                               (heisenberg + dzyaloshinskii_moriya + anisotropy + zeeman).to_array(), rtol=1e-12)


if __name__ == "__main__":
    main()
