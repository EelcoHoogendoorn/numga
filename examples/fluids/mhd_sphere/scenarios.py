"""Differential solar-like rotation on a unit sphere winds a pole-to-pole magnetic field into azimuthal bands until thin current sheets reconnect into magnetic islands."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice, tee

import numpy as np

from numga import stack
from examples.fluids.mhd_sphere import core
from examples.util import shapes

mv = core.mv
n_levels = 5
rotation = 1.0
field_strength = 0.25
viscosity = 1e-3
resistivity = 1e-3
dt = 0.02
n_steps = 450
stride = 4
frame_duration_ms = 40
n_animation_colours = 32


# --- plumbing -------------------------------------------------------------------------
def sphere(n_levels: int) -> tuple[core.Topology, core.Point]:
    """Subdividing an icosahedron `n_levels` times produces a triangulated unit sphere with spatial direction vertices (`Point-[V]`)."""
    topology, coordinates = shapes.sphere(n_levels)
    return topology, mv(core.Plane, coordinates).dual().field()


# --- math -----------------------------------------------------------------------------
def initial_state(surface: core.Surface, vertices: core.Point) -> core.Scalar:
    """A degree-three zonal stream function and an equatorial dipole flux function initialize the vorticity and flux state (`[2] Scalar+[V]`)."""
    heights = mv.z & vertices                                                        # Scalar+[V]
    stream = rotation * (5 * heights * heights * heights - 3 * heights) / 2          # Scalar+[V]
    flux = field_strength * (mv.x & vertices)                                        # Scalar+[V]
    return stack((core.laplacian(surface, stream), flux))                            # [2] Scalar+[V]


def energies(surface: core.Surface, states: core.Scalar) -> core.Scalar:
    """Pairing stream and flux functions with Dirichlet stiffness integrates kinetic and magnetic energies (`[2, ...] Scalar+`) for `states` (`[..., 2] Scalar+[V]`)."""
    vorticity, flux = states[..., 0], states[..., 1]                                 # [...] Scalar+[V], [...] Scalar+[V]
    stream = core.stream_function(surface, vorticity)                                # [...] Scalar+[V]
    kinetic = (stream & surface.stiffness(stream)).sites.sum() / 2                   # [...] Scalar+
    magnetic = (flux & surface.stiffness(flux)).sites.sum() / 2                      # [...] Scalar+
    return stack((kinetic, magnetic))                                                # [2, ...] Scalar+


def frame_channels(surface: core.Surface,
                   states: Iterable[core.Scalar]) -> Iterator[tuple[core.Scalar, core.Scalar, core.Scalar]]:
    """Each vorticity-flux state yields vertex current density (`Scalar+[V]`), vertex flux (`Scalar+[V]`), and triangle field strength (`Scalar+[F]`)."""
    for state in states:
        flux = state[1]                                                              # Scalar+[V]
        yield core.laplacian(surface, flux), flux, surface.gradient(flux).norm()


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.mhd_sphere import render

    topology, vertices = sphere(n_levels)
    surface = core.surface(topology, vertices)
    initial = initial_state(surface, vertices)                                       # [2] Scalar+[V]
    n_frames = n_steps // stride + 1
    animated_states, recorded_states = tee(
        islice(core.evolution(surface, initial, viscosity, resistivity, dt, n_steps), 0, None, stride)
    )
    save_animation(render.animation(topology, vertices, frame_channels(surface, animated_states)), "mhd_sphere",
                   frame_duration_ms, colors=n_animation_colours)
    states = stack(tuple(recorded_states))                                           # [n_frames, 2] Scalar+[V]
    kinetic, magnetic = energies(surface, states)                                    # [n_frames] Scalar+, [n_frames] Scalar+
    times = np.arange(n_frames) * (stride * dt)
    save_figure(render.energy_figure(times, kinetic, magnetic), "mhd_sphere_energies")

    # checks
    # Ideal rates conserve total energy, cross helicity, and mean-square flux via bracket antisymmetry.
    vorticity, flux = states[n_frames // 4]                                          # Scalar+[V], Scalar+[V]
    vorticity_rate, flux_rate = core.rate(surface, stack((vorticity, flux)), 0.0, 0.0)   # Scalar+[V], Scalar+[V]
    stream = core.stream_function(surface, vorticity)                                # Scalar+[V]
    current = core.laplacian(surface, flux)                                          # Scalar+[V]
    energy_rate = ((stream & surface.mass(vorticity_rate)) + (current & surface.mass(flux_rate))).sites.sum()          # Scalar+
    cross_helicity_rate = ((flux & surface.mass(vorticity_rate)) + (vorticity & surface.mass(flux_rate))).sites.sum()  # Scalar+
    flux_square_rate = (flux & surface.mass(flux_rate)).sites.sum()                                                    # Scalar+
    exchange_scale = (current & surface.mass(core.bracket(surface, stream, flux))).sites.sum().abs()                   # Scalar+
    for invariant_rate in (energy_rate, cross_helicity_rate, flux_square_rate):
        assert (invariant_rate.abs() < 1e-11 * exchange_scale).all()
    total_energy = kinetic + magnetic                                                # [n_frames] Scalar+
    assert (total_energy[1:] < total_energy[:-1]).all()


if __name__ == "__main__":
    main()
