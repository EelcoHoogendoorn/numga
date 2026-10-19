"""Scenes for neutrino flavor oscillations: coherent oscillation in vacuum, its fading as the wavepackets drift
apart, and the adiabatic conversion of electron neutrinos leaving the Sun."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.quantum.neutrino_isospin import core
from examples.quantum.neutrino_isospin.core import Bivector, Vector

# The vacuum oscillation's rate and mixing angle in degrees, and the steps over two wavelengths.
OMEGA = 1.0
THETA_DEGREES = 30.0
N_STEPS = 300
# The decoherence: its dephasing rate per unit length, the distance followed, and its steps.
GAMMA = 0.05
DECOHERENCE_DISTANCE = 100.0
N_DECOHERENCE_STEPS = 400
# The Sun: the mixing angle in degrees, the distance followed, its steps, the potential at the centre over the
# resonance's, and the scale on which the electrons' density falls.
SOLAR_THETA_DEGREES = 10.0
SOLAR_DISTANCE = 100.0
N_SOLAR_STEPS = 600
CORE_OVER_RESONANCE = 3.0
DENSITY_SCALE = 25.0
# The animation's frames, and the duration of each.
N_FRAMES = 50
DURATION_MS = 50


# --- math -----------------------------------------------------------------------------
def vacuum_oscillations(theta_degrees: float, omega: float, n_steps: int) -> tuple[Iterator[Vector], Bivector, np.ndarray]:
    """Coherent oscillation in vacuum over two wavelengths: the isospin vector at each step, the plane it turns
    in, and the distances."""
    plane = core.vacuum_hamiltonian(omega, np.radians(theta_degrees))           # [] Bivector
    wavelength = 4.0 * np.pi / omega
    distances, dx = np.linspace(0.0, 2.0 * wavelength, n_steps, retstep=True)
    transfer = core.evolution(core.coherent_generator(plane), dx)               # [] State <- State
    return core.evolve_channel(core.state(core.flavor_e), transfer, n_steps), plane, distances


def wavepacket_decoherence(theta_degrees: float, omega: float, gamma: float, distance: float, n_steps: int
                           ) -> tuple[Iterator[Vector], Bivector, np.ndarray, float]:
    """The oscillation fading as the mass states' wavepackets drift apart: the isospin vector at each step, the
    plane it turns in, the distances, and the electron flavor's probability it settles at."""
    theta = np.radians(theta_degrees)
    plane = core.vacuum_hamiltonian(omega, theta)                                # [] Bivector
    mass_plane = core.mixing_rotor(theta) >> core.phase_plane                    # [] Bivector
    distances, dx = np.linspace(0.0, distance, n_steps, retstep=True)
    rates = core.coherent_generator(plane) + core.dephasing_generator(mass_plane, gamma)   # [] State <- State
    settled = float(1.0 - 0.5 * np.sin(2.0 * theta) ** 2)
    return core.evolve_channel(core.state(core.flavor_e), core.evolution(rates, dx), n_steps), plane, distances, settled


def solar_adiabatic(theta_degrees: float, omega: float, distance: float, n_steps: int
                    ) -> tuple[Iterator[tuple[Vector, Bivector]], np.ndarray, np.ndarray]:
    """An electron neutrino leaving the Sun through a density that falls off slowly: the isospin vector and the
    plane at each step, the distances, and the matter potential along the way."""
    theta = np.radians(theta_degrees)
    vacuum = core.vacuum_hamiltonian(omega, theta)                               # [] Bivector
    distances, dx = np.linspace(0.0, distance, n_steps, retstep=True)
    potentials = CORE_OVER_RESONANCE * core.resonance_potential(omega, theta) * np.exp(-distances / DENSITY_SCALE)   # [steps]
    planes = core.matter_hamiltonian(vacuum, potentials)                         # [steps] Bivector
    return core.adiabatic_channel(core.state(core.flavor_e), core.coherent_generator(planes), planes, dx), distances, potentials


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.quantum.neutrino_isospin import render

    states, plane, distances = vacuum_oscillations(THETA_DEGREES, OMEGA, N_STEPS)
    save_figure(render.draw_precession(states, plane, distances), "neutrino_vacuum_oscillation")
    states, plane, distances, settled = wavepacket_decoherence(THETA_DEGREES, OMEGA, GAMMA, DECOHERENCE_DISTANCE, N_DECOHERENCE_STEPS)
    save_figure(render.draw_decoherence(states, plane, distances, settled), "neutrino_decoherence")
    steps, distances, potentials = solar_adiabatic(SOLAR_THETA_DEGREES, OMEGA, SOLAR_DISTANCE, N_SOLAR_STEPS)
    save_figure(render.draw_adiabatic(steps, distances, potentials), "neutrino_solar_msw")
    states, plane, distances = vacuum_oscillations(THETA_DEGREES, OMEGA, N_STEPS)
    save_animation(render.animate_precession(states, plane, distances, N_FRAMES), "neutrino_vacuum_precession", DURATION_MS)


if __name__ == "__main__":
    main()
