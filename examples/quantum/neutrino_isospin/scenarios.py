"""Scenes for neutrino flavor oscillations: coherent oscillation in vacuum, wavepacket decoherence over
long baselines, and adiabatic MSW flavor conversion leaving the solar core."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.quantum.neutrino_isospin import core
from examples.quantum.neutrino_isospin.core import Bivector, Isospin

omega = 1.0
theta_degrees = 30.0
n_steps = 300
gamma = 0.05
decoherence_distance = 100.0
n_decoherence_steps = 400
solar_theta_degrees = 10.0
solar_distance = 100.0
n_solar_steps = 600
core_over_resonance = 3.0
density_scale = 25.0
n_frames = 63
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def vacuum_oscillations(theta_degrees: float, omega: float, n_steps: int
                        ) -> tuple[Iterator[Isospin], Bivector, np.ndarray]:
    """Coherent vacuum propagation over two oscillation wavelengths yields the `n_steps` isospin stream,
    Hamiltonian plane (`[] Bivector+`), and baseline distances."""
    plane = core.vacuum_hamiltonian(omega, np.radians(theta_degrees))        # [] Bivector+
    wavelength = 4.0 * np.pi / omega
    distances, step_size = np.linspace(0.0, 2.0 * wavelength, n_steps, retstep=True)
    rates = core.coherent_generator(plane)                                   # [] State- <- State-
    transfer = core.step_evolution(rates, step_size)                         # [] State- <- State-
    return core.channel_isospins(core.density_state(core.flavor_e), transfer, n_steps), plane, distances


def wavepacket_decoherence(theta_degrees: float, omega: float, gamma: float, distance: float, n_steps: int
                           ) -> tuple[Iterator[Isospin], Bivector, np.ndarray, float]:
    """Wavepacket separation at rate `gamma` in the mass plane damps the `n_steps` isospin stream toward the
    incoherent survival probability `1 - 0.5 * sin(2 * theta)**2`."""
    theta = np.radians(theta_degrees)
    plane = core.vacuum_hamiltonian(omega, theta)                            # [] Bivector+
    mixing = (core.mixing_plane * (-theta)).exp()                            # [] Rotor+
    mass_plane = mixing >> core.phase_plane                                  # [] Bivector+
    distances, step_size = np.linspace(0.0, distance, n_steps, retstep=True)
    turning = core.coherent_generator(plane)                                 # [] State- <- State-
    dephasing = gamma * core.relaxation(mass_plane)                          # [] State- <- State-
    transfer = core.step_evolution(turning + dephasing, step_size)           # [] State- <- State-
    asymptotic_probability = float(1.0 - 0.5 * np.sin(2.0 * theta) ** 2)
    return (core.channel_isospins(core.density_state(core.flavor_e), transfer, n_steps),
            plane, distances, asymptotic_probability)


def solar_adiabatic(theta_degrees: float, omega: float, distance: float, n_steps: int
                    ) -> tuple[Iterator[tuple[Isospin, Bivector]], Bivector, Bivector, np.ndarray, np.ndarray]:
    """Exponentially falling solar electron potential tilts the local Hamiltonian plane across `n_steps` steps
    from above MSW resonance down toward vacuum."""
    theta = np.radians(theta_degrees)
    vacuum = core.vacuum_hamiltonian(omega, theta)                           # [] Bivector+
    resonance = omega * np.cos(2.0 * theta)
    distances, step_size = np.linspace(0.0, distance, n_steps, retstep=True)
    potentials = core_over_resonance * resonance * np.exp(-distances / density_scale)
    planes = vacuum + potentials * core.phase_plane                          # [n_steps] Bivector+
    rates = core.coherent_generator(planes)                                  # [n_steps] State- <- State-
    stream = core.adiabatic_isospins(core.density_state(core.flavor_e), rates, planes, step_size)
    return stream, planes[0], planes[-1], distances, potentials


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.neutrino_isospin import render

    def tap_stream(stream: Iterator) -> tuple[Iterator, list]:
        items: list = []

        def passing() -> Iterator:
            for item in stream:
                items.append(item)
                yield item

        return passing(), items

    vacuum_stream, vacuum_plane, vacuum_distances = vacuum_oscillations(theta_degrees, omega, n_steps)
    passing_vacuum, vacuum_history = tap_stream(vacuum_stream)
    save_animation(render.precession_animation(vacuum_plane, vacuum_distances, n_frames, passing_vacuum),
                   "neutrino_vacuum_precession", frame_duration_ms)
    save_figure(render.precession_figure(vacuum_history, vacuum_plane, vacuum_distances),
                "neutrino_vacuum_oscillation")

    decoherence_stream, decoherence_plane, decoherence_distances, asymptotic_probability = wavepacket_decoherence(
        theta_degrees, omega, gamma, decoherence_distance, n_decoherence_steps
    )
    passing_decoherence, decoherence_history = tap_stream(decoherence_stream)
    save_animation(
        render.decoherence_animation(decoherence_plane, decoherence_distances, asymptotic_probability,
                                     n_frames, passing_decoherence),
        "neutrino_decoherence", frame_duration_ms,
    )
    save_figure(
        render.decoherence_figure(decoherence_history, decoherence_plane, decoherence_distances, asymptotic_probability),
        "neutrino_decoherence",
    )

    solar_stream, start_plane, end_plane, solar_distances, potentials = solar_adiabatic(
        solar_theta_degrees, omega, solar_distance, n_solar_steps
    )
    passing_solar, solar_history = tap_stream(solar_stream)
    save_animation(
        render.adiabatic_animation(solar_distances, potentials, start_plane, end_plane, n_frames, passing_solar),
        "neutrino_solar_msw", frame_duration_ms,
    )
    save_figure(render.adiabatic_figure(solar_history, solar_distances, potentials), "neutrino_solar_msw")

    # checks
    vacuum_path = stack(vacuum_history)                                      # [n_steps] Isospin-
    vacuum_electron, vacuum_muon = core.flavor_probabilities(vacuum_path)    # [n_steps] Scalar+, [n_steps] Scalar+
    expected_vacuum = 1.0 - np.sin(2.0 * np.radians(theta_degrees)) ** 2 * np.sin(0.5 * omega * vacuum_distances) ** 2
    assert ((vacuum_electron + vacuum_muon - 1).abs() < 1e-10).all()
    assert ((vacuum_electron - expected_vacuum).abs() < 1e-4).all()

    decoherence_path = stack(decoherence_history)                            # [n_decoherence_steps] Isospin-
    decoherence_electron, _ = core.flavor_probabilities(decoherence_path)    # [n_decoherence_steps] Scalar+
    assert ((decoherence_electron[-1] - asymptotic_probability).abs() < 2e-3).all()

    solar_path = stack([spin for spin, _ in solar_history])                  # [n_solar_steps] Isospin-
    solar_electron, _ = core.flavor_probabilities(solar_path)                # [n_solar_steps] Scalar+
    assert (solar_electron[0] > 0.99).all() and (solar_electron[-1] < 0.05).all()


if __name__ == "__main__":
    main()
