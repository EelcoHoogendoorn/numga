"""Scenarios for neutrino flavor oscillations and open quantum system dynamics.

Constructs 3 canonical physical scenarios:
1. Vacuum Oscillations: coherent rotation via extensor transfer maps.
2. Wavepacket Decoherence: Lindblad dephasing spiraling onto mass eigenstates.
3. Solar MSW Adiabatic Conversion: complete flavor conversion through decreasing solar density.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import numpy as np

from numga import stack
from examples.animation import save_animation, save_figure
from examples.quantum.neutrino_isospin import core, render
from examples.quantum.neutrino_isospin.core import Bivector, Vector


def vacuum_oscillations(
    theta_deg: float,
    omega: float,
    steps: int,
) -> tuple[str, Iterator[Vector], Bivector, np.ndarray]:
    """Coherent vacuum oscillations as an extensor transfer map."""
    theta = np.radians(theta_deg)
    b_vac = core.vacuum_hamiltonian(omega, theta)
    wavelength = 4.0 * np.pi / omega
    distances, dx = np.linspace(0.0, 2.0 * wavelength, steps, retstep=True)

    turning = core.coherent_generator(b_vac)
    transfer = core.evolution(turning, dx)
    initial_rho = core.state(core.flavor_e)
    return (
        f"Vacuum Oscillations (theta={int(theta_deg)} deg)",
        core.evolve_channel(initial_rho, transfer, steps),
        b_vac,
        distances,
    )


def wavepacket_decoherence(
    theta_deg: float,
    omega: float,
    gamma: float,
    steps: int,
) -> tuple[str, Iterator[Vector], Bivector, np.ndarray, float]:
    """Wavepacket separation dephasing into an incoherent mixture of mass eigenstates."""
    theta = np.radians(theta_deg)
    b_vac = core.vacuum_hamiltonian(omega, theta)
    rotor_mix = core.mixing_rotor(theta)
    mass_axis = rotor_mix >> core.flavor_mu

    distances, dx = np.linspace(0.0, 100.0, steps, retstep=True)

    turning = core.coherent_generator(b_vac)
    dephasing = core.dephasing_generator(mass_axis, gamma)
    generator = turning + dephasing

    transfer = core.evolution(generator, dx)
    initial_rho = core.state(core.flavor_e)
    asymptotic_pe = float(1.0 - 0.5 * np.sin(2.0 * theta) ** 2)

    return (
        "Wavepacket Decoherence",
        core.evolve_channel(initial_rho, transfer, steps),
        b_vac,
        distances,
        asymptotic_pe,
    )


def solar_adiabatic(
    theta_deg: float,
    omega: float,
    steps: int,
) -> tuple[str, Iterator[tuple[Vector, Bivector]], np.ndarray, np.ndarray]:
    """Solar MSW effect: adiabatic following across smoothly decreasing electron density."""
    theta = np.radians(theta_deg)
    b_vac = core.vacuum_hamiltonian(omega, theta)
    v_res = core.resonance_potential(omega, theta)

    distances, dx = np.linspace(0.0, 100.0, steps, retstep=True)
    v_profile = 3.0 * v_res * np.exp(-distances / 25.0)

    rates_profile = [
        (core.coherent_generator(core.matter_hamiltonian(b_vac, v)), core.matter_hamiltonian(b_vac, v))
        for v in v_profile
    ]
    initial_rho = core.state(core.flavor_e)

    return (
        "Solar MSW (Adiabatic Conversion)",
        core.adiabatic_channel(initial_rho, rates_profile, dx),
        distances,
        v_profile,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animate", action="store_true", help="Also save vacuum precession animation.")
    args = parser.parse_args()

    # 1. Vacuum oscillations
    vac_title, vac_states, b_vac, vac_dist = vacuum_oscillations(theta_deg=30.0, omega=1.0, steps=300)
    save_figure(render.draw_precession(vac_title, vac_states, b_vac, vac_dist), "neutrino_vacuum_oscillation")

    # 2. Wavepacket decoherence
    dec_title, dec_states, b_dec, dec_dist, asymp_pe = wavepacket_decoherence(
        theta_deg=30.0, omega=1.0, gamma=0.05, steps=400
    )
    save_figure(render.draw_decoherence(dec_title, dec_states, b_dec, dec_dist, asymp_pe), "neutrino_decoherence")

    # 3. Solar MSW adiabatic conversion
    sun_title, sun_steps, sun_dist, v_prof = solar_adiabatic(theta_deg=10.0, omega=1.0, steps=600)
    save_figure(render.draw_adiabatic(sun_title, sun_steps, sun_dist, v_prof), "neutrino_solar_msw")

    if args.animate:
        _, anim_states, b_anim, anim_dist = vacuum_oscillations(theta_deg=30.0, omega=1.0, steps=300)
        frames = render.animate_precession("Vacuum Precession", anim_states, b_anim, anim_dist, steps=50)
        save_animation(frames, "neutrino_vacuum_precession", 50)
