"""Scenes of the electromagnetic constitutive example: one function per figure.

Each runs the mathematics in `core` and returns what its figure draws.
"""

from __future__ import annotations

import numpy as np

from numga import Extensor

from examples.electromagnetism.constitutive import core

# Parameters:
EPS_GLASS = 2.25
MU_GLASS = 1.0
BETA_BOOST = 0.3
AXION_ALPHA = 0.4


def dispersion_scenario():
    """Execute 1D dispersion scan along z across all media."""
    speeds = np.linspace(0.05, 1.5, 3001)

    glass = core.isotropic_medium(EPS_GLASS, MU_GLASS)
    axion = core.axion_medium(glass, AXION_ALPHA)
    crystal = core.crystal_medium(eps_x=2.25, eps_y=1.5, eps_z=1.5, mu=1.0)
    ferrite = core.ferrite_medium(eps=2.25, mu_inv_x=1.0, mu_inv_y=0.5, mu_inv_z=1.0)
    moving_down = core.boosted_medium(glass, beta=BETA_BOOST, direction=core.z)
    moving_up = core.boosted_medium(glass, beta=BETA_BOOST, direction=-core.z)

    media_dict = {
        "glass at rest": (glass, core.z),
        "glass with axion term": (axion, core.z),
        "crystal along z": (crystal, core.z),
        "ferrite along z": (ferrite, core.z),
        "glass moving with wave": (moving_down, core.z),
        "glass moving against wave": (moving_up, core.z),
    }

    curves = {}
    for name, (medium, k_dir) in media_dict.items():
        svals = core.solve_dispersion_scan(speeds, medium, direction=k_dir)
        curves[name] = svals

    refractive_index = np.sqrt(EPS_GLASS * MU_GLASS)
    with_flow = (1.0 / refractive_index + BETA_BOOST) / (1.0 + BETA_BOOST / refractive_index)
    against_flow = (1.0 / refractive_index - BETA_BOOST) / (1.0 - BETA_BOOST / refractive_index)

    expected = {
        "glass at rest": [1.0 / refractive_index],
        "glass with axion term": [1.0 / refractive_index],
        "crystal along z": [1.0 / np.sqrt(2.25), 1.0 / np.sqrt(1.5)],
        "ferrite along z": [1.0 / np.sqrt(2.25 * 2.0), 1.0 / np.sqrt(2.25)],
        "glass moving with wave": [with_flow],
        "glass moving against wave": [against_flow],
    }

    return speeds, curves, expected


def polarization_scenario() -> list[tuple[float, Extensor]]:
    """Extract physical polarization states for the slow and fast birefringent modes in the crystal."""
    crystal = core.crystal_medium(eps_x=2.25, eps_y=1.5, eps_z=1.5, mu=1.0)
    speeds = np.linspace(0.05, 1.5, 3001)
    svals = core.solve_dispersion_scan(speeds, crystal, direction=core.z)
    mins = core.minimum_speeds(speeds, svals, threshold=2e-3)

    modes = []
    for v in mins:
        k = v * core.t + core.z
        modes.append((float(v), core.polarization_eigenmodes(k, crystal)))
    return modes


def fresnel_surface_scenario(
    n_angles: int = 72,
):
    """Scan the wave map over phase speed and propagation angle in the xz-plane.

    Returns the angles, the smallest singular value per medium [n_speeds, n_angles], and the speeds.
    """
    angles = np.linspace(0, 2 * np.pi, n_angles)
    speeds = np.linspace(0.4, 1.0, 601)
    directions = np.sin(angles) * core.x + np.cos(angles) * core.z      # [n_angles] Vector

    glass = core.isotropic_medium(EPS_GLASS, MU_GLASS)
    crystal = core.crystal_medium(eps_x=2.25, eps_y=1.5, eps_z=1.5, mu=1.0)
    moving = core.boosted_medium(glass, beta=0.35, direction=core.z)

    surfaces = {
        name: core.solve_dispersion_scan(speeds[:, None], medium, direction=directions)
        for name, medium in [("glass (rest)", glass), ("crystal", crystal), ("moving glass (beta=0.35)", moving)]
    }
    return angles, surfaces, speeds


def fresnel_drag_scenario(
    n_betas: int = 15,
):
    """Compute downstream and upstream phase speeds across medium velocity beta, in glass of the given eps and mu."""
    betas = np.linspace(-0.6, 0.6, n_betas)
    speeds = np.linspace(0.05, 1.2, 1151)
    v_down, v_up = core.fresnel_drag_velocities(EPS_GLASS, MU_GLASS, betas, speeds)
    return betas, v_down, v_up, EPS_GLASS, MU_GLASS


def wave_comparison_scenario() -> tuple[list[tuple[float, Extensor]], list[tuple[float, Extensor]]]:
    """Wave modes along z: both polarizations at one speed in glass, split by speed in the crystal."""
    v_slow = 1.0 / np.sqrt(EPS_GLASS)
    v_fast = 1.0 / np.sqrt(1.5)
    glass_modes = [(v_slow, core.x), (v_slow, core.y)]
    crystal_modes = [(v_slow, core.x), (v_fast, core.y)]
    return glass_modes, crystal_modes


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.electromagnetism.constitutive import render

    glass_modes, crystal_modes = wave_comparison_scenario()
    save_figure(render.draw_wave_comparison_figure(glass_modes, crystal_modes), "constitutive_birefringence_3d")
    save_figure(render.draw_dispersion_figure(*dispersion_scenario()), "constitutive_dispersion")
    save_figure(render.draw_polarizations_figure(polarization_scenario()), "constitutive_polarizations")
    save_figure(render.draw_fresnel_surface_figure(*fresnel_surface_scenario()), "constitutive_fresnel_surface")
    save_figure(render.draw_fresnel_drag_figure(*fresnel_drag_scenario()), "constitutive_fresnel_drag")
    save_animation(render.animate_wave_propagation(crystal_modes), "constitutive_wave", 50)
