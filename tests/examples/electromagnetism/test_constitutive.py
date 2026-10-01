"""Unit tests for the constitutive extensor: projectors, media, birefringence, Fresnel drag."""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

from numga import stack

from examples.electromagnetism.constitutive import core, scenarios, render

mv = core.mv
t = core.t
x = core.x
z = core.z


def phase_speeds(medium, direction, speeds):
    return core.minimum_speeds(speeds, core.solve_dispersion_scan(speeds, medium, direction))


SPEEDS = np.linspace(0.05, 1.5, 6001)
SCAN_TOLERANCE = SPEEDS[1] - SPEEDS[0]


def test_observer_projectors_are_complementary_idempotents():
    electric, magnetic = core.observer_projectors()
    np.testing.assert_allclose(electric(electric).kernel, electric.kernel, atol=1e-14)
    np.testing.assert_allclose(magnetic(magnetic).kernel, magnetic.kernel, atol=1e-14)
    np.testing.assert_allclose(electric(magnetic).kernel, 0.0, atol=1e-14)
    F = mv.tx * 1.0 + mv.ty * 2.0 + mv.tz * 5.0 + mv.yz * 3.0 + mv.xy * 4.0
    np.testing.assert_allclose((electric(F) + magnetic(F) - F).kernel, 0.0, atol=1e-14)


def test_isotropic_medium_speed_is_one_over_n():
    for eps, mu in ((2.25, 1.0), (4.0, 1.5)):
        chi = core.isotropic_medium(eps, mu)
        np.testing.assert_allclose(phase_speeds(chi, z, SPEEDS), [1.0 / np.sqrt(eps * mu)], atol=SCAN_TOLERANCE)


def test_crystal_is_birefringent_and_reduces_to_glass_when_isotropic():
    iso = core.crystal_medium(2.25, 2.25, 2.25, 1.0)
    glass = core.isotropic_medium(2.25, 1.0)
    np.testing.assert_allclose(iso.kernel, glass.kernel, atol=1e-14)

    crystal = core.crystal_medium(2.25, 1.5, 1.5, 1.0)
    np.testing.assert_allclose(phase_speeds(crystal, z, SPEEDS), [1.0 / np.sqrt(2.25), 1.0 / np.sqrt(1.5)], atol=SCAN_TOLERANCE)
    np.testing.assert_allclose(phase_speeds(crystal, x, SPEEDS), [1.0 / np.sqrt(1.5)], atol=SCAN_TOLERANCE)


def test_crystal_fields_satisfy_maxwell_and_lie_in_their_material_planes():
    crystal = core.crystal_medium(2.25, 1.5, 1.5, 1.0)
    speeds = 1.0 / np.sqrt(np.array([2.25, 1.5]))
    wave_covectors = t * speeds + z
    fields = core.field_eigenmodes(wave_covectors, crystal)
    electric, magnetic = core.observer_projectors()
    electric_planes = stack((mv.tx, mv.ty))
    magnetic_planes = stack((mv.xz, mv.yz))

    np.testing.assert_allclose((wave_covectors ^ fields).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(wave_covectors.commutator(crystal(fields)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((electric(fields) - electric_planes * (electric_planes | fields)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((magnetic(fields) + magnetic_planes * (magnetic_planes | fields)).kernel, 0.0, atol=1e-12)


def test_vacuum_wave_has_two_transverse_field_modes():
    vacuum = core.isotropic_medium(1.0)
    wave_covector = t + z
    _, singular_values, fields = core.wave_map(wave_covector, vacuum).svd()

    assert np.count_nonzero(singular_values.kernel < 1e-12) == 2
    np.testing.assert_allclose((wave_covector ^ fields[-2:]).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(wave_covector.commutator(vacuum(fields[-2:])).kernel, 0.0, atol=1e-12)


def test_ferrite_lifts_permeability_through_the_dual_field():
    ferrite = core.ferrite_medium(eps=2.25, mu_inv_x=1.0, mu_inv_y=0.5, mu_inv_z=1.0)
    np.testing.assert_allclose(phase_speeds(ferrite, z, SPEEDS), [1.0 / np.sqrt(4.5), 1.0 / np.sqrt(2.25)], atol=SCAN_TOLERANCE)


def test_axion_term_is_invisible_to_bulk_waves():
    glass = core.isotropic_medium(2.25, 1.0)
    for alpha in (0.4, -1.3):
        axion = core.axion_medium(glass, alpha)
        np.testing.assert_allclose(phase_speeds(axion, z, SPEEDS), phase_speeds(glass, z, SPEEDS), atol=SCAN_TOLERANCE)


def test_moving_glass_shows_exact_fresnel_drag():
    eps, mu = 2.25, 1.0
    refractive_index = np.sqrt(eps * mu)
    glass = core.isotropic_medium(eps, mu)
    for beta in (0.0, 0.3, -0.5):
        moving = core.boosted_medium(glass, beta, direction=z)
        np.testing.assert_allclose(phase_speeds(moving, z, SPEEDS), [(1.0 / refractive_index + beta) / (1.0 + beta / refractive_index)], atol=SCAN_TOLERANCE)


def test_fresnel_surface_and_drag_scenarios():
    angles, surfaces, speeds = scenarios.fresnel_surface_scenario(n_angles=12)
    assert len(angles) == 12
    assert surfaces["crystal"].shape == (len(speeds), 12)

    betas, v_down, v_up, eps, mu = scenarios.fresnel_drag_scenario(n_betas=5)
    assert len(betas) == 5
    refractive_index = np.sqrt(eps * mu)
    np.testing.assert_allclose(v_down, (1 / refractive_index + betas) / (1 + betas / refractive_index), atol=2e-3)
    np.testing.assert_allclose(v_up, (1 / refractive_index - betas) / (1 - betas / refractive_index), atol=2e-3)


def test_figures_and_animation_draw():
    """Each scenario feeds its figure; the figures and a short animation draw."""
    glass_modes, crystal_modes = scenarios.wave_comparison_scenario()
    figures = [
        render.draw_wave_comparison_figure(glass_modes, crystal_modes),
        render.draw_dispersion_figure(*scenarios.dispersion_scenario()),
        render.draw_polarizations_figure(scenarios.field_modes_scenario()),
        render.draw_fresnel_surface_figure(*scenarios.fresnel_surface_scenario(n_angles=24)),
        render.draw_fresnel_drag_figure(*scenarios.fresnel_drag_scenario(n_betas=5)),
    ]
    assert all(isinstance(figure, plt.Figure) for figure in figures)
    frames = render.animate_wave_propagation(crystal_modes, n_frames=4)
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
