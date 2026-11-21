"""Unit tests for electromagnetic constitutive maps, stress-energy currents, birefringence, and Fresnel drag."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import dim

from examples.electromagnetism.constitutive import core, render, scenarios

mv = core.mv
t, t_s, x, z = mv.t, core.t_s, mv.x, mv.z
c, mu_0, eps_0, wavenumber = core.c, core.mu_0, core.eps_0, core.wavenumber
Field = core.Field

trial_speeds = np.linspace(0.05, 1.5, 6001) * c
scan_tolerance = trial_speeds[1] - trial_speeds[0]
# The observer's direction in time pairs to one with t.
time = -t.dual()
# Reflection across t keeps magnetic planes and reverses electric ones.
electric_part = (Field - t * Field * t.inverse()) / 2
magnetic_part = Field - electric_part


def isotropic_medium(relative_permittivity: float, relative_permeability: float) -> core.Constitutive:
    """Isotropic medium with uniform relative permittivity and permeability across all three spatial axes."""
    return core.medium(np.full(3, relative_permittivity) * eps_0, np.full(3, 1.0 / relative_permeability) / mu_0)


def phase_speeds(medium_map: core.Constitutive, direction: core.Covector, speeds: core.Scalar) -> core.Scalar:
    """Phase speeds along `direction` located from local minima of the wave residual."""
    wavevectors = wavenumber * (speeds * t_s - direction)
    return speeds[1:-1][core.local_minima(core.wave_residuals(wavevectors, medium_map), 2e-3 * wavenumber)]


def test_unit_medium_matches_vacuum() -> None:
    fields = mv(Field, np.random.default_rng(5).normal(size=(5, 6))) * core.field_dim
    np.testing.assert_allclose(((isotropic_medium(1.0, 1.0)(fields) - core.vacuum(fields)) * mu_0).kernel, 0.0, atol=1e-12)


def test_isotropic_medium_speed() -> None:
    for permittivity, permeability in ((2.25, 1.0), (4.0, 1.5)):
        medium_map = isotropic_medium(permittivity, permeability)
        found = phase_speeds(medium_map, z, trial_speeds)
        expected = c / np.sqrt(permittivity * permeability)
        assert ((found - expected).abs() < scan_tolerance).all()


def test_constitutive_and_stress_energy_forms() -> None:
    permittivity, permeability = 2.25, 1.5
    medium_map = isotropic_medium(permittivity, permeability)
    fields = mv(Field, np.random.default_rng(318).normal(size=(7, 6))) * core.field_dim
    electric_fields, magnetic_fields = electric_part(fields), magnetic_part(fields)
    electric_squared = electric_fields.scalar_product(electric_fields)
    magnetic_squared = magnetic_fields.scalar_product(magnetic_fields)

    bilinear_form = Field & medium_map
    energy_form = (t * Field * t.inverse()) & medium_map
    stress_form = core.Covector & core.stress_energy(fields, medium_map(fields))
    expected_action = (permittivity * electric_squared + magnetic_squared / permeability) / (2 * mu_0)
    expected_energy = (permittivity * electric_squared - magnetic_squared / permeability) / (2 * mu_0)

    energy_unit = mv.scalar(1.0, unit=core.energy_density_dim)
    assert (((bilinear_form(fields, fields) / 2 + expected_action) / energy_unit).abs() < 1e-11).all()
    assert (((energy_form(fields, fields) / 2 - expected_energy) / energy_unit).abs() < 1e-11).all()
    assert (((stress_form(t, time) - expected_energy) / energy_unit).abs() < 1e-11).all()

    flux_field = (mv.tx + mv.zx) * core.field_dim
    flux_form = core.Covector & core.stress_energy(flux_field, medium_map(flux_field))
    assert (((flux_form(z, time) + energy_unit / permeability) / energy_unit).abs() < 1e-11).all()
    assert (((flux_form(t, z.dual()) - permittivity * energy_unit) / energy_unit).abs() < 1e-11).all()

    vacuum_current = core.stress_energy(fields, core.vacuum(fields))
    for direction in (t, x, z):
        diff = (vacuum_current(direction.dual()) - (fields >> direction).dual() / (2 * mu_0)) / energy_unit
        np.testing.assert_allclose(diff.kernel, 0.0, atol=1e-11)


def test_crystal_birefringence_and_ferrite_permeability() -> None:
    _, crystal, ferrite = scenarios.media
    assert ((phase_speeds(crystal, z, trial_speeds) - np.array([1.0 / np.sqrt(2.25), 1.0 / np.sqrt(1.5)]) * c).abs() < scan_tolerance).all()
    assert ((phase_speeds(crystal, x, trial_speeds) - c / np.sqrt(1.5)).abs() < scan_tolerance).all()
    assert ((phase_speeds(ferrite, z, trial_speeds) - np.array([1.0 / np.sqrt(4.5), 1.0 / np.sqrt(2.25)]) * c).abs() < scan_tolerance).all()


def test_crystal_and_vacuum_eigenmodes_satisfy_maxwell() -> None:
    _, (speeds, fields) = scenarios.propagation_modes()
    _, crystal, _ = scenarios.media
    wavevectors = wavenumber * (speeds * t_s - z)
    along_z = z.dual() & time
    across_z = mv.xy.dual()

    np.testing.assert_allclose((wavevectors ^ fields).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((mu_0 * (wavevectors ^ crystal(fields))).kernel, 0.0, atol=1e-11)
    assert ((fields & along_z).abs() < 1e-11).all()
    assert ((fields & across_z).abs() < 1e-11).all()

    wavevector = wavenumber * (c * t_s - z)
    _, singular_values, vacuum_fields = core.wave_map(wavevector, core.vacuum).svd()
    modes = vacuum_fields[-2:] * core.field_dim
    action_form = Field & core.vacuum
    stress_form = core.Covector & core.stress_energy(modes, core.vacuum(modes))

    assert int((singular_values < 1e-11 * wavenumber).sum()) == 2
    np.testing.assert_allclose(((wavevector ^ modes) / (wavenumber * core.field_dim)).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((mu_0 * (wavevector ^ core.vacuum(modes)) / (wavenumber * core.field_dim)).kernel, 0.0, atol=1e-11)
    assert ((action_form(modes, modes) * mu_0 / core.field_dim**2).abs() < 1e-11).all()
    assert (stress_form(t, time) > 0).all()


def test_axion_coupling_leaves_bulk_wave_speeds_and_stress_unchanged() -> None:
    glass, _, _ = scenarios.media
    field = (mv.tx * 2.0 + mv.yz * 3.0) * core.field_dim
    base_form = Field & glass
    base_current = core.stress_energy(field, glass(field))
    for alpha in (0.4, -1.3):
        coupling = mv(core.AxionCoupling, [alpha]) / mu_0
        axion = glass + coupling * Field
        axion_form = Field & axion
        action_change = (axion_form(field, field) - base_form(field, field)) / 2
        assert ((phase_speeds(axion, z, trial_speeds) - phase_speeds(glass, z, trial_speeds)).abs() < scan_tolerance).all()
        np.testing.assert_allclose(((action_change - coupling * (field & field) / 2) / core.energy_density_dim).kernel, 0.0, atol=1e-11)
        assert (((core.stress_energy(field, axion(field)) - base_current) / core.energy_density_dim).svdvals().max() < 1e-11).all()


def test_lorentz_covariance_and_fresnel_drag() -> None:
    permittivity, permeability, fraction = 2.25, 1.5, 0.35
    medium_map = isotropic_medium(permittivity, permeability)
    boost = ((z ^ t) * (np.arctanh(fraction) / 2)).exp()
    observer = boost >> t
    moving = core.moving_medium(medium_map, fraction * c)
    electric = (Field - observer * Field * observer.inverse()) / 2
    rebuilt = (c**2 * (permittivity * eps_0) * electric + (Field - electric) / (permeability * mu_0)).dual()

    fields = mv(Field, np.random.default_rng(627).normal(size=(7, 6))) * core.field_dim
    moved_fields = boost >> fields
    current = core.stress_energy(fields, medium_map(fields))
    moved_current = core.stress_energy(moved_fields, moving(moved_fields))
    form, moved_form = Field & medium_map, Field & moving
    stress_form = core.Covector & current
    moved_stress_form = core.Covector & moved_current

    np.testing.assert_allclose(((moving(fields) - rebuilt(fields)) * mu_0 / core.field_dim).kernel, 0.0, atol=1e-11)
    assert (((moved_form(moved_fields, moved_fields) - form(fields, fields)) / core.energy_density_dim).abs() < 1e-11).all()
    assert (((moved_current - (boost >> current(boost << core.Direction))) / core.energy_density_dim).svdvals().max(axis=-1) < 1e-11).all()
    assert (((moved_stress_form(observer, -observer.dual()) - stress_form(t, time)) / core.energy_density_dim).abs() < 1e-11).all()

    glass, _, _ = scenarios.media
    velocities = np.array([0.0, 0.3, -0.5]) * c
    for sample_velocity, expected in zip(velocities, scenarios.relativistic_sum(velocities)[0]):
        moving_glass = core.moving_medium(glass, sample_velocity)
        assert ((phase_speeds(moving_glass, z, trial_speeds) - expected).abs() < scan_tolerance).all()


def test_render_figures_and_single_animation_frame() -> None:
    glass_modes, crystal_modes = scenarios.propagation_modes()
    distances = np.linspace(0.0, 4.0 * np.pi, render.n_samples) * mv.scalar(1.0, unit=dim.length)
    pairs = zip(
        core.travelling_fields(*glass_modes, distances, 1),
        core.travelling_fields(*crystal_modes, distances, 1),
    )
    frame = next(render.wave_comparison_frames(glass_modes, crystal_modes, distances, pairs))
    assert frame.ndim == 3 and frame.shape[2] == 3

    speeds = np.linspace(0.4, 1.0, 41) * c
    residual = core.wave_residuals((wavenumber * (speeds * t_s - z))[:, None], scenarios.media)
    fig_disp = render.dispersion_figure(speeds, residual, scenarios.expected_speeds)
    fig_pol = render.polarizations_figure(crystal_modes)
    plt.close(fig_disp)
    plt.close(fig_pol)
