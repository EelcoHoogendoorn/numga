"""Unit tests for electromagnetic constitutive maps, stress-energy currents, birefringence, and Fresnel drag."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.electromagnetism.constitutive import core, render, scenarios

mv = core.mv
t, x, z = mv.t, mv.x, mv.z
Field = core.Field

trial_speeds = np.linspace(0.05, 1.5, 6001)
scan_tolerance = trial_speeds[1] - trial_speeds[0]
# The observer's direction in time pairs to one with t.
time = -t.dual()
# Reflection across t keeps magnetic planes and reverses electric ones.
electric_part = (Field - t * Field * t.inverse()) / 2
magnetic_part = Field - electric_part


def euclidean_norm(element):
    """Positive-definite Euclidean norm (`Scalar+`) induced by the rest-frame observer `t`."""
    return element.reverse().scalar_product(t * element.involute() * t.inverse()).square_root()


def isotropic_medium(permittivity: float, permeability: float) -> core.Constitutive:
    """Isotropic medium with uniform `permittivity` and `permeability` across all three spatial axes."""
    return core.medium(np.full(3, permittivity), np.full(3, 1.0 / permeability))


def phase_speeds(medium_map: core.Constitutive, direction: core.Covector, speeds: np.ndarray) -> np.ndarray:
    """Phase speeds along `direction` located from local minima of the wave residual."""
    return speeds[1:-1][core.local_minima(core.wave_residuals(speeds * t - direction, medium_map), 2e-3)]


def test_unit_medium_matches_vacuum() -> None:
    fields = mv.bivector(np.random.default_rng(5).normal(size=(5, 6)))
    assert (euclidean_norm(isotropic_medium(1.0, 1.0)(fields) - core.vacuum(fields)) < 1e-12).all()


def test_isotropic_medium_speed() -> None:
    for permittivity, permeability in ((2.25, 1.0), (4.0, 1.5)):
        medium_map = isotropic_medium(permittivity, permeability)
        assert np.allclose(phase_speeds(medium_map, z, trial_speeds), [1.0 / np.sqrt(permittivity * permeability)], atol=scan_tolerance)


def test_constitutive_and_stress_energy_forms() -> None:
    permittivity, permeability = 2.25, 1.5
    medium_map = isotropic_medium(permittivity, permeability)
    fields = mv.bivector(np.random.default_rng(318).normal(size=(7, 6)))
    electric_fields, magnetic_fields = electric_part(fields), magnetic_part(fields)
    electric_squared = electric_fields.scalar_product(electric_fields)
    magnetic_squared = magnetic_fields.scalar_product(magnetic_fields)

    bilinear_form = Field & medium_map
    energy_form = (t * Field * t.inverse()) & medium_map
    stress_form = core.Covector & core.stress_energy(fields, medium_map(fields))
    expected_action = (permittivity * electric_squared + magnetic_squared / permeability) / 2
    expected_energy = (permittivity * electric_squared - magnetic_squared / permeability) / 2

    assert ((bilinear_form(fields, fields) / 2 + expected_action).abs() < 1e-11).all()
    assert ((energy_form(fields, fields) / 2 - expected_energy).abs() < 1e-11).all()
    assert ((stress_form(t, time) - expected_energy).abs() < 1e-11).all()

    flux_field = mv.tx + mv.zx
    flux_form = core.Covector & core.stress_energy(flux_field, medium_map(flux_field))
    assert ((flux_form(z, time) + 1.0 / permeability).abs() < 1e-11).all()
    assert ((flux_form(t, z.dual()) - permittivity).abs() < 1e-11).all()

    vacuum_current = core.stress_energy(fields, core.vacuum(fields))
    for direction in (t, x, z):
        assert (euclidean_norm(vacuum_current(direction.dual()) - (fields >> direction).dual() / 2) < 1e-11).all()


def test_crystal_birefringence_and_ferrite_permeability() -> None:
    _, crystal, ferrite = scenarios.media
    assert np.allclose(phase_speeds(crystal, z, trial_speeds), [1.0 / np.sqrt(2.25), 1.0 / np.sqrt(1.5)], atol=scan_tolerance)
    assert np.allclose(phase_speeds(crystal, x, trial_speeds), [1.0 / np.sqrt(1.5)], atol=scan_tolerance)
    assert np.allclose(phase_speeds(ferrite, z, trial_speeds), [1.0 / np.sqrt(4.5), 1.0 / np.sqrt(2.25)], atol=scan_tolerance)


def test_crystal_and_vacuum_eigenmodes_satisfy_maxwell() -> None:
    _, (speeds, fields) = scenarios.propagation_modes()
    _, crystal, _ = scenarios.media
    wavevectors = speeds * t - z
    along_z = z.dual() & time
    across_z = mv.xy.dual()

    assert (euclidean_norm(wavevectors ^ fields) < 1e-11).all()
    assert (euclidean_norm(wavevectors ^ crystal(fields)) < 1e-11).all()
    assert ((fields & along_z).abs() < 1e-11).all()
    assert ((fields & across_z).abs() < 1e-11).all()

    wavevector = t - z
    _, singular_values, vacuum_fields = core.wave_map(wavevector, core.vacuum).svd()
    modes = vacuum_fields[-2:]
    action_form = Field & core.vacuum
    stress_form = core.Covector & core.stress_energy(modes, core.vacuum(modes))

    assert int((singular_values < 1e-11).sum()) == 2
    assert (euclidean_norm(wavevector ^ modes) < 1e-11).all()
    assert (euclidean_norm(wavevector ^ core.vacuum(modes)) < 1e-11).all()
    assert (action_form(modes, modes).abs() < 1e-11).all()
    assert (stress_form(t, time) > 0.0).all()


def test_axion_coupling_leaves_bulk_wave_speeds_and_stress_unchanged() -> None:
    glass, _, _ = scenarios.media
    field = mv.tx * 2.0 + mv.yz * 3.0
    base_form = Field & glass
    base_current = core.stress_energy(field, glass(field))
    for alpha in (0.4, -1.3):
        coupling = mv(core.AxionCoupling, [alpha])
        axion = glass + coupling * Field
        axion_form = Field & axion
        action_change = (axion_form(field, field) - base_form(field, field)) / 2
        assert np.allclose(phase_speeds(axion, z, trial_speeds), phase_speeds(glass, z, trial_speeds), atol=scan_tolerance)
        assert (euclidean_norm(action_change - coupling * (field & field) / 2) < 1e-11).all()
        assert ((core.stress_energy(field, axion(field)) - base_current).svdvals().max() < 1e-11).all()


def test_lorentz_covariance_and_fresnel_drag() -> None:
    permittivity, permeability, fraction = 2.25, 1.5, 0.35
    medium_map = isotropic_medium(permittivity, permeability)
    boost = ((z ^ t) * (np.arctanh(fraction) / 2)).exp()
    observer = boost >> t
    moving = core.moving_medium(medium_map, np.array(fraction))
    electric = (Field - observer * Field * observer.inverse()) / 2
    rebuilt = (permittivity * electric + (Field - electric) / permeability).dual()

    fields = mv.bivector(np.random.default_rng(627).normal(size=(7, 6)))
    moved_fields = boost >> fields
    current = core.stress_energy(fields, medium_map(fields))
    moved_current = core.stress_energy(moved_fields, moving(moved_fields))
    form, moved_form = Field & medium_map, Field & moving
    stress_form = core.Covector & current
    moved_stress_form = core.Covector & moved_current

    assert (euclidean_norm(moving(fields) - rebuilt(fields)) < 1e-7).all()
    assert ((moved_form(moved_fields, moved_fields) - form(fields, fields)).abs() < 1e-7).all()
    assert ((moved_current - (boost >> current(boost << core.Direction))).svdvals().max(axis=-1) < 1e-7).all()
    assert ((moved_stress_form(observer, -observer.dual()) - stress_form(t, time)).abs() < 1e-7).all()

    glass, _, _ = scenarios.media
    fractions = np.array([0.0, 0.3, -0.5])
    for sample_fraction, (expected, _) in zip(fractions, scenarios.relativistic_sum(fractions).T):
        moving_glass = core.moving_medium(glass, np.array(sample_fraction))
        assert np.allclose(phase_speeds(moving_glass, z, trial_speeds), [expected], atol=scan_tolerance)


def test_render_figures_and_single_animation_frame() -> None:
    glass_modes, crystal_modes = scenarios.propagation_modes()
    distances = np.linspace(0.0, 4.0 * np.pi, render.n_samples)
    pairs = zip(
        core.travelling_fields(*glass_modes, distances, 1),
        core.travelling_fields(*crystal_modes, distances, 1),
    )
    frame = next(render.wave_comparison_frames(glass_modes, crystal_modes, distances, pairs))
    assert frame.ndim == 3 and frame.shape[2] == 3

    speeds = np.linspace(0.4, 1.0, 41)
    residual = core.wave_residuals((speeds * t - z)[:, None], scenarios.media)
    fig_disp = render.dispersion_figure(speeds, residual, scenarios.expected_speeds)
    fig_pol = render.polarizations_figure(crystal_modes)
    plt.close(fig_disp)
    plt.close(fig_pol)
