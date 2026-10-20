"""Unit tests for the constitutive extensor: media, energy, birefringence, Fresnel drag."""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

from numga import stack

from examples.electromagnetism.constitutive import core, scenarios, render

mv = core.mv
t = core.t
x = core.x
z = core.z
Bivector = core.Bivector


def isotropic(permittivity, permeability):
    return core.medium(np.full(3, permittivity), np.full(3, 1 / permeability))


def phase_speeds(medium, direction, speeds):
    return speeds[1:-1][core.dips(speeds, core.residuals(speeds * t + direction, medium), 2e-3)]


SPEEDS = np.linspace(0.05, 1.5, 6001)
SCAN_TOLERANCE = SPEEDS[1] - SPEEDS[0]
# The observer's reflection keeps magnetic planes and reverses electric ones.
ELECTRIC = (Bivector - (t >> Bivector)) / 2
MAGNETIC = Bivector - ELECTRIC


def test_unit_medium_is_the_vacuum():
    fields = mv.bivector(np.random.default_rng(5).normal(size=(7, 6)))
    np.testing.assert_allclose((isotropic(1.0, 1.0)(fields) - fields.dual()).kernel, 0.0, atol=1e-14)


def test_isotropic_medium_speed_is_one_over_n():
    for permittivity, permeability in ((2.25, 1.0), (4.0, 1.5)):
        medium = isotropic(permittivity, permeability)
        np.testing.assert_allclose(phase_speeds(medium, z, SPEEDS), [1.0 / np.sqrt(permittivity * permeability)],
                                   atol=SCAN_TOLERANCE)


def test_material_and_stress_forms_give_the_electric_and_magnetic_energy():
    permittivity, permeability = 2.25, 1.5
    medium = isotropic(permittivity, permeability)
    rng = np.random.default_rng(318)
    fields = mv.bivector(rng.normal(size=(17, 6)))
    electric_fields, magnetic_fields = ELECTRIC(fields), MAGNETIC(fields)
    electric_squared = electric_fields.scalar_product(electric_fields)
    magnetic_squared = magnetic_fields.scalar_product(magnetic_fields)
    bilinear_form = Bivector & medium
    energy_form = -(t >> Bivector) & medium
    stress_form = core.Vector & core.stress_energy(fields, medium(fields))
    expected_action = (permittivity * electric_squared + magnetic_squared / permeability) / 2
    expected_energy = (permittivity * electric_squared - magnetic_squared / permeability) / 2

    np.testing.assert_allclose((bilinear_form(fields, fields) / 2 - expected_action).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((energy_form(fields, fields) / 2 - expected_energy).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((stress_form(t, t.dual()) - expected_energy).kernel, 0.0, atol=1e-12)

    flux_field = mv.tx + mv.zx
    flux_form = core.Vector & core.stress_energy(flux_field, medium(flux_field))
    np.testing.assert_allclose(flux_form(z, t.dual()).kernel, -1 / permeability, atol=1e-12)
    np.testing.assert_allclose(flux_form(t, z.dual()).kernel, -permittivity, atol=1e-12)

    # In vacuum the current along a direction is the field's sandwich of the direction, halved.
    vacuum = isotropic(1.0, 1.0)
    vacuum_current = core.stress_energy(fields, vacuum(fields))
    for direction in (t, x, z):
        np.testing.assert_allclose((vacuum_current(direction.dual()) + (fields >> direction).dual() / 2).kernel, 0.0, atol=1e-12)


def test_crystal_is_birefringent_along_z_and_not_along_x():
    _, crystal, _ = scenarios.media()
    np.testing.assert_allclose(phase_speeds(crystal, z, SPEEDS), [1.0 / np.sqrt(2.25), 1.0 / np.sqrt(1.5)], atol=SCAN_TOLERANCE)
    np.testing.assert_allclose(phase_speeds(crystal, x, SPEEDS), [1.0 / np.sqrt(1.5)], atol=SCAN_TOLERANCE)


def test_crystal_fields_satisfy_maxwell_and_lie_in_their_material_planes():
    _, (speeds, fields) = scenarios.modes()
    _, crystal, _ = scenarios.media()
    wavevectors = speeds * t + z
    electric_planes = stack((mv.tx, mv.ty))
    magnetic_planes = stack((mv.xz, mv.yz))

    np.testing.assert_allclose((wavevectors ^ fields).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((wavevectors ^ crystal(fields)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((ELECTRIC(fields) - electric_planes * (electric_planes | fields)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((MAGNETIC(fields) + magnetic_planes * (magnetic_planes | fields)).kernel, 0.0, atol=1e-12)


def test_vacuum_wave_has_two_transverse_field_modes():
    vacuum = isotropic(1.0, 1.0)
    wavevector = t + z
    _, singular_values, fields = core.wave_map(wavevector, vacuum).svd()
    modes = fields[-2:]
    action_form = Bivector & vacuum
    stress_form = core.Vector & core.stress_energy(modes, vacuum(modes))

    assert np.count_nonzero(singular_values.kernel < 1e-12) == 2
    np.testing.assert_allclose((wavevector ^ modes).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((wavevector ^ vacuum(modes)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(action_form(modes, modes).kernel / 2, 0.0, atol=1e-12)
    assert np.all(stress_form(t, t.dual()).kernel > 0.0)


def test_ferrite_lifts_permeability_through_the_dual_field():
    _, _, ferrite = scenarios.media()
    np.testing.assert_allclose(phase_speeds(ferrite, z, SPEEDS), [1.0 / np.sqrt(4.5), 1.0 / np.sqrt(2.25)], atol=SCAN_TOLERANCE)


def test_axion_term_is_invisible_to_bulk_waves():
    glass, _, _ = scenarios.media()
    field = mv.tx * 2.0 + mv.yz * 3.0
    base_form = Bivector & glass
    base_current = core.stress_energy(field, glass(field))
    for alpha in (0.4, -1.3):
        coupling = mv(core.AxionCoupling, [alpha])
        axion = glass + coupling * Bivector
        axion_form = Bivector & axion
        action_change = (axion_form(field, field) - base_form(field, field)) / 2
        np.testing.assert_allclose(phase_speeds(axion, z, SPEEDS), phase_speeds(glass, z, SPEEDS), atol=SCAN_TOLERANCE)
        np.testing.assert_allclose((action_change - coupling * (field & field) / 2).kernel, 0.0, atol=1e-12)
        np.testing.assert_allclose((core.stress_energy(field, axion(field)) - base_current).kernel, 0.0, atol=1e-12)


def test_material_and_stress_forms_transform_with_the_field_and_observer():
    permittivity, permeability, fraction = 2.25, 1.5, 0.35
    medium = isotropic(permittivity, permeability)
    rotation = ((z ^ t) * (np.arctanh(fraction) / 2)).exp()
    observer = rotation >> t
    moving = scenarios.moving(medium, np.array(fraction))
    # The same medium built from the moving observer's electric and magnetic planes.
    electric = (Bivector - (observer >> Bivector)) / 2
    rebuilt = (permittivity * electric + (Bivector - electric) / permeability).dual()
    rng = np.random.default_rng(627)
    fields = mv.bivector(rng.normal(size=(13, 6)))
    moved_fields = rotation >> fields
    current = core.stress_energy(fields, medium(fields))
    moved_current = core.stress_energy(moved_fields, moving(moved_fields))
    form, moved_form = Bivector & medium, Bivector & moving
    stress_form = core.Vector & current
    moved_stress_form = core.Vector & moved_current

    np.testing.assert_allclose((moving(fields) - rebuilt(fields)).kernel, 0.0, atol=1e-7)
    np.testing.assert_allclose((moved_form(moved_fields, moved_fields) - form(fields, fields)).kernel, 0.0, atol=1e-7)
    np.testing.assert_allclose((moved_current - (rotation >> current(rotation << core.Antivector))).kernel, 0.0, atol=1e-7)
    np.testing.assert_allclose((moved_stress_form(observer, observer.dual()) - stress_form(t, t.dual())).kernel, 0.0, atol=1e-7)


def test_moving_glass_shows_exact_fresnel_drag():
    glass, _, _ = scenarios.media()
    fractions = np.array([0.0, 0.3, -0.5])
    for fraction, (expected, _) in zip(fractions, scenarios.relativistic_sum(fractions).T):
        moving = scenarios.moving(glass, np.array(fraction))
        np.testing.assert_allclose(phase_speeds(moving, z, SPEEDS), [expected], atol=SCAN_TOLERANCE)


def test_figures_and_animation_draw():
    """Each scenario feeds its figure; the figures and a short animation draw."""
    glass_modes, crystal_modes = scenarios.modes()
    figures = [
        render.draw_wave_comparison_figure(glass_modes, crystal_modes),
        render.draw_dispersion_figure(*scenarios.dispersion()),
        render.draw_polarizations_figure(crystal_modes),
        render.draw_fresnel_surface_figure(*scenarios.headings(), scenarios.threshold),
        render.draw_fresnel_drag_figure(*scenarios.drag()),
    ]
    assert all(isinstance(figure, plt.Figure) for figure in figures)
    frames = render.animate_wave_comparison(glass_modes, crystal_modes, 4)
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
