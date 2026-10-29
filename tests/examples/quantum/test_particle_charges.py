"""The chiral spinor resolves into the Standard Model multiplets and their gauge actions."""

import numpy as np

from examples.quantum.particle_charges import core, scenarios


def test_spinor_sectors_and_standard_model_charges():
    occupations, projectors, charge, hypercharge, left_isospin, right_isospin, baryon_minus_lepton, _ = scenarios.sectors()
    pseudoscalar_action = core.action(core.pseudoscalar)
    dimension = projectors.trace()
    intersections = projectors[:, None](projectors[None, :])
    tolerance = 1e-12
    expected_charges = np.array([0, 1, -1 / 3, 2 / 3, -1 / 3, 2 / 3, -2 / 3, 1 / 3,
                                 -1 / 3, 2 / 3, -2 / 3, 1 / 3, -2 / 3, 1 / 3, -1, 0])

    # checks
    np.testing.assert_allclose((pseudoscalar_action(pseudoscalar_action) + core.State).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((projectors.sum(axis=0) - core.State).kernel, 0, atol=tolerance)
    np.testing.assert_allclose(dimension.to_array(), 2, atol=tolerance)
    np.testing.assert_allclose((intersections - projectors[:, None] * np.eye(len(projectors))).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((pseudoscalar_action(projectors) - projectors(pseudoscalar_action)).kernel, 0, atol=tolerance)
    np.testing.assert_allclose(charge.to_array(), expected_charges, atol=tolerance)
    np.testing.assert_allclose((charge - hypercharge - left_isospin).to_array(), 0, atol=tolerance)
    np.testing.assert_allclose((hypercharge - right_isospin - baryon_minus_lepton / 2).to_array(), 0, atol=tolerance)
    np.testing.assert_array_equal(occupations.sum(axis=-1) % 2, 0)
    assert len(core.State.output_subspace) == 32


def test_gauge_actions_and_additional_unified_connections():
    occupations, projectors, charge, hypercharge, left_isospin, _, _, _ = scenarios.sectors()
    colour, weak, mixing, colour_strength, weak_strength, mixing_strength = scenarios.interactions(projectors)
    hypercharge_map = (hypercharge * projectors).sum(axis=0)
    charge_map = (charge * projectors).sum(axis=0)
    colour_casimir = -colour(colour).sum(axis=0)
    weak_casimir = -weak(weak).sum(axis=0)
    is_coloured = np.isin(occupations[:, :3].sum(axis=-1), [1, 2])
    is_doublet = occupations[:, 3] != occupations[:, 4]
    pairing = core.State.reverse().scalar_product(core.State)
    tolerance = 1e-12

    # checks
    np.testing.assert_allclose((colour[:, None](weak[None, :]) - weak[None, :](colour[:, None])).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((colour(charge_map) - charge_map(colour)).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((weak(hypercharge_map) - hypercharge_map(weak)).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((colour_casimir(projectors) - (4 / 3) * is_coloured * projectors).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((weak_casimir(projectors) - (3 / 4) * is_doublet * projectors).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((pairing(colour, core.State) + pairing(core.State, colour)).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((pairing(weak, core.State) + pairing(core.State, weak)).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((weak[0](weak[1]) - weak[1](weak[0]) - weak[2]).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((core.action(core.pseudoscalar)(weak[2])(projectors) - left_isospin * projectors).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((colour[:, None](projectors[None, ~is_coloured])).kernel, 0, atol=tolerance)
    np.testing.assert_allclose((weak[:, None](projectors[None, ~is_doublet])).kernel, 0, atol=tolerance)
    assert np.count_nonzero(np.triu(colour_strength.to_array(), 1) > tolerance) == 12
    assert np.count_nonzero(np.triu(weak_strength.to_array(), 1) > tolerance) == 4
    assert np.count_nonzero(np.triu(mixing_strength.to_array(), 1) > tolerance) == 4
    mixing_edges = mixing_strength.to_array() > tolerance
    charge_changes = (charge[:, None] - charge[None, :]).to_array()
    np.testing.assert_allclose(np.abs(charge_changes[mixing_edges]), 4 / 3, atol=tolerance)
    assert np.count_nonzero(mixing_edges & (is_coloured[:, None] != is_coloured[None, :])) == 4
