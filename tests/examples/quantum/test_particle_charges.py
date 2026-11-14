"""The chiral Spin(10) spinor resolves into Standard Model multiplets and their gauge actions."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.quantum.particle_charges import core, render, scenarios


def test_spinor_sectors_and_standard_model_charges():
    (
        sector_patterns, projectors, charges, hypercharges,
        left_isospins, right_isospins, baryon_minus_leptons, _,
    ) = scenarios.charge_sectors()
    pseudoscalar_action = core.action(core.pseudoscalar)
    dimensions = projectors.trace()
    intersections = projectors[:, None](projectors[None, :])
    expected_charges = np.array([
        0, 1, -1 / 3, 2 / 3, -1 / 3, 2 / 3, -2 / 3, 1 / 3,
        -1 / 3, 2 / 3, -2 / 3, 1 / 3, -2 / 3, 1 / 3, -1, 0,
    ])
    tolerance = 1e-10

    # checks
    assert ((pseudoscalar_action(pseudoscalar_action) + core.State).scalar_norm_squared().contract() < tolerance).all()
    assert ((projectors.sum(axis=0) - core.State).scalar_norm_squared().contract() < tolerance).all()
    assert ((dimensions - 2).abs() < tolerance).all()
    assert ((intersections - projectors[:, None] * np.eye(len(projectors))).scalar_norm_squared().contract() < tolerance).all()
    assert ((pseudoscalar_action(projectors) - projectors(pseudoscalar_action)).scalar_norm_squared().contract() < tolerance).all()
    assert ((charges - expected_charges).abs() < tolerance).all()
    assert ((charges - hypercharges - left_isospins).abs() < tolerance).all()
    assert ((hypercharges - right_isospins - baryon_minus_leptons / 2).abs() < tolerance).all()
    assert (sector_patterns.sum(axis=-1) % 2 == 0).all()
    assert len(core.State.output_subspace) == 32


def test_gauge_actions_and_connections_figure():
    (
        sector_patterns, projectors, charges, hypercharges,
        left_isospins, right_isospins, _, colour_weights,
    ) = scenarios.charge_sectors()
    (
        colour_generators, weak_generators, mixing_generators,
        colour_strength, weak_strength, mixing_strength,
    ) = scenarios.gauge_interactions(projectors)
    hypercharge_map = (hypercharges * projectors).sum(axis=0)
    charge_map = (charges * projectors).sum(axis=0)
    colour_casimir = -colour_generators(colour_generators).sum(axis=0)
    weak_casimir = -weak_generators(weak_generators).sum(axis=0)
    coloured_mask = np.isin(sector_patterns[:, :3].sum(axis=-1), [1, 2])
    doublet_mask = sector_patterns[:, 3] != sector_patterns[:, 4]
    pairing = core.State.reverse().scalar_product(core.State)
    tolerance = 1e-10

    figure = render.connections_figure(
        sector_patterns, left_isospins, right_isospins, colour_weights,
        stack((colour_strength, weak_strength, mixing_strength)),
    )
    plt.close(figure)

    # checks
    assert len(figure.axes) == 3
    assert ((colour_generators[:, None](weak_generators[None, :]) - weak_generators[None, :](colour_generators[:, None])).scalar_norm_squared().contract() < tolerance).all()
    assert ((colour_generators(charge_map) - charge_map(colour_generators)).scalar_norm_squared().contract() < tolerance).all()
    assert ((weak_generators(hypercharge_map) - hypercharge_map(weak_generators)).scalar_norm_squared().contract() < tolerance).all()
    assert ((colour_casimir(projectors) - (4 / 3) * coloured_mask * projectors).scalar_norm_squared().contract() < tolerance).all()
    assert ((weak_casimir(projectors) - (3 / 4) * doublet_mask * projectors).scalar_norm_squared().contract() < tolerance).all()
    assert ((pairing(colour_generators, core.State) + pairing(core.State, colour_generators)).scalar_norm_squared().contract(1, 2).contract() < tolerance).all()
    assert ((pairing(weak_generators, core.State) + pairing(core.State, weak_generators)).scalar_norm_squared().contract(1, 2).contract() < tolerance).all()
    assert ((weak_generators[0](weak_generators[1]) - weak_generators[1](weak_generators[0]) - weak_generators[2]).scalar_norm_squared().contract() < tolerance).all()
    assert ((core.action(core.pseudoscalar)(weak_generators[2])(projectors) - left_isospins * projectors).scalar_norm_squared().contract() < tolerance).all()
    assert (colour_generators[:, None](projectors[None, ~coloured_mask]).scalar_norm_squared().contract() < tolerance).all()
    assert (weak_generators[:, None](projectors[None, ~doublet_mask]).scalar_norm_squared().contract() < tolerance).all()
    assert int(np.triu(colour_strength > tolerance, k=1).sum()) == 12
    assert int(np.triu(weak_strength > tolerance, k=1).sum()) == 4
    assert int(np.triu(mixing_strength > tolerance, k=1).sum()) == 4
    mixing_edges = mixing_strength > tolerance
    charge_steps = (charges[:, None] - charges[None, :]).abs()
    assert (((charge_steps - 4 / 3).abs() < tolerance) | ~mixing_edges).all()
    assert int((mixing_edges & (coloured_mask[:, None] != coloured_mask[None, :])).sum()) == 4
