"""Products in the split algebra represent every linear map on exterior states."""

from functools import reduce
from operator import mul

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import Algebra, NumpyContext, stack
from examples.math.doubled_algebra import core, render, scenarios


def test_null_generators_insert_remove_and_recover_the_geometric_product():
    insertions = core.representations(core.creators)
    removals = core.representations(core.removers)

    assert (core.Full * (core.projector.squared() - core.projector)).svdvals().max() < 1e-12
    assert ((core.Full * (core.removers * core.projector)).svdvals().max(axis=-1) < 1e-12).all()
    assert (core.state_readout(core.embedder) - core.State).svdvals().max() < 1e-12
    assert ((insertions - (core.basis ^ core.State)).svdvals().max(axis=-1) < 1e-12).all()
    assert ((removals - core.basis.left_contraction(core.State)).svdvals().max(axis=-1) < 1e-12).all()
    assert ((core.representations(core.basis) - core.basis * core.State).svdvals().max(axis=-1) < 1e-12).all()

    insert_twice = insertions[:, None](insertions[None, :]) + insertions[None, :](insertions[:, None])
    remove_twice = removals[:, None](removals[None, :]) + removals[None, :](removals[:, None])
    mixed_pair = removals[:, None](insertions[None, :]) + insertions[None, :](removals[:, None])
    metric_pair = core.basis[:, None].scalar_product(core.basis[None, :]) * core.State
    assert (insert_twice.svdvals().max(axis=-1) < 1e-12).all()
    assert (remove_twice.svdvals().max(axis=-1) < 1e-12).all()
    assert ((mixed_pair - metric_pair).svdvals().max(axis=-1) < 1e-12).all()


def test_dyads_extend_to_blades_by_the_product_rule():
    dyads = core.representations(core.creators[:, None] * core.removers[None, :])
    wedge_contraction = core.basis[:, None] ^ core.basis[None, :].left_contraction(core.State)
    assert ((dyads - wedge_contraction).svdvals().max(axis=-1) < 1e-12).all()

    blade_action = dyads(core.Vector ^ core.Vector)
    product_rule = (dyads(core.Vector) ^ core.Vector) + (core.Vector ^ dyads(core.Vector))
    diff = blade_action(core.basis[:, None], core.basis[None, :]) - product_rule(core.basis[:, None], core.basis[None, :])
    assert (diff.scalar_norm_squared() < 1e-12).all()

    count_map = core.representations(core.grade_count)
    state = 3 + core.mv.x - 2 * core.mv.y + 4 * core.mv.xy
    assert (count_map(state) - (core.mv.x - 2 * core.mv.y + 8 * core.mv.xy)).scalar_norm_squared() < 1e-12


def test_exponentiating_a_shear_moves_vectors_and_their_wedge_together():
    shear_rates = np.array([-1.25, 0.0, 0.75])
    motion = core.representations((core.shear_generator * shear_rates).exp())
    shear_formula = core.Vector + shear_rates * core.mv.x * core.mv.y.scalar_product(core.Vector)
    assert ((motion(core.Vector) - shear_formula).svdvals().max(axis=-1) < 1e-12).all()

    area_diff = motion(core.mv.x ^ core.mv.y) - (motion(core.mv.x) ^ motion(core.mv.y))
    assert (area_diff.scalar_norm_squared() < 1e-12).all()
    assert ((motion(core.mv.xy) - core.mv.xy).scalar_norm_squared() < 1e-12).all()
    assert ((motion(core.mv.scalar()) - core.mv.scalar()).scalar_norm_squared() < 1e-12).all()


def test_mixed_grade_exchange_and_product_are_map_composition():
    representation = core.representations(core.exchange)
    state = 2 + 3 * core.mv.x - core.mv.y + 5 * core.mv.xy
    assert (representation(state) - (5 + 2 * core.mv.xy)).scalar_norm_squared() < 1e-12
    assert (representation(representation(state)) - (2 + 5 * core.mv.xy)).scalar_norm_squared() < 1e-12
    assert (core.representations(core.exchange.squared()) - representation(representation)).svdvals().max() < 1e-12


def test_arbitrary_maps_encode_as_multivectors_and_compose_by_multiplication():
    rng = np.random.default_rng(0)
    maps = core.context.extensor(core.StateMap, rng.normal(size=(2, 4, 4)))
    first_map, second_map = maps
    first_element = core.map_multivector(first_map)
    second_element = core.map_multivector(second_map)

    assert (core.representations(first_element) - first_map).svdvals().max() < 1e-12
    assert (core.representations(second_element) - second_map).svdvals().max() < 1e-12
    composition = core.representations(first_element * second_element)
    assert (composition - first_map(second_map)).svdvals().max() < 1e-12


@pytest.mark.parametrize("n_dims", [2, 3, 4])
def test_split_algebra_recovers_arbitrary_maps_across_dimensions(n_dims: int):
    positive_names, negative_names = "xyzt"[:n_dims], "uvws"[:n_dims]
    signature = "".join(name + "+" for name in positive_names) + "".join(name + "-" for name in negative_names)
    ga = Algebra(signature)
    context = NumpyContext(ga)
    mv = context.multivector
    Full = ga.gatype.full()
    State = Full.derive.within(" ".join(positive_names))
    positive = [getattr(mv, name) for name in positive_names]
    negative = [getattr(mv, name) for name in negative_names]
    creators = [(first + second) / 2 for first, second in zip(positive, negative)]
    removers = [(first - second) / 2 for first, second in zip(positive, negative)]
    projector = reduce(mul, (remove * create for remove, create in zip(removers, creators)))
    n_state_components = 2 ** n_dims
    embedder = State * projector
    state_readout = (n_state_components * Full).cast(State)
    representations = state_readout(Full * embedder)
    rng = np.random.default_rng(0)
    arbitrary = context.extensor(ga.gatype((State, State)), rng.normal(size=(n_state_components, n_state_components)))
    pairing = Full.scalar_product(Full)
    encoding = pairing.solve(representations(Full, arbitrary).trace(0, 2) / n_state_components)

    assert (representations(encoding) - arbitrary).svdvals().max() < 1e-12


def test_shear_figure_renders_grid_and_square():
    grid = scenarios.grid_segments(core.basis, scenarios.extent, 3)
    transformation = core.representations((core.shear_generator * scenarios.shear_amount).exp())
    vector_shear = transformation(core.Vector).cast(core.Vector)
    figure = render.shear_figure(
        grid, vector_shear(grid), scenarios.boundary, vector_shear(scenarios.boundary), vector_shear(core.basis),
    )
    assert len(figure.axes) == 1
    plt.close(figure)
