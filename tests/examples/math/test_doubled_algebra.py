"""Products in the split algebra act as every linear map on exterior states."""

from functools import reduce
from operator import mul

import numpy as np
import pytest

from numga import Algebra, NumpyContext, stack
from examples.math.doubled_algebra import core
from examples.math.extensor_representations.core import readout


def test_null_generators_insert_remove_and_recover_the_familiar_product():
    vectors = stack([core.mv.x, core.mv.y])
    insertions = core.representations(core.creators)
    removals = core.representations(core.removers)

    np.testing.assert_array_equal((core.projector.squared() - core.projector).kernel, 0)
    np.testing.assert_array_equal((core.removers * core.projector).kernel, 0)
    np.testing.assert_array_equal((core.state_readout(core.embedder) - core.State).kernel, 0)
    np.testing.assert_array_equal((insertions - (vectors ^ core.State)).kernel, 0)
    np.testing.assert_array_equal((removals - vectors.left_contraction(core.State)).kernel, 0)
    np.testing.assert_array_equal((core.representations(vectors) - vectors * core.State).kernel, 0)

    # Repeated insertion/removal vanishes; the mixed pair evaluates the vector.
    insert_twice = insertions[:, None](insertions[None, :]) + insertions[None, :](insertions[:, None])
    remove_twice = removals[:, None](removals[None, :]) + removals[None, :](removals[:, None])
    mixed = removals[:, None](insertions[None, :]) + insertions[None, :](removals[:, None])
    np.testing.assert_array_equal(insert_twice.kernel, 0)
    np.testing.assert_array_equal(remove_twice.kernel, 0)
    np.testing.assert_array_equal((mixed - vectors[:, None].scalar_product(vectors[None, :]) * core.State).kernel, 0)


def test_dyads_extend_to_blades_by_the_product_rule():
    dyads = core.representations(core.creators[:, None] * core.removers[None, :])
    vectors = stack([core.mv.x, core.mv.y])
    expected = vectors[:, None] ^ vectors[None, :].left_contraction(core.State)
    np.testing.assert_array_equal((dyads - expected).kernel, 0)

    extension = dyads(core.Vector ^ core.Vector)
    product_rule = dyads(core.Vector) ^ core.Vector
    product_rule = product_rule + (core.Vector ^ dyads(core.Vector))
    np.testing.assert_array_equal((extension - product_rule).kernel, 0)

    # Each diagonal dyad counts its own factor in a blade; together they count the grade.
    count = core.representations((core.creators * core.removers).sum(axis=0))
    state = 3 + core.mv.x - 2 * core.mv.y + 4 * core.mv.xy
    np.testing.assert_array_equal((count(state) - (core.mv.x - 2 * core.mv.y + 8 * core.mv.xy)).kernel, 0)


def test_exponentiating_a_shear_moves_vectors_and_their_wedge_together():
    amounts = np.array([-1.25, 0., .75])
    motion = core.shear(amounts)
    expected = core.Vector + amounts * core.mv.x * core.mv.y.scalar_product(core.Vector)
    np.testing.assert_allclose((motion(core.Vector) - expected).kernel, 0, atol=1e-12)

    carried_area = motion(core.Vector ^ core.Vector)
    rebuilt_area = motion(core.Vector) ^ motion(core.Vector)
    np.testing.assert_allclose((carried_area - rebuilt_area).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((motion(core.mv.xy) - core.mv.xy).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((motion(core.mv.scalar()) - core.mv.scalar()).kernel, 0, atol=1e-12)


def test_mixed_grade_exchange_and_product_are_map_composition():
    exchange = core.creators[0] * core.creators[1] + core.removers[1] * core.removers[0]
    representation = core.representations(exchange)
    state = 2 + 3 * core.mv.x - core.mv.y + 5 * core.mv.xy
    np.testing.assert_array_equal((representation(state) - (5 + 2 * core.mv.xy)).kernel, 0)
    np.testing.assert_array_equal((representation(representation(state)) - (2 + 5 * core.mv.xy)).kernel, 0)
    np.testing.assert_array_equal((core.representations(exchange.squared()) - representation(representation)).kernel, 0)


def test_arbitrary_maps_encode_as_multivectors_and_compose_by_multiplication():
    rng = np.random.default_rng(0)
    maps = core.context.extensor(core.StateMap, rng.normal(size=(2, 4, 4)))
    first, second = maps
    first_element, second_element = core.encode(first), core.encode(second)

    np.testing.assert_allclose((core.representations(first_element) - first).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((core.representations(second_element) - second).kernel, 0, atol=1e-12)
    composition = core.representations(first_element * second_element)
    np.testing.assert_allclose((composition - first(second)).kernel, 0, atol=1e-12)


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_split_algebra_recovers_arbitrary_maps_through_eight_dimensions(dimension):
    positive_names, negative_names = "xyzt"[:dimension], "uvws"[:dimension]
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
    state_size = 2 ** dimension
    embedder = State * projector
    state_readout = (state_size * Full).cast(State)
    representations = state_readout(Full * embedder)
    rng = np.random.default_rng(0)
    arbitrary = context.extensor(ga.gatype((State, State)), rng.normal(size=(state_size, state_size)))
    recovered = readout(representations, arbitrary, Full)

    # Recovering unrestricted maps exercises the complete algebra, rather than one grade.
    np.testing.assert_allclose((representations(recovered) - arbitrary).kernel, 0, atol=1e-12)
