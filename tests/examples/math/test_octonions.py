"""The octonion product, its Clifford actions, and its exceptional rotation symmetries."""

import matplotlib.pyplot as plt
import numpy as np

from numga import Algebra, NumpyContext, stack
from examples.math.octonions import core, render, scenarios


def test_norm_composition_alternativity_and_nonassociativity():
    rng = np.random.default_rng(0)
    first, second, third = core.mv(core.Octonion, rng.normal(size=(3, 12, 8)))
    product = core.multiply(first, second)
    error = core.norm_squared(product) - core.norm_squared(first) * core.norm_squared(second)
    np.testing.assert_allclose(error.kernel, 0, atol=1e-10)
    np.testing.assert_allclose(core.associator(first, first, third).kernel, 0, atol=1e-10)
    np.testing.assert_allclose(core.associator(first, second, second).kernel, 0, atol=1e-10)
    np.testing.assert_array_equal(
        (core.associator(core.mv.a, core.mv.b, core.mv.d) - 2 * core.mv.g).kernel, 0,
    )


def test_left_multiplications_satisfy_all_clifford_relations():
    vectors = core.mv.vector(np.eye(core.ga.dimension))
    actions = core.multiply(vectors, core.Octonion)
    compositions = actions[:, None](actions[None, :]) + actions[None, :](actions[:, None])
    reference = 2 * vectors[:, None].scalar_product(vectors[None, :]) * core.Octonion
    np.testing.assert_array_equal((compositions - reference).kernel, 0)


def test_fourteen_generators_obey_the_product_rule_and_exponentiate_to_automorphisms():
    generators = core.symmetry_generators(scenarios.TOLERANCE)
    assert len(generators) == 14
    basis = core.mv(core.Octonion, np.eye(8))
    first, second = basis[:, None], basis[None, :]
    defect = core.derivation_defect(generators[:, None, None], first, second)
    np.testing.assert_allclose(defect.kernel, 0, atol=1e-12)

    # The infinitesimal symmetries close under commutators, and finite turns preserve the product.
    brackets = generators[:, None].commutator(generators[None, :])
    np.testing.assert_allclose(brackets.commutator(core.orientation).kernel, 0, atol=1e-12)
    rotors = (generators * (scenarios.ANGLE / 2)).exp()
    finite = core.multiplication_defect(rotors[:, None, None], first, second)
    np.testing.assert_allclose(finite.kernel, 0, atol=1e-12)


def test_the_fano_plane_draws_the_seven_triples_in_the_order_of_their_products():
    assert render.triples(core.cycles) == ["abc", "ade", "agf", "bdf", "beg", "cdg", "cfe"]
    plt.close(render.draw_fano(core.cycles))


def test_doubling_in_the_algebra_of_space_gives_the_same_octonions_and_the_split_ones():
    space = NumpyContext(core.space).multivector
    a, b, d = space.x, space.y, space.z
    c, e, f = core.space_multiply(a, b), core.space_multiply(a, d), core.space_multiply(b, d)
    g = core.space_multiply(c, d)
    images = stack([a, b, c, d, e, f, g])                                     # [images] SpaceOctonion
    both_orders = core.space_multiply(images[:, None], images[None, :]) + core.space_multiply(images[None, :], images[:, None])
    assert np.abs((core.space_multiply(images, images) + 1).kernel).max() == 0
    assert np.abs(both_orders[~np.eye(images.shape[0], dtype=bool)].kernel).max() == 0
    # The three triples not used to build the images hold by themselves.
    assert np.abs((core.space_multiply(a, g) - f).kernel).max() == np.abs((core.space_multiply(b, e) - g).kernel).max() == np.abs((core.space_multiply(c, f) - e).kernel).max() == 0
    # Lengths multiply in both signatures; in the split one, 1 + z has length zero and its conjugate kills it.
    rng = np.random.default_rng(0)
    for algebra in (core.space, Algebra("x+y+z-")):
        mv = NumpyContext(algebra).multivector
        first, second = mv.full(rng.normal(size=(2, 12, 8)))
        error = core.space_multiply(first, second).scalar_norm_squared() - first.scalar_norm_squared() * second.scalar_norm_squared()
        assert np.abs(error.kernel).max() < 1e-10
    null = 1 + NumpyContext(Algebra("x+y+z-")).multivector.z
    assert np.abs(null.scalar_norm_squared().kernel).max() == 0 and np.abs(core.space_multiply(null, null.scalar_negation()).kernel).max() == 0
