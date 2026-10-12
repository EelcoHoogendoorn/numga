"""The octonion product, its Clifford actions, and its exceptional rotation symmetries."""

import matplotlib.pyplot as plt
import numpy as np

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
