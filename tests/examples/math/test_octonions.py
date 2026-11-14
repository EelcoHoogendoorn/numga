"""The octonion product, its Clifford actions, and its exceptional rotation symmetries."""

import matplotlib.pyplot as plt
import numpy as np

from numga import Algebra, NumpyContext, stack
from examples.math.octonions import core, render, scenarios


def test_norm_composition_alternativity_and_nonassociativity():
    rng = np.random.default_rng(0)
    first, second, third = core.mv(core.Octonion, rng.normal(size=(3, 4, 8)))
    combined = core.product(first, second)
    error = combined.symmetric_conjugate_product() - first.symmetric_conjugate_product() * second.symmetric_conjugate_product()
    assert (error.abs() < 1e-10).all()
    assert (core.associator(first, first, third).scalar_norm_squared().abs() < 1e-10).all()
    assert (core.associator(first, second, second).scalar_norm_squared().abs() < 1e-10).all()
    assert (core.associator(core.mv.a, core.mv.b, core.mv.d) - 2 * core.mv.g).scalar_norm_squared().abs() < 1e-10


def test_left_multiplications_satisfy_all_clifford_relations():
    vectors = core.mv.vector(np.eye(core.ga.dimension))
    actions = core.product(vectors, core.Octonion)
    compositions = actions[:, None](actions[None, :]) + actions[None, :](actions[:, None])
    reference = 2 * vectors[:, None].scalar_product(vectors[None, :]) * core.Octonion
    assert ((compositions - reference).svdvals() < 1e-10).all()


def test_the_rotations_preserving_the_product_have_fourteen_generators():
    assert len(core.symmetry_generators(scenarios.tolerance)) == 14


def test_doubling_in_the_algebra_of_space_gives_the_same_octonions_and_the_split_ones():
    space = NumpyContext(core.space).multivector
    a, b, d = space.x, space.y, space.z
    c, e, f = core.space_product(a, b), core.space_product(a, d), core.space_product(b, d)
    g = core.space_product(c, d)
    images = stack([a, b, c, d, e, f, g])                                       # [7] Full+
    both_orders = core.space_product(images[:, None], images[None, :]) + core.space_product(images[None, :], images[:, None])
    assert ((core.space_product(images, images) + 1).scalar_norm_squared().abs() < 1e-10).all()
    assert (both_orders[~np.eye(7, dtype=bool)].scalar_norm_squared().abs() < 1e-10).all()
    # The three triples not used to build the images hold by themselves.
    assert (core.space_product(a, g) - f).scalar_norm_squared().abs() < 1e-10
    assert (core.space_product(b, e) - g).scalar_norm_squared().abs() < 1e-10
    assert (core.space_product(c, f) - e).scalar_norm_squared().abs() < 1e-10
    # Lengths multiply in both signatures; in the split one, 1 + z has length zero and its conjugate kills it.
    rng = np.random.default_rng(0)
    for algebra in (core.space, Algebra("x+y+z-")):
        mv = NumpyContext(algebra).multivector
        first, second = mv.full(rng.normal(size=(2, 4, 8)))
        error = core.space_product(first, second).scalar_norm_squared() - first.scalar_norm_squared() * second.scalar_norm_squared()
        assert (error.abs() < 1e-10).all()
    null_element = 1 + NumpyContext(Algebra("x+y+z-")).multivector.z
    assert null_element.scalar_norm_squared().abs() < 1e-10
    assert core.space_product(null_element, null_element.scalar_negation()).scalar_norm_squared().abs() < 1e-10


def test_the_fano_plane_draws():
    figure = render.fano_figure(core.cycles)
    plt.close(figure)
