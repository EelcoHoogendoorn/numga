import numpy as np
import pytest

from numga.algebra import Algebra
from numga.backend import NumpyContext


def test_cl2_even_geometric_product_is_complex_multiplication():
    algebra = Algebra("x+y+")
    spaces = algebra.subspace
    factory = algebra.operator
    context = NumpyContext(algebra)
    even = spaces.even()
    product = factory.geometric_product(even, even)

    assert product.axes == (even, even, even)
    assert product.kernel.values.tolist() == [
        [[1, 0], [0, -1]],
        [[0, 1], [1, 0]],
    ]

    left = context.extensor(even, [1, 2])
    right = context.extensor(even, [3, 4])
    result = product(left, right)

    np.testing.assert_allclose(
        result.kernel,
        [-5, 10],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


def test_euclidean_3d_cross_product_orientation():
    algebra = Algebra("x+y+z+")
    spaces = algebra.subspace
    factory = algebra.operator
    context = NumpyContext(algebra)
    vector = spaces.vector()
    cross = factory.cross(vector)
    x = context.extensor(vector, [1, 0, 0])
    y = context.extensor(vector, [0, 1, 0])

    assert cross.axes == (vector, vector, vector)
    np.testing.assert_allclose(
        cross(x, y).kernel,
        [0, 0, 1],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        cross(y, x).kernel,
        [0, 0, -1],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        cross(x, x).kernel,
        [0, 0, 0],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


def test_direct_pga2_inertia_expression_batches_reduces_and_applies():
    algebra = Algebra("x+y+w0")
    spaces = algebra.subspace
    context = NumpyContext(algebra)
    point = spaces.antivector()
    bivector = spaces.bivector()
    dual_bivector = spaces.antibivector()

    assert tuple(algebra.blade_name(mask) for mask in bivector.masks) == (
        "xy",
        "xw",
        "yw",
    )
    assert tuple(algebra.blade_name(mask) for mask in dual_bivector.masks) == (
        "x",
        "y",
        "w",
    )

    # With canonical axes (xy, xw, yw), an affine point (x, y) has
    # homogeneous dual-vector coordinates (1, -y, x).
    points = context.extensor(
        point,
        [
            [1, 0, 0],  # (0, 0)
            [1, 0, 1],  # (1, 0)
            [1, -1, 0],  # (0, 1)
        ],
    )

    point_inertia = point.regressive(point.commutator(bivector))
    assert point_inertia.context is algebra.exact
    assert point_inertia.axes == (dual_bivector, point, point, bivector)
    assert point_inertia.arity == 3

    # Binding both point positions atomically retains the repeated-argument
    # relationship for future diagonal/symmetry inference.
    point_inertias = point_inertia.bind({0: points, 1: points})
    direct_point_inertias = points.regressive(points.commutator(bivector))

    expected_point_inertias = np.asarray(
        [
            [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
            [[-1, 0, 1], [0, -1, 0], [1, 0, -1]],
            [[0, 0, 1], [-1, -1, 0], [1, 1, 0]],
        ],
        dtype=float,
    )
    assert point_inertias.shape == (3,)
    assert point_inertias.axes == (dual_bivector, bivector)
    assert point_inertias.arity == 1
    np.testing.assert_allclose(
        point_inertias.kernel,
        expected_point_inertias,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        direct_point_inertias.kernel,
        point_inertias.kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )

    # The opposite commutator order is equally composable, but antisymmetry
    # makes it the negative map rather than the inertia convention used here.
    opposite_order = points.regressive(bivector.commutator(points))
    np.testing.assert_allclose(
        opposite_order.kernel,
        -expected_point_inertias,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )

    inertia = point_inertias.sum(axis=0)
    expected_inertia = np.asarray(
        [[-1, 0, 3], [-1, -3, 0], [2, 1, -1]],
        dtype=float,
    )
    assert inertia.shape == ()
    assert inertia.axes == (dual_bivector, bivector)
    assert inertia.arity == 1
    np.testing.assert_allclose(
        inertia.kernel,
        expected_inertia,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )

    rates = context.extensor(bivector, [[2, 3, 5], [-1, 4, 2]])
    momenta = inertia(rates)
    assert momenta.shape == (2,)
    assert momenta.subspace is dual_bivector
    np.testing.assert_allclose(
        momenta.kernel,
        [[13, -11, 2], [7, -11, 0]],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )

    energy_twice = rates.regressive(momenta)
    assert energy_twice.subspace is spaces.scalar()
    np.testing.assert_allclose(
        energy_twice.kernel,
        [[102], [58]],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    assert np.all(energy_twice.kernel >= 0)

    opposite_momenta = opposite_order.sum(axis=0)(rates)
    np.testing.assert_allclose(
        rates.regressive(opposite_momenta).kernel,
        -energy_twice.kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )

    per_point_momenta = point_inertias.reshape(3, 1)(rates.reshape(1, 2))
    assert per_point_momenta.shape == (3, 2)
    np.testing.assert_allclose(
        per_point_momenta.kernel,
        [
            [[5, -3, 0], [2, -4, 0]],
            [[3, -3, -3], [3, -4, -3]],
            [[5, -5, 5], [2, -3, 3]],
        ],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        per_point_momenta.sum(axis=0).kernel,
        momenta.kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


def test_sandwich_expression_repeats_sandwicher_and_broadcasts_unary_map():
    algebra = Algebra("x+y+")
    spaces = algebra.subspace
    factory = algebra.operator
    context = NumpyContext(algebra)
    sandwicher = spaces.even()
    passenger = spaces.vector()
    sandwich = factory.sandwich(sandwicher, passenger)

    assert sandwicher.sandwich(passenger) is sandwich
    assert algebra.gatype.even().sandwich(algebra.gatype.vector()) is sandwich
    assert sandwich.context is algebra.exact
    assert sandwich.arity == 3
    assert sandwich.axes == (passenger, sandwicher, passenger, sandwicher)
    assert sandwich.kernel.shape == (2, 2, 2, 2)

    sandwichers = context.extensor(
        sandwicher,
        [
            [[1, 0]],
            [[1, 1]],
        ],
    )
    maps = sandwichers.sandwich(passenger)
    explicit = sandwich.bind({0: sandwichers, 2: sandwichers})

    expected_maps = np.asarray(
        [
            [[[1, 0], [0, 1]]],
            [[[0, 2], [-2, 0]]],
        ],
        dtype=float,
    )
    assert maps.shape == (2, 1)
    assert maps.axes == (passenger, passenger)
    assert maps.arity == 1
    np.testing.assert_allclose(
        maps.kernel,
        expected_maps,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        maps.kernel,
        explicit.kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    composed_maps = sandwichers.sandwich(factory.identity(passenger))
    assert composed_maps.shape == maps.shape
    assert composed_maps.axes == maps.axes
    assert composed_maps.arity == maps.arity
    np.testing.assert_allclose(
        composed_maps.kernel,
        maps.kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )

    points = context.multivector.vector(
        [[[2, 3], [-1, 4], [0.5, -2]]],
    )
    transformed = maps(points)
    direct = sandwichers.sandwich(points)

    expected_points = np.asarray(
        [
            [[2, 3], [-1, 4], [0.5, -2]],
            [[6, -4], [8, 2], [-4, -1]],
        ],
        dtype=float,
    )
    assert transformed.shape == (2, 3)
    assert transformed.axes == (passenger,)
    assert transformed.arity == 0
    np.testing.assert_allclose(
        transformed.kernel,
        expected_points,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        direct.kernel,
        transformed.kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )

    with pytest.raises(ValueError, match="repeated sandwicher must be nullary"):
        factory.identity(sandwicher).sandwich(passenger)
