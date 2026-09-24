import numpy as np

from numga import (
    Algebra,
    CoefficientOrthogonal,
    ExtensionMethod,
    Extensor,
    GATypePattern,
    NumpyContext,
)


def test_one_inverse_extension_dispatches_nullary_and_unary_extensors():
    algebra = Algebra("x+y+")
    spaces = algebra.subspace
    gatypes = algebra.gatype
    context = NumpyContext(algebra)

    rotor = context.extensor(gatypes.rotor(), [0, 1])
    rotor_inverse = rotor.inverse()
    assert rotor_inverse.gatype is rotor.gatype
    np.testing.assert_array_equal(rotor_inverse.kernel, [0, -1])

    vector = spaces.vector()
    rotation_type = gatypes(
        (vector, vector),
        (CoefficientOrthogonal,),
    )
    exact_rotation = algebra.exact.extensor(
        rotation_type,
        [[0, -1], [1, 0]],
    )
    exact_inverse = exact_rotation.inverse()
    assert exact_inverse.context is algebra.exact
    assert exact_inverse.gatype is rotation_type
    assert exact_inverse.kernel.values.tolist() == [
        [0, 1],
        [-1, 0],
    ]

    coefficients = np.asarray(
        [
            [[0, -1], [1, 0]],
            [[0, 1], [-1, 0]],
        ],
        dtype=float,
    )
    rotations = context.extensor(rotation_type, coefficients)
    rotation_inverses = rotations.inverse()

    assert "inverse" in dir(rotations)
    assert rotation_inverses is not rotations
    assert rotation_inverses.shape == (2,)
    assert rotation_inverses.gatype is rotation_type
    np.testing.assert_array_equal(
        rotation_inverses.kernel,
        coefficients.transpose(0, 2, 1),
    )
    np.testing.assert_array_equal(rotations.kernel, coefficients)

    plain_matrix = context.extensor(
        gatypes((vector, vector)),
        coefficients[0],
    )
    np.testing.assert_array_equal(
        plain_matrix.inverse().kernel, coefficients[0].T
    )

    # Neither registration came from this algebra's GAType factory.
    other = Algebra("a+b+c+")
    other_rotor = NumpyContext(other).extensor(
        other.gatype.rotor(),
        [0, 1, 0, 0],
    )
    np.testing.assert_array_equal(other_rotor.inverse().kernel, [0, -1, 0, 0])


def test_explicit_pattern_supports_an_unconstrained_extensor_arity():
    Extensor.kind = ExtensionMethod("kind")

    @Extensor.kind.register(GATypePattern(arity=0))
    def any_value(_value):
        return "value"

    @Extensor.kind.register(GATypePattern.map())
    def any_map(_value):
        return "map"

    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    value = context.extensor(algebra.subspace.vector(), [1, 2])
    vector = algebra.subspace.vector()
    map_value = context.extensor(
        algebra.gatype((vector, vector)),
        [[1, 0], [0, 1]],
    )

    assert value.kind() == "value"
    assert map_value.kind() == "map"


def test_opaque_gatype_predicates_are_a_prioritized_low_level_tier():
    Extensor.strategy = ExtensionMethod("strategy")

    @Extensor.strategy.register(GATypePattern(arity=0))
    def generic(_value):
        return "generic"

    optimized_algebra = Algebra("x+y+z+w0")
    optimized_axis = optimized_algebra.subspace.even()
    calls = []

    # This can be imported after the generic implementation and still wins.
    @Extensor.strategy.register(
        lambda gatype: (
            calls.append(gatype) is None
            and gatype.algebra is optimized_algebra
            and gatype.subspaces == (optimized_axis,)
        )
    )
    def optimized(_value):
        return "optimized"

    context = NumpyContext(optimized_algebra)
    rotor = context.extensor(optimized_algebra.gatype.rotor(), [1] * 8)

    assert rotor.strategy() == "optimized"
    assert calls == [rotor.gatype]

    other = Algebra("x+y+z+w0")
    other_value = NumpyContext(other).extensor(other.subspace.even(), [1] * 8)
    assert other_value.strategy() == "generic"


def test_opaque_predicate_position_controls_only_predicate_order():
    Extensor.route = ExtensionMethod("route")

    @Extensor.route.register(lambda _gatype: True)
    def first(_value):
        return "first"

    @Extensor.route.register(lambda _gatype: True, position=0)
    def inserted(_value):
        return "inserted"

    algebra = Algebra("x+")
    value = NumpyContext(algebra).extensor(algebra.subspace.scalar(), [1])
    assert value.route() == "inserted"


