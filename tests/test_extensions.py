import numpy as np
import pytest

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
    assert exact_inverse.kernel.to_object_array().tolist() == [
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


def test_concrete_gatype_registrations_are_deliberately_algebra_local():
    Extensor.origin = ExtensionMethod("origin")
    left = Algebra("x+y+")
    right = Algebra("x+y+")
    unregistered = Algebra("x+y+")

    @Extensor.origin.register(left.gatype.scalar())
    def left_origin(_value):
        return "left"

    @Extensor.origin.register(right.gatype.scalar())
    def right_origin(_value):
        return "right"

    left_value = NumpyContext(left).extensor(left.subspace.scalar(), [1])
    right_value = NumpyContext(right).extensor(right.subspace.scalar(), [1])
    unregistered_value = NumpyContext(unregistered).extensor(
        unregistered.subspace.scalar(),
        [1],
    )

    assert left_value.origin() == "left"
    assert right_value.origin() == "right"
    assert "origin" in dir(left_value)
    assert "origin" in dir(unregistered_value)
    with pytest.raises(LookupError, match="no 'origin' implementation"):
        unregistered_value.origin()


def test_extension_method_infers_multiple_dispatch_arity():
    Extensor.choose = ExtensionMethod("choose")
    left = Algebra("x+y+")
    right = Algebra("x+y+")

    @Extensor.choose.register(left.gatype.vector(), left.gatype.vector())
    def choose_left(first, second):
        return first, second, "left"

    @Extensor.choose.register(right.gatype.vector(), right.gatype.vector())
    def choose_right(first, second):
        return first, second, "right"

    left_context = NumpyContext(left)
    right_context = NumpyContext(right)
    left_first = left_context.extensor(left.subspace.vector(), [1, 2])
    left_second = left_context.extensor(left.subspace.vector(), [3, 4])
    right_first = right_context.extensor(right.subspace.vector(), [1, 2])
    right_second = right_context.extensor(right.subspace.vector(), [3, 4])

    assert Extensor.choose.operand_count == 2
    assert left_first.choose(left_second) == (left_first, left_second, "left")
    assert Extensor.choose(right_first, right_second) == (
        right_first,
        right_second,
        "right",
    )

    with pytest.raises(TypeError, match="dispatches 2 operands, not 1"):
        Extensor.choose.register(right.gatype.vector())


def test_extension_registration_requires_complete_gatypes():
    Extensor.invalid = ExtensionMethod("invalid")
    algebra = Algebra("x+y+")

    with pytest.raises(TypeError, match="must be a GAType"):
        Extensor.invalid.register(algebra.subspace.vector())
    assert Extensor.invalid.operand_count is None

    abandoned = Extensor.invalid.register(algebra.gatype.vector())
    assert Extensor.invalid.operand_count is None
    with pytest.raises(TypeError, match="must be callable"):
        abandoned(None)
    assert Extensor.invalid.operand_count is None

    @Extensor.invalid.register(
        algebra.gatype.vector(),
        algebra.gatype.vector(),
    )
    def valid_binary(left, right):
        return left, right

    assert Extensor.invalid.operand_count == 2


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


def test_multi_operand_predicate_receives_only_complete_gatypes():
    Extensor.type_pair = ExtensionMethod("type_pair")
    observed = []

    @Extensor.type_pair.register(
        lambda left_type, right_type: (
            observed.append((left_type, right_type)) is None
        )
    )
    def selected(left, right):
        return left, right

    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    left = context.extensor(algebra.subspace.vector(), [1, 2])
    right = context.extensor(algebra.subspace.scalar(), [3])

    assert left.type_pair(right) == (left, right)
    assert Extensor.type_pair.operand_count == 2
    assert observed == [(left.gatype, right.gatype)]


def test_trace_extension_square_maps_and_scalars():
    from fractions import Fraction

    algebra = Algebra("x+y+z+")
    context = NumpyContext(algebra)
    V = algebra.subspace.vector()
    I_v = algebra.operator.identity(V)

    # 1. Identity extensor
    ext_I = context.extensor(I_v.gatype, np.eye(3))
    tr_I = ext_I.trace()
    assert tr_I.subspace is algebra.subspace.scalar()
    np.testing.assert_allclose(tr_I.kernel, [3.0])

    # 2. Batch of operators
    batch_mat = np.stack([np.eye(3) * i for i in range(4)])
    batch_op = context.extensor(I_v.gatype, batch_mat)
    tr_batch = batch_op.trace()
    assert tr_batch.shape == (4,)
    np.testing.assert_allclose(tr_batch.kernel, [[0.0], [3.0], [6.0], [9.0]])

    # 3. Exact rational trace
    exact_op = algebra.exact.extensor(I_v.gatype, np.eye(3, dtype=int))
    tr_exact = exact_op.trace()
    assert tr_exact.context is algebra.exact
    assert tr_exact.kernel.to_object_array().tolist() == [Fraction(3, 1)]

    # 4. Permuted blade layout (realigned via AxisTransform)
    V_rev = algebra.subspace.from_blades(["z", "y", "x"])
    perm_mat = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    perm_op = context.extensor(algebra.gatype((V_rev, V)), perm_mat)
    np.testing.assert_allclose(perm_op.trace().kernel, [3.0])

    # 5. Scalar trace
    scalar = context.multivector.scalar([42.0])
    np.testing.assert_allclose(scalar.trace().kernel, [42.0])

    # 6. Incompatible / non-square maps raise TypeError
    B = algebra.subspace.bivector()
    non_square = context.extensor(algebra.gatype((V, B)), np.zeros((3, 3)))
    with pytest.raises(TypeError, match="trace requires an endomorphism with matching subspace support"):
        non_square.trace()
