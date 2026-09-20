from fractions import Fraction

import numpy as np
import pytest

from numga.algebra import Algebra
from numga.backend import NumpyContext
from numga.binding import BindingPlan, TypeRules
from numga.extensor import Extensor
from numga.gatype import GAType
from numga.operator import SymbolicKernel


@pytest.mark.parametrize("execution", ("dense", "sparse"))
@pytest.mark.parametrize("operand_arity", (0, 1))
def test_warm_matrix_application_reuses_static_plans_without_recoercing_results(
    monkeypatch, execution, operand_arity,
):
    algebra = Algebra("x+y+z+")
    context = NumpyContext(algebra, execution=execution)
    vector = algebra.subspace.vector()
    matrix_type = algebra.gatype((vector, vector))
    left = context.extensor(matrix_type, np.arange(9.).reshape(3, 3))
    right = (
        context.extensor(matrix_type, np.eye(3)) if operand_arity
        else context.multivector.vector([1., 2., 3.])
    )
    expected = left(right).kernel
    other_left = context.extensor(matrix_type, 2 * left.kernel)

    def unexpected_coercion(self, kernel):
        raise AssertionError("computed results must not re-enter public input coercion")

    monkeypatch.setattr(NumpyContext, "prepare_kernel", unexpected_coercion)
    def repeated_planning(*args, **kwargs):
        raise AssertionError("warm application must not repeat binding setup")

    monkeypatch.setattr(BindingPlan, "build", repeated_planning)
    monkeypatch.setattr(BindingPlan, "from_types", repeated_planning)
    monkeypatch.setattr(BindingPlan, "__hash__", repeated_planning)
    monkeypatch.setattr(TypeRules, "bind", repeated_planning)
    monkeypatch.setattr(NumpyContext, "lower", repeated_planning)
    monkeypatch.setattr(NumpyContext, "execute_bind", repeated_planning)
    result = left(right)
    batched = left.reshape(1)(right.broadcast_to((2,)))

    np.testing.assert_allclose(result.kernel, expected)
    np.testing.assert_allclose(other_left(right).kernel, 2 * expected)
    np.testing.assert_allclose(batched.kernel, np.broadcast_to(expected, (2,) + expected.shape))
    assert not result.kernel.flags.writeable


def test_operator_and_extensor_shapes_are_output_first():
    algebra = Algebra("x+y+z+")
    spaces = algebra.subspace
    output = spaces.vector()
    first = spaces.scalar()
    second = spaces.even()
    gatype = GAType((output, first, second))
    coefficients = np.arange(12).reshape(3, 1, 4)

    operator = algebra.operator.build((output, first, second), coefficients)
    extensor = NumpyContext(algebra).extensor(
        gatype, np.broadcast_to(coefficients, (5, 3, 1, 4))
    )

    assert operator.axes == (output, first, second)
    assert operator.kernel.shape == (3, 1, 4)
    assert extensor.axes == (output, first, second)
    assert extensor.structural_shape == (3, 1, 4)
    assert extensor.shape == (5,)
    assert extensor.kernel.shape == (5, 3, 1, 4)

    with pytest.raises(ValueError, match="structural shape"):
        algebra.operator.build(
            (output, first, second), np.zeros((1, 4, 3), dtype=int)
        )


def test_exact_extensor_full_and_partial_bind_are_exact():
    algebra = Algebra("x+y+")
    spaces = algebra.subspace
    factory = algebra.operator
    even = spaces.even()
    product = factory.geometric_product(even, even)
    left = factory.build((even,), [Fraction(1, 2), Fraction(1, 3)])
    right = factory.build((even,), [3, 6])

    partial = product.bind({0: left})
    sequential = partial.bind({0: right})
    full = product(left, right)

    assert isinstance(partial, Extensor)
    assert partial.axes == (even, even)
    assert partial.kernel.shape == (2, 2)
    assert sequential.gatype == full.gatype
    assert sequential.kernel == full.kernel
    assert full.axes == (even,)
    assert full.kernel.to_object_array().tolist() == [
        Fraction(-1, 2),
        Fraction(4),
    ]


def test_exact_commutator_and_regressive_sign_conventions():
    algebra = Algebra("x-y+w0")
    spaces = algebra.subspace

    def blade(name: str):
        return spaces.from_masks((algebra.parse_blade(name).mask,))

    x = blade("x")
    y = blade("y")
    xy = blade("xy")
    xw = blade("xw")
    yw = blade("yw")
    w = blade("w")

    xy_commutator = x.commutator(y)
    yx_commutator = y.commutator(x)
    assert xy_commutator.axes == (xy, x, y)
    assert yx_commutator.axes == (xy, y, x)
    assert xy_commutator.kernel.to_object_array().tolist() == [[[Fraction(1)]]]
    assert yx_commutator.kernel.to_object_array().tolist() == [[[Fraction(-1)]]]

    # This fixes the right-Hodge signs in a signature containing both a
    # negative generator and a null generator.
    regressive = xw.regressive(yw)
    assert regressive.axes == (w, xw, yw)
    assert regressive.kernel.to_object_array().tolist() == [[[Fraction(-1)]]]

    # GAType and SubSpace holes select the same cached exact implementation.
    assert algebra.gatype(x).commutator(algebra.gatype(y)) is xy_commutator
    assert algebra.gatype(xw).regressive(yw) is regressive


def test_sandwich_accepts_any_carrier_and_squeezes_exact_symmetry_zeros():
    algebra = Algebra("x+y+")
    spaces = algebra.subspace
    vector = spaces.vector()
    full = spaces.full()

    vector_sandwich = vector.sandwich(vector)
    assert vector_sandwich.axes == (vector, vector, vector, vector)

    general_sandwich = full.sandwich(vector)
    assert general_sandwich.axes == (spaces.from_grades((0, 1)), full, vector, full)


def test_positive_arity_operand_inputs_are_spliced_at_the_bound_slot():
    algebra = Algebra("x+y+z+")
    spaces = algebra.subspace
    factory = algebra.operator
    output = spaces.k_vector(2)
    replaced = spaces.vector()
    untouched = spaces.full()
    inserted_first = spaces.scalar()
    inserted_second = spaces.even()

    target_values = np.arange(3 * 3 * 8).reshape(3, 3, 8)
    operand_values = np.arange(3 * 1 * 4).reshape(3, 1, 4)
    target = factory.build((output, replaced, untouched), target_values)
    operand = factory.build(
        (replaced, inserted_first, inserted_second), operand_values
    )

    exact = target.bind({0: operand})
    expected = np.zeros((3, 1, 4, 8), dtype=object)
    for out_index in range(3):
        for first_index in range(1):
            for second_index in range(4):
                for untouched_index in range(8):
                    expected[out_index, first_index, second_index, untouched_index] = sum(
                        Fraction(target_values[out_index, contracted, untouched_index])
                        * Fraction(operand_values[contracted, first_index, second_index])
                        for contracted in range(3)
                    )

    assert exact.axes == (
        output,
        inserted_first,
        inserted_second,
        untouched,
    )
    assert exact.kernel.shape == (3, 1, 4, 8)
    assert exact.kernel == SymbolicKernel(expected)

    context = NumpyContext(algebra)
    concrete = context.lower(target).bind({0: context.lower(operand)})
    assert concrete.axes == exact.axes
    assert concrete.kernel.shape == (3, 1, 4, 8)
    np.testing.assert_allclose(
        concrete.kernel,
        expected.astype(float),
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


def test_implicit_embedding_and_explicit_casts_have_clear_zero_fill_semantics():
    algebra = Algebra("x+y+z+")
    spaces = algebra.subspace
    factory = algebra.operator
    context = NumpyContext(algebra)
    vector = spaces.vector()
    x_axis = spaces.from_masks((algebra.parse_blade("x").blade,))
    x = context.extensor(x_axis, [5])
    general_vector = context.extensor(vector, [5, 7, 11])

    implicit = factory.identity(vector).bind(x)
    embedded = factory.cast(x_axis, vector).bind(x)
    projected = factory.cast(vector, x_axis).bind(general_vector)

    assert implicit.subspace is vector
    np.testing.assert_array_equal(implicit.kernel, [5, 0, 0])
    np.testing.assert_array_equal(embedded.kernel, [5, 0, 0])
    assert projected.subspace is x_axis
    np.testing.assert_array_equal(projected.kernel, [5])

    with pytest.raises(ValueError, match="project"):
        factory.identity(x_axis).bind(general_vector)


def test_numpy_operations_broadcast_all_batch_shape_axes():
    algebra = Algebra("x+y+")
    spaces = algebra.subspace
    factory = algebra.operator
    context = NumpyContext(algebra)
    even = spaces.even()
    product = factory.geometric_product(even, even)
    left_values = np.asarray([[[1, 2]], [[3, 4]]], dtype=float)
    right_values = np.asarray([[[5, 6], [7, 8], [9, 10]]], dtype=float)
    left = context.extensor(even, left_values)
    right = context.extensor(even, right_values)

    added = left + right
    forward = product.bind({0: left, 1: right})
    reverse = product.bind({1: right, 0: left})
    positional = product(left, right)
    sequential = product.bind({0: left}).bind({0: right})

    left_scalar, left_bivector = np.broadcast_arrays(
        left_values[..., 0], right_values[..., 0]
    )[0], np.broadcast_arrays(left_values[..., 1], right_values[..., 1])[0]
    right_scalar = np.broadcast_to(right_values[..., 0], (2, 3))
    right_bivector = np.broadcast_to(right_values[..., 1], (2, 3))
    expected = np.stack(
        (
            left_scalar * right_scalar - left_bivector * right_bivector,
            left_scalar * right_bivector + left_bivector * right_scalar,
        ),
        axis=-1,
    )

    assert left.shape == (2, 1)
    assert right.shape == (1, 3)
    assert added.shape == (2, 3)
    np.testing.assert_allclose(
        added.kernel,
        left_values + right_values,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )

    assert isinstance(forward, Extensor)
    assert forward.arity == 0
    assert forward.shape == (2, 3)
    assert forward.kernel.shape == (2, 3, 2)
    np.testing.assert_allclose(
        forward.kernel,
        expected,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    for actual in (reverse, positional, sequential):
        assert actual.gatype == forward.gatype
        np.testing.assert_allclose(
            actual.kernel,
            forward.kernel,
            rtol=1e-14,
            atol=1e-14,
            equal_nan=False,
        )


def test_exact_kernel_has_empty_shape_and_broadcasts_as_a_batch_constant():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    even = algebra.subspace.even()
    identity = algebra.operator.identity(even)
    coefficients = np.arange(24, dtype=float).reshape(2, 3, 2, 2)
    matrices = context.extensor(identity.gatype, coefficients)

    result = identity + matrices

    assert identity.shape == ()
    assert matrices.shape == (2, 3)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(
        result.kernel,
        coefficients + np.eye(2),
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


def test_extensor_addition_unions_structural_axes_and_broadcasts_shape():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    scalar = algebra.subspace.scalar()
    vector = algebra.subspace.vector()
    scalar_vector = scalar.union(vector)

    exact = algebra.operator.identity(scalar) + algebra.operator.identity(vector)
    assert exact.context is algebra.exact
    assert exact.axes == (scalar_vector, scalar_vector)
    assert exact.kernel == SymbolicKernel(np.eye(3, dtype=int))

    scalar_values = np.asarray([[[1]], [[2]]], dtype=float)
    vector_values = np.asarray([[[10, 20], [30, 40], [50, 60]]], dtype=float)
    scalars = context.extensor(scalar, scalar_values)
    vectors = context.extensor(vector, vector_values)

    result = scalars + vectors

    expected = np.empty((2, 3, 3), dtype=float)
    expected[..., :1] = np.broadcast_to(scalar_values, (2, 3, 1))
    expected[..., 1:] = np.broadcast_to(vector_values, (2, 3, 2))
    assert result.shape == (2, 3)
    assert result.subspace is scalar_vector
    np.testing.assert_allclose(
        result.kernel,
        expected,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


def test_extensor_and_context_are_immutable_and_extensor_owns_its_buffer():
    algebra = Algebra("x+y+")
    spaces = algebra.subspace
    context = NumpyContext(algebra)
    coefficients = np.asarray([1.0, 2.0])
    value = context.extensor(spaces.even(), coefficients)

    coefficients[0] = 99
    assert value.kernel.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="read-only"):
        value.kernel[0] = 10
    with pytest.raises(AttributeError, match="immutable"):
        value.gatype = GAType((spaces.even(),))
    with pytest.raises(AttributeError, match="immutable"):
        context.dtype = np.dtype(np.float32)

    operator = algebra.operator.identity(spaces.even())
    with pytest.raises(AttributeError):
        operator.kernel = SymbolicKernel.identity(2)


def test_collection_rules_preserve_or_erase_stub_traits_conservatively():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    rotor_type = algebra.gatype.rotor()
    plain_even = algebra.gatype.even()
    rotors = context.extensor(rotor_type, [[1.0, 0.0], [0.0, 1.0]])

    assert rotors[0].gatype is rotor_type
    assert rotors.reshape(1, 2).gatype is rotor_type
    assert rotors.broadcast_to((3, 2)).gatype is rotor_type
    assert Extensor.stack((rotors, rotors)).gatype is rotor_type
    assert Extensor.concatenate((rotors, rotors)).gatype is rotor_type

    assert rotors.sum().gatype is plain_even
    assert rotors.mean().gatype is plain_even
    assert rotors.at[0].set([2.0, 0.0]).gatype is plain_even
    assert (-rotors).gatype is rotor_type


def test_warm_extension_results_bypass_public_construction(monkeypatch):
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    scalar = context.multivector.scalar([4.0])
    rotor = context.multivector.even([3.0, 4.0])
    vector = algebra.subspace.vector()
    matrix = context.extensor(algebra.gatype((vector, vector)), [[2.0, 1.0], [0.0, 3.0]])

    def evaluate():
        return (
            scalar.norm(), scalar.square_root(), scalar.inverse_square_root(),
            scalar.inverse(), scalar.exp(), scalar.log(),
            rotor.normalized(), matrix.inverse(),
        )

    expected = evaluate()

    def fail(*args, **kwargs):
        raise AssertionError("an internal result used public construction")

    monkeypatch.setattr(Extensor, "__init__", fail)
    monkeypatch.setattr(NumpyContext, "prepare_kernel", fail)
    for actual, reference in zip(evaluate(), expected):
        assert actual.gatype is reference.gatype
        np.testing.assert_allclose(actual.kernel, reference.kernel)

    buffer = np.asarray([2.0])
    monkeypatch.setattr(np, "sqrt", lambda values: buffer)
    assert scalar.square_root()._kernel is buffer


def test_map_kernel_passes_raw_storage_and_wraps_the_returned_buffer(monkeypatch):
    context = NumpyContext(Algebra("x+y+"))
    value = context.multivector.vector([1, 2])
    result_buffer = np.asarray([3.0, 4.0])

    def transform(kernel):
        assert kernel is value._kernel
        return result_buffer

    def fail(*args, **kwargs):
        raise AssertionError("map_kernel used public construction")

    monkeypatch.setattr(Extensor, "__init__", fail)
    monkeypatch.setattr(NumpyContext, "prepare_kernel", fail)
    result = value.map_kernel(transform)
    assert result.context is context
    assert result._kernel is result_buffer
    assert not result_buffer.flags.writeable


def test_numpy_context_rejects_lossy_numeric_kind_changes():
    algebra = Algebra("x+y+")
    spaces = algebra.subspace
    value = NumpyContext(algebra).extensor(spaces.even(), [1, 2])

    np.testing.assert_allclose(
        (2 * value).kernel,
        [2, 4],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        (value + value).kernel,
        [2, 4],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    with pytest.raises(TypeError, match="without changing numeric kind"):
        value * (1 + 2j)
    with pytest.raises(TypeError, match="real or complex floating dtype"):
        NumpyContext(algebra, dtype=int)
