import numpy as np

from numga import Algebra, Extensor, NumpyContext


def test_squaring_an_open_vector_map_keeps_its_inputs() -> None:
    algebra = Algebra("x+y+")
    vector = algebra.operator.identity(algebra.subspace.vector())
    squared = vector.squared()

    assert squared.gatype is vector.gatype.squared
    assert squared.output_subspace is algebra.subspace.scalar()
    assert squared.arity == 2
    assert not squared.gatype.reduces_to_scalar(3)
    assert vector.symmetric_reverse_product().gatype is vector.gatype.symmetric_reverse
    assert not vector.gatype.symmetric_reverse.reduces_to_scalar(3)

    x = NumpyContext(algebra).multivector.vector([3, 4])
    np.testing.assert_allclose(squared(x, x).kernel, [25])


def test_geometric_product_expression_stages_by_operand_kind() -> None:
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    V = algebra.subspace.vector()
    assert V.gatype is algebra.gatype(V)
    assert not V.gatype.traits

    x = context.extensor(V, [1, 2])
    y = context.extensor(V, [3, 4])

    binary = V * V
    ternary = V * V * V
    left_multiply = x * V
    right_multiply = V * y
    product = x * y

    values = (binary, ternary, left_multiply, right_multiply, product)
    assert all(type(value) is Extensor for value in values)
    assert tuple(value.arity for value in values) == (2, 3, 1, 1, 0)

    assert binary.context is algebra.exact
    assert ternary.context is algebra.exact
    assert left_multiply.context is context
    assert right_multiply.context is context
    assert product.context is context
    np.testing.assert_allclose(
        product.kernel,
        [11, -2],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )

    equivalent_products = (
        binary(x, y),
        left_multiply(y),
        right_multiply(x),
    )
    for actual in equivalent_products:
        assert actual.gatype == product.gatype
        np.testing.assert_allclose(
            actual.kernel,
            product.kernel,
            rtol=1e-14,
            atol=1e-14,
            equal_nan=False,
        )

    ternary_result = ternary(x, y, x)
    sequential_result = product * x
    assert ternary_result.gatype == sequential_result.gatype
    np.testing.assert_allclose(
        ternary_result.kernel,
        sequential_result.kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


def test_a_call_with_fewer_operands_binds_the_leading_slots():
    import numpy as np
    from numga import NumpyContext
    from numga.algebras import PGA3D

    mv = NumpyContext(PGA3D).multivector
    Point = PGA3D.gatype.antivector()
    join = Point & Point                                     # Line <- (Point, Point)
    origin = mv.antivector([0.0, 0.0, 0.0, 1.0])
    target = mv.antivector([1.0, 2.0, 3.0, 1.0])

    ray = join(origin)                                       # Line <- Point

    assert ray.arity == 1
    assert ray.gatype == join.bind(origin).gatype
    np.testing.assert_allclose(ray(target).kernel, join(origin, target).kernel)
