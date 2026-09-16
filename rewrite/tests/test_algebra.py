from __future__ import annotations

import numpy as np
import pytest

from numga.algebra import Algebra, AlgebraDescription, BladeProduct, OrientedBlade


def multiply_terms(
    algebra: Algebra,
    left: tuple[int, int],
    right: tuple[int, int],
) -> tuple[int, int]:
    left_coefficient, left_blade = left
    right_coefficient, right_blade = right
    product = algebra.geometric_product(left_blade, right_blade)
    return (
        left_coefficient * right_coefficient * product.coefficient,
        product.blade,
    )


def test_compact_description_and_structured_constructors() -> None:
    description = AlgebraDescription.parse("x+y-z0")

    assert description.basis_names == ("x", "y", "z")
    assert description.signature == (1, -1, 0)
    assert description.dimension == 3
    assert description.pqr == (1, 1, 1)
    assert description.signature_string == "+-0"
    assert description.to_compact_string() == "x+y-z0"
    assert Algebra(description) == Algebra("x+y-z0")
    assert Algebra((1, 1, 1)).signature == (1, -1, 0)
    assert Algebra.from_signature((1, 0), ("space", "e0")).basis_names == (
        "space",
        "e0",
    )


@pytest.mark.parametrize(
    "specification",
    ["x", "+", "x++", "x+leftover"],
)
def test_compact_description_rejects_malformed_input(specification: str) -> None:
    with pytest.raises(ValueError):
        Algebra(specification)


def test_description_validates_metric_and_names() -> None:
    with pytest.raises(ValueError, match="same length"):
        AlgebraDescription(("x",), (1, -1))
    with pytest.raises(ValueError, match="unique"):
        AlgebraDescription(("x", "x"), (1, 1))
    with pytest.raises(ValueError, match=r"-1, 0, or \+1"):
        AlgebraDescription(("x",), (2,))
    with pytest.raises(ValueError, match="non-negative"):
        Algebra.from_pqr(1, -1, 0)


def test_algebra_exposes_canonical_masks_and_metric_masks() -> None:
    algebra = Algebra("x+y-z0")

    assert algebra.dimension == 3
    assert algebra.blade_count == 8
    assert algebra.blade_masks == range(8)
    assert algebra.basis_vector_masks == (0b001, 0b010, 0b100)
    assert algebra.pseudoscalar_mask == 0b111
    assert algebra.positive_mask == 0b001
    assert algebra.negative_mask == 0b010
    assert algebra.degenerate_mask == 0b100
    assert algebra.blade_dtype == np.dtype(np.uint8)


def test_spelled_blades_preserve_orientation_before_subspaces_exist() -> None:
    algebra = Algebra("x+y+z+")

    assert algebra.parse_blade("1") == OrientedBlade(0, 1)
    assert algebra.parse_blade("xy") == OrientedBlade(0b011, 1)
    assert algebra.parse_blade("yx") == OrientedBlade(0b011, -1)
    assert algebra.parse_blade("zx") == OrientedBlade(0b101, -1)
    assert algebra.parse_blade("zyx") == OrientedBlade(0b111, -1)
    assert algebra.blade_name(0b101) == "xz"

    with pytest.raises(ValueError, match="repeat"):
        algebra.parse_blade("xx")
    with pytest.raises(ValueError, match="unknown"):
        algebra.parse_blade("q")


def test_multicharacter_blades_have_unambiguous_spellings() -> None:
    algebra = Algebra.from_signature((1, 1), ("time", "space"))

    assert algebra.parse_blade(("space", "time")) == OrientedBlade(0b11, -1)
    assert algebra.parse_blade("space^time") == OrientedBlade(0b11, -1)
    assert algebra.blade_name(0b11) == "time^space"

    with pytest.raises(ValueError, match="multi-character"):
        algebra.parse_blade("timespace")


def test_euclidean_plane_geometric_product() -> None:
    algebra = Algebra("x+y+")
    scalar, x, y, xy = 0b00, 0b01, 0b10, 0b11

    assert algebra.geometric_product(scalar, xy) == BladeProduct(1, xy)
    assert algebra.geometric_product(x, x) == BladeProduct(1, scalar)
    assert algebra.geometric_product(y, y) == BladeProduct(1, scalar)
    assert algebra.geometric_product(x, y) == BladeProduct(1, xy)
    assert algebra.geometric_product(y, x) == BladeProduct(-1, xy)
    assert algebra.geometric_product(xy, xy) == BladeProduct(-1, scalar)
    assert algebra.pseudoscalar_squared == -1


def test_metric_sign_and_degeneracy_enter_only_shared_generators() -> None:
    algebra = Algebra("x-y+w0")
    x, y, w = algebra.basis_vector_masks

    assert algebra.geometric_product(x, x) == BladeProduct(-1, 0)
    assert algebra.geometric_product(y, y) == BladeProduct(1, 0)
    assert algebra.geometric_product(w, w) == BladeProduct(0, 0)
    assert algebra.geometric_product(w, x) == BladeProduct(-1, w | x)
    assert algebra.geometric_product(x, w) == BladeProduct(1, w | x)


@pytest.mark.parametrize("specification", ["a+b+c+", "a-b+c0"])
def test_geometric_product_is_associative_exhaustively(specification: str) -> None:
    algebra = Algebra(specification)
    for left in algebra.blade_masks:
        for middle in algebra.blade_masks:
            for right in algebra.blade_masks:
                lhs = multiply_terms(
                    algebra,
                    tuple(algebra.geometric_product(left, middle)),
                    (1, right),
                )
                rhs = multiply_terms(
                    algebra,
                    (1, left),
                    tuple(algebra.geometric_product(middle, right)),
                )
                assert lhs == rhs


def test_grade_complement_and_involutions() -> None:
    algebra = Algebra("a+b+c+d+")

    assert [algebra.grade(mask) for mask in algebra.blade_masks] == [
        0,
        1,
        1,
        2,
        1,
        2,
        2,
        3,
        1,
        2,
        2,
        3,
        2,
        3,
        3,
        4,
    ]
    assert [algebra.reverse_sign((1 << grade) - 1) for grade in range(5)] == [
        1,
        1,
        -1,
        -1,
        1,
    ]
    assert [algebra.involute_sign((1 << grade) - 1) for grade in range(5)] == [
        1,
        -1,
        1,
        -1,
        1,
    ]
    assert [algebra.conjugate_sign((1 << grade) - 1) for grade in range(5)] == [
        1,
        -1,
        -1,
        1,
        1,
    ]
    assert all(
        algebra.complement(algebra.complement(mask)) == mask
        for mask in algebra.blade_masks
    )


def test_bulk_product_table_agrees_with_scalar_api_and_is_immutable() -> None:
    algebra = Algebra("x+y+z+")
    left = (0, 1, 0b110)
    right = (0b010, 0b111)

    table = algebra.geometric_product_table(left, right)

    assert table.shape == (3, 2)
    for i, left_blade in enumerate(left):
        for j, right_blade in enumerate(right):
            product = algebra.geometric_product(left_blade, right_blade)
            assert table.blades[i, j] == product.blade
            assert table.coefficients[i, j] == product.coefficient
    with pytest.raises(ValueError):
        table.blades[0, 0] = 99
    with pytest.raises(ValueError):
        table.coefficients.flags.writeable = True


def test_product_algebra_requires_disjoint_generator_names() -> None:
    product = Algebra("x+") * Algebra("y-z0")
    assert product == Algebra("x+y-z0")

    with pytest.raises(ValueError, match="duplicate"):
        Algebra("x+") * Algebra("x-")


@pytest.mark.parametrize("mask", [-1, 4])
def test_blade_operations_reject_masks_outside_the_algebra(mask: int) -> None:
    algebra = Algebra("x+y+")

    with pytest.raises(ValueError, match="outside"):
        algebra.grade(mask)
    with pytest.raises(ValueError, match="outside"):
        algebra.geometric_product(0, mask)


def test_vectorized_blade_operations_and_compatibility_api() -> None:
    algebra = Algebra("x+y-z0")
    blades = np.arange(algebra.blade_count, dtype=algebra.blade_dtype)

    # Vectorized grade, complement, involutions
    grades = algebra.grade(blades)
    assert isinstance(grades, np.ndarray)
    assert np.array_equal(grades, [algebra.grade(int(b)) for b in blades])

    complements = algebra.complement(blades)
    assert isinstance(complements, np.ndarray)
    assert np.array_equal(complements, [algebra.complement(int(b)) for b in blades])

    involutes = algebra.involute(blades)
    assert isinstance(involutes, np.ndarray)
    assert np.array_equal(involutes, [algebra.involute_sign(int(b)) for b in blades])

    reverses = algebra.reverse(blades)
    assert isinstance(reverses, np.ndarray)
    assert np.array_equal(reverses, [algebra.reverse_sign(int(b)) for b in blades])

    conjugates = algebra.conjugate(blades)
    assert isinstance(conjugates, np.ndarray)
    assert np.array_equal(conjugates, [algebra.conjugate_sign(int(b)) for b in blades])

    # Vectorized bit_dot, cayley, product
    dot = algebra.bit_dot(blades[:, None], blades[None, :])
    assert dot.shape == (8, 8)
    assert dot[0b011, 0b110] == 1

    cayley, swaps = algebra.cayley(blades[:, None], blades[None, :])
    assert cayley.shape == (8, 8)
    assert swaps.shape == (8, 8)

    prod_blades, prod_signs = algebra.product(blades, blades)
    assert prod_blades.shape == (8, 8)
    assert prod_signs.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            expected = algebra.geometric_product(i, j)
            assert prod_blades[i, j] == expected.blade
            assert prod_signs[i, j] == expected.coefficient

    # 1.x compatibility aliases
    assert algebra.negatives == algebra.negative_mask
    assert algebra.positives == algebra.positive_mask
    assert algebra.zeros == algebra.degenerate_mask
    assert algebra.blade_nbytes == algebra.blade_dtype.itemsize
    assert algebra.pseudo_scalar_squared == algebra.pseudoscalar_squared
    assert algebra.n_dimensions == algebra.dimension
    assert algebra.n_blades == algebra.blade_count
    assert algebra.n_grades == algebra.dimension + 1
    assert len(algebra) == algebra.blade_count

