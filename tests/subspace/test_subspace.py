from __future__ import annotations

import pytest

from numga.subspace import SubSpace, SubSpaceFactory


class StubAlgebra:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.basis_names = tuple("xyzw"[:dimension])
        self.blade_masks = range(1 << dimension)

    def grade(self, mask: int) -> int:
        if mask < 0 or mask >= 1 << self.dimension:
            raise ValueError(mask)
        return mask.bit_count()


def test_subspace_is_an_immutable_structural_value_with_canonical_order() -> None:
    algebra = StubAlgebra()
    left = SubSpace(algebra, [12, 0, 2, 9, 1, 3])
    right = SubSpace(algebra, (mask for mask in [3, 1, 9, 0, 12, 2]))

    assert left == right
    assert left is not right
    assert left.masks == (0, 1, 2, 3, 9, 12)
    assert hash(left) == hash(right)
    with pytest.raises(AttributeError, match="immutable"):
        left.masks = ()


def test_grade_major_mask_order_is_explicit_in_four_dimensions() -> None:
    algebra = StubAlgebra()
    grade_two = SubSpace(algebra, [12, 10, 9, 6, 5, 3])

    # Integer-mask order is the deterministic within-grade policy for the
    # unsigned foundation.  The later explicit-layout phase may preserve a
    # user-specified order through a distinct construction path.
    assert grade_two.masks == (3, 5, 6, 9, 10, 12)


@pytest.mark.parametrize("masks", [[1, 1], [-1], [16], [True], ["x"]])
def test_invalid_masks_are_rejected(masks: list[object]) -> None:
    algebra = StubAlgebra()
    with pytest.raises((TypeError, ValueError)):
        SubSpace(algebra, masks)


def test_support_relations_include_algebra_identity() -> None:
    algebra = StubAlgebra()
    other_algebra = StubAlgebra()
    vector = SubSpace(algebra, [1, 2, 4, 8])
    plane = SubSpace(algebra, [1, 2])
    reordered_input = SubSpace(algebra, [8, 4, 2, 1])
    foreign_vector = SubSpace(other_algebra, [1, 2, 4, 8])

    assert vector.same_support(reordered_input)
    assert vector.support_key == reordered_input.support_key
    assert plane.support_is_subset_of(vector)
    assert not vector.support_is_subset_of(plane)
    assert not vector.same_support(foreign_vector)
    assert vector.support_key != foreign_vector.support_key


def test_factory_is_per_algebra_and_builds_required_spaces() -> None:
    algebra = StubAlgebra()
    factory = SubSpaceFactory(algebra)

    assert factory.from_masks([2, 1]) is factory.from_masks([1, 2])
    assert factory.empty().masks == ()
    assert factory.scalar().masks == (0,)
    assert factory.vector().masks == (1, 2, 4, 8)
    assert factory.k_vector(2).masks == (3, 5, 6, 9, 10, 12)
    assert factory.full().masks == (
        0,
        1,
        2,
        4,
        8,
        3,
        5,
        6,
        9,
        10,
        12,
        7,
        11,
        13,
        14,
        15,
    )
    assert factory.even().masks == (0, 3, 5, 6, 9, 10, 12, 15)
    assert factory.multivector() is factory.full()
    assert factory.even_grade() is factory.even()
    assert factory.from_masks([2, 1]) == SubSpace(algebra, [1, 2])


def test_factory_rejects_an_invalid_grade() -> None:
    factory = SubSpaceFactory(StubAlgebra())
    with pytest.raises(ValueError, match="outside the valid range"):
        factory.k_vector(5)


def test_factory_construction_does_not_eagerly_expand_the_full_algebra() -> None:
    algebra = StubAlgebra(64)
    factory = SubSpaceFactory(algebra)

    assert factory.scalar().masks == (0,)
    assert factory.vector().masks == tuple(1 << index for index in range(64))


def test_algebra_owns_the_canonical_subspace_factory() -> None:
    from numga.algebra import Algebra

    algebra = Algebra("x+y+z+")

    assert algebra.subspace.vector() is algebra.subspace.from_masks((1, 2, 4))
    assert algebra.subspace.flyweight_count == 1


def test_subspace_addition_is_canonical_support_union() -> None:
    from numga.algebra import Algebra

    algebra = Algebra("x+y+z+")
    scalar = algebra.subspace.scalar()
    vector = algebra.subspace.vector()
    scalar_vector = algebra.subspace.from_masks((0, 1, 2, 4))

    assert scalar + vector is scalar_vector
    assert vector + scalar is scalar_vector
    assert vector + vector is vector

    foreign = Algebra("x+y+z+").subspace.vector()
    with pytest.raises(ValueError, match="different algebras"):
        _ = vector + foreign


def test_subspace_factory_attribute_access_constructs_blades() -> None:
    from numga.algebra import Algebra

    algebra = Algebra("x+y+z+")
    spaces = algebra.subspace

    assert spaces.x == spaces.from_blades("x")
    assert spaces.xy == spaces.from_blades("xy")
    assert spaces.xy_z == spaces.from_blades("xy z")

    with pytest.raises(AttributeError):
        _ = spaces.w
    with pytest.raises(AttributeError):
        _ = spaces.unknown_attr
