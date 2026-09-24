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
