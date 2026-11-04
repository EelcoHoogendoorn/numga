"""Types narrowed by parts of their algebra, the complementary type, and the type of an expression: a part is an algebra
whose generators the type's algebra shares by name and square."""

import pytest

from numga import Algebra, GATypeError, NumpyContext
from numga.algebras import PGA3D, STA, VGA3D
from numga.gatype import Orientation

space = Algebra("x+y+z+")
time = Algebra("t-")


def test_within_a_part_keeps_the_blades_of_its_generators():
    planes = STA.gatype.vector().derive.within(space)
    assert planes.output_subspace.same_support(STA.subspace("x y z"))
    # The scalar lies within every part: the rotations of space are its scalar and its bivectors.
    rotations = STA.gatype.even().derive.within(space)
    assert rotations.output_subspace.same_support(STA.subspace("1 yz zx xy"))
    # Parts match by name and square, whatever their factories: the algebra of space within its projective algebra.
    assert PGA3D.gatype.vector().derive.within(VGA3D) == PGA3D.gatype.vector().derive.nondegenerate
    with pytest.raises(GATypeError):
        STA.gatype.vector().derive.within(Algebra("t+"))


def test_touching_and_without_split_by_parts():
    amounts = Algebra("v0m0s0")
    plane = Algebra("x+y+")
    ga = plane * amounts
    drives = ga.gatype.bivector().derive.touching(plane, amounts)
    assert drives.output_subspace.same_support(ga.subspace("xv yv xm ym xs ys"))
    # In spacetime the bivectors through time and those of space alone make up every bivector.
    temporal = STA.gatype.bivector().derive.touching(time)
    spatial = STA.gatype.bivector().derive.without(time)
    assert spatial == STA.gatype.bivector().derive.within(space)
    assert set(temporal.output_subspace.masks) | set(spatial.output_subspace.masks) == set(STA.subspace.bivector().masks)
    assert not set(temporal.output_subspace.masks) & set(spatial.output_subspace.masks)


def test_the_dual_type_is_the_type_of_a_dual():
    mv = NumpyContext(STA).multivector
    planes = STA.gatype.vector().derive.within(space)
    directions = planes.derive.dual
    assert mv(planes, [1.0, 2.0, 3.0]).dual().gatype == directions
    assert directions.orientation is Orientation.intrinsic
    # The directions of space in spacetime are the points through time.
    assert directions == STA.gatype.antivector().derive.containing("t")
    assert directions.derive.dual == planes


def test_an_expression_gives_the_type_of_its_values():
    ga = Algebra("x+y+z+X+Y+Z+")
    mv = NumpyContext(ga).multivector
    first = ga.gatype.even().derive.within(Algebra("x+y+z+"))
    second = ga.gatype.even().derive.within(Algebra("X+Y+Z+"))
    spinors = ga.gatype(first * second)
    # A product of values has that type, with the facts the product certifies about them on top.
    product = mv(first, [1.0, 2.0, 3.0, 4.0]) * mv(second, [5.0, 6.0, 7.0, 8.0])
    assert product.gatype.derive.structural == spinors
    states = STA.gatype(STA.gatype.antibivector() + STA.gatype.antiscalar())
    assert states.orientation is Orientation.intrinsic
    assert states.output_subspace.same_support(STA.subspace.bivector() + STA.subspace.pseudoscalar())
    with pytest.raises(ValueError):
        STA.gatype(first * second)
