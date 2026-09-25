"""Typed motion maps, consistent intersections/joins, and mirrored normals."""

import numpy as np
import pytest

from numga import Algebra, CoefficientOrthogonal, Extensor, NumpyContext, ReverseProductOne, Versor
from numga.algebras import PGA3D
from numga.extensions.inverse import inverse_orthogonal


def test_plane_line_and_point_maps_are_distinct_but_describe_the_same_motion():
    mv = NumpyContext(PGA3D).multivector
    planes = PGA3D.subspace.vector()
    lines = PGA3D.subspace.bivector()
    points = PGA3D.subspace.antivector()

    # A bivector exponential supplies a unit motor, with rotation and translation.
    # Unit matters: a scaled sandwich would not preserve wedge products exactly.
    motor = mv.bivector([0.2, -0.1, 0.3, 0.5, -0.25, 0.1]).exp()
    plane_map = motor.sandwich(planes)
    line_map = motor.sandwich(lines)
    point_map = motor.sandwich(points)

    assert plane_map.axes == (planes, planes)
    assert line_map.axes == (lines, lines)
    assert point_map.axes == (points, points)
    assert plane_map.gatype is not point_map.gatype

    # Both are 4x4 matrices, but a point is not a plane. Shape alone is not a type.
    assert plane_map.kernel.shape == point_map.kernel.shape == (4, 4)
    # Translation acts differently on plane equations and point coordinates:
    # these are genuinely different matrices, not just differently typed copies.
    assert not np.allclose(plane_map.kernel, point_map.kernel)
    with pytest.raises(ValueError, match="cannot bind output axis"):
        plane_map(point_map)
    with pytest.raises(ValueError, match="cannot bind output axis"):
        line_map(plane_map)

    # Applying the motion twice composes within each space, in exactly the same
    # order as multiplying the motors. The resulting maps remain unary.
    following = mv.bivector([-0.1, 0.3, 0, 0.1, 0, 0.2]).exp()
    combined = following * motor
    for space, first in ((planes, plane_map), (lines, line_map), (points, point_map)):
        composed = following.sandwich(space)(first)
        direct = combined.sandwich(space)
        assert composed.axes == (space, space)
        np.testing.assert_allclose(composed.kernel, direct.kernel, atol=1e-9)

    # Concrete geometry: intersect the planes x=1, y=2, z=3, then move the
    # intersection; or move the planes first and intersect them afterwards.
    a, b, c = mv.vector([[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3]])
    intersection = a.wedge(b).wedge(c)
    moved_intersection = plane_map(a).wedge(plane_map(b)).wedge(plane_map(c))
    np.testing.assert_allclose(
        point_map(intersection).kernel, moved_intersection.kernel, atol=1e-9,
    )

    # Stronger: compare the complete open operators, not just sampled geometry.
    # Two planes intersect in a line; a line and a plane intersect in a point.
    move_intersection = line_map(planes.wedge(planes))
    intersect_moved = plane_map.wedge(plane_map)
    assert move_intersection.axes == intersect_moved.axes == (lines, planes, planes)
    np.testing.assert_allclose(move_intersection.kernel, intersect_moved.kernel, atol=1e-9)

    move_intersection = point_map(lines.wedge(planes))
    intersect_moved = line_map.wedge(plane_map)
    assert move_intersection.axes == intersect_moved.axes == (points, lines, planes)
    np.testing.assert_allclose(move_intersection.kernel, intersect_moved.kernel, atol=1e-9)

    # Dually, two points join to a line; a line and a point join to a plane.
    move_join = line_map(points.regressive(points))
    join_moved = point_map.regressive(point_map)
    assert move_join.axes == join_moved.axes == (lines, points, points)
    np.testing.assert_allclose(move_join.kernel, join_moved.kernel, atol=1e-9)

    move_join = plane_map(lines.regressive(points))
    join_moved = line_map.regressive(point_map)
    assert move_join.axes == join_moved.axes == (planes, lines, points)
    np.testing.assert_allclose(move_join.kernel, join_moved.kernel, atol=1e-9)


def test_wedged_normals_follow_a_mirror_without_normal_specific_rules():
    # In Euclidean 3-space an oriented surface normal is represented by its
    # tangent bivector, not converted into another grade-1 vector.
    algebra = Algebra("x+y+z+")
    mv = NumpyContext(algebra).multivector
    directions = algebra.subspace.vector()
    normals = algebra.subspace.bivector()
    mirror = mv.vector([1, 0, 0]).normalized()  # Mirror in the yz plane.

    # Hyperplane reflection is m involute(X) reverse(m). Sandwich alone is
    # untwisted conjugation. Use the same grade-involution rule for every grade.
    direction_map = mirror.sandwich(directions).involute()
    normal_map = mirror.sandwich(normals).involute()

    u = mv.vector([1, 2, 0])
    v = mv.vector([0, 1, 3])
    normal = u.wedge(v)
    mirrored_normal = direction_map(u).wedge(direction_map(v))

    # Verify an actual reflection, not the half-turn that conjugation alone
    # gives on Euclidean vectors. The oriented area flips xy and xz, not yz.
    np.testing.assert_allclose(direction_map(u).kernel, [-1, 2, 0])
    np.testing.assert_allclose(direction_map(v).kernel, [0, 1, 3])
    np.testing.assert_allclose(mirrored_normal.kernel, [-1, -3, 6])
    np.testing.assert_allclose(normal_map(normal).kernel, mirrored_normal.kernel)

    # Both maps are 3x3, but feeding a normal into the direction map is wrong.
    assert direction_map.axes == (directions, directions)
    assert normal_map.axes == (normals, normals)
    assert not np.allclose(direction_map.kernel, normal_map.kernel)
    with pytest.raises(ValueError, match="cannot bind output axis"):
        direction_map(normal)

    # The entire wedge operator commutes with reflection, not just this sample.
    reflect_wedge = normal_map(directions.wedge(directions))
    wedge_reflected = direction_map.wedge(direction_map)
    assert reflect_wedge.axes == wedge_reflected.axes == (normals, directions, directions)
    np.testing.assert_allclose(reflect_wedge.kernel, wedge_reflected.kernel)

    # The reflection keeps the useful promises, not just the right coefficients:
    # unit inputs stay unit, and inverse dispatch still selects transpose.
    for mapping, unit in ((direction_map, u.normalized()), (normal_map, normal.normalized())):
        reflected = mapping(unit)
        assert reflected.gatype.entails((ReverseProductOne, Versor))
        assert mapping.gatype.entails(CoefficientOrthogonal)
        assert Extensor.inverse._dispatch.resolve(mapping.gatype) is inverse_orthogonal
        inverse = mapping.inverse()
        np.testing.assert_allclose(inverse.kernel, mapping.kernel.T)
        np.testing.assert_allclose(inverse(reflected).kernel, unit.kernel, atol=1e-14)


def test_sandwich_shift_operators_are_the_sandwich_from_either_side():
    mv = NumpyContext(PGA3D).multivector
    points = mv.antivector([1, 2, 3, 1])
    lines = PGA3D.subspace.bivector()

    motor = mv.bivector([0.2, -0.1, 0.3, 0.5, -0.25, 0.1]).exp()

    # >> is the sandwich, << the sandwich from the other side; for a unit motor it undoes >>
    moved = motor >> points
    assert moved.gatype == (motor.sandwich(points)).gatype
    np.testing.assert_allclose(moved.kernel, motor.sandwich(points).kernel)

    recovered = motor << moved
    assert recovered.gatype == (motor.reverse().sandwich(moved)).gatype
    np.testing.assert_allclose(recovered.kernel, points.kernel, atol=1e-12)

    # SubSpace hole operand works with >> and <<
    pullback = motor << lines
    assert pullback.axes == (lines, lines)
    np.testing.assert_allclose(pullback.kernel, motor.reverse().sandwich(lines).kernel)
