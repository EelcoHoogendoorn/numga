"""The values of a subalgebra in an algebra containing its generators: the algebra of space in its projective
algebra, by kind, with products, joins, pairings and maps carried over."""

import numpy as np
import pytest

from numga import Algebra, GATypeError, NumpyContext, stack
from numga.algebras import PGA3D, STA, VGA2D, VGA3D

space = NumpyContext(VGA3D).multivector
projective = NumpyContext(PGA3D).multivector
origin = projective.w.dual()

blade_names = ["x", "y", "z", "yz", "zx", "xy", "xyz"]


def assert_same(actual, expected):
    assert actual.gatype.algebra is expected.gatype.algebra
    assert actual.gatype.orientations == expected.gatype.orientations
    np.testing.assert_array_equal(actual.cast(expected.output_subspace).kernel, expected.kernel)
    np.testing.assert_array_equal(expected.cast(actual.output_subspace).kernel, actual.kernel)


@pytest.mark.parametrize("name", blade_names)
def test_an_extrinsic_blade_keeps_its_name(name):
    assert_same(projective.context.include_from_algebra(getattr(space, name)), getattr(projective, name))


@pytest.mark.parametrize("name", blade_names)
def test_an_intrinsic_blade_keeps_its_complement(name):
    assert_same(projective.context.include_from_algebra(getattr(space, name).dual()), getattr(projective, name).dual())


def test_scalars_and_volumes_by_kind():
    assert_same(projective.context.include_from_algebra(space.scalar([2.0])), projective.scalar([2.0]))
    # The unit volume of space is the projective unit volume; space's intrinsic scalar one, the complement of its
    # volume, is the complement of that volume's blade here, an intrinsic vector along the degenerate generator.
    assert_same(projective.context.include_from_algebra(space.antiscalar([1.0])), projective.antiscalar([1.0]))
    assert_same(projective.context.include_from_algebra(space.antipseudoscalar([1.0])), projective.xyz.dual())


def test_a_point_of_space_is_a_direction_and_with_the_origin_a_point():
    coordinates = [[1.0, 2.0, 3.0], [-0.5, 0.0, 4.0]]
    offsets = space.vector(coordinates).dual()                                   # [points] Antivector-
    assert_same(projective.context.include_from_algebra(offsets), projective.direction(coordinates))
    assert_same(origin + projective.context.include_from_algebra(offsets), projective.point(coordinates))
    # An observer's slice of spacetime has the same spatial points, independent of their event times.
    spacetime = NumpyContext(STA).multivector
    events = spacetime.vector([[1, 2, 3, -7], [-0.5, 0, 4, 9]]).dual()
    time = -spacetime.t.dual()
    slice_of_space = spacetime.t ^ (STA.gatype.antivector() & time)
    assert_same(projective.context.include_from_algebra(slice_of_space(events)), projective.direction(coordinates))
    with pytest.raises(GATypeError):
        projective.context.include_from_algebra(events)


def test_linear_over_batches():
    rng = np.random.default_rng(0)
    first = space.vector(rng.normal(size=(5, 3))).dual()
    second = space.vector(rng.normal(size=(5, 3))).dual()
    weights = rng.normal(size=5)
    combined = projective.context.include_from_algebra(first * weights + second * 2.0)
    parts = projective.context.include_from_algebra(first) * weights + projective.context.include_from_algebra(second) * 2.0
    np.testing.assert_allclose(combined.kernel, parts.kernel, atol=1e-14)


def test_products_joins_meets_and_pairings_carry_over():
    rng = np.random.default_rng(1)
    planes = space.vector(rng.normal(size=(4, 3)))                               # [4] Vector+
    points = space.vector(rng.normal(size=(4, 3))).dual()                        # [4] Antivector-
    rotor = (space.bivector(rng.normal(size=3)) * 0.4).exp()                     # [] Rotor+
    included = projective.context.include_from_algebra
    for actual, expected in [
        (included(planes * planes[::-1]), included(planes) * included(planes[::-1])),
        (included(planes ^ planes[::-1]), included(planes) ^ included(planes[::-1])),
        (included(points & points[::-1]), included(points) & included(points[::-1])),
        (included(planes & points), included(planes) & included(points)),
        (included(planes * points), included(planes) * included(points)),
        (included(rotor >> points), included(rotor) >> included(points)),
        (included(rotor >> planes), included(rotor) >> included(planes)),
    ]:
        assert actual.gatype.orientations == expected.gatype.orientations
        np.testing.assert_allclose(actual.cast(expected.output_subspace).kernel, expected.kernel, atol=1e-13)
    # The inclusion preserves the geometric product, so a rotor of space is a rotor here, about the origin.
    assert included(rotor).gatype.traits == rotor.gatype.traits
    np.testing.assert_allclose((included(rotor) >> origin).kernel, origin.kernel, atol=1e-14)


def test_a_map_carries_each_axis_by_its_kind():
    Point = VGA3D.gatype.antivector()
    axes = stack((space.x, space.y, space.z))                                    # [3] Vector+
    # A polarity of space, from points to planes, with a shear so that it is no sum of axis dyads alone.
    polarity = (axes * np.array([2.0, 3.0, 5.0]) * (axes & Point)).sum(axis=-1) + 0.5 * space.y * (space.x & Point)
    points = space.vector([[1.0, 0.5, -2.0], [0.0, 1.0, 1.0]]).dual()           # [2] Antivector-
    included = projective.context.include_from_algebra(polarity)                              # [] Vector+ <- Antivector-
    assert included.gatype.orientations == polarity.gatype.orientations
    np.testing.assert_allclose(included(projective.context.include_from_algebra(points)).kernel,
                               projective.context.include_from_algebra(polarity(points)).kernel, atol=1e-14)
    # Where the map has an inverse, the inclusion of the inverse is the inverse of the inclusion.
    np.testing.assert_allclose(projective.context.include_from_algebra(polarity.inverse()).kernel, included.inverse().kernel,
                               atol=1e-14)
    # The same map can live inside spacetime; each of its axes still belongs to space.
    spacetime = NumpyContext(STA).multivector
    via_spacetime = projective.context.include_from_algebra(spacetime.context.include_from_algebra(polarity))
    assert_same(via_spacetime, included)


def test_a_dual_quadric_of_space_lifts_to_the_solid_about_the_origin():
    Plane = PGA3D.gatype.vector()
    semi_axes = np.array([1.0, 2.0, 3.0])
    axes = stack((space.x, space.y, space.z))                                    # [3] Vector+
    # An ellipsoid's dual quadric in space: each plane through the origin to the point its tangent plane touches.
    dual = (axes.dual() * semi_axes**2 * (VGA3D.gatype.vector() | axes)).sum(axis=-1)   # [] Antivector- <- Vector+
    quadric = (projective.context.include_from_algebra(dual) - origin * (Plane & origin)).inverse()  # [] Plane+ <- Point-
    on_surface = projective.point(np.diag(semi_axes))                            # [3] Point-
    np.testing.assert_allclose((quadric(on_surface) & on_surface).to_array(), 0, atol=1e-14)
    assert (quadric(origin) & origin).to_array() < 0


def test_into_the_same_algebra_and_from_the_plane_into_space():
    rotor = (space.xy * 0.3).exp() * (space.yz * -0.2).exp()
    assert_same(space.context.include_from_algebra(rotor), rotor)
    plane = NumpyContext(VGA2D).multivector
    offset = plane.vector([1.0, 2.0]).dual()                                     # [] Antivector-
    assert_same(projective.context.include_from_algebra(offset), projective.direction([1.0, 2.0, 0.0]))


def test_the_projective_plane_lies_in_projective_space_at_its_origin():
    from numga.algebras import PGA2D

    plane = NumpyContext(PGA2D).multivector
    starts = plane.point([[1.0, 2.0], [0.5, -1.0]])                              # [2] Antivector-
    ends = plane.point([[3.0, 0.0], [1.0, 1.0]])                                 # [2] Antivector-
    assert_same(projective.context.include_from_algebra(starts), projective.point([[1.0, 2.0, 0.0], [0.5, -1.0, 0.0]]))
    assert_same(projective.context.include_from_algebra(starts & ends),
                projective.context.include_from_algebra(starts) & projective.context.include_from_algebra(ends))


def test_exact_values_stay_exact():
    exact = PGA3D.exact.multivector
    assert_same(PGA3D.exact.include_from_algebra(VGA3D.exact.multivector.zx.dual()), exact.zx.dual())
    with pytest.raises(TypeError):
        exact.context.include_from_algebra(space.x)


@pytest.mark.parametrize("larger", [Algebra("x+y+"), Algebra("x+y+z-"), Algebra("x+y+u+")])
def test_refused_where_a_generator_is_missing_or_squares_otherwise(larger):
    with pytest.raises(GATypeError):
        NumpyContext(larger).include_from_algebra(space.vector([1.0, 2.0, 3.0]))


def test_jax_agrees_with_numpy_under_jit():
    jax = pytest.importorskip("jax")
    from numga.backend.jax import JaxContext

    coordinates = np.array([[1.0, 2.0, 3.0], [-0.5, 0.0, 4.0]])
    jax_space = JaxContext(VGA3D).multivector
    jax_projective = JaxContext(PGA3D).multivector
    traced = jax.jit(lambda offsets: jax_projective.context.include_from_algebra(offsets))(jax_space.vector(coordinates).dual())
    assert isinstance(traced.kernel, jax.Array)
    expected = projective.context.include_from_algebra(space.vector(coordinates).dual())
    assert traced.gatype is expected.gatype
    np.testing.assert_array_equal(np.asarray(traced.kernel), expected.kernel)
    spacetime = JaxContext(STA).multivector
    slice_of_space = spacetime.t ^ (STA.gatype.antivector() & -spacetime.t.dual())
    events = spacetime.vector([[1, 2, 3, -7], [-0.5, 0, 4, 9]]).dual()
    sliced = jax.jit(lambda event: jax_projective.context.include_from_algebra(slice_of_space(event)))(events)
    np.testing.assert_array_equal(np.asarray(sliced.kernel), expected.kernel)
