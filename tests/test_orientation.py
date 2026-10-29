"""Intrinsic and extrinsic orientation: every type carries one, operations combine them, sums and bindings check
them, and acting with a versor gives the geometric motion of every element, reflections included.

A reflection reverses the orientation of space, so elements oriented within themselves (points, joins:
intrinsic) and elements oriented by what crosses them (planes, meets: extrinsic) transform with opposite
signs under it. The physics tests mirror a scene and ask for the physics of the mirrored scene; the remaining
tests state the type rules."""

import numpy as np
import pytest

from numga import GATypeError, Orientation
from numga import NumpyContext
from numga.algebras import PGA3D

mv = NumpyContext(PGA3D).multivector
Line = PGA3D.gatype.bivector()
OriginPlane = PGA3D.gatype(PGA3D.subspace("x y z"), orientation=Orientation.extrinsic)


def point(coords) -> object:
    return (mv(OriginPlane, np.atleast_2d(coords)) + mv.w).dual()


def direction(coords) -> object:
    return mv(OriginPlane, np.atleast_2d(coords)).dual()


def body(seed: int):
    """Mass points and the inertia map that takes a rate to the body's momentum."""
    rng = np.random.default_rng(seed)
    points = point(rng.normal(size=(5, 3)))
    masses = rng.uniform(0.5, 1.5, 5)
    return points, masses, points.regressive(points.commutator(Line) * masses).sum()


# A wall in general position, the plane n.x + d = 0.
NORMAL = np.array([1.0, 0.3, -0.2]) / np.linalg.norm([1.0, 0.3, -0.2])
OFFSET = 0.4
wall = mv.x * NORMAL[0] + mv.y * NORMAL[1] + mv.z * NORMAL[2] + mv.w * OFFSET


def reflect_point(coords: np.ndarray) -> np.ndarray:
    return coords - 2 * ((coords @ NORMAL) + OFFSET)[..., None] * NORMAL


def scalar(value) -> float:
    return float(np.asarray(value.to_array()).ravel()[0])


# --- physics of a mirrored scene ------------------------------------------------------------------------
def test_a_mirrored_spinning_body_keeps_its_kinetic_energy():
    points, masses, inertia = body(0)
    rate = mv.bivector(np.random.default_rng(1).normal(size=6))
    momentum = inertia(rate)
    energy = scalar((momentum & rate) * 0.5)
    assert energy > 0
    np.testing.assert_allclose(scalar(((wall >> momentum) & (wall >> rate)) * 0.5), energy, rtol=1e-12)


def test_a_mirrored_duck_stays_on_its_side_of_the_water():
    water = mv.z
    wall_through_water = mv.x - 0.5 * mv.w
    duck = point([2.0, 1.0, 0.3])
    before = scalar(duck & water)
    after = scalar((wall_through_water >> duck) & (wall_through_water >> water))
    assert np.sign(after) == np.sign(before)


def test_points_and_planes_mirror_to_their_mirror_images():
    rng = np.random.default_rng(2)
    coords = rng.normal(size=(4, 3))
    mirrored = wall >> point(coords)
    np.testing.assert_allclose(mirrored.kernel, point(reflect_point(coords)).kernel, atol=1e-12)
    # A plane through three points mirrors to the plane through their mirror images, facing the same way.
    plane = point(coords[0]) & point(coords[1]) & point(coords[2])
    image = point(reflect_point(coords[0])) & point(reflect_point(coords[1])) & point(reflect_point(coords[2]))
    np.testing.assert_allclose((wall >> plane).kernel, image.kernel, atol=1e-12)


def test_the_momentum_of_the_mirrored_body_is_the_mirrored_momentum():
    points, masses, inertia = body(3)
    rate = mv.bivector(np.random.default_rng(4).normal(size=6))
    mirrored_points = wall >> points
    mirrored_inertia = mirrored_points.regressive(mirrored_points.commutator(Line) * masses).sum()
    np.testing.assert_allclose(mirrored_inertia(wall >> rate).kernel, (wall >> inertia(rate)).kernel, atol=1e-12)
    # The mirrored rate moves every mirrored point with its mirrored velocity.
    np.testing.assert_allclose(
        mirrored_points.commutator(wall >> rate).kernel, (wall >> points.commutator(rate)).kernel, atol=1e-12)


def test_power_survives_a_mirror_and_pitch_changes_sign():
    _, _, inertia = body(5)
    rate = mv.bivector(np.random.default_rng(6).normal(size=6))
    forque = inertia(rate)
    np.testing.assert_allclose(scalar((wall >> forque) & (wall >> rate)), scalar(forque & rate), rtol=1e-12)
    np.testing.assert_allclose(scalar((wall >> rate) & (wall >> rate)), -scalar(rate & rate), rtol=1e-12)


def test_every_product_commutes_with_a_mirror():
    rng = np.random.default_rng(7)
    samples = {
        "plane": mv.vector(rng.normal(size=4)),
        "point": point(rng.normal(size=3)),
        "rate": mv.bivector(rng.normal(size=6)),
        "forque": point(rng.normal(size=3)) & direction(rng.normal(size=3)),
    }
    products = {
        "*": lambda a, b: a * b, "^": lambda a, b: a ^ b, "&": lambda a, b: a & b, "|": lambda a, b: a | b,
        "commutator": lambda a, b: a.commutator(b),
    }
    for name, product in products.items():
        for left in samples.values():
            for right in samples.values():
                value = product(left, right)
                np.testing.assert_allclose(
                    product(wall >> left, wall >> right).kernel, (wall >> value).kernel, atol=1e-10, err_msg=name)


# --- the type rules ---------------------------------------------------------------------------------------
def test_constructors_carry_their_orientation():
    from numga.gatype import Orientation

    gatype = PGA3D.gatype
    assert gatype.vector().orientation is Orientation.extrinsic
    assert gatype.bivector().orientation is Orientation.extrinsic
    assert gatype.motor().orientation is Orientation.extrinsic
    assert gatype.antivector().orientation is Orientation.intrinsic
    assert gatype.antibivector().orientation is Orientation.intrinsic
    # In 3D PGA a rate and a forque share their blades, and still differ in type.
    assert gatype.antibivector() != gatype.bivector()
    assert mv.antibivector(np.ones(6)).gatype.orientation is Orientation.intrinsic


def test_orientations_combine_like_signs():
    from numga.gatype import Orientation

    intrinsic, extrinsic = Orientation.intrinsic, Orientation.extrinsic
    plane, place = mv.vector([1.0, 0, 0, 0.5]), point([1.0, 2.0, 3.0])
    rate = mv.bivector(np.arange(1.0, 7.0))
    assert (plane ^ plane).gatype.orientation is extrinsic          # a meet of planes: an axis
    assert (place & place).gatype.orientation is intrinsic          # a join of points: a spear
    assert place.commutator(rate).gatype.orientation is intrinsic  # a point's velocity
    assert plane.dual().gatype.orientation is intrinsic
    assert (rate & rate).gatype.orientation is intrinsic            # pitch, a chirality
    momentum = place.regressive(place.commutator(rate))
    assert momentum.gatype.orientation is intrinsic
    assert (momentum & rate).gatype.orientation is extrinsic        # power, an ordinary number


def test_an_inertia_map_takes_rates_to_forques():
    from numga.gatype import Orientation

    _, _, inertia = body(8)
    assert inertia.gatype.orientations == (Orientation.intrinsic, Orientation.extrinsic)
    assert inertia.inverse().gatype.orientations == (Orientation.extrinsic, Orientation.intrinsic)


def test_a_rate_and_a_forque_do_not_add():
    _, _, inertia = body(9)
    rate = mv.bivector(np.ones(6))
    with pytest.raises(GATypeError, match="orientation"):
        rate + inertia(rate)


def test_a_forque_does_not_bind_where_a_rate_is_expected():
    _, _, inertia = body(10)
    rate = mv.bivector(np.ones(6))
    with pytest.raises(GATypeError, match="orientation"):
        inertia(inertia(rate))


def test_only_rates_exponentiate():
    _, _, inertia = body(11)
    forque = inertia(mv.bivector(np.ones(6) * 0.1))
    with pytest.raises(GATypeError):
        forque.exp()


def test_an_inertia_map_has_no_eigenvectors_of_its_own():
    _, _, inertia = body(12)
    with pytest.raises(GATypeError):
        inertia.eigh()
