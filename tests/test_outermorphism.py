"""The outermorphism extends a map through the exterior product of its own space."""

import numpy as np
import pytest

from numga import Algebra, NumpyContext
from numga.algebras import PGA3D, STA


def random_map(context, gatype, rng, batch=()):
    n = len(gatype.output_subspace.masks)
    return context.extensor(context.algebra.gatype((gatype, gatype)), rng.normal(size=batch + (n, n)))


@pytest.mark.parametrize("ga", [PGA3D, STA, Algebra("x+y+z+w+e-")], ids=str)
def test_vector_maps_extend_through_the_wedge(ga):
    context, rng = NumpyContext(ga), np.random.default_rng(0)
    mv, Vector, Bivector = context.multivector, ga.gatype.vector(), ga.gatype.bivector()
    Pseudoscalar = ga.gatype(ga.subspace.pseudoscalar())
    t, s = random_map(context, Vector, rng), random_map(context, Vector, rng)
    a, b = mv.vector(rng.normal(size=ga.dimension)), mv.vector(rng.normal(size=ga.dimension))
    I = mv(ga.subspace.pseudoscalar(), [1.0])

    np.testing.assert_allclose(t.outermorphism(Vector)(a).kernel, t(a).kernel, atol=1e-12)
    np.testing.assert_allclose(t.outermorphism(Bivector)(a ^ b).kernel, (t(a) ^ t(b)).kernel, atol=1e-12)
    np.testing.assert_allclose(t.outermorphism(Pseudoscalar)(I).kernel, (t.det() * I).kernel, atol=1e-12)
    np.testing.assert_allclose(
        t(s).outermorphism(Bivector).kernel, t.outermorphism(Bivector)(s.outermorphism(Bivector)).kernel, atol=1e-12)


def test_point_maps_extend_through_the_join():
    context, rng = NumpyContext(PGA3D), np.random.default_rng(1)
    mv = context.multivector
    Point, Line, Plane, Scalar = PGA3D.gatype.antivector(), PGA3D.gatype.bivector(), PGA3D.gatype.vector(), PGA3D.gatype.scalar()
    T = random_map(context, Point, rng)
    p, q, r = (mv.antivector(rng.normal(size=4)) for _ in range(3))

    np.testing.assert_allclose(T.outermorphism(Line)(p & q).kernel, (T(p) & T(q)).kernel, atol=1e-12)
    np.testing.assert_allclose(T.outermorphism(Plane)(p & q & r).kernel, (T(p) & T(q) & T(r)).kernel, atol=1e-12)
    np.testing.assert_allclose(T.outermorphism(Scalar)(mv.scalar([1.0])).kernel, T.det().kernel, atol=1e-12)


def test_batched_maps_extend_per_item():
    context, rng = NumpyContext(PGA3D), np.random.default_rng(2)
    Vector, Bivector = PGA3D.gatype.vector(), PGA3D.gatype.bivector()
    t = random_map(context, Vector, rng, batch=(5,))
    lifted = t.outermorphism(Bivector)
    assert lifted.shape == (5,)
    np.testing.assert_allclose(lifted[3].kernel, t[3].outermorphism(Bivector).kernel, atol=1e-12)


def test_a_type_the_products_do_not_span_is_refused():
    context, rng = NumpyContext(PGA3D), np.random.default_rng(3)
    Vector, Direction = PGA3D.gatype.vector(), PGA3D.gatype.from_blades("x y z")
    with pytest.raises(TypeError):
        random_map(context, Vector, rng).outermorphism(PGA3D.gatype(PGA3D.subspace.from_grades([2, 4])))
    with pytest.raises(TypeError):
        random_map(context, Direction, rng).outermorphism(PGA3D.gatype.bivector())      # spans only xy, xz, yz
