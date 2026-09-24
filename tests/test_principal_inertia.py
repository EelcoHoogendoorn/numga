"""A four-number inertia is a drop-in Extensor: dense storage is an implementation detail."""

import numpy as np

from numga import Extensor, NumpyContext
from numga.algebras import PGA3D
from numga.extensions.optimized import PrincipalInertiaPGA3, principal_inertia_pga3


def test_principal_inertia_behaves_as_its_dense_map():
    context, rng = NumpyContext(PGA3D), np.random.default_rng(0)
    mv = context.multivector
    mass = mv.scalar(rng.uniform(1, 2, size=(50, 1)))
    moments = mv(PGA3D.subspace("yz zx xy"), rng.uniform(1, 2, size=(50, 3)))
    inertia = principal_inertia_pga3(mass, moments)
    dense = Extensor(context, inertia.gatype, inertia.kernel.copy())
    rate = mv.bivector(rng.normal(size=(50, 6)))
    motor = mv.bivector(rng.normal(size=(50, 6))).exp()

    assert isinstance(inertia, Extensor) and isinstance(inertia.inverse(), PrincipalInertiaPGA3)
    np.testing.assert_allclose(inertia(rate).kernel, dense(rate).kernel)
    np.testing.assert_allclose(inertia.inverse()(rate).kernel, dense.inverse()(rate).kernel, atol=1e-12)
    np.testing.assert_allclose(inertia.inverse()(inertia(rate)).kernel, rate.kernel, atol=1e-12)
    np.testing.assert_allclose((motor >> inertia).kernel, (motor >> dense).kernel, atol=1e-12)
    np.testing.assert_allclose(inertia.solve(inertia(rate)).kernel, rate.kernel, atol=1e-12)
    assert type(inertia[3]) is Extensor and type(inertia * 2) is Extensor
    assert inertia.shape == (50,) and inertia.__slots__ == ("upper", "lower")      # the dense map is never kept


def test_principal_inertia_matches_six_unit_masses():
    """Unit masses at the six unit axis points: mass 6, moment 4 about each axis."""
    mv = NumpyContext(PGA3D).multivector
    points = mv.antivector([[1, 0, 0, 1], [-1, 0, 0, 1], [0, 1, 0, 1], [0, -1, 0, 1], [0, 0, 1, 1], [0, 0, -1, 1]])
    cloud = (points & points.commutator(PGA3D.gatype.bivector())).sum(axis=0)
    inertia = principal_inertia_pga3(mv.scalar([6.0]), mv(PGA3D.subspace("yz zx xy"), [4.0, 4.0, 4.0]))
    np.testing.assert_allclose(inertia.kernel, cloud.kernel)
