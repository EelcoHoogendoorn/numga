"""A bare SubSpace or nullary GAType in linear arithmetic is the identity map on it."""

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D


def test_subspace_promotes_to_identity_beside_a_map():
    ga = PGA3D
    mv = NumpyContext(ga).multivector
    B = ga.subspace.bivector()
    L = mv.bivector(np.array([0.3, -0.2, 0.5, 0.1, 0.7, -0.4]))
    ad = L.commutator(B)
    complement = B - ad
    assert complement.gatype.axes == (B, B)
    np.testing.assert_allclose(complement.kernel, np.eye(6) - ad.kernel)
    np.testing.assert_allclose((ad + B).kernel, np.eye(6) + ad.kernel)
    np.testing.assert_allclose((B + ad).kernel, np.eye(6) + ad.kernel)


def test_scaled_and_negated_subspaces_are_scaled_identities():
    ga = PGA3D
    V = ga.subspace.vector()
    np.testing.assert_allclose((2.5 * V).kernel.materialize(), 2.5 * np.eye(4))
    np.testing.assert_allclose((V * 3).kernel.materialize(), 3 * np.eye(4))
    np.testing.assert_allclose((-V).kernel.materialize(), -np.eye(4))


def test_subspace_plus_subspace_is_still_the_union():
    ga = PGA3D
    both = ga.subspace.vector() + ga.subspace.bivector()
    assert len(both) == 10


def test_gatype_promotes_like_its_subspace_and_rejects_positive_arity():
    ga = PGA3D
    mv = NumpyContext(ga).multivector
    Vector = ga.gatype.vector()
    x = mv.vector(np.array([1.0, 2.0, 3.0, 4.0]))
    dyad = x * (x | ga.subspace.vector())
    np.testing.assert_allclose((Vector - dyad).kernel, np.eye(4) - dyad.kernel)
    try:
        dyad.gatype + dyad
    except TypeError:
        pass
    else:
        raise AssertionError("a unary GAType must not promote to an identity")


def test_subspace_dual_is_the_open_dual_map():
    ga = PGA3D
    mv = NumpyContext(ga).multivector
    B = ga.subspace.bivector()
    L = mv.bivector(np.array([0.3, -0.2, 0.5, 0.1, 0.7, -0.4]))
    np.testing.assert_allclose(B.dual()(L).kernel, L.dual().kernel)
    np.testing.assert_allclose(B.dual().dual_inverse()(L).cast(B).kernel, L.cast(B).kernel)
