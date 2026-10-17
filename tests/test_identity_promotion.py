"""A nullary GAType in linear arithmetic is the identity map on it; a bare SubSpace has no orientation and is
refused."""

from fractions import Fraction

import numpy as np
import pytest

from numga import NumpyContext
from numga.algebras import PGA3D


def test_type_promotes_to_identity_beside_a_map():
    ga = PGA3D
    mv = NumpyContext(ga).multivector
    B = ga.gatype.bivector()
    L = mv.bivector(np.array([0.3, -0.2, 0.5, 0.1, 0.7, -0.4]))
    ad = L.commutator(B)
    complement = B - ad
    assert complement.gatype == ga.gatype((B, B))
    np.testing.assert_allclose(complement.kernel, np.eye(6) - ad.kernel)
    np.testing.assert_allclose((ad + B).kernel, np.eye(6) + ad.kernel)
    np.testing.assert_allclose((B + ad).kernel, np.eye(6) + ad.kernel)
    with pytest.raises(TypeError):
        ad + ga.subspace.bivector()
    with pytest.raises(TypeError):
        ga.subspace.bivector() - ad


def test_scaled_and_negated_types_are_scaled_identities():
    ga = PGA3D
    V = ga.gatype.vector()
    np.testing.assert_allclose((Fraction(5, 2) * V).kernel.materialize(), 2.5 * np.eye(4))
    np.testing.assert_allclose((V * 3).kernel.materialize(), 3 * np.eye(4))
    np.testing.assert_allclose((-V).kernel.materialize(), -np.eye(4))


def test_nullary_gatype_promotes_and_positive_arity_is_rejected():
    ga = PGA3D
    mv = NumpyContext(ga).multivector
    Vector = ga.gatype.vector()
    x = mv.vector(np.array([1.0, 2.0, 3.0, 4.0]))
    dyad = x * (x | Vector)
    np.testing.assert_allclose((Vector - dyad).kernel, np.eye(4) - dyad.kernel)
    try:
        dyad.gatype + dyad
    except TypeError:
        pass
    else:
        raise AssertionError("a unary GAType must not promote to an identity")


def test_type_dual_is_the_open_dual_map():
    ga = PGA3D
    mv = NumpyContext(ga).multivector
    B = ga.gatype.bivector()
    L = mv.bivector(np.array([0.3, -0.2, 0.5, 0.1, 0.7, -0.4]))
    np.testing.assert_allclose(B.dual()(L).kernel, L.dual().kernel)
    np.testing.assert_allclose(B.dual().dual_inverse()(L).cast(B).kernel, L.cast(B).kernel)
