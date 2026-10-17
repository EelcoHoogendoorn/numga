"""Simplex quadrature integrates second moments and inertia with the same mass scaling."""

import numpy as np
import pytest

from numga import Algebra, NumpyContext
from numga.algebras import ProjectiveMultivectorFactory
from examples.mechanics import simplex
from examples.mechanics.inertia import second_moment


@pytest.mark.parametrize("signature, count", [
    ("w0x+", 2),
    ("w0x+y+", 3),
    ("w0x+y+z+", 3),
    ("w0x+y+z+", 4),
    ("w0x+y+z+v+", 5),
])
def test_second_moment_matches_exact_barycentric_integral(signature, count):
    ga = Algebra(signature, multivector_factory=ProjectiveMultivectorFactory)
    mv = NumpyContext(ga).multivector
    Plane = ga.gatype.vector()
    rng = np.random.default_rng(4)
    corners = mv.point(rng.normal(size=(2, count, ga.dimension - 1)))

    # Uniform barycentric coordinates have diagonal second moments twice the off-diagonal ones.
    total = corners.sum(axis=-1)
    expected = ((corners * (Plane & corners)).sum(axis=-1) + total * (Plane & total)) / (count * (count + 1))
    moment = simplex.simplex_second_moment_lumped(corners)
    np.testing.assert_allclose(moment.kernel, expected.kernel, rtol=1e-11, atol=1e-11)

    # The homogeneous entries contain total mass and the centroid, not just spatial spread.
    np.testing.assert_allclose((mv.w & moment(mv.w)).kernel, 1.0, atol=1e-11)
    np.testing.assert_allclose(moment(mv.w).normalized().kernel,
                               total.normalized().kernel, rtol=1e-11, atol=1e-11)
    mass = 3.7
    weighted = simplex.simplex_second_moment(mass * simplex.simplex_inertia_weights(count), corners)
    np.testing.assert_allclose(weighted.kernel, (mass * moment).kernel, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("signature", ["w0x+y+z+", "w0x+y+z+v+"])
def test_second_moment_agrees_with_inertia_recovery(signature):
    ga = Algebra(signature, multivector_factory=ProjectiveMultivectorFactory)
    mv = NumpyContext(ga).multivector
    rng = np.random.default_rng(7)
    corners = mv.point(rng.normal(size=(2, ga.dimension, ga.dimension - 1)))
    moment = simplex.simplex_second_moment_lumped(corners)
    inertia = simplex.simplex_inertia_lumped(corners)
    recovered = second_moment(inertia)
    np.testing.assert_allclose(recovered.kernel, moment.kernel, rtol=1e-11, atol=1e-11)
