"""Simplex quadrature integrates second-moment and inertia maps with consistent mass scaling."""

from __future__ import annotations

import numpy as np
import pytest

from numga import Algebra, NumpyContext
from numga.algebras import ProjectiveMultivectorFactory
from examples.mechanics import simplex
from examples.mechanics.inertia import second_moment


@pytest.mark.parametrize("signature, n_corners", [
    ("w0x+", 2),
    ("w0x+y+", 3),
    ("w0x+y+z+", 3),
    ("w0x+y+z+", 4),
    ("w0x+y+z+v+", 5),
])
def test_second_moment_matches_exact_barycentric_integral(signature: str, n_corners: int) -> None:
    """Exact lumped simplex weights reproduce the closed-form barycentric second moment, mass, and centroid."""
    ga = Algebra(signature, multivector_factory=ProjectiveMultivectorFactory)
    mv = NumpyContext(ga).multivector
    Plane = ga.gatype.vector()
    rng = np.random.default_rng(4)
    corners = mv.point(rng.normal(size=(2, n_corners, ga.dimension - 1)))            # [2, n_corners] Point-

    # Uniform barycentric coordinates have diagonal second moments equal to twice the off-diagonal ones.
    corner_sum = corners.sum(axis=-1)                                                # [2] Point-
    expected_moment = (
        (corners * (Plane & corners)).sum(axis=-1) + corner_sum * (Plane & corner_sum)
    ) / (n_corners * (n_corners + 1))                                                # [2] Point- <- Plane+
    moment = simplex.simplex_second_moment_lumped(corners)                           # [2] Point- <- Plane+
    n_planes = ga.dimension
    basis_planes = mv.vector(np.eye(n_planes, dtype=int))                            # [n_planes] Plane+
    residual = basis_planes[:, None, None] & (moment - expected_moment)[:, None](basis_planes[None, None, :])
    assert (residual.abs() < 1e-11).all()

    # The homogeneous entries carry total mass and the centroid alongside spatial spread.
    weight_plane = mv.w                                                              # [] Plane+
    assert (((weight_plane & moment(weight_plane)) - 1).abs() < 1e-11).all()
    centroid_residual = basis_planes[:, None] & (moment(weight_plane).normalized() - corner_sum.normalized())[None, :]
    assert (centroid_residual.abs() < 1e-11).all()

    mass = 3.7
    weighted_moment = simplex.simplex_second_moment(mass * simplex.simplex_inertia_weights(n_corners), corners)  # [2] Point- <- Plane+
    scaled_residual = basis_planes[:, None, None] & (weighted_moment - mass * moment)[:, None](basis_planes[None, None, :])
    assert (scaled_residual.abs() < 1e-11).all()


@pytest.mark.parametrize("signature", ["w0x+y+z+", "w0x+y+z+v+"])
def test_second_moment_agrees_with_inertia_recovery(signature: str) -> None:
    """Recovering the second moment from the lumped simplex inertia reproduces the direct lumped second moment."""
    ga = Algebra(signature, multivector_factory=ProjectiveMultivectorFactory)
    mv = NumpyContext(ga).multivector
    rng = np.random.default_rng(7)
    n_corners = ga.dimension
    corners = mv.point(rng.normal(size=(2, n_corners, ga.dimension - 1)))            # [2, n_corners] Point-
    moment = simplex.simplex_second_moment_lumped(corners)                           # [2] Point- <- Plane+
    inertia = simplex.simplex_inertia_lumped(corners)                                # [2] Antibivector- <- Bivector+
    recovered_moment = second_moment(inertia)                                        # [2] Point- <- Plane+
    basis_planes = mv.vector(np.eye(n_corners, dtype=int))                           # [n_corners] Plane+
    residual = basis_planes[:, None, None] & (recovered_moment - moment)[:, None](basis_planes[None, None, :])
    assert (residual.abs() < 1e-11).all()
