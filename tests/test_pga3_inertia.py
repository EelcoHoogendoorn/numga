"""A reusable inertia map for six unit masses, in the preferred PGA3 layout."""

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D


def test_inertia_from_points_is_independent_of_their_storage_layout():
    spaces = PGA3D.subspace
    mv = NumpyContext(PGA3D).multivector

    # Point coordinates read directly as (x, y, z, 1) in this default layout.
    points = mv.antivector([
        [1, 0, 0, 1], [-1, 0, 0, 1],
        [0, 1, 0, 1], [0, -1, 0, 1],
        [0, 0, 1, 1], [0, 0, -1, 1],
    ])

    per_point = points.regressive(points.commutator(spaces.bivector()))
    inertia = per_point.sum(axis=0)

    # Angular components first, translation components second.
    rates = mv.bivector([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [1, 2, 3, 4, 5, 6],
    ])
    momenta = inertia(rates)
    energy = rates.regressive(momenta) * 0.5

    assert per_point.shape == (6,)
    assert inertia.shape == ()
    assert inertia.axes == (spaces.bivector(), spaces.bivector())
    # Total mass 6; moment about each coordinate axis is 4. Momentum uses
    # the dual pairing, so the angular and linear blocks exchange places.
    np.testing.assert_allclose(momenta.kernel, [
        [0, 0, 0, 4, 0, 0],
        [6, 0, 0, 0, 0, 0],
        [24, 30, 36, 4, 8, 12],
    ])
    np.testing.assert_allclose(energy.kernel, [[2], [3], [259]])

    # Store the same points using different blade order AND orientation.
    other_points = points.select_subspace(spaces("xyz xyw xzw yzw"))
    other_inertia = other_points.regressive(
        other_points.commutator(spaces.bivector())
    ).sum(axis=0)

    assert other_points.gatype != points.gatype
    assert other_inertia.gatype is inertia.gatype
    np.testing.assert_allclose(other_inertia.kernel, inertia.kernel)
    np.testing.assert_allclose(other_inertia(rates).kernel, momenta.kernel)
