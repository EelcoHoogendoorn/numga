"""The inertia example diagonalizes inertia in spherical and Euclidean PGA, in four and five dimensions."""

from __future__ import annotations

import pytest

from examples.mechanics.inertia import main


@pytest.mark.parametrize("signature", ["x+y+z+w0", "x+y+z+w+", "x+y+z+v+w0"])
def test_principal_frames_diagonalize_inertia(signature):
    """main asserts both aligned energy forms are diagonal and the round trip recovers its spectrum."""
    main(signature, 0)
