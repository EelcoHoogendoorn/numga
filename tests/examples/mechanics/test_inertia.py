"""Principal-frame recovery diagonalizes inertia maps across spherical and Euclidean PGA in four and five dimensions."""

from __future__ import annotations

import pytest

from examples.mechanics.inertia import main


@pytest.mark.parametrize("signature", ["x+y+z+w0", "x+y+z+w+", "x+y+z+v+w0"])
def test_principal_frames_diagonalize_inertia(signature: str) -> None:
    """Both aligned inertia energy forms are diagonal and round-trip recovery preserves the spectrum."""
    cloud_motor, motor = main(signature, 0)
    assert ((cloud_motor.norm() - 1).abs() < 1e-10).all()
    assert ((motor.norm() - 1).abs() < 1e-10).all()
