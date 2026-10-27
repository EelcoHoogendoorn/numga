"""Gradient descent on the reprojection misfit in JAX recovers the camera pose."""

import pytest

pytest.importorskip("jax")

from examples.estimation.camera_fit import main


def test_descent_recovers_the_pose():
    """The example's own checks assert the recovered pose matches the truth."""
    main()
