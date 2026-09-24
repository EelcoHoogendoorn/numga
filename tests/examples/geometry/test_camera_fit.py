"""Camera pose recovery by gradient descent on the image misfit, in JAX."""

import pytest

pytest.importorskip("jax")

from examples.geometry.camera_fit import main


def test_descent_recovers_the_pose():
    """The example's own checks assert the recovered pose matches the truth."""
    main()
