"""Coverage test runs one gradient-descent step on the camera reprojection misfit."""

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("jax")

from examples.estimation.camera_fit import fitted_generator, mv, reprojection_misfit, screen_images


def test_one_descent_step_reduces_reprojection_misfit() -> None:
    """One JAX gradient-descent step reduces the reprojection misfit on a four-point scene."""
    world_points = mv.point(jnp.asarray(np.random.default_rng(0).normal(size=(4, 3))))
    true_generator = mv.yz * 0.1 - mv.zx * 0.2 + mv.zw * 2.5
    target_images = screen_images(world_points, true_generator)
    initial_generator = true_generator + (mv.yz + mv.xw + mv.zw) * 0.2
    updated_generator = fitted_generator(world_points, target_images, initial_generator, 1, 0.25)
    assert (reprojection_misfit(world_points, target_images, updated_generator)
            < reprojection_misfit(world_points, target_images, initial_generator)).all()
