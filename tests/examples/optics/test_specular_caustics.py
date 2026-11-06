"""Moving-water caustics at a few rays: the first frame of the caustics, drawn, the folds over a coarse chart and the
three paths to one receiver point."""

import pytest

jax = pytest.importorskip("jax")

from examples.optics.specular_caustics import render, scenarios

n_pixels, blur_pixels, exposure = 16, 0.95, 0.3


def test_the_caustics_folds_and_paths_run():
    # One compiled call: run eagerly, every small operation would compile on its own.
    @jax.jit
    def scenes():
        positions, ray_area, wavevectors = scenarios.water(n_rays=4)
        landings = next(scenarios.caustics(positions, wavevectors))
        return ray_area, landings, scenarios.folds(wavevectors, n_derivative=4), scenarios.three_paths()

    ray_area, landings, _, _ = scenes()
    next(render.animate([landings], scenarios.receiver_halfwidth, ray_area, n_pixels, blur_pixels, exposure))
