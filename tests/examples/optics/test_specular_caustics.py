"""Tests for coarse-grid specular caustics, Jacobian fold contours, and multi-path refraction solves."""

import matplotlib.pyplot as plt
import pytest

jax = pytest.importorskip("jax")

from examples.optics.specular_caustics import render, scenarios

n_pixels, blur_pixels, exposure = 16, 0.95, 0.3


def test_caustics_folds_and_paths():
    @jax.jit
    def coarse_scene():
        positions, ray_area, wavevectors = scenarios.stratified_water_rays(4)
        landings = next(scenarios.caustic_states(positions, wavevectors, 1))
        return ray_area, landings, scenarios.fold_chart(wavevectors, 8), scenarios.specular_paths()

    ray_area, landings, (chart_landings, area_ratios), (
        surface_profile, fan_surface, fan_landing, path_surface, receiver_target, path_residuals
    ) = coarse_scene()

    assert (path_residuals.scalar_norm_squared().abs() < 1e-12).all()
    frame = next(render.caustics_animation(scenarios.receiver_halfwidth, ray_area, n_pixels, blur_pixels, exposure,
                                           [landings]))
    assert frame.shape == (n_pixels, n_pixels, 3)

    folds_fig = render.folds_figure(landings, chart_landings, area_ratios, scenarios.receiver_halfwidth, ray_area,
                                    n_pixels, blur_pixels, exposure)
    folds_fig.canvas.draw()
    plt.close(folds_fig)

    paths_fig = render.paths_figure(surface_profile, fan_surface, fan_landing, path_surface, receiver_target,
                                    scenarios.incident_height)
    paths_fig.canvas.draw()
    plt.close(paths_fig)
