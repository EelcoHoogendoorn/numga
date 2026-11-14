"""Coverage tests for earthquake focal mechanisms, P- and S-wave radiation, and scene rendering."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.mechanics.earthquake import core, render, scenarios


def coordinates(directions: core.Point) -> np.ndarray:
    """Pairing `directions` (`[...] Point-`) with the north, east and down coordinate planes reads out `[..., 3]` components."""
    mv = core.mv
    return (stack((mv.x, mv.y, mv.z)) & directions[..., None]).to_array()


def test_fault_frame_matches_aki_and_richards():
    """Fault normal and slip directions produced by `fault_frame` match the Aki and Richards strike, dip and rake formulas."""
    strikes, dips, rakes = np.deg2rad([0.0, 75.0, 210.0]), np.deg2rad([90.0, 35.0, 60.0]), np.deg2rad([0.0, 120.0, -45.0])
    frames = core.fault_frame(strikes, dips, rakes)
    normals = coordinates(frames >> -core.down)
    slips = coordinates(frames >> core.north)
    expected_normals = np.stack([-np.sin(dips) * np.sin(strikes), np.sin(dips) * np.cos(strikes), -np.cos(dips)], axis=-1)
    expected_slips = np.stack([np.cos(rakes) * np.cos(strikes) + np.sin(rakes) * np.cos(dips) * np.sin(strikes),
                               np.cos(rakes) * np.sin(strikes) - np.sin(rakes) * np.cos(dips) * np.cos(strikes),
                               -np.sin(rakes) * np.sin(dips)], axis=-1)
    np.testing.assert_allclose(normals, expected_normals, atol=1e-12)
    np.testing.assert_allclose(slips, expected_slips, atol=1e-12)


def test_p_wave_is_twice_the_product_of_components():
    """Unit double-couple P-wave push equals twice the normal-by-slip directional product, and S-wave motion stays transverse."""
    frame = core.fault_frame(np.deg2rad(40.0), np.deg2rad(55.0), np.deg2rad(20.0))
    normal, slip = frame >> -core.down, frame >> core.north
    fault, auxiliary = frame >> core.ground, frame >> core.prime_vertical
    grid = core.sphere_directions(np.linspace(0.0, np.pi, 5), np.linspace(0.0, 2 * np.pi, 7))
    push, shear = core.radiation(core.double_couple(fault, auxiliary, 1.0), grid)
    assert ((push - 2 * (grid.dot(normal) * grid.dot(slip))).abs() < 1e-12).all()
    assert (shear.dot(grid).abs() < 1e-12).all()


def test_the_scenes_draw():
    import matplotlib.pyplot as plt

    _, _, moment = scenarios.fault_mechanisms(np.deg2rad(scenarios.strikes[:1]), np.deg2rad(scenarios.dips[:1]), np.deg2rad(scenarios.rakes[:1]))
    small_hemisphere = core.sphere_directions(np.linspace(0.0, np.pi / 2, 4), np.linspace(0.0, 2 * np.pi, 7))
    _, axes = core.principal_axes(moment)
    push, _ = core.radiation(moment[:, None, None], small_hemisphere)
    fig = render.draw_mechanisms(small_hemisphere, push, axes)
    plt.close(fig)
    list(render.animate_sweep(scenarios.hemisphere(), scenarios.rake_sweep(1)))
