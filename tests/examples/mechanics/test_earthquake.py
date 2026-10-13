"""Earthquake beachballs: the fault rotor matches the angles' formulas, the P wave is the moment's form
along each direction, and the scenes pass their checks and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.earthquake import core, render, scenarios


def test_fault_frame_matches_aki_and_richards():
    """The normal and slip of the fault rotor are the textbook formulas in strike, dip and rake."""
    strikes, dips, rakes = np.deg2rad([0.0, 75.0, 210.0]), np.deg2rad([90.0, 35.0, 60.0]), np.deg2rad([0.0, 120.0, -45.0])
    frames = core.fault_frame(strikes, dips, rakes)
    normals = render.coordinates(frames >> -core.down)
    slips = render.coordinates(frames >> core.north)
    expected_normals = np.stack([-np.sin(dips) * np.sin(strikes), np.sin(dips) * np.cos(strikes), -np.cos(dips)], axis=-1)
    expected_slips = np.stack([np.cos(rakes) * np.cos(strikes) + np.sin(rakes) * np.cos(dips) * np.sin(strikes),
                               np.cos(rakes) * np.sin(strikes) - np.sin(rakes) * np.cos(dips) * np.cos(strikes),
                               -np.sin(rakes) * np.sin(dips)], axis=-1)
    np.testing.assert_allclose(normals, expected_normals, atol=1e-13)
    np.testing.assert_allclose(slips, expected_slips, atol=1e-13)


def test_p_wave_is_twice_the_product_of_components():
    """Along a unit direction, the P wave of a unit double couple is twice its component along the
    normal times its component along the slip."""
    frames = core.fault_frame(np.deg2rad(40.0), np.deg2rad(55.0), np.deg2rad(20.0))
    normal, slip = frames >> -core.down, frames >> core.north
    grid = scenarios.sphere()
    push, shear = core.radiation(core.double_couple(normal, slip, 1.0), grid)
    np.testing.assert_allclose(push.to_array(), 2 * ((grid | normal) * (grid | slip)).to_array(), atol=1e-13)
    np.testing.assert_allclose((shear | grid).to_array(), 0.0, atol=1e-13)


def test_scenes_pass_their_checks_and_draw():
    figure = render.draw_mechanisms(scenarios.mechanisms(), scenarios.NAMES)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    frames = core.fault_frame(np.deg2rad(30.0), np.deg2rad(50.0), np.deg2rad([45.0]))
    balls = core.mechanism(frames, 1.0, scenarios.hemisphere())
    lobes = core.mechanism(frames, 1.0, scenarios.sphere())
    assert len(render.animate_sweep(balls, lobes)) == 1
    scenarios.sweep()
