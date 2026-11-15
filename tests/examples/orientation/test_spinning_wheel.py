"""Tests angular momentum, kinetic energy, torque, power, reflection symmetry, and rendering for spinning wheels in the plane and in space."""

import matplotlib.pyplot as plt
import numpy as np

from examples import instantiate
from examples.orientation.spinning_wheel import render, scenarios


def test_wheels_in_the_plane_and_in_space():
    for scene in (scenarios.wheel_in_the_plane, scenarios.wheel_in_space):
        points, pushes, rates, momenta, energies, torques, powers = scene()
        core = instantiate(scenarios.CORE, rates.algebra)

        assert ((energies - scenarios.masses.sum() * scenarios.speed ** 2 / 2).abs() < 1e-9).all()
        assert ((powers - scenarios.push_strength * scenarios.speed).abs() < 1e-9).all()
        assert (momenta[1] - (core.mv.x >> momenta[0])).antinorm() < 1e-9
        assert (torques[1] - (core.mv.x >> torques[0])).antinorm() < 1e-9

        state = next(core.turning(points, pushes, rates, np.zeros(1)))
        figure = render.draw_wheels(points, rates, scenarios.masses, state)
        figure.canvas.draw()
        plt.close(figure)
