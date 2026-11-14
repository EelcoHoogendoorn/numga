"""Coarse-sphere coverage exercises one magnetohydrodynamic step, energy evaluation, sphere shading, and the energy chart."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.fluids.mhd_sphere import core, render, scenarios


def test_coarse_sphere_step_and_rendering():
    topology, vertices = scenarios.sphere(1)
    surface = core.surface(topology, vertices)
    initial = scenarios.initial_state(surface, vertices)
    states = stack(tuple(core.evolution(surface, initial, scenarios.viscosity, scenarios.resistivity, scenarios.dt, 1)))
    kinetic, magnetic = scenarios.energies(surface, states)
    frames = list(render.animation(topology, vertices, scenarios.frame_channels(surface, states[:1])))
    figure = render.energy_figure(np.array([0.0, scenarios.dt]), kinetic, magnetic)
    plt.close(figure)
    assert len(frames) == 1
