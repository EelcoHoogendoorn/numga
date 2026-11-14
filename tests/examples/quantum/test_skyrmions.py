"""Magnetic skyrmions: one step of the spins on the smallest lattice, its energy and charge, and the lattice and the
frames drawn, alone and side by side."""

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.skyrmions import core, render, scenarios


def test_one_step_runs_and_draws():
    topology, vertices, bonds, periods, carriers = scenarios.film(n_cells=4)
    areas, gradients, cells = core.triangles(topology, bonds)
    spins = scenarios.noise(topology, scenarios.n_smoothing, scenarios.seed)   # Spin-[vertices]
    # Two frames, one step apart, in the crystal's field, driven by the current and kicked by the thermal field.
    n_frames = 2
    applied = core.film * np.full(n_frames, scenarios.crystal_field)          # [frames] Field+
    drifts = scenarios.current_direction * np.full(n_frames, scenarios.drift_speed)   # [frames] Point-
    states = list(core.dynamics(topology, carriers, areas, gradients, cells, spins, applied, drifts,
                                np.full(n_frames, scenarios.dt), scenarios.damping, scenarios.sweep_temperature, 1,
                                scenarios.seed))
    core.energy(topology, carriers, applied[0], states[-1])
    core.charges(topology, states[-1])
    next(render.animate(topology, vertices, bonds, periods, states))
    next(render.animate_together(topology, vertices, bonds, periods, [(states[-1],) * 3]))
    plt.close(render.draw_lattice(topology, vertices, bonds, periods))
