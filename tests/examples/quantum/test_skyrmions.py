"""Coverage test for magnetic skyrmions exercises one rotor step, energy, topological charge, and every figure and frame renderer on a minimal torus."""

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.skyrmions import core, render, scenarios
from numga import stack


def test_one_step_runs_and_draws():
    topology, vertices, bonds, periods, carriers = scenarios.torus_lattice(n_axis_sites=4)
    V, E, F = topology.n_elements
    spins = scenarios.smoothed_noise(topology, scenarios.n_smoothing, scenarios.seed)   # Spin-[V]
    n_frames = 2
    applied_fields = core.film * np.full(n_frames, scenarios.crystal_field)   # [n_frames] Field+
    drifts = scenarios.current_direction * np.full(n_frames, scenarios.drift_speed)   # [n_frames] Point-
    time_steps = np.full(n_frames, scenarios.time_step)
    states = list(core.dynamics(
        topology,
        bonds,
        carriers,
        spins,
        applied_fields,
        drifts,
        time_steps,
        scenarios.damping,
        scenarios.sweep_temperature,
        1,
        scenarios.seed,
    ))
    assert bool(((states[-1].norm().batch() - 1).abs() < 1e-10).all())
    frame_energies = stack([core.energy(topology, carriers, applied_fields[0], state) for state in states]) / V   # [n_frames] Scalar+
    frame_charges = stack([core.charges(topology, state).sites.sum() for state in states])   # [n_frames] Scalar+
    assert bool(frame_energies.isfinite().all())
    assert bool(((frame_charges * np.pi).sin().abs() < 1e-7).all())
    frame_changes = stack([(states[1] - states[0]).scalar_norm_squared().sites.mean().square_root()])   # [1] Scalar+

    next(render.spin_frames(topology, vertices, bonds, periods, states))
    next(render.comparison_frames(topology, vertices, bonds, periods, [(states[-1],) * 3]))
    plt.close(render.lattice_figure(topology, vertices, bonds, periods))
    plt.close(render.transient_figure(np.array([0.0, scenarios.time_step]), frame_charges, frame_energies, frame_changes))
