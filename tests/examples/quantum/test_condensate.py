"""Coverage test for the rotating Bose-Einstein condensate evolution, vortex counting, and rendering."""

import matplotlib.pyplot as plt

from examples.quantum.condensate import core, render, scenarios


def test_condensate_step_and_rendering():
    topology, vertices, dual_areas, states = scenarios.trap_evolution(2, 1, 1)
    evolved_states = list(states)
    settled_wave = evolved_states[-1]                                         # Wave+[V]
    atom_count = (dual_areas * settled_wave.scalar_norm_squared()).sites.sum()   # [] Scalar+
    assert (atom_count - 1).abs() < 1e-10

    frames = list(render.wave_animation(topology, vertices, scenarios.rotation, scenarios.step, evolved_states))
    assert len(frames) == 2
    assert frames[-1].shape == (render.n_pixels, render.n_pixels, 3)

    counts = core.vortex_counts(topology, vertices, settled_wave, scenarios.counting_radii, scenarios.vortex_threshold)
    figure = render.count_figure(scenarios.counting_radii, counts, scenarios.rotation * scenarios.counting_radii ** 2)
    plt.close(figure)
