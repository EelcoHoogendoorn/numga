"""A rotating Bose-Einstein condensate: its descent toward the lattice of vortices, counted and drawn, on the smallest trap."""

from examples.quantum.condensate import render, scenarios


def test_one_step_of_the_descent_is_counted_and_drawn():
    topology, vertices, frames = scenarios.settled(n_rings=2, n_steps=1, stride=1)
    wave = list(frames)[-1]
    counted = scenarios.counted(topology, vertices, wave)
    next(render.animate(topology, vertices, [wave.scalar_norm_squared()]))
    render.draw_count(scenarios.counting_radii, counted, scenarios.rotation * scenarios.counting_radii ** 2)
