"""A cellular flame: the mixture on a coarse disk burning for a step a frame, and its first shown frame drawn."""

from examples.thermodynamics.flame import core, render, scenarios


def test_burning_runs_and_draws():
    topology, vertices, frames = scenarios.spreading(n_rings=4, n_frames=scenarios.n_unshown + 1, n_stride=1)
    next(render.animate(topology, vertices, scenarios.radius, core.energy, frames))
