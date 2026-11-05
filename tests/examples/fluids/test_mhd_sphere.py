"""Magnetohydrodynamics on a sphere: the winding start on a coarse sphere, stepped once, its energies and its two frames
drawn."""

from __future__ import annotations

from examples.fluids.mhd_sphere import core, render, scenarios


def test_a_coarse_sphere_steps_once_and_draws():
    topology, vertices = scenarios.sphere(n_levels=1)
    surface = core.surface(topology, vertices)
    start = scenarios.winding(surface, vertices)
    states = list(core.evolve(surface, start, scenarios.viscosity, scenarios.resistivity, scenarios.dt, 1))
    scenarios.energies(surface, states[-1])
    list(render.animate(topology, vertices, scenarios.frames(surface, states)))
