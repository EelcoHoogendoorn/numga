"""A coarse paramagnetic crystal slab runs through the equation-of-state chart, ideal cycle chart, resting geometry
figure, and animation renderer."""

from itertools import islice

from examples.thermodynamics.paramagnetic_salt import render, scenarios


def test_salt_runs_and_draws():
    render.draw_equation_of_state(*scenarios.equation_of_state(), scenarios.moment)
    render.draw_cycle(*scenarios.corners(), *scenarios.lines())
    topology, vertices, cell_spins, sites, frames = scenarios.refrigerator(n_along=3, n_across=3)
    render.draw_crystal(topology, vertices, sites, scenarios.bath_temperature, scenarios.load_temperature)
    rendered = list(render.animate_crystal(topology, vertices, cell_spins, sites, scenarios.bath_temperature,
                                           scenarios.load_temperature, scenarios.highest_field, *scenarios.corners(),
                                           *scenarios.lines(), islice(frames, 2)))
    assert len(rendered) == 2
