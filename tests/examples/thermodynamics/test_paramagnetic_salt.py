"""Adiabatic demagnetization in a crystal: the salt's equation of state, the ideal cycle and a coarse crystal drawn, and
the crystal run for a frame and drawn."""

from itertools import islice

from examples.thermodynamics.paramagnetic_salt import render, scenarios


def test_salt_runs_and_draws():
    render.draw_equation_of_state(*scenarios.equation_of_state(), scenarios.moment)
    render.draw_cycle(*scenarios.corners(), *scenarios.lines())
    topology, vertices, sites, frames = scenarios.refrigerator(n_along=3, n_across=3)
    render.draw_crystal(topology, vertices, sites, scenarios.bath_temperature, scenarios.load_temperature)
    list(render.animate_crystal(topology, vertices, sites, scenarios.bath_temperature, scenarios.load_temperature,
                                scenarios.highest_field, scenarios.moment, islice(frames, 2)))
