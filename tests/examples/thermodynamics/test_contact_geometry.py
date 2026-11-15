"""Checks that the ideal gas equilibrium surface, constant-volume section, horizontal contact lift, and Legendre
animation frame execute cleanly."""
import numpy as np

from examples.thermodynamics.contact_geometry import render, scenarios


def test_legendre_runs_and_draws():
    entropies = np.array([-0.8, 1.2])
    family, transformed_family = scenarios.surfaces(entropies, np.array([1.0]))
    section = scenarios.section(entropies, 1.0)
    scenarios.horizontal_lift()
    next(render.legendre(family, transformed_family, *section, scenarios.tangent_stages(*section)))
