"""The ideal gas's equilibrium surface and its Legendre change: the surface, a section, the horizontal lift, and the
first frame drawn, at two entropies."""
import numpy as np

from examples.thermodynamics.contact_geometry import render, scenarios


def test_legendre_runs_and_draws():
    entropies = np.array([-0.8, 1.2])
    family, changed_family = scenarios.surfaces(entropies, np.array([1.0]))
    section = scenarios.section(entropies, 1.0)
    scenarios.horizontal_lift()
    next(render.legendre(family, changed_family, *section, scenarios.moving(*section)))
