"""Equilibrium is Legendrian and the thermodynamic Legendre changes preserve contact."""
import itertools

import numpy as np
import pytest

from numga.backend.jax import derivative
from examples.thermodynamics.contact_geometry import core, render, scenarios

round_off = 1e-11
mv = core.mv


def test_ideal_gas_equilibrium_is_legendrian():
    states = mv.s.dual() * np.array([-0.5, 0.2, 0.8]) + mv.v.dual() * np.array([0.6, 1.1, 1.8])
    embedding = lambda state: core.equilibrium(state, scenarios.heat_capacity)
    equilibrium = embedding(states)
    tangent = derivative(embedding)(states)
    tangent_area = tangent(mv.s.dual()) & tangent(mv.v.dual())
    contact_volume = core.contact_element(equilibrium) ^ core.contact_derivative ^ core.contact_derivative
    np.testing.assert_allclose((core.contact_element(equilibrium) & tangent).kernel, 0, atol=round_off)
    np.testing.assert_allclose((tangent_area & core.contact_derivative).kernel, 0, atol=round_off)
    np.testing.assert_allclose(((mv.u & equilibrium) - scenarios.heat_capacity * (mv.t & equilibrium)).kernel, 0,
                               atol=round_off)
    np.testing.assert_allclose(((mv.p & equilibrium) * (mv.v & equilibrium) - (mv.t & equilibrium)).kernel, 0,
                               atol=round_off)
    np.testing.assert_allclose(contact_volume.kernel, 2, atol=round_off)


@pytest.mark.parametrize("change", [core.helmholtz, core.enthalpy])
def test_legendre_change_preserves_contact_away_from_equilibrium(change):
    points = mv.antivector([[0.4, -0.6, 0.7, 1.1, 1.8], [1.9, 0.8, 1.6, 0.5, 0.7]])
    pullback = core.contact_element(change(points)) & derivative(change)(points)
    gibbs = core.helmholtz(core.enthalpy(points))
    np.testing.assert_allclose((pullback - (core.contact_element(points) & core.Ambient)).kernel, 0, atol=round_off)
    np.testing.assert_allclose((gibbs - core.enthalpy(core.helmholtz(points))).kernel, 0, atol=round_off)
    np.testing.assert_allclose(((mv.u & gibbs) - (mv.u & points) + (mv.t & points) * (mv.s & points)
                                - (mv.p & points) * (mv.v & points)).kernel, 0, atol=round_off)


def test_horizontal_lift_gains_the_enclosed_area():
    path, transformed_path = scenarios.horizontal_lift()
    steps, middles = path[:, -1] - path[:, 0], path[:, len(path[0]) // 2]
    changed_steps, changed_middles = transformed_path[:, -1] - transformed_path[:, 0], transformed_path[:, len(path[0]) // 2]
    gap = path[-1, -1] - path[0, 0]
    entropy_span = mv.s & (path[1, -1] - path[1, 0])
    temperature_span = mv.t & (path[0, -1] - path[0, 0])
    np.testing.assert_allclose((core.contact_element(middles) & steps).kernel, 0, atol=round_off)
    np.testing.assert_allclose((core.contact_element(changed_middles) & changed_steps).kernel, 0, atol=round_off)
    np.testing.assert_allclose((gap - mv.u.dual() * entropy_span * temperature_span).kernel, 0, atol=round_off)


def test_section_tangents_preserve_contact_and_trade_slopes():
    points, tangents, changed_points, changed_tangents = scenarios.section(np.linspace(-0.8, 1.2, 9), 1.0)
    np.testing.assert_allclose((core.contact_element(points) & tangents).kernel, 0, atol=round_off)
    np.testing.assert_allclose((core.contact_element(changed_points) & changed_tangents).kernel, 0, atol=round_off)
    np.testing.assert_allclose(((mv.u & tangents) / (mv.s & tangents) - (mv.t & points)).kernel, 0, atol=round_off)
    np.testing.assert_allclose(((mv.u & changed_tangents) / (mv.s & changed_tangents) + (mv.s & points)).kernel, 0,
                               atol=round_off)


def test_legendre_frames_are_drawn():
    entropies = np.linspace(-0.8, 1.2, 9)
    family, changed_family = scenarios.surfaces(entropies, np.array([0.8, 1.0]))
    section = scenarios.section(entropies, 1.0)
    frame = next(itertools.islice(render.legendre(family, changed_family, *section, scenarios.moving(*section)), 4, None))
    assert frame.ndim == 3 and frame.dtype == np.uint8
