"""Variational polarization, coupled mechanical derivatives and stable electrostriction."""

from dataclasses import replace

import numpy as np
import pytest

from numga.backend.jax import derivative
from examples.electromagnetism.electrostriction import core, scenarios

ROUND_OFF = 1e-12
# Newton steps converge quadratically; seven reach round-off at the strongest field.
CONVERGED_STEPS = 7


@pytest.fixture(scope="module")
def loading():
    model = scenarios.lattice()
    fields = core.mv.x * np.array([0, scenarios.FIELD, -scenarios.FIELD])
    positions = model.equilibrium(fields, CONVERGED_STEPS)
    return model, fields, positions


def test_polarization_minimizes_the_full_electrical_energy(loading):
    model, fields, positions = loading
    response = model.response(positions)
    dipoles = response.solve(fields)
    np.testing.assert_allclose((response(dipoles) - fields).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((response - response.adjoint()).kernel, 0, atol=ROUND_OFF, rtol=0)

    def electrical(polarization):
        return ((polarization | response(polarization)) / 2
                - (fields | polarization)).sites.sum()

    np.testing.assert_allclose(derivative(electrical)(dipoles).kernel, 0, atol=ROUND_OFF, rtol=0)
    reduced = model.energy(positions, fields) - model.elastic_energy(positions)
    np.testing.assert_allclose((electrical(dipoles) - reduced).kernel, 0, atol=ROUND_OFF, rtol=0)

    assert response.eigvalsh().to_array().min() > 20


def test_typed_derivatives_predict_changes_of_energy_and_response(loading):
    model, fields, positions = loading
    position, field = positions[1], fields[1]
    particles = model.rest.gatype.site_shape[0]
    rng = np.random.default_rng(2)
    displacement = core.mv.vector(rng.normal(size=(particles, 2))).field() * 0.01
    other = core.mv.vector(rng.normal(size=(particles, 2))).field() * 0.01
    increment = 1e-3

    def energy(value):
        return model.energy(value, field)

    gradient = derivative(energy)(position)
    curvature = derivative(derivative(energy))(position)
    upper, lower = position + increment * displacement, position - increment * displacement
    energy_change = (energy(upper) - energy(lower)) / (2 * increment)
    gradient_change = (derivative(energy)(upper) - derivative(energy)(lower)) / (2 * increment)
    # These bounds test central-difference accuracy rather than round-off.
    np.testing.assert_allclose(gradient(displacement).kernel, energy_change.kernel, atol=3e-9, rtol=1e-6)
    np.testing.assert_allclose(curvature(displacement).kernel, gradient_change.kernel, atol=3e-9, rtol=1e-6)
    np.testing.assert_allclose((curvature(displacement, other) - curvature(other, displacement)).kernel,
                               0, atol=ROUND_OFF, rtol=0)

    # Differentiating a field map appends a field slot for the displacement. The dipole input
    # stays open until it is bound, independently of that displacement slot.
    dipoles = model.response(position).solve(field)
    change = derivative(model.response)(position)
    predicted = change(dipoles, displacement)
    measured = (model.response(upper)(dipoles) - model.response(lower)(dipoles)) / (2 * increment)
    np.testing.assert_allclose(predicted.kernel, measured.kernel, atol=3e-9, rtol=1e-6)


def test_dipole_readjustment_changes_stiffness_but_not_the_stationary_first_derivative(loading):
    model, fields, positions = loading
    position, field = positions[1], fields[1]
    dipoles = model.response(position).solve(field)

    def relaxed(value):
        return model.energy(value, field)

    def fixed_dipoles(value):
        electrical = ((dipoles | model.response(value)(dipoles)) / 2
                      - (field | dipoles)).sites.sum()
        return model.elastic_energy(value) + electrical

    np.testing.assert_allclose((derivative(relaxed)(position) - derivative(fixed_dipoles)(position)).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    full = derivative(derivative(relaxed))(position)
    frozen = derivative(derivative(fixed_dipoles))(position)
    compression = -core.mv.x * (core.mv.x | position)
    # Allowing an internal degree of freedom to relax softens the mechanical response.
    assert (frozen(compression, compression) - full(compression, compression)).to_array() > 0.1


def test_mechanical_equilibria_are_stable_and_even_in_the_field(loading):
    model, fields, positions = loading
    energy = lambda value: model.energy(value, fields)
    gradient = derivative(energy)(positions)
    curvature = derivative(derivative(energy))(positions)
    np.testing.assert_allclose(gradient.kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((positions[0] - model.rest).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((positions[1] - positions[2]).kernel, 0, atol=ROUND_OFF, rtol=0)
    dipoles = model.response(positions).solve(fields)
    np.testing.assert_allclose((dipoles[1] + dipoles[2]).kernel, 0, atol=ROUND_OFF, rtol=0)
    assert core.strain(positions[1], model.rest, core.mv.x).to_array() < -0.05
    assert curvature.eigvalsh().to_array().min() > 0.05


def test_rotating_the_lattice_and_field_carries_the_whole_response(loading):
    model, fields, positions = loading
    turn = (core.mv.xy * -0.37).exp()
    turned = replace(model, rest=turn >> model.rest)
    np.testing.assert_allclose((turned.energy(turn >> positions, turn >> fields)
                               - model.energy(positions, fields)).kernel, 0, atol=ROUND_OFF, rtol=0)
    dipoles = model.response(positions).solve(fields)
    rotated_dipoles = turned.response(turn >> positions).solve(turn >> fields)
    np.testing.assert_allclose((rotated_dipoles - (turn >> dipoles)).kernel, 0, atol=ROUND_OFF, rtol=0)


def test_figures_and_animation_draw(loading):
    import matplotlib.pyplot as plt
    from examples.electromagnetism.electrostriction import render

    model, fields, positions = loading
    dipoles = model.response(positions).solve(fields)
    shape = render.draw_lattice(model.rest, model.bonds, positions[1], dipoles[1], fields[1],
                                 scenarios.DIPOLE_SCALE, scenarios.FIELD_SCALE)
    parallel = core.strain(positions[:2], model.rest, core.mv.x)
    transverse = core.strain(positions[:2], model.rest, core.mv.y)
    curve = render.draw_loading(np.array([0, scenarios.FIELD]), parallel, transverse)
    lattice = render.draw_rest(model.rest, model.bonds)
    for figure in (shape, curve, lattice):
        figure.canvas.draw()
        plt.close(figure)
    frames = render.animate_lattice(model.rest, model.bonds, positions[:2], dipoles[:2], fields[:2],
                                     scenarios.DIPOLE_SCALE, scenarios.FIELD_SCALE)
    assert len(frames) == 2
    assert np.any(frames[0] != frames[1])
