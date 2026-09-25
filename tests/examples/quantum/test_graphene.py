"""Graphene: the Hamiltonian squares to the field's length, transport around a circle turns a frame
by the solid angle it encloses, and the scenarios pass their checks and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.graphene import core, render, scenarios


def test_the_hamiltonian_squares_to_the_fields_length():
    """H(H(psi)) == (field | field) * psi, so the eigenvalues are plus and minus the field's length."""
    field = core.mv.vector(np.array([0.3, -1.1, 0.7]))
    psi = core.mv(core.Even.output_subspace, np.array([0.4, -0.2, 1.3, 0.5]))
    H = core.hamiltonian(field)
    np.testing.assert_allclose((H(H(psi)) - (field | field) * psi).kernel, 0.0, atol=1e-12)


def test_transport_around_a_circle_of_latitude_turns_by_the_enclosed_solid_angle():
    """Around the circle at polar angle theta the enclosed solid angle is 2 pi (1 - cos theta), and
    the holonomy's scalar part is the cosine of half of it."""
    theta, count = np.array([0.3, 1.0, 2.0]), 400
    along = core.circle(count)[:, None] * core.mv.scalar(np.sin(theta)[:, None])
    directions = core.mv.z * core.mv.scalar(np.cos(theta)[:, None]) + along       # [count + 1, theta] Vector
    holonomy = core.transport(directions)[-1]                                      # [theta] Rotor
    np.testing.assert_allclose(holonomy.select[0].to_array(), np.cos(np.pi * (1 - np.cos(theta))), atol=1e-3)


def test_scenarios_pass_their_checks_and_draw():
    radii, gaps = np.linspace(0.01, 0.4, 5), np.array([0.0, 0.5])
    momenta, field, values = scenarios.bands(4.5, 21, 0.0)
    for figure in (render.draw_field(momenta, field),
                   render.draw_bands(momenta, values),
                   render.draw_textures(*scenarios.textures(0.5, 7, 0.3)),
                   render.draw_berry(radii, gaps, *scenarios.berry(radii, gaps, 200))):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
