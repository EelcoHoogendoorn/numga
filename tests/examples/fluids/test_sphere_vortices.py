"""Vortices on the sphere: a vortex's flow is irrotational away from its two points; a vortex with its
bivector and circulation both negated is the same vortex; two vortices alone turn rigidly by their pair's
rotor; the two time steps compare as they should; and the animation draws."""

from __future__ import annotations

import numpy as np

from examples.fluids.sphere_vortices import core, render, scenarios


def test_a_vortex_flow_is_irrotational_away_from_its_two_points():
    """The circulation around every circle about the vortex is the vortex's own."""
    vortex = core.pivot(np.array([0.0, 0.0, 1.0]))
    for angle in (0.2, 1.0, np.pi / 2, 2.6):
        place = core.pivot(np.array([np.sin(angle), 0.0, np.cos(angle)]))
        speed = place.commutator(vortex / (2 * np.pi * (place & vortex).norm_squared())).norm().to_array()
        np.testing.assert_allclose(speed * 2 * np.pi * np.sin(angle), 1.0, rtol=1e-12)


def test_a_vortex_negated_whole_is_the_same_vortex():
    """Negating some vortices' bivectors together with their circulations changes neither their motion,
    by either step, nor their energy or impulse."""
    rng = np.random.default_rng(1)
    vortices = core.Vortices(core.pivot(rng.normal(size=(5, 3))), np.array([1.0, -0.7, 0.5, 1.2, -1.0]), 0.05)
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    negated = core.Vortices(vortices.pivots * signs, vortices.circulations * signs, vortices.core_radius)
    for step in (lambda current: core.midpoint(current, 0.1, 4), lambda current: core.paired(current, core.schedule(5), 0.1)):
        np.testing.assert_allclose((step(vortices).pivots - step(negated).pivots * signs).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(core.energy(vortices).to_array(), core.energy(negated).to_array(), atol=1e-14)
    np.testing.assert_allclose(core.impulse(vortices).kernel, core.impulse(negated).kernel, atol=1e-14)


def test_two_vortices_alone_turn_rigidly():
    """The pair's rotor keeps the angle between two vortices and their impulse, over a long step."""
    pair = core.Vortices(core.pivot(np.array([[1.0, 0.0, 0.3], [1.0, 0.4, -0.2]])), np.array([1.0, 0.6]), 0.05)
    moved = core.paired(pair, core.schedule(2), 3.0)
    np.testing.assert_allclose((moved.pivots[0] & moved.pivots[1]).norm().to_array(), (pair.pivots[0] & pair.pivots[1]).norm().to_array(), rtol=1e-12)
    np.testing.assert_allclose(core.impulse(moved).kernel, core.impulse(pair).kernel, atol=1e-14)


def test_the_steps_compare_and_the_animation_draws():
    by_pairs, _ = scenarios.compared()
    assert len(list(render.animate([by_pairs[:3]], scenarios.TRAVEL_VIEW, 2))) == 2
