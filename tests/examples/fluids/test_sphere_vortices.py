"""Vortices on the sphere: a vortex's flow is irrotational away from its two points; two vortices alone turn rigidly by their pair's rotor; the two time steps keep the impulse; and the travelling pairs step once, their first frame drawn."""

from __future__ import annotations

import numpy as np

from examples.fluids.sphere_vortices import core, render, scenarios


def test_a_vortex_flow_is_irrotational_away_from_its_two_points():
    """The circulation around every circle about the vortex is the vortex's own."""
    angles = np.array([0.2, 1.0, np.pi / 2, 2.6])
    places = scenarios.sphere_points(angles, np.zeros(len(angles)))                 # [4] Pivot+
    vortex = core.mv.xy                                                              # [] Pivot+
    speeds = places.commutator(vortex / (2 * np.pi * (places & vortex).norm_squared())).norm()   # [4] Scalar+
    assert ((speeds * (2 * np.pi * np.sin(angles)) - 1).abs() < 1e-12).all()


def test_two_vortices_alone_turn_rigidly():
    """The pair's rotor keeps the angle between two vortices and their impulse, over a long step."""
    pair = core.mv.bivector([[1, 0, 0.3], [1, 0.4, -0.2]]).normalized() * np.array([1, 0.6])   # [2] Pivot+
    successors = core.pair_step(pair, scenarios.schedule(2), 0.05, 3.0)              # [2] Pivot+
    squared_sines = (successors[0].normalized() & successors[1].normalized()).norm_squared()
    assert (squared_sines - (pair[0].normalized() & pair[1].normalized()).norm_squared()).abs() < 1e-12
    assert (successors.sum(axis=-1) - pair.sum(axis=-1)).norm() < 1e-12


def test_the_steps_keep_the_impulse():
    """One step of five vortices by the pairs' rotors keeps their impulse to round-off, the midpoint turns closely."""
    rng = np.random.default_rng(1)
    vortices = core.mv.bivector(rng.normal(size=(5, 3))).normalized() * np.array([1, -0.7, 0.5, 1.2, -1])   # [5] Pivot+
    impulse = vortices.sum(axis=-1)                                                  # [] Pivot+
    assert (core.pair_step(vortices, scenarios.schedule(5), 0.05, 0.1).sum(axis=-1) - impulse).norm() < 1e-12
    assert (core.midpoint_step(vortices, 0.05, 0.1, 4).sum(axis=-1) - impulse).norm() < 1e-3


def test_the_travelling_pairs_step_once_and_draw():
    next(render.animate(scenarios.travelling_pairs(1), scenarios.travel_view, 1))
