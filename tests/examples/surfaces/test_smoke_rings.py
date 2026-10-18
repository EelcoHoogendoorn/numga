"""The smoke ring flow: each vertex moves across both its edges, a ring flies rigidly, the flows keep
their edges and impulse; a Bäcklund transformation is a step of the flow with unit edges; a soliton runs
along a straight line; two vortex rings leapfrog under the Biot–Savart law; and the figures draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numga import concatenate

from examples.surfaces.smoke_rings import core, render, scenarios


def test_each_vertex_moves_across_both_its_edges():
    """The velocity is perpendicular to the edges in and out of its vertex, so no edge changes length."""
    points = core.torus_knot(*scenarios.KNOT_WINDINGS, *scenarios.KNOT_RADII, scenarios.KNOT_COUNT)
    outgoing = core.edges(points)
    incoming = concatenate([outgoing[-1:], outgoing[:-1]])
    velocities = core.velocity(points, scenarios.CIRCULATION)
    np.testing.assert_allclose((velocities | outgoing).to_array(), 0.0, atol=1e-13)
    np.testing.assert_allclose((velocities | incoming).to_array(), 0.0, atol=1e-13)


def test_a_backlund_transformation_is_a_step_of_the_flow():
    """The transformed polygon has unit edges, and lies a time of half the parameter along the smoke ring
    flow, to first order in the parameter."""
    points = core.wavy_ring(scenarios.DISCRETE_COUNT, scenarios.WAVY_MODE, scenarios.WAVY_TILT)
    for parameter in (0.02, 0.01):
        moved = core.advance(points, core.mv.xyz * parameter)
        np.testing.assert_allclose(core.edges(moved).norm().to_array(), 1.0, atol=1e-13)
        step = moved - (points + core.velocity(points, scenarios.CIRCULATION) * (parameter / 2))
        # The tolerance tests the first order: the remainder shrinks with the square of the parameter.
        assert step.norm().to_array().max() < 2 * parameter**2


def test_a_fold_is_a_lorentz_transformation_of_the_offsets():
    """Folding an offset about the quad's diagonal is the Lorentz rotor of the edge acting on the offset's
    null paravector."""
    rng = np.random.default_rng(0)
    edge = core.mv.vector(rng.normal(size=3)).normalized()
    offsets = core.mv.vector(rng.normal(size=(6, 3))).normalized()
    twist = 0.3 * core.mv.xyz
    rotor = (-(1 + 2 * twist.inverse()).log() * edge / 2).exp()
    image = rotor * (1 + offsets) * rotor.reverse()
    folded = image.cast(core.Vector) / image.cast(core.Scalar)
    np.testing.assert_allclose((folded - core.fold(edge, offsets, twist)).kernel, 0.0, atol=1e-13)


def test_the_scenes_pass_their_checks_and_draw():
    scenarios.flying_ring()
    curves = scenarios.sweep(core.wavy_ring(scenarios.WAVY_COUNT, scenarios.WAVY_MODE, scenarios.WAVY_TILT),
                             scenarios.SURFACE_STEP, 20, 10)
    plt.close(render.draw_surface(core.closed(curves)))
    assert len(render.animate_curves(core.closed(curves[:1]))) == 1
    solitons = scenarios.soliton()
    plt.close(render.draw_turns(core.turns(solitons)))
    assert len(render.animate_trail(solitons[:2])) == 2
    frogs = scenarios.leapfrog()
    plt.close(render.draw_swept(core.closed(frogs)))
    assert len(render.animate_curves(core.closed(frogs[:1]))) == 1
