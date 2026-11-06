"""The smoke ring flow: each vertex moves across both its edges; a Bäcklund transformation is a step of the flow with
unit edges; a fold is a Lorentz transformation of the offsets; and a coarse wavy ring steps once, its frame drawn."""

from __future__ import annotations

import numpy as np

from examples.fluids.smoke_rings import core, render, scenarios
from examples.util.topology import Topology


def test_each_vertex_moves_across_both_its_edges():
    """The velocity is perpendicular to the edges in and out of its vertex, so no edge changes length."""
    points = core.torus_knot(*scenarios.KNOT_WINDINGS, *scenarios.KNOT_RADII, scenarios.N_KNOT)
    outgoing = core.edges(points)
    arriving = ~Topology.loops([len(points)]).incidence[1, 0][1]
    incoming = (arriving * outgoing.field()).batch()
    velocities = core.velocity(points, scenarios.circulation)
    np.testing.assert_allclose((velocities | outgoing).to_array(), 0.0, atol=1e-13)
    np.testing.assert_allclose((velocities | incoming).to_array(), 0.0, atol=1e-13)


def test_a_backlund_transformation_is_a_step_of_the_flow():
    """The transformed polygon has unit edges, and lies a time of half the parameter along the smoke ring
    flow, to first order in the parameter."""
    points = core.wavy_ring(scenarios.N_DISCRETE, scenarios.WAVY_MODE, scenarios.WAVY_TILT)
    for parameter in (0.02, 0.01):
        moved = core.advance(points, core.mv(core.Parameter, [parameter]))
        np.testing.assert_allclose(core.edges(moved).norm().to_array(), 1.0, atol=1e-13)
        step = moved - (points + core.velocity(points, scenarios.circulation) * (parameter / 2))
        # The tolerance tests the first order: the remainder shrinks with the square of the parameter.
        assert step.norm().to_array().max() < 2 * parameter**2


def test_a_fold_is_a_lorentz_transformation_of_the_offsets():
    """Folding an offset about the quad's diagonal is the Lorentz rotor of the edge, a boost along the
    direction the edge's point stands for and a turn about it, acting on the offset's null paravector."""
    rng = np.random.default_rng(0)
    edge = core.mv.vector(rng.normal(size=3)).dual().normalized()
    offsets = core.mv.vector(rng.normal(size=(6, 3))).dual().normalized()
    parameter = 0.3
    # The turn's angle flips with the parameter under a mirror: intrinsic, as the parameter is.
    rapidity, angle = np.log(1 + 4 / parameter**2) / 2, core.mv(core.Parameter, [-np.arctan(2 / parameter)])
    rotor = (-(rapidity * edge.dual_inverse() + angle * edge) / 2).exp()
    image = rotor * (1 + offsets.dual_inverse()) * rotor.reverse()
    folded = (image.cast(core.Vector) / image.cast(core.Scalar)).dual()
    twist = core.mv(core.Parameter, [parameter])
    np.testing.assert_allclose((folded - core.fold(edge, offsets, twist)).kernel, 0.0, atol=1e-13)


def test_a_coarse_wavy_ring_steps_once_and_draws():
    n_vertices = 8
    curves = scenarios.sweep(core.wavy_ring(n_vertices, scenarios.WAVY_MODE, scenarios.WAVY_TILT), scenarios.SURFACE_STEP,
                             1, 1)                                           # [times, vertices] Point-
    next(render.animate_curves(curves[-1:], scenarios.flying(curves[-1:]).norm()))
