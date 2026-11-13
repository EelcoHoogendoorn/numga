"""The smoke ring flow, each vertex moving across both its edges; a Bäcklund transformation as a step of the flow with
unit edges; a fold as a Lorentz transformation of the offsets; and a coarse wavy ring stepping once, its frame drawn."""

from __future__ import annotations

import numpy as np

from numga import concatenate
from examples.fluids.smoke_rings import core, render, scenarios


def test_each_vertex_moves_across_both_its_edges():
    """The velocity is perpendicular to the edges in and out of its vertex, so no edge changes length."""
    points = scenarios.torus_knot(*scenarios.knot_windings, *scenarios.knot_radii, scenarios.n_knot)
    outgoing = core.edges(points)
    incoming = concatenate([outgoing[-1:], outgoing[:-1]])
    velocities = core.velocity(points, scenarios.circulation)
    assert ((velocities | outgoing).abs() < 1e-13).all()
    assert ((velocities | incoming).abs() < 1e-13).all()


def test_a_backlund_transformation_is_a_step_of_the_flow():
    """The transformed polygon has unit edges, and lies a time of half the parameter along the smoke ring flow, to first
    order in the parameter."""
    points = scenarios.wavy_ring(scenarios.n_discrete, scenarios.wavy_mode, scenarios.wavy_tilt)
    for parameter in (0.02, 0.01):
        next_points = core.time_step(points, core.mv(core.Parameter, [parameter]))
        assert ((core.edges(next_points).norm() - 1).abs() < 1e-13).all()
        remainder = next_points - (points + core.velocity(points, scenarios.circulation) * (parameter / 2))
        # The tolerance tests the first order; the remainder shrinks with the square of the parameter.
        assert (remainder.norm() < 2 * parameter**2).all()


def test_a_fold_is_a_lorentz_transformation_of_the_offsets():
    """Folding an offset about the quad's diagonal is the Lorentz rotor of the edge, a boost along the direction the
    edge's point stands for and a turn about it, acting on the offset's null paravector."""
    rng = np.random.default_rng(0)
    edge = core.mv.vector(rng.normal(size=3)).dual().normalized()
    offsets = core.mv.vector(rng.normal(size=(6, 3))).dual().normalized()
    parameter = 0.3
    # The turn's angle flips with the parameter under a mirror, intrinsic as the parameter is.
    rapidity, angle = np.log(1 + 4 / parameter**2) / 2, core.mv(core.Parameter, [-np.arctan(2 / parameter)])
    rotor = (-(rapidity * edge.dual_inverse() + angle * edge) / 2).exp()
    image = rotor * (1 + offsets.dual_inverse()) * rotor.reverse()
    folded = (image.cast(core.Vector) / image.cast(core.Scalar)).dual()
    twist = core.mv(core.Parameter, [parameter])
    np.testing.assert_allclose((folded - core.next_offset(edge, offsets, twist)).kernel, 0.0, atol=1e-13)


def test_a_coarse_wavy_ring_steps_once_and_draws():
    n_vertices = 8
    curves = scenarios.flow_curves(scenarios.wavy_ring(n_vertices, scenarios.wavy_mode, scenarios.wavy_tilt),
                                   scenarios.surface_step, 1, 1)             # [2, n_vertices] Point-
    next(render.animate_curves(curves[-1:], scenarios.smoke_ring_velocity(curves[-1:]).norm()))
