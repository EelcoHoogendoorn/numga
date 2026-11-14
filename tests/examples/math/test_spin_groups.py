"""Coverage tests for spin_groups verify the signature (3, 1) involution, spinor centre, isoclinic factorization, linking, and flow rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.math.spin_groups import core, render, scenarios


def test_rotations_and_boosts_of_the_signature_3_1():
    """In signature (3, 1) the involution fixes rotation planes and negates boost planes, and the
    pseudoscalar squares to minus one on spinors."""
    Bivector, Even, positives, pseudoscalar = core.signature(3, 1)
    involution = positives >> Bivector
    assert (involution(core.mv.xy) - core.mv.xy).scalar_norm_squared() < 1e-12
    assert (involution(core.mv.xt) + core.mv.xt).scalar_norm_squared() < 1e-12
    action = (pseudoscalar * Even).eigvals()
    assert ((action.squared() + 1).abs() < 1e-12).all()


def test_one_signature_and_one_frame_of_the_flows_draw():
    """Evaluates one signature row, spinor centre, isoclinic split, and linking integral, and
    renders one static flow figure and one animation frame."""
    list(scenarios.signature_rows(((3, 1),), scenarios.seed))
    list(scenarios.spinor_centres(((3, 1),)))
    generator = core.mv.xy + core.mv.zw
    left_factor, right_factor = core.isoclinic_factors(generator)
    assert (left_factor * right_factor - generator.exp()).scalar_norm_squared() < 1e-12
    left_orbits, right_orbits, knotted_orbits = core.isoclinic_orbits(2, scenarios.heights[:1], 2)
    assert core.linking(left_orbits[0, 0], right_orbits[0, 1]).antinorm() >= 0
    figure = render.draw_flows(left_orbits, right_orbits, knotted_orbits)
    plt.close(figure)
    next(render.animate_flows(left_orbits, right_orbits, knotted_orbits, scenarios.flow_steps(2, 1)))
