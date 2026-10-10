"""Cubic curves with negative weights and a point at infinity, the three conics, their tangent planes, and
curves of motors."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.geometry.bezier import core

# Samples along each curve, and every how many samples a tangent plane is drawn.
SAMPLES = 400
TANGENT_STRIDE = 25
# Cubic curves: control points, and per case the weights; the third case turns the second control
# point into a point at infinity heading up and to the right.
CUBIC_POINTS = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]])
CUBIC_HEADING = np.array([1.0, 1.0])
CUBIC_WEIGHTS = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, -0.6, 1.0], [1.0, 1.0, 1.0, 1.0]])
# Quadratic curves on one control triangle: the middle weight against the end weights makes an
# ellipse, a parabola and a hyperbola.
QUADRATIC_POINTS = np.array([[-1.0, 0.0], [0.0, 1.5], [1.0, 0.0]])
QUADRATIC_WEIGHTS = np.array([[1.0, 0.5, 1.0], [1.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
# The middle weight of the quadratic curve swept from an ellipse through a parabola to a hyperbola and
# back, the frames of the sweep, and the duration of each.
SWEEP = (0.3, 2.0)
SWEEP_FRAMES = 48
DURATION_MS = 80
# Motors: positions and angles of the control placements, the triangle they carry, and every how many
# samples it is drawn.
PLACES = np.array([[0.0, 0.0], [1.5, 2.5], [3.0, -1.0], [4.5, 1.5]])
TURNS = np.array([0.0, 3.0, 6.0, 9.0])
TRIANGLE = np.array([[0.3, 0.0], [-0.15, 0.15], [-0.15, -0.15]])
FRAME_STRIDE = 40


# --- math -----------------------------------------------------------------------------
def parameters() -> np.ndarray:
    return np.linspace(0.0, 1.0, SAMPLES + 1)                                  # [samples + 1]


def cubic() -> tuple[core.Point, core.Point, core.Point]:
    """The control points of each case, the curves, and the complementary curves."""
    points = core.point(CUBIC_POINTS)                                          # [controls] Point
    ideal = stack([points[0], core.heading(CUBIC_HEADING), points[2], points[3]])   # [controls] Point
    controls = stack([points, points, ideal])                                  # [cases, controls] Point
    curves = core.blend(controls, CUBIC_WEIGHTS, parameters())                 # [cases, samples + 1] Point
    complements = core.blend(controls, core.complementary(CUBIC_WEIGHTS), parameters())   # [cases, samples + 1] Point
    return controls, curves, complements


def conics() -> tuple[core.Point, core.Point, core.Point, core.Conic, core.Scalar]:
    """The quadratic curves of each case, their complements, their conics, and each conic's form on
    directions."""
    controls = core.point(QUADRATIC_POINTS)                                    # [controls] Point
    curves = core.blend(controls, QUADRATIC_WEIGHTS, parameters())             # [cases, samples + 1] Point
    complements = core.blend(controls, core.complementary(QUADRATIC_WEIGHTS), parameters())   # [cases, samples + 1] Point
    shapes = core.conic(controls, QUADRATIC_WEIGHTS)                           # [cases] Plane <- Point
    return controls, curves, complements, shapes, core.at_infinity(shapes)     # [cases, 2] Scalar


def tangent_planes() -> core.Plane:
    """The tangent planes of the quadratic curves, as quadratic curves of planes."""
    planes, weights = core.tangents(core.point(QUADRATIC_POINTS), QUADRATIC_WEIGHTS)   # [cases, 3] Plane
    return core.blend(planes, weights, parameters()[::TANGENT_STRIDE])          # [cases, tangents] Plane


def sweep() -> tuple[core.Point, core.Point, core.Point, core.Conic]:
    """As the middle weight sweeps out and back: the weighted control points of the quadratic curve and of
    its complement, the curve, its complement, and its conic."""
    phase = np.linspace(0, 2 * np.pi, SWEEP_FRAMES, endpoint=False)
    middle = np.exp(np.log(SWEEP).mean() - np.log(SWEEP[1] / SWEEP[0]) / 2 * np.cos(phase))   # [frames]
    weights = np.stack([np.ones_like(middle), middle, np.ones_like(middle)], axis=-1)   # [frames, 3]
    controls = core.point(QUADRATIC_POINTS)                                    # [controls] Point
    weighted = stack([controls * weights, controls * core.complementary(weights)], axis=1)   # [frames, 2, controls] Point
    curves = core.blend(controls, weights, parameters())                       # [frames, samples + 1] Point
    complements = core.blend(controls, core.complementary(weights), parameters())   # [frames, samples + 1] Point
    return weighted, curves, complements, core.conic(controls, weights)        # [frames] Plane <- Point


def motors() -> tuple[core.Point, core.Point, core.Point, core.Point]:
    """For the control motors, their blend renormalised, and their curve on the group: where each carries
    the origin, and the triangle it carries, every so many samples."""
    controls = (((core.mv.xw * PLACES[:, 0] + core.mv.yw * PLACES[:, 1]) * 0.5).exp()
                * (core.mv.xy * (-TURNS / 2)).exp())                           # [controls] Motor
    blended = core.blend(controls, np.ones(len(TURNS)), parameters()).normalized()   # [samples + 1] Motor
    moving = core.geodesic(controls, parameters())                             # [samples + 1] Motor
    curves = stack([blended, moving])                                          # [curves, samples + 1] Motor
    triangle = core.point(TRIANGLE)                                            # [corners] Point
    origin = core.mv.w.dual()                                                  # [] Point
    return (controls >> origin,                                                # [controls] Point
            controls[:, None] >> triangle,                                     # [controls, corners] Point
            curves >> origin,                                                  # [curves, samples + 1] Point
            curves[:, ::FRAME_STRIDE, None] >> triangle)                       # [curves, frames, corners] Point


# --- plumbing -------------------------------------------------------------------------
CUBIC_BOX = (-1.5, 5.5, -2.5, 3.5)
CONIC_BOX = (-3.0, 3.0, -2.5, 3.0)


def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.geometry.bezier import render

    controls, curves, complements = cubic()
    save_figure(render.draw_curves(controls, curves, complements, CUBIC_BOX), "bezier_cubics")
    triangle, quadratics, quadratic_complements, shapes, kinds = conics()
    save_figure(render.draw_conics(triangle, quadratics, quadratic_complements, shapes, CONIC_BOX), "bezier_conics")
    planes = tangent_planes()
    figure = render.draw_conics(triangle, quadratics, quadratic_complements, shapes, CONIC_BOX)
    save_figure(render.draw_tangents(planes, figure), "bezier_tangents")
    save_animation(render.animate_rays(*sweep()), "bezier_cone", DURATION_MS)
    centres, placements, paths, frames = motors()
    save_figure(render.draw_motors(centres, placements, paths, frames), "bezier_motors")

    # checks
    # Each curve starts at its first control point.
    assert (render.scalars((curves[:2, 0] & controls[:2, 0]).norm()) < 1e-10).all()
    # Both arcs of each quadratic curve lie on its conic.
    assert np.abs(render.scalars(quadratics & shapes[:, None](quadratics))).max() < 1e-10
    assert np.abs(render.scalars(quadratic_complements & shapes[:, None](quadratic_complements))).max() < 1e-10
    # On directions the conic is definite, degenerate and indefinite: ellipse, parabola, hyperbola.
    eigenvalues = render.scalars(kinds)
    assert (eigenvalues[0] > 0).all() and np.abs(eigenvalues[1]).min() < 1e-10 and eigenvalues[2].prod() < 0
    # Every tangent plane passes through its point of the curve and lies on the conic's envelope.
    touching = quadratics[:, ::TANGENT_STRIDE]                                 # [cases, tangents] Point
    assert np.abs(render.scalars(planes & touching)).max() < 1e-10
    assert np.abs(render.scalars(planes & shapes.inverse()[:, None](planes))).max() < 1e-10
    # Both curves of motors start and end at the end placements.
    assert render.scalars((paths[:, [0, -1]] & centres[[0, -1]]).norm()).max() < 1e-10


if __name__ == "__main__":
    main()
