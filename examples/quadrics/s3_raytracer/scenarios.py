"""Four identical ellipsoids along the line of sight, and an eye walking towards them."""

from __future__ import annotations

import numpy as np

from examples.quadrics.s3_raytracer.core import (
    Plane, direction, inside, mv, origin, outlines, pixel_chart, project, reproject,
)


def walk(steps: int):
    # An ellipsoid: the dual quadric with the origin as centre, principal half-widths along the
    # basis directions, each the tangent of an angular half-width, and the -1 on the centre that
    # closes it. Four copies of one shape, carried to increasing angular distances along +x and
    # offset so they don't overlap.
    axes = direction(np.eye(3))
    ellipsoid = (axes * (axes & Plane) * np.tan(np.array([0.2, 0.28, 0.15])) ** 2).sum(axis=0) - origin * (origin & Plane)
    distances = np.array([0.7, 1.4, 2.1, 2.6])
    sideways, upward = np.array([-0.4, 0.4, -0.4, 0.4]), np.array([-0.3, -0.3, 0.3, 0.3])
    colors = np.array([[0.9, 0.3, 0.3], [0.3, 0.8, 0.4], [0.3, 0.5, 0.95], [0.95, 0.8, 0.3]])
    placed = (mv.xw * (distances / 2)).exp() * (mv.yw * (sideways / 2)).exp() * (mv.zw * (upward / 2)).exp() * (mv.xy * 0.4).exp()
    # The dual quadrics in the world, and their primal forms, which map a point to its polar plane.
    bodies = placed >> ellipsoid(placed << Plane)                       # [bodies] Point <- Plane
    surfaces = bodies.inverse()                                   # [bodies] Plane <- Point
    light = direction(np.array([-0.4, 0.6, 0.7]))
    fov = np.radians(80.0)

    # The eye walks along the x geodesic; the scene remains fixed in the world.
    eye_frames = (mv.xw * (np.linspace(0.0, 1.2, steps, endpoint=False) / 2)).exp()

    # --- checks ---------------------------------------------------------------------------
    # A ray is inside a body's outline cone exactly when its great circle origin + chart * lam
    # meets the body: the quadratic a * lam**2 + 2 * b * lam + c has real roots. Compared away
    # from the outline itself.
    chart = pixel_chart(fov, (90, 120))
    covered = inside(outlines(mv.rotor(), surfaces), chart)
    k_eye, k_dir = surfaces.reshape(-1, 1)(origin), surfaces.reshape(-1, 1)(chart)
    a, b, c = (chart & k_dir).to_array(), (chart & k_eye).to_array(), (origin & k_eye).to_array()
    disc = b * b - a * c
    clear = np.abs(disc) > 1e-3 * np.abs(disc).max(axis=1, keepdims=True)
    assert np.array_equal(covered[clear], (disc >= 0.0)[clear])
    # Reprojected hits lie on their surfaces.
    conics, polars = project(mv.rotor(), surfaces)
    np.testing.assert_allclose(conics.reshape(-1, 1)(chart, chart).to_array(), -disc / c**2, atol=1e-8)
    for body in range(4):
        pixels = np.nonzero(covered[body])[0][::10]
        hit = (origin * reproject(conics[body], polars[body], chart[pixels]) + chart[pixels]).normalized()
        np.testing.assert_allclose((hit & surfaces[body](hit)).to_array(), 0.0, atol=1e-8)
    return eye_frames, surfaces, colors, light, fov


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.quadrics.s3_raytracer import render

    scene = walk(48)
    save_figure(render.draw_walk(*scene, (180, 240), 2), "s3_raytracer")
    save_animation(render.frames(*scene, (180, 240), 2), "s3_raytracer", 80)
