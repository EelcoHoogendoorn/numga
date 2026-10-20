"""Four identical ellipsoids along the line of sight on the 3-sphere, and an eye walking towards them."""

from __future__ import annotations

import numpy as np

from examples.quadrics.s3_raytracer.core import (
    Plane, direction, inside, mv, origin, outlines, pixel_chart, project, reproject,
)

# The ellipsoid's angular half-widths along its axes; how far each copy lies along the line of sight, and
# its offset sideways and upward, by angle; its turn; and its colour.
HALF_WIDTHS = np.array([0.2, 0.28, 0.15])
DISTANCES = np.array([0.7, 1.4, 2.1, 2.6])
SIDEWAYS = np.array([-0.4, 0.4, -0.4, 0.4])
UPWARD = np.array([-0.3, -0.3, 0.3, 0.3])
TURN = 0.8
COLOURS = np.array([[0.9, 0.3, 0.3], [0.3, 0.8, 0.4], [0.3, 0.5, 0.95], [0.95, 0.8, 0.3]])
# The light's direction, the field of view, and how far the eye walks along the line of sight.
LIGHT = np.array([-0.4, 0.6, 0.7])
FIELD_OF_VIEW = np.radians(80.0)
WALK = 1.2
# The frames of the walk, the image's pixels and their supersampling, and the duration of a frame in
# milliseconds; the coarser chart the checks run on, and every how many covered pixels a hit is checked.
N_FRAMES = 48
SHAPE = (180, 240)
SUPERSAMPLE = 2
FRAME_DURATION = 80
CHECK_SHAPE = (90, 120)
CHECK_STRIDE = 10


# --- math -----------------------------------------------------------------------------
def walk(n_frames: int):
    """The eye's frames along the walk, the bodies as maps from points to their polar planes, their
    colours, the light and the field of view."""
    # An ellipsoid: the dual quadric with the origin as centre, principal half-widths along the basis
    # directions, each the tangent of an angular half-width, and the -1 on the centre that closes it.
    axes = direction(np.eye(3))                                                # [3] Point
    ellipsoid = (axes * (axes & Plane) * np.tan(HALF_WIDTHS) ** 2).sum(axis=0) - origin * (origin & Plane)   # [] Point <- Plane
    # Four copies of one shape, carried to increasing angular distances along x, offset so they don't overlap.
    placed = ((mv.xw * (DISTANCES / 2)).exp() * (mv.yw * (SIDEWAYS / 2)).exp() * (mv.zw * (UPWARD / 2)).exp()
              * (mv.xy * (TURN / 2)).exp())                                    # [bodies] Motor
    bodies = placed >> ellipsoid(placed << Plane)                              # [bodies] Point <- Plane
    # The eye walks along the x geodesic; the scene stays fixed in the world.
    eye_frames = (mv.xw * (np.linspace(0.0, WALK, n_frames, endpoint=False) / 2)).exp()   # [frames] Motor
    return eye_frames, bodies.inverse(), COLOURS, direction(LIGHT), FIELD_OF_VIEW


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.quadrics.s3_raytracer import render

    scene = walk(N_FRAMES)
    save_figure(render.draw_walk(*scene, SHAPE, SUPERSAMPLE), "s3_raytracer")
    save_animation(render.frames(*scene, SHAPE, SUPERSAMPLE), "s3_raytracer", FRAME_DURATION)

    # checks
    # A ray is inside a body's outline cone exactly when its great circle `origin + chart * t` meets the
    # body, its quadratic `a * t**2 + 2 * b * t + c` having real roots; compared away from the outline
    # itself. The screen conic is that discriminant, and reprojected hits lie on their surfaces.
    _, surfaces, _, _, _ = scene
    chart = pixel_chart(FIELD_OF_VIEW, CHECK_SHAPE)
    covered = inside(outlines(mv.rotor(), surfaces), chart)
    at_eye, along = surfaces.reshape(-1, 1)(origin), surfaces.reshape(-1, 1)(chart)
    a, b, c = (chart & along).to_array(), (chart & at_eye).to_array(), (origin & at_eye).to_array()
    discriminant = b * b - a * c
    clear = np.abs(discriminant) > 1e-3 * np.abs(discriminant).max(axis=1, keepdims=True)
    assert np.array_equal(covered[clear], (discriminant >= 0.0)[clear])
    conics, polars = project(mv.rotor(), surfaces)
    np.testing.assert_allclose(conics.reshape(-1, 1)(chart, chart).to_array(), -discriminant / c**2, atol=1e-8)
    for body in range(len(DISTANCES)):
        pixels = np.nonzero(covered[body])[0][::CHECK_STRIDE]
        hit = (origin * reproject(conics[body], polars[body], chart[pixels]) + chart[pixels]).normalized()
        np.testing.assert_allclose((hit & surfaces[body](hit)).to_array(), 0.0, atol=1e-8)


if __name__ == "__main__":
    main()
