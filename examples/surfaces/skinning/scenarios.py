"""A two-bone cylinder twisted at the tip compares normalised motor blending, motor slerp, and point-map blending."""

from __future__ import annotations

import numpy as np

from examples.surfaces.skinning.core import Point, blend_skin, cylinder, mv

# The cylinder's rings along the bones and vertices around each, and how far the tip bone is twisted.
n_rings, n_around = 12, 24
twist_angle = np.radians(150.0)


# --- math -----------------------------------------------------------------------------
def skinning(vertices: Point, weights: np.ndarray, twist_angle: float) -> tuple[Point, Point, Point]:
    """A fixed root bone and a tip bone twisted by `twist_angle` deform `vertices` (`[...] Point-`) under all three blends."""
    root = mv.rotor()                                                          # [] Motor+
    # Turn the tip bone about the shared x axis.
    tip = (mv.yz * (twist_angle / 2)).exp()                                    # [] Motor+
    return blend_skin(vertices, weights, root, tip)                            # [...] Point-, Point-, Point-


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.surfaces.skinning import render

    vertices, weights = cylinder(n_rings, n_around)                            # [n_rings * n_around] Point-, [n_rings * n_around]
    motor_skin, slerp_skin, matrix_skin = skinning(vertices, weights, twist_angle)
    save_figure(render.draw_skinning(motor_skin, slerp_skin, matrix_skin, n_rings, n_around), "skinning")

    # checks
    # Both motor blends stay rigid about the axis so every vertex keeps unit radius, whereas the
    # point-map blend averages two circle points onto the chord between them.
    assert (((mv.yz & motor_skin.normalized()).norm() - 1).abs() < 1e-12).all()
    assert (((mv.yz & slerp_skin.normalized()).norm() - 1).abs() < 1e-12).all()
    chord_radius = np.sqrt(1 - 2 * weights * (1 - weights) * (1 - np.cos(twist_angle)))
    assert (((mv.yz & matrix_skin.normalized()).norm() - chord_radius).abs() < 1e-12).all()


if __name__ == "__main__":
    main()
