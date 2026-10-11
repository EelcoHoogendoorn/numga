"""An octahedron refined with weighted vertices, and refined before and after a perspective."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.surfaces.projective_loop import core

# The octahedron: its vertices, the one at TOP on top, and its faces, counter-clockwise from outside.
OCTAHEDRON = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
TOP = 4
FACES = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
# Loop steps taken.
DEPTH = 3
# The weights of the vertices per case: all alike, the top one heavy, the top one light.
WEIGHTS = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 4.0, 1.0], [1.0, 1.0, 1.0, 1.0, 0.25, 1.0]])
# The octahedron stretched and set in front of the perspective, and the perspective: its centre at the
# origin, its axis the plane z = 0, and its strength, one over the distance at which it halves sizes.
STRETCH = np.array([1.0, 1.0, 1.5])
OFFSET = np.array([0.0, 0.0, 1.6])
STRENGTH = 1.0
# The top vertex's weight swept from light to heavy and back, the frames of the sweep, and the duration
# of each.
SWEEP = (0.25, 4.0)
SWEEP_FRAMES = 48
DURATION_MS = 60


# --- math -----------------------------------------------------------------------------
def weighted() -> tuple[core.Point, core.Point, np.ndarray]:
    """The weighted octahedra of each case, the surfaces they refine to, and the surfaces' faces."""
    subdivide, refined = core.subdivision(core.Mesh(core.point(OCTAHEDRON).field(), FACES), DEPTH)   # [refined, vertices] Scalar
    controls = (core.point(OCTAHEDRON) * WEIGHTS).field()                     # [cases] Point[vertices]
    return controls, subdivide * controls, refined.faces                       # [cases] Point[refined]


def sweep() -> tuple[core.Point, core.Point, np.ndarray]:
    """The octahedron as its top vertex's weight sweeps out and back along a cosine of its logarithm, and
    the surfaces it refines to, all frames through the one composed map."""
    subdivide, refined = core.subdivision(core.Mesh(core.point(OCTAHEDRON).field(), FACES), DEPTH)   # [refined, vertices] Scalar
    phase = np.linspace(0, 2 * np.pi, SWEEP_FRAMES, endpoint=False)
    top = np.exp(np.log(SWEEP).mean() - np.log(SWEEP[1] / SWEEP[0]) / 2 * np.cos(phase))   # [frames]
    weights = np.where(np.arange(len(OCTAHEDRON)) == TOP, top[:, None], 1.0)  # [frames, vertices]
    controls = (core.point(OCTAHEDRON) * weights).field()                     # [frames] Point[vertices]
    return controls, subdivide * controls, refined.faces                       # [frames] Point[refined]


def perspective() -> tuple[core.Point, core.Point, core.Point, np.ndarray]:
    """The octahedron after the perspective, and the surface refined after and before it: as points with
    their weights, and as positions, each point divided by its weight before the refinement."""
    subdivide, refined = core.subdivision(core.Mesh(core.point(OCTAHEDRON).field(), FACES), DEPTH)   # [refined, vertices] Scalar
    shape = core.point(OCTAHEDRON * STRETCH + OFFSET).field()                  # Point[vertices]
    seen = core.collineation(core.mv.w.dual(), core.mv.z, STRENGTH)           # [] Point <- Point
    weighted_orders = stack([subdivide * seen(shape), seen(subdivide * shape)])   # [orders] Point[refined]
    positional_orders = stack([subdivide * core.unit(seen(shape)), seen(subdivide * core.unit(shape))])   # [orders] Point[refined]
    return seen(shape), weighted_orders, positional_orders, refined.faces


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.surfaces.projective_loop import render

    controls, surfaces, faces = weighted()
    save_figure(render.draw_weights(controls, FACES, surfaces, faces), "projective_loop_weights")
    save_animation(render.animate_weights(*sweep()[:2], FACES, faces), "projective_loop_sweep", DURATION_MS)
    seen, weighted_orders, positional_orders, faces = perspective()
    save_figure(render.draw_orders(seen, FACES, weighted_orders, positional_orders, faces), "projective_loop_perspective")

    # checks
    # Refining points with their weights commutes with the perspective; refining positions does not.
    assert render.separation(weighted_orders) < 1e-12
    assert render.separation(positional_orders) > 1e-2
    # With all weights alike the two schemes agree.
    assert render.separation(stack([surfaces[0], core.unit(surfaces[0])])) < 1e-12


if __name__ == "__main__":
    main()
