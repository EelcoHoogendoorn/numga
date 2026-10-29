"""Penrose's eight-conic configuration deforming: three couplings swept around a loop, the eighth quadric
keeping its double contact with its three neighbours throughout."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.math.eight_conics import core


# --- math -----------------------------------------------------------------------------
def cube() -> tuple[core.Quadric, core.Plane, core.Point, np.ndarray]:
    """Seven compatible ellipsoids and their eighth `[frames, 8]`, the three face couplings swept around a loop
    a third of a turn apart."""
    n_frames = 216
    mean_couplings = np.array([0.30, -0.24, 0.27])
    coupling_amplitude = 0.25
    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    angles = np.arange(3) * (2 * np.pi / 3)
    couplings = mean_couplings + coupling_amplitude * np.cos(phases[:, None] + angles)
    offsets = np.array([0.20, 0.24, 0.16])
    # Strong contact planes make the eight quadrics distinct ellipsoids; stronger ones turn some into hyperboloids.
    plane_weight = 1.3
    semi_axes = stack((core.mv.x.dual(), core.mv.y.dual()))                 # [semi_axes] Point-
    center = core.mv.w.dual()                                                # [] Point-

    dual_sphere = ((semi_axes * (core.Plane & semi_axes)).sum(axis=0)
                   - center * (core.Plane & center))                         # [] Point- <- Plane+
    sphere = dual_sphere.inverse()                                           # [] Plane+ <- Point-
    normals = (core.mv.xy * (-angles / 2)).exp() >> core.mv.x               # [neighbours] Plane+
    contact_planes = (normals - core.mv.w * offsets) * plane_weight         # [neighbours] Plane+
    quadrics, edge_planes, completions = core.configuration(sphere, contact_planes, couplings)
    contacts = core.intersections(quadrics[..., core.edges[:, 0]], edge_planes)  # [frames, edges, 2] Point-
    return quadrics, edge_planes, contacts, core.edges


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.math.eight_conics import render

    duration_ms = 40
    # The edges joining the eighth quadric to its three neighbours.
    closing_edges = np.array([9, 10, 11])
    cube_quadrics, contact_planes, cube_contacts, edges = cube()
    frames = render.animate_eighth(cube_quadrics, contact_planes, cube_contacts, edges, closing_edges)
    save_animation(frames, "eight_conics_eighth", duration_ms)

    # checks
    # Every contact point lies on both quadrics of its edge, at every frame: each quadric pairs it with its
    # own polar plane to zero.
    for vertex in range(2):
        ends = cube_quadrics[..., edges[:, vertex]][..., None]               # [frames, edges, 1] Plane+ <- Point-
        np.testing.assert_allclose((ends(cube_contacts) & cube_contacts).to_array(), 0, atol=1e-9)


if __name__ == "__main__":
    main()
