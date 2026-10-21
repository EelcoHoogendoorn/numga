"""Penrose's eight-conic configuration deforming: three couplings swept around a loop, the eighth conic
keeping its double contact with its three neighbours throughout."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.math.eight_conics import core


# --- math -----------------------------------------------------------------------------
def cube() -> tuple[core.Conic, core.Plane, core.Point, np.ndarray]:
    """Seven compatible ellipses and their eighth `[frames, 8]`, the three face couplings swept around a loop
    a third of a turn apart."""
    n_frames = 96
    mean_couplings = np.array([0.30, -0.24, 0.27])
    coupling_amplitude = 0.25
    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    angles = np.arange(3) * (2 * np.pi / 3)
    couplings = mean_couplings + coupling_amplitude * np.cos(phases[:, None] + angles)
    offsets = np.array([0.20, 0.24, 0.16])
    # Strong chords make the eight conics distinct ellipses; stronger ones turn some into hyperbolas.
    chord_weight = 1.3
    semi_axes = stack((core.mv.x.dual(), core.mv.y.dual()))                     # [2] Point
    center = core.mv.w.dual()                                                # [] Point

    dual_circle = ((semi_axes * (core.Plane & semi_axes)).sum(axis=0)
                   - center * (core.Plane & center))                         # [] Point <- Plane
    circle = dual_circle.inverse()                                          # [] Plane <- Point
    normals = (core.mv.xy * (-angles / 2)).exp() >> core.mv.x                  # [3] Plane
    chords = (normals - core.mv.w * offsets) * chord_weight                   # [3] Plane
    conics, edge_chords, completions = core.configuration(circle, chords, couplings)
    contacts = core.intersections(conics[..., core.EDGES[:, 0]], edge_chords)  # [frames, edges, 2] Point
    return conics, edge_chords, contacts, core.EDGES


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.math.eight_conics import render

    duration_ms = 90
    # The edges joining the eighth conic to its three neighbours.
    closing_edges = np.array([9, 10, 11])
    cube_conics, chords, cube_contacts, edges = cube()
    save_animation(render.animate_eighth(cube_conics, chords, cube_contacts, edges, closing_edges), "eight_conics_eighth",
                   duration_ms)

    # checks
    # Every contact point lies on both conics of its edge, at every frame: each conic pairs it with its
    # own polar line to zero.
    for vertex in range(2):
        ends = cube_conics[..., edges[:, vertex]][..., None]                  # [frames, edges, 1] Conic
        np.testing.assert_allclose((ends(cube_contacts) & cube_contacts).to_array(), 0, atol=1e-9)


if __name__ == "__main__":
    main()
