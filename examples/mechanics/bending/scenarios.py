"""Thin-walled beam bending and twisting scenes compare an I-section, a turned I-section, a Z-section,
and a slotted tube under vertical tip loads through their centroids, a rotating tip load on the
Z-section, a translating tip load passing through the slotted tube's shear centre, and eccentric axial
pushes inside and outside each section's kern."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from numga import stack

from examples.mechanics.bending import core
from examples.mechanics.bending.mesh import section_names, section_walls

mv = core.mv
# Upward direction in the cross-section plane.
up_direction = mv.y.dual()                                                  # [] Point-
# Material moduli, cantilever geometry, tip-load magnitude, and animation timing.
modulus = 1.0
poisson_ratio = 0.3
shear_modulus = modulus / (2 * (1 + poisson_ratio))
beam_length = 10.0
n_pieces = 40
tip_load = 1e-5
n_frames = 72
frame_duration_ms = 40
# Start and end coordinates `(x, y)` of the slotted tube's translating tip load.
across_endpoints = [[0.4, 0.0], [-1.15, 0.0]]
# Eccentric axial push magnitude and `(x, y)` points inside and outside every section's kern.
axial_push = 1e-3
push_coordinates = [[0.03, 0.04], [0.1, 0.3]]
gallery_dpi = 64


# --- plumbing -------------------------------------------------------------------------
def streamed(states: Iterable, sink: list) -> Iterator:
    """Yields items from `states` once while appending each item to `sink`."""
    for state in states:
        sink.append(state)
        yield state


def main() -> None:
    """Renders all section stills and cantilever load animations to `PLOT_DIR` and checks the computed
    deflections, twist rates, kerns, and Betti reciprocity."""
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.bending import render

    sections = section_walls()
    down_load = tip_load * -up_direction                                    # [] Point-
    centres = [core.centroid(topology.incidence[1, 0] * nodes, thickness)
               for topology, nodes, thickness in sections]
    shear_centres = [core.shear_centre(*section) for section in sections]
    beams = [core.cantilever(*section, core.tip_points(centre, beam_length) & down_load,
                             beam_length, modulus, shear_modulus, n_pieces)
             for section, centre in zip(sections, centres)]
    push_points = core.section_point(np.array(push_coordinates))            # [2] Point-
    pushes = [core.axial_pushes(*section, push_points, modulus, axial_push) for section in sections]

    z_index = section_names.index("Z-section")
    turned_history: list = []
    turning = core.turning_states(*sections[z_index], centres[z_index], down_load,
                                  beam_length, modulus, shear_modulus, n_pieces, n_frames)

    tube_index = section_names.index("slotted tube")
    start_point, end_point = core.section_point(np.array(across_endpoints))  # [] Point-, [] Point-
    crossed_history: list = []
    crossing = core.crossing_states(*sections[tube_index], start_point, end_point, down_load,
                                    beam_length, modulus, shear_modulus, n_pieces, n_frames)

    save_figure(render.draw_sections(sections), "bending_cross_sections")
    save_figure(render.draw_walls(sections, centres, shear_centres), "bending_walls")
    save_figure(render.draw_beams(sections, centres, shear_centres, down_load, beams), "bending_sections")
    save_figure(render.draw_pushes(sections, centres, push_points, pushes), "bending_kern")
    save_animation(render.animate_loads(sections[z_index], centres[z_index], shear_centres[z_index],
                                        streamed(turning, turned_history)),
                   "bending_turning", frame_duration_ms)
    save_animation(render.animate_loads(sections[tube_index], centres[tube_index], shear_centres[tube_index],
                                        streamed(crossing, crossed_history)),
                   "bending_across", frame_duration_ms)

    # checks
    displaced_centres = stack([frames[-1] >> centre for centre, (_, frames, _, _) in zip(centres, beams)])  # [4] Point-
    rest_centres = stack(centres)                                           # [4] Point-
    sideways = (mv.x & displaced_centres) / (mv.w & displaced_centres) - (mv.x & rest_centres) / (mv.w & rest_centres)  # [4] Scalar+
    falls = (mv.y & displaced_centres) / (mv.w & displaced_centres) - (mv.y & rest_centres) / (mv.w & rest_centres)  # [4] Scalar+
    assert (falls < 0).all() and bool(sideways[0].abs() < 1e-12)
    assert bool(sideways[1:3].abs().min() > 0.5 * falls[1:3].abs().max())

    # Translating the downward load across the slotted tube reverses the twist across the shear centre.
    crossed_points = stack([state[0] for state in crossed_history])         # [n_frames] Point-
    crossed_strains = stack([state[2][0] for state in crossed_history])     # [n_frames] Twist+
    twist_rates = crossed_strains & core.shift_along_beam                   # [n_frames] Scalar+
    assert bool(twist_rates[0] * twist_rates[-1] < 0)
    shear_offsets = ((mv.x & crossed_points) / (mv.w & crossed_points)
                     - (mv.x & shear_centres[tube_index]) / (mv.w & shear_centres[tube_index])).abs()  # [n_frames] Scalar+
    passing_frame = int(shear_offsets.argmin())
    assert bool(twist_rates[passing_frame].abs() < 0.1 * twist_rates.abs().max())

    for push_strains, push_stresses, polars in pushes:
        # Pushing inside the kern compresses every wall end; pushing outside puts some wall end in tension.
        assert (push_stresses[0] < 0).all() and (push_stresses[1] > 0).any()
        # Betti reciprocity holds on the two push strains already solved: each stretches the other's fibre equally.
        reciprocity_gap = ((push_strains[0] & (core.beam_direction & push_points[1]))
                           - (push_strains[1] & (core.beam_direction & push_points[0])))  # [] Scalar+
        assert bool(reciprocity_gap.abs() < 1e-12)


def pinned() -> None:
    """Saves and pins `bending.gif` showing the slotted tube twisting one way and then the other as
    its tip load translates across the shear centre."""
    from examples.util.animation import pin_animation
    from examples.mechanics.bending import render

    sections = section_walls()
    tube_index = section_names.index("slotted tube")
    tube = sections[tube_index]
    topology, nodes, thickness = tube
    centre = core.centroid(topology.incidence[1, 0] * nodes, thickness)     # [] Point-
    shear = core.shear_centre(*tube)                                        # [] Point-
    down_load = tip_load * -up_direction                                    # [] Point-
    start_point, end_point = core.section_point(np.array(across_endpoints))  # [] Point-, [] Point-
    crossing = core.crossing_states(*tube, start_point, end_point, down_load,
                                    beam_length, modulus, shear_modulus, n_pieces, n_frames)
    pin_animation(render.sized_loads(tube, centre, shear, gallery_dpi, crossing), "bending",
                  frame_duration_ms, forward_then_back=True)


if __name__ == "__main__":
    main()
