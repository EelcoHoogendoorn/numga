"""Scenes for origami fold a square sheet of paper into a kabuto helmet through the eleven-stage sequence of the ganja.js example by Steven De Keninck, https://github.com/enkimute/ganja.js/blob/master/examples/example_pga3d_origami.html."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack

from examples.geometry.origami import core

n_half_steps = 13
frame_duration_ms = 40
side = 2
sheet = core.mv.z                                                               # [] Plane+
crease_tolerance = side * 1e-10
# Each corner flap carries both layers joined along the folded edge; the horns turn
# those paired layers together, while the two brim folds separate front from back.
selections = (
    (0,), (0, 1), (1, 3), (0, 1, 3, 4),
    (2, 7), (0, 6),
    (0, 7), (4, 12),
    (8,), (8, 9), (20,),
)
# Ordinary folds pass halfway through their turn; stages five and six fold flat and reopen.
peaks = np.pi * np.array([0.5, 0.5, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, -0.5])
ends = np.pi * np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, -1])


# --- math -----------------------------------------------------------------------------
def kabuto_recipe() -> tuple[core.PaperState, core.Plane]:
    """A square sheet of half-side `side` and eleven point- and line-bisector crease planes specify the kabuto helmet."""
    corner = core.mv.point([-side, side, 0])                                    # [] Point-
    quarter_turns = (core.mv.xy * (-np.arange(4) * np.pi / 4)).exp()            # [4] Motor+
    corners = quarter_turns >> corner                                           # [4] Point-
    upper_left, lower_left, lower_right, upper_right = corners                  # [] Point-, [] Point-, [] Point-, [] Point-
    centre = corners.mean(axis=0)                                               # [] Point-
    top = (upper_right + upper_left) / 2                                        # [] Point-
    left = (upper_left + lower_left) / 2                                        # [] Point-
    quarter = (upper_left + centre) / 2                                         # [] Point-
    upper_centre = (centre + top) / 2                                           # [] Point-
    left_centre = (centre + left) / 2                                           # [] Point-
    eighth = (centre + quarter) / 2                                             # [] Point-

    # Point coincidences set the main flaps; line coincidences open the two horns,
    # and the last two stages fold opposite layers around the same brim crease.
    creases = stack([                                                           # [11] Plane+
        core.point_bisector(upper_left, lower_right),
        core.point_bisector(upper_right, lower_right),
        core.point_bisector(lower_left, lower_right),
        core.point_bisector(centre, upper_left),
        core.point_bisector(top, centre),
        core.point_bisector(left, centre),
        core.line_bisector(left_centre & quarter, quarter & centre, quarter),
        core.line_bisector(upper_centre & quarter, quarter & centre, quarter),
        core.point_bisector(eighth, upper_left),
        core.point_bisector(centre, upper_left),
        core.point_bisector(centre, upper_left),
    ])
    offsets = core.mv.direction([[0, 0, 0]]).field()                            # Direction-[1]
    initial_state = (corners.field(), np.array([len(corners)]), offsets, sheet)
    return initial_state, creases


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.geometry.origami import render
    from examples.util.animation import save_animation, save_figure

    initial_state, creases = kabuto_recipe()
    kept_states: list[core.PaperState] = []

    def streaming_states() -> Iterator[core.PaperState]:
        for state in core.folding_states(
            initial_state, creases, selections, peaks, ends, n_half_steps, crease_tolerance,
        ):
            kept_states.append(state)
            yield state

    save_animation(render.folding_animation(streaming_states()), "origami_kabuto", frame_duration_ms)
    final_state = kept_states[-1]
    save_figure(render.paper_figure(final_state), "origami_kabuto")

    # checks
    final_points, _, _, final_sheet = final_state
    horn_tips = core.mv.point([[-1, 1 - np.sqrt(2), 0], [np.sqrt(2) - 1, 1, 0]])  # [2] Point-
    tip_distances = (final_points.batch()[:, None] & horn_tips).antinorm()      # [V, 2] Scalar+
    assert (tip_distances.min(axis=0) < 1e-6).all()
    assert ((final_sheet & final_points).abs() < 1e-11).all()


if __name__ == "__main__":
    main()
