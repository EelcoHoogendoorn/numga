"""Folding a Kabuto helmet: the eleven-fold sequence of the origami example of ganja.js by Steven De Keninck,
https://github.com/enkimute/ganja.js/blob/master/examples/example_pga3d_origami.html."""

import numpy as np
from numga import stack

from examples.geometry.origami import core

N_HALF_STEPS = 8
DURATION_MS = 65
SIDE = 2.0
SHEET = core.mv.z
CREASE_TOLERANCE = SIDE * 1e-10
# Each corner flap carries both layers joined along the folded edge. The horns
# turn those paired layers together; the two brim folds use the front sheet.
SELECTIONS = ((0,), (0, 1), (1, 3), (0, 1, 3, 4), (2, 7), (0, 6),
              (0, 7), (4, 12), (8,), (8, 9), (20,))
# Ordinary folds pass halfway through their turn; stages five and six reopen.
PEAKS = np.pi * np.array([0.5, 0.5, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.5, 0.5, -0.5])
ENDS = np.pi * np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, -1])


# --- math -----------------------------------------------------------------------------
def kabuto() -> tuple[core.Paper, core.Plane]:
    corner = core.mv.point([-SIDE, SIDE, 0])
    quarter_turns = np.arange(4) * np.pi / 2
    corners = (core.mv.xy * (-quarter_turns / 2)).exp() >> corner
    upper_left, lower_left, lower_right, upper_right = corners
    centre = corners.mean(axis=0)
    top = (upper_right + upper_left) / 2
    left = (upper_left + lower_left) / 2
    quarter = (upper_left + centre) / 2
    upper_centre = (centre + top) / 2
    left_centre = (centre + left) / 2
    eighth = (centre + quarter) / 2

    # Point coincidences set the main flaps; line coincidences open the two horns.
    # The last two folds turn opposite layers around the same brim crease.
    planes = stack([
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
    offsets = core.mv(core.Direction, [[0, 0, 0]])
    return core.Paper(corners, np.array([len(corners)]), offsets, SHEET), planes


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.geometry.origami import render

    paper, planes = kabuto()
    states = tuple(core.folding(paper, planes, SELECTIONS, PEAKS, ENDS, N_HALF_STEPS, CREASE_TOLERANCE))
    save_figure(render.draw(states[-1]), "origami_kabuto")
    save_animation(render.animate(iter(states)), "origami_kabuto", DURATION_MS)


if __name__ == "__main__":
    main()
