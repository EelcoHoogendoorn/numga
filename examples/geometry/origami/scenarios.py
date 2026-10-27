"""Scenes for origami fold a square sheet of paper into a kabuto helmet through the eleven-stage sequence of the ganja.js example by Steven De Keninck, https://github.com/enkimute/ganja.js/blob/master/examples/example_pga3d_origami.html."""

import numpy as np
from numga import stack

from examples.geometry.origami import core

N_HALF_STEPS = 8
DURATION_MS = 65
SIDE = 2.0
SHEET = core.mv.z                                                               # [] Plane+
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
    """The kabuto helmet folds from a square sheet along eleven crease planes."""
    corner = core.mv.point([-SIDE, SIDE, 0])                                    # [] Point-
    quarter_turns = (core.mv.xy * (-np.arange(4) * np.pi / 4)).exp()            # [corners] Motor+
    corners = quarter_turns >> corner                                           # [corners] Point-
    upper_left, lower_left, lower_right, upper_right = corners                  # [] Point-, Point-, Point-, Point-
    centre = corners.mean(axis=0)                                               # [] Point-
    top = (upper_right + upper_left) / 2                                        # [] Point-
    left = (upper_left + lower_left) / 2                                        # [] Point-
    quarter = (upper_left + centre) / 2                                         # [] Point-
    upper_centre = (centre + top) / 2                                           # [] Point-
    left_centre = (centre + left) / 2                                           # [] Point-
    eighth = (centre + quarter) / 2                                             # [] Point-

    # Point coincidences set the main flaps; line coincidences open the two horns.
    # The last two folds turn opposite layers around the same brim crease.
    creases = stack([                                                           # [stages] Plane+
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
    offsets = core.mv(core.Direction, [[0, 0, 0]])                              # [faces] Direction-
    return core.Paper(corners, np.array([len(corners)]), offsets, SHEET), creases


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.geometry.origami import render

    paper, creases = kabuto()
    states = tuple(core.folding(paper, creases, SELECTIONS, PEAKS, ENDS, N_HALF_STEPS, CREASE_TOLERANCE))
    save_figure(render.draw(states[-1]), "origami_kabuto")
    save_animation(render.animate(iter(states)), "origami_kabuto", DURATION_MS)

    # checks
    tips = core.mv.point([[-1, 1 - np.sqrt(2), 0], [np.sqrt(2) - 1, 1, 0]])
    squared_distance = (states[-1].points[:, None] - tips).dual().scalar_norm_squared()
    np.testing.assert_allclose(squared_distance.kernel.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose((states[-1].sheet & states[-1].points).kernel, 0.0, atol=1e-11)


if __name__ == "__main__":
    main()
