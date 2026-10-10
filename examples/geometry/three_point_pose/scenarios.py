"""Locate a rigid marker triangle from its three measured positions."""

import numpy as np

from examples.geometry.three_point_pose import core

STEPS = 24
DURATION_MS = 65


# --- math -----------------------------------------------------------------------------
def markers() -> tuple[core.Point, core.Point, core.Motor]:
    source = core.point([[-0.8, -0.4, 0.0], [0.9, -0.3, 0.0], [-0.3, 0.8, 0.2]])
    displacement = core.mv.xw * 0.7 + core.mv.yw * 0.2 + core.mv.zw * 0.4
    rotation = core.mv.xy * 0.6 + core.mv.yz * 0.2
    pose = displacement.exp() * rotation.exp()
    target = pose >> source
    return source, target, core.reconstruct(source, target)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.geometry.three_point_pose import render

    source, target, increments = markers()
    fractions = np.linspace(0, 1, STEPS + 1)[1:]
    fractions = fractions * fractions * (3 - 2 * fractions)
    save_figure(render.draw_stages(source, target, core.placements(source, increments)), "three_point_pose")
    save_animation(render.animate(core.alignments(source, increments, fractions), source, target),
                   "three_point_pose", DURATION_MS)


if __name__ == "__main__":
    main()
