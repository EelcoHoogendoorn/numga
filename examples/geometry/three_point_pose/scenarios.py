"""Scenes for three-point pose reconstruction align a displaced marker triangle through its three incremental motors."""

import numpy as np

from examples.geometry.three_point_pose import core

n_steps = 39
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def marker_scene() -> tuple[core.Point, core.Point, core.Motor]:
    """A rigid motion displaces three `source` markers (`[3] Point-`) to `target` (`[3] Point-`), from which `alignment_motors` recovers the three stage motors (`[3] Motor+`)."""
    source = core.mv.point([[-0.8, -0.4, 0.0], [0.9, -0.3, 0.0], [-0.3, 0.8, 0.2]])  # [3] Point-
    displacement = core.mv.xw * 0.7 + core.mv.yw * 0.2 + core.mv.zw * 0.4              # [] Bivector+
    rotation = core.mv.xy * 0.6 + core.mv.yz * 0.2                                     # [] Bivector+
    pose = displacement.exp() * rotation.exp()                                         # [] Motor+
    target = pose >> source                                                            # [3] Point-
    return source, target, core.alignment_motors(source, target)                       # [3] Point-, [3] Point-, [3] Motor+


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Saves the three-stage alignment animation and checks that the final placement matches the target markers."""
    from examples.geometry.three_point_pose import render
    from examples.util.animation import save_animation

    source, target, increments = marker_scene()
    fractions = np.linspace(0, 1, n_steps + 1)[1:]
    fractions = fractions * fractions * (3 - 2 * fractions)
    states = list(core.alignments(source, increments, fractions))
    save_animation(render.pose_animation(source, target, iter(states)), "three_point_pose", frame_duration_ms)

    # checks
    assert ((states[-1] & target).norm() < 1e-10).all()


if __name__ == "__main__":
    main()
