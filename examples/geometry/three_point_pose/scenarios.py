"""Scenes for pose from three points align a displaced marker triangle through its three incremental motors."""

import numpy as np

from examples.geometry.three_point_pose import core

N_STEPS = 39
DURATION_MS = 40


# --- math -----------------------------------------------------------------------------
def markers() -> tuple[core.Point, core.Point, core.Motor]:
    """The source marker triple, its displaced target triple, and the three incremental alignment motors."""
    source = core.mv.point([[-0.8, -0.4, 0.0], [0.9, -0.3, 0.0], [-0.3, 0.8, 0.2]])  # [markers] Point-
    displacement = core.mv.xw * 0.7 + core.mv.yw * 0.2 + core.mv.zw * 0.4              # [] Bivector+
    rotation = core.mv.xy * 0.6 + core.mv.yz * 0.2                                     # [] Bivector+
    pose = displacement.exp() * rotation.exp()                                         # [] Motor+
    target = pose >> source                                                            # [markers] Point-
    return source, target, core.reconstruct(source, target)                            # [markers] Point-, [markers] Point-, [stages] Motor+


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.geometry.three_point_pose import render

    source, target, increments = markers()
    fractions = np.linspace(0, 1, N_STEPS + 1)[1:]
    fractions = fractions * fractions * (3 - 2 * fractions)
    frames = list(core.alignments(source, increments, fractions))
    save_animation(render.animate(iter(frames), source, target), "three_point_pose", DURATION_MS)

    # checks
    np.testing.assert_allclose((frames[-1] - target).kernel, 0.0, atol=1e-12)


if __name__ == "__main__":
    main()
