"""The planes across the cube's rotation axes, checked against the cube's rotations."""

from __future__ import annotations

import numpy as np

from examples.quadrics.conformal_elliptical.core import mv, planes


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.quadrics.conformal_elliptical import render

    save_figure(render.draw_planes(planes), "cube_axis_planes")

    # checks
    # A quarter turn about a face axis and a third of a turn about a corner axis generate the rotations of
    # the cube; each carries every plane onto one of the planes, up to its sense.
    quarter = (mv.xy * (np.pi / 4)).exp()
    third = ((mv.xy + mv.yz + mv.zx).normalized() * (np.pi / 3)).exp()
    for rotor in (quarter, third):
        alignment = ((rotor >> planes)[:, None] | planes).abs().to_array()
        np.testing.assert_allclose(alignment.max(axis=1), 1.0, atol=1e-12)


if __name__ == "__main__":
    main()
