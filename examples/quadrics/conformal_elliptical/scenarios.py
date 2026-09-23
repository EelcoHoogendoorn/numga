"""The octahedral mirror planes, checked against the rotations of the cube."""

from __future__ import annotations

import numpy as np

from examples.quadrics.conformal_elliptical.core import Vector, mv, octahedral_planes


def octahedral_sphere() -> Vector:
    planes = octahedral_planes()

    # --- checks ---------------------------------------------------------------------------
    # A quarter turn about an axis and a third turn about a body diagonal generate the
    # rotations of the cube; each carries every mirror plane onto a mirror plane.
    quarter = (mv.xy * (np.pi / 4)).exp()
    third = ((mv.xy + mv.yz + mv.zx).normalized() * (np.pi / 3)).exp()
    for rotor in (quarter, third):
        alignment = ((rotor >> planes)[:, None] | planes).abs().to_array()
        np.testing.assert_allclose(alignment.max(axis=1), 1.0, atol=1e-12)
    return planes


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.quadrics.conformal_elliptical import render

    save_figure(render.draw_octahedral_sphere(octahedral_sphere()), "sphere_rigid")
