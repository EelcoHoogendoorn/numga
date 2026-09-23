"""Scenes for the projective camera example.

One function per figure. Each builds the concrete scene, hands it to the mathematics
in `core`, and returns the resulting geometry for `render`.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from examples.geometry.projection.core import CUBE_EDGES, circle, cube, direction, ga, point, shadows, stereo

ctx = NumpyContext(ga)
mv = ctx.multivector
origin = mv.zyx


def projection():
    """A lit body over the ground, and the same rig posed twice over one subject."""
    body = (mv.zw * 0.75).exp() >> cube(1.0)
    ground = mv.z
    point_light = point(np.array([1.0, -1.0, 4.0]))
    sun = direction(np.array([-1.0, 0.6, -2.5]))
    light_path = (mv.yw * -0.5 + mv.zw * 2.0).exp() >> circle(24)

    # Two copies of the rig, each translated 0.6 along x and turned 0.12 rad about the y
    # axis so they converge.
    screen = mv.z - mv.w
    rig_1 = (mv.xw * -0.3).exp() * (mv.xz * +0.06).exp()
    rig_2 = (mv.xw * +0.3).exp() * (mv.xz * -0.06).exp()
    subject = (mv.zw * 2.5).exp() >> cube(1.6)

    cast = shadows(body, ground, point_light, sun, body[7], light_path)
    views = stereo(subject, origin, screen, rig_1, rig_2)
    return body, CUBE_EDGES, cast, point_light, sun, views, rig_1, rig_2


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.geometry.projection import render

    save_figure(render.draw_projection(*projection()), "projection")
