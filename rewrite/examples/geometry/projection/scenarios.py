"""Scenes and entry points for the projective camera example.

One function per figure. Each builds the concrete scene, hands it to the
mathematics in `core`, and hands the resulting geometry to `render`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D
from examples import PLOT_DIR
from examples.pga3d import direction, point

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Line = ga.gatype.antibivector()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Pseudoscalar = ga.gatype.pseudoscalar()
Camera = ga.gatype((Point, Point))
LineCamera = ga.gatype((Line, Line))
ShadowTrail = ga.gatype((Point, Point))
Correspondence = ga.gatype((Pseudoscalar, Point, Point))
origin = mv.zyx

# Which corner pairs of `cube` are joined by an edge.
CUBE_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def circle(n: int) -> Point:
    """Construct n points around the unit circle in the xy plane, centred on the origin."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return point(np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1))


def cube(size: float) -> Point:
    """Construct the eight corner points of an axis-aligned cube centred on the origin."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    return point(corners * (size / 2.0))


def projection_figure(plot_path: str = str(PLOT_DIR / "projection.png")) -> plt.Figure:
    """A lit body over the ground, and the same rig posed twice over one subject."""
    from examples.geometry.projection import core, render

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

    shadows = core.shadows(body, ground, point_light, sun, body[7], light_path)
    stereo = core.stereo(subject, origin, screen, rig_1, rig_2)

    return render.draw_projection(
        body, CUBE_EDGES, shadows, point_light, sun, stereo, rig_1, rig_2, plot_path
    )


def main(plot_path: str = str(PLOT_DIR / "projection.png")) -> plt.Figure:
    """Render the projective camera figure."""
    return projection_figure(plot_path)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
    plt.show()
