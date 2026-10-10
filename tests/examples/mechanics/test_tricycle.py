"""The tricycle drives the same track in the plane and in space, passes its checks in both, and draws
in both views."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from examples.mechanics.tricycle import render, scenarios


def test_the_plane_and_space_drive_the_same_track_and_draw():
    plane, space = (scenarios.run(core) for core in scenarios.SCENES.values())
    np.testing.assert_allclose(render.ground(plane[3]), render.ground(space[3]), atol=1e-9)
    for drive, draw in ((plane, render.plane), (space, render.space)):
        frames = render.animate(draw, scenarios.moments(*drive, 2))
        assert len(frames) == 2
    plt.close("all")
