"""The planes across the cube's rotation axes on the unit sphere."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.conformal_elliptical import render
from examples.quadrics.conformal_elliptical.core import mv, planes


def test_thirteen_distinct_unit_planes_closed_under_the_cubes_rotations():
    """The planes are unit, no two of them equal up to sense, and every rotation of the cube permutes them."""
    assert planes.shape == (13,)
    np.testing.assert_allclose(planes.scalar_norm_squared().to_array(), 1.0, atol=1e-12)
    alignment = (planes[:, None] | planes).abs().to_array()
    assert (alignment[~np.eye(13, dtype=bool)] < 1.0 - 1e-6).all()
    third = ((mv.xy + mv.yz + mv.zx).normalized() * (np.pi / 3)).exp()
    turned = ((third >> planes)[:, None] | planes).abs().to_array()
    np.testing.assert_allclose(turned.max(axis=1), 1.0, atol=1e-12)


def test_figure_renders():
    assert isinstance(render.draw_planes(planes), plt.Figure)
