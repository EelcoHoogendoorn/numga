"""The thin-lens figures, and the optical train's imaging."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.optics.thin_lens import render, scenarios


def test_mathematics_does_not_import_plotting():
    """The math layer must stay free of the plotting stack, transitively."""
    probe = (
        "import examples.optics.thin_lens.core as c, sys; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')]; "
        "print(bad)"
    )
    root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": str(root)}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == "[]", f"plotting reached the math layer: {out.stdout}"


def test_train_images_the_subject_through_every_placement():
    """The composed train maps the fan onto the last leg, and every output ray passes through the image."""
    for subject, planes, legs, train, picture in scenarios.train(12):
        np.testing.assert_allclose((train(legs[0]) ^ picture).kernel, 0.0, atol=1e-10)
        np.testing.assert_allclose((train(legs[0]) - legs[-1]).kernel, 0.0, atol=1e-10)


def test_figure_and_animation_draw():
    assert isinstance(render.draw_lenses(*scenarios.lenses()), plt.Figure)
    frames = render.animate_train(scenarios.train(4))
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
