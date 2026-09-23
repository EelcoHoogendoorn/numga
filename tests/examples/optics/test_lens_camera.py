"""The lens camera stills, with their checks, and a short zoom animation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from examples.optics.lens_camera import render, scenarios


def test_mathematics_does_not_import_plotting():
    """The math layer must stay free of the plotting stack, transitively."""
    probe = (
        "import examples.optics.lens_camera.core as c, sys; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')]; "
        "print(bad)"
    )
    root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": str(root)}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == "[]", f"plotting reached the math layer: {out.stdout}"


def test_figure_and_animation_draw():
    """The stills run their checks; the figure and a short animation draw."""
    assert isinstance(render.draw_stills(scenarios.stills()), plt.Figure)
    frames = render.animate_camera(scenarios.motion(2), scenarios.SCENE)
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
