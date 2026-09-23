"""Pose filtering on the motor manifold in PGA2D."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.geometry.kalman import render, scenarios


def test_filter_beats_dead_reckoning_and_renders():
    """Sparse noisy pose readings keep the filtered path far closer to the truth than the
    dead-reckoned one, and the figure renders."""
    tracking = scenarios.tracking()
    *_, dead_error, filtered_error, _ = tracking
    assert filtered_error.mean(axis=0).to_array() < 0.25 * dead_error.mean(axis=0).to_array()

    figure = render.draw_tracking(*tracking)
    assert isinstance(figure, plt.Figure)


def test_mathematics_does_not_import_plotting():
    """The math layer stays free of the plotting stack, transitively."""
    import subprocess
    import sys

    probe = (
        "import examples.geometry.kalman.core, examples.geometry.kalman.scenarios, sys; "
        "print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"plotting reached the math layer: {out.stdout}"
