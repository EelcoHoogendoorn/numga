"""Run the comparison notebook through its figures, animation and mathematical checks."""

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from examples.util import animation


def test_comparison_notebook_renders_both_formulations(monkeypatch):
    stills, frames = [], []

    def capture_figure(figure, name):
        stills.append(animation.capture(figure))
        plt.close(figure)

    def capture_animation(sequence, name, duration_ms):
        frames.extend(sequence)

    monkeypatch.setattr(animation, "save_figure", capture_figure)
    monkeypatch.setattr(animation, "show_animation", capture_animation)
    notebook_path = Path(__file__).parents[3] / "examples/relativity/kerr/kerr_comparison.ipynb"
    notebook = json.loads(notebook_path.read_text())
    namespace = {"__name__": __name__}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Keep the notebook's geometry, transport and checks, with a smaller camera image.
        source = re.sub(r"(?m)^image_width = .*", "image_width = 12", source)
        # Notebook setup magics do not apply to the installed, headless test environment.
        source = re.sub(r"(?m)^([ \t]*)%.*", r"\1pass", source)
        exec(compile(source, f"{notebook_path.name}:{cell['id']}", "exec"), namespace)

    expected_shape = (namespace["image_height"], 2 * namespace["image_width"], 3)
    assert len(stills) == 2
    assert stills[-1].shape == expected_shape
    assert len(frames) == namespace["n_polarizer_frames"]
    for frame in frames:
        assert frame.shape == expected_shape
        assert frame.dtype == np.uint8
    assert np.isfinite(list(namespace["readout"].values())).all()
