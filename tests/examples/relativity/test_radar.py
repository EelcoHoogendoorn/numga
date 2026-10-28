"""Null propagation, reconstruction from clock readings, and covariance of the radar experiment."""

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples.relativity.radar import core, scenarios
from examples.util import animation

ATOL = 1e-9


@pytest.mark.parametrize("velocity", [-0.6, 0, 0.25, 0.85])
def test_echo_maps_and_reconstruction(velocity):
    distance = 2.3
    times = np.array([0.2, 0.6, 1.1])
    placement = (core.mv.xw * (distance / 2)).exp() * (core.mv.xt * (np.arctanh(velocity) / 2)).exp()
    target = placement >> core.mv.x
    emitted = core.origin + core.time * times
    reflection, echo = core.echo_maps(core.mv.x, target, core.outgoing, core.incoming, core.origin)
    reflected, received = reflection(emitted), echo(emitted)

    # Analytic round-trip times, independently of the homogeneous intersection construction.
    expected_stretch = (1 + velocity) / (1 - velocity)
    expected_return = 2 * distance / (1 - velocity) + expected_stretch * times
    np.testing.assert_allclose((core.mv.t & received).to_array(), expected_return, atol=ATOL, rtol=0)
    np.testing.assert_allclose((core.mv.x & reflected).to_array(),
                               distance + velocity * (core.mv.t & reflected).to_array(), atol=ATOL, rtol=0)
    for incidence in (target & reflected, core.mv.x & received):
        np.testing.assert_allclose(incidence.to_array(), 0, atol=ATOL)
    for rays in (emitted & reflected, reflected & received):
        np.testing.assert_allclose(rays.squared().to_array(), 0, atol=ATOL)
    assert np.all((core.mv.t & (reflected - emitted)).to_array() > 0)
    assert np.all((core.mv.t & (received - reflected)).to_array() > 0)

    located, simultaneous, radar_time, radar_range = core.reconstruct(
        emitted, received, core.mv.t, core.origin, core.time, core.across)
    np.testing.assert_allclose((located - reflected).kernel, 0, atol=ATOL)
    np.testing.assert_allclose((core.mv.t & (located - simultaneous)).to_array(), 0, atol=ATOL)
    np.testing.assert_allclose(radar_time.to_array(), (core.mv.t & reflected).to_array(), atol=ATOL)
    np.testing.assert_allclose(radar_range.to_array(), (core.mv.x & reflected).to_array(), atol=ATOL)
    stretch, inferred_velocity, initial_range = core.infer_motion(emitted, received, core.mv.t)
    np.testing.assert_allclose([stretch.to_array(), inferred_velocity.to_array(), initial_range.to_array()],
                               [expected_stretch, velocity, distance], atol=ATOL, rtol=0)


@pytest.mark.parametrize("mirrored", [False, True])
def test_common_frame_change_preserves_events_and_clock_readings(mirrored):
    _, emitted, reflected, received = scenarios.exchange()
    transform = (core.mv.xw * 0.4 - core.mv.tw * 0.3).exp() * (core.mv.xt * -0.4).exp()
    if mirrored:
        transform = transform * core.mv.x
    target_placement = core.mv.xw.exp() * (core.mv.xt * (np.arctanh(0.25) / 2)).exp()
    target = transform >> (target_placement >> core.mv.x)
    clock, reference = transform >> core.mv.t, transform >> core.origin
    sent = transform >> emitted
    reflection, echo = core.echo_maps(transform >> core.mv.x, target,
                                      transform >> core.outgoing, transform >> core.incoming, reference)
    hit, returned = reflection(sent), echo(sent)
    np.testing.assert_allclose((hit - (transform >> reflected)).kernel, 0, atol=ATOL)
    np.testing.assert_allclose((returned - (transform >> received)).kernel, 0, atol=ATOL)
    np.testing.assert_allclose((clock & returned).to_array(), (core.mv.t & received).to_array(), atol=ATOL)
    located, _, _, ranges = core.reconstruct(sent, returned, clock, reference,
                                            transform >> core.time, transform >> core.across)
    np.testing.assert_allclose((located - hit).kernel, 0, atol=ATOL)
    np.testing.assert_allclose(ranges.to_array(), (core.mv.x & reflected).to_array(), atol=ATOL)
    stretch, velocity, distance = core.infer_motion(sent, returned, clock)
    np.testing.assert_allclose([stretch.to_array(), velocity.to_array(), distance.to_array()],
                               [5 / 3, 0.25, 2], atol=ATOL, rtol=0)


def test_notebook_executes_and_renders(monkeypatch):
    stills, movies = [], []

    def capture_figure(figure, name):
        figure.canvas.draw()
        stills.append(np.asarray(figure.canvas.buffer_rgba()).copy())
        plt.close(figure)

    def capture_animation(sequence, name, duration_ms, **options):
        movies.append(list(sequence))

    monkeypatch.setattr(animation, "save_figure", capture_figure)
    monkeypatch.setattr(animation, "show_animation", capture_animation)
    notebook_path = Path(__file__).parents[3] / "examples/relativity/radar/radar.ipynb"
    notebook = json.loads(notebook_path.read_text())
    namespace = {"__name__": __name__}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = re.sub(r"(?m)^n_frames, duration_ms = .*", "n_frames, duration_ms = 3, 40", source)
        source = re.sub(r"(?m)^([ \t]*)%.*", r"\1pass", source)
        exec(compile(source, f"{notebook_path.name}:{cell['id']}", "exec"), namespace)

    assert len(stills) == 1
    assert len(movies) == 2
    for movie in movies:
        assert len(movie) == 3
        for frame in movie:
            assert frame.shape == movie[0].shape
            assert frame.dtype == np.uint8
        assert not np.array_equal(movie[0], movie[-1])
    np.testing.assert_allclose((namespace["located"] - namespace["reflected"]).kernel, 0, atol=ATOL)
    np.testing.assert_allclose(namespace["inferred_velocity"].to_array(), 0.25, atol=ATOL)
