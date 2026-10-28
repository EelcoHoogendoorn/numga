"""Null propagation, frame changes, clock recovery, and predicted navigation uncertainty."""

import numpy as np
import pytest

from numga import stack

from examples.relativity.gps import core, scenarios

# A real receiver's range error, in Earth radii: small enough for the linearised covariance to hold.
SIGMA = 3 / scenarios.EARTH_RADIUS_METRES


@pytest.mark.parametrize("phase_degrees", [5, 20, 50])
def test_light_cones_recover_the_future_reception_event(phase_degrees):
    phases = core.mv.scalar((scenarios.PHASE_OFFSETS + np.deg2rad(phase_degrees))[..., None])
    planes, radius, angular_speed, receiver = scenarios.planes, scenarios.ORBIT_RADIUS, scenarios.ANGULAR_SPEED, scenarios.receiver
    emitted = core.emitted_events(receiver, planes, phases, radius, angular_speed, 3)
    cones = core.light_cones(emitted)

    np.testing.assert_allclose(cones(emitted).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((receiver & cones(receiver)).kernel, 0, atol=1e-12)
    # The joined signal lines are null under the algebra's mixed signature too.
    np.testing.assert_allclose((emitted & receiver).squared().kernel, 0, atol=1e-12)

    candidates = core.intersect_cones(cones[:4])
    np.testing.assert_allclose(
        (candidates[:, None] & cones[:4](candidates[:, None])).kernel, 0, atol=1e-11)
    located = core.reception(candidates, emitted[:4])
    np.testing.assert_allclose((located - receiver).kernel, 0, atol=1e-11)
    assert np.all((core.mv.t & (located - emitted)).kernel > 0)


def test_light_cones_and_their_intersections_follow_a_common_frame_change():
    phases = core.mv.scalar((scenarios.PHASE_OFFSETS + np.deg2rad(20))[..., None])
    planes, radius, angular_speed, receiver = scenarios.planes, scenarios.ORBIT_RADIUS, scenarios.ANGULAR_SPEED, scenarios.receiver
    emitted = core.emitted_events(receiver, planes, phases, radius, angular_speed, 3)
    cones = core.light_cones(emitted)
    transform = (core.mv.xw * 0.3 - core.mv.tw * 0.15).exp() * (
        core.mv.xt * 0.2 + core.mv.yz * 0.1).exp()
    moved_emissions = transform >> emitted
    moved_receiver = transform >> receiver

    transported = transform >> cones(transform << core.Event)
    rebuilt = core.light_cones(moved_emissions)
    np.testing.assert_allclose((transported - rebuilt).kernel, 0, atol=1e-12)
    candidates = core.intersect_cones(rebuilt[:4])
    # Reception after transmission is the same in every frame a forward-in-time change reaches.
    nearest = core.reception(candidates, moved_emissions[:4])
    np.testing.assert_allclose((nearest - moved_receiver).kernel, 0, atol=1e-11)

    moved_clock = transform >> core.mv.t
    np.testing.assert_allclose(
        (moved_clock & (nearest - moved_emissions)).kernel,
        (core.mv.t & (receiver - emitted)).kernel, atol=1e-11, rtol=0)
    assert np.all((moved_clock & (nearest - moved_emissions)).kernel > 0)


@pytest.mark.parametrize("phase_degrees,reception_time", [(5, -3), (20, 0), (50, 4)])
def test_noiseless_fit_recovers_position_and_receiver_clock_error(phase_degrees, reception_time):
    phases = core.mv.scalar((scenarios.PHASE_OFFSETS + np.deg2rad(phase_degrees))[..., None])
    planes, radius, angular_speed, receiver, sigma, clock_error = scenarios.planes, scenarios.ORBIT_RADIUS, scenarios.ANGULAR_SPEED, scenarios.receiver, SIGMA, scenarios.clock_reading
    receiver = receiver + core.time * reception_time
    clock_reading = (core.mv.t & receiver) + clock_error
    emitted = core.emitted_events(receiver, planes, phases, radius, angular_speed, 3)
    initial = receiver + core.mv.y.dual() * 0.02 + core.mv.z.dual() * 0.01 + core.time * clock_error
    estimate, information, covariance = core.locate(emitted, initial, sigma, 5)

    np.testing.assert_allclose((estimate - receiver).kernel, 0, atol=1e-12)
    recovered_clock_error = clock_reading - (core.mv.t & estimate)
    np.testing.assert_allclose((recovered_clock_error - clock_error).kernel, 0, atol=1e-12)
    assert np.all((core.mv.t & (estimate - emitted)).kernel > 0)

    # Greater timestamp uncertainty changes the uncertainty of the answer, not its location.
    same_estimate, _, larger_covariance = core.locate(emitted, initial, sigma * 2, 5)
    np.testing.assert_allclose((same_estimate - estimate).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(larger_covariance.kernel, 4 * covariance.kernel, rtol=1e-9, atol=1e-24)

    directions = stack((core.across, core.mv.y.dual(), core.mv.z.dual(), core.time))
    assert np.all((directions & information(directions)).kernel > 0)
    np.testing.assert_allclose((covariance - covariance.adjugate()).kernel, 0, atol=1e-23)


def test_cone_readouts_predict_changes_of_the_intervals_under_event_displacements():
    phases = core.mv.scalar((scenarios.PHASE_OFFSETS + np.deg2rad(20))[..., None])
    planes, radius, angular_speed, receiver = scenarios.planes, scenarios.ORBIT_RADIUS, scenarios.ANGULAR_SPEED, scenarios.receiver
    emitted = core.emitted_events(receiver, planes, phases, radius, angular_speed, 3)
    cones = core.light_cones(emitted)
    trial = receiver + core.mv.z.dual() * 0.12 + core.time * 0.03
    _, readout, _ = core.intervals(cones, emitted, trial)
    directions = stack((core.across, core.mv.y.dual(), core.mv.z.dual(), core.time))
    epsilon = 1e-4
    plus, _, _ = core.intervals(cones, emitted, trial + directions * epsilon)
    minus, _, _ = core.intervals(cones, emitted, trial - directions * epsilon)
    measured = (plus - minus) / (2 * epsilon)
    predicted = readout & directions[:, None]
    np.testing.assert_allclose((measured - predicted).kernel, 0, atol=1e-8)


def test_timestamp_noise_matches_the_predicted_event_covariance():
    phases = core.mv.scalar((scenarios.PHASE_OFFSETS + np.deg2rad(20))[..., None])
    planes, radius, angular_speed, receiver, sigma, clock_reading = scenarios.planes, scenarios.ORBIT_RADIUS, scenarios.ANGULAR_SPEED, scenarios.receiver, SIGMA, scenarios.clock_reading
    emitted = core.emitted_events(receiver, planes, phases, radius, angular_speed, 3)
    n_trials, n_satellites = 512, emitted.shape[-1]
    noise = core.mv.scalar(np.random.default_rng(2026).normal(size=(n_trials, n_satellites, 1)) * sigma)
    observed = emitted + core.time * noise
    initial = receiver + core.mv.y.dual() * 0.02 + core.mv.z.dual() * 0.01 + core.time * clock_reading
    estimates, _, _ = core.locate(observed, initial, sigma, 5)
    _, _, covariance = core.locate(emitted, initial, sigma, 5)

    # Mixed readouts also test position-clock and position-position covariances.
    probes = stack((core.mv.x, core.mv.y, core.mv.z, core.mv.t,
                    core.mv.x + core.mv.t, core.mv.y + core.mv.z,
                    core.mv.x + core.mv.y - core.mv.t))
    errors = (estimates - receiver)[:, None] & probes
    mean = errors.sum(axis=0) / n_trials
    variance = ((errors - mean) * (errors - mean)).sum(axis=0) / (n_trials - 1)
    predicted = probes & covariance(probes)
    np.testing.assert_allclose((variance / predicted).kernel, 1, atol=0.3, rtol=0)
    np.testing.assert_allclose((mean / predicted.square_root()).kernel, 0, atol=0.2)


def test_notebook_executes_and_renders(monkeypatch):
    import json
    from pathlib import Path
    import re

    import matplotlib.pyplot as plt

    from examples.util import animation

    stills, movies = [], []

    def capture_figure(figure, name):
        figure.canvas.draw()
        stills.append(np.asarray(figure.canvas.buffer_rgba()).copy())
        plt.close(figure)

    def capture_animation(sequence, name, duration_ms, **options):
        movies.append(list(sequence))

    monkeypatch.setattr(animation, "save_figure", capture_figure)
    monkeypatch.setattr(animation, "show_animation", capture_animation)
    notebook_path = Path(__file__).parents[3] / "examples/relativity/gps/gps.ipynb"
    notebook = json.loads(notebook_path.read_text())
    namespace = {"__name__": __name__}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = re.sub(r"(?m)^n_frames, duration_ms = .*", "n_frames, duration_ms = 3, 65", source)
        source = re.sub(r"(?m)^([ \t]*)%.*", r"\1pass", source)
        exec(compile(source, f"{notebook_path.name}:{cell['id']}", "exec"), namespace)

    assert len(stills) == 2
    assert len(movies) == 1
    movie = movies[0]
    assert len(movie) == 3
    for frame in movie:
        assert frame.shape == movie[0].shape
        assert frame.dtype == np.uint8
    assert not np.array_equal(movie[0], movie[-1])
    np.testing.assert_allclose((namespace["located"] - namespace["receiver"]).kernel, 0, atol=1e-11)
