"""Tests for null light-cone intersections, Poincaré equivariance, Gauss-Newton clock/position recovery, and GPS rendering."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import dim, stack

from examples.relativity.gps import core, render, scenarios

# Realistic 3 m range noise in Earth radii so linearized covariance predictions stay accurate.
realistic_sigma = core.mv.scalar(3 / scenarios.earth_radius_metres, unit=dim.length)
readouts = stack((core.mv.x, core.mv.y, core.mv.z, core.mv.t))
probe_events = stack((
    core.origin,
    core.across * dim.length,
    core.mv.y.dual() * dim.length,
    core.mv.z.dual() * dim.length,
    core.time * dim.length,
))


def _assert_event_close(difference: core.Event, tol: float) -> None:
    """Assert weight error is zero and spacetime coordinate errors fall below `tol` in `dim.length`."""
    assert ((core.mv.w & difference).abs() < tol).all()
    assert ((readouts & difference[..., None]).abs() < core.mv.scalar(tol, unit=dim.length)).all()


@pytest.mark.parametrize("phase_degrees", [5, 20, 50])
def test_light_cones_recover_future_reception_event(phase_degrees):
    phases = scenarios.phase_offsets + np.deg2rad(phase_degrees)
    planes = scenarios.orbital_planes
    radius, angular_speed, receiver = scenarios.orbit_radius, scenarios.angular_speed, scenarios.receiver
    emitted = core.emitted_events(receiver, planes, phases, radius, angular_speed, 3)
    cones = core.light_cones(emitted)                                             # [6] Slice+ <- Event-

    assert ((cones(emitted)[..., None] & probe_events).abs() < core.mv.scalar(1e-12, unit=dim.length**2)).all()
    assert ((cones(receiver) & receiver).abs() < core.mv.scalar(1e-12, unit=dim.length**2)).all()
    assert ((emitted & receiver).scalar_norm_squared().abs() < core.mv.scalar(1e-12, unit=dim.length**2)).all()

    candidates = core.cone_intersections(cones[:4])                               # [2] Event-
    assert ((cones[:4](candidates[:, None]) & candidates[:, None]).abs() < core.mv.scalar(1e-11, unit=dim.length**2)).all()
    located = core.causal_reception(candidates, emitted[:4])                      # [] Event-
    _assert_event_close(located - receiver, 1e-11)
    assert ((core.mv.t & (located - emitted)) > 0).all()


def test_light_cones_and_intersections_follow_poincare_transform():
    phases = scenarios.phase_offsets + np.deg2rad(20)
    planes = scenarios.orbital_planes
    radius, angular_speed, receiver = scenarios.orbit_radius, scenarios.angular_speed, scenarios.receiver
    emitted = core.emitted_events(receiver, planes, phases, radius, angular_speed, 3)
    cones = core.light_cones(emitted)
    shift = ((core.mv.xw * 0.3 - core.mv.tw * 0.15) * dim.length).exp()
    lorentz = (core.mv.xt * 0.2 + core.mv.yz * 0.1).exp()
    transform = shift * lorentz                                                   # [] Motor+
    moved_emissions = transform >> emitted
    moved_receiver = transform >> receiver

    transported = transform >> cones(transform << core.Event)
    rebuilt = core.light_cones(moved_emissions)
    assert (((transported - rebuilt)(probe_events[:, None])[..., None] & probe_events).abs() < core.mv.scalar(1e-12, unit=dim.length**2)).all()

    candidates = core.cone_intersections(rebuilt[:4])
    nearest = core.causal_reception(candidates, moved_emissions[:4])
    _assert_event_close(nearest - moved_receiver, 1e-11)

    moved_clock = transform >> core.mv.t
    assert (((moved_clock & (nearest - moved_emissions)) - (core.mv.t & (receiver - emitted))).abs() < core.mv.scalar(1e-11, unit=dim.length)).all()
    assert ((moved_clock & (nearest - moved_emissions)) > 0).all()


@pytest.mark.parametrize("phase_degrees,reception_time", [(5, -3), (20, 0), (50, 4)])
def test_noiseless_fit_recovers_position_and_clock_error(phase_degrees, reception_time):
    phases = scenarios.phase_offsets + np.deg2rad(phase_degrees)
    planes = scenarios.orbital_planes
    radius, angular_speed = scenarios.orbit_radius, scenarios.angular_speed
    clock_offset = scenarios.clock_offset
    t_rec = core.mv.scalar(float(reception_time), unit=dim.time)
    receiver = scenarios.receiver + core.time * (core.c * t_rec)
    clock_reading = (core.mv.t & receiver) / core.c + clock_offset
    emitted = core.emitted_events(receiver, planes, phases, radius, angular_speed, 3)
    initial = (receiver + core.mv.y.dual() * scenarios.initial_y
               + core.mv.z.dual() * scenarios.initial_z + core.time * (core.c * clock_offset))
    estimate, information, covariance = core.fitted_reception(emitted, initial, realistic_sigma, 5)
    assert information.unit == 1 / dim.length**2
    assert covariance.unit == dim.length**2

    _assert_event_close(estimate - receiver, 1e-12)
    recovered_clock_offset = clock_reading - (core.mv.t & estimate) / core.c
    assert recovered_clock_offset.unit == dim.time
    assert (recovered_clock_offset - clock_offset).abs() < core.mv.scalar(1e-12, unit=dim.time)
    assert ((core.mv.t & (estimate - emitted)) > 0).all()

    same_estimate, _, larger_covariance = core.fitted_reception(emitted, initial, realistic_sigma * 2, 5)
    _assert_event_close(same_estimate - estimate, 1e-12)
    cov_entries = readouts[:, None] & covariance(readouts)                        # [4, 4] Scalar+
    larger_entries = readouts[:, None] & larger_covariance(readouts)              # [4, 4] Scalar+
    assert ((larger_entries - 4 * cov_entries).abs() < 1e-9 * cov_entries.abs().max()).all()

    directions = stack((core.across, core.mv.y.dual(), core.mv.z.dual(), core.time)) * dim.length  # [4] Step-
    assert ((information(directions) & directions) > 0).all()
    sym_diff = readouts[:, None] & (covariance - covariance.adjugate())(readouts)
    assert (sym_diff.abs() < 1e-12 * cov_entries.abs().max()).all()


def test_timestamp_noise_and_rendering():
    phases = scenarios.phase_offsets + np.deg2rad(20)
    planes = scenarios.orbital_planes
    radius, angular_speed, receiver = scenarios.orbit_radius, scenarios.angular_speed, scenarios.receiver
    emitted = core.emitted_events(receiver, planes, phases, radius, angular_speed, 3)
    n_trials = 256
    noise = np.random.default_rng(2026).normal(size=(n_trials, scenarios.n_satellites)) * (realistic_sigma / core.c)
    observed = emitted + core.time * (core.c * noise)                             # [n_trials, 6] Event-
    initial = (receiver + core.mv.y.dual() * scenarios.initial_y
               + core.mv.z.dual() * scenarios.initial_z + core.time * (core.c * scenarios.clock_offset))
    estimates, _, _ = core.fitted_reception(observed, initial, realistic_sigma, 5)
    estimate, _, covariance = core.fitted_reception(emitted, initial, realistic_sigma, 5)

    probes = stack((core.mv.x, core.mv.y, core.mv.z, core.mv.t,
                    core.mv.x + core.mv.t, core.mv.y + core.mv.z))                 # [6] Readout+
    errors = probes & (estimates - receiver)[:, None]                             # [n_trials, 6] Scalar+
    mean = errors.sum(axis=0) / n_trials                                          # [6] Scalar+
    variance = ((errors - mean) * (errors - mean)).sum(axis=0) / (n_trials - 1)   # [6] Scalar+
    predicted = probes & covariance(probes)                                       # [6] Scalar+
    assert (((variance / predicted) - 1).abs() < 0.35).all()
    assert ((mean / predicted.square_root()).abs() < 0.25).all()

    semi_axes = core.position_semi_axes(covariance)                               # [3] Event-
    assert semi_axes.unit == dim.length
    constellation_fig = render.constellation_figure(emitted, receiver, radius)
    constellation_fig.canvas.draw()
    plt.close(constellation_fig)

    positioning_fig = render.positioning_figure(emitted, receiver, estimate, semi_axes, radius)
    positioning_fig.canvas.draw()
    plt.close(positioning_fig)

    frame = next(render.geometry_animation([(emitted, receiver, estimate, semi_axes)]))
    assert frame.ndim == 3 and frame.shape[2] == 3
