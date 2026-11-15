"""Coverage tests for twistor incidence, the Robinson congruence, linked Hopf fibres, and twistor rendering."""

import matplotlib.pyplot as plt
import numpy as np

from examples.relativity.twistors import core, render, scenarios
from numga import Algebra
from numga.topology import Topology


tolerance = 1e-18


def test_robinson_twistors_stay_orthogonal_along_straight_future_null_rays():
    rng = np.random.default_rng(823)
    events = core.mv(core.Event.derive.dual, rng.normal(size=(8, 4))).dual()
    twistors = core.selection(core.conformal_point(events), core.robinson_twistor)
    extrinsic_volume = core.extrinsic_readout(core.Full * core.extrinsic_embedding)(core.pseudoscalar)
    directions = core.robinson_direction(events)
    times = np.linspace(-1.0, 1.0, 5)
    along_rays = events[:, None] + directions[:, None] * times

    assert (core.pairing(twistors, core.robinson_twistor).scalar_norm_squared() < tolerance).all()
    assert (core.pairing(twistors, extrinsic_volume(core.robinson_twistor)).scalar_norm_squared() < tolerance).all()
    assert (directions.squared().scalar_norm_squared() < tolerance).all()
    assert (((core.mv.t & directions) - 1).scalar_norm_squared() < tolerance).all()
    assert ((core.robinson_direction(along_rays) - directions[:, None]).scalar_norm_squared() < tolerance).all()
    assert (core.incidence(core.conformal_point(along_rays), twistors[:, None]).scalar_norm_squared() < tolerance).all()


polar_angles = np.pi * np.array([0.4, 0.65])
n_circles = 3
n_fibre_samples = 192
sample_times = np.array([-0.8, 0.0, 0.8])
tangent_tolerance = 1e-5


def test_flow_and_field_lines_follow_robinson_and_hopfion_directions():
    curves = core.hopf_fibres(polar_angles, n_circles, n_fibre_samples)
    steps = curves[..., 2:] - curves[..., :-2]
    tangent = steps / steps.antinorm()
    flow = core.robinson_direction(curves[..., 1:-1]).cast(core.Spatial)
    assert ((tangent & flow).scalar_norm_squared() < tangent_tolerance).all()

    electric = scenarios.electric_turn >> curves
    magnetic_turn = (core.mv.yz * (np.pi / 4)).exp()
    magnetic = magnetic_turn >> curves
    times = sample_times[:, None, None, None]
    electric_events = electric + core.robinson_direction(electric) * times
    magnetic_events = magnetic + core.robinson_direction(magnetic) * times
    temporal = core.Bivector.derive.touching(Algebra("t-"))
    electric_field = core.hopfion_field(electric_events).cast(temporal)[..., 1:-1]
    magnetic_field = (core.spacetime_pseudoscalar * core.hopfion_field(magnetic_events)).cast(temporal)[..., 1:-1]
    electric_steps = electric_events[..., 2:] - electric_events[..., :-2]
    magnetic_steps = magnetic_events[..., 2:] - magnetic_events[..., :-2]
    electric_sweeps = ((electric_steps / electric_steps.antinorm()) & core.time).dual()
    magnetic_sweeps = ((magnetic_steps / magnetic_steps.antinorm()) & core.time).dual()
    assert (electric_sweeps.commutator(electric_field / electric_field.squared().square_root()).scalar_norm_squared() < tangent_tolerance).all()
    assert (magnetic_sweeps.commutator(magnetic_field / magnetic_field.squared().square_root()).scalar_norm_squared() < tangent_tolerance).all()


def test_two_fibres_link_once():
    curves = core.hopf_fibres(polar_angles, n_circles, n_fibre_samples)
    loop = Topology.loops([n_fibre_samples])
    fibres = curves[0, :2].field()
    first_steps, second_steps = (loop.d[0] * fibres).batch()
    first_midpoints, second_midpoints = (loop.incidence[1, 0].mean(axis=0) * fibres).batch()
    offsets = first_midpoints[:, None] - second_midpoints[None, :]
    distance_squared = offsets.scalar_norm_squared()
    volume = core.mv.xyz & (offsets & first_steps[:, None] & second_steps[None, :])
    linking = (volume / (distance_squared * distance_squared.square_root())).sum() / (4 * np.pi)
    assert ((linking.abs() - 1).abs() < 1e-3).all()


def test_incidence_and_rendering():
    twistors, events, rays = scenarios.ray_incidence()
    rays_of_twistors = core.ray_readout(twistors, twistors)
    recovered = core.ray_events(rays_of_twistors[:, None], core.mv.t & events)
    assert ((recovered - events).scalar_norm_squared() < tolerance).all()
    assert (core.incidence(core.conformal_point(rays), twistors[:, None]).scalar_norm_squared() < tolerance).all()

    small_curves = core.hopf_fibres(np.pi * np.array([0.5]), 2, 16)
    plt.close(render.incidence_figure(events, rays))
    plt.close(render.congruence_figure(small_curves))
    plt.close(render.fields_figure(small_curves))
    frame = next(render.fields_animation(core.propagated_lines(small_curves, np.array([0.0]))))
    assert frame.ndim == 3 and frame.shape[2] == 3
