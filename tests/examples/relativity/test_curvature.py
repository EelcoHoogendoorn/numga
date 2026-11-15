"""Geometric and physical checks for the gravitational-wave curvature example."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import stack

from examples.relativity.curvature import render, scenarios
from examples.relativity.curvature.core import (
    Antibivector,
    Antivector,
    boosted_observers,
    curvature_of_strain,
    integrated_displacements,
    mv,
    polarization_curvatures,
    strain_patterns,
    tidal_map,
    time_direction,
    travel,
    wave_packet,
    wave_vector,
)


def rest_ring(n_beads: int) -> Antivector:
    """Unit rest separations sit evenly around the transverse xy circle (`[n_beads] Antivector-`)."""
    angles = np.linspace(0, 2 * np.pi, n_beads, endpoint=False)
    bead_turns = (mv.xy * (-angles / 2)).exp()                      # [n_beads] Rotor+
    return bead_turns >> mv.x.dual()                                # [n_beads] Antivector-


def sample_areas(seed: int) -> Antibivector:
    """Random pairs of spacetime directions span test area elements (`[5] Antibivector-`)."""
    first, second = mv.vector(np.random.default_rng(seed).normal(size=(2, 5, 4))).dual()
    return first & second                                           # [5] Antibivector-


@pytest.mark.parametrize("amplitudes", [(1.0, 0.0), (0.0, 1.0), (0.6, -0.8)])
def test_type_n_null_alignment_and_spin_two_reversal(amplitudes: tuple[float, float]) -> None:
    plus, cross = polarization_curvatures()
    curvature = amplitudes[0] * plus + amplitudes[1] * cross
    singular_values = curvature.svdvals()
    assert int((singular_values > 1e-8).sum()) == 2
    assert ((wave_vector ^ curvature).svdvals() < 1e-10).all()
    quarter_turn = (mv.xy * (np.pi / 4)).exp()
    reversed_sum = (quarter_turn >> curvature(quarter_turn << Antibivector)) + curvature
    assert (reversed_sum.svdvals() < 1e-10).all()


def test_pair_symmetry_bianchi_identity_and_vacuum_trace() -> None:
    plus, cross = polarization_curvatures()
    curvature = 1.3 * plus - 0.7 * cross
    first_area = sample_areas(7)
    second_area = sample_areas(8)
    pair_residual = (curvature(second_area) & first_area) - (curvature(first_area) & second_area)
    assert (pair_residual.abs() < 1e-10).all()

    edge_a, edge_b = mv.vector(np.random.default_rng(9).normal(size=(2, 4))).dual()
    bianchi = (
        (curvature(edge_a & edge_b) & Antivector)
        + (curvature(edge_b & Antivector) & edge_a)
        + (curvature(Antivector & edge_a) & edge_b)
    )
    assert (bianchi.svdvals() < 1e-10).all()

    directions = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))
    boosts = ((directions & time_direction).dual() * np.array([0.0, 0.4, -0.8]) / 2).exp()
    observers = boosts >> time_direction
    tides = tidal_map(curvature, observers)(Antivector) & Antivector
    traces = tides.eigvals(Antivector.dot(Antivector)).sum(axis=-1)
    assert (traces.abs() < 1e-10).all()


def test_transverse_tidal_spectrum_and_doppler_scaling() -> None:
    plus, _ = polarization_curvatures()
    rapidities = np.array([-0.7, 0.0, 0.5])
    observers = boosted_observers(rapidities, mv.z.dual())
    responses = tidal_map(plus, observers)
    amplitudes = (-responses.dual()).svdvals()[..., 0]
    assert ((amplitudes - np.exp(-2 * rapidities)).abs() < 1e-10).all()

    basis_directions = stack((mv.t, mv.x, mv.y, mv.z)).dual()
    for index, rapidity in enumerate(rapidities):
        expected = np.exp(-2 * rapidity)
        response = responses[index]
        observer = observers[index]
        eigenvalues = (response(Antivector) & Antivector).eigvals(Antivector.dot(Antivector))
        assert ((eigenvalues * (eigenvalues - expected) * (eigenvalues + expected)).abs() < 1e-10).all()
        assert (eigenvalues.sum().abs() < 1e-10).all()
        assert ((eigenvalues.abs().sum() - 2 * expected).abs() < 1e-10).all()
        assert ((response(observer) & basis_directions).abs() < 1e-10).all()
        assert ((response(travel) & basis_directions).abs() < 1e-10).all()


def test_lorentz_covariance_of_curvature_and_tidal_binding() -> None:
    plus, _ = polarization_curvatures()
    rotor = (mv.tx * 0.31).exp() * (mv.yz * -0.23).exp()
    boosted_wave = rotor >> wave_vector
    first_plane = boosted_wave ^ (rotor >> mv.x)
    second_plane = boosted_wave ^ (rotor >> mv.y)
    transformed = first_plane * (first_plane & Antibivector) - second_plane * (second_plane & Antibivector)
    conjugated = rotor >> plus(rotor << Antibivector)
    assert ((transformed - conjugated).svdvals() < 1e-10).all()

    boosted_observer = rotor >> time_direction
    bound_after = tidal_map(transformed, boosted_observer)
    bound_before = rotor >> tidal_map(plus, time_direction)(rotor << Antivector)
    assert ((bound_after - bound_before).svdvals() < 1e-10).all()


def test_strain_map_matches_integrated_ring_and_reconstructs_curvature() -> None:
    plus_strain, cross_strain = strain_patterns()
    basis_directions = stack((mv.t, mv.x, mv.y, mv.z)).dual()
    for separation, image in (
        (mv.x.dual(), mv.x),
        (mv.y.dual(), -mv.y),
        (time_direction, 0 * mv.x),
        (mv.z.dual(), 0 * mv.x),
    ):
        assert (((plus_strain(separation) - image) & basis_directions).abs() < 1e-10).all()

    plus, cross = polarization_curvatures()
    reference = rest_ring(8)
    basis_slices = stack((mv.t, mv.x, mv.y, mv.z))
    max_errors = []
    for n_steps in (161, 321):
        time = np.linspace(0.0, 6.0, n_steps)
        profiles, second_derivatives = wave_packet(time, 6.0, 3, 1e-4)
        curvature_weights = -0.5 * second_derivatives[:, None] * scenarios.polarization_weights
        waves = curvature_weights[..., 0] * plus + curvature_weights[..., 1] * cross
        response = tidal_map(waves, time_direction)
        acceleration = -response[:, :, None](reference).dual()
        displacement = integrated_displacements(time, acceleration)

        strain_weights = 0.5 * profiles[:, None] * scenarios.polarization_weights
        strain_map = strain_weights[..., 0] * plus_strain + strain_weights[..., 1] * cross_strain
        predicted = strain_map[:, :, None](reference).dual()
        component_errors = (basis_slices & (displacement - predicted)[..., None]).abs()
        max_errors.append(component_errors.max())

    assert max_errors[1] < max_errors[0] / 10
    assert max_errors[1] < 1e-8
    assert (predicted.antinorm().max() > 4e-5)

    second_weights = 0.5 * second_derivatives[:, None] * scenarios.polarization_weights
    second_strain = second_weights[..., 0] * plus_strain + second_weights[..., 1] * cross_strain
    assert ((response + second_strain).svdvals() < 1e-10).all()

    two_form = curvature_of_strain(wave_vector, second_strain)
    sample_edge = mv.vector(np.random.default_rng(5).normal(size=4)).dual()
    assert ((two_form(sample_edge) - waves(sample_edge & Antivector)).svdvals() < 1e-10).all()


def test_figures_and_animation_render() -> None:
    time, profiles, second_derivatives, reference, displacement, acceleration = scenarios.detector_response(4)
    for figure in (
        render.curvature_map_figure(*scenarios.curvature_event()),
        render.doppler_figure(*scenarios.doppler_amplitudes(2)),
        render.detector_figure(time, reference, displacement, acceleration, scenarios.amplification),
        render.packet_figure(time, profiles, -0.5 * second_derivatives),
    ):
        plt.close(figure)
    frames = list(
        render.detector_animation(time[:1], reference, displacement[:1], acceleration[:1], scenarios.amplification)
    )
    assert len(frames) == 1
