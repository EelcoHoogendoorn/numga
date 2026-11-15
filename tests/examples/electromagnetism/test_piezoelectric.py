"""Patch tests, electromechanical reciprocity, charge balance, beam convergence, and rendering checks."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import stack
from examples.electromagnetism.piezoelectric import core, render, scenarios


WIDTH = 5
VOLTAGE = 1e-4
FORCE = 1e-4


@pytest.fixture(scope="module")
def material():
    return scenarios.pzt5a(stack((core.mv.x, core.mv.y)))


@pytest.fixture(scope="module")
def device(material):
    topology, coordinates, layer_poling, top_weights, tip_weights = scenarios.bimorph(10, 1, 16, 4)
    vertices = core.mv(core.Plane, coordinates).dual().field()
    poling = core.mv.scalar(layer_poling).field()
    top_electrode = core.mv.scalar(top_weights).field()
    tip_distribution = core.mv.scalar(tip_weights).field()
    areas, gradients = core.triangles(topology, vertices)
    blocks = core.assemble(topology, areas, gradients, *material, poling, WIDTH)
    return topology, vertices, areas, gradients, poling, top_electrode, tip_distribution, blocks


@pytest.mark.parametrize("motion", [(core.mv.xy * 0.27).exp(), core.mv.x], ids=["rotation", "reflection"])
def test_material_follows_its_directions_under_rigid_transformations(material, motion):
    elastic, polarization, permittivity = material
    reoriented_material = scenarios.pzt5a(motion >> stack((core.mv.x, core.mv.y)))
    displacement = 0.12 * core.axis - 0.3 * core.pole
    test_displacement = -0.2 * core.axis + 0.1 * core.pole
    gradient = 0.7 * core.mv.x + 0.2 * core.mv.y
    test_gradient = -0.6 * core.mv.x + 0.8 * core.mv.y
    electric_field = 0.3 * core.mv.xt + 0.7 * core.mv.yt

    work = elastic(displacement, gradient, test_displacement, test_gradient)
    reoriented_work = reoriented_material[0](
        motion >> displacement, motion >> gradient, motion >> test_displacement, motion >> test_gradient,
    )
    assert (reoriented_work - work).abs() < 1e-12
    assert ((reoriented_material[1](motion >> displacement, motion >> gradient)
             - (motion >> polarization(displacement, gradient))).norm() < 1e-12).all()
    assert ((reoriented_material[2](motion >> electric_field)
             - (motion >> permittivity(electric_field))).norm() < 1e-12).all()


@pytest.mark.parametrize("electric_field, extension, shear, free_permittivity, direction", [
    (core.mv.yt, (-171e-12, 374e-12), 0, 1700, core.pole),
    (core.mv.xt, (0, 0), 584e-12, 1730, core.axis),
], ids=["parallel", "transverse"])
def test_stress_free_response_matches_measured_pzt5a_data(
        material, electric_field, extension, shear, free_permittivity, direction):
    elastic, polarization, permittivity = material
    changes = (
        (core.axis, core.mv.x, extension[0] * scenarios.field_unit),
        (core.pole, core.mv.y, extension[1] * scenarios.field_unit),
        (core.axis, core.mv.y, shear * scenarios.field_unit / 2),
        (core.pole, core.mv.x, shear * scenarios.field_unit / 2),
    )
    for test_displacement, test_gradient in ((core.axis, core.mv.x), (core.pole, core.mv.y), (core.axis, core.mv.y)):
        mechanical = sum(
            elastic(displacement, gradient, test_displacement, test_gradient) * amplitude
            for displacement, gradient, amplitude in changes
        )
        electrical = electric_field & polarization(test_displacement, test_gradient)
        assert (mechanical - electrical).abs() < 1e-5

    total_excitation = sum(
        (polarization(displacement, gradient) * amplitude for displacement, gradient, amplitude in changes),
        permittivity(electric_field),
    )
    measured = (direction & core.time) * (free_permittivity * 8.8541878128e-12 / scenarios.permittivity_unit)
    assert ((total_excitation - measured).norm() < 1e-5).all()


def solve_state(topology, top_electrode, tip_distribution, blocks, voltage, force, floating_nodes):
    free_motion, free_voltage = scenarios.boundary_maps(topology, floating_nodes)
    displacement, potential = core.equilibrium(
        *blocks, free_motion, free_voltage, voltage * top_electrode, (core.mv.y * force) * tip_distribution,
    )
    charges = -(blocks[1].adjugate()(displacement) + blocks[2](potential))
    return displacement, potential, charges


def test_affine_fields_and_rigid_motion_are_reproduced(device):
    topology, vertices, _, gradients, _, top_electrode, _, (mechanical, coupling, dielectric) = device
    horizontal, vertical = core.mv.x & vertices, core.mv.y & vertices
    translation = vertices * 0 + core.axis * 0.3 + core.pole * 0.7
    turning = vertices.commutator(core.mv.xy * 0.2)
    for displacement in (translation, turning):
        local_strain = core.strain(topology, displacement, gradients)
        assert ((local_strain(core.axis) & core.axis).abs() < 1e-12).all()
        assert ((local_strain(core.pole) & core.pole).abs() < 1e-12).all()
        assert ((local_strain(core.pole) & core.axis).abs() < 1e-12).all()
        assert ((core.mv.x & mechanical(displacement)).abs() < 1e-10).all()
        assert ((core.mv.y & mechanical(displacement)).abs() < 1e-10).all()
        assert (coupling.adjugate()(displacement).antinorm() < 1e-10).all()

    displacement = (
        core.axis * (0.2 * horizontal + 0.07 * vertical)
        + core.pole * (0.07 * horizontal - 0.3 * vertical)
    )
    local_strain = core.strain(topology, displacement, gradients)
    assert (((local_strain(core.axis) & core.axis) - 0.2).abs() < 1e-12).all()
    assert (((local_strain(core.pole) & core.pole) + 0.3).abs() < 1e-12).all()
    assert (((local_strain(core.pole) & core.axis) - 0.07).abs() < 1e-12).all()

    potential = 2 * horizontal - 3 * vertical + 0.7
    expected = (2 * core.mv.x - 3 * core.mv.y) ^ (-core.mv.t)
    electric_field = ((topology.incidence[2, 0] * potential).batch() * (gradients.batch() ^ (-core.mv.t))).sum(axis=-2)
    assert (((electric_field - expected) & (core.axis & core.time)).abs() < 1e-11).all()
    assert (((electric_field - expected) & (core.pole & core.time)).abs() < 1e-11).all()
    constant = 0 * top_electrode + 1
    assert ((core.mv.x & coupling(constant)).abs() < 1e-11).all()
    assert ((core.mv.y & coupling(constant)).abs() < 1e-11).all()
    assert (dielectric(constant).antinorm() < 1e-11).all()


def test_actuation_and_short_circuit_sensing_are_reciprocal(device):
    topology, _, _, _, _, top_electrode, tip_distribution, blocks = device
    fixed_nodes = np.array([], dtype=int)
    actuator_motion, _, _ = solve_state(topology, top_electrode, tip_distribution, blocks, VOLTAGE, 0, fixed_nodes)
    _, _, sensor_charges = solve_state(topology, top_electrode, tip_distribution, blocks, 0, FORCE, fixed_nodes)
    tip_motion = ((core.mv.y & actuator_motion) * tip_distribution).sites.sum()
    motion_per_voltage = tip_motion / VOLTAGE
    charge_per_force = (top_electrode & sensor_charges).sites.sum() / FORCE

    assert motion_per_voltage.abs() > 1
    assert (motion_per_voltage - charge_per_force).abs() < 1e-9


def test_floating_electrode_is_equipotential_and_conserves_charge(device):
    topology, _, _, _, _, top_electrode, tip_distribution, (mechanical, coupling, dielectric) = device
    displacement, potential, charges = solve_state(
        topology, top_electrode, tip_distribution, (mechanical, coupling, dielectric), 0, FORCE, topology.top_nodes,
    )
    V, E, F = topology.n_elements
    interior_nodes = np.setdiff1d(np.arange(V), np.concatenate((topology.bottom_nodes, topology.top_nodes)))
    free_nodes = np.setdiff1d(np.arange(V), topology.clamp_nodes)
    top = topology.reading(topology.top_nodes, 0)
    voltage = (top * potential).sites.mean()

    assert voltage.abs() > 1e-6
    assert ((top * potential - voltage).abs() < 1e-12).all()
    assert ((topology.reading(topology.bottom_nodes, 0) * potential).abs() < 1e-12).all()
    assert (top_electrode & charges).sites.sum().abs() < 1e-11
    assert ((topology.reading(interior_nodes, 0) * charges).antinorm() < 1e-11).all()
    assert charges.sites.sum().antinorm() < 1e-11

    load = (core.mv.y * FORCE) * tip_distribution
    reaction = mechanical(displacement) + coupling(potential) - load
    assert ((core.mv.x & (topology.reading(free_nodes, 0) * reaction)).abs() < 1e-11).all()
    assert ((core.mv.y & (topology.reading(free_nodes, 0) * reaction)).abs() < 1e-11).all()
    assert ((topology.reading(topology.clamp_nodes, 0) * displacement).antinorm() < 1e-12).all()
    assert (core.mv.y & (reaction.sites.sum() + load.sites.sum())).abs() < 1e-11


def test_reversing_poling_reverses_the_coupled_responses(device, material):
    topology, _, areas, gradients, poling, top_electrode, tip_distribution, blocks = device
    opposite_blocks = core.assemble(topology, areas, gradients, *material, -poling, WIDTH)
    fixed_nodes = np.array([], dtype=int)
    displacement, potential, charges = solve_state(
        topology, top_electrode, tip_distribution, blocks, VOLTAGE, 0, fixed_nodes,
    )
    reverse_displacement, reverse_potential, reverse_charges = solve_state(
        topology, top_electrode, tip_distribution, opposite_blocks, VOLTAGE, 0, fixed_nodes,
    )
    assert ((displacement + reverse_displacement).antinorm() < 1e-11).all()
    assert ((potential - reverse_potential).abs() < 1e-11).all()
    assert ((charges - reverse_charges).antinorm() < 1e-11).all()

    displacement, potential, _ = solve_state(
        topology, top_electrode, tip_distribution, blocks, 0, FORCE, topology.top_nodes,
    )
    reverse_displacement, reverse_potential, _ = solve_state(
        topology, top_electrode, tip_distribution, opposite_blocks, 0, FORCE, topology.top_nodes,
    )
    assert ((displacement - reverse_displacement).antinorm() < 1e-11).all()
    assert ((potential + reverse_potential).abs() < 1e-11).all()


def test_assembled_loads_match_enthalpy_variation(device, material):
    topology, vertices, areas, gradients, poling, _, _, (mechanical, coupling, dielectric) = device
    horizontal, vertical = core.mv.x & vertices, core.mv.y & vertices
    displacement = (
        core.axis * (0.001 * horizontal.squared() + 0.002 * horizontal * vertical)
        + core.pole * (0.003 * horizontal * vertical - 0.004 * vertical.squared())
    )
    potential = 0.002 * horizontal * vertical + 0.003 * horizontal.squared() - 0.004 * vertical
    perturbation = (
        core.axis * (0.3 * horizontal * vertical)
        + core.pole * (0.2 * horizontal.squared() + 0.4 * vertical.squared())
    )
    voltage_step = 0.3 * horizontal.squared() - 0.7 * horizontal * vertical + 0.2 * vertical.squared()

    def total_enthalpy(trial_displacement, trial_potential):
        electric_field = (
            (topology.incidence[2, 0] * trial_potential).batch() * (gradients.batch() ^ (-core.mv.t))
        ).sum(axis=-2)
        density = core.enthalpy(
            core.strain(topology, trial_displacement, gradients), electric_field, *material, poling.batch(),
        )
        return (WIDTH * areas.batch() * density).sum()

    step = 1e-4
    variation = (
        total_enthalpy(displacement + step * perturbation, potential + step * voltage_step)
        - total_enthalpy(displacement - step * perturbation, potential - step * voltage_step)
    ) / (2 * step)
    work = ((mechanical(displacement) + coupling(potential)) & perturbation).sites.sum()
    work += (voltage_step & (coupling.adjugate()(displacement) + dielectric(potential))).sites.sum()
    assert (variation - work).abs() < 1e-10


def test_refinement_approaches_the_slender_bimorph_response(material):
    strip_length, strip_thickness = 10, 1
    compliance, free_coupling = 16.4e-12, -171e-12
    free_permittivity = 1700 * 8.8541878128e-12
    stiffness = 1 / compliance / scenarios.stress_unit
    piezo_constant = free_coupling / compliance * scenarios.field_unit / scenarios.stress_unit
    permittivity = (free_permittivity - free_coupling**2 / compliance) / scenarios.permittivity_unit
    beam_response = (
        1.5 * piezo_constant * permittivity
        / (stiffness * permittivity + piezo_constant**2 / 4) * (strip_length / strip_thickness)**2
    )
    errors = []
    for n_along, n_across in ((12, 2), (24, 4), (48, 8)):
        topology, coordinates, layer_poling, top_weights, tip_weights = scenarios.bimorph(
            strip_length, strip_thickness, n_along, n_across,
        )
        vertices = core.mv(core.Plane, coordinates).dual().field()
        poling = core.mv.scalar(layer_poling).field()
        top_electrode = core.mv.scalar(top_weights).field()
        tip_distribution = core.mv.scalar(tip_weights).field()
        areas, gradients = core.triangles(topology, vertices)
        blocks = core.assemble(topology, areas, gradients, *material, poling, WIDTH)
        displacement, _, _ = solve_state(
            topology, top_electrode, tip_distribution, blocks, VOLTAGE, 0, np.array([], dtype=int),
        )
        tip_motion = ((core.mv.y & displacement) * tip_distribution).sites.sum()
        response = float((tip_motion / VOLTAGE).to_array())
        errors.append(abs(response / beam_response - 1))

    assert errors[1] < 0.7 * errors[0]
    assert errors[2] < 0.7 * errors[1]
    assert errors[2] < 0.08


def test_render_strip_device_and_single_frame(device):
    topology, vertices, areas, gradients, _, top_electrode, tip_distribution, blocks = device
    displacement, potential, _ = solve_state(
        topology, top_electrode, tip_distribution, blocks, VOLTAGE, 0, np.array([], dtype=int),
    )
    axial_strain = core.strain(topology, displacement, gradients)(core.axis) & core.axis
    strip_fig = render.draw_strip(topology, vertices)
    device_fig = render.draw_device(
        topology, vertices, areas, displacement, potential, axial_strain, 100, VOLTAGE, 2e-5,
    )
    frames = list(render.animate(
        topology, vertices, areas, 100, VOLTAGE, 2e-5,
        core.load_cycle(displacement, potential, axial_strain, 2),
    ))
    plt.close(strip_fig)
    plt.close(device_fig)
    assert len(frames) == 2
    assert frames[0].ndim == 3 and frames[0].shape[-1] == 3
