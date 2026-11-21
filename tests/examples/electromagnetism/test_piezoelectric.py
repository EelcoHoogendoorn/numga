"""Patch tests, electromechanical reciprocity, charge balance, beam convergence, and rendering checks."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import GATypeError, dim, stack
from examples.electromagnetism.piezoelectric import core, render, scenarios
from examples.electromagnetism.piezoelectric.numerical import solve_coupled


UNIT_LENGTH = core.mv.scalar(1.0, unit=dim.length)
UNIT_VOLTAGE = core.mv.scalar(1.0, unit=dim.voltage)
UNIT_FIELD = UNIT_VOLTAGE / UNIT_LENGTH
UNIT_FORCE = core.mv.scalar(1.0, unit=scenarios.force_dim)
UNIT_CHARGE = core.mv.scalar(1.0, unit=dim.charge)
UNIT_ENERGY = core.mv.scalar(1.0, unit=scenarios.energy_dim)
UNIT_STRESS = core.mv.scalar(1.0, unit=scenarios.stress_dim)
UNIT_PIEZO = core.mv.scalar(1.0, unit=scenarios.piezo_dim)

WIDTH = core.mv.scalar(5.0, unit=dim.length)
VOLTAGE = core.mv.scalar(1e-4, unit=dim.voltage)
FORCE = core.mv.scalar(1e-4, unit=scenarios.force_dim)
ZERO_VOLTAGE = core.mv.scalar(0.0, unit=dim.voltage)
ZERO_FORCE = core.mv.scalar(0.0, unit=scenarios.force_dim)


@pytest.fixture(scope="module")
def material():
    return scenarios.pzt5a(stack((core.mv.x, core.mv.y)))


@pytest.fixture(scope="module")
def device(material):
    topology, coordinates, layer_poling, top_weights, tip_weights = scenarios.bimorph(10, 1, 16, 4)
    vertices = core.mv(core.Plane, coordinates).dual().field() * dim.length
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
    displacement = (0.12 * core.axis - 0.3 * core.pole) * UNIT_LENGTH
    test_displacement = (-0.2 * core.axis + 0.1 * core.pole) * UNIT_LENGTH
    gradient = (0.7 * core.mv.x + 0.2 * core.mv.y) / UNIT_LENGTH
    test_gradient = (-0.6 * core.mv.x + 0.8 * core.mv.y) / UNIT_LENGTH
    electric_field = (0.3 * core.mv.xt + 0.7 * core.mv.yt) * UNIT_FIELD

    work = elastic(displacement, gradient, test_displacement, test_gradient)
    reoriented_work = reoriented_material[0](
        motion >> displacement, motion >> gradient, motion >> test_displacement, motion >> test_gradient,
    )
    assert (reoriented_work - work).abs() < 1e-12 * UNIT_STRESS
    assert ((reoriented_material[1](motion >> displacement, motion >> gradient)
             - (motion >> polarization(displacement, gradient))).norm() < 1e-12 * UNIT_PIEZO).all()
    assert ((reoriented_material[2](motion >> electric_field)
             - (motion >> permittivity(electric_field))).norm() < 1e-12 * UNIT_PIEZO).all()


@pytest.mark.parametrize("electric_field, extension, shear, free_permittivity, direction", [
    (core.mv.yt, (-171e-12, 374e-12), 0, 1700, core.pole),
    (core.mv.xt, (0, 0), 584e-12, 1730, core.axis),
], ids=["parallel", "transverse"])
def test_stress_free_response_matches_measured_pzt5a_data(
        material, electric_field, extension, shear, free_permittivity, direction):
    elastic, polarization, permittivity = material
    field = electric_field * UNIT_FIELD
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
        electrical = field & polarization(test_displacement, test_gradient)
        assert (mechanical - electrical).abs() < 1e-5 * UNIT_STRESS

    total_excitation = sum(
        (polarization(displacement, gradient) * amplitude for displacement, gradient, amplitude in changes),
        permittivity(field),
    )
    measured = (direction & core.time) * (free_permittivity * 8.8541878128e-12 / scenarios.permittivity_unit * UNIT_PIEZO)
    assert ((total_excitation - measured).norm() < 1e-5 * UNIT_PIEZO).all()


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
    translation = vertices * 0 + (core.axis * 0.3 + core.pole * 0.7) * UNIT_LENGTH
    turning = vertices.commutator(core.mv.xy * 0.2)
    for displacement in (translation, turning):
        local_strain = core.strain(topology, displacement, gradients)
        assert ((local_strain(core.axis) & core.axis).abs() < 1e-12).all()
        assert ((local_strain(core.pole) & core.pole).abs() < 1e-12).all()
        assert ((local_strain(core.pole) & core.axis).abs() < 1e-12).all()
        assert ((core.mv.x & mechanical(displacement)).abs() < 1e-10 * UNIT_FORCE).all()
        assert ((core.mv.y & mechanical(displacement)).abs() < 1e-10 * UNIT_FORCE).all()
        assert (coupling.adjugate()(displacement).antinorm() < 1e-10 * UNIT_CHARGE).all()

    displacement = (
        core.axis * (0.2 * horizontal + 0.07 * vertical)
        + core.pole * (0.07 * horizontal - 0.3 * vertical)
    )
    local_strain = core.strain(topology, displacement, gradients)
    assert (((local_strain(core.axis) & core.axis) - 0.2).abs() < 1e-12).all()
    assert (((local_strain(core.pole) & core.pole) + 0.3).abs() < 1e-12).all()
    assert (((local_strain(core.pole) & core.axis) - 0.07).abs() < 1e-12).all()

    potential = (2 * horizontal - 3 * vertical) * UNIT_FIELD + 0.7 * UNIT_VOLTAGE
    expected = ((2 * core.mv.x - 3 * core.mv.y) ^ (-core.mv.t)) * UNIT_FIELD
    electric_field = ((topology.incidence[2, 0] * potential).batch() * (gradients.batch() ^ (-core.mv.t))).sum(axis=-2)
    assert (((electric_field - expected) & (core.axis & core.time)).abs() < 1e-11 * UNIT_FIELD).all()
    assert (((electric_field - expected) & (core.pole & core.time)).abs() < 1e-11 * UNIT_FIELD).all()
    constant = (0 * top_electrode + 1) * UNIT_VOLTAGE
    assert ((core.mv.x & coupling(constant)).abs() < 1e-11 * UNIT_FORCE).all()
    assert ((core.mv.y & coupling(constant)).abs() < 1e-11 * UNIT_FORCE).all()
    assert (dielectric(constant).antinorm() < 1e-11 * UNIT_CHARGE).all()


def test_actuation_and_short_circuit_sensing_are_reciprocal(device):
    topology, _, _, _, _, top_electrode, tip_distribution, blocks = device
    fixed_nodes = np.array([], dtype=int)
    actuator_motion, _, _ = solve_state(topology, top_electrode, tip_distribution, blocks, VOLTAGE, ZERO_FORCE, fixed_nodes)
    _, _, sensor_charges = solve_state(topology, top_electrode, tip_distribution, blocks, ZERO_VOLTAGE, FORCE, fixed_nodes)
    tip_motion = ((core.mv.y & actuator_motion) * tip_distribution).sites.sum()
    motion_per_voltage = tip_motion / VOLTAGE
    charge_per_force = (top_electrode & sensor_charges).sites.sum() / FORCE

    assert motion_per_voltage.unit == dim.length / dim.voltage
    assert charge_per_force.unit == dim.length / dim.voltage
    assert motion_per_voltage.abs() > UNIT_LENGTH / UNIT_VOLTAGE
    assert (motion_per_voltage - charge_per_force).abs() < 1e-9 * motion_per_voltage.abs()


def test_floating_electrode_is_equipotential_and_conserves_charge(device):
    topology, _, _, _, _, top_electrode, tip_distribution, (mechanical, coupling, dielectric) = device
    displacement, potential, charges = solve_state(
        topology, top_electrode, tip_distribution, (mechanical, coupling, dielectric), ZERO_VOLTAGE, FORCE, topology.top_nodes,
    )
    V, E, F = topology.n_elements
    interior_nodes = np.setdiff1d(np.arange(V), np.concatenate((topology.bottom_nodes, topology.top_nodes)))
    free_nodes = np.setdiff1d(np.arange(V), topology.clamp_nodes)
    top = topology.reading(topology.top_nodes, 0)
    voltage = (top * potential).sites.mean()

    assert voltage.abs() > 1e-6 * UNIT_VOLTAGE
    assert ((top * potential - voltage).abs() < 1e-12 * UNIT_VOLTAGE).all()
    assert ((topology.reading(topology.bottom_nodes, 0) * potential).abs() < 1e-12 * UNIT_VOLTAGE).all()
    assert (top_electrode & charges).sites.sum().abs() < 1e-11 * UNIT_CHARGE
    assert ((topology.reading(interior_nodes, 0) * charges).antinorm() < 1e-11 * UNIT_CHARGE).all()
    assert charges.sites.sum().antinorm() < 1e-11 * UNIT_CHARGE

    load = (core.mv.y * FORCE) * tip_distribution
    reaction = mechanical(displacement) + coupling(potential) - load
    assert ((core.mv.x & (topology.reading(free_nodes, 0) * reaction)).abs() < 1e-11 * UNIT_FORCE).all()
    assert ((core.mv.y & (topology.reading(free_nodes, 0) * reaction)).abs() < 1e-11 * UNIT_FORCE).all()
    assert ((topology.reading(topology.clamp_nodes, 0) * displacement).antinorm() < 1e-12 * UNIT_LENGTH).all()
    assert (core.mv.y & (reaction.sites.sum() + load.sites.sum())).abs() < 1e-11 * UNIT_FORCE


def test_reversing_poling_reverses_the_coupled_responses(device, material):
    topology, _, areas, gradients, poling, top_electrode, tip_distribution, blocks = device
    opposite_blocks = core.assemble(topology, areas, gradients, *material, -poling, WIDTH)
    fixed_nodes = np.array([], dtype=int)
    displacement, potential, charges = solve_state(
        topology, top_electrode, tip_distribution, blocks, VOLTAGE, ZERO_FORCE, fixed_nodes,
    )
    reverse_displacement, reverse_potential, reverse_charges = solve_state(
        topology, top_electrode, tip_distribution, opposite_blocks, VOLTAGE, ZERO_FORCE, fixed_nodes,
    )
    assert ((displacement + reverse_displacement).antinorm() < 1e-11 * UNIT_LENGTH).all()
    assert ((potential - reverse_potential).abs() < 1e-11 * UNIT_VOLTAGE).all()
    assert ((charges - reverse_charges).antinorm() < 1e-11 * UNIT_CHARGE).all()

    displacement, potential, _ = solve_state(
        topology, top_electrode, tip_distribution, blocks, ZERO_VOLTAGE, FORCE, topology.top_nodes,
    )
    reverse_displacement, reverse_potential, _ = solve_state(
        topology, top_electrode, tip_distribution, opposite_blocks, ZERO_VOLTAGE, FORCE, topology.top_nodes,
    )
    assert ((displacement - reverse_displacement).antinorm() < 1e-11 * UNIT_LENGTH).all()
    assert ((potential + reverse_potential).abs() < 1e-11 * UNIT_VOLTAGE).all()


def test_assembled_loads_match_enthalpy_variation(device, material):
    topology, vertices, areas, gradients, poling, _, _, (mechanical, coupling, dielectric) = device
    horizontal = (core.mv.x & vertices) / UNIT_LENGTH
    vertical = (core.mv.y & vertices) / UNIT_LENGTH
    displacement = (
        core.axis * (0.001 * horizontal.squared() + 0.002 * horizontal * vertical)
        + core.pole * (0.003 * horizontal * vertical - 0.004 * vertical.squared())
    ) * UNIT_LENGTH
    potential = (0.002 * horizontal * vertical + 0.003 * horizontal.squared() - 0.004 * vertical) * UNIT_VOLTAGE
    perturbation = (
        core.axis * (0.3 * horizontal * vertical)
        + core.pole * (0.2 * horizontal.squared() + 0.4 * vertical.squared())
    ) * UNIT_LENGTH
    voltage_step = (0.3 * horizontal.squared() - 0.7 * horizontal * vertical + 0.2 * vertical.squared()) * UNIT_VOLTAGE

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
    assert variation.unit == scenarios.energy_dim
    assert work.unit == scenarios.energy_dim
    assert (variation - work).abs() < 1e-10 * UNIT_ENERGY


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
        vertices = core.mv(core.Plane, coordinates).dual().field() * dim.length
        poling = core.mv.scalar(layer_poling).field()
        top_electrode = core.mv.scalar(top_weights).field()
        tip_distribution = core.mv.scalar(tip_weights).field()
        areas, gradients = core.triangles(topology, vertices)
        blocks = core.assemble(topology, areas, gradients, *material, poling, WIDTH)
        displacement, _, _ = solve_state(
            topology, top_electrode, tip_distribution, blocks, VOLTAGE, ZERO_FORCE, np.array([], dtype=int),
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
        topology, top_electrode, tip_distribution, blocks, VOLTAGE, ZERO_FORCE, np.array([], dtype=int),
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


def test_units_refuse_inconsistent_loads_and_coupled_blocks(device):
    topology, _, _, _, _, top_electrode, tip_distribution, blocks = device
    mechanical, coupling, dielectric = blocks
    free_motion, free_voltage = scenarios.boundary_maps(topology, np.array([], dtype=int))
    with pytest.raises(GATypeError):
        core.equilibrium(
            mechanical, coupling, dielectric, free_motion, free_voltage,
            FORCE * top_electrode, (core.mv.y * ZERO_FORCE) * tip_distribution,
        )
    elastic = free_motion.adjugate() * mechanical * free_motion
    mixed = free_motion.adjugate() * coupling * free_voltage
    capacitance = free_voltage.adjugate() * dielectric * free_voltage
    load = free_motion.adjugate() * ((core.mv.y * FORCE) * tip_distribution)
    charge = -(free_voltage.adjugate() * dielectric(ZERO_VOLTAGE * top_electrode))
    with pytest.raises(GATypeError):
        solve_coupled(elastic, mixed * UNIT_LENGTH, capacitance, load, charge)
    with pytest.raises(GATypeError):
        solve_coupled(elastic, mixed, capacitance, load * UNIT_LENGTH, charge)
