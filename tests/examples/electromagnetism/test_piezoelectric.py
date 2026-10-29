"""Continuum patch tests, electromechanical reciprocity, charge balance, and beam convergence."""

import numpy as np
import pytest

from numga import stack
from numga.sparse import SparseExtensor
from examples.electromagnetism.piezoelectric import core, scenarios


WIDTH = 5
VOLTAGE = 1e-4
FORCE = 1e-4


@pytest.fixture(scope="module")
def material():
    return scenarios.pzt5a(stack((core.mv.x, core.mv.y)))


@pytest.fixture(scope="module")
def device(material):
    mesh, vertices, poling = core.strip(10, 1, 40, 6)
    areas, gradients = core.triangles(mesh, vertices)
    return mesh, vertices, gradients, poling, core.assemble(mesh, areas, gradients, *material, poling, WIDTH)


@pytest.mark.parametrize("motion", [(core.mv.xy * .27).exp(), core.mv.x], ids=["rotation", "reflection"])
def test_material_follows_its_directions_under_rigid_transformations(material, motion):
    elastic, polarization, permittivity = material
    transformed = scenarios.pzt5a(motion >> stack((core.mv.x, core.mv.y)))
    displacement = .12 * core.axis - .3 * core.pole
    test_displacement = -.2 * core.axis + .1 * core.pole
    gradient = .7 * core.mv.x + .2 * core.mv.y
    test_gradient = -.6 * core.mv.x + .8 * core.mv.y
    electric = .3 * core.mv.xt + .7 * core.mv.yt

    work = elastic(displacement, gradient, test_displacement, test_gradient)
    transformed_work = transformed[0](motion >> displacement, motion >> gradient,
                                      motion >> test_displacement, motion >> test_gradient)
    np.testing.assert_allclose((transformed_work - work).kernel, 0, atol=1e-13)
    np.testing.assert_allclose((transformed[1](motion >> displacement, motion >> gradient)
                               - (motion >> polarization(displacement, gradient))).kernel, 0, atol=1e-13)
    np.testing.assert_allclose((transformed[2](motion >> electric)
                               - (motion >> permittivity(electric))).kernel, 0, atol=1e-13)


@pytest.mark.parametrize("electric, extension, shear, free_permittivity, direction", [
    (core.mv.yt, (-171e-12, 374e-12), 0, 1700, core.pole),
    (core.mv.xt, (0, 0), 584e-12, 1730, core.axis),
], ids=["parallel", "transverse"])
def test_stress_free_response_matches_measured_pzt5a_data(
        material, electric, extension, shear, free_permittivity, direction):
    elastic, polarization, permittivity = material
    # Table A3 gives free piezoelectric strains and permittivity. The constitutive maps
    # use fixed-strain data, so allowing these strains must recover the measurements.
    changes = ((core.axis, core.mv.x, extension[0] * scenarios.field_unit),
               (core.pole, core.mv.y, extension[1] * scenarios.field_unit),
               (core.axis, core.mv.y, shear * scenarios.field_unit / 2),
               (core.pole, core.mv.x, shear * scenarios.field_unit / 2))
    for test_displacement, test_gradient in ((core.axis, core.mv.x),
                                            (core.pole, core.mv.y), (core.axis, core.mv.y)):
        mechanical = sum(elastic(displacement, gradient, test_displacement, test_gradient) * amplitude
                         for displacement, gradient, amplitude in changes)
        electrical = electric & (polarization(test_displacement, test_gradient) & core.time)
        # The published constants are rounded before conversion to the stored material data.
        np.testing.assert_allclose((mechanical - electrical).kernel, 0, atol=1e-5)

    induced = sum((polarization(displacement, gradient) * amplitude
                   for displacement, gradient, amplitude in changes), permittivity(electric))
    measured = direction * (free_permittivity * 8.8541878128e-12 / scenarios.permittivity_unit)
    np.testing.assert_allclose((induced - measured).kernel, 0, atol=1e-5)


def solve(mesh, blocks, voltage, force, floating):
    movement, electric = core.boundary_maps(mesh, floating)
    displacement, potential = core.equilibrium(
        *blocks, movement, electric, core.electrode_potential(mesh, voltage), core.tip_load(mesh, force))
    charges = -(blocks[1].adjugate()(displacement) + blocks[2](potential))
    return displacement, potential, charges


def test_affine_fields_and_rigid_motion_are_reproduced(device):
    mesh, vertices, gradients, _, (mechanical, coupling, dielectric) = device
    horizontal, vertical = core.mv.x & vertices, core.mv.y & vertices
    translation = vertices * 0 + core.axis * .3 + core.pole * .7
    turning = vertices.commutator(core.mv.xy * .2)
    for displacement in (translation, turning):
        np.testing.assert_allclose(core.strain(mesh, displacement, gradients).kernel, 0, atol=1e-12)
        np.testing.assert_allclose(mechanical(displacement).kernel, 0, atol=1e-10)
        np.testing.assert_allclose(coupling.adjugate()(displacement).kernel, 0, atol=1e-10)

    displacement = (core.axis * (.2 * horizontal + .07 * vertical)
                    + core.pole * (.07 * horizontal - .3 * vertical))
    strain = core.strain(mesh, displacement, gradients)
    np.testing.assert_allclose((core.axis & strain(core.axis)).kernel, .2, atol=1e-12, rtol=0)
    np.testing.assert_allclose((core.pole & strain(core.pole)).kernel, -.3, atol=1e-12, rtol=0)
    np.testing.assert_allclose((core.axis & strain(core.pole)).kernel, .07, atol=1e-12, rtol=0)

    potential = 2 * horizontal - 3 * vertical + .7
    expected = -(2 * core.mv.x - 3 * core.mv.y) ^ core.mv.t
    electric = -((mesh.incidence[2, 0] * potential).batch() * (gradients.batch() ^ core.mv.t)).sum(axis=-2)
    np.testing.assert_allclose((electric - expected).kernel, 0, atol=1e-11)
    constant = core.electrode_potential(mesh, 0) + 1
    np.testing.assert_allclose(coupling(constant).kernel, 0, atol=1e-11)
    np.testing.assert_allclose(dielectric(constant).kernel, 0, atol=1e-11)


def test_actuation_and_short_circuit_sensing_are_reciprocal(device):
    mesh, _, _, _, blocks = device
    fixed = np.array([], dtype=int)
    actuator, _, _ = solve(mesh, blocks, VOLTAGE, 0, fixed)
    _, _, sensor_charge = solve(mesh, blocks, 0, FORCE, fixed)
    tip = mesh.reading(mesh.tip)
    top = mesh.reading(mesh.top)
    weights = np.ones(len(mesh.tip))
    weights[[0, -1]] = 0.5
    tip_motion = ((core.mv.y & (tip * actuator)) * core.mv.scalar(weights[:, None]).field()).sites.sum() / weights.sum()
    motion_per_voltage = tip_motion / VOLTAGE
    charge_per_force = (core.mv.scalar([1]) & (top * sensor_charge)).sites.sum() / FORCE

    assert np.abs(motion_per_voltage.kernel).max() > 1
    np.testing.assert_allclose((motion_per_voltage - charge_per_force).kernel, 0, atol=1e-9)


def test_floating_electrode_is_equipotential_and_conserves_charge(device):
    mesh, _, _, _, (mechanical, coupling, dielectric) = device
    displacement, potential, charges = solve(mesh, (mechanical, coupling, dielectric), 0, FORCE, mesh.top)
    n_vertices = mesh.n_vertices
    interior = np.setdiff1d(np.arange(n_vertices), np.concatenate((mesh.bottom, mesh.top)))
    free = np.setdiff1d(np.arange(n_vertices), mesh.clamped)
    top = mesh.reading(mesh.top)
    voltage = (top * potential).sites.mean()

    assert np.abs(voltage.kernel).max() > 1e-6
    np.testing.assert_allclose((potential.batch()[mesh.top] - voltage).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(potential.batch()[mesh.bottom].kernel, 0, atol=1e-12)
    np.testing.assert_allclose((core.mv.scalar([1]) & (top * charges)).sites.sum().kernel, 0, atol=1e-11)
    np.testing.assert_allclose(charges.batch()[interior].kernel, 0, atol=1e-11)
    np.testing.assert_allclose(charges.sites.sum().kernel, 0, atol=1e-11)

    # The clamp carries the applied force; all other nodes are in mechanical equilibrium.
    load = core.tip_load(mesh, FORCE)
    reaction = mechanical(displacement) + coupling(potential) - load
    np.testing.assert_allclose(reaction.batch()[free].kernel, 0, atol=1e-11)
    np.testing.assert_allclose(displacement.batch()[mesh.clamped].kernel, 0, atol=1e-12)
    np.testing.assert_allclose((reaction.sites.sum() + load.sites.sum()).kernel, 0, atol=1e-11)


def test_reversing_poling_reverses_the_coupled_responses(device, material):
    mesh, vertices, gradients, poling, blocks = device
    areas, _ = core.triangles(mesh, vertices)
    reversed_blocks = core.assemble(mesh, areas, gradients, *material, -poling, WIDTH)
    fixed = np.array([], dtype=int)
    displacement, potential, charges = solve(mesh, blocks, VOLTAGE, 0, fixed)
    reverse_displacement, reverse_potential, reverse_charges = solve(mesh, reversed_blocks, VOLTAGE, 0, fixed)
    np.testing.assert_allclose((displacement + reverse_displacement).kernel, 0, atol=1e-11)
    np.testing.assert_allclose((potential - reverse_potential).kernel, 0, atol=1e-11)
    np.testing.assert_allclose((charges - reverse_charges).kernel, 0, atol=1e-11)

    displacement, potential, _ = solve(mesh, blocks, 0, FORCE, mesh.top)
    reverse_displacement, reverse_potential, _ = solve(mesh, reversed_blocks, 0, FORCE, mesh.top)
    np.testing.assert_allclose((displacement - reverse_displacement).kernel, 0, atol=1e-11)
    np.testing.assert_allclose((potential + reverse_potential).kernel, 0, atol=1e-11)


def test_assembled_loads_are_the_variation_of_material_enthalpy(material):
    mesh, vertices, poling = core.strip(10, 1, 16, 4)
    areas, gradients = core.triangles(mesh, vertices)
    mechanical, coupling, dielectric = core.assemble(mesh, areas, gradients, *material, poling, WIDTH)
    horizontal, vertical = core.mv.x & vertices, core.mv.y & vertices
    displacement = (core.axis * (.001 * horizontal.squared() + .002 * horizontal * vertical)
                    + core.pole * (.003 * horizontal * vertical - .004 * vertical.squared()))
    potential = .002 * horizontal * vertical + .003 * horizontal.squared() - .004 * vertical
    movement = core.axis * (.3 * horizontal * vertical) + core.pole * (.2 * horizontal.squared() + .4 * vertical.squared())
    voltage = .3 * horizontal.squared() - .7 * horizontal * vertical + .2 * vertical.squared()

    def enthalpy(displacement, potential):
        density = core.enthalpy(core.strain(mesh, displacement, gradients),
                                -((mesh.incidence[2, 0] * potential).batch() * (gradients.batch() ^ core.mv.t)).sum(axis=-2), *material, poling.batch())
        return (WIDTH * areas.batch() * density).sum()

    step = 1e-4
    difference = (enthalpy(displacement + step * movement, potential + step * voltage)
                  - enthalpy(displacement - step * movement, potential - step * voltage)) / (2 * step)
    work = ((mechanical(displacement) + coupling(potential)) & movement).sites.sum()
    work += ((coupling.adjugate()(displacement) + dielectric(potential)) & voltage).sites.sum()
    np.testing.assert_allclose((difference - work).kernel, 0, atol=1e-10)


def test_refinement_approaches_the_slender_bimorph_response(material):
    length, thickness = 10, 1
    # Zero transverse stress gives these axial coefficients. Constant electric displacement
    # through the thickness includes the electric field induced by the bending itself.
    compliance, free_coupling = 16.4e-12, -171e-12
    free_permittivity = 1700 * 8.8541878128e-12
    stiffness = 1 / compliance / scenarios.stress_unit
    piezoelectric = free_coupling / compliance * scenarios.field_unit / scenarios.stress_unit
    permittivity = (free_permittivity - free_coupling**2 / compliance) / scenarios.permittivity_unit
    beam_response = (1.5 * piezoelectric * permittivity
                     / (stiffness * permittivity + piezoelectric**2 / 4) * (length / thickness)**2)
    errors = []
    for n_length, n_thickness in ((20, 4), (40, 8), (80, 16)):
        mesh, vertices, poling = core.strip(length, thickness, n_length, n_thickness)
        areas, gradients = core.triangles(mesh, vertices)
        blocks = core.assemble(mesh, areas, gradients, *material, poling, WIDTH)
        displacement, _, _ = solve(mesh, blocks, VOLTAGE, 0, np.array([], dtype=int))
        tip = mesh.reading(mesh.tip)
        weights = np.ones(len(mesh.tip))
        weights[[0, -1]] = 0.5
        tip_motion = ((core.mv.y & (tip * displacement)) * core.mv.scalar(weights[:, None]).field()).sites.sum() / weights.sum()
        response = tip_motion.kernel[0] / VOLTAGE
        errors.append(abs(response / beam_response - 1))

    assert errors[1] < .7 * errors[0]
    assert errors[2] < .7 * errors[1]
    assert errors[2] < .08
