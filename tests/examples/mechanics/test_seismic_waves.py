"""Affine material response, rigid motions, and free elastic time evolution."""

import numpy as np
import pytest

from numga import stack
from numga.algebras import VGA2D, VGA3D
from examples import instantiate
from examples.mechanics.seismic_waves import core


@pytest.fixture(scope="module")
def disk():
    mesh, vertices = core.Mesh.concentric_disk(np.array([0.6, 1.0]), 6)
    centres = (mesh.incidence[2, 0].mean(axis=0) * vertices).batch()
    inner = centres.norm_squared() < 0.6**2
    density = core.as_scalar(np.where(inner, 2.0, 1.0)).field()
    lame_lambda = core.as_scalar(np.where(inner, 3.0, 2.0))
    lame_mu = core.as_scalar(np.where(inner, 1.4, 1.0))
    material = core.constitutive(lame_lambda, lame_mu)
    operators = core.Operators(mesh, vertices, material, core.viscous(core.as_scalar(0.0)), density)
    return operators, lame_lambda.field(), lame_mu.field()


def test_rigid_motions_do_no_elastic_work(disk):
    operators, *_ = disk
    vertices = operators.vertices
    translation = vertices * 0 + 0.3 * core.mv.x.dual() - 0.2 * core.mv.y.dual()
    rotation = vertices.commutator(0.2 * core.mv.xy)
    for displacement in (translation, rotation):
        np.testing.assert_allclose(operators.stiffness(displacement).kernel, 0, atol=1e-12)


def test_affine_strain_recovers_heterogeneous_energy(disk):
    operators, lame_lambda, lame_mu = disk
    longitudinal, transverse, shear = 0.12, -0.04, 0.035
    x, y = core.mv.x & operators.vertices, core.mv.y & operators.vertices
    displacement = (core.mv.x.dual() * (longitudinal * x + shear * y)
                    + core.mv.y.dual() * (shear * x + transverse * y))

    density = (lame_lambda * ((longitudinal + transverse)**2 / 2)
               + lame_mu * (longitudinal**2 + transverse**2 + 2 * shear**2))
    expected = (density * operators.areas).sites.sum()
    actual = core.energy(operators, displacement, displacement * 0)
    np.testing.assert_allclose(actual.kernel, expected.kernel, rtol=1e-11, atol=1e-12)

    _, dilation_rate, vorticity = next(core.reconstruct(operators, [(displacement * 0, displacement)]))
    np.testing.assert_allclose(dilation_rate.kernel, longitudinal + transverse, atol=1e-12)
    np.testing.assert_allclose(vorticity.kernel, 0, atol=1e-12)
    np.testing.assert_allclose(operators.stiffness(displacement).sites.sum().kernel, 0, atol=1e-12)

    spinning = core.mv.y.dual() * x - core.mv.x.dual() * y
    _, dilation_rate, vorticity = next(core.reconstruct(operators, [(spinning * 0, spinning)]))
    np.testing.assert_allclose(dilation_rate.kernel, 0, atol=1e-12)
    np.testing.assert_allclose(vorticity.kernel, 2, atol=1e-12)


def test_free_dynamics_preserves_momentum_with_bounded_energy(disk):
    operators, *_ = disk
    vertices = operators.vertices
    centre = -0.4 * core.mv.x.dual() + 0.2 * core.mv.y.dual()
    envelope = (-(vertices - centre).norm_squared() / 0.08).exp()
    velocity = 0.02 * envelope * core.mv.x.dual()
    velocity = velocity - (operators.masses * velocity).sites.sum() / operators.masses.sites.sum()
    displacement = velocity * 0
    initial_energy = core.energy(operators, displacement, velocity)
    initial_momentum = (operators.masses * velocity).sites.sum()
    energies = []
    for displacement, velocity in core.evolve(operators, displacement, velocity, 0.0, 0.002, 20, 20):
        energies.append(core.energy(operators, displacement, velocity))
        momentum = (operators.masses * velocity).sites.sum()
        np.testing.assert_allclose((momentum - initial_momentum).kernel, 0, atol=1e-12)
    relative_energy = (stack(energies) / initial_energy).kernel
    assert np.max(np.abs(relative_energy - 1)) < 0.002
    assert np.max(displacement.norm().kernel) > 1e-4


@pytest.mark.parametrize("algebra", [VGA2D, VGA3D])
def test_concentric_disk_has_oriented_faces_in_plane_and_space(algebra):
    mesh_module = instantiate("examples.util.mesh", algebra)
    mesh, vertices = mesh_module.Mesh.concentric_disk(np.array([0.6, 1.0]), 5)
    corners = mesh.incidence[2, 0] * vertices
    patches = (corners[1] - corners[0]) & (corners[2] - corners[0])
    assert np.all((mesh_module.mv.xy & patches) > 0)
    np.testing.assert_allclose(vertices.batch()[0].kernel, 0, atol=0)


def test_inviscid_fluid_has_no_shear_restoring_force():
    mesh, vertices = core.Mesh.concentric_disk(np.array([1.0]), 5)
    fluid = core.constitutive(core.as_scalar(3.0), core.as_scalar(0.0))
    operators = core.Operators(mesh, vertices, fluid, core.viscous(core.as_scalar(0.0)), core.as_scalar(2.0))
    x, y = core.mv.x & vertices, core.mv.y & vertices
    shearing = 0.1 * (core.mv.x.dual() * y + core.mv.y.dual() * x)
    np.testing.assert_allclose(operators.stiffness(shearing).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(core.energy(operators, shearing, shearing * 0).kernel, 0, atol=1e-12)
    compression = -0.1 * vertices
    assert core.energy(operators, compression, compression * 0) > 0


def test_viscous_liquid_drags_shear_but_not_rigid_motion():
    mesh, vertices = core.Mesh.concentric_disk(np.array([1.0]), 5)
    fluid = core.constitutive(core.as_scalar(3.0), core.as_scalar(0.0))
    operators = core.Operators(mesh, vertices, fluid, core.viscous(core.as_scalar(0.1)), core.as_scalar(2.0))
    translation = vertices * 0 + 0.3 * core.mv.x.dual() - 0.2 * core.mv.y.dual()
    spin = vertices.commutator(0.2 * core.mv.xy)
    for rigid in (translation, spin):
        np.testing.assert_allclose(operators.damping(rigid).kernel, 0, atol=1e-12)
    # A shearing flow is dragged, and the drag takes energy at a positive rate.
    x, y = core.mv.x & vertices, core.mv.y & vertices
    shearing = 0.1 * (core.mv.x.dual() * y + core.mv.y.dual() * x)
    assert (operators.damping(shearing) & shearing).sites.sum() > 0
