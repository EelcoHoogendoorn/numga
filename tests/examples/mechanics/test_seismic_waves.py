"""Affine material response, rigid motions, free elastic evolution, and cross-section rendering."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import NumpyContext, stack
from numga.algebras import VGA2D, VGA3D
from examples.mechanics.seismic_waves import core, render, scenarios
from examples.util.shapes import concentric_disk


@pytest.fixture(scope="module")
def disk():
    topology, coordinates = concentric_disk(np.array([0.6, 1.0]), 5)
    vertices = core.mv.vector(coordinates).dual().field()                     # Point-[V]
    centres = topology.incidence[2, 0].mean(axis=0) * vertices                # Point-[F]
    regions = (centres.antinorm() > 0.6).astype(int)                          # [F]
    density = core.mv.scalar(np.array([2.0, 1.0])[regions, None]).field()     # Scalar+[F]
    lame_lambda = core.mv.scalar(np.array([3.0, 2.0])[regions, None]).field()  # Scalar+[F]
    lame_mu = core.mv.scalar(np.array([1.4, 1.0])[regions, None]).field()     # Scalar+[F]
    areas = core.triangle_areas(topology, vertices)                           # Scalar+[F]
    gradients = core.simplex_gradients(topology, vertices)                    # [3] Plane+[F]
    stiffness = core.assembled_stiffness(topology, gradients, areas, core.constitutive_form(lame_lambda, lame_mu))
    damping = core.assembled_stiffness(topology, gradients, areas, core.viscous_form(core.mv.scalar([0.0])))
    masses = core.vertex_masses(topology, areas, density)                     # Scalar+[V]
    divergence = core.divergence_operator(topology, gradients)                # (Scalar+ <- Point-)[F <- V]
    rotation = core.rotation_operator(topology, gradients)                    # (Bivector+ <- Point-)[F <- V]
    return topology, vertices, regions, areas, masses, stiffness, damping, divergence, rotation, lame_lambda, lame_mu


def test_rigid_motions_do_no_elastic_work(disk):
    _, vertices, _, _, _, stiffness, *_ = disk
    translation = 0 * vertices + 0.3 * core.mv.x.dual() - 0.2 * core.mv.y.dual()
    spin = vertices.commutator(0.2 * core.mv.xy)
    for displacement in (translation, spin):
        assert (stiffness(displacement).norm() < 1e-12).all()


def test_affine_strain_recovers_heterogeneous_energy(disk):
    _, vertices, _, areas, masses, stiffness, _, divergence, rotation, lame_lambda, lame_mu = disk
    longitudinal, transverse, shear = 0.12, -0.04, 0.035
    x, y = core.mv.x & vertices, core.mv.y & vertices
    displacement = (core.mv.x.dual() * (longitudinal * x + shear * y)
                    + core.mv.y.dual() * (shear * x + transverse * y))

    energy_density = (lame_lambda * ((longitudinal + transverse)**2 / 2)
                      + lame_mu * (longitudinal**2 + transverse**2 + 2 * shear**2))
    expected = (energy_density * areas).sites.sum()
    actual = core.field_energy(masses, stiffness, displacement, 0 * displacement)
    assert ((actual - expected).abs() < 1e-11).all()

    _, dilation_rate, vorticity = next(core.wave_readings(divergence, rotation, [(0 * displacement, displacement)]))
    assert ((dilation_rate - (longitudinal + transverse)).abs() < 1e-12).all()
    assert (vorticity.abs() < 1e-12).all()
    assert (stiffness(displacement).sites.sum().norm() < 1e-12).all()

    spinning = core.mv.y.dual() * x - core.mv.x.dual() * y
    _, dilation_rate, vorticity = next(core.wave_readings(divergence, rotation, [(0 * spinning, spinning)]))
    assert (dilation_rate.abs() < 1e-12).all()
    assert ((vorticity - 2).abs() < 1e-12).all()


def test_free_dynamics_preserves_momentum_with_bounded_energy(disk):
    _, vertices, _, _, masses, stiffness, damping, *_ = disk
    centre = -0.4 * core.mv.x.dual() + 0.2 * core.mv.y.dual()
    envelope = (-(vertices - centre).scalar_norm_squared() / 0.08).exp()
    velocity = 0.02 * envelope * core.mv.x.dual()
    velocity = velocity - (masses * velocity).sites.sum() / masses.sites.sum()
    displacement = 0 * velocity
    inverse_inertia = (1 / masses) * core.Force.dual()
    acceleration_map = -inverse_inertia(stiffness)
    drag_map = -inverse_inertia(damping)
    initial_energy = core.field_energy(masses, stiffness, displacement, velocity)
    initial_momentum = (masses * velocity).sites.sum()
    energies = []
    for displacement, velocity in core.wave_evolution(acceleration_map, drag_map, displacement, velocity, 0.0, 0.002, 5, 4):
        energies.append(core.field_energy(masses, stiffness, displacement, velocity))
        momentum = (masses * velocity).sites.sum()
        assert ((momentum - initial_momentum).antinorm() < 1e-12).all()
    relative_energy = stack(tuple(energies)) / initial_energy
    assert ((relative_energy - 1).abs() < 0.002).all()
    assert (displacement.antinorm() > 1e-4).any()


@pytest.mark.parametrize("algebra", [VGA2D, VGA3D])
def test_concentric_disk_has_oriented_faces_in_plane_and_space(algebra):
    mv = NumpyContext(algebra).multivector
    topology, coordinates = concentric_disk(np.array([0.6, 1.0]), 5)
    vertices = mv(algebra.gatype.vector().derive.within(VGA2D), coordinates).dual().field()
    corners = topology.incidence[2, 0] * vertices
    patches = (corners[1] - corners[0]) & (corners[2] - corners[0])
    assert ((mv.xy & patches) > 0).all()
    assert (vertices.batch()[0].antinorm() <= 0).all()


def test_inviscid_fluid_and_viscous_shear_drag():
    topology, coordinates = concentric_disk(np.array([1.0]), 4)
    vertices = core.mv.vector(coordinates).dual().field()
    areas = core.triangle_areas(topology, vertices)
    gradients = core.simplex_gradients(topology, vertices)
    stiffness = core.assembled_stiffness(topology, gradients, areas,
                                         core.constitutive_form(core.mv.scalar([3.0]), core.mv.scalar([0.0])))
    damping = core.assembled_stiffness(topology, gradients, areas, core.viscous_form(core.mv.scalar([0.1])))
    masses = core.vertex_masses(topology, areas, core.mv.scalar([2.0]))

    x, y = core.mv.x & vertices, core.mv.y & vertices
    shearing = 0.1 * (core.mv.x.dual() * y + core.mv.y.dual() * x)
    assert (stiffness(shearing).norm() < 1e-12).all()
    assert (core.field_energy(masses, stiffness, shearing, 0 * shearing).abs() < 1e-12).all()
    assert ((damping(shearing) & shearing).sites.sum() > 0).all()

    compression = -0.1 * vertices
    assert (core.field_energy(masses, stiffness, compression, 0 * compression) > 0).all()

    translation = 0 * vertices + 0.3 * core.mv.x.dual() - 0.2 * core.mv.y.dual()
    spin = vertices.commutator(0.2 * core.mv.xy)
    for rigid in (translation, spin):
        assert (damping(rigid).norm() < 1e-12).all()


def test_normal_modes_and_cross_section_rendering(disk):
    topology, vertices, regions, _, masses, stiffness, _, divergence, rotation, *_ = disk
    squared_frequencies, shapes = core.normal_modes(stiffness, masses, 6)
    assert (squared_frequencies[:3].abs() < 1e-10).all()
    assert (squared_frequencies[3:] > 0).all()

    figure = render.materials_figure(topology, vertices, regions)
    plt.close(figure)

    moon, velocity = scenarios.impact_setup(vertices)
    readings = core.wave_readings(divergence, rotation, [(0 * velocity, velocity)])
    wave_frame = next(render.waves_animation(topology, vertices, regions, moon[:1], scenarios.moon_radius,
                                             scenarios.displacement_scale, scenarios.dilation_scale,
                                             scenarios.vorticity_scale, readings))
    assert wave_frame.shape[0] == render.n_pixels

    travelling = core.travelling_waves(divergence, rotation, shapes[3:4], shapes[4:5], scenarios.mode_peak, 1)
    mode_frame = next(render.modes_animation(topology, vertices, np.ones_like(regions),
                                             scenarios.displacement_scale, travelling))
    assert mode_frame.shape == wave_frame.shape
