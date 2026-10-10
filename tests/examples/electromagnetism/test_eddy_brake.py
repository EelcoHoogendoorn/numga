"""Charge conservation, magnetic braking, and field covariance of the conducting disc solve."""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.sparse import SparseExtensor
from examples.electromagnetism.eddy_brake import core


def test_a_uniform_field_is_cancelled_by_the_charge_distribution():
    radius, rings, sectors = 0.1, 6, 36
    strength, speed, conductance = 0.2, 1.0, 5.8e4
    mesh = core.Mesh.disk(radius, rings, sectors)
    conductivity = conductance * (core.Vector + core.mv.x * (core.mv.x | core.Vector))
    response = core.solve(mesh, core.mv.xy * strength, core.mv.xy * speed, conductivity)

    # The midpoint edge integral is exact for rigid rotation in a uniform field.
    potential = strength * speed * mesh.vertices.scalar_norm_squared() / 2
    potential = potential - potential.batch()[0]
    np.testing.assert_allclose(response.potential.kernel, potential.kernel, atol=1e-12)
    np.testing.assert_allclose(response.flux.kernel, 0, atol=1e-9)
    np.testing.assert_allclose(response.current.kernel, 0, atol=1e-8)
    np.testing.assert_allclose(response.torque.kernel, 0, atol=1e-12)


def test_localized_field_conserves_charge_and_turns_mechanical_power_into_heat():
    radius, rings, sectors = 0.1, 8, 48
    strength, width, offset = 0.2, 0.02, 0.055
    speed, conductance = 1.0, 5.8e4
    mesh = core.Mesh.disk(radius, rings, sectors)
    magnetic = core.field(mesh.edge_midpoints, core.mv.x * offset, core.mv.xy, strength, width)
    spin = core.mv.xy * speed
    principal = conductance * np.array([[1.0, 1.0], [1.9, 0.1], [0.1, 1.9]])
    conductivity = core.fibre_conductivity(mesh, principal, 0, radius, conductance)
    response = core.solve(mesh, magnetic, spin, conductivity)

    charge_balance = ~mesh.d0 * response.flux
    heat = response.heating.sites.sum()
    mechanical_loss = spin | response.torque
    np.testing.assert_allclose(charge_balance.kernel, 0, atol=1e-9)
    np.testing.assert_allclose(mechanical_loss.kernel, heat.kernel, atol=1e-12)
    assert np.all(heat.kernel > 0)
    assert np.all(np.sum(spin.kernel * response.torque.kernel, axis=-1) < 0)
    # With the same mean conductance, resistance across the fibres reduces both current loops.
    assert np.all(heat[1:].kernel < heat[0].kernel / 2)


def test_speed_field_and_conductance_have_their_physical_scalings():
    radius, rings, sectors = 0.1, 6, 36
    strength, width, offset = 0.2, 0.02, 0.055
    conductance = 5.8e4
    speeds = np.array([1.0, 2.0, 1.0, -1.0])
    strengths = np.array([1.0, 1.0, 2.0, 1.0])
    mesh = core.Mesh.disk(radius, rings, sectors)
    magnetic = core.field(mesh.edge_midpoints, core.mv.x * offset, core.mv.xy, strength, width)
    response = core.solve(mesh, magnetic * strengths,
                          core.mv.xy * speeds, conductance * core.Vector)
    doubled = core.solve(mesh, magnetic, core.mv.xy, conductance * 2 * core.Vector)

    current_factors = speeds * strengths
    torque_factors = speeds * strengths ** 2
    heat_factors = speeds ** 2 * strengths ** 2
    np.testing.assert_allclose(response.current.kernel,
                               (response.current[0] * current_factors).kernel, atol=1e-8)
    np.testing.assert_allclose(response.torque.kernel,
                               (response.torque[0] * torque_factors).kernel, atol=1e-12)
    np.testing.assert_allclose(response.heating.kernel,
                               (response.heating[0] * heat_factors).kernel, atol=1e-12)
    np.testing.assert_allclose(doubled.potential.kernel, response.potential[0].kernel, atol=1e-12)
    np.testing.assert_allclose(doubled.current.kernel, (2 * response.current[0]).kernel, atol=1e-8)
    np.testing.assert_allclose(doubled.torque.kernel, (2 * response.torque[0]).kernel, atol=1e-12)
    np.testing.assert_allclose(doubled.heating.kernel, (2 * response.heating[0]).kernel, atol=1e-12)


def test_turning_the_disc_and_magnet_turns_the_solution():
    radius, rings, sectors = 0.1, 6, 36
    strength, width, offset = 0.2, 0.02, 0.055
    conductance, angle = 5.8e4, 0.7
    mesh = core.Mesh.disk(radius, rings, sectors)
    centre, plane, spin = core.mv.x * offset, core.mv.xy, core.mv.xy
    turn = (core.mv.xz * (-angle / 2)).exp()
    turned_mesh = mesh.copy(vertices=turn >> mesh.vertices)
    magnetic = core.field(mesh.edge_midpoints, centre, plane, strength, width)
    turned_magnetic = core.field(turned_mesh.edge_midpoints, turn >> centre,
                                 turn >> plane, strength, width)
    conductivity = conductance * (core.Vector + core.mv.x * (core.mv.x | core.Vector))
    turned_conductivity = turn >> conductivity(turn << core.Vector)
    response = core.solve(mesh, magnetic, spin, conductivity)
    turned = core.solve(turned_mesh, turned_magnetic, turn >> spin, turned_conductivity)

    np.testing.assert_allclose(turned.potential.kernel, response.potential.kernel, atol=1e-12)
    np.testing.assert_allclose(turned.flux.kernel, response.flux.kernel, atol=1e-9)
    np.testing.assert_allclose(turned.current.kernel, (turn >> response.current).kernel, atol=1e-8)
    np.testing.assert_allclose((turned.torque - (turn >> response.torque)).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(turned.heating.kernel, response.heating.kernel, atol=1e-12)


def test_braking_power_converges_under_mesh_refinement():
    radius, strength, width, offset = 0.1, 0.2, 0.02, 0.055
    conductance, rings, sectors = 5.8e4, 4, 24
    powers = []
    for refinement in (1, 2, 4):
        mesh = core.Mesh.disk(radius, rings * refinement, sectors * refinement)
        magnetic = core.field(mesh.edge_midpoints, core.mv.x * offset, core.mv.xy, strength, width)
        principal = conductance * np.array([[1.0, 1.0], [1.9, 0.1], [0.1, 1.9]])
        conductivity = core.fibre_conductivity(mesh, principal, 0, radius, conductance)
        response = core.solve(mesh, magnetic, core.mv.xy, conductivity)
        powers.append(response.heating.sites.sum().kernel)
    # Midpoint edge integration and the polygonal rim both approach the smooth-disc problem.
    assert np.all(abs(powers[2] - powers[1]) < 0.4 * abs(powers[1] - powers[0]))


def test_annular_fibres_lose_mesh_orientation_bias_under_uniform_refinement():
    from matplotlib.tri import LinearTriInterpolator, Triangulation

    radius, strength, width, offset, conductance = 0.1, 1.0, 0.02, 0.055, 5.8e4
    inner_radius, outer_radius = 0.2 * radius, 0.85 * radius
    material_radii = np.array([inner_radius, outer_radius, radius])
    divisions = (6, 12, 24)
    principal = conductance * np.array([[1.9, 0.1], [0.1, 1.9]])
    angles = np.array([0, np.pi / 12, np.pi / 6])
    turns = (core.mv.xy * (-angles / 2)).exp()
    errors = np.empty((len(divisions), len(principal)))

    for index, density in enumerate(divisions):
        mesh = core.Mesh.concentric_disk(material_radii, density)
        conductivity = core.fibre_conductivity(mesh, principal, inner_radius, outer_radius, conductance)

        # Zero streamfunction on the rim gives closed currents. Solve their full steady
        # response so this check measures spatial discretization, without a modal cutoff.
        boundary = np.unique(mesh.edges[mesh.boundary_edges])
        vertex_count = len(mesh.vertices.batch())
        interior = np.setdiff1d(np.arange(vertex_count), boundary)
        embedder = ~SparseExtensor.selection(core.context, interior, vertex_count) * core.Scalar
        gradient = mesh.reconstruction((mesh.d0 * core.Scalar)(embedder))
        current = core.spdiag(mesh.face_planes | core.Vector)(gradient)
        resistivity = conductivity.batch().lstsq(core.Vector).field()
        resistance = current.adjoint()(core.spdiag(mesh.triangle_areas * resistivity)(current))
        # Pull the stationary laboratory magnet into each orientation's material frame.
        centres = turns << (core.mv.x * offset)
        magnetic = core.field(mesh.face_centers, centres, core.mv.xy, strength, width)
        electric = magnetic | -(core.mv.xy | mesh.face_centers)
        drive = current.adjoint()(mesh.triangle_areas * electric)
        streamfunction = embedder(resistance.solve(drive[:, None]))

        # checks: compare the world fields on one fixed grid, including amplitude errors.
        sample_count = 61
        axis = np.linspace(-0.9 * radius, 0.9 * radius, sample_count)
        horizontal, vertical = np.meshgrid(axis, axis)
        inside = horizontal ** 2 + vertical ** 2 < (0.9 * radius) ** 2
        horizontal, vertical = horizontal[inside], vertical[inside]
        points = (turns >> mesh.vertices).cast(core.ga.subspace("x y")).kernel
        fields = np.array([
            [np.asarray(LinearTriInterpolator(Triangulation(point[:, 0], point[:, 1], mesh.faces), values)
                        (horizontal, vertical)) for values in cases]
            for point, cases in zip(points, streamfunction.kernel[..., 0])
        ])
        relative_error = np.linalg.norm(fields - fields[0], axis=-1) / np.linalg.norm(fields[0], axis=-1)
        errors[index] = relative_error.max(axis=0)

    assert np.all(errors[1:] < 0.65 * errors[:-1])
    assert np.all(errors[-1] < 0.025)


def test_isotropic_material_recovers_the_cotangent_stiffness():
    radius, rings, sectors, conductance = 0.1, 6, 36, 2.5
    mesh = core.Mesh.disk(radius, rings, sectors)
    gradient = mesh.reconstruction(mesh.d0 * core.Scalar)
    material = core.spdiag(mesh.triangle_areas * conductance * core.Vector)
    balance = gradient.adjoint()(material(gradient))
    cotangent = (~mesh.d0 * core.spdiag(conductance * mesh.edge_ratio) * mesh.d0) * core.Scalar
    # The face gradient is constant, so integrating its material pairing at the centroid is exact.
    np.testing.assert_allclose((balance - cotangent).cells.kernel, 0, atol=1e-10)


def test_free_rotation_converts_kinetic_energy_into_heat_with_closed_currents():
    radius, rings, sectors = 0.1, 6, 36
    strength, width, offset, conductance = 0.2, 0.02, 0.055, 5.8e4
    inertia, step, steps = 0.005, 0.1, 30
    initial_speeds = np.array([2.0, 1.0])
    mesh = core.Mesh.disk(radius, rings, sectors)
    conductivity = conductance * (0.1 * core.Vector + 1.8 * core.mv.x * (core.mv.x | core.Vector))
    conductivity = conductivity.broadcast_to(initial_speeds.shape)
    orientation = core.mv.rotor().broadcast_to(initial_speeds.shape)
    heat = (mesh.triangle_areas * 0).broadcast_to(initial_speeds.shape)
    states = list(core.braking(mesh, core.mv.x * offset, core.mv.xy, strength, width, conductivity,
                               inertia, orientation, core.mv.xy * initial_speeds, heat, step, steps))

    energy = stack([state.heat.sites.sum() + inertia / 2 * state.spin.scalar_norm_squared() for state in states])
    speed = stack([-(core.mv.xy | state.spin) for state in states])
    flux = stack([mesh.reconstruction.adjoint()(mesh.triangle_areas * state.current) for state in states])
    np.testing.assert_allclose((energy - energy[0]).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((~mesh.d0 * flux).kernel, 0, atol=1e-9)
    assert np.all(speed.kernel > 0)
    assert np.all(np.diff(speed.kernel, axis=0) < 0)
    assert np.all(states[-1].heat.sites.sum().kernel > 0)


def test_free_rotation_converges_at_second_order_in_time():
    radius, rings, sectors = 0.1, 4, 24
    strength, width, offset, conductance = 0.2, 0.02, 0.055, 5.8e4
    inertia, initial_speed, duration, steps = 0.005, 2.0, 2.0, 10
    mesh = core.Mesh.disk(radius, rings, sectors)
    conductivity = conductance * (0.5 * core.Vector + core.mv.x * (core.mv.x | core.Vector))
    heat = mesh.triangle_areas * 0
    final = []
    for refinement in (1, 2, 4):
        count = steps * refinement
        states = core.braking(mesh, core.mv.x * offset, core.mv.xy, strength, width, conductivity,
                              inertia, core.mv.rotor(), core.mv.xy * initial_speed, heat, duration / count, count)
        final.append(list(states)[-1])

    # Halving the step quarters the endpoint error in both speed and orientation.
    coarse_spin = np.linalg.norm((final[1].spin - final[0].spin).kernel)
    fine_spin = np.linalg.norm((final[2].spin - final[1].spin).kernel)
    coarse_turn = np.linalg.norm((final[1].orientation - final[0].orientation).kernel)
    fine_turn = np.linalg.norm((final[2].orientation - final[1].orientation).kernel)
    assert fine_spin < 0.35 * coarse_spin
    assert fine_turn < 0.35 * coarse_turn


def test_axisymmetric_materials_brake_in_batches():
    radius, rings, sectors = 0.1, 6, 36
    strength, width, offset, conductance = 0.2, 0.02, 0.055, 5.8e4
    inertia, initial_speed, step, steps = 0.005, 2.0, 0.1, 40
    principal = conductance * np.array([[1.0, 1.0], [1.9, 0.1], [0.1, 1.9]])
    mesh = core.Mesh.disk(radius, rings, sectors)
    conductivity = core.fibre_conductivity(mesh, principal, 0, radius, conductance)
    orientation = core.mv.rotor().broadcast_to(conductivity.shape)
    spin = (core.mv.xy * initial_speed).broadcast_to(conductivity.shape)
    heat = (mesh.triangle_areas * 0).broadcast_to(conductivity.shape)
    states = list(core.braking(mesh, core.mv.x * offset, core.mv.xy, strength, width, conductivity,
                               inertia, orientation, spin, heat, step, steps))

    energy = stack([state.heat.sites.sum() + inertia / 2 * state.spin.scalar_norm_squared() for state in states])
    speed = stack([-(core.mv.xy | state.spin) for state in states])
    np.testing.assert_allclose((energy - energy[0]).kernel, 0, atol=1e-12)
    assert np.all(np.diff(speed.kernel, axis=0) < 0)
    # With the same mean conductance, the isotropic disc brakes hardest and circumferential fibres least.
    assert np.all(np.diff(speed[-1].kernel, axis=0) > 0)


def test_moving_material_faces_gain_heat_and_preserve_total_energy():
    radius, rings, sectors = 0.1, 4, 24
    strength, width, offset, conductance = 0.2, 0.02, 0.055, 5.8e4
    inertia, step, steps = 0.005, 0.1, 20
    initial_speeds = np.array([2.0, 1.0])
    initial_angles = np.array([0.0, 0.3])
    principal = conductance * np.array([[1.9, 0.1], [0.1, 1.9]])
    mesh = core.Mesh.disk(radius, rings, sectors)
    conductivity = core.fibre_conductivity(mesh, principal, 0, radius, conductance)
    orientation = (core.mv.xy * (-initial_angles / 2)).exp()
    heat = (mesh.triangle_areas * 0).broadcast_to(conductivity.shape)
    states = list(core.braking(mesh, core.mv.x * offset, core.mv.xy,
                               strength, width, conductivity, inertia, orientation,
                               core.mv.xy * initial_speeds, heat, step, steps))

    thermal = stack([state.heat for state in states])
    energy = stack([state.heat.sites.sum() + inertia / 2 * state.spin.scalar_norm_squared()
                    for state in states])
    turns = stack([state.orientation for state in states])
    vertices = stack([state.vertices for state in states])
    np.testing.assert_allclose((energy - energy[0]).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((vertices - (turns >> mesh.vertices)).kernel, 0, atol=1e-12)
    assert np.all(np.diff(thermal.kernel, axis=0) >= -1e-14)
    assert np.all(states[-1].heat.sites.sum().kernel > 0)
    assert np.linalg.norm((vertices[-1] - vertices[0]).kernel) > radius


def test_coasting_transports_nonuniform_heat_with_the_material():
    radius, rings, sectors = 0.1, 4, 24
    strength, width, offset, conductance = 0.0, 0.02, 0.055, 5.8e4
    inertia, step, steps = 0.005, 0.1, 12
    speeds = np.array([2.0, -1.0])
    angles = np.array([0.0, 0.3])
    mesh = core.Mesh.disk(radius, rings, sectors)
    directions = stack((core.mv.x, core.mv.y))
    conductivity = (conductance * directions * (directions | core.Vector)).sum(axis=-1)
    conductivity = conductivity.broadcast_to(speeds.shape)
    orientation = (core.mv.xy * (-angles / 2)).exp()
    spin = core.mv.xy * speeds
    # A warmer side identifies material faces even when no current adds further heat.
    heat = mesh.triangle_areas * (1 + (mesh.face_centers | core.mv.x) / radius)
    heat = heat.broadcast_to(conductivity.shape)
    states = list(core.braking(mesh, core.mv.x * offset, core.mv.xy,
                               strength, width, conductivity, inertia,
                               orientation, spin, heat, step, steps))

    thermal = stack([state.heat for state in states])
    spins = stack([state.spin for state in states])
    elapsed = step * steps
    turn = (spin * (-elapsed / 2)).exp()
    np.testing.assert_allclose((thermal - heat).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((spins - spin).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((states[-1].vertices - (turn >> states[0].vertices)).kernel,
                               0, atol=1e-12)

    # The heat-weighted position turns in world space while each face keeps its own heat.
    first_centers = mesh.copy(states[0].vertices).face_centers
    last_centers = mesh.copy(states[-1].vertices).face_centers
    first = (first_centers * heat).sites.sum() / heat.sites.sum()
    last = (last_centers * heat).sites.sum() / heat.sites.sum()
    np.testing.assert_allclose((last - (turn >> first)).kernel, 0, atol=1e-12)
    assert np.linalg.norm((last - first).kernel) > radius / 4


def test_material_frame_heating_matches_a_solve_on_the_rotated_world_mesh():
    radius, rings, sectors = 0.1, 4, 24
    strength, width, offset, conductance = 0.2, 0.02, 0.055, 5.8e4
    inertia, speed, angle, step, steps = 0.005, 2.0, 0.4, 0.1, 1
    mesh = core.Mesh.disk(radius, rings, sectors)
    tilt = (core.mv.xz * 0.2).exp()
    plane = tilt >> core.mv.xy
    centre = tilt >> (core.mv.x * offset)
    orientation = (plane * (-angle / 2)).exp() * tilt
    spin = plane * speed
    radial = mesh.face_centers.normalized()
    conductivity = conductance * (0.1 * core.Vector + 1.8 * radial * (radial | core.Vector))
    heat = mesh.triangle_areas * 0
    states = list(core.braking(mesh, centre, plane, strength, width,
                               conductivity[None], inertia, orientation[None],
                               spin[None], heat[None], step, steps))

    # Solve independently in world coordinates, leaving the magnet fixed as the mesh turns.
    midpoint = (spin * (-step / 4)).exp() * orientation
    world_mesh = mesh.copy(vertices=midpoint >> mesh.vertices)
    magnetic = core.field(world_mesh.edge_midpoints, centre, plane, strength, width)
    material = midpoint >> conductivity(midpoint << core.Vector)
    response = core.solve(world_mesh, magnetic, plane, material)
    drag = plane | response.torque
    midpoint_spin = spin / (1 + drag * step / (2 * inertia))
    gained_heat = step * midpoint_spin.scalar_norm_squared() * response.heating
    np.testing.assert_allclose((states[-1].heat[0] - gained_heat).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((states[-1].spin[0] - (2 * midpoint_spin - spin)).kernel, 0, atol=1e-12)


def test_material_heat_and_motion_converge_at_second_order_in_time():
    radius, rings, sectors = 0.1, 4, 24
    strength, width, offset, conductance = 0.2, 0.02, 0.055, 5.8e4
    inertia, speed, elapsed, steps = 0.005, 2.0, 1.6, 8
    mesh = core.Mesh.disk(radius, rings, sectors)
    radial = mesh.face_centers.normalized()
    conductivity = conductance * (0.3 * core.Vector + 1.4 * radial * (radial | core.Vector))
    orientation = (core.mv.xy * 0).exp().broadcast_to((1,))
    spin = (core.mv.xy * speed).broadcast_to((1,))
    heat = (mesh.triangle_areas * 0)[None]
    final = []
    for refinement in (1, 2, 4):
        count = steps * refinement
        states = core.braking(mesh, core.mv.x * offset, core.mv.xy, strength, width,
                              conductivity[None], inertia, orientation, spin,
                              heat, elapsed / count, count)
        final.append(list(states)[-1])

    # Refining time resolves both the magnetic heating swept over material faces and the turn.
    coarse_heat = np.linalg.norm((final[1].heat - final[0].heat).kernel)
    fine_heat = np.linalg.norm((final[2].heat - final[1].heat).kernel)
    coarse_turn = np.linalg.norm((final[1].orientation - final[0].orientation).kernel)
    fine_turn = np.linalg.norm((final[2].orientation - final[1].orientation).kernel)
    assert fine_heat < 0.4 * coarse_heat
    assert fine_turn < 0.4 * coarse_turn


def test_inductive_currents_exchange_energy_with_rotation_and_remain_closed():
    radius, rings, sectors, count = 0.1, 3, 16, 8
    strength, width, offset, conductance = 1.0, 0.02, 0.055, 5.8e4
    permeability, inertia = 4e-7 * np.pi, 0.005
    step, steps, substeps = 0.0001, 30, 2
    speeds = np.array([120.0, 0.0, 0.0])
    stored_current = np.array([0.0, 1.0, 0.0])
    preparation_time = 0.01
    principal = conductance * np.array([[1.0, 1.0], [1.5, 0.5], [1.5, 0.5]])
    mesh = core.Mesh.disk(radius, rings, sectors)
    conductivity = core.fibre_conductivity(mesh, principal, 0, radius, conductance)
    modes = core.current_modes(mesh, conductivity, permeability, count)
    magnetic = core.field(mesh.face_centers, core.mv.x * offset, core.mv.xy, strength, width)
    electric = magnetic | -(core.mv.xy | mesh.face_centers)
    amplitudes = -preparation_time * modes.drive(mesh, electric) * stored_current
    orientation = (core.mv.xy * np.zeros(len(speeds))).exp()
    heat = (mesh.triangle_areas * 0).broadcast_to(conductivity.shape)
    harmonics = 16
    forcing = core.periodic_drive(mesh, modes, core.mv.x * offset, core.mv.xy,
                                   strength, width, orientation, harmonics)
    motor_steps = 10
    motor_torque = core.mv.xy * (np.array([0.0, 0.0, 20.0]) *
        (np.arange(steps) < motor_steps)[:, None, None] * np.ones((1, substeps, 1)))
    states = list(core.inductive_braking(mesh, modes, forcing, core.mv.xy,
                                         inertia, orientation, core.mv.xy * speeds,
                                         heat, amplitudes, motor_torque, step))

    thermal = stack([state.heat for state in states])
    magnetic_energy = stack([state.magnetic_energy for state in states])
    spin = stack([state.spin for state in states])
    currents = stack([state.current for state in states])
    streams = stack([state.streamfunction for state in states])
    modal_heat = stack([state.thermal_energy for state in states])
    motor_work = stack([state.motor_work for state in states])
    energy = thermal.sites.sum() + magnetic_energy + inertia / 2 * spin.scalar_norm_squared()
    np.testing.assert_allclose((energy - energy[0] - motor_work).kernel, 0, atol=1e-9)
    np.testing.assert_allclose((thermal.sites.sum() - modal_heat).kernel, 0, atol=1e-11)
    assert np.all(np.diff(thermal.kernel, axis=0) >= -1e-14)
    assert magnetic_energy[1, 0].kernel[0] > 0
    # The prepared opposing current accelerates the initially stationary second rotor.
    assert -(core.mv.xy | states[1].spin[1]).kernel[0] > 0
    # The motor supplies exactly the third disc's energy, then it coasts and brakes.
    assert states[-1].motor_work[2].kernel[0] > 0
    np.testing.assert_allclose((motor_work[motor_steps:, 2] - motor_work[motor_steps, 2]).kernel, 0, atol=1e-12)
    assert (states[-1].spin[2].scalar_norm_squared() < states[motor_steps].spin[2].scalar_norm_squared())

    # The stored magnetic energy agrees with the integrated interaction of the actual currents.
    coupling = core.magnetic_coupling(mesh, permeability)
    integrated = (currents | coupling(currents)).sites.sum() / 2
    np.testing.assert_allclose((integrated - magnetic_energy).kernel, 0, atol=1e-11)

    # Each face's outward flux is the oriented difference of its streamfunction along that
    # edge. Shared edges therefore cancel and the constant boundary stream gives no leakage.
    outward = mesh.face_planes | mesh.triangle_edges
    normal_flux = currents[..., None] | outward
    differences = mesh.d0 * streams
    orientation_by_corner = core.as_scalar(mesh.face_edge_orientation.T).field()
    facing = SparseExtensor.selection(core.context, mesh.face_edges.T, len(mesh.edges))   # [3] [F, E] Scalar
    expected_flux = facing * differences[..., None] * orientation_by_corner
    np.testing.assert_allclose((normal_flux - expected_flux).kernel, 0, atol=1e-9)
    np.testing.assert_allclose(differences.batch()[..., mesh.boundary_edges].kernel, 0, atol=1e-12)

    # Between angular samples, the periodic drive agrees with projecting the magnet directly.
    sample_angles = np.array([-0.37, 0.31, 2.11])
    turns = (core.mv.xy * (-sample_angles[:, None] / 2)).exp() * orientation
    positions = turns >> mesh.face_centers
    magnetic = turns << core.field(positions, core.mv.x * offset, core.mv.xy, strength, width)
    electric = magnetic | -(core.mv.xy | mesh.face_centers)
    projected = modes.drive(mesh, electric)
    periodic = forcing(core.mv.scalar(sample_angles[:, None, None]))
    assert np.linalg.norm((periodic - projected).kernel) < 1e-3 * np.linalg.norm(projected.kernel)


def test_unforced_current_memory_decays_to_heat_at_second_order_in_time():
    radius, rings, sectors, count = 0.1, 3, 16, 8
    strength, width, offset, conductance = 0.0, 0.02, 0.055, 5.8e4
    permeability, inertia, speed = 4e-7 * np.pi, 0.005, 20.0
    step, steps = 0.0002, 10
    mesh = core.Mesh.disk(radius, rings, sectors)
    conductivity = core.fibre_conductivity(mesh, conductance * np.ones((1, 2)), 0, radius, conductance)
    modes = core.current_modes(mesh, conductivity, permeability, count)
    amplitudes = core.mv.scalar(np.full((1, count, 1), 0.02)).field()
    orientation = (core.mv.xy * np.zeros(1)).exp()
    spin = (core.mv.xy * speed).broadcast_to((1,))
    heat = (mesh.triangle_areas * 0)[None]
    elapsed = step * steps
    expected_amplitudes = amplitudes * (-modes.decay * elapsed).exp()
    expected_current = modes.currents(expected_amplitudes)
    harmonics = 16
    forcing = core.periodic_drive(mesh, modes, core.mv.x * offset, core.mv.xy,
                                   strength, width, orientation, harmonics)
    errors = []
    for substeps in (1, 2, 4):
        motor_torque = core.mv.xy * np.zeros((steps, substeps, 1))
        states = list(core.inductive_braking(mesh, modes, forcing, core.mv.xy,
                                             inertia, orientation, spin,
                                             heat, amplitudes, motor_torque, step))
        errors.append(np.linalg.norm((states[-1].current - expected_current).kernel))

    thermal = stack([state.heat for state in states])
    magnetic_energy = stack([state.magnetic_energy for state in states])
    spins = stack([state.spin for state in states])
    energy = thermal.sites.sum() + magnetic_energy
    np.testing.assert_allclose((energy - energy[0]).kernel, 0, atol=1e-11)
    np.testing.assert_allclose((spins - spin).kernel, 0, atol=1e-12)
    assert np.all(np.diff(thermal.kernel, axis=0) >= -1e-14)
    assert np.all(np.diff(magnetic_energy.kernel, axis=0) < 0)
    # The zero-field current has an independent exponential solution, including every mode.
    assert errors[1] < 0.3 * errors[0]
    assert errors[2] < 0.3 * errors[1]
