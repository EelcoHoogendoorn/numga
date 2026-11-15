"""Coverage tests for thin-walled beam bending and twisting verify section area and second moments of
area, slotted-tube circular-arc formulas, shear-centre decoupling, Saint-Venant torsion, neutral axes
and principal bending directions, kern reciprocity, and single-piece rendering."""

from __future__ import annotations

import numpy as np
from numga import stack

from examples.mechanics.bending import core, render, scenarios
from examples.mechanics.bending.mesh import (
    flange_width,
    section_names,
    section_turns,
    section_walls,
    slot_angle,
    tube_radius,
    tube_wall,
    wall_thickness,
    web_height,
)


def closed_form_second_moments() -> np.ndarray:
    """Closed-form midline second moments of area (`[3, 2, 2]`) describe the I-section, turned
    I-section, and Z-section."""
    flange_area = flange_width * wall_thickness
    across = 2 * wall_thickness * flange_width**3 / 12
    up = 2 * flange_area * (web_height / 2)**2 + wall_thickness * web_height**3 / 12
    aligned = np.diag([across, up])
    angle = np.deg2rad(section_turns[1])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    product = 2 * flange_area * (flange_width / 2) * (web_height / 2)
    z_section = np.array([[across + 2 * flange_area * (flange_width / 2)**2, product], [product, up]])
    return np.stack([aligned, rotation @ aligned @ rotation.T, z_section])


def test_stiffness_holds_the_area_and_the_second_moments():
    area = (2 * flange_width + web_height) * wall_thickness
    for (topology, nodes, thickness), spread in zip(section_walls(), closed_form_second_moments()):
        walls = topology.incidence[1, 0] * nodes
        rigidity = core.bending_stiffness(walls, thickness, scenarios.modulus)
        axial_energy = core.shift_along_beam & rigidity(core.shift_along_beam)
        assert (axial_energy - scenarios.modulus * area).abs() < 1e-12 * area
        square = stack([core.mv.y, core.mv.x])                              # [2] Plane+
        centre = core.centroid(walls, thickness)                            # [] Point-
        turns = (square - (square & centre) * core.mv.w) ^ core.section_plane  # [2] Strain+
        energies = turns & rigidity(turns)                                  # [2] Scalar+
        expected = scenarios.modulus * np.array([spread[1, 1], spread[0, 0]])
        assert ((energies - expected).abs() < 1e-12 * np.abs(expected)).all()


def test_slotted_tube_matches_a_thin_circular_arc():
    topology, nodes, thickness = section_walls()[-1]
    walls = topology.incidence[1, 0] * nodes
    tail, head = walls
    sweep = 2 * np.pi - slot_angle
    half_sweep = sweep / 2
    area = sweep * tube_radius * tube_wall
    centre_x = -tube_radius * np.sin(half_sweep) / half_sweep
    radial = tube_radius**3 * tube_wall
    across = radial * (sweep - np.sin(slot_angle)) / 2 - area * centre_x**2
    up = radial * (sweep + np.sin(slot_angle)) / 2
    shear_x = (-2 * tube_radius * (np.sin(half_sweep) - half_sweep * np.cos(half_sweep))
               / (half_sweep - np.sin(half_sweep) * np.cos(half_sweep)))
    rigidity = core.bending_stiffness(walls, thickness, scenarios.modulus)
    centre = core.centroid(walls, thickness)
    shear = core.shear_centre(topology, nodes, thickness)
    square = stack([core.mv.y, core.mv.x])                                  # [2] Plane+
    turns = (square - (square & centre) * core.mv.w) ^ core.section_plane   # [2] Strain+
    # Polygonal approximation tolerances match the discretized circular arc.
    total_area = ((tail & head).norm() * thickness).sites.sum()
    assert (total_area - area).abs() < 1e-2 * area
    np.testing.assert_allclose(render.section_coordinates(centre), [centre_x, 0.0], rtol=1e-2, atol=1e-12)
    expected = scenarios.modulus * np.array([up, across])
    assert (((turns & rigidity(turns)) - expected).abs() < 1e-2 * expected).all()
    np.testing.assert_allclose(render.section_coordinates(shear), [shear_x, 0.0], rtol=1e-2, atol=1e-12)


def test_warping_from_the_shear_centre_is_invisible_to_bending():
    for topology, nodes, thickness in section_walls():
        shear = core.shear_centre(topology, nodes, thickness)
        couples = core.warping(topology, nodes, thickness, shear).cast(core.Couples)
        assert couples.dual().norm() < 1e-12
    for topology, nodes, thickness in section_walls()[:-1]:
        # Mirror or half-turn symmetry places the shear centre on the centroid.
        shear = core.shear_centre(topology, nodes, thickness)
        centre = core.centroid(topology.incidence[1, 0] * nodes, thickness)
        assert (shear & centre).norm() < 1e-12


def test_torque_twists_about_the_shear_centre():
    torque = 1e-5
    for topology, nodes, thickness in section_walls():
        walls = topology.incidence[1, 0] * nodes
        shear = core.shear_centre(topology, nodes, thickness)
        torsion_constant = core.torsional_stiffness(walls, thickness, scenarios.shear_modulus)
        _, frames, _, _ = core.cantilever(
            topology, nodes, thickness, torque * core.couple_about_beam,
            scenarios.beam_length, scenarios.modulus, scenarios.shear_modulus, 2)
        angle = torque * scenarios.beam_length / torsion_constant
        expected_turn = (core.mv.xy * (-angle / 2)).exp()
        assert ((expected_turn >> core.mv.x.dual()) - (frames[-1] >> core.mv.x.dual())).dual().norm() < 1e-12 * angle
        np.testing.assert_allclose(render.section_coordinates(frames[-1] >> shear),
                                   render.section_coordinates(shear), atol=1e-12)
        # Loading through the shear centre bends the cantilever without twisting it about the beam.
        down_load = scenarios.tip_load * -scenarios.up_direction
        strains, _, _, _ = core.cantilever(
            topology, nodes, thickness, core.tip_points(shear, scenarios.beam_length) & down_load,
            scenarios.beam_length, scenarios.modulus, scenarios.shear_modulus, 2)
        assert ((strains & core.shift_along_beam).abs() < 1e-12).all()


def test_neutral_axes_and_principal_bending_directions():
    down_load = scenarios.tip_load * -scenarios.up_direction
    for topology, nodes, thickness in section_walls():
        centre = core.centroid(topology.incidence[1, 0] * nodes, thickness)
        strains, _, principal, _ = core.cantilever(
            topology, nodes, thickness, core.tip_points(centre, scenarios.beam_length) & down_load,
            scenarios.beam_length, scenarios.modulus, scenarios.shear_modulus, 2)
        assert (centre & strains[0].cast(core.Strain)).norm() < 1e-12
        assert ((centre & principal).norm() < 1e-12).all()


def test_kern_and_reciprocity():
    push_points = core.section_point(np.array(scenarios.push_coordinates))
    for topology, nodes, thickness in section_walls():
        walls = topology.incidence[1, 0] * nodes
        rigidity = core.bending_stiffness(walls, thickness, scenarios.modulus)
        centre = core.centroid(walls, thickness)
        assert core.polar_strains(rigidity, centre).norm() < 1e-12
        _, stresses, polars = core.axial_pushes(
            topology, nodes, thickness, push_points, scenarios.modulus, scenarios.axial_push)
        assert (stresses[0] < 0).all() and (stresses[1] > 0).any()
        direct = core.polar_strains(rigidity, push_points[:, None]) & (core.beam_direction & nodes.batch())
        exchanged = polars & (core.beam_direction & push_points[:, None])
        assert ((exchanged - direct).abs() < 1e-12).all()
        corners = render.kern_polygon(polars, centre)
        corner_stretches = (core.polar_strains(rigidity, corners[:, None])
                            & (core.beam_direction & nodes.batch())).to_array().min(axis=-1)
        np.testing.assert_allclose(corner_stretches, 0.0, atol=1e-9)


def test_beams_of_one_piece_and_the_scenes_draw():
    import matplotlib.pyplot as plt

    sections = section_walls()
    down_load = scenarios.tip_load * -scenarios.up_direction                # [] Point-
    centres = [core.centroid(topology.incidence[1, 0] * nodes, thickness)
               for topology, nodes, thickness in sections]
    shears = [core.shear_centre(*section) for section in sections]
    beams = [core.cantilever(*section, core.tip_points(centre, scenarios.beam_length) & down_load,
                             scenarios.beam_length, scenarios.modulus, scenarios.shear_modulus, 1)
             for section, centre in zip(sections, centres)]
    push_points = core.section_point(np.array(scenarios.push_coordinates))  # [2] Point-
    pushes = [core.axial_pushes(*section, push_points, scenarios.modulus, scenarios.axial_push)
              for section in sections]
    plt.close(render.draw_sections(sections))
    plt.close(render.draw_walls(sections, centres, shears))
    plt.close(render.draw_beams(sections, centres, shears, down_load, beams))
    plt.close(render.draw_pushes(sections, centres, push_points, pushes))
    z_index = section_names.index("Z-section")
    turning = core.turning_states(*sections[z_index], centres[z_index], down_load,
                                  scenarios.beam_length, scenarios.modulus, scenarios.shear_modulus, 1, 1)
    list(render.animate_loads(sections[z_index], centres[z_index], shears[z_index], turning))
    start_point, end_point = core.section_point(np.array(scenarios.across_endpoints))  # [] Point-, [] Point-
    next(core.crossing_states(*sections[-1], start_point, end_point, down_load,
                              scenarios.beam_length, scenarios.modulus, scenarios.shear_modulus, 1, 1))
