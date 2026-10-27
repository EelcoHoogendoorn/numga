"""Thin-walled bending and twisting: the stiffness holds the area and the second moments of area of the
walls, a small tip load deflects the tip by the cantilever formula along the inverse second moment and
slides it along the inverse shear area, the
slotted tube has the area, moments and shear centre of a thin circular arc, the warping seen from the
shear centre is invisible to bending, a torque twists the beam about its shear centre by the torque
over the torsional stiffness, and the scenes pass their checks and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numga import stack

from examples.mechanics.bending import core, render, scenarios
from examples.mechanics.bending.mesh import HEIGHT, FLANGE, THICKNESS, TURNS, TUBE_RADIUS, TUBE_WALL, SLOT_ANGLE, section_walls


def shear_areas() -> np.ndarray:
    """Closed-form shear areas of the I-sections and the Z-section: each wall carries a slide along
    itself, so the flanges carry a slide across and the web a slide up."""
    aligned = np.diag([2 * FLANGE * THICKNESS, HEIGHT * THICKNESS])
    angle = np.deg2rad(TURNS[1])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.stack([aligned, rotation @ aligned @ rotation.T, aligned])


def second_moments() -> np.ndarray:
    """Closed-form thin-walled section moments of the I-sections and the Z-section, from two flanges
    and a web along their midlines, independent of the quadrature."""
    flange_area = FLANGE * THICKNESS
    across = 2 * THICKNESS * FLANGE**3 / 12
    up = 2 * flange_area * (HEIGHT / 2)**2 + THICKNESS * HEIGHT**3 / 12
    aligned = np.diag([across, up])
    angle = np.deg2rad(TURNS[1])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    product = 2 * flange_area * (FLANGE / 2) * (HEIGHT / 2)
    z_section = np.array([[across + 2 * flange_area * (FLANGE / 2)**2, product], [product, up]])
    return np.stack([aligned, rotation @ aligned @ rotation.T, z_section])


def test_stiffness_holds_the_area_and_the_second_moments():
    area = (2 * FLANGE + HEIGHT) * THICKNESS
    for (nodes, ends, thickness), spread in zip(section_walls(), second_moments()):
        walls = nodes[ends]
        rigidity = core.bending(walls, thickness, scenarios.MODULUS)
        np.testing.assert_allclose((core.shift_along_beam & rigidity(core.shift_along_beam)).to_array(),
                                   scenarios.MODULUS * area, rtol=1e-12)
        # Turns about the lines through the centroid along x and along y: the meets of the section's plane
        # with the planes through the centroid square to y and to x.
        square = stack([core.mv.y, core.mv.x])                              # [2] Plane
        turns = (square - (square & core.centroid(walls, thickness)) * core.mv.w) ^ core.section_plane   # [2] Line
        energies = (turns & rigidity(turns)).to_array()
        expected = scenarios.MODULUS * np.array([spread[1, 1], spread[0, 0]])
        np.testing.assert_allclose(energies, expected, rtol=1e-12)


def test_small_tip_load_deflects_by_the_cantilever_formula():
    """Bending along the inverse second moment, and the slide along the inverse shear area, on the slender
    beams and on a stubby one, about twice as long as it is deep, where the slide is a large part of the deflection."""
    load = 1e-3 * scenarios.LOAD
    stubby_length = 2.0
    for length in (scenarios.LENGTH, stubby_length):
        for (nodes, ends, thickness), spread, shear in zip(section_walls(), second_moments(), shear_areas()):
            centre = core.centroid(nodes[ends], thickness)
            _, frames, _, _ = scenarios.cantilever(nodes, ends, thickness,
                                                   scenarios.tip(centre, length) & (load * -scenarios.up_direction),
                                                   length, scenarios.SHEAR_MODULUS)
            across = render.flat(frames[-1] >> centre) - render.flat(centre)
            expected = (length**3 / (3 * scenarios.MODULUS) * np.linalg.solve(spread, [0.0, -load])
                        + length / scenarios.SHEAR_MODULUS * np.linalg.solve(shear, [0.0, -load]))
            # This tolerance tests the approximation by constant-strain pieces.
            np.testing.assert_allclose(np.linalg.norm(across - expected) / np.linalg.norm(expected), 0.0, atol=1e-3)


def test_slotted_tube_matches_a_thin_circular_arc():
    nodes, ends, thickness = section_walls()[-1]
    walls = nodes[ends]
    sweep = 2 * np.pi - SLOT_ANGLE
    half = sweep / 2
    area = sweep * TUBE_RADIUS * TUBE_WALL
    centre_x = -TUBE_RADIUS * np.sin(half) / half
    radial = TUBE_RADIUS**3 * TUBE_WALL
    across = radial * (sweep - np.sin(SLOT_ANGLE)) / 2 - area * centre_x**2
    up = radial * (sweep + np.sin(SLOT_ANGLE)) / 2
    shear_x = -2 * TUBE_RADIUS * (np.sin(half) - half * np.cos(half)) / (half - np.sin(half) * np.cos(half))
    rigidity = core.bending(walls, thickness, scenarios.MODULUS)
    centre = core.centroid(walls, thickness)
    square = stack([core.mv.y, core.mv.x])                                  # [2] Plane
    turns = (square - (square & centre) * core.mv.w) ^ core.section_plane   # [2] Line
    # The tolerances measure the polygonal approximation to the arc.
    np.testing.assert_allclose(((walls[..., 0] & walls[..., 1]).norm() * thickness).sum(axis=0).to_array(), area, rtol=1e-2)
    np.testing.assert_allclose(render.flat(centre), [centre_x, 0.0], rtol=1e-2, atol=1e-12)
    np.testing.assert_allclose((turns & rigidity(turns)).to_array(),
                               scenarios.MODULUS * np.array([up, across]), rtol=1e-2)
    np.testing.assert_allclose(render.flat(core.shear_centre(nodes, ends, thickness)),
                               [shear_x, 0.0], rtol=1e-2, atol=1e-12)


def test_warping_from_the_shear_centre_is_invisible_to_bending():
    for nodes, ends, thickness in section_walls():
        walls = nodes[ends]
        shear = core.shear_centre(nodes, ends, thickness)
        couples = core.warping(nodes, ends, thickness, shear).cast(core.Couples)
        np.testing.assert_allclose(couples.kernel, 0.0, atol=1e-12)
    for nodes, ends, thickness in section_walls()[:-1]:
        # Mirror or point symmetry puts the shear centre on the centroid.
        shear = core.shear_centre(nodes, ends, thickness)
        np.testing.assert_allclose(render.flat(shear), render.flat(core.centroid(nodes[ends], thickness)), atol=1e-12)


def test_torque_twists_about_the_shear_centre():
    torque = 1e-5
    for nodes, ends, thickness in section_walls():
        walls = nodes[ends]
        shear = core.shear_centre(nodes, ends, thickness)
        stiffness = core.twisting(walls, thickness, scenarios.SHEAR_MODULUS).to_array()
        _, frames, _, _ = scenarios.cantilever(nodes, ends, thickness, torque * core.couple_about_beam, scenarios.LENGTH,
                                               scenarios.SHEAR_MODULUS)
        np.testing.assert_allclose(render.twist(frames[-1]), torque * scenarios.LENGTH / stiffness, rtol=1e-12)
        np.testing.assert_allclose(render.flat(frames[-1] >> shear), render.flat(shear), atol=1e-12)
        # A load through the shear centre bends the beam without turning it about the beam.
        strains, _, _, _ = scenarios.cantilever(nodes, ends, thickness,
                                                scenarios.tip(shear, scenarios.LENGTH) & (scenarios.LOAD * -scenarios.up_direction),
                                                scenarios.LENGTH, scenarios.SHEAR_MODULUS)
        np.testing.assert_allclose((strains & core.shift_along_beam).to_array(), 0.0, atol=1e-12)


def test_neutral_axes_and_principal_bending_directions():
    for nodes, ends, thickness in section_walls():
        centre = core.centroid(nodes[ends], thickness)
        strains, frames, principal, stress = scenarios.cantilever(
            nodes, ends, thickness, scenarios.tip(centre, scenarios.LENGTH) & (scenarios.LOAD * -scenarios.up_direction),
            scenarios.LENGTH, scenarios.SHEAR_MODULUS)
        # The bending part of the strain, without the twist about the shear centre.
        np.testing.assert_allclose((centre & strains[0].cast(core.Strain)).kernel, 0.0, atol=1e-12)
        np.testing.assert_allclose((centre & principal).kernel, 0.0, atol=1e-12)


def test_kern_and_reciprocity():
    points = core.point(scenarios.PUSH_POINTS)
    for nodes, ends, thickness in section_walls():
        rigidity = core.bending(nodes[ends], thickness, scenarios.MODULUS)
        centre = core.centroid(nodes[ends], thickness)
        np.testing.assert_allclose(core.pulls(rigidity, centre).norm().to_array(), 0.0, atol=1e-12)
        strains, stress, polars = scenarios.push_section(nodes, ends, thickness, points)
        squeezed = stress.to_array()
        assert np.all(squeezed[0] < 0) and squeezed[1].max() > 0
        # Each node's polar stretches the fibre through a point as much as a pull through the point
        # stretches the node's fibre.
        direct = core.pulls(rigidity, points[:, None]) & (core.beam_direction & nodes)
        np.testing.assert_allclose((polars & (core.beam_direction & points[:, None])).to_array(), direct.to_array(), atol=1e-12)
        # A push through a corner of the kern leaves some fibre unstretched and squeezes the rest.
        corners = core.point(render.kern(polars, centre))                   # [corners] Point
        least = (core.pulls(rigidity, corners[:, None]) & (core.beam_direction & nodes)).to_array().min(axis=-1)
        np.testing.assert_allclose(least, 0.0, atol=1e-9)


def test_scenes_draw():
    sections = section_walls()
    down = scenarios.LOAD * -scenarios.up_direction
    centres = [core.centroid(nodes[ends], thickness) for nodes, ends, thickness in sections]
    shears = [core.shear_centre(*section) for section in sections]
    beams = [scenarios.cantilever(*section, scenarios.tip(centre, scenarios.LENGTH) & down, scenarios.LENGTH,
                                  scenarios.SHEAR_MODULUS) for section, centre in zip(sections, centres)]
    points = core.point(scenarios.PUSH_POINTS)
    pushes = [scenarios.push_section(*section, points) for section in sections]
    for figure in (render.draw_walls(sections, centres, shears), render.draw_beams(sections, centres, shears, down, beams),
                   render.draw_pushes(sections, centres, points, pushes)):
        plt.close(figure)
    tube = sections[-1]
    across = scenarios.across_points()[:1]
    moved = scenarios.cantilever(*tube, scenarios.tip(across, scenarios.LENGTH) & down, scenarios.LENGTH, scenarios.SHEAR_MODULUS)
    assert len(list(render.animate_loads(tube, centres[-1], shears[-1], across, down, moved))) == 1
    plt.close(render.draw_sections(sections))
