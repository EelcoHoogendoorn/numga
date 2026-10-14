"""Unsymmetric bending: the stiffness holds the area and the second moments of area of the
rectangles, a small tip load deflects the tip by the cantilever formula along the inverse second
moment, and the scenes pass their checks and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numga import stack

from examples.mechanics.bending import core, render, scenarios
from examples.mechanics.bending.mesh import HEIGHT, FLANGE, THICKNESS, TURNS, TUBE_RADIUS, TUBE_WALL, SLOT_ANGLE, section_meshes


def second_moments() -> np.ndarray:
    """Closed-form section moments from two flanges and a web, independent of the triangulation."""
    flange_area = FLANGE * THICKNESS
    web_height = HEIGHT - 2 * THICKNESS
    web_area = web_height * THICKNESS
    offset_y = (HEIGHT - THICKNESS) / 2
    offset_x = (FLANGE - THICKNESS) / 2
    across = 2 * flange_area * FLANGE**2 / 12 + web_area * THICKNESS**2 / 12
    up = 2 * flange_area * (offset_y**2 + THICKNESS**2 / 12) + web_area * web_height**2 / 12
    aligned = np.diag([across, up])
    angle = np.deg2rad(TURNS[1])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    z_section = np.array([[across + 2 * flange_area * offset_x**2, 2 * flange_area * offset_x * offset_y],
                          [2 * flange_area * offset_x * offset_y, up]])
    return np.stack([aligned, rotation @ aligned @ rotation.T, z_section])


def test_stiffness_holds_the_area_and_the_second_moments():
    area = 2 * FLANGE * THICKNESS + (HEIGHT - 2 * THICKNESS) * THICKNESS
    for (triangles, boundary), spread in zip(section_meshes(), second_moments()):
        rigidity = core.stiffness(triangles, scenarios.MODULUS)
        np.testing.assert_allclose((core.shift_along_beam & rigidity(core.shift_along_beam)).to_array(),
                                   scenarios.MODULUS * area, rtol=1e-12)
        turns = core.centroid(triangles) & stack([core.mv.x, core.mv.y]).dual()
        energies = (turns & rigidity(turns)).to_array()
        expected = scenarios.MODULUS * np.array([spread[1, 1], spread[0, 0]])
        np.testing.assert_allclose(energies, expected, rtol=1e-12)


def test_small_tip_load_deflects_by_the_cantilever_formula():
    load = 1e-3 * scenarios.LOAD
    for (triangles, boundary), spread in zip(section_meshes(), second_moments()):
        _, frames, _, _ = scenarios.cantilever(triangles, load * -scenarios.up_direction)
        centre = core.centroid(triangles)
        across = render.flat(frames[-1] >> centre) - render.flat(centre)
        expected = scenarios.LENGTH**3 / (3 * scenarios.MODULUS) * np.linalg.solve(spread, [0.0, -load])
        # This tolerance tests the approximation by constant-strain pieces.
        np.testing.assert_allclose(np.linalg.norm(across - expected) / np.linalg.norm(expected), 0.0, atol=1e-3)


def test_slotted_tube_matches_annular_sector_moments():
    triangles, boundary = section_meshes()[-1]
    inner = TUBE_RADIUS - TUBE_WALL
    sweep = 2 * np.pi - SLOT_ANGLE
    area = sweep * (TUBE_RADIUS**2 - inner**2) / 2
    centre_x = -2 * np.sin(SLOT_ANGLE / 2) * (TUBE_RADIUS**3 - inner**3) / (3 * area)
    radial = (TUBE_RADIUS**4 - inner**4) / 4
    across = radial * (sweep - np.sin(SLOT_ANGLE)) / 2 - area * centre_x**2
    up = radial * (sweep + np.sin(SLOT_ANGLE)) / 2
    rigidity = core.stiffness(triangles, scenarios.MODULUS)
    centre = core.centroid(triangles)
    turns = centre & stack([core.mv.x, core.mv.y]).dual()
    # The tolerance measures the polygonal approximation to the curved annular sector.
    np.testing.assert_allclose(core.areas(triangles).sum(axis=0).to_array(), area, rtol=0.01)
    np.testing.assert_allclose(render.flat(centre), [centre_x, 0.0], rtol=0.01, atol=1e-12)
    np.testing.assert_allclose((turns & rigidity(turns)).to_array(),
                               scenarios.MODULUS * np.array([up, across]), rtol=0.01)


def test_neutral_axes_and_principal_bending_directions():
    quarter = (core.mv.xy * (-np.pi / 4)).exp()
    for triangles, boundary in section_meshes():
        strains, frames, principal, stress = scenarios.cantilever(triangles, scenarios.LOAD * -scenarios.up_direction)
        centre = core.centroid(triangles)
        np.testing.assert_allclose((centre & strains[0]).kernel, 0.0, atol=1e-12)
        np.testing.assert_allclose((centre & principal).kernel, 0.0, atol=1e-12)
        load = scenarios.LOAD * (quarter >> (principal[0].normalized() ^ core.mv.w))
        _, along_principal, _, _ = scenarios.cantilever(triangles, load)
        plane = (centre & core.beam_direction) & load
        np.testing.assert_allclose(((along_principal[-1] >> centre) & plane).to_array(), 0.0, atol=1e-12)


def test_kern_and_reciprocity():
    points, view = core.point(scenarios.PUSH_POINTS), scenarios.kern_view()
    for triangles, boundary in section_meshes():
        rigidity = core.stiffness(triangles, scenarios.MODULUS)
        centred = core.pulls(rigidity, core.centroid(triangles))
        np.testing.assert_allclose(centred.norm().to_array(), 0.0, atol=1e-12)
        strains, stress, corners = scenarios.push_section(triangles, points, view)
        squeezed = stress.to_array()
        assert np.all(squeezed[0] < 0) and squeezed[1].max() > 0


def test_scenes_draw_and_opposite_loads_bend_oppositely():
    sections = section_meshes()
    beams = [scenarios.cantilever(triangles, scenarios.LOAD * -scenarios.up_direction) for triangles, _ in sections]
    points, view = core.point(scenarios.PUSH_POINTS), scenarios.kern_view()
    pushes = [scenarios.push_section(triangles, points, view) for triangles, _ in sections]
    for figure in (render.draw_meshes(sections), render.draw_beams(sections, scenarios.LOAD * -scenarios.up_direction, beams),
                   render.draw_pushes(sections, points, view, pushes)):
        plt.close(figure)
    triangles, boundary = sections[scenarios.NAMES.index("Z-section")]
    loads = scenarios.turning_loads()
    strains, frames, principal, stress = scenarios.cantilever(triangles, loads)
    centre = core.centroid(triangles)
    across = render.flat(frames[:, -1] >> centre) - render.flat(centre)
    half = scenarios.FRAMES // 2
    np.testing.assert_allclose(np.linalg.norm(across[:half] + across[half:], axis=-1)
                               / np.linalg.norm(across[:half], axis=-1), 0.0, atol=2e-2)
    one = (strains[:1], frames[:1], principal, stress[:1])
    assert len(render.animate_turning(triangles, boundary, loads[:1], one)) == 1
