"""Unsymmetric bending: the stiffness holds the area and the second moments of area of the
rectangles, a small tip load deflects the tip by the cantilever formula along the inverse second
moment, and the scenes pass their checks and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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
        np.testing.assert_allclose((core.slide & rigidity(core.slide)).to_array(),
                                   scenarios.MODULUS * area, rtol=1e-12)
        turns = core.centroid(triangles) & core.direction([[1.0, 0.0], [0.0, 1.0]])
        energies = (turns & rigidity(turns)).to_array()
        expected = scenarios.MODULUS * np.array([spread[1, 1], spread[0, 0]])
        np.testing.assert_allclose(energies, expected, rtol=1e-12)


def test_small_tip_load_deflects_by_the_cantilever_formula():
    load = 1e-3 * scenarios.LOAD
    for (triangles, boundary), spread in zip(section_meshes(), second_moments()):
        beam = scenarios.cantilever(triangles, boundary, load * -scenarios.up)
        across = render.flat(beam.centres[-1]) - render.flat(core.centroid(triangles))
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
    turns = centre & core.direction([[1.0, 0.0], [0.0, 1.0]])
    # The tolerance measures the polygonal approximation to the curved annular sector.
    np.testing.assert_allclose(core.areas(triangles).sum(axis=0).to_array(), area, rtol=0.01)
    np.testing.assert_allclose(render.flat(centre), [centre_x, 0.0], rtol=0.01, atol=1e-12)
    np.testing.assert_allclose((turns & rigidity(turns)).to_array(),
                               scenarios.MODULUS * np.array([up, across]), rtol=0.01)


def test_neutral_axes_and_principal_bending_directions():
    beams = scenarios.downward()
    quarter = (core.mv.xy * (-np.pi / 4)).exp()
    for beam in beams:
        centre = core.centroid(beam.triangles)
        np.testing.assert_allclose((centre & beam.strains[0]).kernel, 0.0, atol=1e-12)
        np.testing.assert_allclose((centre & beam.principal).kernel, 0.0, atol=1e-12)
        load = scenarios.LOAD * (quarter >> (beam.principal[0].normalized() ^ core.mv.w))
        principal = scenarios.cantilever(beam.triangles, beam.boundary, load)
        plane = (centre & core.along) & load
        np.testing.assert_allclose((principal.centres[-1] & plane).to_array(), 0.0, atol=1e-12)
    # Tip displacements are ideal points: read their headings, not their homogeneous positions.
    deflections = np.array([render.flat(beam.centres[-1]) - render.flat(core.centroid(beam.triangles)) for beam in beams])
    sideways, down = deflections.T
    np.testing.assert_allclose(sideways[0], 0.0, atol=1e-12)
    assert np.all(down < 0) and abs(sideways[1]) > 0.1 * abs(down[1]) and abs(sideways[2]) > abs(down[2])


def test_kern_and_reciprocity():
    for pushed in scenarios.pushes():
        rigidity = core.stiffness(pushed.triangles, scenarios.MODULUS)
        centred = core.pulls(rigidity, core.centroid(pushed.triangles))
        np.testing.assert_allclose(centred.norm().to_array(), 0.0, atol=1e-12)
        first, second = core.pulls(rigidity, pushed.points[0]), core.pulls(rigidity, pushed.points[1])
        np.testing.assert_allclose((first & (core.along & pushed.points[1])).to_array(),
                                   (second & (core.along & pushed.points[0])).to_array(), rtol=1e-10)
        squeezed = pushed.stress.to_array()
        assert np.all(squeezed[0] < 0) and squeezed[1].max() > 0


def test_scenes_draw_and_opposite_loads_bend_oppositely():
    figures = [render.draw_meshes(section_meshes(), scenarios.NAMES),
               render.draw_beams(scenarios.downward(), scenarios.NAMES),
               render.draw_pushes(scenarios.pushes(), scenarios.NAMES)]
    for figure in figures:
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
    turning = scenarios.turning()
    across = render.flat(turning.centres[:, -1]) - render.flat(core.centroid(turning.triangles))
    half = scenarios.FRAMES // 2
    np.testing.assert_allclose(np.linalg.norm(across[:half] + across[half:], axis=-1)
                               / np.linalg.norm(across[:half], axis=-1), 0.0, atol=2e-2)
    one = core.Beam(turning.triangles, turning.boundary, turning.loads[:1], turning.strains[:1], turning.frames[:1],
                    turning.centres[:1], turning.principal, turning.stress[:1])
    assert len(render.animate_turning(one)) == 1
