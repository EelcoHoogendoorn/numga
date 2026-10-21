"""Scenes for the symmetry example: a conducting crystal, a cubic lattice, and optically active crystals."""

from __future__ import annotations

import numpy as np

from numga import Extensor
from examples.mechanics.symmetry import core
from examples.mechanics.symmetry.core import Direction, Field, mv

# The measured responses' principal frame, turned in two planes; the conductivity's gains along its axes,
# and the rotary powers along them.
POSE_XY, POSE_YZ = 0.3, 0.16
GAINS = np.array([4.5, 1.5, 0.6])
POWERS = np.array([3.0, -1.0, 0.5])
# The seed bonds of the cubic lattice: along an axis, and along a face diagonal.
SEEDS = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]


# --- math -----------------------------------------------------------------------------
def principal_axes() -> Direction:
    """The measured responses' principal axes, a tilted frame of directions."""
    pose = (mv.xy * POSE_XY + mv.yz * POSE_YZ).exp()
    return pose >> Extensor.stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))    # [3] Direction


def heat_conduction():
    """A tilted positive conductivity, averaged over three rotation groups, probed on the unit sphere:
    the conductivities, the heat-flow surfaces of every unit driving field, the driving field drawn, and
    the heat flows it drives.

    Half turns align the ellipsoid's axes; quarter turns about z make a spheroid; cube rotations make a
    sphere. This isotropy follows for rank-two conductivity; rank-four cubic elasticity can still be
    anisotropic.
    """
    driving = (mv.x + mv.y + mv.z).normalized()                                # [] Field
    sphere = core.fields()                                                     # [latitudes, longitudes] Field
    # Half turns about x and z; quarter turns about z; the rotations of a cube.
    groups = [(core.turns(mv.yz, 2)[:, None] * core.turns(mv.xy, 2)[None, :]).reshape(-1),
              core.turns(mv.xy, 4), core.cube_rotations()]
    responses = core.conductivities(principal_axes(), GAINS, groups)           # [candidate and groups] Direction <- Field
    return responses, responses[:, None, None](sphere[None, :, :]), driving, responses(driving)


def crystal_lattice():
    """Axial and face-diagonal bonds averaged over the cube rotations, probed on the unit sphere: the
    lattice sites and the two neighbour shells, the stiffness surface, and the conductivity and
    elasticity."""
    seeds = mv.vector(SEEDS).dual().normalized()                               # [families] Direction
    sites, axial, diagonal = core.lattice_samples()
    sphere = core.directions()                                                 # [latitudes, longitudes] Direction
    conductivity, elasticity = core.lattice_responses(seeds, core.cube_rotations())
    # A unit direction in every slot probes imposed uniaxial strain; the stiffness becomes a radius.
    stiffness_surface = sphere * elasticity(sphere, sphere, sphere, sphere)    # [latitudes, longitudes] Direction
    return sites, axial, diagonal, stiffness_surface, conductivity, elasticity


def optical_activity():
    """A tilted candidate rotary power, averaged over five point groups, probed on the unit sphere: the
    rotary powers, their values along every direction, and the sphere.

    The rotations of a square keep two rotary powers, along its axis and across it; adding the inversion
    removes both. Half turns with quarter turns through the inversion keep rotary powers of opposite sense
    along x and y and none along z: such a crystal is its own mirror image, yet turns light along the
    directions its improper rotations exchange. The cube's rotations leave one rotary power, the same in
    every direction, as in a sugar solution; adding the inversion removes it.
    """
    sphere = core.directions()
    inversion = mv.xyz
    quarter = core.turns(mv.xy, 4)
    cube = core.cube_rotations()
    # Each group's rotations, and its improper rotations; chiral groups have none, an empty batch: the
    # quarter turns, with the inversion, through the inversion, the cube's rotations, and with the inversion.
    groups = [(quarter, inversion * quarter[:0]), (quarter, inversion * quarter), (quarter[::2], inversion * quarter[1::2]),
              (cube, inversion * cube[:0]), (cube, inversion * cube)]
    responses = core.rotary_powers(principal_axes(), POWERS, groups)          # [candidate and groups] Scalar <- (Direction, Direction)
    return responses, responses[:, None, None](sphere[None, :, :], sphere[None, :, :]), sphere


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_figure
    from examples.mechanics.symmetry import render

    responses, surfaces, driving, fluxes = heat_conduction()
    save_figure(render.draw_conduction(responses, driving, fluxes), "symmetry_conductivity")
    sites, axial, diagonal, stiffness, conductivity, elasticity = crystal_lattice()
    save_figure(render.draw_crystal(sites, axial, diagonal, conductivity, stiffness), "symmetry_crystal_lattice")
    rotary, rotary_surfaces, sphere = optical_activity()
    save_figure(render.draw_optical_activity(rotary), "symmetry_optical_activity")

    # checks
    # Averaging keeps the trace, so the cube-invariant conductivity drives the mean gain along every field;
    # quarter turns about z conduct equally along x and y.
    assert (surfaces[-1] - core.directions() * GAINS.mean()).norm().to_array().max() < 1e-7
    quarter = responses[2]
    assert abs(((mv.x & quarter(mv.x)) - (mv.y & quarter(mv.y))).to_array()).max() < 1e-6
    # Lattice conduction is isotropic, 2/3 in every direction; stiffness is 1/2 along the cube axes and 1/3
    # along the body diagonals.
    probes = core.fields()
    np.testing.assert_allclose((probes & conductivity(probes)).to_array(), 2 / 3, rtol=1e-7)
    axis_direction, diagonal_direction = mv.x.dual(), (mv.x + mv.y + mv.z).dual().normalized()
    np.testing.assert_allclose(elasticity(*[axis_direction] * 4).to_array(), 1 / 2, rtol=1e-7)
    np.testing.assert_allclose(elasticity(*[diagonal_direction] * 4).to_array(), 1 / 3, rtol=1e-7)
    # The inversion removes every rotary power; the cube's rotations leave the mean one in every direction;
    # the rotoinversion leaves opposite senses along x and y and none along z. The same principal values as
    # a conductivity, which pairs a field with a direction, survive the inversion.
    handedness = mv.xyz.dual()
    for centrosymmetric in (rotary_surfaces[2], rotary_surfaces[5]):
        assert centrosymmetric.norm().to_array().max() < 1e-12
    assert (rotary_surfaces[4] - handedness * POWERS.mean()).norm().to_array().max() < 1e-12
    rotoinversion = rotary[3]
    along_x, along_y, along_z = mv.x.dual(), mv.y.dual(), mv.z.dual()
    assert (rotoinversion(along_x, along_x) + rotoinversion(along_y, along_y)).norm().to_array() < 1e-12
    assert rotoinversion(along_z, along_z).norm().to_array() < 1e-12
    candidate = core.conductivities(principal_axes(), POWERS, [])[0]
    inverted = mv.xyz >> candidate(mv.xyz << Field)
    assert (inverted(probes) - candidate(probes)).norm().to_array().max() < 1e-12


if __name__ == "__main__":
    main()
