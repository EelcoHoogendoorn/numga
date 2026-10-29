"""Scenes for the symmetry example: a conducting crystal, a cubic lattice, and optically active crystals."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.mechanics.symmetry import core
from examples.mechanics.symmetry.core import Conductivity, Direction, Elasticity, Field, Gyration, Point, RotaryPower, mv

# The measured responses' principal frame, turned in two planes; the conductivity's gains along its axes,
# and the rotary powers' values across its planes.
pose_xy, pose_yz = 0.3, 0.16
gains = np.array([4.5, 1.5, 0.6])
rotary_values = np.array([3.0, -1.0, 0.5])
# The seed bonds of the cubic lattice: along an axis, and along a face diagonal.
seed_bonds = [[1, 0, 0], [1, 1, 0]]

pose = (mv.xy * pose_xy + mv.yz * pose_yz).exp()                                # [] Rotor+
# The measured responses' principal axes, a tilted frame of directions, and the planes across them.
principal_axes = pose >> stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))         # [axes] Direction-
principal_planes = pose >> stack((mv.x, mv.y, mv.z))                            # [axes] Plane+
cube = core.cube_rotations()                                                    # [rotations] Rotor+


# --- math -----------------------------------------------------------------------------
def heat_conduction() -> tuple[Conductivity, Direction, Field, Direction]:
    """A tilted positive conductivity, averaged over three rotation groups, probed on the unit sphere:
    the conductivities, the heat-flow surfaces of every unit driving field, the driving field drawn, and
    the heat flows it drives.

    Half turns align the ellipsoid's axes; quarter turns in `mv.xy` make a spheroid; cube rotations make a
    sphere. This isotropy holds for a conductivity, a map from fields to directions; an elasticity, a form on
    four planes, can still be anisotropic under the cube.
    """
    driving = (mv.x + mv.y + mv.z).normalized()                                # [] Field+
    # Half turns in `mv.yz` and `mv.xy`; quarter turns in `mv.xy`; the rotations of a cube.
    groups = [(core.turns(mv.yz, 2)[:, None] * core.turns(mv.xy, 2)[None, :]).reshape(-1),
              core.turns(mv.xy, 4), cube]
    responses = core.conductivities(principal_axes, gains, groups)             # [candidate and groups] Direction- <- Field+
    return responses, responses[:, None, None](core.unit_fields[None, :, :]), driving, responses(driving)


def crystal_lattice() -> tuple[Point, Point, Point, Point, Direction, Conductivity, Elasticity]:
    """Axial and face-diagonal bonds averaged over the cube rotations, probed on the unit sphere: the
    lattice sites by their shells about the centre, the stiffness surface, and the conductivity and
    elasticity."""
    seeds = mv.vector(seed_bonds).dual().normalized()                          # [families] Direction-
    centre, axial, diagonal, corners = core.lattice_samples()
    conductivity, elasticity = core.lattice_responses(seeds, cube)
    # The unit plane across each direction of the sphere probes a uniaxial strain along it; the stiffness becomes
    # a radius.
    probes = core.unit_fields                                                  # [latitudes, longitudes] Plane+
    stiffness_surface = core.sphere * elasticity(probes, probes, probes, probes)   # [latitudes, longitudes] Direction-
    return centre, axial, diagonal, corners, stiffness_surface, conductivity, elasticity


def optical_activity() -> tuple[Gyration, RotaryPower]:
    """A tilted candidate rotary power, averaged over five point groups, probed on the unit sphere: the
    rotary powers, and their values along every direction.

    The rotations of a square keep two rotary powers, along its axis and across it; adding the inversion
    removes both. Half turns with quarter turns through the inversion keep rotary powers of opposite sense
    along x and y and none along z: such a crystal is its own mirror image, yet turns light along the
    directions its improper rotations exchange. The cube's rotations leave one rotary power, the same in
    every direction, as in a sugar solution; adding the inversion removes it.
    """
    powers = mv(RotaryPower, rotary_values[:, None])                           # [axes] RotaryPower-
    # The inversion is the pseudoscalar as an odd versor.
    inversion = mv.xyz                                                         # [] ImproperRotation+
    quarter = core.turns(mv.xy, 4)                                             # [turns] Rotor+
    # Each group's rotations, and its improper rotations; chiral groups have none, an empty batch: the
    # quarter turns, with the inversion, through the inversion, the cube's rotations, and with the inversion.
    groups = [(quarter, inversion * quarter[:0]), (quarter, inversion * quarter), (quarter[::2], inversion * quarter[1::2]),
              (cube, inversion * cube[:0]), (cube, inversion * cube)]
    responses = core.rotary_powers(principal_planes, powers, groups)          # [candidate and groups] RotaryPower- <- (Direction-, Direction-)
    return responses, responses[:, None, None](core.sphere[None, :, :], core.sphere[None, :, :])


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.mechanics.symmetry import render

    responses, surfaces, driving, fluxes = heat_conduction()
    save_figure(render.draw_conduction(responses, driving, fluxes), "symmetry_conductivity")
    centre, axial, diagonal, corners, stiffness, conductivity, elasticity = crystal_lattice()
    save_figure(render.draw_crystal(centre, axial, diagonal, corners, conductivity, stiffness), "symmetry_crystal_lattice")
    rotary, rotary_surfaces = optical_activity()
    save_figure(render.draw_optical_activity(rotary), "symmetry_optical_activity")

    # checks
    # Averaging keeps the trace, so the cube-invariant conductivity drives the mean gain along every field;
    # quarter turns in `mv.xy` conduct equally along x and y.
    assert (surfaces[-1] - core.sphere * gains.mean()).norm().to_array().max() < 1e-7
    quarter = responses[2]
    assert abs(((mv.x & quarter(mv.x)) - (mv.y & quarter(mv.y))).to_array()).max() < 1e-6
    # Lattice conduction is isotropic, 2/3 in every direction; stiffness is 1/2 across the cube's faces and 1/3
    # across the planes normal to its body diagonals.
    probes = core.unit_fields
    np.testing.assert_allclose((probes & conductivity(probes)).to_array(), 2 / 3, rtol=1e-7)
    cube_face, octahedral_face = mv.x, (mv.x + mv.y + mv.z).normalized()
    np.testing.assert_allclose(elasticity(*[cube_face] * 4).to_array(), 1 / 2, rtol=1e-7)
    np.testing.assert_allclose(elasticity(*[octahedral_face] * 4).to_array(), 1 / 3, rtol=1e-7)
    # The inversion removes every rotary power; the cube's rotations leave the mean one in every direction;
    # the rotoinversion leaves opposite senses along x and y and none along z. The same principal values as
    # a conductivity, which pairs a field with a direction, survive the inversion.
    for centrosymmetric in (rotary_surfaces[2], rotary_surfaces[5]):
        assert centrosymmetric.norm().to_array().max() < 1e-12
    mean_power = mv(RotaryPower, [rotary_values.mean()])
    assert (rotary_surfaces[4] - mean_power).norm().to_array().max() < 1e-12
    rotoinversion = rotary[3]
    x_direction, y_direction, z_direction = mv.x.dual(), mv.y.dual(), mv.z.dual()
    assert (rotoinversion(x_direction, x_direction) + rotoinversion(y_direction, y_direction)).norm() < 1e-12
    assert rotoinversion(z_direction, z_direction).norm() < 1e-12
    candidate = core.conductivities(principal_axes, rotary_values, [])[0]
    inverted = mv.xyz >> candidate(mv.xyz << Field)
    assert (inverted(probes) - candidate(probes)).norm().to_array().max() < 1e-12


if __name__ == "__main__":
    main()
