"""Scenarios and sampling grids for material symmetry: heat conduction, cubic lattice elasticity, and optical activity."""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D
from examples.mechanics.symmetry import core
from examples.mechanics.symmetry.core import Conductivity, Direction, Elasticity, Field, Gyration, Plane, RotaryPower, Rotor, mv

pga_mv = NumpyContext(PGA3D).multivector
LatticePoint = PGA3D.gatype.antivector()                                        # Antivector-

# Resolution of the latitude-longitude sphere grid used to sample directional surfaces.
n_latitudes, n_longitudes = 33, 65

# Tilted principal frame, principal conductivity gains, principal rotary powers, and simple-cubic seed bonds.
pose_xy, pose_yz = 0.3, 0.16
gains = np.array([4.5, 1.5, 0.6])
rotary_values = np.array([3.0, -1.0, 0.5])
seed_bond_coords = [[1, 0, 0], [1, 1, 0]]

pose = (mv.xy * pose_xy + mv.yz * pose_yz).exp()                                # [] Rotor+
principal_axes = pose >> stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))         # [3] Direction-
principal_planes = pose >> stack((mv.x, mv.y, mv.z))                            # [3] Plane+
cube = core.cube_rotations()                                                    # [24] Rotor+


# --- plumbing -------------------------------------------------------------------------
def sphere_grid() -> tuple[Direction, Plane]:
    """Latitude-longitude rotations produce unit directions and unit planes (`[n_latitudes, n_longitudes]`) across the sphere."""
    longitude = np.linspace(0, 2 * np.pi, n_longitudes)[None, :]
    latitude = np.linspace(-np.pi / 2, np.pi / 2, n_latitudes)[:, None]
    grid_turns = (mv.xy * (-longitude / 2)).exp() * (mv.xz * (-latitude / 2)).exp()  # [n_latitudes, n_longitudes] Rotor+
    return grid_turns >> mv.x.dual(), grid_turns >> mv.x


sphere_directions, unit_fields = sphere_grid()                                  # [n_latitudes, n_longitudes] Direction-, Plane+


def lattice_shells() -> tuple[LatticePoint, LatticePoint, LatticePoint, LatticePoint]:
    """Projective sites of a `3 x 3 x 3` simple-cubic block partition into centre, axial, face-diagonal, and corner shells."""
    grid_i, grid_j, grid_k = np.indices((3, 3, 3)) - 1
    offsets = np.stack([grid_i, grid_j, grid_k], axis=-1)
    sites = pga_mv.point(offsets)                                               # [3, 3, 3] Antivector-
    shell = grid_i * grid_i + grid_j * grid_j + grid_k * grid_k
    return sites[shell == 0], sites[shell == 1], sites[shell == 2], sites[shell == 3]


# --- math -----------------------------------------------------------------------------
def heat_conduction() -> tuple[Conductivity, Direction, Field, Direction]:
    """A tilted positive conductivity averaged over three rotation groups yields `[4]` conductivities, flow surfaces, driving field, and fluxes.

    Half turns align the ellipsoid's principal axes, quarter turns in the horizontal plane make an axisymmetric
    spheroid, and the 24 cube rotations produce an isotropic sphere.
    """
    driving = (mv.x + mv.y + mv.z).normalized()                                 # [] Field+
    orthorhombic_turns = (core.turns(mv.yz, 2)[:, None] * core.turns(mv.xy, 2)[None, :]).reshape(-1)  # [4] Rotor+
    groups = [orthorhombic_turns, core.turns(mv.xy, 4), cube]
    responses = core.conductivities(principal_axes, gains, groups)              # [4] Direction- <- Field+
    surfaces = responses[:, None, None](unit_fields[None, :, :])                # [4, n_latitudes, n_longitudes] Direction-
    return responses, surfaces, driving, responses(driving)


def crystal_lattice() -> tuple[LatticePoint, LatticePoint, LatticePoint, LatticePoint, Direction, Conductivity, Elasticity]:
    """Simple-cubic axial and face-diagonal bonds averaged over `cube` yield the four site shells, stiffness surface, conductivity, and elasticity."""
    seed_bonds = mv.vector(seed_bond_coords).dual().normalized()                # [2] Direction-
    centre, axial, diagonal, corners = lattice_shells()
    conductivity, elasticity = core.lattice_responses(seed_bonds, cube)
    probes = unit_fields                                                        # [n_latitudes, n_longitudes] Plane+
    stiffness_surface = sphere_directions * elasticity(probes, probes, probes, probes)  # [n_latitudes, n_longitudes] Direction-
    return centre, axial, diagonal, corners, stiffness_surface, conductivity, elasticity


def optical_activity() -> tuple[Gyration, RotaryPower]:
    """A tilted gyration map averaged over five point groups yields `[6]` gyration maps and their directional rotary-power surfaces.

    Pure fourfold rotations keep independent axial and transverse rotary powers, fourfold rotoinversions keep
    opposite senses along the horizontal axes with zero along the vertical axis, pure cube rotations leave a single
    isotropic rotary power, and any group containing spatial inversion cancels optical activity everywhere.
    """
    powers = mv(RotaryPower, rotary_values[:, None])                            # [3] RotaryPower-
    inversion = mv.xyz                                                          # [] ImproperRotation+
    quarter_turns = core.turns(mv.xy, 4)                                        # [4] Rotor+
    groups = [
        (quarter_turns, inversion * quarter_turns[:0]),
        (quarter_turns, inversion * quarter_turns),
        (quarter_turns[::2], inversion * quarter_turns[1::2]),
        (cube, inversion * cube[:0]),
        (cube, inversion * cube),
    ]
    gyrations = core.rotary_powers(principal_planes, powers, groups)            # [6] RotaryPower- <- (Direction-, Direction-)
    rotary_surfaces = gyrations[:, None, None](sphere_directions[None, :, :], sphere_directions[None, :, :])  # [6, n_latitudes, n_longitudes] RotaryPower-
    return gyrations, rotary_surfaces


def main() -> None:
    """Saves the three material-symmetry figures and verifies the group-averaged invariants."""
    from examples.util.animation import save_figure
    from examples.mechanics.symmetry import render

    responses, surfaces, driving, fluxes = heat_conduction()
    save_figure(render.conduction_figure(responses, driving, fluxes), "symmetry_conductivity")
    centre, axial, diagonal, corners, stiffness_surface, conductivity, elasticity = crystal_lattice()
    save_figure(render.crystal_figure(centre, axial, diagonal, corners, conductivity, stiffness_surface), "symmetry_crystal_lattice")
    gyrations, rotary_surfaces = optical_activity()
    save_figure(render.optical_activity_figure(gyrations), "symmetry_optical_activity")

    # checks
    # Cube rotations leave the trace mean along every direction; fourfold turns equate horizontal conductivities.
    assert ((surfaces[-1] - sphere_directions * gains.mean()).norm() < 1e-7).all()
    tetragonal = responses[2]
    assert (((mv.x & tetragonal(mv.x)) - (mv.y & tetragonal(mv.y))).abs() < 1e-6).all()
    # Simple-cubic bond conduction is 2/3 in every direction, while uniaxial stiffness is 1/2 on cube faces and 1/3 on body diagonals.
    probes = unit_fields
    assert (((probes & conductivity(probes)) - 2 / 3).abs() < 1e-7).all()
    cube_face, octahedral_face = mv.x, (mv.x + mv.y + mv.z).normalized()
    assert ((elasticity(cube_face, cube_face, cube_face, cube_face) - 1 / 2).abs() < 1e-7).all()
    assert ((elasticity(octahedral_face, octahedral_face, octahedral_face, octahedral_face) - 1 / 3).abs() < 1e-7).all()
    # Spatial inversion kills rotary power, pure cube rotations keep the mean rotary power, and fourfold rotoinversion alternates signs.
    for centrosymmetric in (rotary_surfaces[2], rotary_surfaces[5]):
        assert (centrosymmetric.norm() < 1e-9).all()
    mean_power = mv(RotaryPower, [rotary_values.mean()])
    assert ((rotary_surfaces[4] - mean_power).norm() < 1e-9).all()
    rotoinversion = gyrations[3]
    x_dir, y_dir, z_dir = mv.x.dual(), mv.y.dual(), mv.z.dual()
    assert ((rotoinversion(x_dir, x_dir) + rotoinversion(y_dir, y_dir)).norm() < 1e-9).all()
    assert (rotoinversion(z_dir, z_dir).norm() < 1e-9).all()
    inverted_conduction = mv.xyz >> responses[0](mv.xyz << Field)
    assert ((inverted_conduction(probes) - responses[0](probes)).norm() < 1e-9).all()


if __name__ == "__main__":
    main()
