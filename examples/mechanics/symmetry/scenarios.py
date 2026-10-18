"""Scenes for the symmetry example: a conducting crystal, a three-armed flywheel, a cubic lattice,
and optically active crystals."""

from __future__ import annotations

import numpy as np

from numga import Extensor
from examples.mechanics.symmetry import core
from examples.mechanics.symmetry.core import Vector, mv, pga_mv


def heat_conduction():
    """A tilted positive conductivity, averaged over three rotation groups, probed on the unit sphere.

    Mapping unit driving fields gives an ellipsoid of heat-flow vectors. Half turns
    align its axes; quarter turns about z make a spheroid; cube rotations make a sphere.
    This isotropy follows for rank-two conductivity; rank-four cubic elasticity can
    still be anisotropic.
    """
    # Inputs: a positive conductivity in a tilted principal frame, and candidate symmetries.
    pose = (mv.xy * 0.3 + mv.yz * 0.16).exp()
    axes = pose >> Extensor.stack((mv.x, mv.y, mv.z))
    gains = np.array([4.5, 1.5, 0.6])
    driving = (mv.x + mv.y + mv.z).normalized()
    sphere = core.directions()
    groups = {
        "Half-turns about x and z\n3 independent components": (core.turns(mv.yz, 2)[:, None] * core.turns(mv.xy, 2)[None, :]).reshape(-1),
        "Quarter-turns about z\n2 independent components": core.turns(mv.xy, 4),
        "Rotations of a cube\n1 independent component": core.cube_rotations(),
    }
    responses = core.conductivities(axes, gains, list(groups.values()))

    # Half turns eliminate off-diagonal coupling; quarter turns also equate x and y.
    # For this rank-two response, fourfold symmetry already implies axial symmetry.
    # Cube rotations equate all three axes, so heat always flows along the driving field.
    surfaces = responses[:, None, None](sphere[None, :, :])
    fluxes = responses(driving)

    # --- checks
    # Averaging keeps the trace, so the cube-invariant conductivity is the mean gain times the identity.
    assert (surfaces[-1] - sphere * 2.2).norm().to_array().max() < 1e-7
    # Quarter turns about z conduct equally along x and y.
    quarter = responses[2]
    assert abs(((quarter(mv.x) | mv.x) - (quarter(mv.y) | mv.y)).to_array()).max() < 1e-6
    return ["Measured candidate\n6 independent components", *groups], surfaces, driving, fluxes


def flywheel():
    """Three unit-mass arms 120 degrees apart, and moments about axes in the wheel's plane.

    The axes pass through the hub, not each arm's individual center of mass.
    """
    arm = core.arm_samples()
    rotations = (pga_mv.xy * (np.arange(3) * np.pi / 3)).exp()
    angles = np.linspace(0, 2 * np.pi, 181)
    probes = (pga_mv.xy * (-angles / 2)).exp() >> pga_mv.yz
    arm_inertia, inertia = core.flywheel_inertia(arm, rotations)
    arms = rotations[:, None, None] >> arm[None, :, :]

    # Unit bivectors describe rotation about axes in the wheel's plane. The regressive
    # pairing of motion and momentum gives the moment about each chosen axis.
    arm_moments = probes & arm_inertia(probes)
    wheel_moments = probes & inertia(probes)

    # --- checks
    # Threefold symmetry makes every transverse moment equal; about z it is twice as large.
    transverse = wheel_moments.to_array()
    assert np.ptp(transverse) < 1e-8 * transverse.max()
    axial = (pga_mv.xy & inertia(pga_mv.xy)).to_array()
    np.testing.assert_allclose(axial, 2 * transverse.mean(), rtol=1e-8)
    return arms, angles, Extensor.stack((arm_moments, wheel_moments))


def crystal_lattice():
    """Axial and face-diagonal bonds averaged over the cube rotations, probed on the unit sphere.

    For each unit direction `sphere`, the plotted conduction radius is
    `sphere | conductivity(sphere)` and the plotted stiffness is
    `elasticity(sphere, sphere, sphere, sphere)`.
    """
    seeds = Extensor.stack((mv.x, (mv.x + mv.y).normalized()))
    sites, axial, diagonal = core.lattice_samples()
    sphere = core.directions()
    conductivity, elasticity = core.lattice_responses(seeds, core.cube_rotations())

    # Bind a unit direction into every slot to probe imposed uniaxial strain.
    # The response values become radii, making isotropy versus cubic lobes visible.
    conduction_surface = sphere * (sphere | conductivity(sphere))
    stiffness_surface = sphere * elasticity(sphere, sphere, sphere, sphere)

    # --- checks
    # Conduction is isotropic: 2/3 in every direction. Stiffness is 1/2 along the cube
    # axes and 1/3 along the body diagonals.
    np.testing.assert_allclose((sphere | conductivity(sphere)).to_array(), 2 / 3, rtol=1e-7)
    body_diagonal = (mv.x + mv.y + mv.z).normalized()
    np.testing.assert_allclose(elasticity(mv.x, mv.x, mv.x, mv.x).to_array(), 1 / 2, rtol=1e-7)
    np.testing.assert_allclose(elasticity(body_diagonal, body_diagonal, body_diagonal, body_diagonal).to_array(), 1 / 3, rtol=1e-7)
    return sites, axial, diagonal, conduction_surface, stiffness_surface


def optical_activity():
    """A tilted candidate rotary power, averaged over five point groups, probed on the unit sphere.

    The rotations of a square keep two rotary powers, along its axis and across it; adding the inversion
    removes both. Half turns with quarter turns through the inversion keep rotary powers of opposite
    sense along x and y and none along z: such a crystal is its own mirror image, yet turns light along
    the directions its improper rotations exchange. The cube's rotations leave one rotary power, the same in every direction, as in a
    sugar solution; adding the inversion removes it.
    """
    pose = (mv.xy * 0.3 + mv.yz * 0.16).exp()
    axes = pose >> Extensor.stack((mv.x, mv.y, mv.z))
    powers = np.array([3.0, -1.0, 0.5])
    sphere = core.directions()
    inversion = mv.xyz
    quarter = core.turns(mv.xy, 4)
    cube = core.cube_rotations()
    # A group's improper rotations follow its rotations; chiral groups have none, an empty batch.
    groups = {
        "4\nquarter turns": (quarter, inversion * quarter[:0]),
        "4/m\nwith the inversion": (quarter, inversion * quarter),
        "-4\nquarter turns through the inversion": (quarter[::2], inversion * quarter[1::2]),
        "432\nrotations of a cube": (cube, inversion * cube[:0]),
        "m-3m\nwith the inversion": (cube, inversion * cube),
    }
    responses = core.rotary_powers(axes, powers, list(groups.values()))
    surfaces = responses[:, None, None](sphere[None, :, :], sphere[None, :, :])

    # --- checks
    handedness = mv.xyz.dual()
    # The inversion removes every rotary power.
    for centrosymmetric in (surfaces[2], surfaces[5]):
        assert centrosymmetric.norm().to_array().max() < 1e-12
    # The cube's rotations leave the mean rotary power, in every direction.
    assert (surfaces[4] - handedness * powers.mean()).norm().to_array().max() < 1e-12
    # Rotoinversion: opposite senses along x and y, none along z.
    rotoinversion = responses[3]
    assert (rotoinversion(mv.x, mv.x) + rotoinversion(mv.y, mv.y)).norm().to_array() < 1e-12
    assert rotoinversion(mv.z, mv.z).norm().to_array() < 1e-12
    # The same principal values as an extrinsic response, a conductivity, survive the inversion.
    conductivity = (axes * (axes | Vector) * powers).sum(axis=0)
    inverted = inversion >> conductivity(inversion << Vector)
    assert (inverted(sphere) - conductivity(sphere)).norm().to_array().max() < 1e-12
    return ["Measured candidate", *groups], surfaces, sphere


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.mechanics.symmetry import render

    save_figure(render.draw_conduction(*heat_conduction()), "symmetry_conductivity")
    save_figure(render.draw_flywheel(*flywheel()), "symmetry_flywheel")
    save_figure(render.draw_crystal(*crystal_lattice()), "symmetry_crystal_lattice")
    save_figure(render.draw_optical_activity(*optical_activity()), "symmetry_optical_activity")
