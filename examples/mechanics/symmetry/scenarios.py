"""Scenes for the symmetry example: a conducting crystal, a three-armed flywheel, a cubic lattice."""

from __future__ import annotations

import numpy as np

from numga import Extensor
from examples.mechanics.symmetry import core
from examples.mechanics.symmetry.core import mv, pga_mv


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
    gains = mv.scalar([[4.5], [1.5], [0.6]])
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
    assert (surfaces[-1] - sphere * 2.2).norm().to_array().max() < 1e-8
    # Quarter turns about z conduct equally along x and y.
    quarter = responses[2]
    assert abs(((quarter(mv.x) | mv.x) - (quarter(mv.y) | mv.y)).to_array()).max() < 1e-8
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

    The plotted conduction radius is d · K(d); the plotted stiffness is C(d,d,d,d).
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
    np.testing.assert_allclose((sphere | conductivity(sphere)).to_array(), 2 / 3, rtol=1e-8)
    body_diagonal = (mv.x + mv.y + mv.z).normalized()
    np.testing.assert_allclose(elasticity(mv.x, mv.x, mv.x, mv.x).to_array(), 1 / 2, rtol=1e-8)
    np.testing.assert_allclose(elasticity(body_diagonal, body_diagonal, body_diagonal, body_diagonal).to_array(), 1 / 3, rtol=1e-8)
    return sites, axial, diagonal, conduction_surface, stiffness_surface


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.mechanics.symmetry import render

    save_figure(render.draw_conduction(*heat_conduction()), "symmetry_conductivity")
    save_figure(render.draw_flywheel(*flywheel()), "symmetry_flywheel")
    save_figure(render.draw_crystal(*crystal_lattice()), "symmetry_crystal_lattice")
