"""Magnetohydrodynamics on a sphere: the ideal rates keep energy, cross helicity and mean square flux for any flow and
field, the field takes energy from the rotation that winds it up, dissipation drains it, and the pictures draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.fluids.mhd_sphere import core, render, scenarios
from examples.util import shapes

mv = core.mv
unit = mv.scalar([1.0])


def coarse_sphere():
    topology, coordinates = shapes.sphere(3)
    vertices = mv(core.Plane, coordinates).dual().field()
    return topology, vertices, core.surface(topology, vertices)


def test_the_ideal_rates_keep_energy_cross_helicity_and_flux_for_any_flow_and_field():
    topology, vertices, surface = coarse_sphere()
    x, y, z = mv.x & vertices, mv.y & vertices, mv.z & vertices
    # A flow and a field with no symmetry between them.
    vorticity = core.laplacian(surface, x * y + z * z * z - 0.3 * y * z)
    flux = 0.3 * (x * z + y * y * x - 0.2 * z)
    vorticity_rate, flux_rate = core.rate(surface, stack((vorticity, flux)), 0.0, 0.0)
    stream, current = core.stream_function(surface, vorticity), core.laplacian(surface, flux)
    m = surface.mass
    scale = ((current * m(core.bracket(surface, stream, flux))).sites.sum() & unit).abs().to_array()
    for invariant_rate in ((stream * m(vorticity_rate) + current * m(flux_rate)).sites.sum(),
                           (flux * m(vorticity_rate) + vorticity * m(flux_rate)).sites.sum(),
                           (flux * m(flux_rate)).sites.sum()):
        np.testing.assert_allclose((invariant_rate & unit).to_array(), 0.0, atol=1e-13 * scale)


def test_the_rotation_winds_up_the_field_dissipation_drains_it_and_the_pictures_draw():
    topology, vertices, surface = coarse_sphere()
    start = scenarios.winding(surface, vertices)
    states = stack(tuple(core.evolve(surface, start, 1e-3, 1e-3, 0.02, 40)))   # [steps + 1, 2] Scalar+[vertices]
    kinetic, magnetic = scenarios.energies(surface, states)
    kinetic, magnetic = (kinetic & unit).to_array(), (magnetic & unit).to_array()
    assert magnetic[-1] > 2 * magnetic[0]
    assert np.all(np.diff(kinetic + magnetic) < 0)
    assert len(list(render.animate(topology, vertices, scenarios.frames(surface, states[::20])))) == 3
    energies = scenarios.energies(surface, states)
    figure = render.draw_energies(np.arange(41) * 0.02, energies[0], energies[1])
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
