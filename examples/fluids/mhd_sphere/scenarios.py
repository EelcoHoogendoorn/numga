"""A sun-like sphere whose differential rotation winds up a field running from pole to pole, until the field's
tension pushes back and its wound-up current sheets reconnect into magnetic islands."""

from __future__ import annotations

from itertools import islice, tee

import numpy as np

from numga import stack
from examples.fluids.mhd_sphere import core
from examples.util import shapes

mv = core.mv
# The sphere's subdivision levels.
n_levels = 5
# The differential rotation's strength, and the strength of the field it winds up.
rotation = 1.0
field_strength = 0.25
# Viscosity and resistivity, the Runge–Kutta step, the steps run and the stride between frames, the frame duration, and
# the colours of the animation's palette.
viscosity = 1e-3
resistivity = 1e-3
dt = 0.02
n_steps = 450
stride = 4
frame_duration_ms = 40
n_animation_colours = 32


# --- plumbing -------------------------------------------------------------------------
def sphere() -> tuple[core.Topology, core.Point]:
    """The unit sphere, its vertices directions of space."""
    topology, coordinates = shapes.sphere(n_levels)
    return topology, mv(core.Plane, coordinates).dual().field()


# --- math -----------------------------------------------------------------------------
def winding(surface: core.Surface, vertices: core.Point) -> core.Scalar:
    """The starting vorticity and flux functions `[2] Scalar+[vertices]`: differential rotation, the equator turning
    against the middle latitudes, and a field running from pole to pole, its lines circling the two points where x
    meets the equator."""
    heights = mv.z & vertices                                                    # Scalar+[vertices]
    # The zonal stream function of degree three, and the flux function of degree one.
    stream = rotation * (5 * heights * heights * heights - 3 * heights) / 2      # Scalar+[vertices]
    flux = field_strength * (mv.x & vertices)                                    # Scalar+[vertices]
    return stack((core.laplacian(surface, stream), flux))                        # [2] Scalar+[vertices]


def energies(surface: core.Surface, states: core.Scalar) -> core.Scalar:
    """The kinetic and magnetic energies of states `[..., 2] Scalar+[vertices]`, `[2, ...] Amount-`: half of
    each function paired with the stiffness's amount of it, the squared gradients integrated over the sphere."""
    vorticity, flux = states[..., 0], states[..., 1]                             # [...] Scalar+[vertices], [...] Scalar+[vertices]
    stream = core.stream_function(surface, vorticity)                            # [...] Scalar+[vertices]
    return stack(((stream * surface.stiffness(stream)).sites.sum(), (flux * surface.stiffness(flux)).sites.sum())) / 2


def frames(surface: core.Surface, states: core.Scalar):
    """The current and flux function of each state, and the field's strength on each face: the steepness of the flux
    function, how much flux crosses a wall of unit length."""
    for vorticity, flux in states:
        yield core.laplacian(surface, flux), flux, surface.gradient(flux).norm()


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.mhd_sphere import render

    topology, vertices = sphere()
    surface = core.surface(topology, vertices)
    start = winding(surface, vertices)                                           # [2] Scalar+[vertices]
    # Every stride-th state, drawn as it comes and kept for the energies.
    drawn, kept = tee(islice(core.evolve(surface, start, viscosity, resistivity, dt, n_steps), 0, None, stride))
    save_animation(render.animate(topology, vertices, frames(surface, drawn)), "mhd_sphere", frame_duration_ms,
                   colors=n_animation_colours)
    states = stack(tuple(kept))                                                  # [frames, 2] Scalar+[vertices]
    kinetic, magnetic = energies(surface, states)                                # [frames] Amount-, [frames] Amount-
    save_figure(render.draw_energies(np.arange(0, n_steps + 1, stride) * dt, kinetic, magnetic), "mhd_sphere_energies")

    # checks
    # Without viscosity and resistivity the flow and field trade energy and keep the total, the cross helicity and the
    # mean square flux: at a state from the run, the rates pair to nothing with each.
    vorticity, flux = states[len(states) // 4]                                   # Scalar+[vertices], Scalar+[vertices]
    vorticity_rate, flux_rate = core.rate(surface, stack((vorticity, flux)), 0.0, 0.0)   # Scalar+[vertices], Scalar+[vertices]
    stream, current = core.stream_function(surface, vorticity), core.laplacian(surface, flux)   # Scalar+[vertices], Scalar+[vertices]
    unit = mv.scalar([1.0])
    energy_rate = (stream * surface.mass(vorticity_rate) + current * surface.mass(flux_rate)).sites.sum() & unit
    cross_helicity_rate = (flux * surface.mass(vorticity_rate) + vorticity * surface.mass(flux_rate)).sites.sum() & unit
    flux_square_rate = (flux * surface.mass(flux_rate)).sites.sum() & unit
    # The scale of each rate: the energy the flow carries through the field, as fast as it does.
    scale = ((current * surface.mass(core.bracket(surface, stream, flux))).sites.sum() & unit).abs()
    for invariant_rate in (energy_rate, cross_helicity_rate, flux_square_rate):
        np.testing.assert_allclose(invariant_rate.to_array(), 0.0, atol=1e-13 * scale.to_array())
    # With them, the total energy only falls.
    assert np.all(np.diff(((kinetic + magnetic) & unit).to_array()) < 0)


if __name__ == "__main__":
    main()
