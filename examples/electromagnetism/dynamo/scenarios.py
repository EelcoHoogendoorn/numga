"""A helical flow stirred in a conducting core, wrapped in a poorly conducting mantle, grows a seed field until the
field's force on the current it carries brakes the flow, and the two settle."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import stack
from examples.electromagnetism.dynamo import core
from examples.util import shapes

mv = core.mv
# The sphere's subdivision levels, the ball's shells and the mantle's outer radius, the core's radius being one, and
# the solid inner core's radius, on the first of the core's shells: Earth's is 0.35 of its core's.
n_levels = 2
n_shells = 6
mantle_radius = 1.5
inner_radius = 0.25
# How much poorer a conductor the mantle is than the core.
mantle_conductance = 1e-2
# The magnetic Reynolds number, the conductivity of the core for a flow whose fastest point moves at unit speed, the
# step, the steps run, those before the animation starts, while the field grows to saturation, the stride between
# frames, and the frame duration.
reynolds = 160.0
dt = 0.05
n_steps = 2400
n_skipped_steps = 1200
stride = 10
frame_duration_ms = 40
# The core's slowest scalar modes the flow's modes are built from, the flow's modes, the fluid's viscosity, and the
# rate at which the stirring pulls the flow back toward the helical flow.
n_scalar_modes = 80
n_modes = 60
# The flow modes drawn, from the slowest to a fine one.
shown_modes = np.array([0, 3, 8, 40])
viscosity = 0.01
pull = 1.0


# --- plumbing -------------------------------------------------------------------------
def ball() -> tuple[core.Topology, np.ndarray, core.Point]:
    """The ball of core and mantle: its topology, its vertices' coordinates and the vertices, directions of space."""
    topology, coordinates = shapes.ball(n_levels, n_shells)
    coordinates = mantle_radius * coordinates
    return topology, coordinates, mv(core.Plane, coordinates).dual().field()


def outer_core(topology: core.Topology, vertices: core.Point) -> tuple[core.Topology, core.Point, core.SparseExtensor]:
    """The fluid outer core, between the inner core and the mantle: its tetrahedra as a topology on its own vertices,
    which run without a gap among the ball's, shell after shell, those vertices, and the map placing a field on its
    tetrahedra among the ball's."""
    centres = topology.incidence[3, 0].mean(axis=0) * vertices                   # Point-[tets]
    radii = centres.squared().abs().square_root()                                # Scalar+[tets]
    fluid = ((radii > inner_radius) & (radii < 1)).reshape(-1)                   # [tets]
    simplices = topology.elements[3][fluid]                                      # [fluid, 4]
    first, last = simplices.min(), simplices.max()
    fluid_vertices = topology.reading(np.arange(first, last + 1)) * vertices     # Point-[fluid vertices]
    return core.Topology(last + 1 - first, simplices - first), fluid_vertices, ~topology.reading(np.flatnonzero(fluid), 3)


# --- math -----------------------------------------------------------------------------
def spheromak(points: core.Point, degree: int, root: float) -> core.Point:
    """The Chandrasekhar–Kendall field of the given degree about z at points `Point-[...]` in the unit ball: a Beltrami
    field, its curl `root` times itself, tangent to the ball's surface, where the spherical Bessel function of that
    degree has the root. It sweeps out along the radius by the Legendre pattern, along the meridians by the slope of the
    radial profile, and turns about z as strongly as the profile."""
    radii = points.squared().abs().square_root()                                 # Scalar+[...]
    cosines = (mv.z & points) / radii                                            # Scalar+[...]
    # The sine and cosine of the radial phase, read off a turn by it.
    phases = (mv.xy * (root * radii)).exp()                                      # Rotor+[...]
    sines, cosine_phases = phases.scalar_product(~mv.xy), phases.scalar_product(mv.scalar([1.0]))   # Scalar+[...], Scalar+[...]
    x = root * radii                                                             # Scalar+[...]
    # The spherical Bessel functions of degrees zero to two, and the slope of x times that of the degree.
    bessels = (sines / x, sines / (x * x) - cosine_phases / x,
               (3 / (x * x) - 1) * sines / x - 3 * cosine_phases / (x * x))
    bessel = bessels[degree]                                                     # Scalar+[...]
    slope = x * bessels[degree - 1] - degree * bessel                            # Scalar+[...]
    # The Legendre pattern of the degree, and its slope across the meridian over the sine of the colatitude.
    legendre, across = ((cosines, -1.0), ((3 * cosines * cosines - 1) / 2, -3 * cosines))[degree - 1]
    radial, up = points / radii, mv.z.dual()                                     # Point-[...], [] Point-
    sweep = radial * (degree * (degree + 1) * bessel * legendre / radii) + (radial * cosines - up) * (slope * across / radii)
    turn = points.commutator(mv.xy) * (-root * bessel * across / radii)
    return sweep + turn


def helical_flow(topology: core.Topology, vertices: core.Point) -> core.Point:
    """The flow in the core at each tetrahedron, `Point-[tets]`, nothing in the mantle: the spheromak of degree one
    about z and that of degree two turned onto x, scaled so the fastest point moves at unit speed."""
    # The roots of the spherical Bessel functions of degrees one and two, where each field is tangent to the core's
    # surface.
    first_root, second_root = 4.493409457909064, 5.763459196894550
    onto_x = (mv.zx * (np.pi / 4)).exp()                                         # [] Rotor+
    # The flow at each tetrahedron's centroid, none of which lies at the centre where the radius vanishes.
    centres = topology.incidence[3, 0].mean(axis=0) * vertices                   # Point-[tets]
    flow = spheromak(centres, 1, first_root) + (onto_x >> spheromak(~onto_x >> centres, 2, second_root))   # Point-[tets]
    inside = mv.scalar((centres.squared().abs().square_root() < 1)[:, None]).field()   # Scalar+[tets]
    flow = flow * inside                                                         # Point-[tets]
    return flow / flow.squared().abs().square_root().batch().max()


def seed(topology: core.Topology, vertices: core.Point, field: core.Field) -> core.Scalar:
    """The potential of a uniform field on the edges, `Scalar+[edges]`: half the field's flux through the triangle from
    the centre to each edge."""
    tails, heads = topology.incidence[1, 0] * vertices                           # Point-[edges], Point-[edges]
    return (field & (tails & heads)) / 2


def frames(operators: core.Ball, flow: core.Flow,
           states: Iterable[tuple[core.Scalar, core.Scalar]]) -> Iterator[tuple[core.Magnetic, core.Point]]:
    """The field and the flow in each tetrahedron, `Magnetic+[tets]` and `Point-[tets]`, frame after frame."""
    for amplitudes, potential in states:
        yield operators.curl(potential), flow.velocity(flow.circulation(amplitudes))


def main() -> None:
    from itertools import islice, tee
    from examples.util.animation import save_animation, save_figure
    from examples.electromagnetism.dynamo import render

    topology, coordinates, vertices = ball()
    operators = core.Ball(topology, vertices)
    fluid_topology, fluid_vertices, embedding = outer_core(topology, vertices)
    flow = core.Flow(core.Ball(fluid_topology, fluid_vertices), embedding, n_scalar_modes, n_modes)
    save_figure(render.draw_modes(operators, mantle_radius, flow.velocity(flow.modes[shown_modes])), "dynamo_modes")
    # The stirred flow, the helical flow's share in each mode.
    stirred = flow.reading(operators.prisms.norm() * helical_flow(topology, vertices).dual())   # Scalar+[modes]
    radii = operators.centres.squared().abs().square_root()                      # Scalar+[tets]
    conductivity = mv.scalar(np.where((radii < 1).reshape(-1), reynolds, reynolds * mantle_conductance)[:, None]).field()   # Scalar+[tets]
    potential = seed(topology, vertices, mv.xy * 1e-2 + mv.yz * 3e-3)            # Scalar+[edges]
    states = core.dynamo(operators, flow, conductivity, viscosity, pull, stirred, potential, dt, n_steps)
    # The saturated dynamo, every stride-th state drawn as it comes from both views, and the whole run kept for the
    # energies.
    above, cutaway, kept = tee(states, 3)
    save_animation(render.animate_lines(operators, frames(operators, flow, islice(above, n_skipped_steps, None, stride))),
                   "dynamo_lines", frame_duration_ms)
    save_animation(render.animate(operators, mantle_radius, frames(operators, flow, islice(cutaway, n_skipped_steps, None, stride))),
                   "dynamo", frame_duration_ms)
    amplitudes, potentials = map(stack, zip(*islice(kept, 0, None, stride)))   # [frames] Scalar+[modes], [frames] Scalar+[edges]
    kinetic = (amplitudes * amplitudes).sites.sum() / 2                          # [frames] Scalar+
    magnetic = (potentials * operators.stiffness(potentials)).sites.sum() / 2    # [frames] Amount-
    save_figure(render.draw_energies(np.arange(0, n_steps + 1, stride) * dt, kinetic, magnetic), "dynamo_energies")

if __name__ == "__main__":
    main()
