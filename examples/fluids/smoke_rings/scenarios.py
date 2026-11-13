"""Scenes for the smoke ring flow, a ring flying along its axis without changing shape, a ring carrying a standing wave
and the Hashimoto surface it sweeps, and a trefoil knot of vortex flowing; the same flow in discrete time by Bäcklund
transformations; a soliton, a loop running along a straight vortex line, made by one transformation and carried by
the others; and two coaxial vortex rings leapfrogging under the Biot–Savart law, and the tubes they sweep."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import concatenate, stack
from examples.fluids.smoke_rings import core

mv = core.mv
# The ring's vertices; the wavy ring's vertices, the waves around it and the tilt of its edges; the trefoil's windings
# around the axis and around the tube, the tube's radii and its vertices.
n_ring = 60
n_wavy = 90
wavy_mode = 3
wavy_tilt = 0.4
knot_windings = (2, 3)
knot_radii = (6.0, 2.5)
n_knot = 240
# The flows' steps, their numbers of steps and the steps between kept curves; the wavy ring's unit edges allow a
# longer step than the knot's shorter ones. The duration of a frame.
surface_step = 0.4
n_surface_steps = 750
surface_stride = 10
knot_step = 1 / 30
n_knot_steps = 3600
knot_stride = 40
frame_duration_ms = 40
# The discrete-time flow, its rings' vertices, the Bäcklund parameter, and the steps, each a time of about half the
# parameter along the flow.
n_discrete = 40
discrete_twist = mv(core.Parameter, [0.5])                                       # [] Parameter-
n_discrete_steps = 100
# The leapfrogging rings, their radius, vertices, heights along the axis, circulations and core radius; the flow's
# step and number of steps, through two cycles of leapfrogging, and the steps between frames.
frog_radius = 5.0
n_frog = 48
frog_heights = np.array([0.0, 2.5])
frog_circulations = mv(core.Circulation, [[10.0], [10.0]])                       # [2] Circulation-
frog_core_radius = 0.4
frog_step = 0.4 / 3
n_frog_steps = 150
frog_stride = 2
# The soliton's straight line of unit edges; the length of the offsets that make the loop, their transformation's
# parameter, and how far the first offset leans off the line; the parameter of the transformations that carry it,
# and their number, one a frame.
n_soliton = 75
soliton_offset_length = 4.0
soliton_twist = mv(core.Parameter, [2.0])                                        # [] Parameter-
soliton_lean = 1e-5
carry_twist = mv(core.Parameter, [1.0])                                          # [] Parameter-
n_soliton_steps = 90
# The smoke ring flow's own circulation, one in its rescaled time.
circulation = mv(core.Circulation, [1.0])                                        # [] Circulation-


# --- shapes ---------------------------------------------------------------------------
def ring(n_vertices: int) -> core.Point:
    """A regular polygon of unit edges around the z axis."""
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    around = (mv.xy * (-angles / 2)).exp()                                       # [n_vertices] Rotor+
    return (around >> mv.x.dual()) / (2 * np.sin(np.pi / n_vertices))            # [n_vertices] Point-


def coaxial_rings(radius: float, n_vertices: int, heights: np.ndarray) -> core.Point:
    """Regular polygons of the given radius around the z axis, one at each height along it."""
    scaled = ring(n_vertices) * (radius * 2 * np.sin(np.pi / n_vertices))        # [n_vertices] Point-
    return scaled[None, :] + mv.z.dual() * heights[:, None]                      # [..., n_vertices] Point-


def wavy_ring(n_vertices: int, mode: int, tilt: float) -> core.Point:
    """A closed polygon of unit edges around the z axis, each edge tilted out of the plane by `tilt` times the sine of
    `mode` times its angle around the axis, a ring with a standing wave on it. The tilts' sines and cosines have no
    component going once around the axis, so the edges close."""
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    around = (mv.xy * (-angles / 2)).exp()                                       # [n_vertices] Rotor+
    # Each edge tilted out of the plane, before turning around the axis.
    tilts = (mv.yz * (-tilt * np.sin(mode * angles) / 2)).exp()                  # [n_vertices] Rotor+
    unit_edges = around >> (tilts >> mv.y.dual())                                # [n_vertices] Point-
    return concatenate([mv.vector([[0, 0, 0]]).dual(), unit_edges[:-1]]).cumsum(axis=0)   # [n_vertices] Point-


def torus_knot(around_axis: int, around_tube: int, major_radius: float, minor_radius: float, n_vertices: int) -> core.Point:
    """A closed polygon on a torus knot, winding `around_axis` times around the z axis and `around_tube` times around
    the tube of the given radii."""
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    tube = (mv.zx * (-around_tube * angles / 2)).exp() >> (mv.x.dual() * minor_radius)   # [n_vertices] Point-
    return (mv.xy * (-around_axis * angles / 2)).exp() >> (mv.x.dual() * major_radius + tube)   # [n_vertices] Point-


# --- math -----------------------------------------------------------------------------
def smoke_ring_velocity(points: core.Point) -> core.Point:
    """The smoke ring flow's velocity at every vertex."""
    return core.velocity(points, circulation)                                    # [...] Point-


def line_speeds(curves: core.Point) -> core.Scalar:
    """The smoke ring flow's speed at every vertex of open lines, inside them as on a closed curve, and at the two
    ends, which the closing edge would bend, their neighbours'."""
    inner = smoke_ring_velocity(curves).norm()[..., 1:-1]                        # [...] Scalar+
    return concatenate([inner[..., :1], inner, inner[..., -1:]], axis=-1)        # [...] Scalar+


def leapfrog_velocity(points: core.Point) -> core.Point:
    """The velocity the leapfrogging rings induce at every vertex of both."""
    return core.induced_velocity(points, frog_circulations, frog_core_radius)    # [2, n_frog] Point-


def flow_curves(points: core.Point, step: float, n_steps: int, stride: int) -> core.Point:
    """The curves of the smoke ring flow every `stride` steps."""
    return stack(list(islice(core.flow(points, smoke_ring_velocity, step), 0, n_steps + 1, stride)))   # [...] Point-


def wavy_surface() -> core.Point:
    """The Hashimoto surface of the wavy ring, its curves through the flow."""
    wavy = wavy_ring(n_wavy, wavy_mode, wavy_tilt)                               # [n_wavy] Point-
    return flow_curves(wavy, surface_step, n_surface_steps, surface_stride)      # [...] Point-


def knot_curves() -> core.Point:
    """The trefoil of vortex through the flow."""
    knot = torus_knot(*knot_windings, *knot_radii, n_knot)                       # [n_knot] Point-
    return flow_curves(knot, knot_step, n_knot_steps, knot_stride)               # [...] Point-


def discrete_curves() -> core.Point:
    """The plain and the wavy ring through the discrete-time flow, one Bäcklund transformation per step, the wavy
    ring's curves sweeping a discrete Hashimoto surface."""
    rings = stack([ring(n_discrete), wavy_ring(n_discrete, wavy_mode, wavy_tilt)])   # [2, n_discrete] Point-

    def discrete_steps() -> Iterator[core.Point]:
        curves = rings
        while True:
            yield curves
            curves = core.time_step(curves, discrete_twist)

    return stack(list(islice(discrete_steps(), n_discrete_steps + 1)))           # [n_discrete_steps + 1, 2, n_discrete] Point-


def soliton_curves() -> core.Point:
    """A loop made on a straight vortex line by one Bäcklund transformation, from an offset leaning just off the line's
    forward direction, and carried along it by the discrete-time flow."""
    line = mv.x.dual() * (np.arange(n_soliton) - n_soliton / 2)                  # [n_soliton] Point-
    start = ((mv.zx * soliton_lean).exp() >> mv.x.dual()) * soliton_offset_length   # [] Point-

    def soliton_steps() -> Iterator[core.Point]:
        curve = core.backlund(line, start, soliton_twist)                        # [n_soliton] Point-
        while True:
            yield curve
            # Offsets of unit length from minus the first edge; on the straight end each vertex moves onto the one
            # before, and relabelled the end stays where it is.
            relabelled = core.backlund(curve, curve[0] - curve[1], carry_twist)[1:]   # [n_soliton - 1] Point-
            # The line runs on straight past its last vertex.
            curve = concatenate([relabelled, 2 * relabelled[-1:] - relabelled[-2:-1]])   # [n_soliton] Point-

    return stack(list(islice(soliton_steps(), n_soliton_steps + 1)))             # [n_soliton_steps + 1, n_soliton] Point-


def leapfrog_rings() -> core.Point:
    """Two coaxial vortex rings moving each other by the Biot–Savart law through two cycles of leapfrogging."""
    rings = coaxial_rings(frog_radius, n_frog, frog_heights)                     # [2, n_frog] Point-
    return stack(list(islice(core.flow(rings, leapfrog_velocity, frog_step), n_frog_steps + 1)))   # [n_frog_steps + 1, 2, n_frog] Point-


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.smoke_rings import render

    ring_velocities = smoke_ring_velocity(ring(n_ring))                          # [n_ring] Point-
    surface = wavy_surface()                                                     # [...] Point-
    save_figure(render.draw_surface(surface), "smoke_rings_surface")
    save_animation(render.animate_curves(surface, smoke_ring_velocity(surface).norm()), "smoke_rings_wavy",
                   frame_duration_ms)
    trefoil = knot_curves()                                                      # [...] Point-
    save_animation(render.animate_curves(trefoil, smoke_ring_velocity(trefoil).norm()), "smoke_rings_knot",
                   frame_duration_ms)
    discrete = discrete_curves()                                                 # [n_discrete_steps + 1, 2, n_discrete] Point-
    save_figure(render.draw_surface(discrete[:, 1]), "smoke_rings_discrete")
    save_animation(render.animate_curves(discrete, smoke_ring_velocity(discrete).norm()), "smoke_rings_discrete_flow",
                   frame_duration_ms)
    solitons = soliton_curves()                                                  # [n_soliton_steps + 1, n_soliton] Point-
    save_animation(render.animate_trail(solitons, line_speeds(solitons)), "smoke_rings_soliton", frame_duration_ms)
    save_figure(render.draw_turns(core.turns(solitons)), "smoke_rings_turns")
    frogs = leapfrog_rings()                                                     # [n_frog_steps + 1, 2, n_frog] Point-
    save_figure(render.draw_swept(frogs), "smoke_rings_swept")
    # Each time's rings pushed by both; the induced flow couples every vertex of a time, and times are a batch.
    frog_speeds = stack([leapfrog_velocity(rings) for rings in frogs]).norm()    # [n_frog_steps + 1, 2, n_frog] Scalar+
    save_animation(render.animate_curves(frogs[::frog_stride], frog_speeds[::frog_stride]), "smoke_rings_leapfrog",
                   frame_duration_ms)

    # checks
    # The ring flies rigidly along its axis at twice the tangent of half its turn per vertex.
    assert ((ring_velocities - ring_velocities[0]).norm() < 1e-13).all()
    assert (((ring_velocities & mv.z) + 2 * np.tan(np.pi / n_ring)).abs() < 1e-12).all()
    # The flows keep every edge's length and the polygon's vector area; the tolerance on the area tests the time
    # steps.
    for curves in (surface, trefoil):
        lengths = core.edges(curves).norm()                                      # [...] Scalar+
        assert ((lengths - lengths[0]).abs() < 1e-9).all()
        areas = core.impulse(curves)                                             # [...] Patch-
        assert ((areas - areas[0]).norm() < 1e-3 * areas[0].norm()).all()
    # In discrete time every edge stays exactly of unit length, and the ring, a discrete elastic curve, moves rigidly,
    # all its points keeping one radius and one height.
    assert ((core.edges(discrete).norm() - 1).abs() < 1e-12).all()
    final_ring = discrete[-1, 0]                                                 # [n_discrete] Point-
    radial = final_ring - mv.z.dual() * (final_ring & mv.z)                      # [n_discrete] Point-
    assert ((radial.norm() - radial[0].norm()).abs() < 1e-12 * radial[0].norm()).all()
    assert (((final_ring - final_ring[0]) & mv.z).abs() < 1e-12).all()
    # The soliton keeps unit edges, and the tangent turns once around through the loop at every step; the tolerance
    # allows for the loop's tails beyond the line's ends.
    assert (((solitons[:, 1:] - solitons[:, :-1]).norm() - 1).abs() < 1e-12).all()
    angles = core.turns(solitons).log().norm()                                   # [n_soliton_steps + 1, n_soliton - 2] Scalar+
    assert ((angles.sum(axis=-1) - 2 * np.pi).abs() < 1e-3).all()
    # The loop keeps its shape, up to how the vertices sample it, and travels further than its width.
    along = np.arange(n_soliton - 2)
    centres = (angles * along).sum(axis=-1) / (2 * np.pi)                        # [n_soliton_steps + 1] Scalar+
    spreads = (angles * (centres[:, None] - along).squared()).sum(axis=-1) / (2 * np.pi)   # [n_soliton_steps + 1] Scalar+
    assert ((spreads - spreads[0]).abs() < 1e-2 * spreads[0]).all()
    assert (centres[-1] - centres[0] > 2 * spreads[0].square_root()).all()
    # The leapfrogging rings keep their impulse, each ring's vector area times its circulation; the tolerance tests
    # the steps along the filaments and in time.
    impulses = (core.impulse(frogs) * frog_circulations).sum(axis=-1)            # [n_frog_steps + 1] Patch-
    assert ((impulses - impulses[0]).norm() < 3e-3 * impulses[0].norm()).all()
    # The rings take turns, each passing through the other twice in each cycle, the narrower one passing.
    heights = (frogs & mv.z).mean(axis=-1)                                       # [n_frog_steps + 1, 2] Scalar+
    radii = (frogs - mv.z.dual() * (frogs & mv.z)).norm().mean(axis=-1)          # [n_frog_steps + 1, 2] Scalar+
    ahead = heights[:, 1] > heights[:, 0]
    crossings = np.flatnonzero(ahead[1:] != ahead[:-1])
    assert len(crossings) >= 4 and (radii[crossings].argmin(axis=-1) == heights[crossings].argmin(axis=-1)).all()


if __name__ == "__main__":
    main()
