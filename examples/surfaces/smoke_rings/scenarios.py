"""Scenes for the smoke ring flow: a ring flying along its axis without changing shape, a ring carrying a
standing wave and the Hashimoto surface it sweeps, and a trefoil knot of vortex flowing; the same flow in
discrete time, by Bäcklund transformations; a soliton, a loop running along a straight vortex line, made
by one transformation and carried by the others; and two coaxial vortex rings leapfrogging under the
Biot–Savart law, and the tubes they sweep."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate, stack
from examples.surfaces.smoke_rings import core

mv = core.mv
# The ring's vertices; the wavy ring's vertices, the waves around it and the tilt of its edges; the
# trefoil's windings around the axis and around the tube, the tube's radii and its vertices.
RING_COUNT = 60
WAVY_COUNT = 90
WAVY_MODE = 3
WAVY_TILT = 0.4
KNOT_WINDINGS = (2, 3)
KNOT_RADII = (6.0, 2.5)
KNOT_COUNT = 240
# The flows' steps, their numbers of steps and the steps between kept curves: the wavy ring's unit edges
# allow a longer step than the knot's shorter ones. The duration of a frame.
SURFACE_STEP = 0.4
SURFACE_STEPS = 750
SURFACE_EVERY = 10
KNOT_STEP = 0.05
KNOT_STEPS = 2400
KNOT_EVERY = 40
DURATION_MS = 60
# The discrete-time flow: its rings' vertices, the Bäcklund parameter, and the steps, each a time of about
# half the parameter along the flow.
DISCRETE_COUNT = 40
PARAMETER = 0.5
DISCRETE_STEPS = 100
# The leapfrogging rings: their radius, vertices, heights along the axis, circulations and core radius;
# the flow's step, number of steps and steps between kept rings, through two cycles of leapfrogging.
FROG_RADIUS = 5.0
FROG_COUNT = 48
FROG_HEIGHTS = np.array([0.0, 2.5])
FROG_CIRCULATIONS = mv(core.Circulation, [[10.0], [10.0]])
FROG_CORE = 0.4
FROG_STEP = 0.4
FROG_STEPS = 50
FROG_EVERY = 1


# The soliton's straight line of unit edges; the length of the offsets that make the loop, their
# transformation's parameter, and how far the first offset leans off the line; the parameter of the
# transformations that carry it, their number, and the steps between frames.
SOLITON_COUNT = 75
SOLITON_REACH = 4.0
SOLITON_PARAMETER = 2.0
SOLITON_LEAN = 1e-5
CARRY_PARAMETER = 1.0
SOLITON_STEPS = 90
SOLITON_EVERY = 2

# The smoke ring flow's own circulation, one in its rescaled time, and its velocity.
CIRCULATION = mv(core.Circulation, [1.0])


def flying(points: core.Vector) -> core.Vector:
    return core.velocity(points, CIRCULATION)

# --- math -----------------------------------------------------------------------------
def flying_ring() -> core.Vector:
    """A ring's velocities `[n] Vector`: all one, along its axis, at twice the tangent of half its turn
    per vertex."""
    velocities = core.velocity(core.ring(RING_COUNT), CIRCULATION)               # [n] Vector

    # --- checks
    np.testing.assert_allclose((velocities - velocities[0]).norm().to_array(), 0.0, atol=1e-13)
    np.testing.assert_allclose((velocities | mv.z).to_array(), -2 * np.tan(np.pi / RING_COUNT), rtol=1e-12)
    return velocities


def sweep(points: core.Vector, step: float, steps: int, every: int) -> core.Vector:
    """The curves of the smoke ring flow `[times, n] Vector`: kept edge lengths and impulse checked."""
    curves = stack(list(core.flow(points, flying, step, steps, every)))          # [times, n] Vector

    # --- checks
    # The flow keeps every edge's length and the polygon's vector area; the tolerance on the area tests
    # the time steps.
    lengths = core.edges(curves).norm().to_array()                               # [times, n]
    np.testing.assert_allclose(lengths, np.broadcast_to(lengths[0], lengths.shape), atol=1e-9)
    areas = core.impulse(curves).kernel                                          # [times, 3]
    np.testing.assert_allclose(areas, np.broadcast_to(areas[0], areas.shape), rtol=1e-3, atol=1e-3 * np.abs(areas[0]).max())
    return curves


def wavy_surface() -> core.Vector:
    """The Hashimoto surface of the wavy ring: its curves through the flow `[times, n] Vector`."""
    return sweep(core.wavy_ring(WAVY_COUNT, WAVY_MODE, WAVY_TILT), SURFACE_STEP, SURFACE_STEPS, SURFACE_EVERY)


def knot() -> core.Vector:
    """The trefoil of vortex through the flow `[times, n] Vector`."""
    return sweep(core.torus_knot(*KNOT_WINDINGS, *KNOT_RADII, KNOT_COUNT), KNOT_STEP, KNOT_STEPS, KNOT_EVERY)


def discrete() -> core.Vector:
    """The wavy ring through the discrete-time flow, one Bäcklund transformation per step: its curves
    `[steps + 1, n] Vector`, which sweep a discrete Hashimoto surface."""
    rings = stack([core.ring(DISCRETE_COUNT), core.wavy_ring(DISCRETE_COUNT, WAVY_MODE, WAVY_TILT)])   # [rings, n] Vector
    twist = mv.xyz * PARAMETER                                                   # [] Pseudoscalar

    def stepped() -> Iterator[core.Vector]:
        curves = rings
        yield curves
        for _ in range(DISCRETE_STEPS):
            curves = core.advance(curves, twist)
            yield curves

    swept = stack(list(stepped()))                                               # [steps + 1, rings, n] Vector

    # --- checks
    # Every edge stays exactly of unit length.
    np.testing.assert_allclose(core.edges(swept).norm().to_array(), 1.0, atol=1e-12)
    # The ring, a discrete elastic curve, moves rigidly: all its points keep one radius and one height.
    ring = swept[-1, 0]
    np.testing.assert_allclose((ring - mv.z * (ring | mv.z)).norm().to_array(), (ring[0] - mv.z * (ring[0] | mv.z)).norm().to_array(), rtol=1e-12)
    np.testing.assert_allclose((ring | mv.z).to_array(), (ring[0] | mv.z).to_array(), atol=1e-12)
    # Each step is a time of about half the parameter along the smoke ring flow, closer the smaller the
    # parameter; the tolerance tests the discrete time against Runge–Kutta's, relative to the travel.
    continuous = list(core.flow(rings[1], flying, PARAMETER / 2, DISCRETE_STEPS, DISCRETE_STEPS))[-1]
    travel = (continuous - rings[1]).norm().to_array().max()
    assert (swept[-1, 1] - continuous).norm().to_array().max() < 0.2 * travel
    return swept[:, 1]


def soliton() -> core.Vector:
    """A loop made on a straight vortex line by one Bäcklund transformation, from an offset leaning just
    off the line's forward direction, and carried along it by the discrete-time flow: the curves
    `[steps + 1, n] Vector`."""
    line = mv.x * (np.arange(SOLITON_COUNT) - SOLITON_COUNT / 2)                 # [n] Vector
    start = ((mv.zx * SOLITON_LEAN).exp() >> mv.x) * SOLITON_REACH               # [] Vector
    twist, carry = mv.xyz * SOLITON_PARAMETER, mv.xyz * CARRY_PARAMETER          # [] Pseudoscalar

    def carried() -> Iterator[core.Vector]:
        curve = core.backlund(line, start, twist)                    # [n] Vector
        yield curve
        for _ in range(SOLITON_STEPS):
            # Offsets of unit length from minus the first edge: on the straight end, each vertex moves
            # onto the one before, and relabelled the end stays where it is.
            moved = core.backlund(curve, curve[0] - curve[1], carry)[1:]     # [n - 1] Vector
            # The line runs on straight past its last vertex.
            curve = concatenate([moved, 2 * moved[-1:] - moved[-2:-1]])        # [n] Vector
            yield curve

    curves = stack(list(carried()))                                              # [steps + 1, n] Vector

    # --- checks
    # Every edge stays exactly of unit length.
    np.testing.assert_allclose((curves[:, 1:] - curves[:, :-1]).norm().to_array(), 1.0, atol=1e-12)
    # The tangent turns once around through the loop, at every step; the tolerance allows for the loop's
    # tails beyond the line's ends.
    angles = core.turns(curves).log().norm().to_array()                          # [steps + 1, n - 2]
    np.testing.assert_allclose(angles.sum(axis=-1), 2 * np.pi, atol=1e-3)
    # The loop keeps its shape: the spread of the turning along the line stays the same as it travels,
    # up to how the vertices sample it.
    along = np.arange(angles.shape[-1])
    centres = (angles * along).sum(axis=-1) / (2 * np.pi)                       # [steps + 1]
    spreads = (angles * (along - centres[:, None]) ** 2).sum(axis=-1) / (2 * np.pi)
    np.testing.assert_allclose(spreads, spreads[0], rtol=1e-2)
    # It travels further than its own width.
    assert centres[-1] - centres[0] > 2 * np.sqrt(spreads[0])
    return curves


def leapfrog() -> core.Vector:
    """Two coaxial vortex rings moving each other by the Biot–Savart law, through two cycles of
    leapfrogging: the rings `[times, rings, n] Vector`."""
    rings = core.coaxial_rings(FROG_RADIUS, FROG_COUNT, FROG_HEIGHTS)              # [rings, n] Vector
    induced = lambda points: core.induced(points, FROG_CIRCULATIONS, FROG_CORE)
    curves = stack(list(core.flow(rings, induced, FROG_STEP, FROG_STEPS, FROG_EVERY)))   # [times, rings, n] Vector

    # --- checks
    # A ring alone flies along its axis without changing shape.
    alone = core.induced(rings[:1], FROG_CIRCULATIONS[:1], FROG_CORE)            # [1, n] Vector
    np.testing.assert_allclose((alone - alone[:, :1]).norm().to_array(), 0.0, atol=1e-12)
    # The rings' impulse, each ring's vector area times its circulation, is kept; the tolerance tests the
    # steps along the filaments and in time.
    impulse = (core.impulse(curves) * FROG_CIRCULATIONS).sum(axis=-1).kernel     # [times, 3]
    np.testing.assert_allclose(impulse, np.broadcast_to(impulse[0], impulse.shape), atol=3e-3 * np.abs(impulse[0]).max())
    # The rings take turns: each passes through the other, twice in each cycle, and the ring passing is
    # the narrower one.
    heights = (curves | mv.z).to_array().mean(axis=-1)                           # [times, rings]
    radii = (curves - mv.z * (curves | mv.z)).norm().to_array().mean(axis=-1)    # [times, rings]
    passes = np.flatnonzero(np.diff(np.sign(heights[:, 1] - heights[:, 0])))   # the times just before a pass
    assert len(passes) >= 4
    behind = np.argmin(heights[passes], axis=-1)                                 # [passes]: the ring overtaking
    assert np.all(radii[passes, behind] < radii[passes, 1 - behind])
    return curves


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.surfaces.smoke_rings import render

    flying_ring()
    save_figure(render.draw_surface(core.closed(wavy_surface())), "smoke_rings_surface")
    save_animation(render.animate_curves(core.closed(knot())), "smoke_rings_knot", DURATION_MS)
    save_figure(render.draw_surface(core.closed(discrete())), "smoke_rings_discrete")
    solitons = soliton()
    save_animation(render.animate_trail(solitons[::SOLITON_EVERY]), "smoke_rings_soliton", DURATION_MS)
    save_figure(render.draw_turns(core.turns(solitons)), "smoke_rings_turns")
    frogs = leapfrog()
    save_figure(render.draw_swept(core.closed(frogs)), "smoke_rings_swept")
    save_animation(render.animate_curves(core.closed(frogs)), "smoke_rings_leapfrog", DURATION_MS)
