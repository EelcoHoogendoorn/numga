"""Scenes for the smoke ring flow: a ring flying along its axis without changing shape, a ring carrying a
standing wave and the Hashimoto surface it sweeps, and a trefoil knot of vortex flowing; the same flow in
discrete time, by Bäcklund transformations; a soliton, a loop running along a straight vortex line, made
by one transformation and carried by the others; and two coaxial vortex rings leapfrogging under the
Biot–Savart law, and the tubes they sweep."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import concatenate, stack
from examples.fluids.smoke_rings import core

mv = core.mv
# The ring's vertices; the wavy ring's vertices, the waves around it and the tilt of its edges; the
# trefoil's windings around the axis and around the tube, the tube's radii and its vertices.
N_RING = 60
N_WAVY = 90
WAVY_MODE = 3
WAVY_TILT = 0.4
KNOT_WINDINGS = (2, 3)
KNOT_RADII = (6.0, 2.5)
N_KNOT = 240
# The flows' steps, their numbers of steps and the steps between kept curves: the wavy ring's unit edges
# allow a longer step than the knot's shorter ones. The duration of a frame.
SURFACE_STEP = 0.4
N_SURFACE_STEPS = 750
SURFACE_STRIDE = 10
KNOT_STEP = 0.05
N_KNOT_STEPS = 2400
KNOT_STRIDE = 40
DURATION_MS = 60
# The discrete-time flow: its rings' vertices, the Bäcklund parameter, and the steps, each a time of about
# half the parameter along the flow.
N_DISCRETE = 40
PARAMETER = 0.5
N_DISCRETE_STEPS = 100
# The leapfrogging rings: their radius, vertices, heights along the axis, circulations and core radius;
# the flow's step and number of steps, through two cycles of leapfrogging.
FROG_RADIUS = 5.0
N_FROG = 48
FROG_HEIGHTS = np.array([0.0, 2.5])
frog_circulations = mv(core.Circulation, [[10.0], [10.0]])
FROG_CORE = 0.4
FROG_STEP = 0.4
N_FROG_STEPS = 50
# The soliton's straight line of unit edges; the length of the offsets that make the loop, their
# transformation's parameter, and how far the first offset leans off the line; the parameter of the
# transformations that carry it, their number, and the steps between frames.
N_SOLITON = 75
SOLITON_OFFSET_LENGTH = 4.0
SOLITON_PARAMETER = 2.0
SOLITON_LEAN = 1e-5
CARRY_PARAMETER = 1.0
N_SOLITON_STEPS = 90
SOLITON_STRIDE = 2
# The smoke ring flow's own circulation, one in its rescaled time.
circulation = mv(core.Circulation, [1.0])


# --- math -----------------------------------------------------------------------------
def flying(points: core.Point) -> core.Point:
    """The smoke ring flow's velocity at every vertex."""
    return core.velocity(points, circulation)


def line_speeds(curves: core.Point) -> core.Scalar:
    """The smoke ring flow's speed at every vertex of open lines `[..., n]`: inside them as on a closed curve; at the
    two ends, which the closing edge would bend, their neighbours'."""
    inner = flying(curves).norm()[..., 1:-1]                                     # [..., n - 2] Scalar+
    return concatenate([inner[..., :1], inner, inner[..., -1:]], axis=-1)        # [..., n] Scalar+


def pushed(points: core.Point) -> core.Point:
    """The velocity the leapfrogging rings induce at every vertex of both `[rings, n]`."""
    return core.induced(points, frog_circulations, FROG_CORE)


def sweep(points: core.Point, step: float, n_steps: int, stride: int) -> core.Point:
    """The curves of the smoke ring flow, every `stride` steps `[times, n] Point`."""
    return stack(list(islice(core.flow(points, flying, step), 0, n_steps + 1, stride)))


def wavy_surface() -> core.Point:
    """The Hashimoto surface of the wavy ring: its curves through the flow `[times, n] Point`."""
    return sweep(core.wavy_ring(N_WAVY, WAVY_MODE, WAVY_TILT), SURFACE_STEP, N_SURFACE_STEPS, SURFACE_STRIDE)


def knot() -> core.Point:
    """The trefoil of vortex through the flow `[times, n] Point`."""
    return sweep(core.torus_knot(*KNOT_WINDINGS, *KNOT_RADII, N_KNOT), KNOT_STEP, N_KNOT_STEPS, KNOT_STRIDE)


def discrete() -> core.Point:
    """The plain and the wavy ring through the discrete-time flow, one Bäcklund transformation per step:
    their curves `[steps + 1, rings, n] Point`, the wavy ring's sweeping a discrete Hashimoto surface."""
    rings = stack([core.ring(N_DISCRETE), core.wavy_ring(N_DISCRETE, WAVY_MODE, WAVY_TILT)])   # [rings, n] Point-
    twist = mv(core.Parameter, [PARAMETER])                                      # [] Parameter-

    def stepped() -> Iterator[core.Point]:
        curves = rings
        while True:
            yield curves
            curves = core.advance(curves, twist)

    return stack(list(islice(stepped(), N_DISCRETE_STEPS + 1)))                  # [steps + 1, rings, n] Point-


def soliton() -> core.Point:
    """A loop made on a straight vortex line by one Bäcklund transformation, from an offset leaning just
    off the line's forward direction, and carried along it by the discrete-time flow: the curves
    `[steps + 1, n] Point`."""
    line = mv.x.dual() * (np.arange(N_SOLITON) - N_SOLITON / 2)                  # [n] Point-
    start = ((mv.zx * SOLITON_LEAN).exp() >> mv.x.dual()) * SOLITON_OFFSET_LENGTH   # [] Point-
    twist, carry = mv(core.Parameter, [SOLITON_PARAMETER]), mv(core.Parameter, [CARRY_PARAMETER])   # [] Parameter-, Parameter-

    def carried() -> Iterator[core.Point]:
        curve = core.backlund(line, start, twist)                                # [n] Point-
        while True:
            yield curve
            # Offsets of unit length from minus the first edge: on the straight end, each vertex moves
            # onto the one before, and relabelled the end stays where it is.
            moved = core.backlund(curve, curve[0] - curve[1], carry)[1:]         # [n - 1] Point-
            # The line runs on straight past its last vertex.
            curve = concatenate([moved, 2 * moved[-1:] - moved[-2:-1]])          # [n] Point-

    return stack(list(islice(carried(), N_SOLITON_STEPS + 1)))                   # [steps + 1, n] Point-


def leapfrog() -> core.Point:
    """Two coaxial vortex rings moving each other by the Biot–Savart law, through two cycles of
    leapfrogging: the rings `[times, rings, n] Point`."""
    rings = core.coaxial_rings(FROG_RADIUS, N_FROG, FROG_HEIGHTS)                # [rings, n] Point-
    return stack(list(islice(core.flow(rings, pushed, FROG_STEP), N_FROG_STEPS + 1)))   # [times, rings, n] Point-


# --- plumbing -------------------------------------------------------------------------
def passes(curves: core.Point) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The mean height and radius of each ring at each time `[times, rings]`, and the times just before
    one ring passes through the other."""
    # Each point paired with the axis: its height along it.
    heights = (curves & mv.z).to_array().mean(axis=-1)                           # [times, rings]
    radii = (curves - mv.z.dual() * (curves & mv.z)).norm().to_array().mean(axis=-1)   # [times, rings]
    return heights, radii, np.flatnonzero(np.diff(np.sign(heights[:, 1] - heights[:, 0])))


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.smoke_rings import render

    velocities = flying(core.ring(N_RING))                                       # [n] Point-
    surface, trefoil = wavy_surface(), knot()
    save_figure(render.draw_surface(core.closed(surface)), "smoke_rings_surface")
    save_animation(render.animate_curves(core.closed(trefoil), core.closed(flying(trefoil).norm())), "smoke_rings_knot",
                   DURATION_MS)
    stepped = discrete()
    save_figure(render.draw_surface(core.closed(stepped[:, 1])), "smoke_rings_discrete")
    solitons = soliton()
    save_animation(render.animate_trail(solitons[::SOLITON_STRIDE], line_speeds(solitons[::SOLITON_STRIDE])),
                   "smoke_rings_soliton", DURATION_MS)
    save_figure(render.draw_turns(core.turns(solitons)), "smoke_rings_turns")
    frogs = leapfrog()
    save_figure(render.draw_swept(core.closed(frogs)), "smoke_rings_swept")
    # Each time's rings pushed by both: the induced flow couples every vertex of a time, and times are a batch.
    frog_speeds = stack([pushed(rings) for rings in frogs]).norm()               # [times, rings, n] Scalar+
    save_animation(render.animate_curves(core.closed(frogs), core.closed(frog_speeds)), "smoke_rings_leapfrog", DURATION_MS)

    # checks
    # The ring flies rigidly along its axis at twice the tangent of half its turn per vertex.
    np.testing.assert_allclose((velocities - velocities[0]).norm().to_array(), 0.0, atol=1e-13)
    np.testing.assert_allclose((velocities & mv.z).to_array(), -2 * np.tan(np.pi / N_RING), rtol=1e-12)
    # The flows keep every edge's length and the polygon's vector area; the tolerance on the area tests
    # the time steps.
    for curves in (surface, trefoil):
        lengths = core.edges(curves).norm().to_array()                           # [times, n]
        np.testing.assert_allclose(lengths, np.broadcast_to(lengths[0], lengths.shape), atol=1e-9)
        areas = core.impulse(curves).kernel                                      # [times, 3]
        np.testing.assert_allclose(areas, np.broadcast_to(areas[0], areas.shape), rtol=1e-3, atol=1e-3 * np.abs(areas[0]).max())
    # In discrete time every edge stays exactly of unit length, and the ring, a discrete elastic curve,
    # moves rigidly: all its points keep one radius and one height.
    np.testing.assert_allclose(core.edges(stepped).norm().to_array(), 1.0, atol=1e-12)
    moved = stepped[-1, 0]
    np.testing.assert_allclose((moved - mv.z.dual() * (moved & mv.z)).norm().to_array(), (moved[0] - mv.z.dual() * (moved[0] & mv.z)).norm().to_array(), rtol=1e-12)
    np.testing.assert_allclose((moved & mv.z).to_array(), (moved[0] & mv.z).to_array(), atol=1e-12)
    # Each step is a time of about half the parameter along the smoke ring flow, closer the smaller the
    # parameter; the tolerance tests the discrete time against Runge–Kutta's, relative to the travel.
    continuous = list(islice(core.flow(stepped[0, 1], flying, PARAMETER / 2), N_DISCRETE_STEPS + 1))[-1]
    travel = (continuous - stepped[0, 1]).norm().to_array().max()
    assert (stepped[-1, 1] - continuous).norm().to_array().max() < 0.2 * travel
    # The soliton keeps unit edges, and the tangent turns once around through the loop at every step;
    # the tolerance allows for the loop's tails beyond the line's ends.
    np.testing.assert_allclose((solitons[:, 1:] - solitons[:, :-1]).norm().to_array(), 1.0, atol=1e-12)
    angles = core.turns(solitons).log().norm().to_array()                        # [steps + 1, n - 2]
    np.testing.assert_allclose(angles.sum(axis=-1), 2 * np.pi, atol=1e-3)
    # The loop keeps its shape, up to how the vertices sample it, and travels further than its width.
    along = np.arange(angles.shape[-1])
    centres = (angles * along).sum(axis=-1) / (2 * np.pi)                        # [steps + 1]
    spreads = (angles * (along - centres[:, None]) ** 2).sum(axis=-1) / (2 * np.pi)
    np.testing.assert_allclose(spreads, spreads[0], rtol=1e-2)
    assert centres[-1] - centres[0] > 2 * np.sqrt(spreads[0])
    # A ring alone flies along its axis without changing shape, and the leapfrogging rings keep their
    # impulse, each ring's vector area times its circulation; the tolerance tests the steps along the
    # filaments and in time.
    alone = core.induced(frogs[0, :1], frog_circulations[:1], FROG_CORE)         # [1, n] Point-
    np.testing.assert_allclose((alone - alone[:, :1]).norm().to_array(), 0.0, atol=1e-12)
    impulse = (core.impulse(frogs) * frog_circulations).sum(axis=-1).kernel      # [times, 3]
    np.testing.assert_allclose(impulse, np.broadcast_to(impulse[0], impulse.shape), atol=3e-3 * np.abs(impulse[0]).max())
    # The rings take turns, each passing through the other twice in each cycle, the narrower one passing.
    heights, radii, crossings = passes(frogs)
    behind = np.argmin(heights[crossings], axis=-1)                              # [passes]: the ring overtaking
    assert len(crossings) >= 4 and np.all(radii[crossings, behind] < radii[crossings, 1 - behind])


if __name__ == "__main__":
    main()
