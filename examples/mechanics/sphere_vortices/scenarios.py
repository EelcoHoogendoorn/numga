"""Scenes of point vortices on the sphere, each vortex drawn at both of its antipodal points:
rings of five, six and seven equal vortices around the point facing the viewer, of which the ring of
seven breaks up; a dipole and two leapfrogging pairs travelling around a great circle; and the two time
steps compared on the leapfrogging pairs."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from numga import stack
from examples.mechanics.sphere_vortices import core

# Rings of equal vortices around the pole: their numbers, the angle from the pole to the ring, and the
# vortices' core, small against the ring.
RING_COUNTS = (5, 6, 7)
RING_ANGLE = np.radians(25.0)
RING_CORE = 0.02
# A dipole and two leapfrogging pairs on the equator's first meridian, each vortex an angle from the
# equator, positive vortices to the north; the vortices' core.
DIPOLE_ANGLES = np.array([0.3, -0.3])
DIPOLE_CIRCULATIONS = np.array([1.0, -1.0])
LEAPFROG_ANGLES = np.array([0.5, -0.5, 0.275, -0.275])
LEAPFROG_CIRCULATIONS = np.array([1.0, -1.0, 1.0, -1.0])
PAIR_CORE = 0.05
# The view of each animation as (elevation, azimuth) in degrees: the rings from above the pole, the
# travelling pairs from just above the equator.
RING_VIEW, TRAVEL_VIEW = (70.0, -60.0), (20.0, 0.0)
# How far each place is turned off at random, and the seed.
DISTURBANCE = 1e-4
SEED = 0
# The time step, the midpoint step's fixed-point sweeps, the steps, the steps compared between the two
# time steps, the steps between frames, and the duration of a frame.
DT = 0.05
SWEEPS = 3
STEPS = 1200
COMPARED_STEPS = 100
EVERY = 4
DURATION_MS = 60


# --- math -----------------------------------------------------------------------------
def history(vortices: core.Vortices, step: Callable[[core.Vortices], core.Vortices], steps: int) -> core.Pivot:
    """The vortices' pivots at every step `[steps + 1, vortices] Pivot`."""
    def stepped() -> Iterator[core.Pivot]:
        current = vortices
        yield current.pivots
        for _ in range(steps):
            current = step(current)
            yield current.pivots

    return stack(list(stepped()))


def turned(colatitudes: np.ndarray, longitudes: np.ndarray) -> core.Pivot:
    """Points at the given angles from the pole and around it `[vortices]`: the pole, the meet of the
    great circles x = 0 and y = 0, turned about the point (0, 1, 0), then about itself."""
    pole, across = core.mv.x ^ core.mv.y, core.mv.z ^ core.mv.x                      # [] Pivot each
    tilted = (across * (-colatitudes / 2)).exp() >> pole                             # [vortices] Pivot
    return (pole * (-longitudes / 2)).exp() >> tilted                               # [vortices] Pivot


def stepped(vortices: core.Vortices) -> core.Pivot:
    return history(vortices, lambda current: core.midpoint(current, DT, SWEEPS), STEPS)


def rings() -> list[core.Pivot]:
    """Rings of five, six and seven equal vortices around the pole, each place slightly disturbed: the
    pivots `[steps + 1, vortices] Pivot` of each ring."""
    rng = np.random.default_rng(SEED)

    def ring(count: int) -> core.Pivot:
        places = turned(np.full(count, RING_ANGLE), 2 * np.pi * np.arange(count) / count)   # [vortices] Pivot
        nudges = (core.pivot(rng.normal(size=(count, 3))) * DISTURBANCE).exp()       # [vortices] Rotor
        return stepped(core.Vortices(nudges >> places, np.ones(count), RING_CORE))

    histories = [ring(count) for count in RING_COUNTS]

    # --- checks
    # The ring of five keeps its shape, the ring of six nearly; the ring of seven, neutral in the plane,
    # breaks up on the sphere.
    changes = [chord_change(pivots) for pivots in histories]
    assert changes[0] < 0.01 and changes[1] < 0.05 and changes[2] > 0.2
    return histories


def travelling() -> list[core.Pivot]:
    """A dipole and two leapfrogging pairs travelling around the equator: their pivots
    `[steps + 1, vortices] Pivot`."""
    histories = [stepped(core.Vortices(turned(np.pi / 2 - angles, np.zeros(len(angles))), circulations, PAIR_CORE))
                 for angles, circulations in ((DIPOLE_ANGLES, DIPOLE_CIRCULATIONS), (LEAPFROG_ANGLES, LEAPFROG_CIRCULATIONS))]

    # --- checks
    # The dipole keeps its chord, closely: two vortices alone turn rigidly about their centre of
    # vorticity, which the midpoint turns, each vortex about the other, follow to their second order; the
    # tolerance tests that. The leapfrogging pairs stay mirror images across the equator, the mirror
    # taking each positive vortex onto the negative one beside it.
    dipole, frogs = histories
    chords = (dipole[:, 0] - dipole[:, 1]).norm_squared().to_array()               # [steps + 1]
    np.testing.assert_allclose(chords, chords[0], rtol=1e-2)
    # The mirror in the equator takes each positive vortex's flow, its circulation times its bivector, onto the
    # flow of the negative vortex beside it.
    equator = core.mv.z                                                              # [] Plane
    mirrored = equator >> (frogs[:, ::2] * LEAPFROG_CIRCULATIONS[::2])               # [steps + 1, pairs] Pivot
    np.testing.assert_allclose(np.abs((mirrored - frogs[:, 1::2] * LEAPFROG_CIRCULATIONS[1::2]).kernel).max(), 0.0, atol=1e-8)
    return histories


def chord_change(pivots: core.Pivot) -> float:
    """The largest change of any squared chord between two vortices, over the steps `[steps + 1, vortices]`."""
    chords = (pivots[:, :, None] - pivots[:, None, :]).norm_squared().to_array()         # [steps + 1, vortices, vortices]
    return np.abs(chords - chords[0]).max()


def compared() -> tuple[core.Pivot, core.Pivot]:
    """The leapfrogging pairs by the rotors of pairs and by midpoint turns, over the first steps: the
    impulse kept exactly by the one and closely by the other."""
    frogs = core.Vortices(turned(np.pi / 2 - LEAPFROG_ANGLES, np.zeros(len(LEAPFROG_ANGLES))), LEAPFROG_CIRCULATIONS, PAIR_CORE)
    by_pairs = history(frogs, lambda current: core.paired(current, core.schedule(len(frogs.circulations)), DT), COMPARED_STEPS)
    by_midpoints = history(frogs, lambda current: core.midpoint(current, DT, SWEEPS), COMPARED_STEPS)

    # --- checks
    # The pairs keep the impulse to round-off; the midpoint turns, each vortex by its own rotor, only
    # closely, the tolerance testing their accuracy; and both steps agree closely, being of second order.
    for pivots, tolerance in ((by_pairs, 1e-12), (by_midpoints, 1e-2)):
        impulses = core.impulse(core.Vortices(pivots, frogs.circulations, PAIR_CORE)).kernel   # [steps + 1, 3]
        np.testing.assert_allclose(impulses, np.broadcast_to(impulses[0], impulses.shape), atol=tolerance)
    parting = np.sqrt((by_pairs[-1] - by_midpoints[-1]).norm_squared().to_array().max())
    travel = np.sqrt((by_midpoints[-1] - by_midpoints[0]).norm_squared().to_array().max())
    assert parting < 0.05 * travel
    return by_pairs, by_midpoints


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.mechanics.sphere_vortices import render

    save_animation(render.animate(rings(), RING_VIEW, EVERY), "sphere_vortices_rings", DURATION_MS)
    save_animation(render.animate(travelling(), TRAVEL_VIEW, EVERY), "sphere_vortices_travelling", DURATION_MS)
    compared()


if __name__ == "__main__":
    main()
