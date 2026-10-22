"""Scenes of point vortices on the sphere, each vortex drawn at both of its antipodal points:
rings of five, six and seven equal vortices around the point facing the viewer, of which the ring of
seven breaks up; a dipole and two leapfrogging pairs travelling around a great circle; and the two time
steps compared on the leapfrogging pairs."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace

import numpy as np

from numga import concatenate, stack
from examples.fluids.sphere_vortices import core

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
N_SWEEPS = 3
N_STEPS = 1200
N_COMPARED_STEPS = 100
N_STEPS_PER_FRAME = 8
DURATION_MS = 80


# --- math -----------------------------------------------------------------------------
def history(vortices: core.Vortices, step: Callable[[core.Vortices], core.Vortices], n_steps: int) -> core.Pivot:
    """The vortices' pivots at every step `[steps + 1, vortices] Pivot`."""
    def stepped() -> Iterator[core.Pivot]:
        current = vortices
        yield current.pivots
        for _ in range(n_steps):
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
    """The vortices' pivots at every step `[steps + 1, vortices] Pivot`, the midpoint steps compiled into one loop by
    `jax.lax.scan`: a handful of vortices takes a few hundred small operations a step, the rotor exponentials above all,
    which the compiled loop does at once."""
    import jax
    from numga.backend.jax import JaxContext

    def step(pivots: core.Pivot, _) -> tuple[core.Pivot, core.Pivot]:
        later = core.midpoint(replace(vortices, pivots=pivots), DT, N_SWEEPS).pivots
        return later, later

    start = vortices.pivots.materialize(JaxContext(core.ga))                         # [vortices] Pivot
    _, later = jax.lax.scan(step, start, length=N_STEPS)                            # [steps, vortices] Pivot
    return concatenate([start[None], later]).materialize(core.mv.context)


def rings() -> list[core.Pivot]:
    """Rings of five, six and seven equal vortices around the pole, each place slightly disturbed: the
    pivots `[steps + 1, vortices] Pivot` of each ring."""
    rng = np.random.default_rng(SEED)

    def ring(size: int) -> core.Pivot:
        places = turned(np.full(size, RING_ANGLE), 2 * np.pi * np.arange(size) / size)   # [vortices] Pivot
        nudges = (core.pivot(rng.normal(size=(size, 3))) * DISTURBANCE).exp()       # [vortices] Rotor
        return stepped(core.Vortices(nudges >> places, np.ones(size), RING_CORE))

    return [ring(size) for size in RING_COUNTS]


def travelling() -> list[core.Pivot]:
    """A dipole and two leapfrogging pairs travelling around the equator: their pivots
    `[steps + 1, vortices] Pivot`."""
    return [stepped(core.Vortices(turned(np.pi / 2 - angles, np.zeros(len(angles))), circulations, PAIR_CORE))
            for angles, circulations in ((DIPOLE_ANGLES, DIPOLE_CIRCULATIONS), (LEAPFROG_ANGLES, LEAPFROG_CIRCULATIONS))]


def chord_change(pivots: core.Pivot) -> float:
    """The largest change of any squared chord between two vortices, over the steps `[steps + 1, vortices]`."""
    chords = (pivots[:, :, None] - pivots[:, None, :]).norm_squared().to_array()         # [steps + 1, vortices, vortices]
    return np.abs(chords - chords[0]).max()


def compared() -> tuple[core.Pivot, core.Pivot]:
    """The leapfrogging pairs by the rotors of pairs and by midpoint turns, over the first steps: the
    impulse kept exactly by the one and closely by the other."""
    frogs = core.Vortices(turned(np.pi / 2 - LEAPFROG_ANGLES, np.zeros(len(LEAPFROG_ANGLES))), LEAPFROG_CIRCULATIONS, PAIR_CORE)
    by_pairs = history(frogs, lambda current: core.paired(current, core.schedule(len(frogs.circulations)), DT), N_COMPARED_STEPS)
    by_midpoints = history(frogs, lambda current: core.midpoint(current, DT, N_SWEEPS), N_COMPARED_STEPS)
    return by_pairs, by_midpoints


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.fluids.sphere_vortices import render

    histories = rings()
    dipole, frogs = travelling()
    by_pairs, by_midpoints = compared()
    save_animation(render.animate(histories, RING_VIEW, N_STEPS_PER_FRAME), "sphere_vortices_rings", DURATION_MS)
    save_animation(render.animate([dipole, frogs], TRAVEL_VIEW, N_STEPS_PER_FRAME), "sphere_vortices_travelling", DURATION_MS)

    # checks
    # The ring of five keeps its shape, the ring of six nearly; the ring of seven, neutral in the plane,
    # breaks up on the sphere.
    changes = [chord_change(pivots) for pivots in histories]
    assert changes[0] < 0.01 and changes[1] < 0.05 and changes[2] > 0.2
    # The dipole keeps its chord, closely: two vortices alone turn rigidly about their centre of
    # vorticity, which the midpoint turns, each vortex about the other, follow to their second order; the
    # tolerance tests that. The leapfrogging pairs stay mirror images across the equator, the mirror
    # taking each positive vortex onto the negative one beside it.
    chords = (dipole[:, 0] - dipole[:, 1]).norm_squared().to_array()               # [steps + 1]
    np.testing.assert_allclose(chords, chords[0], rtol=1e-2)
    # The mirror in the equator takes each positive vortex's flow, its circulation times its bivector, onto the
    # flow of the negative vortex beside it.
    equator = core.mv.z                                                              # [] Plane
    mirrored = equator >> (frogs[:, ::2] * LEAPFROG_CIRCULATIONS[::2])               # [steps + 1, pairs] Pivot
    np.testing.assert_allclose(np.abs((mirrored - frogs[:, 1::2] * LEAPFROG_CIRCULATIONS[1::2]).kernel).max(), 0.0, atol=1e-8)
    # The pairs keep the impulse to round-off; the midpoint turns, each vortex by its own rotor, only
    # closely, the tolerance testing their accuracy; and both steps agree closely, being of second order.
    for pivots, tolerance in ((by_pairs, 1e-12), (by_midpoints, 1e-2)):
        impulses = core.impulse(core.Vortices(pivots, LEAPFROG_CIRCULATIONS, PAIR_CORE)).kernel   # [steps + 1, 3]
        np.testing.assert_allclose(impulses, np.broadcast_to(impulses[0], impulses.shape), atol=tolerance)
    parting = np.sqrt((by_pairs[-1] - by_midpoints[-1]).norm_squared().to_array().max())
    travel = np.sqrt((by_midpoints[-1] - by_midpoints[0]).norm_squared().to_array().max())
    assert parting < 0.05 * travel


if __name__ == "__main__":
    main()
