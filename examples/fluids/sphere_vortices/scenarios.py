"""Scenes of point vortices on the sphere, each vortex drawn at both of its antipodal points: rings of five, six and seven equal vortices around the point facing the viewer, of which the ring of seven breaks up; a dipole and two leapfrogging pairs travelling around a great circle; and the leapfrogging pairs stepped by the exact rotors of pairs beside the midpoint turns."""

from __future__ import annotations

from collections.abc import Callable

import jax
import numpy as np

from numga import concatenate
from numga.backend.jax import JaxContext
from examples.fluids.sphere_vortices import core

mv = core.mv
# Rings of equal vortices around the pole: their sizes, the angle from the pole to the ring, and the
# vortices' core, small against the ring.
n_ring_vortices = (5, 6, 7)
ring_angle = np.radians(25.0)
ring_core = 0.02
# A dipole and two leapfrogging pairs on the equator's first meridian, each vortex an angle from the
# equator, positive vortices to the north; the vortices' core.
dipole_angles = np.array([0.3, -0.3])
dipole_circulations = np.array([1, -1])
leapfrog_angles = np.array([0.5, -0.5, 0.275, -0.275])
leapfrog_circulations = np.array([1, -1, 1, -1])
pair_core = 0.05
# The view of each animation as (elevation, azimuth) in degrees: the rings from above the pole, the
# travelling pairs from just above the equator.
ring_view, travel_view = (70.0, -60.0), (20.0, 0.0)
# How far each place is turned off at random, and the seed.
disturbance = 1e-4
seed = 0
# The time step, the midpoint step's fixed-point sweeps, the steps of the rings and of the travelling pairs and the
# steps between their frames, the steps compared between the two time steps, and the duration of a frame.
dt = 0.05
n_sweeps = 3
n_ring_steps, ring_stride = 320, 1
n_travel_steps, travel_stride = 640, 2
n_compared_steps = 100
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def history(vortices: core.Pivot, step: Callable[[core.Pivot], core.Pivot], n_steps: int) -> core.Pivot:
    """The vortices at every step, the steps compiled into one loop by `jax.lax.scan`: a handful of vortices takes a few hundred small operations a step, the rotor exponentials above all, which the compiled loop does at once."""
    def advance(current: core.Pivot, _: None) -> tuple[core.Pivot, core.Pivot]:
        successor = step(current)
        return successor, successor

    # The vortices carried into JAX for the loop, and the history back.
    start = vortices.materialize(JaxContext(core.ga))                                # [...] Pivot+
    _, successors = jax.lax.scan(advance, start, length=n_steps)                     # [n_steps, ...] Pivot+
    return concatenate([start[None], successors]).materialize(mv.context)            # [n_steps + 1, ...] Pivot+


def sphere_points(colatitudes: np.ndarray, longitudes: np.ndarray) -> core.Pivot:
    """Points at the given angles from the pole and around it: the pole, the meet of the great circles x = 0 and y = 0, turned about the point (0, 1, 0), then about itself."""
    tilts = (mv.zx * (-colatitudes / 2)).exp()                                      # [...] Rotor+
    spins = (mv.xy * (-longitudes / 2)).exp()                                       # [...] Rotor+
    return spins >> (tilts >> mv.xy)                                                # [...] Pivot+


def rings() -> list[core.Pivot]:
    """Rings of five, six and seven equal vortices of unit circulation around the pole at every step, each vortex turned a little off its place at random."""
    rng = np.random.default_rng(seed)
    histories = []
    for n_vortices in n_ring_vortices:
        places = sphere_points(np.full(n_vortices, ring_angle), 2 * np.pi * np.arange(n_vortices) / n_vortices)   # [n_vortices] Pivot+
        nudges = (mv.bivector(rng.normal(size=(n_vortices, 3))).normalized() * disturbance).exp()   # [n_vortices] Rotor+
        histories.append(history(nudges >> places, lambda current: core.midpoint_step(current, ring_core, dt, n_sweeps),
                                 n_ring_steps))                                      # [n_ring_steps + 1, n_vortices] Pivot+
    return histories


def leapfrogs() -> core.Pivot:
    """The two leapfrogging pairs at their start, each vortex its point scaled by its circulation."""
    return sphere_points(np.pi / 2 - leapfrog_angles, np.zeros(len(leapfrog_angles))) * leapfrog_circulations


def travelling_pairs(n_steps: int) -> list[core.Pivot]:
    """A dipole and two leapfrogging pairs travelling around the equator, at every step, by midpoint turns."""
    dipole = sphere_points(np.pi / 2 - dipole_angles, np.zeros(len(dipole_angles))) * dipole_circulations   # [2] Pivot+
    return [history(vortices, lambda current: core.midpoint_step(current, pair_core, dt, n_sweeps), n_steps)
            for vortices in (dipole, leapfrogs())]


def leapfrogs_by_pair_rotors(n_steps: int) -> core.Pivot:
    """The two leapfrogging pairs at every step, by the exact rotors of pairs."""
    partners = schedule(len(leapfrog_angles))                                        # [n_rounds, 4]
    return history(leapfrogs(), lambda current: core.pair_step(current, partners, pair_core, dt), n_steps)


# --- plumbing -------------------------------------------------------------------------
def schedule(n_vortices: int) -> np.ndarray:
    """A round robin over the vortices by the circle method: every vortex's partner in every round, each pair meeting once. For an odd number of vortices every vortex sits one round out, as its own partner."""
    # An empty seat for an odd number of vortices.
    seats = np.arange(n_vortices + n_vortices % 2)
    partners = []
    for shift in range(len(seats) - 1):
        order = np.concatenate([seats[:1], np.roll(seats[1:], -shift)])
        firsts, seconds = order[: len(seats) // 2], order[::-1][: len(seats) // 2]
        partner = np.concatenate([np.arange(n_vortices), [n_vortices]])
        partner[firsts], partner[seconds] = seconds, firsts
        # A vortex facing the empty seat keeps itself.
        partners.append(np.where(partner[:n_vortices] == n_vortices, np.arange(n_vortices), partner[:n_vortices]))
    return np.stack(partners)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.fluids.sphere_vortices import render

    ring_histories = rings()
    save_animation(render.animate(ring_histories, ring_view, ring_stride), "sphere_vortices_rings", frame_duration_ms)
    dipole, frogs = travelling_pairs(n_travel_steps)
    save_animation(render.animate([dipole, frogs], travel_view, travel_stride), "sphere_vortices_travelling",
                   frame_duration_ms)
    # The pairs' rotors beside the first steps of the midpoint turns, already taken above.
    pair_frogs, midpoint_frogs = leapfrogs_by_pair_rotors(n_compared_steps), frogs[:n_compared_steps + 1]
    save_animation(render.animate([pair_frogs, midpoint_frogs], travel_view, 1), "sphere_vortices_compared",
                   frame_duration_ms)

    # checks
    # The ring of five keeps its shape, the ring of six nearly; the ring of seven, neutral in the plane,
    # breaks up on the sphere.
    changes = []
    for places in ring_histories:
        chords = (places[:, :, None] - places[:, None, :]).norm_squared()             # [n_ring_steps + 1, ...] Scalar+
        changes.append((chords - chords[0]).abs().max())
    assert changes[0] < 0.01 and changes[1] < 0.05 and changes[2] > 0.2
    # The dipole keeps the angle between its vortices, closely: two vortices alone turn rigidly about their sum,
    # which the midpoint turns, each vortex about the other, follow to their second order.
    squared_sines = (dipole[:, 0].normalized() & dipole[:, 1].normalized()).norm_squared()   # [n_travel_steps + 1] Scalar+
    assert ((squared_sines / squared_sines[0] - 1).abs() < 1e-2).all()
    # The mirror in the equator takes each positive vortex onto the negative vortex beside it.
    mirror_images = mv.z >> frogs[:, ::2]                                             # [n_travel_steps + 1, 2] Pivot+
    assert ((mirror_images - frogs[:, 1::2]).norm() < 1e-8).all()
    # The pairs keep the impulse to round-off; the midpoint turns, each vortex by its own rotor, only closely;
    # and both steps agree closely, being of second order.
    for vortices, tolerance in ((pair_frogs, 1e-10), (midpoint_frogs, 1e-2)):
        impulses = vortices.sum(axis=-1)                                              # [n_compared_steps + 1] Pivot+
        assert ((impulses - impulses[0]).norm() < tolerance).all()
    assert (pair_frogs[-1] - midpoint_frogs[-1]).norm().max() < 0.05 * (midpoint_frogs[-1] - midpoint_frogs[0]).norm().max()


if __name__ == "__main__":
    main()
