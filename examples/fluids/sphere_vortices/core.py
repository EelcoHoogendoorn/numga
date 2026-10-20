"""Point vortices on the sphere, in the geometric algebra of the sphere.

A vortex is a bivector: a point of the sphere, the meet of two great circles, oriented by the circulation
around it. Its rotor turns the sphere about the point, and the turn leaves exactly two places fixed, the
antipodal pair the bivector stands for; both are the vortex's points, the invariants of its turn, and a
vortex with both its bivector and its circulation negated is the same vortex. A place on the sphere is held the same way, by its bivector. The flow of the vortex turns every
place about the vortex at a rate of its circulation over two pi times the squared norm of `A & B`, the join
of the vortex and the place, the great circle through the two, whose norm is the sine of the angle between
them. Along each circle about the vortex the speed is the circulation over two pi over the circle's radius, so the circulation around every such circle is the vortex's: the flow is irrotational
everywhere except at the vortex's two points, where its vorticity sits, the circulation at the one and its
negative at the other, as a flow on a closed surface must have. A core radius softens the rate near the two points. The
rate is the same expression as in the plane, where the norm of the join of two points is their distance;
only the algebra's signature changes.

The flow at a place is the sum of the vortices' bivectors, each weighted by its rate there: one
bivector, the generator of the turn the fluid makes there. A vortex moves with the flow of the others.
Two vortices alone keep the angle between them, so the sum of their two weighted bivectors stays the
same while they move, and they turn rigidly about it: about the circulation-weighted sum of their
bivectors. Two time steps follow. The explicit one turns the vortices pair by pair, each pair by its
exact rotor, over the rounds of a round-robin schedule and back. The implicit one turns every vortex by
the rotor of the flow of the others at the middle of its step, found by a few sweeps of fixed-point
iteration. The flow keeps the vortices' impulse, the circulation-weighted sum of their bivectors, for its
symmetry under the turns of the sphere; the pairs' rotors keep it exactly, the midpoint turns closely.

In the notation of point-vortex dynamics on the projective plane, a vortex on the sphere comes with its
antipodal image of opposite circulation, the velocity reads $\\dot{\\mathbf x}_k = \\frac{1}{2\\pi}
\\sum_{j \\ne k} \\Gamma_j \\frac{\\mathbf x_j \\times \\mathbf x_k}{|\\mathbf x_j \\times \\mathbf x_k|^2}$,
and the energy is $H = -\\frac{1}{2\\pi} \\sum_{j < k} \\Gamma_j \\Gamma_k \\log \\tan(\\theta_{jk} / 2)$
with $\\cos \\theta_{jk} = \\mathbf x_j \\cdot \\mathbf x_k$.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from numga import NumpyContext
from numga.algebras import Spherical3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# A point of the sphere oriented by the turn around it, the meet of two great circles: a vortex, or a
# place the flow turns.
Pivot = ga.gatype.bivector()
Rotor = ga.gatype.rotor()


@dataclass(frozen=True)
class Vortices:
    pivots: Pivot                                            # [..., vortices] Pivot
    circulations: np.ndarray                                 # [vortices] circulation of each
    core_radius: float


# --- math -----------------------------------------------------------------------------
def flow(vortices: Vortices, at: Pivot) -> Pivot:
    """The flow of the other vortices at each vortex's place `at` `[..., vortices]`: their bivectors,
    each weighted by its circulation over `2 pi` times the softened squared norm of the join."""
    squared = (at.normalized()[..., :, None] & vortices.pivots.normalized()[..., None, :]).norm_squared()   # [..., at, vortices] Scalar
    others = vortices.circulations * ~np.eye(len(vortices.circulations), dtype=bool)   # [at, vortices]
    rates = others / (2 * np.pi * (squared + vortices.core_radius**2))               # [..., at, vortices] Scalar
    return (vortices.pivots.normalized()[..., None, :] * rates).sum(axis=-1)          # [..., at] Pivot


def midpoint(vortices: Vortices, dt: float, sweeps: int) -> Vortices:
    """One step turning every vortex by the rotor of the flow of the others at the middle of its step,
    the middles found by `sweeps` of fixed-point iteration."""
    later = vortices.pivots                                                            # [..., vortices] Pivot
    for _ in range(sweeps):
        middle = replace(vortices, pivots=((vortices.pivots + later) / 2).normalized())
        later = (flow(middle, middle.pivots) * (-dt / 2)).exp() >> vortices.pivots
    return replace(vortices, pivots=later)


def paired(vortices: Vortices, partners: np.ndarray, dt: float) -> Vortices:
    """One step by the exact rotors of pairs: through the rounds `[rounds, vortices]` of partners and back,
    for half the step each way. A vortex its own partner sits the round out: the turn about itself
    leaves it in place."""
    pivots = vortices.pivots                                                             # [..., vortices] Pivot
    for partner in np.concatenate([partners, partners[::-1]]):
        others = pivots[..., partner]                                                  # [..., vortices] Pivot
        rate = 1 / (2 * np.pi * ((pivots.normalized() & others.normalized()).norm_squared() + vortices.core_radius**2))
        # The two bivectors weighted by circulation: the point the pair turns about.
        both = (pivots * vortices.circulations + others * vortices.circulations[partner]) * rate
        pivots = (both * (-dt / 4)).exp() >> pivots                                   # [..., vortices] Pivot
    return replace(vortices, pivots=pivots)


def impulse(vortices: Vortices) -> Pivot:
    """The circulation-weighted sum of the vortices' bivectors."""
    return (vortices.pivots.normalized() * vortices.circulations).sum(axis=-1)        # [...] Pivot


def energy(vortices: Vortices) -> Scalar:
    """The interaction energy of the vortices, the log of the tangent of half the angle between every two
    of them, softened by the core: with `c` the cosine of the angle, minus `log((b - c) / (b + c))` times
    their circulations over `4 pi b`, with `b` the square root of one plus the squared core radius."""
    pivots = vortices.pivots.normalized()
    # The inner product of two of the bivectors is minus the cosine of the angle between them.
    cosines = -(pivots[..., :, None] | pivots[..., None, :])                            # [..., vortices, vortices] Scalar
    reach = np.sqrt(1 + vortices.core_radius**2)
    products = np.outer(vortices.circulations, vortices.circulations) * ~np.eye(len(vortices.circulations), dtype=bool)
    halves = ((reach - cosines) / (reach + cosines)).log() / (4 * np.pi * reach)     # [..., vortices, vortices] Scalar
    return -(products * halves).sum(axis=-1).sum(axis=-1) / 2                        # [...] Scalar


# --- plumbing -------------------------------------------------------------------------
def pivot(directions: np.ndarray) -> Pivot:
    """The points of the sphere in the given directions `[..., 3]`, of unit size: on the bivectors `yz`,
    `zx` and `xy`, the meets of the great circles y = 0 and z = 0, z = 0 and x = 0, x = 0 and y = 0."""
    return mv.bivector(directions / np.linalg.norm(directions, axis=-1, keepdims=True))


def schedule(count: int) -> np.ndarray:
    """A round robin over `count` vortices by the circle method: every vortex's partner in every round
    `[rounds, count]`, each pair meeting once. For an odd count every vortex sits one round out, as its
    own partner."""
    seats = np.arange(count + count % 2)                                             # an empty seat for odd counts
    partners = []
    for shift in range(len(seats) - 1):
        order = np.concatenate([seats[:1], np.roll(seats[1:], -shift)])
        firsts, seconds = order[: len(seats) // 2], order[::-1][: len(seats) // 2]
        partner = np.concatenate([np.arange(count), [count]])
        partner[firsts], partner[seconds] = seconds, firsts
        # A vortex facing the empty seat keeps itself.
        partner = np.where(partner[:count] == count, np.arange(count), partner[:count])
        partners.append(partner)
    return np.stack(partners)                                                        # [rounds, count]
