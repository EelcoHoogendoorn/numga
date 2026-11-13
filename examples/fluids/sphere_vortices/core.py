r"""Point vortices on the sphere, in the geometric algebra of the sphere.

This example moves point vortices over a sphere by the flow they make together. A point vortex is modelled as a point with a circulation around it, the fluid swirling about it and irrotational everywhere else, and each vortex drifts with the flow of all the others. Equal vortices in a ring turn rigidly about its centre until, past a critical number, the ring breaks up; a pair of opposite vortices travels around the whole sphere, and two such pairs leapfrog as they go.

A vortex is modelled as the bivector of the turn about its point, the meet of two great circles, scaled by its circulation, so that negating both its orientation and its circulation leaves the same element. The flow at a place is one bivector, the sum of the vortices each weighted by its rate there, and that rate is the circulation over two pi times the squared norm of the join `A & B` of vortex and place, the great circle through both: the expression of the plane, where that norm is a distance, read here as a sine. Two time steps are built from rotors: one turns pairs of vortices rigidly by their exact rotors, keeping the impulse `vortices.sum()` to round-off; the other turns every vortex by the rotor of the flow at the middle of its step, `(flow * (-dt / 2)).exp() >> vortices`.

In Gibbs–Heaviside vector notation, each vortex is a unit vector $\mathbf x_k$ accompanied by its antipodal image $-\mathbf x_k$ of circulation $-\Gamma_k$, the velocity sums the two, $\dot{\mathbf x}_k = \frac{1}{4\pi} \sum_{j \ne k} \Gamma_j \, (\mathbf x_j \times \mathbf x_k) \left( \frac{1}{1 - \mathbf x_j \cdot \mathbf x_k} + \frac{1}{1 + \mathbf x_j \cdot \mathbf x_k} \right)$, and an explicit step renormalizes $\mathbf x_k$ back onto the sphere, while turning a pair rigidly about its axis $\mathbf n \propto \Gamma_j \mathbf x_j + \Gamma_k \mathbf x_k$ takes Rodrigues' formula $\mathbf x \cos\phi + (\mathbf n \times \mathbf x) \sin\phi + \mathbf n \, (\mathbf n \cdot \mathbf x)(1 - \cos\phi)$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import Spherical3D as ga

mv = NumpyContext(ga).multivector
# A point of the sphere as the meet of two great circles, oriented by the turn about it: scaled by its
# circulation, a vortex; summed over the vortices by their rates, the flow at a place.
Pivot = ga.gatype.bivector()                               # Bivector+


# --- math -----------------------------------------------------------------------------
def flow(vortices: Pivot, core_radius: float) -> Pivot:
    """The flow of the other vortices at each vortex: their bivectors, each weighted by one over two pi times the squared norm of its join with the place, softened by the core radius."""
    n_vortices = vortices.shape[-1]
    units = vortices.normalized()                          # [..., n_vortices] Pivot+
    # The join of two points is the great circle through them; its norm is the sine of the angle between them,
    # the same from either point of an antipodal pair.
    squared_sines = (units[..., :, None] & units[..., None, :]).norm_squared()   # [..., n_vortices, n_vortices] Scalar+
    # No vortex is moved by its own flow.
    rates = (1 - np.eye(n_vortices)) / (2 * np.pi * (squared_sines + core_radius**2))   # [..., n_vortices, n_vortices] Scalar+
    return (vortices[..., None, :] * rates).sum(axis=-1)   # [..., n_vortices] Pivot+


def midpoint_step(vortices: Pivot, core_radius: float, dt: float, n_sweeps: int) -> Pivot:
    """The vortices one step later, each turned by the rotor of the flow of the others at the middle of its step, the middles found by fixed-point sweeps."""
    successors = vortices                                  # [..., n_vortices] Pivot+
    for _ in range(n_sweeps):
        rotors = (flow((vortices + successors) / 2, core_radius) * (-dt / 2)).exp()   # [...] Rotor+
        successors = rotors >> vortices                    # [...] Pivot+
    return successors


def pair_step(vortices: Pivot, partners: np.ndarray, core_radius: float, dt: float) -> Pivot:
    """The vortices one step later by the exact rotors of pairs, through the rounds of partners for half the step and back for the other half. A vortex that is its own partner sits the round out, the turn about itself leaving it in place."""
    for partner in np.concatenate([partners, partners[::-1]]):
        others = vortices[..., partner]                    # [...] Pivot+
        rates = 1 / (2 * np.pi * ((vortices.normalized() & others.normalized()).norm_squared() + core_radius**2))   # [...] Scalar+
        # Two vortices alone keep the angle between them and turn rigidly about their sum, weighted by the rate.
        rotors = ((vortices + others) * rates * (-dt / 4)).exp()   # [...] Rotor+
        vortices = rotors >> vortices                      # [...] Pivot+
    return vortices
