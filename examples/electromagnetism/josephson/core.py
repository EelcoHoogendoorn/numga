"""Fluxons in a planar Josephson junction: the phase difference across the junction's barrier as a direction in a plane
of its own, joined to the junction's spacetime, whose lines of magnetic flux travel through it as relativistic strings:
they contract as they speed up, reflect from the junction's edges as their opposites, and collapse under their own
tension.

Two superconductors joined across a thin barrier share their condensates' phases: a supercurrent tunnels across by the
sine of the phase difference, and the phase difference turns at the rate of the voltage across, Josephson's two
relations. Across a wide junction the phase difference also changes with the magnetic field in the barrier, and with
Maxwell's equations the three make the sine-Gordon equation, whose waves travel at most at the junction's own light
speed, the Swihart velocity. Its kinks, lines where the phase turns once around, each carry one flux quantum per unit
of their length.

The algebra is the junction's spacetime, x+y+t-, joined with a plane for the phase, a+b+: `Algebra("x+y+t-") *
Algebra("a+b+")`. The phase difference at each place is modelled as a direction in the phase plane, `mv.a` turned by
the phase, `((mv.ab) * (-phase / 2)).exp() >> mv.a`, and its rate of change as a bivector in that plane. A direction is
pulled towards the bias direction `mv.a` by the tunnelling and towards its neighbours by the field's tension, and the
torque on it is its wedge with the pull, `phases ^ pull`: the sine of the phase difference and the curvature of the
phase in one product. The phase's gradient lies in the mixed planes `xa xb ya yb`, its norm the field's strength. The
phase plane commutes with the junction's spacetime, so boosts and turns of the junction's plane, rotors in its
spacetime, carry phase directions along unchanged: a moving fluxon is a resting one seen from a boosted frame, and an
oblique one a straight one turned.

Lengths are in Josephson penetration depths and times in inverse plasma frequencies, so that the Swihart velocity is
one.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

# The junction's spacetime and the plane of the phase, which no measure of length or time reaches.
ga = Algebra("x+y+t-") * Algebra("a+b+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Plane = ga.gatype.vector()                                                    # Vector+
Point = ga.gatype.antivector()                                                # Antivector-
Phase = Plane.derive.within(Algebra("a+b+"))                                  # Vector+
Rate = ga.gatype.bivector().derive.within(Algebra("a+b+"))                    # Bivector+
# A turn of the phase plane, which takes the bias direction to a phase.
Turn = (mv.ab * 0.0).exp().gatype                                             # Rotor+
# Unit steps along the junction's plane and a unit of time, as events from the origin.
axis = mv.x.dual()                                                            # [] Point-
across = mv.y.dual()                                                          # [] Point-
time = -mv.t.dual()                                                           # [] Point-
# A gradient of the phase read as the supercurrent it drives along the junction, one in these units.
supercurrent = Plane.dual()                                                   # [] Point- <- Plane+


# --- plumbing -------------------------------------------------------------------------
def junction(n_columns: int, n_rows: int, width: float, height: float) -> tuple[Topology, np.ndarray]:
    """A junction `width` across and `height` up around its middle as a grid of triangles: its topology, and its
    places, across and up `[places, 2]`."""
    across_places, up_places = np.meshgrid(np.linspace(-width / 2, width / 2, n_columns),
                                           np.linspace(-height / 2, height / 2, n_rows))
    return Topology.grid(n_rows, n_columns), np.stack([across_places.ravel(), up_places.ravel()], axis=-1)


def assembled(topology: Topology, local: Scalar) -> SparseExtensor:
    """A form on each triangle's corners `[corners, corners] Scalar+[faces]` gathered onto the vertices it joins,
    `Scalar+[places <- places]`."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- places]
    return (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=-1).sum(axis=-1)


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, vertices: Point) -> tuple[Scalar, Plane, Scalar]:
    """Each triangle's area `Scalar+[faces]`, each corner's hat gradient `[corners] Plane+[faces]`, the facing side
    met with the triangle's patch, and each place's share of the area around it `Scalar+[places]`."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)[..., None]  # [corners] Point-[faces]
    patches = (sides[..., 0] & sides[..., 1]) / 2                             # [] Antibivector-[faces]
    areas = patches.squared().square_root()                                   # Scalar+[faces]
    gradients = (sides | patches[..., None].inverse()) / 2                    # [corners] Plane+[faces]
    return areas, gradients, (~topology.incidence[2, 0] * (areas / 3)).sum(axis=-1)


def fluxon(events: Point, rapidity: np.ndarray, sense: float) -> Turn:
    """The turns `[...] Turn+[places]` that take the bias direction to the phase of a fluxon line along y, or an
    antifluxon's for a negative sense, at these events `Point-[places]`, moving along x at each rapidity `[...]`: the
    resting kink, a turn by `4 * arctan(exp(x))` of its distance along x, read at each event from a frame boosted by
    the rapidity."""
    # The resting frame's events: the events boosted back by the rapidity, read by the plane x.
    resting = mv.x & ((mv.xt * (-rapidity / 2)).exp() >> events)            # [...] Scalar+[places]
    # The kink turns by four times the arctangent of the exponential of that distance.
    angles = 4 * (sense * resting).exp().arctan()                             # [...] Scalar+[places]
    return (mv.ab * (-angles / 2)).exp()


def evolving(tension: SparseExtensor, tunnelling: Scalar, phases: Phase, rates: Rate, dt: float, n_steps: int,
             stride: int) -> Iterator[Phase]:
    """The phase directions `[...] Phase+[places]` every `stride` steps of `dt`, from these and these rates
    `[...] Rate+[places]`: each pulled towards the bias direction by the tunnelling `Scalar+[places]` and towards its
    neighbours by the field's tension `Scalar+[places <- places]`, turned by the torque of the pull, its wedge with
    it."""
    for step in range(n_steps + 1):
        if step % stride == 0:
            yield phases
        pull = tunnelling * mv.a - tension * phases                           # [...] Phase+[places]
        rates = rates + dt * (phases ^ pull)                                  # [...] Rate+[places]
        phases = (rates * (-dt / 2)).exp() >> phases                          # [...] Phase+[places]
