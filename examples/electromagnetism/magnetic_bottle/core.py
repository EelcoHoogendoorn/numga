"""Charged particles gyrate, bounce, or escape through the loss cone of a two-coil magnetic bottle in
spacetime geometric algebra.

Events, four-velocities, spatial separations, and steady wire current elements are modelled as intrinsic
antivectors, while the magnetic field is modelled as an extrinsic bivector built from the coil segments
by the Biot–Savart commutator `currents.commutator(separation)`. Pairing the field with the four-velocity
via `magnetic & velocity` yields the Lorentz four-force covector, whose spacetime dual
`velocity.commutator(magnetic)` gives the proper-time rate of turn integrated by a rotor step at exact
unit speed.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("x+y+z+t-")
mv = NumpyContext(ga).multivector

Scalar = ga.gatype.scalar()                                                   # Scalar+
Field = ga.gatype.bivector()                                                  # Bivector+
Antivector = ga.gatype.antivector()                                           # Antivector-

# Dualizing the spacetime basis planes gives laboratory time and orthonormal spatial axes.
time = -mv.t.dual()                                                           # [] Antivector-
x_axis, y_axis, z_axis = stack([mv.x, mv.y, mv.z]).dual()                     # [] Antivector-, Antivector-, Antivector-


# --- math -----------------------------------------------------------------------------
def coils(radius: float, heights: np.ndarray, current: float, n_pieces: int) -> tuple[Antivector, Antivector]:
    """Circular coils of `radius` and `current` at `heights` (`[n_coils]`) discretize into straight
    current elements and their midpoints (`[n_coils * n_pieces] Antivector-`)."""
    n_coils = len(heights)
    angles = np.linspace(0, 2 * np.pi, n_pieces + 1)
    turns = (mv.xy * (-angles / 2)).exp()                                     # [n_pieces + 1] Rotor+
    rim = turns >> (x_axis * radius)                                          # [n_pieces + 1] Antivector-
    wire = rim + z_axis * heights[:, None]                                    # [n_coils, n_pieces + 1] Antivector-
    currents = current * (wire[:, 1:] - wire[:, :-1]).reshape(n_coils * n_pieces)   # [n_coils * n_pieces] Antivector-
    midpoints = ((wire[:, 1:] + wire[:, :-1]) / 2).reshape(n_coils * n_pieces)      # [n_coils * n_pieces] Antivector-
    return currents, midpoints


def field(events: Antivector, currents: Antivector, midpoints: Antivector) -> Field:
    """Steady wire `currents` and `midpoints` produce the Biot–Savart magnetic field (`[...] Field+`)
    at `events` (`[...] Antivector-`)."""
    # Intersecting each event's laboratory worldline with the constant-time hyperplane reads its spatial position.
    position = mv.t ^ (events & time)                                         # [...] Antivector-
    separation = position[..., None] - midpoints                              # [...] Antivector-
    distance = separation.antinorm()                                          # [...] Scalar+
    # The commutator of two spatial antivectors spans their extrinsic magnetic plane.
    planes = currents.commutator(separation)                                  # [...] Field+
    return (planes / (distance * distance.squared())).sum(axis=-1)            # [...] Field+


def step(
    event: Antivector,
    velocity: Antivector,
    currents: Antivector,
    midpoints: Antivector,
    proper_step: float,
) -> tuple[Antivector, Antivector]:
    """Leapfrog proper-time integration advances `event` and turns `velocity` (`[...] Antivector-`)
    through `proper_step` at unit charge-to-mass ratio."""
    midpoint_event = event + velocity * (proper_step / 2)                     # [...] Antivector-
    magnetic = field(midpoint_event, currents, midpoints)                     # [...] Field+
    # Exponentiating half the field step integrates velocity.commutator(magnetic) at exact unit speed.
    turn = (magnetic * (proper_step / 2)).exp()                               # [...] Rotor+
    velocity = turn >> velocity                                               # [...] Antivector-
    return midpoint_event + velocity * (proper_step / 2), velocity


def evolve(
    event: Antivector,
    velocity: Antivector,
    currents: Antivector,
    midpoints: Antivector,
    proper_step: float,
    n_steps: int,
) -> Iterator[tuple[Antivector, Antivector]]:
    """Yields the particle events and four-velocities (`[...] Antivector-`) across `n_steps`
    proper-time steps."""
    for _ in range(n_steps):
        yield event, velocity
        event, velocity = step(event, velocity, currents, midpoints, proper_step)


def moment(velocity: Antivector, magnetic: Field) -> Scalar:
    """The Lorentz four-force covector `magnetic & velocity` determines the orbital magnetic moment
    (`[...] Scalar+`) at unit charge-to-mass ratio."""
    force = magnetic & velocity                                               # [...] Vector+
    strength = magnetic.scalar_norm_squared().square_root()                   # [...] Scalar+
    # For a purely magnetic field, the spatial force covector squares to perpendicular momentum squared times field squared.
    return force.squared() / (2 * strength * strength.squared())              # [...] Scalar+
