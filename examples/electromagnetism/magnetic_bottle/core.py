"""The magnetic bottle: charged particles trapped between two coils, bouncing back and forth along the
field, in the spacetime algebra.

A charged particle in a magnetic field circles around the field's lines while it drifts along them. The
magnetic force does no work, so the particle's speed never changes; and where the field changes slowly
on the scale of the circles, the circling keeps its magnetic moment, the momentum of the circling
squared over the field. Moving into a stronger field, where the field's lines crowd together near a
coil, the particle must circle faster, which takes speed from its drift, until the drift stops and
reverses: the coil is a magnetic mirror, and two of them make a bottle. Particles aimed too closely
along the field never run out of drift before the field stops growing, and escape through the ends:
the loss cone.

Events and velocities are modelled as antivectors, directions in spacetime, and the magnetic field as a
plane of space, a bivector, extrinsic, built from the coils by Biot–Savart's law: each piece of wire
contributes the plane it spans with the line to the event, `piece.commutator(separation)`, over the cube
of the distance. The field turns the particle's velocity at the rate `velocity.commutator(field)`, and the
Lorentz force is the field paired with the velocity, `velocity & field`. A step in the particle's own time
is the rotor of that turn, so the speed is kept exactly however long the step; the magnetic moment is the
force squared over twice the field cubed.

In the tensor formulation of electrodynamics the equation of motion reads as
$\\frac{du^\\mu}{d\\tau} = \\frac{q}{m} F^{\\mu}{}_{\\nu} u^\\nu$, the field of the coils as
$\\mathbf B = \\frac{\\mu_0 I}{4\\pi} \\oint \\frac{d\\mathbf l \\times \\mathbf r}{r^3}$, and the adiabatic
invariant as $\\frac{p_\\perp^2}{B}$, which keeps the particle out of the mirror where
$\\sin^2\\alpha < \\frac{B_{\\min}}{B_{\\max}}$ for the angle $\\alpha$ between its velocity and the field.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                  # Scalar+
Bivector = ga.gatype.bivector()                              # Bivector+
# Events, velocities and displacements are antivectors: directions in spacetime, intrinsic.
Antivector = ga.gatype.antivector()                          # Antivector-

# One unit along the lab's time, and unit directions of space, the bottle's axis last.
time = -mv.t.dual()                                          # [] Antivector-
across, sideways, axis = stack([mv.x, mv.y, mv.z]).dual()    # [] Antivector-, Antivector-, Antivector-


# --- math -----------------------------------------------------------------------------
def coils(radius: float, heights: np.ndarray, n_pieces: int) -> tuple[Antivector, Antivector]:
    """Circular coils of `radius` around the axis at each height, as straight pieces of wire: each
    piece's displacement along the current, and its midpoint `[coils * pieces]`."""
    angles = np.linspace(0, 2 * np.pi, n_pieces + 1)
    rim = (mv.xy * (-angles / 2)).exp() >> (across * radius)                  # [pieces + 1] Antivector-
    wire = rim + axis * heights[:, None]                                      # [coils, pieces + 1] Antivector-
    pieces = (wire[:, 1:] - wire[:, :-1]).reshape(len(heights) * n_pieces)
    return pieces, ((wire[:, 1:] + wire[:, :-1]) / 2).reshape(len(heights) * n_pieces)


def field(events: Antivector, pieces: Antivector, midpoints: Antivector, current: float) -> Bivector:
    """The magnetic field of steady current in the wire at each event: by Biot–Savart's law, each piece
    contributes the plane it spans with the line to the event, over the cube of the distance."""
    # The event's place in the lab's space.
    position = events - (mv.t & events) * time                                # [...] Antivector-
    separation = position[..., None] - midpoints                              # [..., pieces] Antivector-
    distance = separation.squared().square_root()                             # [..., pieces] Scalar+
    # The commutator of two directions is the bivector that turns one towards the other.
    planes = pieces.commutator(separation)                                    # [..., pieces] Bivector+
    return current * (planes / (distance * distance.squared())).sum(axis=-1)  # [...] Bivector+


def step(event: Antivector, velocity: Antivector, pieces: Antivector, midpoints: Antivector, current: float,
         proper_step: float) -> tuple[Antivector, Antivector]:
    """One step in the particles' own time, of unit charge over mass: half a step along the velocity,
    the velocity turned by the field there, and the other half step."""
    halfway = event + velocity * (proper_step / 2)                            # [...] Antivector-
    # The velocity turns at the rate `velocity.commutator(field)`: the rotor of that turn over the step.
    turn = (field(halfway, pieces, midpoints, current) * (proper_step / 2)).exp()    # [...] Rotor+
    velocity = turn >> velocity                                               # [...] Antivector-
    return halfway + velocity * (proper_step / 2), velocity


def evolve(event: Antivector, velocity: Antivector, pieces: Antivector, midpoints: Antivector, current: float,
           proper_step: float) -> Iterator[tuple[Antivector, Antivector]]:
    """The particles' events and velocities at every step, without end."""
    while True:
        yield event, velocity
        event, velocity = step(event, velocity, pieces, midpoints, current, proper_step)


def moment(velocity: Antivector, magnetic: Bivector) -> Scalar:
    """The magnetic moment of the particles' circling, per unit charge over mass: the Lorentz force
    squared over twice the field cubed, the momentum of the circling squared over twice the field."""
    force = velocity & magnetic                                               # [...] Vector+
    strength = magnetic.scalar_norm_squared().square_root()                   # [...] Scalar+
    # The force of a magnetic field is square to the lab's time, and squares to its size squared.
    return force.squared() / (2 * strength * strength.squared())              # [...] Scalar+
