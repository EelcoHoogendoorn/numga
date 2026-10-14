"""This example computes the shapes of elastic rods loaded only at their ends, the elastica of Euler and
Kirchhoff, in the projective geometric algebra of three-dimensional space.

A rod is a chain of cross-sections, each a rigid slice of the rod with its own placement, a motor. With loads
only at the ends, the part of the rod beyond any cross-section is held in balance by the end load and by the
cross-section alone, so every cross-section carries the same load: one line, fixed in space, a force along it
or a couple. Seen from a cross-section, `frame << load`, the load's moments about the section's own axes bend
it, across and up, and twist it, along the rod, each in proportion over the rod's rigidity against that turn.
A cross-section moves to the next by a shift along the rod and that turn, a line read as a motion; its
exponential over a short piece is the motor from one cross-section to the next, and their running product
places the whole rod.

In motion, the rod is a chain of short rigid segments with mass, each joined to the next by an elastic joint
whose rest placement is the rod's own shape. A joint's deviation from rest is a line; its energy is a quadratic
form of that line, and the load it puts on its two segments is the derivative of that energy, the stiffness
applied to the deviation through the derivative of the exponential. Each segment moves under the loads of its
two joints, its weight and some damping, by Euler's equations for a rigid body: its momentum, the inertia
applied to its rate, changes by the forque on it less the gyroscopic turn of the momentum by the rate.

In the notation of Kirchhoff's rod theory the cross-sections' turn per unit length is
$\\boldsymbol{\\kappa} = B^{-1}\\mathbf{m}$, with $B$ the rigidities and $\\mathbf{m}$ the moment carried, and the
frames solve $\\frac{d}{ds}(R, \\mathbf{r}) = (R\\,[\\boldsymbol{\\kappa}]_\\times, R\\,\\mathbf{e}_3)$.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import PGA3D as ga

context = NumpyContext(ga)
mv = context.multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Line = ga.gatype.bivector()
Motor = ga.gatype.rotor()
# How a cross-section turns per unit length under the load it carries.
Compliance = ga.gatype((Line, Line))                       # Line <- Line
# The load a joint carries per unit of its deviation from rest, and a segment's momentum per unit of rate.
Stiffness = ga.gatype((Line, Line))                        # Line <- Line
Inertia = ga.gatype((Line, Line))                          # Line <- Line

# The centre of the cross-section at the rod's start, the shift carrying a cross-section along the unbent rod
# per unit length, and the axes of a cross-section through its centre: across, up and along the rod. A turn
# about an axis is the axis itself and the couple about it the axis's dual; a shift along an axis is the dual and
# the force along it the axis.
centre = mv.w.dual()                                       # [] Point
shift_along_rod = mv.z ^ mv.w                              # [] Line
section_axes = stack([centre & mv.x.dual(), centre & mv.y.dual(), centre & mv.z.dual()])   # [3] Line


# --- math -----------------------------------------------------------------------------
def compliance(turning: np.ndarray, shifting: np.ndarray) -> Compliance:
    """How a cross-section moves per unit length under a load: about each of its axes `[3]`, a turn by the
    load's couple about the axis times the compliance to turning, and a shift by the load's force along the
    axis times the compliance to shifting; zero to shifting for a rod that only bends and twists."""
    couples = section_axes.dual()                                       # [3] Line
    turns = section_axes * (section_axes & Line) * turning             # [3] Line <- Line
    shifts = couples * (couples & Line) * shifting                     # [3] Line <- Line
    return (turns + shifts).sum(axis=0)                                 # [] Line <- Line


def rod(loads: Line, flex: Compliance, starts: Motor, length: float, pieces: int) -> Motor:
    """The frame of each cross-section of rods of the given length `[..., pieces + 1]`, from the start
    frames along, each rod carrying its load across every cross-section."""
    step = length / pieces

    def frames() -> Iterator[Motor]:
        frame = starts.broadcast_to(np.broadcast_shapes(starts.shape, loads.shape))
        yield frame
        for _ in range(pieces):
            # The load seen from a cross-section turns it, and with the shift moves it along the rod; the
            # turn taken over each piece is the one seen from halfway along it.
            halfway = frame * ((shift_along_rod + flex(frame << loads)) * (step / 4)).exp()   # [...] Motor
            frame = frame * ((shift_along_rod + flex(halfway << loads)) * (step / 2)).exp()
            yield frame
    return stack(list(frames()), axis=-1)                               # [..., pieces + 1] Motor


def stiffness(turning: np.ndarray, shifting: np.ndarray) -> Stiffness:
    """The load a joint carries per unit length of its deviation from rest: about each of the cross-section's
    axes `[3]`, a couple for the turn about it times the rigidity against turning, and a force along it for the
    shift along it times the rigidity against shifting."""
    couples = section_axes.dual()                                       # [3] Line
    turns = couples * (couples & Line) * turning                       # [3] Line <- Line
    shifts = section_axes * (section_axes & Line) * shifting           # [3] Line <- Line
    return (turns + shifts).sum(axis=0)                                 # [] Line <- Line


def segment_inertia(mass: float, length: float, radius: float) -> Inertia:
    """A segment's inertia, from its rate to its momentum: its mass spread over its two ends and a ring of the
    given radius around its middle, each point's momentum its join with its own motion under the rate."""
    spread = points([[radius, 0, 0], [-radius, 0, 0], [0, radius, 0], [0, -radius, 0],
                     [0, 0, length / 2], [0, 0, -length / 2]])            # [6] Point
    return (spread & spread.commutator(Line)).sum(axis=0) * (mass / spread.shape[0])   # [] Line <- Line


def joint_loads(frames: Motor, rest: Motor, stiff: Stiffness, step: float) -> Line:
    """The load each joint between neighbouring frames `[..., segments]` puts on the frame after it, in the
    world: its stiffness applied to its deviation from the rest placement, through the derivative of the
    exponential, so that the load is exactly the derivative of the joint's energy."""
    deviation = (rest.reverse() * frames[..., :-1].reverse() * frames[..., 1:]).log() * 2   # [..., joints] Line
    load = stiff(deviation / step)
    load = load - deviation.commutator(load) / 2 + deviation.commutator(deviation.commutator(load)) / 12
    return frames[..., 1:] >> load                                      # [..., joints] Line


def moving(frames: Motor, rates: Line, rest: Motor, stiff: Stiffness, step: float, inertia: Inertia,
           inverse: Inertia, weight: Point, damping: float, time_step: float) -> tuple[Motor, Line]:
    """One time step of the segments between the first and last frame, which are held: each rate changes by
    the forque on its segment, the loads of its two joints, its weight and its damping, less the gyroscopic
    turn of its momentum; each frame then moves by its rate."""
    loads = joint_loads(frames, rest, stiff, step)                      # [joints] Line
    segments = frames[1:-1]                                             # [segments] Motor
    momenta = inertia(rates)                                            # [segments] Line
    # The weight is a line straight down through the segment's centre; the damping resists its rate.
    forques = (segments << (loads[:-1] - loads[1:])) + (centre & (segments << weight)) - damping * momenta
    rates = rates + inverse(forques - momenta.commutator(rates)) * time_step
    segments = segments * (rates * (-time_step / 2)).exp()
    return concatenate([frames[:1], segments, frames[-1:]]), rates


def between(frames: Motor, count: int) -> Motor:
    """Frames between neighbouring frames `[..., segments]`, each pair joined by its relative motion in
    `count` equal parts `[..., (segments - 1) * count + 1]`."""
    relative = (frames[..., :-1].reverse() * frames[..., 1:]).log()     # [..., joints] Line
    shares = np.arange(count) / count
    parts = frames[..., :-1, None] * (relative[..., None] * shares).exp()   # [..., joints, count] Motor
    return concatenate([parts.reshape(parts.shape[:-2] + (-1,)), frames[..., -1:]], axis=-1)


# --- plumbing -------------------------------------------------------------------------
def points(coordinates: np.ndarray) -> Point:
    """Points at coordinates `[..., 3]`."""
    return (mv("x y z", coordinates) + mv.w).dual()
