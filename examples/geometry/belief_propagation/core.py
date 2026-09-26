"""The most likely poses of a robot's lap, by belief propagation, in plane-based geometric algebra.

A robot drives a lap and reads, at every stop, how far it moved since the last one; each reading is a
little off. Passing a spot it saw before, it reads one more relative pose, closing the loop. Wanted
are the poses most likely given all the readings.

A reading's mismatch is a twist, the log of the motor between what it read and the relative motor of
the current poses: `(reading.inverse() * relative).log() * 2`. Its information, the inverse of its
covariance, is a map `Information = Line <- Twist`, and the most likely poses minimize the sum of
`mismatch & information(mismatch)` over the readings. The motors make this nonlinear. Linearized at
the current poses, with a small error of a pose a twist on its right, `pose * (twist * 0.5).exp()`,
it is a quadratic in the twists of all poses at once.

Belief propagation minimizes it without assembling that quadratic. Every pose holds a belief, a
Gaussian over its twist: an information and an information-weighted mean, with mean
`information.solve(weighted_mean)` and covariance `information.inverse()`. Every reading tells each
of its ends what the belief at its other end implies for it: that belief, without what the reading
told it, carried through the reading, along which covariances add. A pose's belief is the sum of what
it is told. Every round the poses move to their beliefs' means, and the readings are relinearized
there.

On a chain the beliefs are exact once what is told has crossed it. On a loop the means still settle
at the most likely poses, but the covariances do not: what goes round the loop is counted again.

In the notation of Gaussian belief propagation the information reads as the information matrix, the
weighted mean as the information vector, and what a reading tells a pose as a factor-to-variable
message.

The algebra is not fixed here. `ga` is supplied per instance, by
`examples.instantiate("examples.geometry.belief_propagation.core", PGA2D)` for a lap in the plane or
PGA3D for one in space, and the same module serves both.
"""

from collections.abc import Generator
from dataclasses import dataclass

import numpy as np

from numga import Algebra, NumpyContext, concatenate, stack

# Supplied by examples.instantiate.
ga: Algebra
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
Twist = ga.gatype.bivector()
# A line reads out a twist: `line & twist`.
Line = ga.gatype.antibivector()
# A plane reads out a point: `plane & point`.
Plane = ga.gatype.vector()
Information = ga.gatype((Line, Twist))            # Line <- Twist
Covariance = ga.gatype((Twist, Line))             # Twist <- Line
# A quadric: the points where its pairing with a point vanishes.
Quadric = ga.gatype((Plane, Point))               # Plane <- Point


@dataclass(frozen=True)
class PoseGraph:
    """Relative pose readings between pairs of poses, and a prior on every pose."""

    ends: np.ndarray          # [2, readings] the pose at the head of each reading, then the one at its tail
    readings: Motor           # [readings] Motor, the head as read from the tail
    information: Information  # [readings] Information, in the frame of the head
    anchors: Motor            # [poses] Motor, where each pose is believed to be before any reading
    priors: Information       # [poses] Information, how firmly

    @property
    def incidence(self) -> np.ndarray:
        """[2, readings, poses] one where an end of a reading sits at a pose."""
        return (self.ends[..., None] == np.arange(len(self.anchors))).astype(float)

    def untold(self) -> "Belief":
        """What the readings tell their ends before anything has been told: nothing, at either end."""
        return Belief(self.information * np.zeros(self.ends.shape), mv(Line, np.zeros(self.ends.shape + (len(Line.output_subspace),))))

    def continued(self, told: "Belief") -> "Belief":
        """What was told along the readings of a graph that had only the first of these readings, and
        nothing yet along the readings added after them."""
        untold, known = self.untold(), told.information.shape[-1]
        return Belief(concatenate([told.information, untold.information[:, known:]], axis=1),
                      concatenate([told.weighted_mean, untold.weighted_mean[:, known:]], axis=1))


@dataclass(frozen=True)
class Belief:
    """A Gaussian over the twist of a pose, as its information and its information-weighted mean."""

    information: Information  # [...] Information
    weighted_mean: Line       # [...] Line


# --- math -----------------------------------------------------------------------------
def linearize(graph: PoseGraph, poses: Motor) -> tuple[Motor, Twist, Information, Twist]:
    """Every reading as seen from each of its ends: the near pose in the far pose's frame, the
    mismatch, and the information in the near pose's frame; and every pose's offset from its anchor."""
    relative = poses[graph.ends[::-1]].inverse() * poses[graph.ends]           # [2, readings] Motor
    readings = stack([graph.readings, graph.readings.inverse()])              # [2, readings] Motor
    mismatch = (readings.inverse() * relative).log() * 2                      # [2, readings] Twist
    # The information is given in the head's frame; the head in the tail's frame carries it there.
    head = relative[0]                                                        # [readings] Motor
    information = stack([graph.information, head >> graph.information(head << Twist)])   # [2, readings] Information
    offset = (graph.anchors.inverse() * poses).log() * 2                      # [poses] Twist
    return relative, mismatch, information, offset


def beliefs(graph: PoseGraph, told: Belief, weighted_mean: Line) -> Belief:
    """Every pose's belief: its prior, with the given weighted mean, plus all it is told."""
    information = graph.priors + (told.information[..., None] * graph.incidence).sum(axis=(-3, -2))   # [poses] Information
    weighted_mean = weighted_mean + (told.weighted_mean[..., None] * graph.incidence).sum(axis=(-3, -2))   # [..., poses] Line
    return Belief(information, weighted_mean)


def tell(graph: PoseGraph, relative: Motor, mismatch: Twist, information: Information, belief: Belief,
         told: Belief) -> Belief:
    """What every reading tells each of its ends: the belief at its other end, without what the reading
    told that end, carried through the reading."""
    far = graph.ends[::-1]
    rest = belief.information[far] - told.information[::-1]                   # [2, readings] Information
    mean = rest.solve(belief.weighted_mean[..., far] - told.weighted_mean[..., ::-1, :])   # [..., 2, readings] Twist
    # Carried into the near pose's frame by the relative motor.
    covariance = relative << rest.solve(relative >> Line)                 # [2, readings] Covariance
    # Through the reading, the near pose sits its mismatch behind that mean, and the reading's own
    # covariance adds to the carried one.
    implied = (information.inverse() + covariance).inverse()                  # [2, readings] Information
    return Belief(implied, implied((relative << mean) - mismatch))


def propagate(graph: PoseGraph, poses: Motor, told: Belief,
              rounds: int) -> Generator[tuple[Motor, Information], None, tuple[Motor, Belief]]:
    """Belief propagation relinearized every round, from what the readings have told so far: every
    reading tells its ends what it implies, and every pose moves to its belief's mean. Yields the poses
    and their beliefs' information; returns the poses and what is told after the last round."""
    for _ in range(rounds):
        relative, mismatch, information, offset = linearize(graph, poses)
        told = tell(graph, relative, mismatch, information, beliefs(graph, told, -graph.priors(offset)), told)
        belief = beliefs(graph, told, -graph.priors(offset))
        mean = belief.information.solve(belief.weighted_mean)                  # [poses] Twist
        poses = poses * (mean * 0.5).exp()
        # What a pose was told moves with it: its mean is now the move closer.
        told = Belief(told.information, told.weighted_mean - told.information(mean[graph.ends]))
        yield poses, belief.information
    return poses, told


def gradient(graph: PoseGraph, poses: Motor) -> Line:
    """The gradient, with respect to the twist of each pose, of half the sum of the mismatches weighted
    by their information, and of the priors': zero at the most likely poses."""
    _, mismatch, information, offset = linearize(graph, poses)
    return (information(mismatch)[..., None] * graph.incidence).sum(axis=(-3, -2)) + graph.priors(offset)   # [poses] Line


def exact_covariance(graph: PoseGraph, poses: Motor, lines: Line, rounds: int) -> Twist:
    """Each pose's exact covariance applied to each line: how far the pose's most likely twist moves
    when its own weighted mean alone is nudged by the line, with every mismatch zero. Belief
    propagation gets means right on loops too, so this holds there as well."""
    relative, mismatch, information, _ = linearize(graph, poses)
    nudged = np.arange(len(graph.anchors))
    # One pose nudged at a time, by each line in turn.
    nudges = lines[:, None, None] * np.eye(len(nudged))                       # [lines, nudged, poses] Line
    told = graph.untold()
    for _ in range(rounds):
        told = tell(graph, relative, 0 * mismatch, information, beliefs(graph, told, nudges), told)
    belief = beliefs(graph, told, nudges)
    moved = belief.information.solve(belief.weighted_mean)                    # [lines, nudged, poses] Twist
    # How far each nudged pose itself moves.
    return moved[:, nudged, nudged]                                           # [lines, poses] Twist


def position_quadric(poses: Motor, covariance: Covariance, origin: Point, sigmas: float) -> Quadric:
    """The quadric `sigmas` standard deviations out, within which each pose carries the origin: an
    ellipse in the plane, an ellipsoid in space.

    A twist moves the carried point by its commutator with the point. Reading that motion with a plane
    is reading the twist with a line, found by solving the incidence form; so the covariance of the
    twist gives the covariance of the point's motion. Added to the point's dyad it is the point's
    second moment. Its inverse sends the carried point to a plane, its polar; paired twice with any
    point, the inverse is that point's pairing with the polar, squared, times one plus the squared
    number of standard deviations to it. Less that many dyads of the polar, it vanishes on the quadric.
    """
    here = poses >> origin                                                    # [...] Point
    shift = Twist.commutator(here)(poses >> Twist)                            # [...] Point <- Twist
    readout = (Line & Twist).solve(Plane & shift)                             # [...] Line <- Plane
    moment = here * (Plane & here) + shift(covariance(readout))               # [...] Point <- Plane
    inverse = moment.inverse()                                                # [...] Plane <- Point
    polar = inverse(here)                                                     # [...] Plane
    return inverse - (1 + sigmas**2) * polar * (polar & Point) / (polar & here)   # [...] Plane <- Point


# --- plumbing -------------------------------------------------------------------------
def information(translation_std: float, rotation_std: float) -> Information:
    """The information of a reading with isotropic translation noise and rotation noise about the head:
    the inverse of its covariance."""
    twists = mv(Twist, np.eye(len(Twist.output_subspace)))                     # [twists] Twist
    # A basis twist turns if its reverse product is one, and slides if it is zero.
    turning = twists.scalar_norm_squared()                                     # [twists] Scalar
    variances = translation_std**2 + (rotation_std**2 - translation_std**2) * turning   # [twists] Scalar
    return (twists * (twists & Line) * variances).sum(axis=0).inverse()        # [] Line <- Twist
