"""Static shapes and rigid-segment dynamics of elastic rods in three-dimensional projective geometric algebra.

A rod is modelled as a chain of rigid cross-sections placed by motors. Under loads applied only at the ends,
every cross-section carries the same spatial forque, pulls that load into its own frame, and turns per unit
length through its compliance before advancing to the next cross-section by a motor exponential.

In motion, rigid segments with inertia connect across elastic joints whose rest step encodes the natural coil.
Each joint maps its logarithmic twist deviation to a restoring forque through the right-trivialized exponential
derivative, and each interior segment steps its momentum and placement under joint loads, gravity, and damping.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import PGA3D as ga
from numga.backend.jax import JaxContext

context = NumpyContext(ga)
mv = context.multivector
Point = ga.gatype.antivector()                             # Antivector-
Twist = ga.gatype.bivector()                               # Bivector+
Forque = ga.gatype.antibivector()                          # Antibivector-
Motor = ga.gatype.rotor()                                  # Rotor+
Compliance = ga.gatype((Twist, Forque))                    # Twist+ <- Forque-
Stiffness = ga.gatype((Forque, Twist))                     # Forque- <- Twist+
Inertia = ga.gatype((Forque, Twist))                       # Forque- <- Twist+

# Cross-section origin, unit axial shift, and local rotation and force axes (across, up, along).
centre = mv.w.dual()                                                                   # [] Point-
shift_along_rod = mv.zw                                                                # [] Twist+
turn_axes = stack([mv.yz, mv.zx, mv.xy])                                               # [3] Twist+
force_axes = stack([centre & mv.x.dual(), centre & mv.y.dual(), centre & mv.z.dual()])  # [3] Forque-


# --- math -----------------------------------------------------------------------------
def compliance(turning_compliance: np.ndarray, shifting_compliance: np.ndarray) -> Compliance:
    """A dyad sum over the three cross-section axes (`[3]`) maps a carried load to a twist per unit length (`[] Twist+ <- Forque-`)."""
    shift_axes = force_axes.dual()                                                      # [3] Twist+
    turn_terms = turn_axes * (turn_axes & Forque) * turning_compliance                  # [3] Twist+ <- Forque-
    shift_terms = shift_axes * (shift_axes & Forque) * shifting_compliance              # [3] Twist+ <- Forque-
    return (turn_terms + shift_terms).sum(axis=0)                                       # [] Twist+ <- Forque-


def rod_frames(loads: Forque, rod_compliance: Compliance, start_frames: Motor, length: float, n_pieces: int) -> Motor:
    """Midpoint integration across `n_pieces` pieces integrates the carried load into cross-section placements (`[..., n_pieces + 1] Motor+`)."""
    piece_length = length / n_pieces
    frame = start_frames.broadcast_to(np.broadcast_shapes(start_frames.shape, loads.shape))  # [...] Motor+
    frames = [frame]
    for _ in range(n_pieces):
        # Each piece advances along the rod and turns by the load seen halfway across the piece.
        halfway_step = ((shift_along_rod + rod_compliance(frame << loads)) * (piece_length / 4)).exp()  # [...] Motor+
        halfway = frame * halfway_step                                                                  # [...] Motor+
        full_step = ((shift_along_rod + rod_compliance(halfway << loads)) * (piece_length / 2)).exp()   # [...] Motor+
        frame = frame * full_step                                                                       # [...] Motor+
        frames.append(frame)
    return stack(frames, axis=-1)                                                       # [..., n_pieces + 1] Motor+


def stiffness(turning_rigidity: np.ndarray, shifting_rigidity: np.ndarray) -> Stiffness:
    """A dyad sum over the cross-section couple and force axes (`[3]`) maps a twist per unit length to a restoring load (`[] Forque- <- Twist+`)."""
    couple_axes = turn_axes.dual()                                                      # [3] Forque-
    turn_terms = couple_axes * (Twist & couple_axes) * turning_rigidity                 # [3] Forque- <- Twist+
    shift_terms = force_axes * (Twist & force_axes) * shifting_rigidity                 # [3] Forque- <- Twist+
    return (turn_terms + shift_terms).sum(axis=0)                                       # [] Forque- <- Twist+


def segment_inertia(mass: float, length: float, radius: float) -> Inertia:
    """Six point masses across a segment's ends and mid-ring (`[6]`) define its rigid-body inertia (`[] Forque- <- Twist+`)."""
    points = mv.point([
        [radius, 0, 0], [-radius, 0, 0], [0, radius, 0], [0, -radius, 0],
        [0, 0, length / 2], [0, 0, -length / 2],
    ])                                                                                  # [6] Point-
    return ((mass / 6) * points).regressive(points.commutator(Twist)).sum(axis=0)       # [] Forque- <- Twist+


def joint_loads(frames: Motor, rest_step: Motor, joint_stiffness: Stiffness, piece_length: float) -> Forque:
    """Each joint maps its relative twist deviation from `rest_step` to the world-frame load on the following segment (`[...] Forque-`)."""
    deviation = (rest_step.reverse() * frames[..., :-1].reverse() * frames[..., 1:]).log() * 2  # [...] Twist+
    local_load = joint_stiffness(deviation / piece_length)                                      # [...] Forque-
    # The right-trivialized exponential derivative keeps the joint load equal to the energy gradient.
    restoring_load = (
        local_load
        - deviation.commutator(local_load) / 2
        + deviation.commutator(deviation.commutator(local_load)) / 12
    )                                                                                           # [...] Forque-
    return frames[..., 1:] >> restoring_load                                                    # [...] Forque-


def cord_step(
    frames: Motor,
    rates: Twist,
    rest_step: Motor,
    joint_stiffness: Stiffness,
    piece_length: float,
    inertia: Inertia,
    weight: Point,
    damping: float,
    time_step: float,
) -> tuple[Motor, Twist]:
    """Euler–Poincaré integration advances the interior segments and body rates by `time_step` between held end frames."""
    loads = joint_loads(frames, rest_step, joint_stiffness, piece_length)               # [...] Forque-
    segments = frames[1:-1]                                                             # [...] Motor+
    momenta = inertia(rates)                                                            # [...] Forque-
    # Gravity acts along the vertical line through each segment's centre, while damping opposes its momentum.
    forques = (segments << (loads[:-1] - loads[1:])) + (centre & (segments << weight)) - damping * momenta  # [...] Forque-
    next_rates = rates + inertia.solve(forques - momenta.commutator(rates)) * time_step                     # [...] Twist+
    step_motors = (next_rates * (-time_step / 2)).exp()                                                     # [...] Motor+
    next_segments = segments * step_motors                                                                  # [...] Motor+
    return concatenate([frames[:1], next_segments, frames[-1:]]), next_rates


def cord_trajectory(
    rest_frames: Motor,
    rest_step: Motor,
    coil_shift: Twist,
    joint_stiffness: Stiffness,
    piece_length: float,
    inertia: Inertia,
    weight: Point,
    damping: float,
    time_step: float,
    reaches: np.ndarray,
) -> Iterator[Motor]:
    """Yields the segment frames (`[...] Motor+`) at each frame of the handset displacement schedule `reaches`."""
    n_segments = len(rest_frames) - 2
    jax_context = JaxContext(ga)

    def scan_step(state: tuple[Motor, Twist], reach: jax.Array) -> tuple[tuple[Motor, Twist], None]:
        frames, rates = state
        # Shifting the handset along the coil axis updates the far boundary before stepping the interior segments.
        handset_shift = (coil_shift * (reach / 2)).exp()                                # [] Motor+
        handset_frame = handset_shift * rest_frames[-1]                                 # [] Motor+
        next_frames = concatenate([frames[:-1], handset_frame[None]])                   # [...] Motor+
        next_state = cord_step(
            next_frames, rates, rest_step, joint_stiffness, piece_length,
            inertia, weight, damping, time_step,
        )
        return next_state, None

    advance_frame = jax.jit(lambda state, frame_reaches: jax.lax.scan(scan_step, state, frame_reaches)[0])
    initial_rates = (0 * shift_along_rod).broadcast_to((n_segments,)).retyped(Twist)    # [n_segments] Twist+
    state = (
        rest_frames.retyped(Motor).materialize(jax_context),
        initial_rates.materialize(jax_context),
    )
    for frame_reaches in reaches:
        yield state[0].materialize(context)
        state = advance_frame(state, frame_reaches)


def dense_frames(frames: Motor, n_subdivisions: int) -> Motor:
    """Logarithmic relative-twist interpolation inserts `n_subdivisions` equal steps across each joint (`[...] Motor+`)."""
    relative_twists = (frames[..., :-1].reverse() * frames[..., 1:]).log()              # [...] Twist+
    shares = np.arange(n_subdivisions) / n_subdivisions                                 # [n_subdivisions]
    step_motors = (relative_twists[..., None] * shares).exp()                           # [..., n_subdivisions] Motor+
    parts = frames[..., :-1, None] * step_motors                                        # [..., n_subdivisions] Motor+
    return concatenate([parts.reshape(parts.shape[:-2] + (-1,)), frames[..., -1:]], axis=-1)  # [...] Motor+
