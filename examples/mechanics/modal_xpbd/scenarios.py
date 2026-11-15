"""Scene builders and simulation drivers for modal extended position-based dynamics: a spliced
cantilever, a hinged chain swinging under gravity, and a spliced beam clamped at both ends and
compressed until it buckles. Along the chain, link compliances span from an exactly rigid first link
(`0` compliance) at the pin to successively softer links toward the tip within a single solver."""

from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from numga import Extensor, stack
from numga.algebras import PGA2D
from numga.sparse import Sparse
from numga.topology import Topology
from examples import instantiate

# Chain girder geometry, bar stiffness, density, and retained vibration modes.
n_bays, length, height = 4, 1.0, 0.2
stiffness, density, n_modes = 300.0, 1.0, 8
# Moving links in the chain, modal damping ratio, and downward gravity magnitude.
n_links = 4
damping = np.array([0.02])                                                 # [1]
gravity_strength = 4.0
# Relative modal compliance per moving link: zero at the pin, then four times softer per link.
flexibility = np.concatenate([[0.0], 4.0 ** np.arange(n_links - 1)])[None]  # [1, n_links]
# Tiny joint compliance keeping twin splice constraints nonsingular at machine precision.
splice = 1e-14
# Time step, substeps per frame, frame count, and GIF frame delay for the swinging chain.
dt, n_substeps, n_frames = 0.02, 2, 150
frame_duration_ms = 40
# Buckling beam parameters: moving girders, bays of unit length, height, bar stiffness, modal
# damping ratio, peak end displacement, frame count, and substeps per frame.
n_beam_girders, n_beam_bays, beam_height = 8, 32, 1.0
beam_stiffness, beam_damping = 1e11, np.array([3.0])
end_displacement, n_beam_frames, n_beam_substeps, beam_dt = 0.15, 90, 4, 0.02


# --- plumbing -------------------------------------------------------------------------
@dataclass(frozen=True)
class Shape:
    """Rest vertices, bar connectivity, inertia, and retained vibration modes of one girder."""
    # Centred on the girder's centre of mass.
    rest: Extensor                                                         # Antivector-[V]
    modes: Extensor                                                        # [n_modes] Antivector-[V]
    # Inverse squared angular frequencies of the retained modes.
    compliance: Extensor                                                   # [n_modes] Scalar+
    masses: Extensor                                                       # Scalar+[V]
    edges: np.ndarray                                                      # [E, 2]
    inertia: Extensor                                                      # [] Antibivector- <- Bivector+


@dataclass(frozen=True)
class Bodies:
    """Pose motors, body-frame twists, modal states, and inertias of `n_bodies` girders."""
    motor: Extensor                                                        # [...] Rotor+[n_bodies]
    # Body-frame spatial velocity bivector.
    rate: Extensor                                                         # [...] Bivector+[n_bodies]
    amplitudes: Extensor                                                   # [..., n_modes] Scalar+[n_bodies]
    rates: Extensor                                                        # [..., n_modes] Scalar+[n_bodies]
    # Zero on an exactly rigid body.
    compliance: Extensor                                                   # [..., n_modes] Scalar+[n_bodies]
    # Modal damping ratios.
    damping: Extensor                                                      # [..., n_modes] Scalar+[n_bodies]
    masses: Extensor                                                       # [...] Scalar+[n_bodies]
    inertia: Extensor                                                      # [...] (Antibivector- <- Bivector+)[n_bodies]
    # Zero on a fixed support body.
    inverse_inertia: Extensor                                              # [...] (Bivector+ <- Antibivector-)[n_bodies]


@dataclass(frozen=True)
class Constraints:
    """Point constraints joining pairs of anchor points across `n_bodies` girders."""
    # Incidence selecting the two bodies linked by each constraint.
    ends: Sparse                                                           # [2] Scalar+[n_constraints <- n_bodies]
    anchors: Extensor                                                      # [2] Antivector-[n_constraints]
    modes: Extensor                                                        # [n_modes, 2] Antivector-[n_constraints]
    compliance: Extensor                                                   # Scalar+[n_constraints]


def truss(mv, n_bays: int, length: float, height: float) -> tuple[Extensor, np.ndarray]:
    """Rest vertices (`Antivector-[V]`) and bar edge indices (`[E, 2]`) form a cross-braced girder
    of `n_bays` bays."""
    stations = np.arange(n_bays + 1)                                       # [n_bays + 1]
    levels = np.array([-0.5, 0.5])                                         # [2]
    # Offset the origin along x to each station and along y to the lower and upper chords.
    positions = (mv.w.dual() + mv.x.dual() * (stations[:, None] * length / n_bays) + mv.y.dual() * (levels[None, :] * height)).reshape(-1).field()  # Antivector-[V]
    lower = 2 * stations[:-1]                                              # [n_bays]
    # Longitudinal chords, crossed diagonals, and one upright post at every station.
    edges = np.concatenate([
        np.stack([lower, lower + 2], axis=-1),
        np.stack([lower + 1, lower + 3], axis=-1),
        np.stack([lower, lower + 3], axis=-1),
        np.stack([lower + 1, lower + 2], axis=-1),
        np.stack([2 * stations, 2 * stations + 1], axis=-1),
    ])                                                                     # [E, 2]
    return positions, edges


def girders(mv, shape: Shape, fixed: np.ndarray, damping: np.ndarray, flexibility: np.ndarray) -> Bodies:
    """End-to-end girders placed along `x` combine fixed supports and moving bodies scaled by
    `flexibility` (`[n_cases, n_bodies]`)."""
    n_bodies, n_modes, n_cases = len(fixed), shape.modes.shape[0], len(damping)
    motor_type = mv.context.algebra.gatype.rotor()
    rest = shape.rest.batch()                                              # [V] Antivector-
    span = mv.x & (rest[-1] - rest[1])                                     # [] Scalar+
    motor = (mv.xw * ((np.arange(n_bodies) - 0.5) * span) * 0.5).exp().cast(motor_type).broadcast_to((n_cases, n_bodies)).field()  # [n_cases] Rotor+[n_bodies]
    amplitudes = mv.scalar(np.zeros((n_cases, n_modes, n_bodies, 1))).field()  # [n_cases, n_modes] Scalar+[n_bodies]
    moving = ~fixed                                                        # [n_bodies]
    return Bodies(
        motor=motor,
        rate=mv.bivector(np.zeros((n_cases, n_bodies, 3))).field(),
        amplitudes=amplitudes,
        rates=amplitudes,
        compliance=(shape.compliance[:, None] * flexibility[:, None, :] * moving).field(),
        damping=mv.scalar(damping[:, None, None, None]).broadcast_to((n_cases, n_modes, n_bodies)).field(),
        masses=shape.masses.sites.sum().broadcast_to((n_cases, n_bodies)).field(),
        inertia=shape.inertia.broadcast_to((n_cases, n_bodies)).field(),
        inverse_inertia=(shape.inertia.inverse() * moving).broadcast_to((n_cases, n_bodies)).field(),
    )


def point_constraints(mv, shape: Shape, n_bodies: int, body_idx: np.ndarray, corner_idx: np.ndarray) -> Constraints:
    """Anchor vertex selections `corner_idx` (`[2, n_constraints]`) and body incidences `body_idx`
    (`[2, n_constraints]`) form point constraints across `n_bodies` girders."""
    V = len(shape.rest.batch())
    compliance = mv.scalar([splice]).broadcast_to(body_idx.shape[-1:]).field()  # Scalar+[n_constraints]
    ends = Topology(n_bodies, body_idx.T).incidence[1, 0]                  # [2] Scalar+[n_constraints <- n_bodies]
    corners = Sparse.selection(corner_idx, V)                              # [2] Scalar+[n_constraints <- V]
    return Constraints(ends, corners * shape.rest, corners * shape.modes[:, None], compliance)


def splices(mv, shape: Shape, n_bodies: int) -> Constraints:
    """Twin upper- and lower-chord point constraints join each adjacent pair among `n_bodies`
    girders rigidly."""
    V = shape.rest.batch().shape[-1]
    joints = np.arange(n_bodies - 1)                                       # [n_bodies - 1]
    body_idx = np.repeat(np.stack([joints, joints + 1]), 2, axis=-1)       # [2, 2 * n_bodies - 2]
    corner_idx = np.tile([[V - 2, V - 1], [0, 1]], (1, n_bodies - 1))      # [2, 2 * n_bodies - 2]
    return point_constraints(mv, shape, n_bodies, body_idx, corner_idx)


def hinges(mv, shape: Shape, n_bodies: int) -> Constraints:
    """Single upper-chord point constraints pin each adjacent pair among `n_bodies` girders."""
    V = shape.rest.batch().shape[-1]
    joints = np.arange(n_bodies - 1)                                       # [n_bodies - 1]
    return point_constraints(mv, shape, n_bodies, np.stack([joints, joints + 1]), np.tile([[V - 1], [1]], (1, n_bodies - 1)))


def cantilever(mv, shape: Shape, damping: np.ndarray, flexibility: np.ndarray) -> tuple[Bodies, Constraints]:
    """One fixed root girder followed by `n_links` spliced moving girders forms a cantilevered
    beam."""
    n_links = flexibility.shape[-1]
    fixed = np.arange(n_links + 1) == 0                                    # [n_links + 1]
    return girders(mv, shape, fixed, damping, np.pad(flexibility, ((0, 0), (1, 0)))), splices(mv, shape, n_links + 1)


def hinged_chain(mv, shape: Shape, damping: np.ndarray, flexibility: np.ndarray) -> tuple[Bodies, Constraints]:
    """One fixed root girder followed by `n_links` hinged moving girders forms a hanging chain."""
    n_links = flexibility.shape[-1]
    fixed = np.arange(n_links + 1) == 0                                    # [n_links + 1]
    return girders(mv, shape, fixed, damping, np.pad(flexibility, ((0, 0), (1, 0)))), hinges(mv, shape, n_links + 1)


def clamped_beam(mv, shape: Shape, n_moving: int, damping: np.ndarray) -> tuple[Bodies, Constraints]:
    """Spliced moving girders (`n_moving`) clamped between two fixed end girders form a slender
    column."""
    fixed = (np.arange(n_moving + 2) == 0) | (np.arange(n_moving + 2) == n_moving + 1)  # [n_moving + 2]
    return girders(mv, shape, fixed, damping, np.ones((len(damping), n_moving + 2))), splices(mv, shape, n_moving + 2)


# --- math -----------------------------------------------------------------------------
def swing_states(core, shape: Shape, bodies: Bodies, constraints: Constraints, gravity, dt: float, n_substeps: int, n_frames: int) -> Iterator[tuple]:
    """Yields `(world_points, bodies)` across `n_frames` frames under `gravity`, advancing
    `n_substeps` steps per frame."""
    for _ in range(n_frames):
        yield core.world_points(bodies, shape), bodies
        for _ in range(n_substeps):
            bodies = core.step(bodies, constraints, dt, gravity)


def end_loaded_bodies(mv, bodies: Bodies, rest_motor, displacement: float) -> Bodies:
    """Translating the right-hand fixed support inward by `displacement` along `x` updates the
    end-loaded column state."""
    motor_step = (mv.xw * (-displacement / 2)).exp()                       # [] Rotor+
    driven = rest_motor.batch()[..., -1] * motor_step                      # [...] Rotor+
    return replace(bodies, motor=bodies.motor.batch().at[..., -1].set(driven).field().normalized())


def compression_states(core, shape: Shape, bodies: Bodies, constraints: Constraints, displacements: np.ndarray, dt: float, n_substeps: int) -> Iterator[tuple]:
    """Yields `(world_points, midspans)` at each prescribed end displacement after `n_substeps`
    relaxation steps."""
    rest_motor = bodies.motor
    n_bodies = rest_motor.batch().shape[-1]
    middle = n_bodies // 2
    weightless = core.mv.direction([0, 0])                                 # [] Antivector-
    for displacement in displacements:
        bodies = end_loaded_bodies(core.mv, bodies, rest_motor, displacement)
        for _ in range(n_substeps):
            bodies = core.step(bodies, constraints, dt, weightless)
        midspan_motors = bodies.motor.batch()[..., middle - 1:middle + 1]  # [..., 2] Rotor+
        centres = midspan_motors >> core.mv.w.dual()                       # [..., 2] Antivector-
        yield core.world_points(bodies, shape), centres.sum(axis=-1) / 2   # [..., n_bodies] Antivector-[V], [...] Antivector-


def critical_displacement(height: float, span: float) -> float:
    """Euler buckling of a clamped two-chord truss of chord spacing `height` and length `span` sets
    in at end displacement `pi**2 * height**2 / span`."""
    return np.pi**2 * height**2 / span


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from . import render

    core = instantiate("examples.mechanics.modal_xpbd.core", PGA2D)
    shape = core.girder(n_bays, length, height, stiffness, density, n_modes)
    save_figure(render.mode_shapes(shape.rest, shape.rest + shape.modes[:4] * 0.03, shape.edges), "modal_xpbd_modes")

    bodies, constraints = hinged_chain(core.mv, shape, damping, flexibility)
    initial_points = core.world_points(bodies, shape)
    save_figure(render.chain_figure(initial_points, shape.edges), "modal_xpbd_chain")

    gravity = core.mv.direction([0, -gravity_strength])                    # [] Antivector-
    swing_history: list = []

    def chain_frames() -> Iterator:
        for frame_points, state in swing_states(core, shape, bodies, constraints, gravity, dt, n_substeps, n_frames):
            swing_history.append(state)
            yield frame_points

    save_animation(render.swinging_chain(shape.edges, initial_points, chain_frames()), "modal_xpbd_swing", frame_duration_ms)

    beam_shape = core.girder(n_beam_bays, float(n_beam_bays), beam_height, beam_stiffness, density, n_modes)
    beam_bodies, beam_splices = clamped_beam(core.mv, beam_shape, n_beam_girders, beam_damping)
    displacements = end_displacement * np.arange(n_beam_frames) / n_beam_frames  # [n_beam_frames]
    recorded_midspans: list = []

    def beam_frames() -> Iterator:
        for frame_points, midspan in compression_states(core, beam_shape, beam_bodies, beam_splices, displacements, beam_dt, n_beam_substeps):
            recorded_midspans.append(midspan)
            yield frame_points

    save_animation(render.beam_animation(beam_shape.edges, beam_frames()), "modal_xpbd_buckle", frame_duration_ms)
    midspans = stack(recorded_midspans)                                    # [n_beam_frames, 1] Antivector-
    critical = critical_displacement(beam_height, n_beam_girders * n_beam_bays)
    save_figure(render.buckling_figure(displacements, midspans, critical), "modal_xpbd_buckling")

    # checks
    final_bodies = swing_history[-1]
    rigid, _, _, final_gap = core.coupling(final_bodies, constraints)
    rng = np.random.default_rng(0)
    forces = core.mv.vector(rng.normal(size=(1, constraints.ends.shape[0], 3))).cast(core.Force).field()  # [1] Vector+[n_constraints]
    twists = core.mv.bivector(rng.normal(size=(1, n_links + 1, 3))).field()  # [1] Bivector+[n_links + 1]
    work_diff = (rigid.adjugate()(forces) & twists).sites.sum() - (forces & rigid(twists)).sites.sum()
    assert (work_diff.abs() < 1e-12).all()
    assert (final_gap.dual_inverse().norm() < 1e-4).all()
    deflection = (core.mv.y & midspans[:, 0]).abs()                        # [n_beam_frames] Scalar+
    assert (deflection[displacements < 0.8 * critical] < 1e-6).all()
    assert (deflection[displacements > 2 * critical] > 0.5).all()


if __name__ == "__main__":
    main()
