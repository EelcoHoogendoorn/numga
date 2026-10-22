"""Scenes for the XPBD chain: solid ellipsoidal links hanging from a pin, released from a horizontal arc, with
stiff joints and light damping, in gravity of 0.5, stepped with NumPy and under jax.jit."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import Context, Extensor, NumpyContext
from examples.mechanics.lie_integrators import inertia_from_points
from examples.mechanics.xpbd import core
from examples.mechanics.xpbd.core import Chain, Joints, Motor, Scalar

# The chain: its links, the spacing of their joints, the semi-axes of each link's solid ellipsoid, the turn
# between neighbours along its starting arc, the joints' compliance, the damping and gravity.
N_LINKS = 10
LINK_SPACING = 9e-2
SEMI_AXES = np.array([0.62, 0.38, 0.22]) * LINK_SPACING
CURL = 0.3
COMPLIANCE = 1e-9
DAMPING = 1e-3
GRAVITY = 0.5
# The run: its steps, the substeps per step and the time step; and the duration of a frame in milliseconds.
N_STEPS = 100
N_SUBSTEPS = 4
STEP = 0.04
FRAME_DURATION = 60


# --- math -----------------------------------------------------------------------------
def chain(
    context: Context, n_links: int, spacing: float, semi_axes: np.ndarray, curl: float,
    compliance: float, damping: float, gravity: float,
) -> tuple[Chain, list[Joints]]:
    """A chain of solid ellipsoidal links hanging from a pin at the origin, in gravity along -z.

    The pin is body 0, with zero inverse inertia, an infinite mass, so nothing moves it; the links follow.
    Each body is six unit masses on its axes, at `semi_axes * np.sqrt(3 / 5)`, which gives it the inertia of
    a solid ellipsoid with those semi-axes, three different moments. Neighbours are joined at their ends,
    half the spacing along x from their centres, and the chain starts on a horizontal arc: each body is its
    neighbour carried half the spacing along x, turned about z, and carried on. The joints are split
    red/black: even joints connect (0-1, 2-3, ...) and odd joints connect (1-2, 3-4, ...), so the bodies
    within one partition are independent and relax in parallel.
    """
    mv = context.multivector
    origin = mv.w.dual()
    n_bodies = n_links + 1

    # Mass properties of one body, from its point cloud, replicated over all bodies.
    reach = np.diag(semi_axes) * np.sqrt(3 / 5)
    cloud = mv.point(np.concatenate([reach, -reach]))
    inertia, inverse_inertia = inertia_from_points(cloud)

    # The pin has infinite mass: zero inverse inertia.
    mobility = np.concatenate([[0.0], np.ones(n_links)])

    # From one body to the next along the arc; the bodies are its powers.
    half_shift = (mv.xw * (spacing / 4)).exp()
    neighbour = half_shift * (mv.xy * (curl / 2)).exp() * half_shift                   # [] Motor
    motors = (neighbour.log() * np.arange(n_bodies)).exp().cast(Motor.output_subspace)  # [bodies] Motor

    state = Chain(
        motor=motors,
        rate=mv.bivector().broadcast_to(n_bodies),
        first_moment=cloud.sum(axis=-1)[None].broadcast_to(n_bodies),
        inertia=inertia[None].broadcast_to(n_bodies),
        inverse_inertia=inverse_inertia[None].broadcast_to(n_bodies) * mobility,
        damping=(mv.scalar() * damping).broadcast_to(n_bodies),
        gravity=mv.direction([0.0, 0.0, -gravity]).broadcast_to(n_bodies),
    )

    # Joint j connects the +x end of body j to the -x end of body j + 1.
    link = np.arange(n_links)
    pairs = np.array([link, link + 1])
    anchors = origin + mv.direction([1.0, 0.0, 0.0]) * (np.array([[+0.5], [-0.5]]) * spacing * np.ones(n_links))
    compliances = mv.scalar(np.full((n_links, 1), compliance))
    partitions = [Joints(pairs[:, half], anchors[:, half], compliances[half])
                  for half in (slice(0, None, 2), slice(1, None, 2))]
    return state, partitions


def swinging_chain(n_links: int, n_steps: int, n_substeps: int, dt: float) -> tuple[Motor, Scalar]:
    """The bodies' motors over time, and the gap at every joint, stepped with NumPy."""
    initial, partitions = chain(NumpyContext(core.ga), n_links, LINK_SPACING, SEMI_AXES, CURL, COMPLIANCE, DAMPING, GRAVITY)
    return trajectory(list(islice(states(initial, lambda state: core.advance(state, partitions, n_substeps, dt)), n_steps + 1)), partitions)


def swinging_chain_jax(n_links: int, n_steps: int, n_substeps: int, dt: float) -> tuple[Motor, Scalar]:
    """The same chain, each step compiled by jax.jit. Needs JAX installed."""
    import jax
    from numga.backend.jax import JaxContext

    initial, partitions = chain(JaxContext(core.ga), n_links, LINK_SPACING, SEMI_AXES, CURL, COMPLIANCE, DAMPING, GRAVITY)
    compiled = jax.jit(lambda state: core.advance(state, partitions, n_substeps, dt))
    # The first step runs eagerly: it settles the traits the initial state is built with, so that jit
    # traces the step once.
    first = core.advance(initial, partitions, n_substeps, dt)
    return trajectory([initial] + list(islice(states(first, compiled), n_steps)), partitions)


def states(state: Chain, advance) -> Iterator[Chain]:
    """The chain from a state on, one step at a time, without end."""
    while True:
        yield state
        state = advance(state)


def trajectory(history: list[Chain], partitions: list[Joints]) -> tuple[Motor, Scalar]:
    """Every body's motor, and every joint gap, stacked over time."""
    motors = Extensor.stack([state.motor for state in history])                        # [steps + 1, bodies] Motor
    gaps = Extensor.stack([core.joint_gaps(state.motor, partitions) for state in history])   # [steps + 1, joints] Scalar
    return motors, gaps


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.mechanics.xpbd import render

    motors, gaps = swinging_chain(N_LINKS, N_STEPS, N_SUBSTEPS, STEP)
    save_animation(render.animate_chain(motors, SEMI_AXES, LINK_SPACING), "xpbd_chain", FRAME_DURATION)

    # checks
    # The joints stay closed to within a tenth of the link spacing, hidden in the links' overlapping ends, and
    # the pin, of infinite mass, never leaves the origin.
    origin = NumpyContext(core.ga).multivector.w.dual()
    assert gaps.to_array().max() < 0.1 * LINK_SPACING
    assert ((motors[:, 0] >> origin) & origin).norm().to_array().max() < 1e-6


if __name__ == "__main__":
    main()
