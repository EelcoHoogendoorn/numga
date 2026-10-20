"""Scenes for the XPBD chain: links of 5 cm, 9 cm apart, with stiff joints and light damping, in gravity
of 0.5, stepped with NumPy and under jax.jit."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import Context, Extensor, NumpyContext
from examples.mechanics.lie_integrators import inertia_from_points
from examples.mechanics.xpbd import core
from examples.mechanics.xpbd.core import Chain, Joints, Motor, Point, Scalar

# The chain: its links, their spacing and size, the joints' compliance, the damping and gravity.
N_LINKS = 6
LINK_SPACING = 9e-2
LINK_SIZE = 5e-2
COMPLIANCE = 1e-9
DAMPING = 1e-3
GRAVITY = 0.5
# The run: its steps, the substeps per step and the time step; and the duration of a frame in milliseconds.
N_STEPS = 200
N_SUBSTEPS = 5
STEP = 0.02
FRAME_DURATION = 60


# --- math -----------------------------------------------------------------------------
def chain(
    context: Context, bodies: int, distance: float, size: float,
    compliance: float, damping: float, gravity: float,
) -> tuple[Chain, list[Joints]]:
    """A chain of links along x, the first one fixed at the origin, hanging in gravity along -z.

    Each link is six unit masses on its axes, at distances `size / 3`, `2 * size / 3` and
    `size` along x, y and z, so its three moments of inertia differ. Adjacent links are joined
    halfway between their centres. The joints are split red/black: even joints connect
    (0-1, 2-3, ...) and odd joints connect (1-2, 3-4, ...), so the bodies within one
    partition are independent and relax in parallel.
    """
    mv = context.multivector
    origin = mv.w.dual()

    # Mass properties of one link, from its point cloud, replicated over all links.
    reach = np.diag([1.0, 2.0, 3.0]) * size / 3
    cloud = mv.point(np.concatenate([reach, -reach]))
    inertia, inverse_inertia = inertia_from_points(cloud)

    # The fixed link has infinite mass: zero inverse inertia.
    mobility = np.concatenate([[0.0], np.ones(bodies - 1)])

    state = Chain(
        motor=(mv.xw * (np.arange(bodies) * distance / 2)).exp().cast(Motor.output_subspace),
        rate=mv.bivector().broadcast_to(bodies),
        first_moment=cloud.sum(axis=-1)[None].broadcast_to(bodies),
        inertia=inertia[None].broadcast_to(bodies),
        inverse_inertia=inverse_inertia[None].broadcast_to(bodies) * mobility,
        damping=(mv.scalar() * damping).broadcast_to(bodies),
        gravity=mv.direction([0.0, 0.0, -gravity]).broadcast_to(bodies),
    )

    # Joint j connects the +x anchor of link j to the -x anchor of link j + 1.
    link = np.arange(bodies - 1)
    pairs = np.array([link, link + 1])
    anchors = origin + mv.direction([1.0, 0.0, 0.0]) * (np.array([[+0.5], [-0.5]]) * distance * np.ones(bodies - 1))
    compliances = mv.scalar(np.full((bodies - 1, 1), compliance))
    partitions = [Joints(pairs[:, half], anchors[:, half], compliances[half])
                  for half in (slice(0, None, 2), slice(1, None, 2))]
    return state, partitions


def swinging_chain(n_links: int, n_steps: int, n_substeps: int, dt: float) -> tuple[Point, Scalar]:
    """The link centres over time, and the gap at every joint, stepped with NumPy."""
    initial, partitions = chain(NumpyContext(core.ga), n_links, LINK_SPACING, LINK_SIZE, COMPLIANCE, DAMPING, GRAVITY)
    return trajectory(list(islice(states(initial, lambda state: core.advance(state, partitions, n_substeps, dt)), n_steps + 1)), partitions)


def swinging_chain_jax(n_links: int, n_steps: int, n_substeps: int, dt: float) -> tuple[Point, Scalar]:
    """The same chain in single precision, each step compiled by jax.jit. Needs JAX installed."""
    import jax
    from numga.backend.jax import JaxContext

    initial, partitions = chain(JaxContext(core.ga), n_links, LINK_SPACING, LINK_SIZE, COMPLIANCE, DAMPING, GRAVITY)
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


def trajectory(history: list[Chain], partitions: list[Joints]) -> tuple[Point, Scalar]:
    """The origin of every link frame, and every joint gap, stacked over time."""
    motors = Extensor.stack([state.motor for state in history])
    centres = motors >> motors.context.multivector.w.dual()                   # [steps + 1, links] Point
    gaps = Extensor.stack([core.joint_gaps(state.motor, partitions) for state in history])   # [steps + 1, joints] Scalar
    return centres, gaps


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.mechanics.xpbd import render

    centres, gaps = swinging_chain(N_LINKS, N_STEPS, N_SUBSTEPS, STEP)
    save_animation(render.animate_chain(centres), "xpbd_chain", FRAME_DURATION)
    save_figure(render.draw_gaps(gaps), "xpbd_gaps")

    # checks
    # The joints stay closed to within 1% of the link spacing, and the fixed link, of infinite mass, never
    # leaves the origin.
    origin = NumpyContext(core.ga).multivector.w.dual()
    assert gaps.to_array().max() < 1e-2 * LINK_SPACING
    assert (centres[:, 0] & origin).norm().to_array().max() < 1e-6


if __name__ == "__main__":
    main()
