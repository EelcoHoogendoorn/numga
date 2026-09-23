"""Scenes for the XPBD chain: links of 5 cm, 9 cm apart, with stiff joints and light damping,
in gravity of 0.5, stepped with NumPy and under jax.jit."""

from __future__ import annotations

import numpy as np

from numga import Context, Extensor, NumpyContext
from examples.mechanics.lie_integrators import inertia_from_points
from examples.mechanics.xpbd import core
from examples.mechanics.xpbd.core import Chain, Joints, Motor, Point, Scalar

LINK_SPACING = 9e-2


def chain(
    context: Context, bodies: int, distance: float, size: float,
    compliance: float, damping: float, gravity: float,
) -> tuple[Chain, list[Joints]]:
    """A chain of links along x, the first one fixed at the origin, hanging in gravity along -z.

    Each link is six unit masses on its axes, at distances size/3, 2 size/3 and size along
    x, y and z, so its three moments of inertia differ. Adjacent links are joined halfway
    between their centres. The joints are split red/black: even joints connect
    (0-1, 2-3, ...) and odd joints connect (1-2, 3-4, ...), so the bodies within one
    partition are independent and relax in parallel.
    """
    mv = context.multivector
    origin = mv.zyx

    # Mass properties of one link, from its point cloud, replicated over all links.
    reach = np.diag([1.0, 2.0, 3.0]) * size / 3
    cloud = mv("yzw zxw xyw", np.concatenate([reach, -reach])) + origin
    inertia, inertia_inv = inertia_from_points(cloud)

    # The fixed link has infinite mass: zero inverse inertia.
    mobility = np.concatenate([[0.0], np.ones(bodies - 1)])

    state = Chain(
        motor=(mv.xw * (np.arange(bodies) * distance / 2)).exp().cast(Motor.output_subspace),
        rate=mv.bivector().broadcast_to(bodies),
        first_moment=cloud.sum(axis=-1)[None].broadcast_to(bodies),
        inertia=inertia[None].broadcast_to(bodies),
        inertia_inv=inertia_inv[None].broadcast_to(bodies) * mobility,
        damping=(mv.scalar() * damping).broadcast_to(bodies),
        gravity=(mv.xyw * -gravity).broadcast_to(bodies),
    )

    # Joint j connects the +x anchor of link j to the -x anchor of link j + 1.
    link = np.arange(bodies - 1)
    pairs = np.array([link, link + 1])
    anchors = origin + mv.yzw * (np.array([[+0.5], [-0.5]]) * distance * np.ones(bodies - 1))
    compliances = mv.scalar(np.full((bodies - 1, 1), compliance))
    partitions = [Joints(pairs[:, half], anchors[:, half], compliances[half])
                  for half in (slice(0, None, 2), slice(1, None, 2))]
    return state, partitions


def swinging_chain(links: int, steps: int, substeps: int, dt: float) -> tuple[Point, Scalar]:
    """The link centres over time, and the gap at every joint, stepped with NumPy."""
    initial, partitions = chain(NumpyContext(core.ga), links, LINK_SPACING, 5e-2, 1e-9, 1e-3, 0.5)
    states = [initial]
    for _ in range(steps):
        states.append(core.advance(states[-1], partitions, substeps, dt))
    return trajectory(states, partitions)


def swinging_chain_jax(links: int, steps: int, substeps: int, dt: float) -> tuple[Point, Scalar]:
    """The same chain in single precision, each step compiled by jax.jit. Needs JAX installed."""
    import jax
    from numga.backend.jax import JaxContext

    initial, partitions = chain(JaxContext(core.ga), links, LINK_SPACING, 5e-2, 1e-9, 1e-3, 0.5)
    advance = jax.jit(lambda state: core.advance(state, partitions, substeps, dt))
    # The first step runs eagerly: it settles the traits the initial state is built with,
    # so that jit traces the step once.
    states = [initial, core.advance(initial, partitions, substeps, dt)]
    for _ in range(steps - 1):
        states.append(advance(states[-1]))
    return trajectory(states, partitions)


def trajectory(states: list[Chain], partitions: list[Joints]) -> tuple[Point, Scalar]:
    """The origin of every link frame, and every joint gap, stacked over time."""
    motors = Extensor.stack([state.motor for state in states])
    origin = motors.context.multivector.zyx
    centres = motors >> origin
    gaps = Extensor.stack([core.joint_gaps(state.motor, partitions) for state in states])

    # --- checks
    # The joints stay closed to within 1% of the link spacing.
    assert gaps.to_array().max() < 1e-2 * LINK_SPACING
    # The fixed link has infinite mass: its centre never leaves the origin.
    assert (centres[:, 0] & origin).norm().select[0].to_array().max() < 1e-6
    return centres, gaps


if __name__ == "__main__":
    import argparse
    from examples.animation import save_figure
    from examples.mechanics.xpbd import render

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax", action="store_true", help="Also run the chain under jax.jit.")
    args = parser.parse_args()
    save_figure(render.draw_chain(*swinging_chain(6, 60, 5, 0.02)), "xpbd_chain")
    if args.jax:
        save_figure(render.draw_chain(*swinging_chain_jax(6, 60, 5, 0.02)), "xpbd_chain_jax")
