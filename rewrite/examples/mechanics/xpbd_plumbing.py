"""Plumbing, topology, and visualization for XPBD rigid body chain simulations.

Handles bookkeeping outside the mathematical core:
- Link geometry construction and anchor point embedding
- Gather / scatter of constraint partitions across body arrays
- Simulation stepping and constraint violation tracking
- 3D visualization and plot generation
"""

from __future__ import annotations

from typing import List, NamedTuple, Tuple
import numpy as np

from numga import Context, Extensor, NumpyContext
from numga.algebras import PGA3D
from examples.mechanics.lie_integrators import inertia_from_points
from examples.mechanics.xpbd import (
    pre_integrate,
    post_integrate,
    project_distance_constraint,
    project_velocity_constraint,
)


class ChainState(NamedTuple):
    """Batched state of rigid bodies in the chain."""
    motor: Extensor
    rate: Extensor
    first_moment: Extensor
    inertia: Extensor
    inertia_inv: Extensor
    damping: Extensor
    gravity: Extensor


class ConstraintPartition(NamedTuple):
    """A disjoint set of pairwise constraints that can be relaxed simultaneously."""
    body_idx: np.ndarray          # Shape [2, n_constraints]
    anchors: Extensor             # Shape [2, n_constraints], antivector points in body frame
    compliance: Extensor          # Shape [n_constraints], scalar compliance


def make_cube(n: int) -> np.ndarray:
    """Corner vertices of an n-dimensional hypercube."""
    return np.array(np.kron(np.diag(np.arange(n) + 1.0) / n, [+1.0, -1.0]).T)


def setup_chain(
    context: Context | None = None,
    n_bodies: int = 6,
    distance: float = 9e-2,
    size: float = 5e-2,
    compliance: float = 1e-9,
    damping: float = 1e-3,
    fixing: bool = True,
) -> Tuple[ChainState, List[ConstraintPartition]]:
    """Construct an N-link rigid body chain and its red-black constraint partitions.

    Parameters
    ----------
    context : Context, optional
        Context for PGA. Defaults to NumpyContext(PGA3D).
    n_bodies : int
        Number of links in the chain.
    distance : float
        Separation between adjacent link centers.
    size : float
        Physical dimension of each link cube.
    compliance : float
        Inverse stiffness alpha.
    damping : float
        Velocity damping factor.
    fixing : bool
        If True, anchor body 0 at translation directions so the chain hangs.
    """
    if context is None:
        from numga.backend.numpy import NumpyContext
        from numga.algebras import PGA3D
        context = NumpyContext(PGA3D)

    n_dim = context.algebra.dimension
    ndim = n_dim - 1
    origin_mask = 1 << ndim

    # Helper to construct translator motor for displacement d: [..., ndim]
    def translator(dist: np.ndarray) -> Extensor:
        pad = np.pad(dist, [(0, 0)] * (dist.ndim - 1) + [(0, 1)])
        vec = context.multivector.vector(pad)
        origin = context.multivector.vector(
            np.array([0] * ndim + [1], dtype=float)
        )
        generator = (origin * -0.5).wedge(vec)
        return generator.exp()

    # PGA origin point (0, ..., 0, 1) in antivector space
    origin_pt_coords = np.zeros((len(context.algebra.subspace.antivector()),), dtype=float)
    origin_pt_coords[-1] = 1.0
    origin_pt = context.multivector.antivector(origin_pt_coords)

    def point_embed(coords: np.ndarray) -> Extensor:
        t = translator(coords)
        return t >> origin_pt

    # Construct single body mass properties from point cloud
    cube_coords = make_cube(ndim) * size
    points = point_embed(cube_coords)
    inertia_single, inertia_inv_single = inertia_from_points(points)
    first_moment_single = points.sum(axis=-1)

    # Replicate template body across all n_bodies
    motor = context.multivector.rotor().broadcast_to(n_bodies)
    rate = context.multivector.bivector().broadcast_to(n_bodies)
    first_moment = first_moment_single[None].broadcast_to(n_bodies)
    inertia = inertia_single[None].broadcast_to(n_bodies)
    inertia_inv = inertia_inv_single[None].broadcast_to(n_bodies)
    damping_ext = (context.multivector.scalar() * damping).broadcast_to(n_bodies)

    grav_coords = np.zeros((len(context.algebra.subspace.antivector()),), dtype=float)
    grav_coords[-1] = 5e-2  # gravity acceleration along coordinate axis
    gravity = context.multivector.antivector(grav_coords).broadcast_to(n_bodies)

    # Offset each body along the last model axis
    d = np.zeros((n_bodies, ndim), dtype=float)
    d[:, -1] = np.arange(n_bodies) * distance
    qs = translator(d)
    motor = motor * qs

    # Initialize constraints between adjacent links
    i = np.arange(n_bodies - 1)
    body_idx_all = np.array([i, i + 1])

    # Anchor offsets in body-local frame
    a = np.zeros((2, ndim), dtype=float)
    a[0, -1] = +distance / 2.0
    a[1, -1] = -distance / 2.0
    a = a[:, None, :] * np.ones((1, n_bodies - 1, 1), dtype=float)
    anchors_all = point_embed(a)

    compliance_ext = context.multivector.scalar(
        np.full((n_bodies - 1, 1), compliance, dtype=float)
    )

    # Anchor body 0 if requested (zero out translational compliance in inertia_inv)
    if fixing:
        bivector = context.algebra.subspace.bivector()
        trans_indices = [idx for idx, m in enumerate(bivector.masks) if (m & origin_mask)]

        def fix_anchor(kernel: np.ndarray) -> np.ndarray:
            if hasattr(kernel, "at"):
                return kernel.at[0, :, trans_indices].set(0)
            k = np.array(kernel, copy=True)
            k[0, :, trans_indices] = 0
            return k

        inertia_inv = inertia_inv.map_kernel(fix_anchor)

    # Red/black partitioning: separate even and odd constraints
    # Even constraints connect (0-1, 2-3, 4-5...) and odd connect (1-2, 3-4...)
    # Disjoint pairs have independent bodies and can be relaxed in parallel
    partitions = []
    for slc in (slice(0, None, 2), slice(1, None, 2)):
        p_idx = body_idx_all[:, slc]
        p_anchors = anchors_all[:, slc]
        p_comp = compliance_ext[slc]
        if p_idx.shape[1] == 0:
            continue
        partitions.append(ConstraintPartition(p_idx, p_anchors, p_comp))

    state = ChainState(
        motor=motor,
        rate=rate,
        first_moment=first_moment,
        inertia=inertia,
        inertia_inv=inertia_inv,
        damping=damping_ext,
        gravity=gravity,
    )
    return state, partitions


def step_chain(
    state: ChainState,
    partitions: List[ConstraintPartition],
    dt: float,
) -> ChainState:
    """Perform a single XPBD integration step on the chain state."""
    old_motor = state.motor

    # 1. Unconstrained inertial pre-integration:
    motor, rate = pre_integrate(
        state.motor,
        state.rate,
        state.inertia,
        state.inertia_inv,
        state.first_moment,
        state.gravity,
        state.damping,
        dt,
    )

    # 2. Relax position constraints (Gauss-Seidel over red-black partitions):
    for part in partitions:
        m_pair = motor[part.body_idx]
        inv_I_pair = state.inertia_inv[part.body_idx]
        m_updated = project_distance_constraint(
            m_pair, part.anchors, inv_I_pair, part.compliance, dt
        )
        motor = motor.at[part.body_idx].set(m_updated)

    # 3. Post-integration: recover rates from motor displacement:
    motor, rate = post_integrate(old_motor, motor, dt)

    # 4. Resolve velocity constraints:
    for part in partitions:
        m_pair = motor[part.body_idx]
        r_pair = rate[part.body_idx]
        inv_I_pair = state.inertia_inv[part.body_idx]
        r_updated = project_velocity_constraint(
            m_pair, r_pair, part.anchors, inv_I_pair, dt
        )
        rate = rate.at[part.body_idx].set(r_updated)

    return ChainState(
        motor=motor,
        rate=rate,
        first_moment=state.first_moment,
        inertia=state.inertia,
        inertia_inv=state.inertia_inv,
        damping=state.damping,
        gravity=state.gravity,
    )


def compute_violations(motor: Extensor, partitions: List[ConstraintPartition]) -> np.ndarray:
    """Compute the scalar Euclidean joint separation for all constraints."""
    violations = []
    for part in partitions:
        world_anchors = motor[part.body_idx] >> part.anchors
        line = world_anchors[0] & world_anchors[1]
        violations.append(line.norm().kernel.ravel())
    return np.concatenate(violations)


def simulate_chain(
    state: ChainState,
    partitions: List[ConstraintPartition],
    n_steps: int = 50,
    substeps: int = 5,
    dt: float = 0.02,
) -> Tuple[List[ChainState], List[np.ndarray]]:
    """Execute swinging chain simulation over time."""
    dt_sub = dt / substeps
    states = [state]
    violations = [compute_violations(state.motor, partitions)]

    for _ in range(n_steps):
        for _ in range(substeps):
            state = step_chain(state, partitions, dt=dt_sub)
        states.append(state)
        violations.append(compute_violations(state.motor, partitions))

    return states, violations


def draw_chain(
    states: List[ChainState],
    partitions: List[ConstraintPartition],
    save_path: str,
) -> object:
    """Render 3D link trajectories and constraint residual history."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5), dpi=120)

    # 1. 3D link trajectory
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    n_bodies = states[0].motor.shape[0]

    # Extract center positions over time: center is motor >> origin
    origin_pt = states[0].first_moment[0].context.multivector.antivector([0, 0, 0, 1])
    times = [0, len(states) // 4, len(states) // 2, 3 * len(states) // 4, -1]
    colors = plt.cm.viridis(np.linspace(0.2, 1.0, len(times)))

    for t_idx, col in zip(times, colors):
        m = states[t_idx].motor
        pts = (m >> origin_pt).kernel
        pts_norm = pts[:, :3] / pts[:, 3:4]
        ax1.plot(pts_norm[:, 0], pts_norm[:, 1], pts_norm[:, 2], "-o", color=col,
                 label=f"Step {t_idx if t_idx >= 0 else len(states)-1}", markersize=5)

    ax1.set_title("Swinging Chain (XPBD Rigid Body Poses)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend(loc="upper left", fontsize=8)

    # 2. Joint separation history
    ax2 = fig.add_subplot(1, 2, 2)
    all_violations = [compute_violations(s.motor, partitions) for s in states]
    max_v = [float(np.max(v)) for v in all_violations]
    mean_v = [float(np.mean(v)) for v in all_violations]

    ax2.plot(max_v, label="Max joint separation", color="crimson", linewidth=1.5)
    ax2.plot(mean_v, label="Mean joint separation", color="royalblue", linestyle="--")
    ax2.set_yscale("log")
    ax2.set_title("Constraint Violation History", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Anchor distance (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    return fig
