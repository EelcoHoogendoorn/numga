"""Unit tests for XPBD geometric constraint projection and chain dynamics."""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D
from examples.mechanics.xpbd import (
    project_distance_constraint,
    project_velocity_constraint,
    main,
)
from examples.mechanics.xpbd_plumbing import setup_chain, simulate_chain, compute_violations


def test_geometric_distance_constraint_projection():
    """Verify that project_distance_constraint closes anchor gaps between body pairs."""
    context = NumpyContext(PGA3D)
    state, partitions = setup_chain(context, n_bodies=2, distance=0.1, compliance=0.0)

    part = partitions[0]
    initial_violations = compute_violations(state.motor, [part])
    assert initial_violations[0] < 1e-12

    # Perturb body 1 by a random translation
    trans = (context.multivector.vector([0.02, -0.01, 0.03, 0.0]) * -0.5).wedge(
        context.multivector.vector([0.0, 0.0, 0.0, 1.0])
    ).exp()
    perturbed_motor = state.motor.at[1].set(state.motor[1] * trans)

    perturbed_violations = compute_violations(perturbed_motor, [part])
    assert perturbed_violations[0] > 0.01

    # Apply XPBD constraint projections to let non-linear rotational displacement converge
    m_pair = perturbed_motor[part.body_idx]
    inv_I_pair = state.inertia_inv[part.body_idx]
    dt = 0.01

    motor_cur = perturbed_motor
    for _ in range(6):
        relaxed_pair = project_distance_constraint(
            motor_cur[part.body_idx], part.anchors, inv_I_pair, part.compliance, dt=dt
        )
        motor_cur = motor_cur.at[part.body_idx].set(relaxed_pair)

    relaxed_violations = compute_violations(motor_cur, [part])
    assert relaxed_violations[0] < 1e-4


def test_velocity_constraint_projection():
    """Verify that project_velocity_constraint cancels relative anchor velocity."""
    context = NumpyContext(PGA3D)
    state, partitions = setup_chain(context, n_bodies=2, distance=0.1)
    part = partitions[0]

    # Perturb body 1 with a linear velocity along the constraint direction
    delta_v = np.zeros(state.rate.kernel.shape)
    delta_v[1, 0] = 0.5
    perturbed_rate = state.rate.map_kernel(lambda k: k + delta_v, preserve_traits=True)

    m_pair = state.motor[part.body_idx]
    r_pair = perturbed_rate[part.body_idx]
    inv_I_pair = state.inertia_inv[part.body_idx]

    bivector = context.algebra.subspace.bivector()
    anchors_map = part.anchors & part.anchors.commutator(bivector)
    initial_rel_v = (m_pair >> anchors_map(r_pair))[0] - (m_pair >> anchors_map(r_pair))[1]
    assert np.max(initial_rel_v.norm().kernel) > 0.01

    # Project velocity
    resolved_rates = project_velocity_constraint(m_pair, r_pair, part.anchors, inv_I_pair, dt=0.01)
    final_rel_v = (m_pair >> anchors_map(resolved_rates))[0] - (m_pair >> anchors_map(resolved_rates))[1]
    assert np.max(final_rel_v.norm().kernel) < 1e-6


def test_xpbd_chain_simulation():
    """Simulate a swinging chain and verify all joints remain connected."""
    state, partitions = setup_chain(n_bodies=4, distance=0.08)
    states, violations = simulate_chain(
        state,
        partitions,
        n_steps=30,
        substeps=4,
        dt=0.02,
    )
    assert len(states) == 31
    for v in violations:
        assert np.max(v) < 1e-4


def test_main_figure(tmp_path):
    """Verify that main runs and generates the output figure."""
    import matplotlib
    matplotlib.use("Agg")

    out_file = tmp_path / "test_chain.png"
    states, violations = main(n_bodies=4, n_steps=10, substeps=2, plot_path=str(out_file))
    assert out_file.exists()
    assert len(states) == 11
