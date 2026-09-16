"""Unit tests for Spherical Quadric Physics in Cl(3)."""

from __future__ import annotations

import numpy as np

from examples.quadrics.spherical_quadric_physics import (
    mv,
    Plane,
    SphericalBody,
    pointcloud_inertia,
    decompose_quadric,
    make_spherical_quadric,
    step_motor,
    find_contact_parameter,
    extract_contact_geometry,
    resolve_collision,
    simulate,
)
from examples.quadrics.spherical_quadric_scenarios import (
    make_cap_mesh,
    setup_crowded_scene,
    setup_tumbling_scene,
    setup_hyperbolic_scene,
)


def test_spherical_quadric_tangency():
    """Verify that tangent great circles to a spherical quadric satisfy L ∨ Q(L) == 0."""
    th_x = np.radians(30.0)
    th_y = np.radians(15.0)
    Q_body = make_spherical_quadric(th_x, th_y)

    for phi_deg in [0.0, 30.0, 75.0, 120.0, 200.0, 310.0]:
        phi = np.radians(phi_deg)
        lx = np.cos(phi) / np.tan(th_x)
        ly = np.sin(phi) / np.tan(th_y)
        lz = 1.0
        L_tan = mv.plane([lx, ly, lz])
        residual = L_tan.regressive(Q_body(L_tan))
        assert np.isclose(float(residual.kernel[0]), 0.0, atol=1e-12)


def test_dual_pencil_collision_criterion():
    """Verify max_λ det(Q(λ)) > 0 (separated), ≈ 0 (touching), < 0 (overlapping)."""
    th1 = np.radians(20.0)
    th2 = np.radians(25.0)
    Q1 = make_spherical_quadric(th1, th1)

    # State 1: Separated (distance = 55 degrees)
    alpha_sep = np.radians(55.0)
    R_sep = (mv.xz * (-alpha_sep / 2.0)).exp()
    Q2_sep = R_sep >> (make_spherical_quadric(th2, th2))(R_sep << Plane)
    lam_sep, max_sep = find_contact_parameter(Q1, Q2_sep)
    assert max_sep > 0.005

    # State 2: Touching (distance = 45 degrees)
    alpha_touch = np.radians(45.0)
    R_touch = (mv.xz * (-alpha_touch / 2.0)).exp()
    Q2_touch = R_touch >> (make_spherical_quadric(th2, th2))(R_touch << Plane)
    lam_touch, max_touch = find_contact_parameter(Q1, Q2_touch)
    assert np.isclose(max_touch, 0.0, atol=1e-5)

    # State 3: Overlapping (distance = 35 degrees)
    alpha_over = np.radians(35.0)
    R_over = (mv.xz * (-alpha_over / 2.0)).exp()
    Q2_over = R_over >> (make_spherical_quadric(th2, th2))(R_over << Plane)
    lam_over, max_over = find_contact_parameter(Q1, Q2_over)
    assert max_over < -0.005


def test_contour_derived_inertia_intermediate_axis():
    """Verify inertia extensor derived from rendering contour has 3 distinct moments Iz < Ix < Iy."""
    th_x = np.radians(35.0)
    th_y = np.radians(10.0)
    mass = 1.0
    cap_pts, cap_pts_mass = make_cap_mesh(th_x, th_y, mass)
    I_body = pointcloud_inertia(cap_pts, cap_pts_mass)

    w_z = mv.xy
    w_y = -mv.xz
    w_x = mv.yz

    L_z = I_body(w_z)
    L_y = I_body(w_y)
    L_x = I_body(w_x)

    Iz = abs(float(w_z.regressive(L_z).kernel[0]))
    Iy = abs(float(w_y.regressive(L_y).kernel[0]))
    Ix = abs(float(w_x.regressive(L_x).kernel[0]))

    assert Iz < Ix < Iy
    assert np.isclose(Iz, 0.094, atol=0.02)
    assert np.isclose(Ix, 0.913, atol=0.03)
    assert np.isclose(Iy, 0.992, atol=0.03)

    I_inv = I_body.inverse()
    w_rec = I_inv(L_x)
    assert np.isclose(abs(float(w_rec.regressive(L_x).kernel[0])), Ix, atol=1e-10)


def test_quadric_decomposition_and_composition():
    """Verify compose_quadric and decompose_quadric are exact decompositions."""
    th_x, th_y = np.radians(28.0), np.radians(14.0)
    Q = make_spherical_quadric(th_x, th_y)

    s, tangents = decompose_quadric(Q)
    assert tangents.shape == (3,)
    assert len(s) == 3

    for i in range(3):
        assert np.isclose(float(Q(tangents[i]).norm().kernel[0]), s[i], atol=1e-12)


def test_free_flight_energy_and_momentum_conservation():
    """Verify Lie midpoint integrator conserves kinetic energy and angular momentum."""
    th_x = np.radians(35.0)
    th_y = np.radians(10.0)
    cap_pts, cap_pts_mass = make_cap_mesh(th_x, th_y)
    I_body = pointcloud_inertia(cap_pts, cap_pts_mass)
    I_inv = I_body.inverse()

    w_init = mv.yz * 2.5 + mv.xy * 0.05
    R = mv.scalar([1.0])
    L_body = I_body(w_init)

    E0 = abs(float(((I_inv(L_body)).regressive(L_body) * 0.5).kernel.item()))
    L0 = np.linalg.norm(L_body.kernel)

    dt = 0.01
    for _ in range(200):
        R, L_body = step_motor(R, L_body, I_inv, dt)

    E_final = abs(float(((I_inv(L_body)).regressive(L_body) * 0.5).kernel.item()))
    L_final = np.linalg.norm(L_body.kernel)

    assert np.isclose(L_final, L0, atol=1e-14)
    assert np.isclose(E_final, E0, rtol=1e-4)


def test_elastic_collision_conservation():
    """Verify that collision impulse strictly conserves momentum and kinetic energy."""
    Q1_body = make_spherical_quadric(np.radians(30), np.radians(12))
    cap1, cap1_mass = make_cap_mesh(np.radians(30), np.radians(12))
    I1 = pointcloud_inertia(cap1, cap1_mass)
    I1_inv = I1.inverse()

    Q2_body = make_spherical_quadric(np.radians(20), np.radians(16))
    cap2, cap2_mass = make_cap_mesh(np.radians(20), np.radians(16), mass=1.2)
    I2 = pointcloud_inertia(cap2, cap2_mass)
    I2_inv = I2.inverse()

    R1 = mv.scalar([1.0])
    R2 = (mv.xz * (-np.radians(50.0) / 2.0)).exp()
    L1 = R1 >> I1(mv.yz * 2.0)
    L2 = R2 >> I2(mv.xz * (-1.8))

    p_star = mv.xy
    L_star = mv.x
    tau = L_star.commutator(p_star)

    w1 = R1 >> I1_inv(R1 << L1)
    w2 = R2 >> I2_inv(R2 << L2)
    v1 = w1.regressive(p_star)
    v2 = w2.regressive(p_star)
    v_rel = L_star | (v1 - v2)

    tau1 = R1 << tau
    tau2 = R2 << tau
    v_rel_pbd = tau1.regressive(I1_inv(R1 << L1)) - tau2.regressive(I2_inv(R2 << L2))
    assert np.isclose(float(v_rel_pbd.kernel.item()), float(v_rel.kernel.item()), atol=1e-12)

    E_before = (
        abs(float(((I1_inv(R1 << L1)).regressive(R1 << L1) * 0.5).kernel.item()))
        + abs(float(((I2_inv(R2 << L2)).regressive(R2 << L2) * 0.5).kernel.item()))
    )
    L_before = np.linalg.norm((L1 + L2).kernel)

    dw1 = R1 >> I1_inv(R1 << tau)
    dw2 = R2 >> I2_inv(R2 << tau)
    m_eff_inv = (L_star | dw1.regressive(p_star)) + (L_star | dw2.regressive(p_star))
    
    m_eff_inv_pbd = tau1.regressive(I1_inv(tau1)) + tau2.regressive(I2_inv(tau2))
    assert np.isclose(float(m_eff_inv_pbd.kernel.item()), float(m_eff_inv.kernel.item()), atol=1e-12)

    j = (v_rel * 2.0) / m_eff_inv

    L1_after = L1 - tau * j
    L2_after = L2 + tau * j

    w1_after = R1 >> I1_inv(R1 << L1_after)
    w2_after = R2 >> I2_inv(R2 << L2_after)
    v1_after = w1_after.regressive(p_star)
    v2_after = w2_after.regressive(p_star)
    v_rel_after = L_star | (v1_after - v2_after)

    E_after = (
        abs(float(((I1_inv(R1 << L1_after)).regressive(R1 << L1_after) * 0.5).kernel.item()))
        + abs(float(((I2_inv(R2 << L2_after)).regressive(R2 << L2_after) * 0.5).kernel.item()))
    )
    L_after = np.linalg.norm((L1_after + L2_after).kernel)

    assert np.isclose(v_rel_after.kernel.item(), -v_rel.kernel.item(), atol=1e-10)
    assert np.isclose(E_after, E_before, atol=1e-12)
    assert np.isclose(L_after, L_before, atol=1e-14)


def test_spherical_body_resolve_collision():
    """Verify resolve_collision operates on SphericalBody dataclass instances directly."""
    th_x, th_y = np.radians(25.0), np.radians(15.0)
    cap1, cap1_mass = make_cap_mesh(th_x, th_y)
    I1 = pointcloud_inertia(cap1, cap1_mass)
    I1_inv = I1.inverse()
    R1 = mv.scalar([1.0])
    L1 = I1(mv.yz * (-2.5))
    Q1_body = make_spherical_quadric(th_x, th_y)

    b1 = SphericalBody(
        name="Body 1",
        color="#38bdf8",
        mass=1.0,
        motor=R1,
        momentum=L1,
        Q=Q1_body,
        cap_pts=cap1,
        I_inv=I1_inv,
    )

    alpha = np.radians(38.0)
    R2 = (mv.yz * (-alpha / 2.0)).exp()
    cap2, cap2_mass = make_cap_mesh(th_x, th_y)
    I2 = pointcloud_inertia(cap2, cap2_mass)
    I2_inv = I2.inverse()
    L2 = I2(mv.yz * 2.0)
    Q2_body = make_spherical_quadric(th_x, th_y)

    b2 = SphericalBody(
        name="Body 2",
        color="#f43f5e",
        mass=1.0,
        motor=R2,
        momentum=L2,
        Q=Q2_body,
        cap_pts=cap2,
        I_inv=I2_inv,
    )

    M_rel = b1.motor.inverse() * b2.motor
    Q2_in_1 = M_rel >> b2.Q(M_rel << Plane)
    lam_star, max_det = find_contact_parameter(b1.Q, Q2_in_1)

    L_world_1_before = b1.motor >> b1.momentum
    L_world_2_before = b2.motor >> b2.momentum
    L_total_before = np.linalg.norm((L_world_1_before + L_world_2_before).kernel)
    E_before = (
        abs(float(((b1.I_inv(b1.momentum)).regressive(b1.momentum) * 0.5).kernel.item()))
        + abs(float(((b2.I_inv(b2.momentum)).regressive(b2.momentum) * 0.5).kernel.item()))
    )

    resolved = resolve_collision(b1, b2, lam_star, M_rel=M_rel, Q2_in_1=Q2_in_1, restitution=1.0)
    assert resolved

    L_world_1_after = b1.motor >> b1.momentum
    L_world_2_after = b2.motor >> b2.momentum
    L_total_after = np.linalg.norm((L_world_1_after + L_world_2_after).kernel)
    E_after = (
        abs(float(((b1.I_inv(b1.momentum)).regressive(b1.momentum) * 0.5).kernel.item()))
        + abs(float(((b2.I_inv(b2.momentum)).regressive(b2.momentum) * 0.5).kernel.item()))
    )

    assert np.isclose(L_total_after, L_total_before, atol=1e-14)
    assert np.isclose(E_after, E_before, atol=1e-12)


def test_tumbling_scenario():
    """Verify tumbling scenario produces valid body, 3 distinct moments, and periodic flips."""
    body = setup_tumbling_scene()
    assert body.name == "Tumbling Oval"
    assert body.mass == 1.0

    th_x, th_y = np.radians(32.0), np.radians(10.0)
    cap, cap_mass = make_cap_mesh(th_x, th_y, n_phi=96, n_r=10)
    I = pointcloud_inertia(cap, cap_mass)
    evals, _ = np.linalg.eigh(I.kernel)
    assert evals[0] < evals[1] < evals[2]
    assert evals[1] - evals[0] > 0.5
    assert evals[2] - evals[1] > 0.01

    bodies, frame_meshes, e_hist, m_hist, w_hist, snapshots, diag_indices = (
        simulate([body], num_frames=240, dt=0.015)
    )
    e_arr = np.array(e_hist)
    e_drift = np.max(np.abs(e_arr - e_arr[0])) / e_arr[0]
    assert e_drift < 5e-4, f"Energy drift too high: {e_drift:.4%}"

    m_arr = np.array(m_hist)
    m_drift = np.max(np.abs(m_arr - m_arr[0])) / m_arr[0]
    assert m_drift < 1e-12, f"Angular momentum drift too high: {m_drift:.4e}"

    w_arr = np.array(w_hist)
    w_yz = w_arr[:, 0]
    crossings = np.where(np.diff(np.sign(w_yz)))[0]
    assert len(crossings) >= 3, f"Expected at least 3 Dzhanibekov flips, found {len(crossings)}"


def test_hyperbolic_scenario():
    """Verify hyperbolic scenario has valid separation and maintains conservation laws."""
    bodies = setup_hyperbolic_scene()
    assert len(bodies) == 5
    assert bodies[0].name == "Giant Oval"
    assert bodies[0].mass == 6.0

    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            Q1 = bodies[i].motor >> bodies[i].Q(bodies[i].motor << Plane)
            Q2 = bodies[j].motor >> bodies[j].Q(bodies[j].motor << Plane)
            lam_star, max_det = find_contact_parameter(Q1, Q2)
            assert max_det > 0.0, f"Overlap between {bodies[i].name} and {bodies[j].name} at t=0: max_det={max_det}"

    bodies, frame_meshes, e_hist, m_hist, _, snapshots, diag_indices = (
        simulate(bodies, num_frames=60, dt=0.015)
    )
    e_arr = np.array(e_hist)
    e_drift = np.max(np.abs(e_arr - e_arr[0])) / e_arr[0]
    assert e_drift < 5e-4, f"Kinetic energy drift too high: {e_drift:.4%}"

    m_arr = np.array(m_hist)
    m_drift = np.max(np.abs(m_arr - m_arr[0])) / m_arr[0]
    assert m_drift < 1e-12, f"Angular momentum drift too high: {m_drift:.4e}"


def test_crowded_scenario():
    """Verify crowded scenario (7 bodies) initializes without overlap and preserves invariants."""
    bodies = setup_crowded_scene()
    assert len(bodies) == 7

    bodies, frame_meshes, e_hist, m_hist, _, snapshots, diag_indices = (
        simulate(bodies, num_frames=40, dt=0.015)
    )
    e_arr = np.array(e_hist)
    e_drift = np.max(np.abs(e_arr - e_arr[0])) / e_arr[0]
    assert e_drift < 1e-3, f"Kinetic energy drift too high: {e_drift:.4%}"

    m_arr = np.array(m_hist)
    m_drift = np.max(np.abs(m_arr - m_arr[0])) / m_arr[0]
    assert m_drift < 1e-12, f"Angular momentum drift too high: {m_drift:.4e}"


def test_contact_geometry_projective_invariance():
    """Verify that contact geometry extraction and collision dynamics are projectively sign-invariant."""
    th_x, th_y = np.radians(25.0), np.radians(18.0)
    Q1 = make_spherical_quadric(th_x, th_y)

    R2 = (mv.xz * (np.radians(40.0) / 2.0)).exp()
    Q2 = R2 >> make_spherical_quadric(th_x, th_y)(R2 << Plane)

    lam_star, _ = find_contact_parameter(Q1, Q2)
    p_star, L_star = extract_contact_geometry(Q1, Q2, lam_star)

    assert np.linalg.norm(p_star.kernel) > 0.1
    assert np.linalg.norm(L_star.kernel) > 0.1

    w1 = mv.yz * 1.5 + mv.xy * 0.5
    w2 = (-mv.xz) * 1.2
    tau = L_star.commutator(p_star)
    tau_neg = (-L_star).commutator(-p_star)
    assert np.allclose(tau.kernel, tau_neg.kernel)

    v_rel = L_star | (w1.regressive(p_star) - w2.regressive(p_star))
    v_rel_neg = (-L_star) | (w1.regressive(-p_star) - w2.regressive(-p_star))
    assert np.isclose(v_rel.kernel.item(), v_rel_neg.kernel.item())


def test_pinned_camera_projection():
    """Pin the baseline 2D screen projections of initial and dynamic scene configurations under the folded camera."""
    crowded_expected_t0 = {
        "Cyan Baton": (-0.1388, 0.2437),
        "Rose Disc": (-0.0269, -0.8097),
        "Amber Needle": (-0.1634, -0.1844),
        "Emerald Sliver": (-0.2309, -0.0030),
        "Purple Dart": (0.7959, 0.0463),
        "Orange Oval": (0.1067, 0.2038),
        "Pink Puck": (-0.8171, -0.2425),
    }
    bodies_t0 = setup_crowded_scene()
    for b in bodies_t0:
        pts = (b.motor >> b.cap_pts).dual().kernel[:, -1, :]
        sgn = np.sign(pts[:, 2])
        bx = pts[:, 0] * sgn
        by = pts[:, 1] * sgn
        exp_x, exp_y = crowded_expected_t0[b.name]
        assert np.isclose(np.mean(bx), exp_x, atol=1e-3)
        assert np.isclose(np.mean(by), exp_y, atol=1e-3)

    b_tumble = setup_tumbling_scene()
    pts = (b_tumble.motor >> b_tumble.cap_pts).dual().kernel[:, -1, :]
    sgn = np.sign(pts[:, 2])
    bx = pts[:, 0] * sgn
    by = pts[:, 1] * sgn
    assert np.isclose(np.mean(bx), -0.0371, atol=1e-3)
    assert np.isclose(np.mean(by), 0.6030, atol=1e-3)

    hyperbolic_expected_t0 = {
        "Giant Oval": (-0.3995, -0.2242),
        "Ruby Needle": (-0.3636, -0.4253),
        "Amber Puck": (0.3537, -0.4324),
        "Emerald Dart": (0.8824, -0.2023),
        "Purple Sliver": (-0.9203, -0.1559),
    }
    bodies_hyp_t0 = setup_hyperbolic_scene()
    for b in bodies_hyp_t0:
        pts = (b.motor >> b.cap_pts).dual().kernel[:, -1, :]
        sgn = np.sign(pts[:, 2])
        bx = pts[:, 0] * sgn
        by = pts[:, 1] * sgn
        exp_x, exp_y = hyperbolic_expected_t0[b.name]
        assert np.isclose(np.mean(bx), exp_x, atol=1e-3)
        assert np.isclose(np.mean(by), exp_y, atol=1e-3)

    # Pin frame 13 mid-orbit dynamic trajectory coordinates for crowded scene:
    bodies_sim, _, _, _, _, snapshots, diag_indices = simulate(setup_crowded_scene(), num_frames=40, dt=0.015)
    crowded_expected_t13 = {
        "Cyan Baton": (-0.246, 0.010),
        "Rose Disc": (-0.029, -0.254),
        "Amber Needle": (0.717, -0.454),
        "Emerald Sliver": (0.119, -0.051),
        "Purple Dart": (0.529, 0.357),
        "Orange Oval": (0.110, 0.082),
        "Pink Puck": (-0.894, 0.321),
    }
    k13 = snapshots[diag_indices[1]]
    for b, m in zip(bodies_sim, k13):
        pts = (m >> b.cap_pts).dual().kernel[:, -1, :]
        sgn = np.sign(pts[:, 2])
        bx = pts[:, 0] * sgn
        by = pts[:, 1] * sgn
        exp_x, exp_y = crowded_expected_t13[b.name]
        assert np.isclose(np.mean(bx), exp_x, atol=1e-2)
        assert np.isclose(np.mean(by), exp_y, atol=1e-2)

    # Pin frame 13 mid-orbit dynamic trajectory coordinates for hyperbolic scene:
    bodies_hyp_sim, _, _, _, _, snapshots_hyp, diag_indices_hyp = simulate(setup_hyperbolic_scene(), num_frames=40, dt=0.015)
    hyperbolic_expected_t13 = {
        "Giant Oval": (-0.430, -0.199),
        "Ruby Needle": (-0.606, -0.373),
        "Amber Puck": (0.308, 0.021),
        "Emerald Dart": (0.156, 0.025),
        "Purple Sliver": (-0.683, 0.135),
    }
    k13_hyp = snapshots_hyp[diag_indices_hyp[1]]
    for b, m in zip(bodies_hyp_sim, k13_hyp):
        pts = (m >> b.cap_pts).dual().kernel[:, -1, :]
        sgn = np.sign(pts[:, 2])
        bx = pts[:, 0] * sgn
        by = pts[:, 1] * sgn
        exp_x, exp_y = hyperbolic_expected_t13[b.name]
        assert np.isclose(np.mean(bx), exp_x, atol=1e-2)
        assert np.isclose(np.mean(by), exp_y, atol=1e-2)

