"""A connection of motors on flat charts: on the sphere's, each corner's ramp reads one at its corner, each edge's
Whitney form one along its own edge, the rolled plane steps along every edge by the chart's conformal factor, the
motors around each triangle close up and turn by its area on the sphere, and the fan of geodesics meets at the
antipode; on anti-de Sitter's strip, particles thrown at every speed meet again after half a turn of proper time; on a
round sphere and three-sphere, rolled on themselves, paths run along great circles to the antipode."""

import numpy as np

from examples.relativity.motor_connection import scenarios

core = scenarios.plane

mesh, vertices = scenarios.chart()
ramps = core.coframe(mesh, vertices)                                            # [corners] Plane+[faces]
rates = core.sphere_rates(mesh, vertices)                                       # Rate+[edges]


def test_ramps_read_one_at_their_corner_and_whitney_forms_one_along_their_edge():
    corners = mesh.incidence[2, 0] * vertices                                   # [corners] Point-[faces]
    readings = (ramps[:, None] & corners[None, :]).batch()                     # [corners, corners, faces] Scalar+
    np.testing.assert_allclose(readings.to_array(), np.eye(3)[..., None] * np.ones(len(mesh.faces)), atol=1e-12)
    tails, heads = mesh.corners[2, 1]
    whitney = ramps[tails] ^ ramps[heads]                                       # [sides] Bivector+[faces]
    sides = corners[tails] & (corners[heads] - corners[tails])                  # [sides] Line-[faces]
    readings = (whitney[:, None] & sides[None, :]).batch()                      # [sides, sides, faces] Scalar+
    np.testing.assert_allclose(readings.to_array(), np.eye(3)[..., None] * np.ones(len(mesh.faces)), atol=1e-12)


def test_the_rolled_plane_steps_along_each_edge_by_the_conformal_factor():
    edges = mesh.d[0] * vertices                                                # Antivector-[edges]
    midpoints = mesh.incidence[1, 0].mean(axis=0) * vertices                    # Point-[edges]
    conformal = 2 / (1 + core.polar((midpoints - core.origin).cast(core.Step)).squared())    # Scalar+[edges]
    np.testing.assert_allclose((core.origin.commutator(rates) - conformal * edges).kernel, 0, atol=1e-12)


def test_the_motors_around_each_triangle_close_up_and_turn_by_its_area_on_the_sphere():
    # Each side's motor as the triangle runs it; around from the first corner, the side across from the last corner
    # comes first, then the one across from the first, then the one across from the second.
    motors = (mesh.boundary[2, 1] * rates * -0.5).exp().batch()                 # [sides, faces] Motor+
    holonomy = motors[2] * motors[0] * motors[1]                                # [faces] Motor+
    # A rate's turn is its weight, read by the plane at infinity; its step moves the point of contact.
    turns = 2 * (core.infinity & holonomy.log())                                # [faces] Scalar+
    steps = core.polar(core.origin.commutator(holonomy.log())).squared().square_root()   # [faces] Scalar+
    sides = (mesh.boundary[2, 1] * core.origin.commutator(rates)).batch()       # [sides, faces] Antivector-
    perimeters = core.polar(sides.cast(core.Step)).squared().square_root().sum(axis=0)       # [faces] Scalar+
    corners = mesh.incidence[2, 0] * vertices                                   # [corners] Point-[faces]
    centres = (mesh.incidence[2, 0].mean(axis=0) * vertices).batch()            # [faces] Point-
    conformal = 2 / (1 + core.polar((centres - core.origin).cast(core.Step)).squared())      # [faces] Scalar+
    sphere_areas = conformal * conformal * ((corners[0] & corners[1] & corners[2]) / 2).batch()   # [faces] Scalar-

    # Around every triangle the rolled plane comes back to its point of contact, so the connection has no torsion,
    # and turns by the curvature inside it, which adds up to the sphere's area within the chart.
    assert (steps / perimeters).to_array().max() < 0.02
    np.testing.assert_allclose((turns.sum() / sphere_areas.sum()).to_array(), 1, atol=1e-3)


def test_the_fan_meets_at_the_antipode():
    connection = core.connection(mesh, ramps, rates)                            # (Rate+ <- Line-)[faces]
    starts = scenarios.source[None].broadcast_to((scenarios.n_paths,))          # [paths] Point-
    *_, ends = core.geodesics(mesh, vertices, ramps, connection, starts, scenarios.headings, scenarios.step,
                              scenarios.n_half_turn)                            # [paths] Point-
    misses = core.polar((ends - scenarios.antipode).cast(core.Step)).squared().square_root()  # [paths] Scalar+
    assert misses.to_array().max() < 0.05


def test_particles_thrown_in_anti_de_sitter_meet_again_after_half_a_turn():
    spacetime = scenarios.spacetime
    strip_mesh, strip_vertices = scenarios.strip()
    thrown = scenarios.thrown                                                   # [particles] Antivector-
    worldlines = scenarios.rolled(spacetime, strip_mesh, strip_vertices,
                                  spacetime.anti_de_sitter_rates(strip_mesh, strip_vertices),
                                  scenarios.event[None].broadcast_to(thrown.shape), thrown)   # [particles, steps + 1] Point-
    missed = (worldlines[:, scenarios.n_half_turn] - scenarios.meeting).cast(spacetime.Step)   # [particles] Antivector-
    readings = (spacetime.mv.x & missed, spacetime.mv.t & missed)               # [particles] Scalar+, Scalar+
    assert max(np.abs(reading.to_array()).max() for reading in readings) < 0.03


def test_paths_on_the_round_sphere_run_along_great_circles():
    sphere = scenarios.sphere
    sphere_mesh, sphere_vertices = scenarios.icosphere()
    headings = scenarios.sphere_headings                                        # [paths] Antivector-
    paths = scenarios.rolled(sphere, sphere_mesh, sphere_vertices,
                             sphere.round_rates(sphere_mesh, scenarios.great_circle_frames(sphere_vertices)),
                             sphere.origin[None].broadcast_to(headings.shape), headings)   # [paths, steps + 1] Point-
    # A quarter turn along, each path is where its great circle is, the origin turned toward its heading.
    quarter = scenarios.n_half_turn // 2
    angle = quarter * scenarios.step
    expected = sphere.origin * np.cos(angle) + headings * np.sin(angle)       # [paths] Point-
    assert np.abs((paths[:, quarter].normalized() - expected).kernel).max() < 0.02


def test_paths_through_the_600_cell_meet_at_the_antipode():
    three_sphere = scenarios.three_sphere
    cell_mesh, cell_vertices = scenarios.six_hundred_cell()
    headings = scenarios.three_sphere_headings                                  # [paths] Antivector-
    paths = scenarios.rolled(three_sphere, cell_mesh, cell_vertices,
                             three_sphere.round_rates(cell_mesh, scenarios.three_sphere.isoclinic(cell_vertices)),
                             three_sphere.origin[None].broadcast_to(headings.shape), headings)   # [paths, steps + 1] Point-
    # Half a turn along, every path is near the antipode, within what cells a fifth of a half turn wide allow.
    ends = paths[:, scenarios.n_half_turn].normalized()                         # [paths] Point-
    assert np.abs((ends + three_sphere.origin).kernel).max() < 0.3
