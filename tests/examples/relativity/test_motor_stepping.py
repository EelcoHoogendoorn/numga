"""The linearized Chern–Simons form of the 600-cell's edge motors: it vanishes on pure gauge, and each edge's turn
along itself lengthens it by one in its squared length."""

import numpy as np

from examples.relativity.motor_connection import scenarios

core = scenarios.three_sphere
mesh, vertices = scenarios.six_hundred_cell()


def test_the_chern_simons_form_vanishes_on_pure_gauge():
    rng = np.random.default_rng(0)
    gauge = mesh.d[0] * core.mesh_module.as_scalar(rng.normal(size=mesh.n_vertices)).field()   # Scalar+[edges]
    np.testing.assert_allclose((core.chern_simons(mesh, vertices) * gauge).to_array(), 0, atol=1e-13)


def test_each_edge_turns_along_itself_by_one_in_its_squared_length():
    reading = core.length_reading(mesh, vertices)                               # (Scalar+ <- Rate+)[edges]
    np.testing.assert_allclose(reading(core.edge_turns(mesh, vertices)).to_array(), 1, atol=1e-12)
