"""The topology of simplicial meshes: its derivatives following each other to nothing in every dimension, its border,
orientation and manifoldness, its dual, its products and its subdivision, on fields of an algebra."""

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D
from numga.topology import Topology, coherent, turned


def test_boundaries_of_boundaries_vanish_across_dimensions():
    # 1D path, 2D torus, 3D cube of 6 tetrahedra, and 4D pentatope boundary (5 tetrahedra).
    path = Topology.chains([4])
    torus = Topology.torus(6, 5)
    cube_tets = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]])
    solid = Topology(8, cube_tets)
    simplex_4 = Topology(5, [[0, 1, 2, 3, 4]])

    for topology in (torus, solid, simplex_4):
        for k in range(topology.n_dim - 1):
            np.testing.assert_allclose((topology.d[k + 1] * topology.d[k]).cells, 0.0, atol=0.0)
    assert path.n_dim == 1 and path.n_elements == (4, 3)


def test_border_closedness_orientation_and_manifoldness():
    # A closed tetrahedron surface wound coherently has empty border, Euler characteristic 2, and one piece.
    tet_faces = coherent(np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]))
    tet = Topology(4, tet_faces)
    assert tet.is_closed and tet.is_oriented and tet.is_manifold
    assert tet.n_components == 1 and tet.euler_characteristic == 2
    assert len(tet.border.elements[-1]) == 0

    # A 4-panel Möbius strip is a manifold with one closed rim loop, Euler characteristic 0, and non-orientable.
    mobius = Topology(8, np.array([
        [0, 2, 3], [0, 3, 1],
        [2, 4, 5], [2, 5, 3],
        [4, 6, 7], [4, 7, 5],
        [6, 1, 0], [6, 0, 7],
    ]))
    assert mobius.is_manifold and not mobius.is_closed and not mobius.is_oriented
    assert mobius.euler_characteristic == 0
    assert mobius.border.is_closed and mobius.border.n_components == 1

    # Pinched vertex star and 3-fin edge fail manifoldness; opposing vs matching shared edge distinguishes orientation.
    pinched = Topology(5, np.array([[0, 1, 2], [0, 3, 4]]))
    assert not pinched.is_manifold and pinched.n_components == 2
    fin = Topology(5, np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]))
    assert not fin.is_manifold
    assert not Topology(4, np.array([[0, 1, 2], [0, 1, 3]])).is_oriented
    assert Topology(4, np.array([[0, 1, 2], [1, 0, 3]])).is_oriented


def test_the_dual_and_the_subdivision_act_on_fields():
    mv = NumpyContext(VGA3D).multivector
    coarse = Topology(4, coherent(np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])))
    dual = coarse.dual
    rng = np.random.default_rng(0)
    values = mv.scalar(rng.normal(size=(dual.n_elements[0], 1))).field()      # Scalar+[dual vertices]
    tails, heads = dual.incidence[1, 0] * values
    np.testing.assert_allclose((heads - tails).kernel, (dual.d[0] * values).kernel, atol=1e-12)
    np.testing.assert_allclose((dual.d[1] * (dual.d[0] * values)).kernel, 0.0, atol=1e-12)

    pinned = np.array([True, False, False, False])
    finer = coarse.loop_subdivided(pinned, 2)
    assert len(finer.steps) == 2 and finer.is_closed and finer.is_oriented and finer.is_manifold
    coords = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    points = mv.vector(coords).dual().field()
    composed = finer.subdivision * points
    stepwise = finer.steps[1] * (finer.steps[0] * points)
    np.testing.assert_allclose(composed.kernel, stepwise.kernel, atol=1e-12)
    np.testing.assert_allclose(composed.batch()[0].kernel, points.batch()[0].kernel, atol=1e-12)


def test_cone_and_product_preserve_orientation_and_manifoldness():
    loop = Topology.loops([6])
    disk = loop.cone()
    assert disk.n_dim == 2 and disk.is_oriented and disk.is_manifold and not disk.is_closed
    assert disk.euler_characteristic == 1
    np.testing.assert_array_equal(turned(turned(disk.elements[-1])), disk.elements[-1])

    cylinder = loop.product(Topology.chains([3]))
    assert cylinder.n_dim == 2 and cylinder.is_oriented and cylinder.is_manifold and cylinder.euler_characteristic == 0
