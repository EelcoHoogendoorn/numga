"""A map's trace pairs its output with an input slot of the same space; forms need a metric."""

import numpy as np
import pytest

from numga import Algebra, NumpyContext
from numga.algebras import STA


def random_map(mv, basis, space, rng):
    """A map space <- space from dyads; the input pairing uses complements only."""
    n = basis.shape[0]
    return (basis[:, None] * (basis[None, :].dual() & space) * rng.normal(size=(n, n))).sum(axis=0).sum(axis=0)


@pytest.mark.parametrize("signature", ["x+y+z+", "t+x-y-z-", "x+y+z+w0", "x+y+z+p+n-"])
def test_trace_is_the_exterior_contraction(signature):
    """`f.trace() == ((basis.dual() & f(basis)) / (basis.dual() & basis)).sum(axis=0)`: the
    regressive product and complements alone, which holds in degenerate signatures where no
    reciprocal frame exists."""
    ga = Algebra(signature)
    mv = NumpyContext(ga).multivector
    vector = ga.gatype.vector()
    basis = mv.vector(np.eye(len(ga.subspace.vector())))
    f = random_map(mv, basis, vector, np.random.default_rng(0))
    complements = basis.dual()
    exterior = ((complements & f(basis)) / (complements & basis)).sum(axis=0)
    np.testing.assert_allclose(f.trace().kernel, exterior.kernel, atol=1e-12)


def test_trace_is_invariant_under_any_change_of_basis():
    ga = STA
    mv = NumpyContext(ga).multivector
    bivector = ga.gatype.bivector()
    basis = mv.bivector(np.eye(6))
    rng = np.random.default_rng(1)
    f, change = random_map(mv, basis, bivector, rng), random_map(mv, basis, bivector, rng)
    np.testing.assert_allclose(change(f(change.inverse())).trace().kernel, f.trace().kernel, atol=1e-10)


def test_trace_rejects_a_slot_spanning_part_of_the_output():
    ga = Algebra("x+y+z+p+n-")
    mv = NumpyContext(ga).multivector
    direction = ga.gatype.from_blades("x y z")
    partial = mv.vector(np.eye(5))[0] * (mv.x | direction)                  # Vector <- Direction
    with pytest.raises(TypeError):
        partial.trace()


def test_metric_raised_trace_and_relative_det_are_frame_independent():
    """A form's trace and determinant exist relative to a metric, and survive a boost."""
    pytest.importorskip("scipy")
    ga = STA
    mv = NumpyContext(ga).multivector
    vector = ga.gatype.vector()
    t, x, y, z = mv.vector(np.eye(4))
    basis = mv.vector(np.eye(4))
    symmetric = (lambda a: a + a.T)(np.random.default_rng(2).normal(size=(4, 4)))
    form = ((basis[:, None] | vector) * (basis[None, :] | vector) * symmetric).sum(axis=0).sum(axis=0)
    metric = vector | vector
    boost = ((x ^ t) * 0.4).exp()
    boosted = form(boost >> vector, boost >> vector)
    np.testing.assert_allclose(metric.solve(boosted).trace().kernel, metric.solve(form).trace().kernel, atol=1e-10)
    np.testing.assert_allclose(boosted.det(metric).kernel, form.det(metric).kernel, atol=1e-10)


@pytest.mark.parametrize("signature", ["x+y+", "x+y+z+", "t+x-y-z-", "x+y+z+w0"])
def test_pairing_the_output_is_lifting_the_map_to_a_form(signature):
    """`trace` on the output is the trace of the map met with its complement, and `contract` on the
    output is the same trace: the pairing and its inverse cancel, so neither needs an inverse
    metric, and both hold in degenerate signatures."""
    ga = Algebra(signature)
    mv = NumpyContext(ga).multivector
    vector, antivector = ga.gatype.vector(), ga.gatype.antivector()
    basis = mv.vector(np.eye(len(ga.subspace.vector())))
    f = random_map(mv, basis, vector, np.random.default_rng(3))
    np.testing.assert_allclose((antivector & f).trace().kernel, f.trace().kernel, atol=1e-12)
    np.testing.assert_allclose(f.contract().kernel, f.trace().kernel, atol=1e-12)


@pytest.mark.parametrize("signature", ["x+y+", "x+y+z+", "t+x-y-z-"])
def test_contracting_the_output_is_the_inner_product_lift(signature):
    ga = Algebra(signature)
    mv = NumpyContext(ga).multivector
    vector = ga.gatype.vector()
    basis = mv.vector(np.eye(len(ga.subspace.vector())))
    f = random_map(mv, basis, vector, np.random.default_rng(4))
    np.testing.assert_allclose((vector | f).contract().kernel, f.contract().kernel, atol=1e-12)


def test_slots_are_named_unless_only_two_carry_an_index():
    """A scalar output carries no index: a form's two inputs need no names, a map with an open
    vector multiplied in has three slots that do, and a scalar output pairs with nothing."""
    ga = Algebra("x+y+z+")
    mv = NumpyContext(ga).multivector
    vector = ga.gatype.vector()
    f = random_map(mv, mv.vector(np.eye(3)), vector, np.random.default_rng(5))
    np.testing.assert_allclose((vector | f).contract().kernel, (vector | f).contract(1, 2).kernel, atol=1e-12)
    with pytest.raises(TypeError):
        (vector * f).contract()
    with pytest.raises(TypeError):
        (vector | f).contract(0, 1)
