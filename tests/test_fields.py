"""Fields and field maps: a axis that ranges over sites as well as blades. Binding a field into a
field axis sums over the sites, binding it into a axis over blades acts site by site, and the solve
of a field map is the coupled system over every pair of a site and a blade."""

import numpy as np
import pytest

from numga import NumpyContext
from numga.algebras import VGA3D

ctx = NumpyContext(VGA3D)
mv = ctx.multivector
Scalar, Vector, Bivector = VGA3D.gatype.scalar(), VGA3D.gatype.vector(), VGA3D.gatype.bivector()
rng = np.random.default_rng(0)
sites, other_sites, copies = 4, 5, 7


def field_map(output: int, input: int):
    """A random Vector[output] <- Vector[input], from a batch of maps over the pairs of sites."""
    return ctx.extensor(VGA3D.gatype((Vector, Vector)), rng.normal(size=(output, input, 3, 3))).field(0, 1)


def test_products_act_site_by_site_and_broadcast_plain_batches_over_the_sites():
    first = mv.vector(rng.normal(size=(sites, 3))).field()                  # Vector[sites]
    second = mv.vector(rng.normal(size=(sites, 3))).field()                 # Vector[sites]
    batch = mv.vector(rng.normal(size=(copies, 3)))                         # [copies] Vector
    np.testing.assert_allclose((first * second).batch().kernel, (first.batch() * second.batch()).kernel, atol=1e-12)
    # The sites are not a batch axis: a batch of copies stays in front of them.
    np.testing.assert_allclose((first * batch).batch().kernel, (first.batch() * batch[:, None]).kernel, atol=1e-12)
    np.testing.assert_allclose((first + mv.x).batch().kernel, (first.batch() + mv.x).kernel, atol=1e-12)
    # An extension method over blades runs at every site.
    np.testing.assert_allclose(first.norm().batch().kernel, first.batch().norm().kernel, atol=1e-12)


def test_a_field_map_sums_over_the_sites_it_binds_and_composes_as_matrices_over_blocks():
    couplings = field_map(sites, other_sites)                               # Vector[sites] <- Vector[other_sites]
    values = mv.vector(rng.normal(size=(copies, other_sites, 3))).field()   # [copies] Vector[other_sites]
    np.testing.assert_allclose(couplings(values).kernel, np.einsum("pqij,bqj->bpi", couplings.kernel, values.kernel), atol=1e-12)
    inner = field_map(other_sites, 2)
    np.testing.assert_allclose(couplings(inner).kernel, np.einsum("pqij,qrjk->prik", couplings.kernel, inner.kernel), atol=1e-12)


def test_scalar_cells_are_matrix_algebra():
    matrix, vector = rng.normal(size=(sites, sites)), rng.normal(size=sites)
    couplings = ctx.extensor(VGA3D.gatype((Scalar, Scalar)), matrix[..., None, None]).field(0, 1)
    values = mv.scalar(vector[:, None]).field()                             # Scalar[sites]
    np.testing.assert_allclose(couplings(values).kernel[:, 0], matrix @ vector, atol=1e-12)
    np.testing.assert_allclose(couplings(couplings).kernel[..., 0, 0], matrix @ matrix, atol=1e-12)
    np.testing.assert_allclose(couplings.solve(values).kernel[:, 0], np.linalg.solve(matrix, vector), atol=1e-12)


def test_the_solve_of_a_field_map_or_form_inverts_its_binding():
    couplings = field_map(sites, sites)
    values = mv.vector(rng.normal(size=(copies, sites, 3))).field()         # [copies] Vector[sites]
    np.testing.assert_allclose(couplings.solve(couplings(values)).kernel, values.kernel, atol=1e-12)
    # A symmetric form over two fields, as a Hessian is, against the form of one, as a gradient is:
    # Scalar <- (Vector[sites], Vector[sites]).
    blocks = rng.normal(size=(sites, sites, 3, 3))
    blocks = blocks + blocks.transpose(1, 0, 3, 2)
    form = ctx.extensor(VGA3D.gatype((Scalar, Vector, Vector)), blocks[:, :, None]).field(1, 2)
    np.testing.assert_allclose(form.solve(form(values)).kernel, values.kernel, atol=1e-12)



def test_the_adjoint_of_a_field_map_exchanges_its_sites_and_carries_the_summed_scalar_product():
    couplings = field_map(sites, other_sites)                               # Vector[sites] <- Vector[other_sites]
    into = ctx.extensor(VGA3D.gatype((Vector, Vector)), rng.normal(size=(sites, 3, 3))).field()   # Vector[sites] <- Vector
    values = mv.vector(rng.normal(size=(other_sites, 3))).field()           # Vector[other_sites]
    value = mv.vector(rng.normal(size=3))                                   # Vector
    probes = mv.vector(rng.normal(size=(sites, 3))).field()                 # Vector[sites]
    summed = lambda field: field.batch().sum(axis=-1).kernel
    np.testing.assert_allclose(summed(couplings.adjoint()(probes).scalar_product(values)),
                               summed(probes.scalar_product(couplings(values))), atol=1e-12)
    # A map into a field takes the field back to one value, summed over its sites.
    np.testing.assert_allclose(into.adjoint()(probes).scalar_product(value).kernel,
                               summed(probes.scalar_product(into(value))), atol=1e-12)


def test_a_field_of_maps_on_the_diagonal_acts_on_each_site_alone():
    maps = ctx.extensor(VGA3D.gatype((Vector, Vector)), rng.normal(size=(sites, 3, 3))).field()   # Vector[sites] <- Vector
    values = mv.vector(rng.normal(size=(copies, sites, 3))).field()         # [copies] Vector[sites]
    np.testing.assert_allclose(maps.on_diagonal()(values).kernel, maps(values).kernel, atol=1e-12)
    np.testing.assert_allclose(maps.on_diagonal().solve(maps(values)).kernel, values.kernel, atol=1e-12)


def test_a_map_without_sites_adds_to_a_field_map_as_it_applies():
    couplings = field_map(sites, sites)                                     # Vector[sites] <- Vector[sites]
    values = mv.vector(rng.normal(size=(copies, sites, 3))).field()         # [copies] Vector[sites]
    turn = (mv.xy * 0.3).exp() >> Vector                                    # Vector <- Vector
    np.testing.assert_allclose((couplings + Vector)(values).kernel, (couplings(values) + values).kernel, atol=1e-12)
    np.testing.assert_allclose((couplings - turn)(values).kernel, (couplings(values) - turn(values)).kernel, atol=1e-12)


def symmetric_positive(*batch: int):
    """A random symmetric positive-definite Vector <- Vector, a batch of them over the given axes."""
    cells = rng.normal(size=batch + (3, 3))
    return ctx.extensor(VGA3D.gatype((Vector, Vector)), cells @ np.swapaxes(cells, -1, -2) + 3 * np.eye(3))


def test_a_map_or_form_without_sites_acts_at_every_site_of_a_field():
    turn = (mv.xy * 0.3).exp() >> Vector                                    # Vector <- Vector
    pairing = Vector.scalar_product(Vector)                                 # Scalar <- (Vector, Vector)
    values = mv.vector(rng.normal(size=(copies, sites, 3))).field()         # [copies] Vector[sites]
    others = mv.vector(rng.normal(size=(copies, sites, 3))).field()         # [copies] Vector[sites]
    np.testing.assert_allclose(turn(values).batch().kernel, turn(values.batch()).kernel, atol=1e-12)
    # A form bound to a field is a field of forms, Scalar[sites] <- Vector, and to two a field of scalars.
    np.testing.assert_allclose(pairing(values).batch().kernel, pairing(values.batch()).kernel, atol=1e-12)
    np.testing.assert_allclose(pairing(values, others).batch().kernel, pairing(values.batch(), others.batch()).kernel, atol=1e-12)


def test_a_map_without_sites_solves_a_field_at_every_site():
    metric = symmetric_positive()                                           # Vector <- Vector
    metrics = symmetric_positive(copies)                                    # [copies] Vector <- Vector
    values = mv.vector(rng.normal(size=(copies, sites, 3))).field()         # [copies] Vector[sites]
    for solution in (metric.solve(values), metric.lstsq(values)):          # [copies] Vector[sites]
        assert solution.gatype == values.gatype
        np.testing.assert_allclose(metric(solution).kernel, values.kernel, atol=1e-12)
    # A batch of maps solves its own copy of the field, the copies in front of the sites.
    np.testing.assert_allclose(metrics(metrics.solve(values)).kernel, values.kernel, atol=1e-12)


def test_a_form_without_sites_solved_against_a_form_on_a_field_gives_the_field_it_pairs_with():
    # The metric on bivectors, the scalar product taken positive, against readings of a field of
    # bivectors, as the derivative of a sum over the sites gives them: Scalar <- Bivector[sites].
    metric = -Bivector.scalar_product(Bivector)                             # Scalar <- (Bivector, Bivector)
    readings = ctx.extensor(VGA3D.gatype((Scalar, Bivector)), rng.normal(size=(sites, 1, 3))).field(1)
    probes = mv.bivector(rng.normal(size=(copies, sites, 3))).field()       # [copies] Bivector[sites]
    for step in (metric.solve(readings), metric.lstsq(readings)):          # Bivector[sites]
        np.testing.assert_allclose(metric(step, probes).sites.sum().kernel, readings(probes).kernel, atol=1e-12)
    # A field of the forms the metric reads off each site's bivector solves back to the field.
    np.testing.assert_allclose(metric.solve(metric(probes)).kernel, probes.kernel, atol=1e-12)
    # A bilinear right-hand side keeps the sites of its other input: Bivector[sites] <- Vector[other_sites].
    bilinear = ctx.extensor(VGA3D.gatype((Scalar, Vector, Bivector)), rng.normal(size=(other_sites, sites, 1, 3, 3))).field(1, 2)
    values = mv.vector(rng.normal(size=(other_sites, 3))).field()           # Vector[other_sites]
    np.testing.assert_allclose(metric(metric.solve(bilinear)(values), probes).sites.sum().kernel,
                               bilinear(values, probes).kernel, atol=1e-11)


def test_a_map_without_sites_solves_as_the_field_map_on_its_site_diagonal_and_a_field_map_still_couples():
    metric = symmetric_positive()                                           # Vector <- Vector
    diagonal = metric[..., None].broadcast_to((sites,)).field().on_diagonal()   # Vector[sites] <- Vector[sites]
    couplings = diagonal + field_map(sites, sites)                          # Vector[sites] <- Vector[sites]
    values = mv.vector(rng.normal(size=(copies, sites, 3))).field()         # [copies] Vector[sites]
    np.testing.assert_allclose(metric.solve(values).kernel, diagonal.solve(values).kernel, atol=1e-12)
    np.testing.assert_allclose(couplings(couplings.solve(values)).kernel, values.kernel, atol=1e-10)


def test_the_steepest_step_of_a_field_is_solved_at_every_site_under_jit():
    jax = pytest.importorskip("jax")
    from numga.backend.jax import JaxContext, derivative
    jax_mv = JaxContext(VGA3D, np.float64).multivector
    metric = -Bivector.scalar_product(Bivector)                             # Scalar <- (Bivector, Bivector)
    weights = jax_mv.scalar(rng.uniform(1, 2, size=(sites, 1))).field()     # Scalar[sites]
    turns = jax_mv.bivector(rng.normal(size=(sites, 3))).field()            # Bivector[sites]

    @jax.jit
    def steepest(turns):
        # The weighted energy of every site's bivector in the metric, summed over the sites: its
        # derivative is a form on the field, and the step it pairs with is each bivector, weighted.
        slope = derivative(lambda turns: (weights * metric(turns, turns)).sites.sum() / 2)(turns)   # Scalar <- Bivector[sites]
        return metric.solve(slope)

    np.testing.assert_allclose(np.asarray(steepest(turns).kernel), np.asarray((weights * turns).kernel), atol=1e-12)


def test_least_squares_over_a_field_leaves_the_null_space_at_zero():
    # A chain of sites whose differences alone are seen: the field's mean is unseen, and least squares
    # leaves it at zero; the pseudoinverse inverts the rest.
    differences = np.eye(sites, k=1)[:-1] - np.eye(sites)[:-1]                # [sites - 1, sites]
    seen = (mv.scalar(differences[..., None]) * Vector).field(0, 1)          # Vector[sites - 1] <- Vector[sites]
    values = mv.vector(rng.normal(size=(sites, 3))).field()                  # Vector[sites]
    centred = values - values.batch().mean(axis=0)
    np.testing.assert_allclose(seen.lstsq(seen(values), rcond=1e-10).kernel, centred.kernel, atol=1e-12)
    np.testing.assert_allclose(seen.pinv(rcond=1e-10)(seen(values)).kernel, centred.kernel, atol=1e-12)


def test_spectra_and_factors_of_a_field_map_are_those_of_its_blocks_as_one_matrix():
    # Scalar cells times the identity on vectors: the matrix of every site and blade is the cells'
    # matrix repeated over the three blades.
    cells = rng.normal(size=(sites, sites))
    symmetric = cells @ cells.T + sites * np.eye(sites)
    general = (mv.scalar(cells[..., None]) * Vector).field(0, 1)            # Vector[sites] <- Vector[sites]
    positive = (mv.scalar(symmetric[..., None]) * Vector).field(0, 1)
    values = mv.vector(rng.normal(size=(copies, sites, 3))).field()         # [copies] Vector[sites]
    np.testing.assert_allclose(general.inverse()(general(values)).kernel, values.kernel, atol=1e-10)
    np.testing.assert_allclose(general.det().kernel[0], np.linalg.det(cells) ** 3, rtol=1e-10)
    np.testing.assert_allclose(general.trace().kernel[0], 3 * np.trace(cells), rtol=1e-12)
    np.testing.assert_allclose(positive.eigvalsh().kernel[..., 0], np.repeat(np.linalg.eigvalsh(symmetric), 3), rtol=1e-10)
    eigenvalues, modes = positive.eigh()                                    # [modes] Scalar, [modes] Vector[sites]
    np.testing.assert_allclose((positive(modes) - modes * eigenvalues).kernel, 0.0, atol=1e-10)
    # Against a metric: positive(mode) == eigenvalue * metric(mode).
    eigenvalues, modes = general.adjoint()(general).eigh(positive)
    np.testing.assert_allclose((general.adjoint()(general(modes)) - positive(modes) * eigenvalues).kernel, 0.0, atol=1e-9)
    lower = positive.cholesky()
    np.testing.assert_allclose(lower(lower.adjoint()).kernel, positive.kernel, atol=1e-10)
    left, singular, right = general.svd()
    np.testing.assert_allclose((general(right) - left * singular).kernel, 0.0, atol=1e-10)


def test_sums_over_sites_reduce_the_output_and_leave_the_batch():
    values = mv.vector(rng.normal(size=(copies, sites, 3))).field()         # [copies] Vector[sites]
    np.testing.assert_allclose(values.sites.sum().kernel, values.batch().sum(axis=-1).kernel, atol=1e-12)
    np.testing.assert_allclose(values.sites.mean().kernel, values.batch().mean(axis=-1).kernel, atol=1e-12)
    # A map into a field, summed over its output's sites, is one map over blades.
    into = ctx.extensor(VGA3D.gatype((Vector, Vector)), rng.normal(size=(sites, 3, 3))).field()   # Vector[sites] <- Vector
    point = mv.vector(rng.normal(size=3))
    np.testing.assert_allclose(into.sites.sum()(point).kernel, into(point).sites.sum().kernel, atol=1e-12)
