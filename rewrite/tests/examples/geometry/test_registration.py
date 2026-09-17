"""Unit tests for orientation estimation from an open sandwich."""

from __future__ import annotations

import numpy as np

from numga.expression import scalar_product
from numga.gatype.traits import Versor

from examples.geometry.registration import Alignment, main
from examples.geometry.registration_plumbing import ga, ctx, Rotor, cloud, jitter, mv, same_rotor

def best_rotor(alignment):
    values, vectors = ((alignment + alignment.transpose()) * .5).eigh()
    return values[-1].kernel.item(), vectors[-1].normalized()


TRUTH = (mv.xy * 0.4 + mv.yz * -0.3 + mv.zx * 0.7).exp()


def test_alignment_form_is_bilinear_in_the_rotor():
    rng = np.random.default_rng(1)
    source = cloud(20, rng)
    form = (TRUTH >> source).scalar_product(Rotor.sandwich(source)).sum()
    assert form.gatype == Alignment
    assert form.kernel.shape == (1, 4, 4)
    N = form.kernel.squeeze()
    np.testing.assert_allclose(N, N.T, atol=1e-12)


def test_rotor_unit_form_is_euclidean():
    # A rotor mixes grades 0 and 2, so r̃ | r keeps cross terms; the unit condition is the scalar part.
    unit = np.asarray(ctx.lower(scalar_product(ga.operator.reverse(Rotor.output_subspace), Rotor)).kernel)
    np.testing.assert_allclose(unit.squeeze(), np.eye(4))


def test_exact_correspondences_recover_the_rotor_exactly():
    rng = np.random.default_rng(2)
    source = cloud(30, rng)
    _, estimate = best_rotor((TRUTH >> source).scalar_product(Rotor.sandwich(source)).sum())
    assert same_rotor(estimate, TRUTH, atol=1e-12)


def test_noisy_correspondences_recover_the_rotor_within_noise():
    rng = np.random.default_rng(3)
    source = cloud(200, rng)
    target = jitter(TRUTH >> source, 0.05, rng)
    _, estimate = best_rotor(target.scalar_product(Rotor.sandwich(source)).sum())
    assert same_rotor(estimate, TRUTH, atol=0.01)


def test_estimate_matches_svd_procrustes():
    """The eigenvector rotor agrees with the orthogonal Procrustes rotation from an SVD."""
    rng = np.random.default_rng(4)
    source = cloud(100, rng)
    target = jitter(TRUTH >> source, 0.05, rng)
    _, estimate = best_rotor(target.scalar_product(Rotor.sandwich(source)).sum())
    U, _, Vt = np.linalg.svd(target.kernel.T @ source.kernel)
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    np.testing.assert_allclose((estimate >> source).kernel, source.kernel @ R.T, atol=1e-10)


def test_estimate_is_a_trusted_unit_rotor():
    rng = np.random.default_rng(5)
    source = cloud(30, rng)
    _, estimate = best_rotor((TRUTH >> source).scalar_product(Rotor.sandwich(source)).sum())
    assert estimate.gatype == Rotor
    np.testing.assert_allclose(estimate.reverse().scalar_product(estimate).kernel, 1.0, atol=1e-12)


def test_scale_reads_off_the_eigenvalue():
    rng = np.random.default_rng(6)
    source = cloud(100, rng)
    for s in (0.5, 2.0):
        target = jitter((TRUTH >> source) * s, 0.02, rng)
        peak_alignment, rotor = best_rotor(target.scalar_product(Rotor.sandwich(source)).sum())
        scale = peak_alignment / float((source | source).sum().kernel.item())
        assert np.isclose(scale, s, atol=0.01)
        assert same_rotor(rotor, TRUTH, atol=0.01)
        similarity = rotor * mv.scalar(np.array([np.sqrt(scale)])).with_traits(Versor)
        assert np.linalg.norm((target - (similarity >> source)).kernel, axis=-1).mean() < 0.05
        np.testing.assert_allclose(similarity.log().kernel[0], 0.5 * np.log(s), atol=0.01)


def test_tutorial_runs_and_saves(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    out = tmp_path / "registration.png"
    main(plot_path=str(out))
    assert out.exists()
