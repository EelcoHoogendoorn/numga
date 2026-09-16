"""Orientation estimation from point correspondences: Horn's method from an open sandwich.

Given points p and their rotated, noisy images q, the best rotor maximises the alignment
Σ q · (r p r̃). The sandwich is bilinear in r, so with both rotor slots open that sum is a
quadratic form in r, and the best unit rotor is its largest eigenvector. Nothing is
derived by hand: the 4x4 matrix Horn wrote out entry by entry is the form's kernel, and
the eigenvalue next to the eigenvector is the scale.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga.gatype.traits import Versor

from examples import PLOT_DIR
from examples.geometry.registration_plumbing import (
    ga,
    Rotor,
    best_rotor,
    cloud,
    jitter,
    mv,
    new_figure,
    render_registration,
    same_rotor,
)

Scalar = ga.gatype.scalar()
Alignment = ga.gatype((Scalar, Rotor, Rotor))    # q · (r p r̃) <= (r, r)


def main(plot_path: str = str(PLOT_DIR / "registration.png")) -> plt.Figure:
    """Estimate a rotation, then a similarity, from noisy correspondences."""
    rng = np.random.default_rng(0)

    # -----------------------------------------------------------------------
    # 1. Rotation: leave both rotor slots open
    # -----------------------------------------------------------------------
    # The sandwich r p r̃ has r on both sides. With a rotor type in the sandwicher slot
    # and the source points bound, it is a map bilinear in r, one per point. Contracting
    # its output with the target point and summing gives the alignment as a form in r.
    source = cloud(60, rng)
    truth = (mv.xy * 0.4 + mv.yz * -0.3 + mv.zx * 0.7).exp()
    target = jitter(truth >> source, 0.02, rng)

    alignment: Alignment = target.scalar_product(Rotor.sandwich(source)).sum()

    # Maximising r N r subject to r r̃ = 1 is an eigenproblem. A rotor's reverse product is
    # the Euclidean norm of its four coefficients, so the largest eigenvector is the unit
    # rotor, and there is no constraint to handle.
    _, estimate = best_rotor(alignment)
    assert same_rotor(estimate, truth, atol=0.01)
    residual = np.linalg.norm((target - (estimate >> source)).kernel, axis=-1)
    assert residual.mean() < 0.05

    # -----------------------------------------------------------------------
    # 2. Scale: read it off the eigenvalue
    # -----------------------------------------------------------------------
    # At the optimum the eigenvalue is the alignment itself, Σ q · (R p R̃), and the least
    # squares scale is that divided by Σ p · p. Folding the scale into the rotor gives a
    # single even element, √s R, whose sandwich is the similarity. A float multiplier
    # cannot promise it is nonzero, so it would strip the versor fact; a scalar asserted
    # as a versor keeps it, and the log then carries the log-scale next to the rotation.
    scaled = jitter((truth >> source) * 1.7, 0.02, rng)
    peak_alignment, rotor = best_rotor(scaled.scalar_product(Rotor.sandwich(source)).sum())
    scale = peak_alignment / float((source | source).sum().kernel.item())
    similarity = rotor * mv.scalar(np.array([np.sqrt(scale)])).with_traits(Versor)
    np.testing.assert_allclose(scale, 1.7, atol=0.01)
    np.testing.assert_allclose(similarity.log().kernel[0], 0.5 * np.log(1.7), atol=0.01)
    assert np.linalg.norm((scaled - (similarity >> source)).kernel, axis=-1).mean() < 0.05

    # -----------------------------------------------------------------------
    # 3. Translation: centre first
    # -----------------------------------------------------------------------
    # A translation adds nothing to the form once both clouds are centred, so the
    # similarity is estimated from the centred clouds and the translation is what is left
    # between the centroids after applying it.
    offset = mv.vector(np.array([1.5, -0.5, 2.0]))
    moved = jitter((truth >> source) * 1.7 + offset, 0.02, rng)
    source_centred = source - source.mean(axis=0)
    moved_centred = moved - moved.mean(axis=0)

    peak_alignment, rotor = best_rotor(moved_centred.scalar_product(Rotor.sandwich(source_centred)).sum())
    similarity = rotor * np.sqrt(peak_alignment / float((source_centred | source_centred).sum().kernel.item()))
    translation = moved.mean(axis=0) - (similarity >> source.mean(axis=0))
    assert same_rotor(rotor, truth, atol=0.01)
    np.testing.assert_allclose(translation.kernel, offset.kernel, atol=0.05)

    # -----------------------------------------------------------------------
    # 4. Draw
    # -----------------------------------------------------------------------
    fig, axes = new_figure()
    render_registration(axes[0], source, target, estimate >> source, "Rotation")
    render_registration(axes[1], source, moved, (similarity >> source) + translation, "Rotation, scale and translation")
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
