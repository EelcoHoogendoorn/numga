"""Scenes for the spin groups: the table of signatures up to six dimensions, a four-dimensional rotor
split into its two isoclinic factors, and the orbits of the two isoclinic flows on the three-sphere."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.math.spin_groups import core

mv = core.mv
SIGNATURES = ((3, 0), (2, 1), (4, 0), (3, 1), (2, 2), (5, 0), (4, 1), (3, 2), (6, 0), (5, 1), (4, 2), (3, 3))


# --- math -----------------------------------------------------------------------------
def zoo(signatures: tuple[tuple[int, int], ...], seed: int) -> Iterator[tuple]:
    """For each signature: the invariant form against the inner product on a sample bivector, the
    involution's eigenvalues, and the square of the pseudoscalar."""
    rng = np.random.default_rng(seed)
    for p, q in signatures:
        Bivector, Even, positives, pseudoscalar = core.signature(p, q)
        form = core.invariant_form(Bivector)                                   # [] Scalar <- (Bivector, Bivector), exact
        sample = mv(Bivector.output_subspace, rng.normal(size=len(Bivector.output_subspace.masks)))   # [] Bivector
        factor = form.materialize(core.context)(sample, sample) / (sample | sample)   # [] Scalar
        signs = core.involution(Bivector, positives).eigvals()                  # [generators] Scalar
        square = (pseudoscalar * pseudoscalar).select[0]                        # [] Scalar

        # --- checks
        # The invariant form is 2 * (n - 2) times the inner product, exactly; the involution fixes
        # p * (p - 1) // 2 + q * (q - 1) // 2 planes and negates p * q.
        n = p + q
        inner = (Bivector | Bivector).materialize(core.context)
        np.testing.assert_array_equal(form.materialize(core.context).kernel, 2 * (n - 2) * inner.kernel)
        values = np.real(signs.to_array())
        assert (values > 0).sum() == p * (p - 1) // 2 + q * (q - 1) // 2 and (values < 0).sum() == p * q
        yield p, q, factor, signs, square


def centres(signatures: tuple[tuple[int, int], ...]) -> Iterator[tuple]:
    """For each signature of an even number of dimensions: the eigenvalues of the pseudoscalar acting
    on the spinors, plus and minus one when it splits them in two, plus and minus i when it acts as
    the complex unit."""
    for p, q in signatures:
        _, Even, _, pseudoscalar = core.signature(p, q)
        action = (pseudoscalar * Even).eigvals()                               # [spinors] Scalar

        # --- checks
        # The eigenvalues square to the pseudoscalar's square.
        square = (pseudoscalar * pseudoscalar).select[0].to_array()
        np.testing.assert_allclose(action.to_array() ** 2, square, atol=1e-12)
        yield p, q, action


def split(seed: int) -> tuple[core.Extensor, core.Extensor, core.Extensor]:
    """A random rotor of the four Euclidean directions and its two isoclinic factors."""
    Bivector, _, _, pseudoscalar = core.signature(4, 0)
    generator = mv(Bivector.output_subspace, np.random.default_rng(seed).normal(size=6))   # [] Bivector
    along, against = core.isoclinic(generator, pseudoscalar)                   # [] Bivector each
    rotor = generator.exp()                                                    # [] Rotor
    left, right = along.exp(), against.exp()                                   # [] Rotor each

    # --- checks
    # The halves commute, the factors multiply to the rotor, and each factor turns all its planes
    # through one angle.
    np.testing.assert_allclose((along * against - against * along).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((left * right - rotor).kernel, 0.0, atol=1e-8)
    for factor in (left, right):
        angles = np.abs(np.angle((factor >> core.Euclidean).eigvals().to_array()))
        np.testing.assert_allclose(angles, angles[0], atol=1e-9)
    return rotor, left, right


def flows(azimuths: int, heights: np.ndarray, count: int) -> tuple[core.Extensor, core.Extensor, core.Extensor]:
    """The orbits of points of the three-sphere under the two isoclinic flows, turning xy together with
    zw one way and the other, and under a rotation turning xy twice while zw turns three times, the
    two flows combined at different rates, `5 * against - along`, from one start at each height; all
    projected into space."""
    _, _, _, pseudoscalar = core.signature(4, 0)
    # The planes of the two flows: xy turned along with its dual plane, and against it.
    along, against = core.isoclinic(mv.xy, pseudoscalar)                       # [] Bivector each
    # Starting points at a few heights along z, spread around the xy plane.
    phi = np.linspace(0.0, 2 * np.pi, azimuths, endpoint=False)
    around = mv.x * np.cos(phi) + mv.y * np.sin(phi)                           # [azimuths] Vector
    starts = around * np.cos(heights)[:, None] + mv.z * np.sin(heights)[:, None]   # [heights, azimuths] Vector
    # One full turn of every plane: the sandwich turns by twice the rotor's angle.
    turn = np.linspace(0.0, np.pi, count + 1)
    left = core.stereographic((2 * along * turn).exp() >> starts[..., None])  # [heights, azimuths, count + 1] Space
    right = core.stereographic((2 * against * turn).exp() >> starts[..., None])   # [heights, azimuths, count + 1] Space
    knotted = core.stereographic(((2 * mv.xy + 3 * mv.zw) * turn).exp() >> starts[:, :1, None])   # [heights, 1, count + 1] Space

    # --- checks
    # Turning xy twice while zw turns three times is the flow against run five times over and the flow
    # along run once backward. Every orbit closes, and two orbits of one flow link once, with opposite
    # signs for the two flows.
    np.testing.assert_allclose((2 * mv.xy + 3 * mv.zw - (5 * against - along)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((left[..., 0] - left[..., -1]).kernel, 0.0, atol=1e-5)
    np.testing.assert_allclose((knotted[..., 0] - knotted[..., -1]).kernel, 0.0, atol=1e-5)
    forward = core.linking(left[0, 0], left[-1, azimuths // 3]).to_array()
    backward = core.linking(right[0, 0], right[-1, azimuths // 3]).to_array()
    np.testing.assert_allclose(np.abs([forward, backward]), 1.0, atol=1e-2)
    assert np.sign(forward) == -np.sign(backward)
    return left, right, knotted


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.math.spin_groups import render

    print(render.table(zoo(SIGNATURES, 0), centres(tuple(s for s in SIGNATURES if sum(s) % 2 == 0))))
    orbits = flows(12, np.array([0.25, 0.6, 1.0]), 240)
    save_figure(render.draw_flows(*orbits), "spin_groups_isoclinic")
    save_animation(render.animate_flows(*orbits, 80), "spin_groups_isoclinic", 60)
