"""The spin groups: the table of signatures up to six dimensions, a four-dimensional rotor
split into its two isoclinic factors, and the orbits of the two isoclinic flows on the three-sphere."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


from examples.math.spin_groups import core

mv = core.mv
signatures = ((3, 0), (2, 1), (4, 0), (3, 1), (2, 2), (5, 0), (4, 1), (3, 2), (6, 0), (5, 1), (4, 2), (3, 3))
# The seed of the sample bivectors and of the random rotor.
seed = 0
# The starts of the orbits: a ring of azimuths around the xy plane at each height along z, and the
# steps along one full turn.
n_azimuths = 12
heights = np.array([0.25, 0.6, 1.0])
n_steps = 240
# The frames of the animation, one per two steps along the turn, and the time per frame.
n_frames = n_steps // 2
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def zoo(signatures: tuple[tuple[int, int], ...], seed: int) -> Iterator[tuple]:
    """For each signature: the invariant form, exact in integers, and its ratio to the inner product
    on a sample bivector, the involution's eigenvalues, and the square of the pseudoscalar."""
    rng = np.random.default_rng(seed)
    for p, q in signatures:
        Bivector, Even, positives, pseudoscalar = core.signature(p, q)
        # The trace of the double commutator, a form on pairs of bivectors.
        form = Bivector.commutator(Bivector.commutator(Bivector)).trace(0, 3)  # [] Scalar+ <- (Bivector+, Bivector+)
        sample = mv(Bivector, rng.normal(size=len(Bivector.output_subspace.masks)))   # [] Bivector+
        factor = form.materialize(core.context)(sample, sample) / (sample | sample)   # [] Scalar+
        # The bivectors sandwiched with the product of the positive directions.
        involution = positives >> Bivector                                     # [] Bivector+ <- Bivector+
        yield p, q, form, factor, involution.eigvals(), pseudoscalar.squared()


def centres(signatures: tuple[tuple[int, int], ...]) -> Iterator[tuple]:
    """For each signature of an even number of dimensions: the eigenvalues of the pseudoscalar acting
    on the spinors, plus and minus one when it splits them in two, and none real when it turns every
    pair of their coefficients a quarter turn alike."""
    for p, q in signatures:
        _, Even, _, pseudoscalar = core.signature(p, q)
        action = (pseudoscalar * Even).eigvals()                               # [spinors] Scalar+
        yield p, q, action


def flows(n_azimuths: int, heights: np.ndarray, n_steps: int
          ) -> tuple[core.Euclidean, core.Euclidean, core.Euclidean]:
    """The orbits of points of the three-sphere under the two isoclinic flows, turning xy together with
    zw one way and the other, and under a rotation turning xy twice while zw turns three times, the
    two flows combined at different rates, from one start at each height; all projected into space."""
    # The planes of the two flows: xy turned along with its dual plane, and against it.
    along, against = core.plus * mv.xy, core.minus * mv.xy                     # [] Bivector+, Bivector+
    # Turning xy twice while zw turns three times.
    combined = 2 * mv.xy + 3 * mv.zw                                           # [] Bivector+
    # Starting points at a few heights along z, spread around the xy plane: x turned up toward z by
    # each height, then within the xy plane by each azimuth.
    azimuths = np.linspace(0.0, 2 * np.pi, n_azimuths, endpoint=False)
    azimuth_turns = (mv.xy * (-azimuths / 2)).exp()                            # [azimuths] Rotor+
    height_turns = (mv.xz * (-heights[:, None] / 2)).exp()                     # [heights, 1] Rotor+
    starts = (azimuth_turns * height_turns) >> mv.x.dual()                     # [heights, azimuths] Antivector-
    # One full turn of every plane: the sandwich turns by twice the rotor's angle.
    turn_angles = np.linspace(0.0, np.pi, n_steps + 1)
    along_rotors = (2 * along * turn_angles).exp()                             # [steps + 1] Rotor+
    against_rotors = (2 * against * turn_angles).exp()                         # [steps + 1] Rotor+
    combined_rotors = (combined * turn_angles).exp()                           # [steps + 1] Rotor+
    left = core.stereographic(along_rotors >> starts[..., None])               # [heights, azimuths, steps + 1] Antivector-
    right = core.stereographic(against_rotors >> starts[..., None])            # [heights, azimuths, steps + 1] Antivector-
    # The combined rotation from one start at each height.
    knotted = core.stereographic(combined_rotors >> starts[:, :1, None])       # [heights, 1, steps + 1] Antivector-
    return left, right, knotted


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.math.spin_groups import render

    rows = list(zoo(signatures, seed))
    actions = list(centres(tuple(s for s in signatures if sum(s) % 2 == 0)))
    for p, q, _, ratio, signs, square in rows:
        print((p, q), "form / inner:", ratio.to_array(), " involution's eigenvalues:", signs.to_array(), " pseudoscalar squared:", square.to_array())
    for p, q, action in actions:
        print((p, q), "the pseudoscalar's eigenvalues on the spinors:", action.to_array())
    left, right, knotted = flows(n_azimuths, heights, n_steps)
    save_figure(render.draw_flows(left, right, knotted), "spin_groups_isoclinic")
    save_animation(render.animate_flows(left, right, knotted, n_frames), "spin_groups_isoclinic", frame_duration_ms)

    # checks
    # The invariant form is 2 * (n - 2) times the inner product, exactly; the involution fixes
    # p * (p - 1) // 2 + q * (q - 1) // 2 planes and negates p * q.
    for p, q, form, _, signs, _ in rows:
        Bivector, _, _, _ = core.signature(p, q)
        inner = (Bivector | Bivector).materialize(core.context)
        np.testing.assert_array_equal(form.materialize(core.context).kernel, 2 * (p + q - 2) * inner.kernel)
        values = np.real(signs.to_array())
        assert (values > 0).sum() == p * (p - 1) // 2 + q * (q - 1) // 2 and (values < 0).sum() == p * q
    # The eigenvalues of the pseudoscalar on the spinors square to the pseudoscalar's square.
    squares = {(p, q): square for p, q, _, _, _, square in rows}
    for p, q, action in actions:
        np.testing.assert_allclose(action.to_array() ** 2, squares[p, q].to_array(), atol=1e-12)
    # A random rotor of the four Euclidean directions is the product of the exponentials of the two
    # halves of its generator; the halves commute, and each factor turns all its planes through one angle.
    Bivector, _, _, _ = core.signature(4, 0)
    generator = mv(Bivector, np.random.default_rng(seed).normal(size=len(Bivector.output_subspace.masks)))
    along_generator, against_generator = core.plus * generator, core.minus * generator
    left_factor, right_factor = along_generator.exp(), against_generator.exp()
    np.testing.assert_allclose(along_generator.commutator(against_generator).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((left_factor * right_factor - generator.exp()).kernel, 0.0, atol=1e-8)
    for factor in (left_factor, right_factor):
        angles = np.abs(np.angle((factor >> core.Euclidean).eigvals().to_array()))
        np.testing.assert_allclose(angles, angles[0], atol=1e-9)
    # Turning xy twice while zw turns three times is the flow against run five times over and the flow
    # along run once backward. Every orbit closes, and two orbits of one flow link once, with opposite
    # signs for the two flows.
    along, against = core.plus * mv.xy, core.minus * mv.xy
    np.testing.assert_allclose((2 * mv.xy + 3 * mv.zw - (5 * against - along)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((left[..., 0] - left[..., -1]).kernel, 0.0, atol=1e-5)
    np.testing.assert_allclose((knotted[..., 0] - knotted[..., -1]).kernel, 0.0, atol=1e-5)
    forward = core.linking(left[0, 0], left[-1, n_azimuths // 3])
    backward = core.linking(right[0, 0], right[-1, n_azimuths // 3])
    np.testing.assert_allclose(np.abs((mv.xyz & forward).to_array()), 1.0, atol=1e-2)
    np.testing.assert_allclose((forward + backward).kernel, 0.0, atol=1e-2)


if __name__ == "__main__":
    main()
