"""Signature sweep, four-dimensional isoclinic split, and three-sphere flow scenarios for spin groups."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.math.spin_groups import core

mv = core.mv
signatures = ((3, 0), (2, 1), (4, 0), (3, 1), (2, 2), (5, 0), (4, 1), (3, 2), (6, 0), (5, 1), (4, 2), (3, 3))
seed = 0
# Starting orbits form a ring of azimuths around the xy plane at each height angle along z.
n_azimuths = 12
heights = np.array([0.25, 0.6, 1.0])
n_steps = 240
# One animation frame every two steps along the turn at 40 ms per frame.
n_frames = n_steps // 2
frame_duration_ms = 40


def signature_rows(signatures: tuple[tuple[int, int], ...], seed: int) -> Iterator[tuple]:
    """Yields `(p, q, form, ratio, signs, square)` across `signatures`: the exact invariant form,
    its ratio to the inner product on a random bivector, the involution's eigenvalues, and the
    pseudoscalar's square."""
    rng = np.random.default_rng(seed)
    for p, q in signatures:
        Bivector, Even, positives, pseudoscalar = core.signature(p, q)
        n_planes = len(Bivector.output_subspace.masks)
        form = core.invariant_form(Bivector)                                    # [] Scalar+ <- (Bivector+, Bivector+)
        sample = mv(Bivector, rng.normal(size=n_planes))                        # [] Bivector+
        ratio = form.materialize(core.context)(sample, sample) / (sample | sample)  # [] Scalar+
        involution = positives >> Bivector                                      # [] Bivector+ <- Bivector+
        signs = involution.eigvals()                                            # [n_planes] Scalar+
        yield p, q, form, ratio, signs, pseudoscalar.squared()


def spinor_centres(signatures: tuple[tuple[int, int], ...]) -> Iterator[tuple]:
    """Yields `(p, q, action)` across even-dimensional `signatures`: the eigenvalues of the
    pseudoscalar acting on the spinors (`[n_spinors] Scalar+`)."""
    for p, q in signatures:
        _, Even, _, pseudoscalar = core.signature(p, q)
        n_spinors = len(Even.output_subspace.masks)
        action = (pseudoscalar * Even).eigvals()                                # [n_spinors] Scalar+
        yield p, q, action


def flow_steps(n_steps: int, n_frames: int) -> Iterator[int]:
    """Yields the `n_frames` evenly spaced step indices along a closed orbit of `n_steps` segments."""
    for step in np.linspace(0, n_steps, n_frames, endpoint=False).astype(int):
        yield int(step)


def main() -> None:
    from examples.math.spin_groups import render
    from examples.util.animation import save_animation, save_figure

    rows = list(signature_rows(signatures, seed))
    even_signatures = tuple(pair for pair in signatures if sum(pair) % 2 == 0)
    actions = list(spinor_centres(even_signatures))
    for p, q, _, ratio, signs, square in rows:
        print(
            (p, q),
            "form / inner:", ratio.to_array(),
            " involution eigenvalues:", signs.real().to_array(),
            " pseudoscalar squared:", square.to_array(),
        )
    for p, q, action in actions:
        print((p, q), "pseudoscalar eigenvalues on spinors:", action.to_array())

    Bivector_4d, _, _, _ = core.signature(4, 0)
    generator = mv(Bivector_4d, np.random.default_rng(seed).normal(size=len(Bivector_4d.output_subspace.masks)))  # [] Bivector+
    rotor = generator.exp()                                                     # [] Rotor+
    left_factor, right_factor = core.isoclinic_factors(generator)               # [] Rotor+, Rotor+
    left_angles = (left_factor >> core.Euclidean).eigvals().real().arccos()     # [4] Scalar+
    right_angles = (right_factor >> core.Euclidean).eigvals().real().arccos()   # [4] Scalar+

    left_orbits, right_orbits, knotted_orbits = core.isoclinic_orbits(n_azimuths, heights, n_steps)
    linking_along = core.linking(left_orbits[0, 0], left_orbits[-1, n_azimuths // 3])     # [] Antitrivector-
    linking_against = core.linking(right_orbits[0, 0], right_orbits[-1, n_azimuths // 3])  # [] Antitrivector-

    save_figure(render.draw_flows(left_orbits, right_orbits, knotted_orbits), "spin_groups_isoclinic")
    save_animation(
        render.animate_flows(left_orbits, right_orbits, knotted_orbits, flow_steps(n_steps, n_frames)),
        "spin_groups_isoclinic",
        frame_duration_ms,
    )

    # checks
    # Across every signature, the invariant form equals 2 * (n - 2) times the inner product and the
    # involution fixes p * (p - 1) // 2 + q * (q - 1) // 2 rotation planes while negating p * q boost planes.
    for p, q, form, _, signs, _ in rows:
        Bivector, _, _, _ = core.signature(p, q)
        planes = mv(Bivector, np.eye(len(Bivector.output_subspace.masks)))
        difference = (form - 2 * (p + q - 2) * (Bivector | Bivector)).materialize(core.context)(
            planes[:, None], planes[None, :],
        )
        assert (difference.abs() <= 0).all()
        real_signs = signs.real()
        assert (real_signs > 0).sum() == p * (p - 1) // 2 + q * (q - 1) // 2
        assert (real_signs < 0).sum() == p * q
    # Every spinor eigenvalue of the even-dimensional pseudoscalar squares to the pseudoscalar's own square.
    squares = {(p, q): square for p, q, _, _, _, square in rows}
    for p, q, action in actions:
        assert ((action.squared() - squares[p, q]).abs() < 1e-12).all()
    # The four-dimensional rotor factors into two commuting isoclinic halves, each turning every plane by one angle.
    assert (left_factor * right_factor - rotor).scalar_norm_squared() < 1e-12
    assert (left_factor * right_factor - right_factor * left_factor).scalar_norm_squared() < 1e-12
    assert ((left_angles - left_angles[0]).abs() < 1e-9).all()
    assert ((right_angles - right_angles[0]).abs() < 1e-9).all()
    # Closed orbits return to their starts, opposite Hopf fibres link with unit volume of opposite sign,
    # and 2 * xy + 3 * zw decomposes into 5 * against - along.
    along, against = core.plus * mv.xy, core.minus * mv.xy
    assert (2 * mv.xy + 3 * mv.zw - (5 * against - along)).scalar_norm_squared() < 1e-12
    assert ((left_orbits[..., 0] - left_orbits[..., -1]).antinorm() < 1e-5).all()
    assert ((knotted_orbits[..., 0] - knotted_orbits[..., -1]).antinorm() < 1e-5).all()
    assert ((mv.xyz & linking_along).abs() - 1).abs() < 1e-2
    assert (linking_along + linking_against).antinorm() < 1e-2


if __name__ == "__main__":
    main()
