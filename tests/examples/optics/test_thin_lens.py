"""Thin-lens imaging, Gullstrand focusing, and moving-train concurrency and rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.optics.thin_lens import render, scenarios


def test_moving_train_keeps_rays_concurrent():
    """The composed train matches the final bundle leg and brings every outgoing ray through the image crossing."""
    for _, _, legs, train, picture in scenarios.moving_train_states(4):
        assert ((picture & train(legs[0])).abs() < 1e-10).all()
        assert ((train(legs[0]) - legs[-1]).norm() < 1e-10).all()


def test_single_lens_satisfies_thin_lens_equation_and_draws():
    """The static lens systems satisfy the thin-lens equation and render both the static diagram and one train frame."""
    imaging, image, focusing, focus, planes = scenarios.static_lens_systems()
    subject = scenarios.subject
    plane = planes[0]
    object_distance = -(plane & subject) / (scenarios.mv.w & subject)
    image_distance = (plane & image) / (scenarios.mv.w & image)
    assert ((1 / object_distance + 1 / image_distance - 1 / scenarios.first_focal).abs() < 1e-12).all()
    plt.close(render.lenses_figure(subject, imaging, image, focusing, focus, planes))
    next(render.train_animation(scenarios.moving_train_states(1)))
