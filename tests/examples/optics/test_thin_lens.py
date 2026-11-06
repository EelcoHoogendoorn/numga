"""The optical train's imaging, and the thin lens equation."""

from __future__ import annotations

import numpy as np

from examples.optics.thin_lens import render, scenarios


def test_train_images_the_subject_through_every_placement():
    """The composed train maps the fan onto the last leg, and every output ray passes through the image."""
    for subject, planes, legs, train, picture in scenarios.train(12):
        np.testing.assert_allclose((train(legs[0]) ^ picture).kernel, 0.0, atol=1e-10)
        np.testing.assert_allclose((train(legs[0]) - legs[-1]).kernel, 0.0, atol=1e-10)


def test_one_lens_images_by_the_lens_equation():
    """The first lens's image lies where the thin lens equation puts it."""
    _, image, _, _, planes = scenarios.lenses()
    subject = scenarios.subject
    plane = planes[0]
    distances = (-(plane & subject) / (scenarios.mv.w & subject), (plane & image) / (scenarios.mv.w & image))
    np.testing.assert_allclose((1 / distances[0] + 1 / distances[1]).to_array(), 1 / scenarios.first_focal, atol=1e-12)


def test_one_frame_of_the_train_draws():
    next(render.animate_train(scenarios.train(1)))
