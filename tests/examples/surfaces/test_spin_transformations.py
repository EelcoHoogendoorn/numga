"""Spin transformations: both scenes pass their checks, the deformation converges to a conformal
one under refinement, and the figures draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.spin_transformations import core, render, scenarios


def test_conformal_error_falls_with_refinement():
    """The discretization is first order: each split of the faces roughly halves the largest change
    in a corner's cosine."""
    errors = []
    for levels in (2, 3):
        mesh = core.icosphere(levels)
        centroids = core.as_ga_sparse(mesh.faces, core.as_scalar(np.ones_like(mesh.faces) / 3)) * mesh.vertices
        deformed = core.spin_transform_deform(mesh, (centroids | core.context.multivector.y) * 0.5)
        errors.append(np.abs((deformed.corner_cosines() - mesh.corner_cosines()).to_array()).max())
    assert errors[1] < 0.6 * errors[0]


def test_scenes_and_figures():
    figure = render.draw_dipole(*scenarios.dipole())
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    mesh, gallery = scenarios.dirac()
    figure = render.draw_dirac(gallery, mesh)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    frames = render.animate_rounding(scenarios.rounding())
    assert len(frames) == scenarios.STEPS + 1
