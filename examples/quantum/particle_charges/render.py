"""Charge readouts and generator connections between the sixteen particle sectors."""

from __future__ import annotations

from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D

from examples.quantum.particle_charges.core import Scalar

BACKGROUND = "#fbfaf7"
PARTICLE_COLOURS = np.array(["#d6453d", "#3a9a5b", "#2f6fc0", "#d9a227"])
CONNECTION_COLOURS = ("#4a9ea3", "#8a6fc4", "#db663d")


# --- plumbing
def charge_table(
    occupations: np.ndarray, charge: Scalar, hypercharge: Scalar, left_isospin: Scalar,
) -> str:
    """The charges and colour multiplicities of the eight particle multiplets."""
    n_colour = occupations[:, :3].sum(axis=-1)
    n_weak = occupations[:, 3:].sum(axis=-1)
    up = occupations[:, 3] == 1
    groups = (
        ("neutrino L", (n_colour == 3) & up),
        ("electron L", (n_colour == 3) & ~up),
        ("up L", (n_colour == 1) & up),
        ("down L", (n_colour == 1) & ~up),
        ("neutrino R^c", (n_colour == 0) & (n_weak == 0)),
        ("electron R^c", (n_colour == 0) & (n_weak == 2)),
        ("up R^c", (n_colour == 2) & (n_weak == 0)),
        ("down R^c", (n_colour == 2) & (n_weak == 2)),
    )
    readings = np.stack((charge.to_array(), hypercharge.to_array(), left_isospin.to_array()), axis=-1)
    rows = [f"{'field':16s} {'states':>6s} {'Q':>7s} {'Y':>7s} {'T3_L':>7s}"]
    for name, selected in groups:
        values = readings[selected].mean(axis=0)
        fractions = (str(Fraction(float(value)).limit_denominator(12)) for value in values)
        rows.append(f"{name:16s} {selected.sum():6d}" + "".join(f"{value:>8s}" for value in fractions))
    return "\n".join(rows)


def draw_connections(
    occupations: np.ndarray,
    left_isospin: Scalar,
    right_isospin: Scalar,
    colour_weights: Scalar,
    strengths: Scalar,
) -> plt.Figure:
    """Weight diagrams whose edges are the computed actions between particle sectors."""
    left = left_isospin.to_array()
    right = right_isospin.to_array()
    weights = colour_weights.to_array()
    responses = strengths.to_array()
    n_panels = responses.shape[0]

    # The two weak charges separate the four multiplets; colour weights separate their states.
    centres = 4 * np.stack((left, right), axis=-1)
    offsets = Affine2D().rotate_deg(-12).scale(0.85).transform(weights)
    positions = centres + offsets
    n_colour = occupations[:, :3].sum(axis=-1)
    occupied_colour = np.argmax(occupations[:, :3], axis=-1)
    missing_colour = np.argmin(occupations[:, :3], axis=-1)
    colour_indices = np.where(n_colour == 1, occupied_colour, missing_colour)
    colour_indices = np.where((n_colour == 0) | (n_colour == 3), 3, colour_indices)
    colours = PARTICLE_COLOURS[colour_indices]
    doublets = np.abs(left) > 1e-10
    faces = np.where(doublets, colours, BACKGROUND)

    figure, axes = plt.subplots(
        1, n_panels, squeeze=False, figsize=(3.7 * n_panels, 3.7), facecolor=BACKGROUND,
    )
    extent = np.abs(positions).max() + 0.35
    for panel, ax in enumerate(axes[0]):
        response = responses[panel]
        source, target = np.nonzero(np.triu(response > 1e-10, k=1))
        segments = np.stack((positions[source], positions[target]), axis=1)
        widths = 2.4 * np.sqrt(response[source, target] / response.max())
        ax.add_collection(LineCollection(
            segments, colors=CONNECTION_COLOURS[panel], linewidths=widths, alpha=0.82, zorder=1,
        ))
        ax.scatter(
            *positions.T, facecolors=faces, edgecolors=colours, s=78, linewidths=1.8, zorder=3,
        )
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.035)
    plt.close(figure)
    return figure
