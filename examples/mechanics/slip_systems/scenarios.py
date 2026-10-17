"""Which slip system a pull along any direction makes yield first in a face-centred cubic crystal, and the paths
the pull's axis takes through the crystal as it slips."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.mechanics.slip_systems import core

# The normals of the four slip planes, the {111} planes.
NORMALS = [[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]
# The samples across the stereographic disk of pull directions.
GRID = 1000
# The axes the pulls start along, spread over the standard triangle between [001], [011] and [111] away from
# [001], where eight systems nearly tie; the slip of a step on the leading system, and how steeply a system's
# slip falls off below the leading factor, so steeply that the leading system takes almost all of it and two systems with equal factors
# share it alike; the steps taken; and the frames drawn, the duration of each.
STARTS = [[0.05, 0.3, 1], [0.15, 0.32, 1], [0.03, 0.5, 1], [0.2, 0.5, 1], [0.05, 0.75, 1],
          [0.3, 0.6, 1], [0.12, 0.62, 1], [0.4, 0.55, 1]]
AMOUNT = 0.01
STEEPNESS = 30
STEPS = 500
FRAMES = 50
DURATION_MS = 70


# --- math -----------------------------------------------------------------------------
def chart() -> tuple[core.Vector, core.Scalar, core.Vector, core.Vector]:
    """Pull directions over the upper half of the sphere, laid out on the stereographic disk `[rows, columns]`,
    each system's Schmid factor for each `[rows, columns, planes, directions]`, and the slip planes' normals
    and slip directions."""
    # Each point of the chart, the plane across [001], lifts to the direction that projects onto it: [001]
    # reflected in the line along [001] plus the point.
    across = np.linspace(-1, 1, GRID)
    points = core.mv.x * across + core.mv.y * across[:, None]                  # [rows, columns] Vector
    directions = (core.mv.z + points).normalized() >> core.mv.z               # [rows, columns] Vector
    normals = core.mv.vector(NORMALS).normalized()                            # [planes] Vector
    slips = core.slip_directions(normals)                                     # [planes, directions] Vector
    return directions, core.schmid(directions, normals, slips), normals, slips


def rotation(axis: core.Vector, normals: core.Vector, slips: core.Vector) -> Iterator[tuple[core.Vector, core.Scalar]]:
    """Each pull's axis after every slip step, and how much each system slipped in the step: every system
    slips in the sense the pull drives it, at a rate rising steeply with its Schmid factor."""
    for _ in range(STEPS):
        amounts = power_law(core.schmid(axis, normals, slips))                 # [starts, planes, directions] Scalar
        axis = core.slipped(axis, normals, slips, amounts)
        yield axis, amounts                                                   # [starts] Vector


def paths() -> tuple[core.Vector, core.Scalar]:
    """The pulls' axes at every step `[steps + 1, starts]`, and how much each system slipped in each step
    `[steps, starts, planes, directions]`."""
    normals = core.mv.vector(NORMALS).normalized()                            # [planes] Vector
    starts = core.mv.vector(STARTS).normalized()                              # [starts] Vector
    axes, amounts = zip(*rotation(starts, normals, core.slip_directions(normals)))
    return stack([starts, *axes]), stack(amounts)


# --- the step outside the algebra ---------------------------------------------------------
def power_law(factors: core.Scalar) -> core.Scalar:
    """How much each system slips in a step `[..., planes, directions]`: the step's slip, in the sign of the
    Schmid factor, scaled by the factor's size over the leading one raised to the steepness, a power law of
    the resolved shear stress."""
    values = factors.cast(core.ga.subspace.scalar()).kernel[..., 0]
    leading = np.abs(values).max(axis=(-2, -1), keepdims=True)
    return factors * (AMOUNT * np.abs(values) ** (STEEPNESS - 1) / leading ** STEEPNESS)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.mechanics.slip_systems import render

    directions, factors, normals, slips = chart()
    axes, amounts = paths()
    save_figure(render.draw_chart(directions, factors, normals, slips, axes, amounts), "slip_systems")
    save_animation(render.animate_paths(directions, factors, axes, amounts, FRAMES), "slip_systems_paths", DURATION_MS)

    # checks
    values = render.scalars(factors)                                          # [rows, columns, planes, directions]
    # No Schmid factor exceeds a half, and along [001] eight systems share the largest, one over the root of six.
    assert np.abs(values).max() <= 0.5 + 1e-12
    along = np.abs(render.scalars(core.schmid(core.mv.z, normals, slips)))   # [planes, directions]
    assert np.isclose(along.max(), 1 / np.sqrt(6)) and np.isclose(along, along.max()).sum() == 8
    # Every pull's axis ends at [112], two systems carrying nearly all of its slip on the way.
    ends = render.coordinates(axes[-1])                                        # [starts, 3]
    target = np.array([1, 1, 2]) / np.sqrt(6)
    assert (np.sort(ends, axis=-1) @ target > 0.999).all()
    carried = np.sort(np.abs(render.scalars(amounts)).sum(axis=0).reshape(len(STARTS), -1), axis=-1)   # [starts, systems]
    assert (carried[:, -2:].sum(axis=-1) / carried.sum(axis=-1) > 0.95).all()


if __name__ == "__main__":
    main()
