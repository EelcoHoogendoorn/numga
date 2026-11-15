"""Scene parameters and figure/animation generation for gravitational lensing by a binary mass."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from examples.relativity.gravitational_lensing import core

# Two equal half-mass lenses separated by 1.1 Einstein angles along the x axis, producing a six-cusped caustic.
n_masses = 2
positions = core.mv.x.dual() * np.array([-0.55, 0.55])                         # [n_masses] Point-
masses = np.array([0.5, 0.5])                                                  # [n_masses]
# Sky sampling half-width and pixel count per side used to trace the critical curve and caustic.
sky_half_width = 1.9
n_pixels = 480
# Source path crossing the caustic 0.12 Einstein angles above the binary axis, still index, width, and frame duration.
n_steps = 145
path = (core.mv.x * np.linspace(-0.6, 0.6, n_steps) + core.mv.y * 0.12).dual()   # [n_steps] Point-
shown_step = 47
source_width = 0.035
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def binary_source_direction(observed: core.Point) -> core.Point:
    """The two binary masses deflect each `observed` sky direction (`[...] Point-`) to its source direction (`[...] Point-`)."""
    return core.source_direction(observed, positions, masses)                  # [...] Point-


def binary_lens_grid(n_grid_pixels: int) -> tuple[core.Point, core.Scalar, core.LocalMap]:
    """The binary lens over an `[n_grid_pixels, n_grid_pixels]` sky grid determines pixel directions, signed area ratios, and local Jacobians."""
    directions = core.sky_grid(sky_half_width, n_grid_pixels)                  # [n_grid_pixels, n_grid_pixels] Point-
    local = core.local_map(directions, positions, masses)                      # [n_grid_pixels, n_grid_pixels] Point- <- Point-
    area = local.det()                                                         # [n_grid_pixels, n_grid_pixels] Scalar+
    return directions, area, local


# --- plumbing -------------------------------------------------------------------------
def save_states(
    states: Iterable,
    animate: Callable[..., Iterable[np.ndarray]],
    *arguments,
    name: str,
    duration_ms: int,
) -> list:
    """Streams `states` once through `animate` into `save_animation` and returns the collected states."""
    from examples.util.animation import save_animation

    recorded = []

    def passing() -> Iterator:
        for state in states:
            recorded.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms)
    return recorded


def main() -> None:
    from examples.relativity.gravitational_lensing import render
    from examples.util.animation import save_figure

    directions, area, local = binary_lens_grid(n_pixels)
    exposures = save_states(
        core.path_exposures(path, positions, masses, source_width),
        render.lensing_animation,
        directions,
        area,
        binary_source_direction,
        positions,
        name="gravitational_lensing",
        duration_ms=frame_duration_ms,
    )
    fig = render.lensing_figure(directions, area, binary_source_direction, positions, exposures[shown_step])
    save_figure(fig, "gravitational_lensing")
    import matplotlib.pyplot as plt
    plt.close(fig)

    # checks
    # Planar reflections are traceless, so the local map has trace 2 everywhere and its trace-2 inverse formula
    # `(2 * Point - local) / area` recovers the identity on every pixel away from the critical curve.
    assert ((local.trace() - 2.0).abs() < 1e-11).all()
    probe = core.mv.x.dual()                                                   # Point-
    recovered = ((2 * core.Point - local) / area)(local(probe))                # [n_pixels, n_pixels] Point-
    away_from_critical = area.abs() > 1e-2
    assert (((recovered - probe).scalar_norm_squared() < 1e-18) | ~away_from_critical).all()
    assert len(exposures) == n_steps


if __name__ == "__main__":
    main()
