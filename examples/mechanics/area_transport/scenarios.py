"""A sheared cube approaching a flat sheet, including the singular endpoint."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.mechanics.area_transport import core

SIDE = 1.0
# The deformations compared: the solid cube, compressed and sheared, and the flat sheet.
MAX_SHEAR = 0.70
THICKNESSES = np.array([1.0, 0.35, 0.0])
SHEARS = np.array([0.0, 0.45, MAX_SHEAR])
# The frames of one collapse and recovery, and the duration of each.
N_FRAMES = 40
DURATION_MS = 70


# --- math -----------------------------------------------------------------------------
def scene(thicknesses: np.ndarray, shears: np.ndarray) -> tuple[core.Point, core.Point, core.Patch, core.Scalar]:
    """The cube carried by each deformation: its corners, face centres and oriented face patches, and the
    volume factor."""
    deformation = core.deformation(thicknesses, shears)                       # [...] Point <- Point
    areas, _, volume = core.transport(deformation)
    corners, centres, patches = core.cube(SIDE)
    return deformation[..., None](corners), deformation[..., None](centres), areas[..., None](patches), volume


def collapse(n_frames: int) -> Iterator[tuple[core.Point, core.Point, core.Patch, core.Scalar]]:
    """Per frame of a collapse and recovery, the cube's corners, face centres, patches and volume."""
    for phase in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        thickness = np.array((1 + np.cos(phase)) / 2)
        yield scene(thickness, MAX_SHEAR * (1 - thickness))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.mechanics.area_transport import render

    vertices, centres, patches, volume = scene(THICKNESSES, SHEARS)
    save_figure(render.comparison(vertices, centres, patches, volume), "area_transport")
    save_animation(render.animate(collapse(N_FRAMES)), "area_transport", DURATION_MS)

    # checks
    # The deformation and its area map scale every volume by the determinant, at the flat sheet too.
    deformation = core.deformation(THICKNESSES, SHEARS)                       # [cases] Point <- Point
    areas, _, _ = core.transport(deformation)
    volumes = deformation(core.Point) & areas(core.Patch)                     # [cases] Scalar <- (Point, Patch)
    np.testing.assert_allclose((volumes - volume * (core.Point & core.Patch)).kernel, 0, atol=1e-12)


if __name__ == "__main__":
    main()
