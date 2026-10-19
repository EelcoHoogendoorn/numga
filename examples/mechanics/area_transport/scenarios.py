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
def scene(thicknesses: np.ndarray, shears: np.ndarray) -> tuple[core.Vector, core.Vector, core.Bivector, core.Vector, core.Scalar]:
    """The cube carried by each deformation: its corners, face centres, oriented face patches and area
    normals, and the volume factor."""
    deformation = core.deformation(thicknesses, shears)                       # [...] Vector <- Vector
    areas, _, cofactor, volume = core.transport(deformation)
    corners, centres, patches = core.cube(SIDE)
    return (deformation[..., None](corners), deformation[..., None](centres), areas[..., None](patches),
            cofactor[..., None](patches.dual()), volume)


def collapse(n_frames: int) -> Iterator[tuple[core.Vector, core.Vector, core.Vector, core.Scalar]]:
    """Per frame of a collapse and recovery, the cube's corners, face centres, area normals and volume."""
    for phase in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        thickness = np.array((1 + np.cos(phase)) / 2)
        vertices, centres, _, area_normals, volume = scene(thickness, MAX_SHEAR * (1 - thickness))
        yield vertices, centres, area_normals, volume


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.mechanics.area_transport import render

    vertices, centres, patches, area_normals, volume = scene(THICKNESSES, SHEARS)
    save_figure(render.comparison(vertices, centres, area_normals, volume), "area_transport")
    save_animation(render.animate(collapse(N_FRAMES)), "area_transport", DURATION_MS)

    # checks
    # The carried area normals are the carried patches' duals, and the deformation and its area map scale
    # every volume by the determinant, at the flat sheet too.
    np.testing.assert_allclose((area_normals - patches.dual()).kernel, 0, atol=1e-12)
    deformation = core.deformation(THICKNESSES, SHEARS)                       # [cases] Vector <- Vector
    areas, _, _, _ = core.transport(deformation)
    volumes = deformation(core.Vector) ^ areas(core.Bivector)                 # [cases] Pseudoscalar <- (Vector, Bivector)
    np.testing.assert_allclose((volumes - volume * (core.Vector ^ core.Bivector)).kernel, 0, atol=1e-12)


if __name__ == "__main__":
    main()
