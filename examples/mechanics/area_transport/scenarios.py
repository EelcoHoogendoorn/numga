"""A sheared cube approaching a flat sheet, including the singular endpoint."""

from __future__ import annotations

import numpy as np

from examples.mechanics.area_transport import core

SIDE = 1.0
# The crush's frames, from the unit cube to the flat sheet, played forward and back, and the duration of each; the
# slide along x at the flat sheet.
N_FRAMES = 21
DURATION_MS = 70
MAX_SHEAR = 0.70
# The thickness eased from the unit cube's down to the flat sheet, each layer sliding along x as the cube thins.
THICKNESSES = (1 + np.cos(np.linspace(0, np.pi, N_FRAMES))) / 2
SHEARS = MAX_SHEAR * (1 - THICKNESSES)


# --- math -----------------------------------------------------------------------------
def scene(thicknesses: np.ndarray, shears: np.ndarray) -> tuple[core.Point, core.Point, core.Patch, core.Scalar]:
    """The cube carried by each deformation: its corners, face centres and oriented face patches, and the
    volume factor."""
    deformation = core.deformation(thicknesses, shears)                       # [...] Point- <- Point-
    areas, _, volume = core.transport(deformation)
    corners, centres, patches = core.cube(SIDE)
    return deformation[..., None](corners), deformation[..., None](centres), areas[..., None](patches), volume


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.mechanics.area_transport import render

    vertices, centres, patches, volume = scene(THICKNESSES, SHEARS)
    save_animation(render.animate(zip(vertices, centres, patches, volume)), "area_transport", DURATION_MS,
                   forward_then_back=True)

    # checks
    # The deformation and its area map scale every volume by the determinant, at the flat sheet too.
    deformation = core.deformation(THICKNESSES, SHEARS)                       # [frames] Point- <- Point-
    areas, _, _ = core.transport(deformation)
    volumes = deformation(core.Point) & areas(core.Patch)                     # [frames] Scalar+ <- (Point-, Patch-)
    np.testing.assert_allclose((volumes - volume * (core.Point & core.Patch)).kernel, 0, atol=1e-12)


if __name__ == "__main__":
    main()
