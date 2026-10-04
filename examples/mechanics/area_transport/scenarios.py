"""A sheared cube approaching a flat sheet, including the singular endpoint."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.mechanics.area_transport import core

SIDE = 1.0
CASE_NAMES = ("Solid", "Compressed and sheared", "Flat sheet")
MAX_SHEAR = 0.70
THICKNESSES = np.array([1.0, 0.35, 0.0])
SHEARS = np.array([0.0, 0.45, MAX_SHEAR])
FRAMES = 40
DURATION_MS = 70


# --- math -----------------------------------------------------------------------------
def scene(thicknesses: np.ndarray, shears: np.ndarray) -> tuple[core.Transport, core.Surface]:
    """The area maps and the cube they carry, batched over the deformations."""
    transport = core.transport(core.deformation(thicknesses, shears))
    surface = core.carried(core.cube(SIDE), transport)

    # --- checks
    # The carried area normals are the carried patches' duals, and the adjugate undoes the deformation
    # up to its volume, at the flat sheet too.
    np.testing.assert_allclose((surface.area_normals - surface.patches.dual()).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((transport.adjugate(transport.deformation) - transport.volume * core.Vector).kernel, 0, atol=1e-12)
    return transport, surface


def collapse() -> Iterator[tuple[core.Vector, core.Vector, core.Vector, core.Scalar]]:
    """Vertices, face centres, area normals and volume through a collapse and recovery."""
    phase = np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)
    thickness = (1 + np.cos(phase)) / 2
    transport, surface = scene(thickness, MAX_SHEAR * (1 - thickness))
    yield from zip(surface.vertices, surface.centres, surface.area_normals, transport.volume)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.mechanics.area_transport import render

    transport, surface = scene(THICKNESSES, SHEARS)
    save_figure(render.comparison(surface, transport.volume, CASE_NAMES), "area_transport")
    save_animation(render.animate(collapse()), "area_transport", DURATION_MS)


if __name__ == "__main__":
    main()
