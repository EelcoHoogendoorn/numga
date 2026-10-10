"""The same fibres and tension, with the plies ordered to keep or cancel twist."""

import numpy as np

from examples.mechanics.composite_strip import core

ANGLES = np.deg2rad([[-45, -45, 45, 45], [-45, 45, 45, -45]])   # bottom to top
LONGITUDINAL_MODULUS = 130000.0                                # MPa
TRANSVERSE_MODULUS = 10000.0                                   # MPa
POISSON = 0.3
SHEAR_MODULUS = 5000.0                                        # MPa
THICKNESS = 1.0                                               # mm
LENGTH = 100.0                                                # mm
WIDTH = 20.0                                                  # mm
TENSION = 30.0                                                # N/mm of width
LENGTH_SAMPLES = 61
WIDTH_SAMPLES = 13
MAGNIFICATION = 5.0
FIBRE_SPACING = 5.0                                           # mm between drawn fibre lines, across the fibres
SEPARATION = 1.6 * WIDTH                                      # mm between the strips' centres
FRAMES = 60
DURATION_MS = 60
# The pull grows from nothing to its full value and back, once per loop.
LOADS = (1 - np.cos(np.linspace(0, 2 * np.pi, FRAMES, endpoint=False))) / 2


# --- math -----------------------------------------------------------------------------
def scene() -> tuple[core.Vector, core.Vector, core.Scalar]:
    """Both layups solved together: the strips at rest, each frame of the growing pull, and the
    distance across the top ply's fibres over each strip, whose level lines are those fibres."""
    plies = ANGLES.shape[-1]
    ply_thickness = THICKNESS / plies
    # Two samples per ply integrate its quadratic strain energy exactly.
    nodes = np.array([-1, 1]) / np.sqrt(3)
    centres = (np.arange(plies) + 0.5) * ply_thickness - THICKNESS / 2
    heights = core.mv.z * (centres[:, None] + nodes * ply_thickness / 2)  # [plies, samples] Vector
    weights = np.full((plies, len(nodes)), ply_thickness / 2)             # [plies, samples]
    # Rotate every ply's fibre direction; both layups share the thickness samples.
    fibres = (core.mv.xy * (-ANGLES[..., None] / 2)).exp() >> core.mv.x   # [cases, plies, 1] Vector
    # Solve the two layups together, including their force-free width contraction.
    state, width_strain = core.response(fibres, heights, weights, THICKNESS,
                                       LONGITUDINAL_MODULUS, TRANSVERSE_MODULUS, POISSON,
                                       SHEAR_MODULUS, TENSION)          # [cases] State, [cases] Scalar
    # One flat midsurface, sampled along its length and width, placed once for each layup.
    reference = (
        core.mv.x * np.linspace(0, LENGTH, LENGTH_SAMPLES)[:, None]
        + core.mv.y * np.linspace(-WIDTH / 2, WIDTH / 2, WIDTH_SAMPLES)[None, :]
    )                                                                   # [length samples, width samples] Vector
    # Each strip deforms about its own centreline, then sits beside the other.
    placement = core.mv.y * (SEPARATION * np.array([-0.5, 0.5]))[:, None, None]   # [cases, 1, 1] Vector
    # The response is linear, so each frame scales the full one; magnified for display only.
    shown = MAGNIFICATION * LOADS[:, None]                              # [frames, 1]
    frames = core.deform(reference, state * shown, width_strain * shown,
                         THICKNESS) + placement                        # [frames, cases, length samples, width samples] Vector
    # The top ply's fibres run along the level lines of the distance across them.
    top = fibres[:, -1, 0]                                              # [cases] Vector
    across_fibres = reference | (core.mv.xy | top[:, None, None])       # [cases, length samples, width samples] Scalar
    return reference + placement, frames, across_fibres


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.mechanics.composite_strip import render

    placed, frames, across_fibres = scene()
    save_animation(render.animate_strips(placed, frames, across_fibres, FIBRE_SPACING), "composite_strip", DURATION_MS)


if __name__ == "__main__":
    main()
