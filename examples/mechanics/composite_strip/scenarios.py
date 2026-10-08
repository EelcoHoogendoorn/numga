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
FIBRE_SPACING = 7.0                                           # mm between drawn fibre lines
FIBRE_SAMPLES = 9
SEPARATION = 1.6 * WIDTH                                      # mm between the strips' centres
FRAMES = 60
DURATION_MS = 60
# The pull grows from nothing to its full value and back, once per loop.
LOADS = (1 - np.cos(np.linspace(0, 2 * np.pi, FRAMES, endpoint=False))) / 2


# --- math -----------------------------------------------------------------------------
def scene() -> tuple[core.Vector, core.Vector, core.Vector, core.State, core.Scalar]:
    """Both layups solved together: the strips at rest, each frame of the growing pull, the top ply's
    fibres in each frame, and their stretch–twist state and width strain at full load."""
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
    frames = core.deform(reference, state * (MAGNIFICATION * LOADS[:, None]),
                         width_strain * (MAGNIFICATION * LOADS[:, None]),
                         THICKNESS) + placement                        # [frames, cases, length samples, width samples] Vector
    # Lines along each layup's top ply's fibres, carried by the same deformation.
    top = (core.mv.xy * (-ANGLES[:, -1] / 2)).exp() >> core.mv.x       # [cases] Vector
    across = np.linspace(-WIDTH / 2, WIDTH / 2, FIBRE_SAMPLES)          # [samples] mm
    starts = np.arange(WIDTH / 2, LENGTH - WIDTH / 2, FIBRE_SPACING)    # [lines] mm along the centre line
    fibres = (core.mv.x * starts[:, None]
              + top[:, None, None] * (across / (top | core.mv.y)[:, None, None]))   # [cases, lines, samples] Vector
    fibre_frames = core.deform(fibres, state * (MAGNIFICATION * LOADS[:, None]),
                               width_strain * (MAGNIFICATION * LOADS[:, None]),
                               THICKNESS) + placement                   # [frames, cases, lines, samples] Vector
    return reference + placement, frames, fibre_frames, state, width_strain


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.mechanics.composite_strip import render

    placed, frames, fibres, _, _ = scene()
    save_animation(render.animate_strips(placed, frames, fibres), "composite_strip", DURATION_MS)


if __name__ == "__main__":
    main()
