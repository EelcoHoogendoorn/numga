"""Extension–twist coupling: the same fibres and tension, with the plies ordered to keep or cancel twist."""

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
N_LENGTH_SAMPLES = 61
N_WIDTH_SAMPLES = 13
MAGNIFICATION = 5.0
FIBRE_SPACING = 5.0                                           # mm between drawn fibre lines, across the fibres
SEPARATION = 1.6 * WIDTH                                      # mm between the strips' centres
N_FRAMES = 60
DURATION_MS = 60
# The pull grows from nothing to its full value; the animation plays it forward and back, once per loop.
LOADS = (1 - np.cos(np.linspace(0, np.pi, N_FRAMES // 2 + 1))) / 2


# --- math -----------------------------------------------------------------------------
def scene() -> tuple[core.Direction, core.Direction, core.Scalar]:
    """Both layups solved together: the strips at rest, each frame of the growing pull, and the
    distance across the top ply's fibres over each strip, whose level lines are those fibres."""
    n_plies = ANGLES.shape[-1]
    ply_thickness = THICKNESS / n_plies
    # Two samples per ply integrate its quadratic strain energy exactly.
    nodes = np.array([-1, 1]) / np.sqrt(3)
    centres = (np.arange(n_plies) + 0.5) * ply_thickness - THICKNESS / 2
    heights = core.along_z * (centres[:, None] + nodes * ply_thickness / 2)  # [plies, samples] Direction-
    weights = np.full((n_plies, len(nodes)), ply_thickness / 2)             # [plies, samples]
    # Rotate every ply's fibre direction; both layups share the thickness samples.
    plies = (core.mv.xy * (-ANGLES[..., None] / 2)).exp()                   # [cases, plies, 1] Rotor+
    fibres = plies >> core.along_x                                      # [cases, plies, 1] Direction-
    # Solve the two layups together, including their force-free width contraction.
    state, width_strain = core.response(fibres, heights, weights, THICKNESS,
                                       LONGITUDINAL_MODULUS, TRANSVERSE_MODULUS, POISSON,
                                       SHEAR_MODULUS, TENSION)          # [cases] State, [cases] Scalar+
    # One flat midsurface, sampled along its length and width, placed once for each layup.
    reference = (
        core.along_x * np.linspace(0, LENGTH, N_LENGTH_SAMPLES)[:, None]
        + core.along_y * np.linspace(-WIDTH / 2, WIDTH / 2, N_WIDTH_SAMPLES)[None, :]
    )                                                                   # [length samples, width samples] Direction-
    # Each strip deforms about its own centreline, then sits beside the other.
    placement = core.along_y * (SEPARATION * np.array([-0.5, 0.5]))[:, None, None]   # [cases, 1, 1] Direction-
    # The response is linear, so each frame scales the full one; magnified for display only.
    display_scale = MAGNIFICATION * LOADS[:, None]                      # [frames, 1]
    frames = core.deform(reference, state * display_scale, width_strain * display_scale,
                         THICKNESS) + placement                        # [frames, cases, length samples, width samples] Direction-
    # The top ply's fibres run along the level lines of the distance across them: the plane y = 0 turned with the
    # ply, paired with each point.
    across = plies[:, -1, 0] >> core.mv.y                                # [cases] Plane+
    across_fibres = across[:, None, None] & reference                   # [cases, length samples, width samples] Scalar+
    return reference + placement, frames, across_fibres


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.mechanics.composite_strip import render

    placed, frames, across_fibres = scene()
    save_animation(render.animate_strips(placed, frames, across_fibres, FIBRE_SPACING), "composite_strip", DURATION_MS,
                   forward_then_back=True)


if __name__ == "__main__":
    main()
