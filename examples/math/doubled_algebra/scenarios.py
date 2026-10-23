"""A square sheared by the exponential of one mixed product in Cl(2,2)."""

import numpy as np

from examples.math.doubled_algebra import core

# How far y is sheared towards x, and the half width and lines of the drawn grid.
AMOUNT = np.array(0.8)
EXTENT = 1.5
N_LINES = 9


# --- math -----------------------------------------------------------------------------
def sheared_square(amount: np.ndarray, extent: float, lines: int) -> tuple[
    core.Vector, core.Vector, core.Vector, core.Vector, core.Vector,
]:
    """A planar grid and square, their sheared images and the images of the coordinate directions."""
    grid = core.grid_samples(core.directions, extent, lines)
    boundary = core.mv(core.Vector, [[-1, -1], [1, -1], [1, 1], [-1, 1]])
    transformation = core.shear(amount)
    return grid, transformation(grid), boundary, transformation(boundary), transformation(core.directions)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.math.doubled_algebra import render

    sheared = sheared_square(AMOUNT, EXTENT, N_LINES)
    save_figure(render.draw_shear(*sheared), "doubled_algebra_shear")

    # checks
    # The shear keeps the oriented area: the carried area is the area.
    area = core.shear(AMOUNT)(core.mv.xy)
    np.testing.assert_allclose((area - core.mv.xy).kernel, 0, atol=1e-12)


if __name__ == "__main__":
    main()
