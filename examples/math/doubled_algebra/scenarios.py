"""A square sheared by the exponential of one mixed product in Cl(2,2)."""

import numpy as np

from examples.math.doubled_algebra import core

# How far y is sheared towards x, and the half width and lines of the drawn grid.
amount = np.array(0.8)
extent = 1.5
n_lines = 9


# --- math -----------------------------------------------------------------------------
def sheared_square(amount: np.ndarray, extent: float, n_lines: int) -> tuple[
    core.Vector, core.Vector, core.Vector, core.Vector, core.Vector,
]:
    """A planar grid and square, their sheared images and the images of the basis vectors."""
    grid = core.grid_samples(core.basis, extent, n_lines)                    # [lines, endpoints] Vector+
    boundary = core.mv(core.Vector, [[-1, -1], [1, -1], [1, 1], [-1, 1]])    # [corners] Vector+
    transformation = core.representations((core.shear_generator * amount).exp())   # [] State+ <- State+
    # The shear keeps grades, so restricted to vectors it is a map from vectors to vectors.
    vector_shear = transformation(core.Vector).cast(core.Vector)              # [] Vector+ <- Vector+
    return grid, vector_shear(grid), boundary, vector_shear(boundary), vector_shear(core.basis)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.math.doubled_algebra import render

    shear_geometry = sheared_square(amount, extent, n_lines)
    save_figure(render.draw_shear(*shear_geometry), "doubled_algebra_shear")

    # checks
    # The shear keeps the oriented area: the image of the area is the area.
    transformation = core.representations((core.shear_generator * amount).exp())   # [] State+ <- State+
    np.testing.assert_allclose((transformation(core.mv.xy) - core.mv.xy).kernel, 0, atol=1e-12)


if __name__ == "__main__":
    main()
