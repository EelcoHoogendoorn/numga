"""Exponentiating one mixed product in Cl(2, 2) shears a planar square and its reference grid."""

import numpy as np

from numga import concatenate
from examples.math.doubled_algebra import core

shear_amount = 0.8
extent = 1.5
n_lines = 9
boundary = core.mv(core.Vector, [[-1, -1], [1, -1], [1, 1], [-1, 1]])         # [4] Vector+


def grid_segments(basis: core.Vector, extent: float, n_lines: int) -> core.Vector:
    """Horizontal and vertical segments (`[2 * n_lines, 2] Vector+`) form a square reference grid across `[-extent, extent]`."""
    offsets = np.linspace(-extent, extent, n_lines)
    ends = np.array([-extent, extent])
    horizontal = basis[0] * ends[None, :] + basis[1] * offsets[:, None]        # [n_lines, 2] Vector+
    vertical = basis[1] * ends[None, :] + basis[0] * offsets[:, None]          # [n_lines, 2] Vector+
    return concatenate([horizontal, vertical], axis=0)


def main() -> None:
    from examples.math.doubled_algebra import render
    from examples.util.animation import save_figure

    grid = grid_segments(core.basis, extent, n_lines)                          # [2 * n_lines, 2] Vector+
    transformation = core.representations((core.shear_generator * shear_amount).exp())  # [] State+ <- State+
    # The shear preserves grades, so its restriction to vectors maps vectors to vectors.
    vector_shear = transformation(core.Vector).cast(core.Vector)               # [] Vector+ <- Vector+
    grid_image = vector_shear(grid)                                            # [2 * n_lines, 2] Vector+
    boundary_image = vector_shear(boundary)                                    # [4] Vector+
    basis_image = vector_shear(core.basis)                                     # [2] Vector+
    area_image = transformation(core.mv.xy)                                    # [] State+

    save_figure(render.shear_figure(grid, grid_image, boundary, boundary_image, basis_image), "doubled_algebra_shear")

    # checks
    assert (area_image - core.mv.xy).scalar_norm_squared() < 1e-12
    assert (basis_image[0] - core.mv.x).scalar_norm_squared() < 1e-12
    assert (basis_image[1] - (core.mv.y + shear_amount * core.mv.x)).scalar_norm_squared() < 1e-12


if __name__ == "__main__":
    main()
