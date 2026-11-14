"""The common Möbius map of spherical synchronization preserves null rays, cross ratios, and circle angles."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.math.spherical_synchronization import core, render, scenarios


def sample_population() -> core.Point:
    """Five sample null rays on the unit sphere (`[5] Point-`)."""
    longitudes = np.array([-0.6, 0.4, 1.2, 2.3, 4.2])
    latitudes = np.array([-0.3, 0.6, 0.4, -0.2, 0.1])
    return core.sphere_points(longitudes, latitudes) + core.time      # [5] Point-


def test_common_map_preserves_null_rays_cross_ratios_and_circle_angles():
    mv = core.mv
    seeds = sample_population()                                       # [5] Point-
    rotor = (0.37 * mv.xt - 0.21 * mv.yt + 0.18 * mv.yz).exp()        # [] Rotor+
    mapped = core.directions(rotor >> seeds) + core.time              # [5] Point-
    assert (mapped.squared().abs() < 1e-11).all()

    def cross_ratio(points: core.Point):
        return ((points[0] | points[1]) * (points[2] | points[3])
                / ((points[0] | points[2]) * (points[1] | points[3])))

    assert ((cross_ratio(mapped) - cross_ratio(seeds)).abs() < 1e-11).all()

    # Two circles intersect at this direction; their tangent normals determine their intersection angle.
    direction = (mv.x * np.sqrt(0.84) + 0.4 * mv.z).dual()            # [] Point-
    circles = stack((mv.y, mv.z + mv.y / 3 - 0.4 * mv.t))             # [2] Cut+
    mapped_direction = core.directions(rotor >> (direction + core.time))   # [] Point-
    mapped_circles = rotor >> circles                                 # [2] Cut+
    assert ((mapped_circles & (mapped_direction + core.time)).abs() < 1e-11).all()

    def intersection_cosine(point: core.Point, cuts: core.Cut) -> float:
        """Cosine of the Euclidean angle between two circles meeting at `point`."""
        position = render.coordinates(point)
        spatial = (cuts[:, None] & stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))).to_array()
        temporal = (cuts[:, None] & core.time).to_array()
        tangent_normals = spatial + position * temporal
        tangent_normals /= np.linalg.norm(tangent_normals, axis=-1, keepdims=True)
        return float(tangent_normals[0] @ tangent_normals[1])

    assert abs(intersection_cosine(mapped_direction, mapped_circles) - intersection_cosine(direction, circles)) < 1e-11


def test_one_frame_and_figure_render():
    states = scenarios.synchronization(n_particles=2, n_steps=1)
    first_population, first_cuts = next(states)
    fig = render.sphere_figure(first_population, first_cuts)
    plt.close(fig)
    next(render.frames([next(states)]))
