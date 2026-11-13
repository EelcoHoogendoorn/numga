"""The common Möbius map of spherical synchronization keeps null rays, cross ratios and circle angles."""

import numpy as np

from numga import stack
from examples.math.spherical_synchronization import core, render, scenarios


def population():
    longitudes = np.array([-0.6, 0.4, 1.2, 2.3, 4.2])
    latitudes = np.array([-0.3, 0.6, 0.4, -0.2, 0.1])
    return core.sphere_points(longitudes, latitudes) + core.time


def test_common_map_preserves_null_rays_cross_ratios_and_circle_angles():
    mv = core.mv
    seeds = population()
    rotor = (0.37 * mv.xt - 0.21 * mv.yt + 0.18 * mv.yz).exp()
    mapped = core.directions(rotor >> seeds) + core.time
    np.testing.assert_allclose(mapped.squared().kernel, 0, atol=1e-11)

    def cross_ratio(points):
        return ((points[0] | points[1]) * (points[2] | points[3])
                / ((points[0] | points[2]) * (points[1] | points[3])))

    np.testing.assert_allclose(cross_ratio(mapped).kernel, cross_ratio(seeds).kernel,
                               atol=1e-11, rtol=1e-11)

    # Two circles intersect at this direction; their tangent directions determine their angle.
    direction = (mv.x * np.sqrt(0.84) + 0.4 * mv.z).dual()
    circles = stack((mv.y, mv.z + mv.y / 3 - 0.4 * mv.t))
    mapped_direction = core.directions(rotor >> (direction + core.time))
    mapped_circles = rotor >> circles
    np.testing.assert_allclose((mapped_circles & (mapped_direction + core.time)).kernel, 0, atol=1e-11)

    def intersection_angle(point, cuts):
        """The Euclidean angle between two circles at a point on both, from their cut planes' normals."""
        position = render.coordinates(point)
        # Each cut's parts across x, y and z, and along time, read by pairing with the points on those axes.
        spatial = (cuts[:, None] & stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))).to_array()
        temporal = (cuts[:, None] & core.time).to_array()
        tangent_normals = spatial + position * temporal
        tangent_normals /= np.linalg.norm(tangent_normals, axis=-1, keepdims=True)
        return tangent_normals[0] @ tangent_normals[1]

    np.testing.assert_allclose(intersection_angle(mapped_direction, mapped_circles),
                               intersection_angle(direction, circles), atol=1e-11, rtol=1e-11)


def test_one_frame_of_the_synchronization_draws():
    next(render.frames(scenarios.synchronization(n_particles=2, n_steps=1)))
