"""The common Möbius flow reproduces spherical synchronization and preserves its geometry."""

from itertools import islice

import numpy as np
from scipy.integrate import solve_ivp

from numga import stack
from examples.math.spherical_synchronization import core, render


def steps(initial, seeds, rotation, coupling, dt, n_steps):
    return stack(list(islice(core.motion(initial, seeds, rotation, coupling, dt), n_steps + 1)))


def population():
    longitudes = np.array([-0.6, 0.4, 1.2, 2.3, 4.2])
    latitudes = np.array([-0.3, 0.6, 0.4, -0.2, 0.1])
    return core.sphere_points(longitudes, latitudes) + core.time


def test_common_rotor_matches_particle_equations_with_second_order_convergence():
    mv = core.mv
    seeds = population()
    rotation = -0.3 * mv.xy + 0.2 * mv.yz
    coupling, duration = 1.1, 1.6
    initial = render.coordinates(core.directions(seeds))
    omega = np.array([-0.2, 0, 0.3])

    # Integrate each particle's tangential attraction independently of the common rotor.
    def velocity(time, state):
        directions = state.reshape(initial.shape)
        mean = directions.mean(axis=0)
        attraction = mean - np.sum(directions * mean, axis=-1)[:, None] * directions
        return (np.cross(omega, directions) + coupling * attraction).ravel()

    reference = solve_ivp(velocity, (0, duration), initial.ravel(),
                          rtol=1e-10, atol=1e-12).y[:, -1].reshape(initial.shape)
    errors = []
    for dt in (0.04, 0.02):
        rotors = steps(mv.rotor(), seeds, rotation, coupling, dt, round(duration / dt))
        final = render.coordinates(core.directions(rotors[-1] >> seeds))
        errors.append(np.max(np.abs(final - reference)))
    assert errors[1] < 0.0005
    assert errors[1] < errors[0] / 3


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
    circles = stack((mv.y, mv.z + mv.y / 3 + 0.4 * mv.t))
    mapped_direction = core.directions(rotor >> (direction + core.time))
    mapped_circles = rotor >> circles
    np.testing.assert_allclose((mapped_circles & (mapped_direction + core.time)).kernel, 0, atol=1e-11)

    def intersection_angle(point, cuts):
        """The Euclidean angle between two circles at a point on both, from their cut planes' normals."""
        position = render.coordinates(point)
        coefficients = cuts.cast(core.ga.subspace("x y z t")).kernel
        tangent_normals = coefficients[:, :3] - position * coefficients[:, 3:]
        tangent_normals /= np.linalg.norm(tangent_normals, axis=-1, keepdims=True)
        return tangent_normals[0] @ tangent_normals[1]

    np.testing.assert_allclose(intersection_angle(mapped_direction, mapped_circles),
                               intersection_angle(direction, circles), atol=1e-11, rtol=1e-11)


def test_alignment_grows_and_mirroring_inputs_mirrors_the_population():
    mv = core.mv
    seeds = population()
    rotation = -0.3 * mv.xy + 0.2 * mv.yz
    coupling, dt, n_steps = 1.1, 0.025, 160
    rotors = steps(mv.rotor(), seeds, rotation, coupling, dt, n_steps)
    directions = core.directions(rotors[:, None] >> seeds)
    coherence = np.sum(render.coordinates(directions.mean(axis=-1)) ** 2, axis=-1)
    assert np.min(np.diff(coherence)) > -1e-8
    assert coherence[-1] > 0.98

    mirror = (mv.x + mv.y / 3 - mv.z / 5).normalized()
    mirrored_seeds = mirror >> seeds
    reflected = steps(mv.rotor(), mirrored_seeds, mirror >> rotation, coupling, dt, n_steps)
    mirrored_directions = core.directions(reflected[:, None] >> mirrored_seeds)
    np.testing.assert_allclose(mirrored_directions.kernel, (mirror >> directions).kernel,
                               atol=1e-10, rtol=1e-10)
