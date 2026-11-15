"""Scenes and checks for quadric rigid bodies on the 2-sphere and on the 3-sphere.

On the 2-sphere, rigid ellipses seen on the front hemisphere form three scenes: a colliding crowd of needles, ovals, and discs; a single oval flipping periodically about its intermediate axis; and a giant oval surrounded by a fast swarm.

On the 3-sphere, four ray-traced scenes highlight closed spherical geometry: a crowd scattered across the whole 3-sphere; small ellipsoids bouncing inside the equatorial gap around a huge central ellipsoid; a nearly antipodal needle spinning about its long axis; and ellipsoids confined inside a Clifford torus whose cross-section widens and narrows along its core great circle.
"""

from __future__ import annotations

import numpy as np

from numga import concatenate, stack
from numga.algebras import Spherical3D
from examples import instantiate
from examples.quadrics.elliptic_physics import render
from examples.quadrics.s3_raytracer import core as raytracer

S2 = instantiate("examples.quadrics.elliptic_physics.core", Spherical3D)
S3 = instantiate("examples.quadrics.elliptic_physics.core", raytracer.ga)

s2_dt = 0.015
n_s2_frames = 160
n_s2_substeps = 2
n_tumbling_frames = 240
n_tumbling_substeps = 3
n_s3_frames = 160
n_s3_substeps = 6
s3_dt = 0.03
s3_shape = (180, 240)
frame_duration_ms = 40

# --- 2-sphere scene parameters and setup ----------------------------------------------
camera = (S2.mv.yz * 0.2714904022294391 + S2.mv.zx * 0.6097774271693677 + S2.mv.xy * 1.0148400517968847).exp()  # [] Motor+

ellipse_half_widths = np.radians([30.0, 10.0])
ellipse_turn = 0.3

pair_half_angles = np.array([[30.0, 8.0], [17.0, 17.0]])
pair_masses = np.array([1.0, 1.2])
pair_polars = np.array([0.3, 0.3])
pair_azimuths = np.array([0.7 + np.pi, 0.7])
pair_rates = np.array([[-0.5, -2.0, 0.3], [1.0, 2.0, 0.2]])
pair_colours = ["#38bdf8", "#f43f5e"]
n_pair_rings = 96

crowd_half_angles = np.array([
    [30.0, 6.0], [17.0, 17.0], [28.0, 6.5], [24.0, 5.5], [15.0, 4.0], [26.0, 9.0], [9.0, 9.0],
])
crowd_masses = np.array([1.0, 1.2, 0.9, 0.8, 0.5, 1.1, 0.4])
crowd_polars = np.array([0.35, 1.05, 1.15, 1.10, 1.25, 0.90, 1.55])
crowd_azimuths = np.array([0.2, 0.7, 2.1, 3.6, 4.9, -0.67, 2.90])
crowd_rates = np.array([
    [0.8, 2.6, 0.5],
    [1.8, -1.2, 0.6],
    [-1.7, 1.5, -0.6],
    [2.0, 1.0, -0.5],
    [-2.2, -1.8, 0.7],
    [1.6, 1.4, 0.7],
    [-2.4, 0.8, -0.4],
])
crowd_colours = ["#38bdf8", "#f43f5e", "#fbbf24", "#34d399", "#a855f7", "#fb923c", "#ec4899"]
n_crowd_rings = 384


def ellipse_samples(
    half_angles: np.ndarray,
    masses: np.ndarray,
    n_azimuths: int,
    n_radii: int,
) -> tuple[S2.Point, S2.Scalar]:
    """Gnomonic polar quadrature over ellipses of angular `half_angles` (`[n_bodies, 2]`) forms unit interior points (`[n_bodies, n_samples] Point-`) and point masses (`[n_bodies, n_samples] Scalar+`)."""
    n_bodies = len(masses)
    radii, azimuths = np.meshgrid(
        np.linspace(0.0, 1.0, n_radii + 1)[1:],
        np.linspace(0.0, 2.0 * np.pi, n_azimuths, endpoint=False),
    )
    azimuth_turns = (S2.mv.xy * (-azimuths / 2)).exp()                          # [n_azimuths, n_radii] Motor+
    unit_circle = azimuth_turns >> S2.basis[0]                                  # [n_azimuths, n_radii] Point-
    tangent_x = np.tan(half_angles[:, 0])[:, None, None]
    tangent_y = np.tan(half_angles[:, 1])[:, None, None]
    stretch = (
        S2.basis[0] * (S2.mv.x & S2.Point) * tangent_x
        + S2.basis[1] * (S2.mv.y & S2.Point) * tangent_y
    )                                                                           # [n_bodies, 1, 1] Point- <- Point-
    gnomonic = stretch(unit_circle) * radii + S2.origin                         # [n_bodies, n_azimuths, n_radii] Point-
    norm_squared = gnomonic.scalar_norm_squared()                               # [n_bodies, n_azimuths, n_radii] Scalar+
    area_element = (tangent_x * tangent_y * radii) / (norm_squared * norm_squared.square_root())  # [n_bodies, n_azimuths, n_radii] Scalar+
    radial_weights = np.ones(n_radii)
    radial_weights[-1] = 0.5
    weights = (area_element * radial_weights).reshape(n_bodies, -1)             # [n_bodies, n_azimuths * n_radii] Scalar+
    sample_masses = weights * (masses[:, None] / weights.sum(axis=-1)[..., None])  # [n_bodies, n_azimuths * n_radii] Scalar+
    points = gnomonic.normalized().reshape(n_bodies, -1)                        # [n_bodies, n_azimuths * n_radii] Point-
    return points, sample_masses


def polar_placement(polars: np.ndarray, azimuths: np.ndarray) -> S2.Motor:
    """Azimuthal turns about `z` determine meridian rotors (`[...] Motor+`) carrying the pole to polar angles `polars` at `azimuths`."""
    azimuth_turns = (S2.mv.xy * (-azimuths / 2)).exp()                          # [...] Motor+
    meridian_planes = azimuth_turns >> S2.mv.zx                                 # [...] Rate+
    return (meridian_planes * (polars / 2)).exp()                               # [...] Motor+


def ellipse_population(
    half_angles: np.ndarray,
    masses: np.ndarray,
    placements: S2.Motor,
    rates: np.ndarray,
    colors: list[str],
    n_azimuths: int,
) -> tuple[np.ndarray, S2.Motor, S2.Momentum, S2.DualQuadric, S2.Quadric, S2.InverseInertia, S2.Scalar]:
    """Ellipses with `half_angles` (`[n_bodies, 2]` in degrees), `masses`, `placements`, and `rates` (`[n_bodies, 3]`) assemble RGB colours and rigid-body state extensors under `camera`."""
    radians = np.radians(half_angles)
    body_rates = S2.mv.yz * rates[:, 0] + S2.mv.zx * rates[:, 1] + S2.mv.xy * rates[:, 2]  # [n_bodies] Rate+
    duals = S2.ellipsoid_quadric(np.tan(radians))                               # [n_bodies] Point- <- Plane+
    points, sample_masses = ellipse_samples(radians, masses, n_azimuths, 10)
    return render.rgb_colors(colors), *S2.body_state(duals, camera * placements, body_rates, points, sample_masses)


def joined_population(groups: list[tuple]) -> tuple:
    """Concatenates `(colors, motors, momenta, duals, forms, inverse_inertias, bounding_radii)` tuples along the body axis."""
    colors = np.concatenate([group[0] for group in groups], axis=0)
    extensors = tuple(concatenate([group[slot] for group in groups]) for slot in range(1, len(groups[0])))
    return (colors, *extensors)


def selected_population(population: tuple, keep: np.ndarray) -> tuple:
    """Indexes every array and extensor in `population` by `keep` along the body axis."""
    return tuple(entry[keep] for entry in population)


def crowded_scene():
    """Seven ellipses from needles to discs scatter across the 2-sphere (`[7]`)."""
    return ellipse_population(
        crowd_half_angles,
        crowd_masses,
        polar_placement(crowd_polars, crowd_azimuths),
        crowd_rates,
        crowd_colours,
        n_crowd_rings,
    )


def tumbling_scene():
    """One 40-by-8-degree oval spins near its intermediate `yz` plane so it flips over and back periodically."""
    tilt = (S2.mv.xz * (np.radians(15.0) / 2)).exp() * (S2.mv.yz * (np.radians(10.0) / 2)).exp()  # [] Motor+
    return ellipse_population(
        np.array([[40.0, 8.0]]),
        np.array([1.0]),
        tilt.reshape(1),
        np.array([[14.0, -0.1, 0.05]]),
        ["#38bdf8"],
        384,
    )


def hyperbolic_scene():
    """A giant 75-by-48-degree oval and four fast smaller bodies share the narrow channel around the 2-sphere."""
    giant = ellipse_population(
        np.array([[75.0, 48.0]]),
        np.array([6.0]),
        polar_placement(np.zeros(1), np.zeros(1)),
        np.array([[0.3, 0.2, 0.4]]),
        ["#38bdf8"],
        1536,
    )
    swarm = ellipse_population(
        np.array([[18.0, 4.5], [10.0, 10.0], [15.0, 4.0], [16.0, 5.0]]),
        np.array([0.6, 0.7, 0.5, 0.5]),
        polar_placement(np.full(4, 1.57), np.array([0.35, 1.10, 1.85, 2.65])),
        np.array([[2.5, 1.5, -0.8], [-2.2, 1.2, 0.6], [1.8, -2.0, 0.7], [-1.6, -1.8, -0.5]]),
        ["#f43f5e", "#fbbf24", "#34d399", "#a855f7"],
        384,
    )
    return joined_population([giant, swarm])


# --- 3-sphere scene parameters and setup ----------------------------------------------
def sphere_placement(places: S3.Point, spins: S3.Rate) -> S3.Motor:
    """Motors (`[...] Motor+`) carry `origin` to `places` (`[...] Point-`) after turning by `spins` (`[...] Rate+`)."""
    transport = (raytracer.unit(places) / raytracer.origin).square_root()       # [...] Motor+
    turn = (spins / 2).exp()                                                    # [...] Motor+
    return transport * turn                                                     # [...] Motor+


def origin_body(
    color: np.ndarray,
    dual: S3.DualQuadric,
    rate: S3.Rate,
    mass: float,
    rng: np.random.Generator,
) -> tuple:
    """One body at the origin of the 3-sphere with `dual`, `rate`, and `mass` forms a 1-body population tuple."""
    motor = sphere_placement(raytracer.origin.reshape(1), S3.mv.bivector(np.zeros((1, 6))))  # [1] Motor+
    points, sample_masses = S3.interior_samples(dual, np.array([mass]), 400, rng)
    return color[None], *S3.body_state(dual, motor, rate, points, sample_masses)


def random_population(
    rng: np.random.Generator,
    n_candidates: int,
    size_bounds: tuple[float, float],
) -> tuple:
    """Log-uniform half-widths and uniform placements on the 3-sphere build `n_candidates` candidate ellipsoids spinning near their intermediate axes."""
    half_widths = 10 ** rng.uniform(np.log10(size_bounds[0]), np.log10(size_bounds[1]), size=(n_candidates, 3))
    rate_components = rng.normal(size=(n_candidates, 6)) * np.array([0.3, 0.3, 0.3, 0.6, 0.6, 0.6])
    rate_components[np.arange(n_candidates), np.argsort(half_widths, axis=-1)[:, 1]] = rng.choice(
        [-5.0, 5.0], size=n_candidates,
    )
    places = S3.mv.vector(rng.normal(size=(n_candidates, 4))).dual()            # [n_candidates] Point-
    spins = S3.mv.bivector(rng.normal(size=(n_candidates, 6)) * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))  # [n_candidates] Rate+
    colors = render.hue_colors(rng.permutation(n_candidates) / n_candidates)
    duals = S3.ellipsoid_quadric(half_widths)                                   # [n_candidates] Point- <- Plane+
    motors = sphere_placement(places, spins)                                    # [n_candidates] Motor+
    rates = S3.mv.bivector(rate_components)                                     # [n_candidates] Rate+
    points, sample_masses = S3.interior_samples(duals, np.prod(half_widths, axis=-1) * 200.0, 400, rng)
    return colors, *S3.body_state(duals, motors, rates, points, sample_masses)


def admitted_population(population: tuple, n_given: int, n_added: int) -> tuple:
    """Retains the first `n_given` bodies and greedily admits `n_added` non-overlapping candidates from `population`."""
    colors, motors, _, _, forms, _, bounding_radii = population
    n_total = len(colors)
    first_indices, second_indices = S3.candidate_pairs(motors, bounding_radii)
    relative = motors[first_indices].inverse() * motors[second_indices]         # [...] Motor+
    overlapping = np.zeros((n_total, n_total), dtype=bool)
    margins, _ = S3.quadric_overlap(
        forms[first_indices],
        relative >> forms[second_indices](relative << S3.Point),
    )
    overlapping[first_indices, second_indices] = overlapping[second_indices, first_indices] = margins < 0
    kept = list(range(n_given))
    for candidate in range(n_given, n_total):
        if not overlapping[candidate, kept].any():
            kept.append(candidate)
    return selected_population(population, np.array(kept[: n_given + n_added]))


def s3_crowd_scene():
    """Twenty-eight non-overlapping ellipsoids scatter across the 3-sphere under an origin camera and off-frustum point light."""
    rng = np.random.default_rng(3)
    population = admitted_population(random_population(rng, 120, (0.05, 0.5)), 0, 28)
    light_turn = ((S3.mv.xw * -0.34 + S3.mv.zw * 0.94) * (0.8 / 2)).exp()      # [] Motor+
    light = raytracer.unit(light_turn >> raytracer.origin)                      # [] Point-
    return *population, S3.mv.rotor(), light


def s3_gap_scene():
    """Sixty small ellipsoids bounce inside the equatorial belt left around a 60-by-75-by-70-degree central ellipsoid."""
    rng = np.random.default_rng(5)
    huge = origin_body(
        np.array([0.75, 0.7, 0.6]),
        S3.ellipsoid_quadric(np.tan(np.radians([[60.0, 75.0, 70.0]]))),
        S3.mv.bivector(np.zeros((1, 6))),
        500.0,
        rng,
    )
    population = admitted_population(joined_population([huge, random_population(rng, 2000, (0.03, 0.15))]), 1, 60)
    eye = (S3.mv.xw * (np.pi / 4)).exp() * (-S3.mv.xy * (np.pi / 4)).exp()     # [] Motor+
    light_offset = ((S3.mv.xw * -0.3 + S3.mv.yw * 0.25 + S3.mv.zw * 0.2) / 2).exp()  # [] Motor+
    light = (eye * light_offset) >> raytracer.origin                            # [] Point-
    return *population, eye, light


def s3_needle_scene():
    """Forty ellipsoids collide around an 80-degree spinning needle whose antipodal tips leave a 20-degree gap."""
    rng = np.random.default_rng(7)
    needle_body = origin_body(
        np.array([0.9, 0.85, 0.3]),
        S3.ellipsoid_quadric(np.tan(np.radians([[80.0, 4.0, 3.0]]))),
        S3.mv.yz.reshape(1) * 2.0,
        50.0,
        rng,
    )
    population = admitted_population(joined_population([needle_body, random_population(rng, 1000, (0.05, 0.3))]), 1, 40)
    eye = (
        (S3.mv.xw * (np.pi / 4)).exp()
        * (S3.mv.zw * (np.radians(35.0) / 2)).exp()
        * (S3.mv.zx * (-np.pi / 4)).exp()
    )                                                                           # [] Motor+
    light_offset = ((S3.mv.xw * -0.3 + S3.mv.yw * 0.2 + S3.mv.zw * 0.3) / 2).exp()  # [] Motor+
    light = (eye * light_offset) >> raytracer.origin                            # [] Point-
    return *population, eye, light


def s3_tunnel_scene():
    """Fifty ellipsoids bounce inside a Clifford torus whose elliptical tube narrows from 40 degrees to 20 degrees along its core circle."""
    rng = np.random.default_rng(11)
    tube = S3.diagonal_quadric(
        np.array([[-1.0, -1.0 / 1.5**2, 1.0 / np.tan(np.radians(40.0)) ** 2, 1.0 / np.tan(np.radians(20.0)) ** 2]]),
    )                                                                           # [1] Point- <- Plane+
    torus = origin_body(np.array([0.55, 0.65, 0.75]), tube, S3.mv.bivector(np.zeros((1, 6))), 500.0, rng)
    population = admitted_population(joined_population([torus, random_population(rng, 2000, (0.03, 0.1))]), 1, 50)
    eye = (S3.mv.zw * (np.pi / 4)).exp() * (S3.mv.zx * (np.pi / 4)).exp()      # [] Motor+
    camera_point = eye >> raytracer.origin                                      # [] Point-
    side_up = raytracer.direction(np.array([0, 1, 1]))                          # [] Point-
    wall_surface = S3.world_quadrics(torus[1], torus[4])                        # [1] Plane+ <- Point-
    conic, polar = raytracer.project(eye, wall_surface)
    wall_hit = (camera_point * raytracer.reproject(conic, polar, side_up) + (eye >> side_up)).normalized()  # [1] Point-
    light = (camera_point + wall_hit[0]).normalized()                           # [] Point-
    return *population, eye, light


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Streams all seven 2-sphere and 3-sphere rigid-body animations and figures to `plots/` and verifies conservation laws."""
    from examples.util.animation import save_animation, save_figure

    def stream_s2(scene_builder, name: str, n_frames: int, n_substeps: int):
        colors, motors, momenta, duals, forms, inverse_inertias, bounding_radii = scene_builder()
        recorded: list[tuple[S2.Motor, S2.Momentum, int]] = []

        def states():
            for state in S2.collision_frames(
                motors, momenta, forms, duals, inverse_inertias, bounding_radii, n_frames, s2_dt, n_substeps,
            ):
                recorded.append(state)
                yield state

        save_animation(
            render.s2_animation(forms, colors, render.painted_resolution, states()),
            name,
            frame_duration_ms,
        )
        return recorded, inverse_inertias

    def stream_s3(scene_builder, name: str):
        colors, motors, momenta, duals, forms, inverse_inertias, bounding_radii, eye, light = scene_builder()
        recorded: list[tuple[S3.Motor, S3.Momentum, int]] = []
        last_frame: list[np.ndarray] = []

        def frames():
            states = S3.collision_frames(
                motors, momenta, forms, duals, inverse_inertias, bounding_radii, n_s3_frames, s3_dt, n_s3_substeps,
            )
            def recording_states():
                for state in states:
                    recorded.append(state)
                    yield state

            for image in render.s3_animation(forms, colors, eye, light, s3_shape, recording_states()):
                last_frame[:] = [image]
                yield image

        save_animation(frames(), name, frame_duration_ms)
        save_figure(render.frame_figure(last_frame[0]), name)
        plt_close(last_frame)
        return recorded, inverse_inertias

    def plt_close(_unused) -> None:
        import matplotlib.pyplot as plt

        plt.close("all")

    s2_runs = {
        "crowded": stream_s2(crowded_scene, "spherical_quadric_physics", n_s2_frames, n_s2_substeps),
        "tumbling": stream_s2(tumbling_scene, "spherical_tumbling_oval", n_tumbling_frames, n_tumbling_substeps),
        "hyperbolic": stream_s2(hyperbolic_scene, "spherical_hyperbolic_arena", n_s2_frames, n_s2_substeps),
    }
    tumbling_states, tumbling_inverse = s2_runs["tumbling"]
    tumbling_rates = stack([tumbling_inverse(momenta)[0] for _, momenta, _ in tumbling_states])  # [n_tumbling_frames] Rate+
    fig = render.tumbling_figure(tumbling_rates, s2_dt)
    save_figure(fig, "spherical_tumbling_oval")
    plt_close(fig)

    s3_runs = {
        "crowd": stream_s3(s3_crowd_scene, "s3_quadric_physics"),
        "gap": stream_s3(s3_gap_scene, "s3_gap"),
        "needle": stream_s3(s3_needle_scene, "s3_needle"),
        "tunnel": stream_s3(s3_tunnel_scene, "s3_tunnel"),
    }

    # checks
    for scene_name, drift in (("crowded", 1e-2), ("tumbling", 2e-2), ("hyperbolic", 2e-2)):
        states, inverse_inertias = s2_runs[scene_name]
        energies = stack([S2.kinetic_energy(inverse_inertias(momenta), momenta) for _, momenta, _ in states])  # [n_frames] Scalar+
        momenta_norms = stack([S2.world_momentum(motors, momenta).antinorm() for motors, momenta, _ in states])  # [n_frames] Scalar+
        assert (((energies.max() - energies.min()) / energies[0].abs()) < drift).all()
        assert (((momenta_norms.max() - momenta_norms.min()) / momenta_norms[0]) < 1e-11).all()

    intermediate_spin = tumbling_rates & S2.mv.yz.dual()                        # [n_tumbling_frames] Scalar+
    positive = intermediate_spin > 0
    assert int(np.count_nonzero(positive[1:] != positive[:-1])) >= 2

    for scene_name, drift in (("crowd", 2e-2), ("gap", 5e-2), ("needle", 2e-2), ("tunnel", 1.5e-1)):
        s3_states, s3_inverse = s3_runs[scene_name]
        assert s3_states[-1][2] > 0
        s3_energies = stack([S3.kinetic_energy(s3_inverse(momenta), momenta) for _, momenta, _ in s3_states])  # [n_s3_frames] Scalar+
        s3_momenta_norms = stack([S3.world_momentum(motors, momenta).antinorm() for motors, momenta, _ in s3_states])  # [n_s3_frames] Scalar+
        assert (((s3_energies.max() - s3_energies.min()) / s3_energies[0].abs()) < drift).all()
        assert (((s3_momenta_norms.max() - s3_momenta_norms.min()) / s3_momenta_norms[0]) < 1e-11).all()


if __name__ == "__main__":
    main()
