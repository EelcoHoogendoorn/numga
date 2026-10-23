"""An equatorial spin sweep and a camera above a rotating black hole's disk."""

from collections.abc import Iterator

import numpy as np

from examples.relativity.kerr import core

mass = 1.0
n_frames = 80
max_spin = 0.8 * mass
spins = -max_spin * np.cos(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))
duration_ms = 80
extent = 12.0 * mass
n_rays = 31
offsets = np.linspace(-8, 8, n_rays) * mass
step_size = 0.1 * mass
n_steps = 400
stop_fraction = 0.8
n_ring_samples = 128
camera_spin = 0.8 * mass
camera_distance = 28.0 * mass
camera_inclination = np.deg2rad(75.0)
half_view = np.tan(np.deg2rad(19.0))
image_width = 480
image_height = image_width * 2 // 3
camera_step_fraction = 0.2
n_camera_steps = 720
disk_outer_radius = 12.0 * mass
# The disk's optical depth straight through its thickness, at its inner edge.
inner_depth = 4.0
disk_temperature_scale = 58500.0                               # K, (accretion rate c^6 / (σ G^2 M^2))^(1/4)
exposure_temperature = 6500.0                                 # K, shown at mid-grey
escape_radius = 100.0 * mass
# Captured inside every photon orbit, where no ray turns back out.
capture_radius = 1.05 * (mass + np.sqrt(mass**2 - camera_spin**2))
sky_width = 2048
sky_height = sky_width // 2
sky_seed = 8
n_polarizer_frames = 80
polarizer_angles = np.linspace(0, np.pi, n_polarizer_frames, endpoint=False)
polarization_degree = 0.8
n_flow_frames = 80
web_scale = 2 / 3                                               # the camera's animations, box-filtered to this share of its size
flow_time = 50.0 * mass                                         # coordinate time the disk turns through in one loop


# --- math -----------------------------------------------------------------------------
def scene() -> tuple[core.Antivector, core.Antivector]:
    """Rays sent along x past the hole, one sweep of its spin, and the horizon's cut through the
    equatorial plane at each spin."""
    x, y = core.mv.x.dual(), core.mv.y.dual()
    spin = core.mv.xy * spins[:, None]                         # [frames, 1] Spin
    position = -x * extent + y * offsets                        # [rays] Antivector
    momentum = core.initial_momentum(position, x, spin, mass)
    position = position.broadcast_to(momentum.shape)
    paths, _ = core.trace(position, momentum, spin, mass, step_size, n_steps, stop_fraction, core.rates)

    # A constant oblate radius cuts the equatorial plane in this circle.
    horizon = mass + np.sqrt(mass**2 - spins**2)
    radius = np.sqrt(horizon**2 + spins**2)
    angles = np.linspace(0, 2 * np.pi, n_ring_samples + 1)
    turns = (core.mv.xy * (-angles / 2)).exp()
    horizons = (turns >> x) * radius[:, None]
    return paths, horizons


def camera() -> tuple[core.Image, core.Scalar, core.Scalar]:
    """The camera's view, resolved ray by ray; each disk crossing's observed temperature, `[crossings]
    Scalar`; and the power its emission passes through each analyzer, `[analyzers, crossings]
    Scalar`."""
    orientation = (core.mv.zx * (-camera_inclination / 2)).exp()
    eye = (orientation >> core.mv.z.dual()) * camera_distance
    spin = core.mv.xy * camera_spin
    # The thin disk ends at the innermost stable orbit of gas turning with the hole.
    inner_radius = core.innermost_stable_orbit(core.mv.xy, spin, mass)
    position, momentum = core.camera_rays(eye, orientation, image_width, image_height,
                                         half_view, spin, mass)
    screen = core.camera_screen(position, momentum, orientation, spin, mass)
    def opacity(event: core.Antivector, momentum: core.Vector, radius: core.Scalar) -> core.Scalar:
        depth = core.disk_depth(radius, inner_radius, disk_outer_radius, inner_depth)
        return core.disk_opacity(event, momentum, depth, core.mv.z, core.mv.xy, spin, mass)
    image, _ = core.resolve((position, momentum, screen), spin, mass, camera_step_fraction, n_camera_steps,
                            core.polarized_rates, core.mv.z, inner_radius, disk_outer_radius, opacity, escape_radius,
                            capture_radius)
    points, momenta, screens = image.disk
    # The gas orbits with the hole, in its equatorial plane.
    shift = core.frequency_ratio(points, momenta, eye, core.mv.xy, spin, mass)
    # A blackbody shifted in frequency is a blackbody at the shifted temperature.
    temperature = shift * core.disk_temperature(image.disk_radii, core.mv.xy, spin, mass, disk_temperature_scale)
    coherency = core.disk_coherency(points, momenta, screens, core.mv.z, spin, mass, polarization_degree)
    analyzers = (core.mv.xy * (-polarizer_angles / 2)).exp() >> core.mv.x.dual()   # [analyzers] Screen
    return image, temperature, core.transmitted(coherency, analyzers[:, None])


def disk_flow(image: core.Image) -> Iterator[core.Antivector]:
    """Where the gas seen at each disk crossing was at each frame of one loop, and one loop before
    that, `[2, crossings] Antivector` per frame: the flow carries the disk's texture, while its light
    stays steady."""
    points, _, _ = image.disk
    spin = core.mv.xy * camera_spin
    for elapsed in np.linspace(0, flow_time, n_flow_frames, endpoint=False):
        yield core.carried(points, image.disk_radii, core.mv.xy, spin, mass, np.array([elapsed, elapsed - flow_time]))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.relativity.kerr import render

    paths, horizons = scene()
    save_animation(render.animate_rays(paths, horizons, extent), "kerr_spin", duration_ms)
    image, temperature, transmission = camera()
    sky = render.star_texture(sky_width, sky_height, sky_seed)
    shape = (image_height, image_width)
    save_figure(render.draw_camera(image, temperature, exposure_temperature, shape, sky), "kerr_camera")
    save_animation(render.animate_polarizer(image, temperature, exposure_temperature, transmission, shape, sky),
                   "kerr_polarizer", duration_ms, scale=web_scale)
    save_animation(render.animate_disk(image, temperature, exposure_temperature, disk_flow(image), n_flow_frames, shape, sky),
                   "kerr_disk", duration_ms, scale=web_scale)

    # --- checks
    # Each pixel's light is shared at most once between the disk and the sky, all of it when its ray
    # escapes; where the rays cross the disk they stay null, to the integrator's accuracy.
    pixel_count = image_width * image_height
    disk_share = np.bincount(image.disk_pixels, image.disk_weights.kernel[..., 0], pixel_count)
    sky_share = np.bincount(image.sky_pixels, image.sky_weights.kernel[..., 0], pixel_count)
    assert np.all(disk_share + sky_share <= 1 + 1e-13)
    np.testing.assert_allclose((disk_share + sky_share)[image.sky_pixels], 1, atol=1e-13, rtol=0)
    points, momenta, _ = image.disk
    tangent, _ = core.rates(points, momenta, core.mv.xy * camera_spin, mass)
    np.testing.assert_allclose((momenta & tangent).kernel, 0, atol=7e-2, rtol=0)


if __name__ == "__main__":
    main()
