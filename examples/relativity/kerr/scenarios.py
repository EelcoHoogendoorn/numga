"""Equatorial spin sweep and camera scenes above a rotating Kerr black hole's accretion disk."""

from collections.abc import Iterator

import numpy as np

from examples.relativity.kerr import core
from examples.relativity.kerr.plumbing import Image

mass = 1.0
n_frames = 160
n_sweep_frames = n_frames // 2 + 1
max_spin = 0.8 * mass
# One sweep of the spin from one sense to the other; the animation plays it forward and back.
spins = -max_spin * np.cos(np.linspace(0, np.pi, n_sweep_frames))
frame_duration_ms = 40
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
n_image_width = 320
n_image_height = n_image_width * 2 // 3
camera_step_fraction = 0.2
n_camera_steps = 720
disk_outer_radius = 12.0 * mass
# The disk's optical depth straight through its thickness, at its inner edge.
inner_depth = 4.0
# In kelvin: (accretion rate c^6 / (sigma G^2 M^2))^(1/4).
disk_temperature_scale = 58500.0
# In kelvin, shown at mid-grey.
exposure_temperature = 6500.0
escape_radius = 100.0 * mass
# Captured inside every photon orbit, where no ray turns back out.
capture_radius = 1.05 * (mass + np.sqrt(mass**2 - camera_spin**2))
n_sky_width = 2048
n_sky_height = n_sky_width // 2
sky_seed = 8
n_polarizer_frames = n_frames
polarizer_angles = np.linspace(0, np.pi, n_polarizer_frames, endpoint=False)
polarization_degree = 0.8
n_flow_frames = n_frames
flow_time = 50.0 * mass                                         # coordinate time the disk turns through in one loop
# The gallery's camera: its pixels across and down, the tangent of half its view, its frames and its colours.
n_gallery_width = 400
n_gallery_height = n_gallery_width * 2 // 3
gallery_half_view = np.tan(np.deg2rad(21.3))
n_gallery_frames = n_flow_frames
n_gallery_colours = 64


# --- math -----------------------------------------------------------------------------
def spin_sweep_scene(n_trace_steps: int) -> tuple[core.Antivector, core.Antivector]:
    """Traces equatorial light rays (`[n_trace_steps + 1, n_sweep_frames, n_rays] Antivector-`) and horizon outlines (`[n_sweep_frames, n_ring_samples] Antivector-`)."""
    x, y = core.mv.x.dual(), core.mv.y.dual()
    spin = core.mv.xy.dual() * spins[:, None]                                               # [n_sweep_frames, 1] Spin-
    position = -x * extent + y * offsets                                                    # [n_rays] Antivector-
    momentum = core.initial_momentum(position, x, spin, mass)                               # [n_sweep_frames, n_rays] Vector+
    position = position.broadcast_to(momentum.shape)                                        # [n_sweep_frames, n_rays] Antivector-
    paths, _ = core.trace(position, momentum, spin, mass, step_size, n_trace_steps, stop_fraction, core.rates)

    # A constant oblate radius cuts the equatorial plane in this circle.
    horizon = mass + np.sqrt(mass**2 - spins**2)
    radius = np.sqrt(horizon**2 + spins**2)
    angles = np.linspace(0, 2 * np.pi, n_ring_samples, endpoint=False)
    turns = (core.mv.xy * (-angles / 2)).exp()                                              # [n_ring_samples] Rotor+
    horizons = (turns >> x) * radius[:, None]                                               # [n_sweep_frames, n_ring_samples] Antivector-
    return paths, horizons


def camera_scene(n_width: int, n_height: int, view: float, n_trace_steps: int) -> tuple[Image, core.Scalar, core.Scalar]:
    """Resolves a `n_width` by `n_height` camera view, disk temperatures (`[...] Scalar+`), and analyzer transmissions (`[n_polarizer_frames, ...] Scalar+`)."""
    orientation = (core.mv.zx * (-camera_inclination / 2)).exp()                            # [] Rotor+
    eye = (orientation >> core.mv.z.dual()) * camera_distance                               # [] Antivector-
    spin = core.mv.xy.dual() * camera_spin                                                  # [] Spin-
    # The thin disk ends at the innermost stable orbit of gas turning with the hole.
    inner_radius = core.innermost_stable_orbit(core.mv.xy, spin, mass)                      # [] Scalar+
    position, momentum = core.camera_rays(eye, orientation, n_width, n_height, view, spin, mass)
    screen = core.camera_screen(position, momentum, orientation, spin, mass)

    def opacity(event: core.Antivector, ray_momentum: core.Vector, radius: core.Scalar) -> core.Scalar:
        depth = core.disk_depth(radius, inner_radius, disk_outer_radius, inner_depth)
        return core.disk_opacity(event, ray_momentum, depth, core.mv.z, core.mv.xy, spin, mass)

    image, _ = core.resolve((position, momentum, screen), spin, mass, camera_step_fraction, n_trace_steps,
                            core.polarized_rates, core.mv.z, inner_radius, disk_outer_radius, opacity, escape_radius,
                            capture_radius)
    points, momenta, screens = image.disk
    # The gas orbits with the hole, in its equatorial plane.
    shift = core.frequency_ratio(points, momenta, eye, core.mv.xy, spin, mass)              # [...] Scalar+
    # A blackbody shifted in frequency is a blackbody at the shifted temperature.
    temperature = shift * core.disk_temperature(image.disk_radii, core.mv.xy, spin, mass, disk_temperature_scale)
    coherency = core.disk_coherency(points, momenta, screens, core.mv.z, spin, mass, polarization_degree)
    analyzer_rotors = (core.mv.xy * (-polarizer_angles / 2)).exp()                          # [n_polarizer_frames] Rotor+
    analyzers = analyzer_rotors >> core.mv.x                                                # [n_polarizer_frames] ScreenSlice+
    return image, temperature, core.transmitted_power(coherency, analyzers[:, None])


def disk_flow_states(image: Image, n_loop_frames: int) -> Iterator[core.Antivector]:
    """Yields retrodicted disk crossing positions (`[2, ...] Antivector-`) for the current and preceding orbital loops."""
    points, _, _ = image.disk
    spin = core.mv.xy.dual() * camera_spin                                                  # [] Spin-
    for elapsed in np.linspace(0, flow_time, n_loop_frames, endpoint=False):
        yield core.orbital_origins(points, image.disk_radii, core.mv.xy, spin, mass, camera_distance,
                                   np.array([elapsed, elapsed - flow_time]))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Runs the equatorial spin sweep and static, polarized, and orbiting disk camera scenes and saves their outputs."""
    from examples.util.animation import save_animation, save_figure
    from examples.relativity.kerr import render

    paths, horizons = spin_sweep_scene(n_steps)
    save_animation(render.ray_animation(paths, horizons, extent), "kerr_spin", frame_duration_ms, forward_then_back=True)
    image, temperature, transmission = camera_scene(n_image_width, n_image_height, half_view, n_camera_steps)
    sky = render.star_texture(n_sky_width, n_sky_height, sky_seed)
    shape = (n_image_height, n_image_width)
    save_figure(render.camera_figure(image, temperature, exposure_temperature, shape, sky), "kerr_camera")
    save_animation(render.polarizer_animation(image, temperature, exposure_temperature, transmission, shape, sky),
                   "kerr_polarizer", frame_duration_ms)
    flow_states = list(disk_flow_states(image, n_flow_frames))
    save_animation(render.disk_animation(image, temperature, exposure_temperature, flow_states, n_flow_frames, shape, sky),
                   "kerr_disk", frame_duration_ms)

    # checks
    # Each pixel's light is shared at most once between the disk and the sky, all of it when its ray
    # escapes; where the rays cross the disk they stay null, to the integrator's accuracy.
    n_pixels = n_image_width * n_image_height
    disk_share = np.bincount(image.disk_pixels, image.disk_weights.kernel[..., 0], n_pixels)
    sky_share = np.bincount(image.sky_pixels, image.sky_weights.kernel[..., 0], n_pixels)
    assert np.all(disk_share + sky_share <= 1 + 1e-12)
    assert np.all(np.abs((disk_share + sky_share)[image.sky_pixels] - 1) < 1e-12)
    points, momenta, _ = image.disk
    tangent, _ = core.rates(points, momenta, core.mv.xy.dual() * camera_spin, mass)
    assert ((momenta & tangent).abs() < 7e-2).all()
    assert ((flow_states[0][0] - flow_states[0][1]).antinorm() > 0).all()


def pinned() -> None:
    """Renders the repository-tracked gallery animation of the orbiting Kerr accretion disk."""
    from examples.util.animation import pin_animation
    from examples.relativity.kerr import render

    sky = render.star_texture(n_sky_width, n_sky_height, sky_seed)
    gallery_image, gallery_temperature, _ = camera_scene(n_gallery_width, n_gallery_height, gallery_half_view, n_camera_steps)
    pin_animation(render.disk_animation(gallery_image, gallery_temperature, exposure_temperature,
                                        disk_flow_states(gallery_image, n_gallery_frames), n_gallery_frames,
                                        (n_gallery_height, n_gallery_width), sky),
                  "kerr", frame_duration_ms, colors=n_gallery_colours)


if __name__ == "__main__":
    main()
