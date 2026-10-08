"""An equatorial spin sweep and a camera above a rotating black hole's disk."""

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.relativity.kerr import core

MASS = 1.0
FRAMES = 80
MAX_SPIN = 0.8 * MASS
SPINS = -MAX_SPIN * np.cos(np.linspace(0, 2 * np.pi, FRAMES, endpoint=False))
DURATION_MS = 80
EXTENT = 12.0 * MASS
RAYS = 31
OFFSETS = np.linspace(-8, 8, RAYS) * MASS
STEP_SIZE = 0.04 * MASS
STEPS = 1100
STOP_FRACTION = 0.8
RING_SAMPLES = 128
CAMERA_SPIN = 0.8 * MASS
CAMERA_DISTANCE = 28.0 * MASS
CAMERA_INCLINATION = np.deg2rad(75.0)
HALF_VIEW = np.tan(np.deg2rad(19.0))
IMAGE_WIDTH = 480
IMAGE_HEIGHT = IMAGE_WIDTH * 2 // 3
CAMERA_STEP_FRACTION = 0.1
CAMERA_STEPS = 720
DISK_INNER_RADIUS = core.innermost_stable_orbit(CAMERA_SPIN, MASS)   # where the thin disk ends
DISK_OUTER_RADIUS = 12.0 * MASS
# The disk's optical depth straight through its thickness, at its inner edge.
DISK_DEPTH = 4.0
DISK_TEMPERATURE_SCALE = 58500.0                               # K, (accretion rate c^6 / (σ G^2 M^2))^(1/4)
EXPOSURE_TEMPERATURE = 6500.0                                 # K, shown at mid-grey
ESCAPE_RADIUS = 100.0 * MASS
CAPTURE_RADIUS = 1.002 * (MASS + np.sqrt(MASS**2 - CAMERA_SPIN**2))
SKY_WIDTH = 2048
SKY_HEIGHT = SKY_WIDTH // 2
SKY_SEED = 8
POLARIZER_FRAMES = 80
POLARIZER_ANGLES = np.linspace(0, np.pi, POLARIZER_FRAMES, endpoint=False)
POLARIZATION_DEGREE = 0.8
FLOW_FRAMES = 80
WEB_SCALE = 2 / 3                                               # the camera's animations, box-filtered to this share of its size
FLOW_TIME = 50.0 * MASS                                         # coordinate time the disk turns through in one loop


# --- math -----------------------------------------------------------------------------
def scene() -> tuple[core.Vector, core.Vector]:
    spin = core.mv.xy * SPINS[:, None]                         # [frames, 1] Spin
    position = -core.mv.x * EXTENT + core.mv.y * OFFSETS        # [rays] Vector
    momentum = core.initial_momentum(position, core.mv.x, spin, MASS)
    position = position.broadcast_to(momentum.shape)
    paths = stack(tuple(position for position, _ in core.trace(
        position, momentum, spin, MASS, STEP_SIZE, STEPS, STOP_FRACTION, core.rates)))

    # A constant oblate radius cuts the equatorial plane in this circle.
    horizon = MASS + np.sqrt(MASS**2 - SPINS**2)
    radius = np.sqrt(horizon**2 + SPINS**2)
    angles = np.linspace(0, 2 * np.pi, RING_SAMPLES + 1)
    turns = (core.mv.xy * (angles / 2)).exp()
    horizons = (turns >> core.mv.x) * radius[:, None]
    return paths, horizons


def camera() -> tuple[core.Image, core.Scalar, core.Scalar]:
    """The camera's view, resolved ray by ray; each disk crossing's observed temperature, `[crossings]
    Scalar`; and the power its emission passes through each analyzer, `[analyzers, crossings]
    Scalar`."""
    orientation = (core.mv.zx * (CAMERA_INCLINATION / 2)).exp()
    eye = (orientation >> core.mv.z) * CAMERA_DISTANCE
    spin = core.mv.xy * CAMERA_SPIN
    position, momentum = core.camera_rays(eye, orientation, IMAGE_WIDTH, IMAGE_HEIGHT,
                                         HALF_VIEW, spin, MASS)
    screen = core.camera_screen(position, momentum, orientation, spin, MASS)
    steps = core.camera_trace((position, momentum, screen), spin, MASS,
                              CAMERA_STEP_FRACTION, CAMERA_STEPS, core.polarized_rates)
    def opacity(event: core.Vector, momentum: core.Vector, radius: core.Scalar) -> core.Scalar:
        depth = core.disk_depth(radius, DISK_INNER_RADIUS, DISK_OUTER_RADIUS, DISK_DEPTH)
        return core.disk_opacity(event, momentum, depth, core.mv.z, core.mv.xy, spin, MASS)
    image = core.resolve(steps, IMAGE_WIDTH * IMAGE_HEIGHT, core.mv.z, DISK_INNER_RADIUS, DISK_OUTER_RADIUS, opacity,
                         ESCAPE_RADIUS, CAPTURE_RADIUS, spin, MASS)
    points, momenta, screens = image.disk
    # The gas orbits with the hole, in its equatorial plane.
    shift = core.frequency_ratio(points, momenta, eye, core.mv.xy, spin, MASS)
    # A blackbody shifted in frequency is a blackbody at the shifted temperature.
    temperature = shift * core.disk_temperature(image.disk_radii, CAMERA_SPIN, MASS, DISK_TEMPERATURE_SCALE)
    coherency = core.disk_coherency(points, momenta, screens, core.mv.z, spin, MASS, POLARIZATION_DEGREE)
    analyzers = (core.mv.xy * (POLARIZER_ANGLES / 2)).exp() >> core.mv.x       # [analyzers] Screen
    return image, temperature, core.transmitted(coherency, analyzers[:, None])


def disk_flow(image: core.Image) -> Iterator[core.Vector]:
    """Where the gas seen at each disk crossing was at each frame of one loop, and one loop before
    that, `[2, crossings] Vector` per frame: the flow carries the disk's texture, while its light
    stays steady."""
    points, _, _ = image.disk
    spin = core.mv.xy * CAMERA_SPIN
    for elapsed in np.linspace(0, FLOW_TIME, FLOW_FRAMES, endpoint=False):
        yield core.carried(points, image.disk_radii, core.mv.xy, spin, MASS, np.array([elapsed, elapsed - FLOW_TIME]))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.relativity.kerr import render

    paths, horizons = scene()
    save_animation(render.animate_rays(paths, horizons, EXTENT), "kerr_spin", DURATION_MS)
    image, temperature, transmission = camera()
    sky = render.star_texture(SKY_WIDTH, SKY_HEIGHT, SKY_SEED)
    shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    save_figure(render.draw_camera(image, temperature, EXPOSURE_TEMPERATURE, shape, sky), "kerr_camera")
    save_animation(render.animate_polarizer(image, temperature, EXPOSURE_TEMPERATURE, transmission, shape, sky),
                   "kerr_polarizer", DURATION_MS, scale=WEB_SCALE)
    save_animation(render.animate_disk(image, temperature, EXPOSURE_TEMPERATURE, disk_flow(image), FLOW_FRAMES, shape, sky),
                   "kerr_disk", DURATION_MS, scale=WEB_SCALE)


if __name__ == "__main__":
    main()
