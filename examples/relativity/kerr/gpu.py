"""A camera above a rotating black hole's disk, traced on the GPU by a shader written in numga.

The shader is WGSL in numga's dialect: the Kerr geometry, the rays and the disk read as `core` does, the
position gauge field `Antivector + dyad(geometry)` a value the shader passes around and applies, the
camera's screen a map from its two axes that rides along each ray, `-rotation(geometry,
heading).commutator(ray.screen)`. One invocation follows one pixel's light back from the camera,
through the disk, to the sky or into the hole, and returns its linear light; the frames of the orbiting
disk and of the rotating polarizer are each one more trace. Tables the shader looks up, the disk's rest
temperature and blackbody colours, come from `core` and `render`.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.relativity.kerr import core, render, scenarios
from examples.util.wgsl import Shader, sampled, supersampled

types = dict(Vector=core.Vector, Bivector=core.Bivector, Antivector=core.Antivector, Spin=core.Spin,
             Rotor=core.Rotor, Screen=core.Screen, ScreenSlice=core.ScreenSlice, ScreenFrame=core.ScreenFrame,
             Coherency=core.Coherency, GaugeField=core.GaugeField)
# The camera's own axes, which the shader takes as constants.
constants = dict(camera_forward=core.camera_forward, camera_right=core.camera_right, camera_up=core.camera_up)
mv = core.mv
n_table_entries = 256
# The colour table's entries, and its hottest temperature, past any the camera sees.
hottest = 3 * scenarios.disk_temperature_scale                                  # K


# --- math -----------------------------------------------------------------------------
geometry = """
const pi = 3.141592653589793;

// One unit along the lab's time, and the part of a direction within the lab's slice of space.
const time = -mv.t.dual();
const space = Antivector - time * (mv.t & Antivector);

var<uniform> spin: Spin;
var<uniform> mass: f32;

struct Field {
    radius: f32,
    profile: f32,
    null: Antivector,
    profile_gradient: Vector,
    null_gradient: GaugeField,
    null_curl: Bivector,
}

fn field(position: Antivector) -> Field {
    let spatial = space(position);
    // The plane the hole turns in, whose dual is its spin; the volume the position spans with it, and the
    // position's height off it.
    let turning = spin.dual_inverse();
    let volume = spatial.anticommutator(turning);
    let height = volume ^ turning;
    let spin_squared = spin.squared();
    let offset = spatial.squared() - spin_squared;
    let discriminant = (offset.squared() - 4 * volume.squared()).square_root();
    let radius_squared = (offset + discriminant) / 2;
    let radius = radius_squared.square_root();
    // Paired with a small step, how far the step changes the oblate radius.
    let radius_gradient = (radius_squared * spatial - height).dual() / (radius * discriminant);
    // The hole drags the outgoing direction against its turning.
    let direction = (radius * spatial - spatial.commutator(turning) - height / radius) / (radius_squared + spin_squared);
    let lever = spatial - 2 * radius * direction + height / radius_squared;
    let direction_gradient = (
        lever * (radius_gradient & Antivector)
        + radius * space - space.commutator(turning)
        - (space.anticommutator(turning) ^ turning) / radius
    ) / (radius_squared + spin_squared);
    let denominator = radius_squared.squared() - volume.squared();
    let profile = mass * radius**3 / denominator;
    let profile_gradient = profile * (3 * radius_gradient / radius
        - (4 * radius**3 * radius_gradient - 2 * height.dual()) / denominator);
    let null_curl = ((radius_gradient ^ lever.dual()) - 2 * turning) / (radius_squared + spin_squared);
    return Field(radius, profile, time - direction, profile_gradient, -direction_gradient, null_curl);
}

fn null_plane(geometry: Field) -> Vector {
    return -geometry.null.dual();
}

// The position gauge field: the identity plus the profile times the null dyad, which squares to zero.
fn dyad(geometry: Field) -> GaugeField {
    return geometry.profile * geometry.null * (null_plane(geometry) & Antivector);
}

fn gauge(geometry: Field) -> GaugeField {
    return Antivector + dyad(geometry);
}

fn gauge_inverse(geometry: Field) -> GaugeField {
    return Antivector - dyad(geometry);
}

// A momentum's direction in the local frame: raised flat and carried through the gauge field.
fn local(geometry: Field, momentum: Vector) -> Antivector {
    return gauge(geometry)(-momentum.dual());
}

fn proper_squared(geometry: Field, direction: Antivector) -> f32 {
    return -gauge_inverse(geometry)(direction).squared();
}

// The gradient of a momentum's pairing with the null direction: the momentum pulled back.
fn overlap_gradient(geometry: Field, momentum: Vector) -> Vector {
    return geometry.null_gradient.adjugate()(momentum);
}
"""

rays = """
// A ray: its event, its momentum, and the camera's screen carried along it.
struct Ray {
    position: Antivector,
    momentum: Vector,
    screen: ScreenFrame,
}

// The rotation gauge field: the plane the local frame turns in per unit step along a direction.
fn rotation(geometry: Field, direction: Antivector) -> Bivector {
    let null = null_plane(geometry);
    let along = null & direction;
    let turning = geometry.profile_gradient * along + geometry.profile * overlap_gradient(geometry, -direction.dual());
    return -(turning ^ null) - geometry.profile * geometry.null_curl * along;
}

// Hamiltonian motion and parallel transport of the screen, per unit of decreasing coordinate time.
fn rates(ray: Ray) -> Ray {
    let geometry = field(ray.position);
    let overlap = ray.momentum & geometry.null;
    let heading = local(geometry, ray.momentum);
    let tangent = gauge(geometry)(heading);
    let gradient = geometry.profile_gradient * overlap.squared()
        + 2 * geometry.profile * overlap * overlap_gradient(geometry, ray.momentum);
    let screen_rate = -rotation(geometry, heading).commutator(ray.screen);
    return Ray(tangent, -gradient, screen_rate) / -(mv.t & tangent);
}

fn runge_kutta(ray: Ray, step: f32) -> Ray {
    let first = rates(ray);
    let second = rates(ray + first * (step / 2));
    let third = rates(ray + second * (step / 2));
    let fourth = rates(ray + third * step);
    return ray + (first + 2 * second + 2 * third + fourth) * (step / 6);
}
"""

disk = """
// The disk lies in the plane `orbit`, its height measured by `normal`.
const orbit = mv.xy;
const normal = mv.z;

var<uniform> outer_radius: f32;
var<uniform> inner_depth: f32;
var<uniform> polarization_degree: f32;
var<uniform> clump_contrast: f32;
var<uniform> elapsed: f32;
var<uniform> flow_time: f32;
var<uniform> fade: f32;
var<uniform> hottest: f32;
// The disk's temperature scale, (accretion rate c^6 / (σ G^2 M^2))^(1/4), and blackbody light up to the hottest.
var<uniform> temperature_scale: f32;
var blackbodies: texture_2d<f32>;
var filtered: sampler;
// One of the disk's random waves: its whole turns around the disk, waves per unit radius across it, and phase.
struct Mode {
    turns: f32,
    wavenumber: f32,
    phase: f32,
}

var<storage, read> modes: array<Mode>;

// The radius of the innermost stable circular orbit in the disk's plane, the spin counted along the orbit, negative
// for gas orbiting against the hole: gas orbiting with the hole reaches inward of the radius without spin, gas
// orbiting against it outward (Bardeen, Press and Teukolsky, 1972).
fn inner_radius() -> f32 {
    let ratio = (orbit & spin) / mass;
    let first = 1 + pow(1 - ratio * ratio, 1.0 / 3) * (pow(1 + ratio, 1.0 / 3) + pow(1 - ratio, 1.0 / 3));
    let second = sqrt(3 * ratio * ratio + first * first);
    let reach = sqrt((3 - first) * (3 + first + 2 * second));
    return mass * (3 + second - select(reach, -reach, ratio < 0));
}

fn orbital_speed(radius: f32) -> f32 {
    let alignment = orbit & spin;
    return mass.square_root() / (radius * radius.square_root() + alignment * mass.square_root());
}

// The unit velocity of the gas orbiting at an event: a turned point moves by its commutator with the plane.
fn orbiting_gas(position: Antivector, radius: f32) -> Antivector {
    let flow = time + orbital_speed(radius) * space(position).commutator(orbit);
    return flow / proper_squared(field(position), flow).square_root();
}

// The share of a ray's light the disk absorbs where it crosses: its depth over the slant the gas sees.
fn disk_opacity(crossing: Ray, radius: f32) -> f32 {
    let inner = inner_radius();
    let fraction = (radius - inner) / (outer_radius - inner);
    let depth = inner_depth * (1 - fraction) * (1 - fraction);
    let gas = orbiting_gas(crossing.position, radius);
    let slant = abs((crossing.momentum & gas) / (crossing.momentum & normal.dual()));
    return 1 - exp(-depth * slant);
}

// The disk's random waves where the gas seen at `position` was a time `before` ago. The clock starts when light leaves
// the hole's centre for the camera at its `distance`, so that moving the camera does not turn the disk.
fn waves(position: Antivector, radius: f32, before: f32) -> f32 {
    let angle = orbital_speed(radius) * (elapsed - before + (mv.t & position) + distance);
    let carried = (orbit * (angle / 2)).exp() >> position;
    let around = atan2(mv.y & carried, mv.x & carried);
    var total = 0.0;
    for (var index = 0u; index < arrayLength(&modes); index++) {
        let mode = modes[index];
        total += sin(mode.turns * around + mode.wavenumber * radius + mode.phase);
    }
    return total / sqrt(f32(arrayLength(&modes)) / 2);
}

// A relativistic thin disk's temperature at an oblate radius, from the flux of Page and Thorne (1974): zero at the
// innermost stable orbit, falling as the radius to the power -3/4 far out.
fn rest_temperature(radius: f32) -> f32 {
    let ratio = (orbit & spin) / mass;
    let root_radius = sqrt(radius / mass);
    let edge = sqrt(inner_radius() / mass);
    // The roots of the cubic `root_radius**3 - 3 * root_radius + 2 * ratio`, and the weight of each in the torque.
    let roots = 2 * cos((acos(ratio) + vec3<f32>(-pi, pi, 3 * pi)) / 3);
    let weights = (roots - ratio) * (roots - ratio) / (roots * (roots * roots - 1));
    // The torque the disk carries, zero at the innermost stable orbit.
    let terms = weights * log((root_radius - roots) / (edge - roots));
    let torque = max(root_radius - edge - 1.5 * ratio * log(root_radius / edge) - (terms.x + terms.y + terms.z), 0.0);
    let cubic = root_radius * root_radius * root_radius - 3 * root_radius + 2 * ratio;
    let flux = 3 / (8 * pi) * torque / (root_radius * root_radius * root_radius * root_radius * cubic);
    return temperature_scale * sqrt(sqrt(flux));
}

// Linear light the disk emits towards the camera: a blackbody at the shifted temperature, textured.
fn disk_light(crossing: Ray, radius: f32) -> vec3<f32> {
    let gas = orbiting_gas(crossing.position, radius);
    // A blackbody seen at a shifted frequency is a blackbody at the shifted temperature.
    let shift = (crossing.momentum & at_rest()) / (crossing.momentum & gas);
    // The table's entries sit at its texel centres, from zero to the hottest temperature.
    let entries = f32(textureDimensions(blackbodies).x);
    let entry = clamp(shift * rest_temperature(radius) / hottest, 0.0, 1.0) * (entries - 1) + 0.5;
    let colour = textureSampleLevel(blackbodies, filtered, vec2<f32>(entry / entries, 0.5), 0.0).rgb;
    let rings = 0.90 + 0.045 * sin(11 * radius) + 0.025 * sin(19 * radius);
    let clumps = cos(fade) * waves(crossing.position, radius, 0.0) + sin(fade) * waves(crossing.position, radius, flow_time);
    return colour * rings * exp(clump_contrast * clumps);
}

// The disk's emission, a covariance from the camera's slices to screen directions: partly polarized square to the
// magnetic field circling the normal.
fn disk_coherency(crossing: Ray) -> Coherency {
    let heading = local(field(crossing.position), crossing.momentum);
    let direction = space(heading) / (mv.t & heading);
    let meridian = space(crossing.position).dual() ^ normal;
    let polarization_axis = meridian.commutator(direction) / meridian.norm();
    let polarization = crossing.screen.adjoint()(polarization_axis);
    let polarized_fraction = polarization_degree * polarization.squared();
    // Unpolarized light is round, where the metric enters; the polarized part weights the polarization by each
    // slice's reading of it.
    return (1 - polarized_fraction) / 2 * ScreenSlice.dual() + polarization_degree * polarization * (ScreenSlice & polarization);
}
"""

camera = """
var<uniform> orientation: Rotor;
var<uniform> distance: f32;
var<uniform> analyzer: ScreenSlice;
var<uniform> half_view: f32;
var<uniform> step_fraction: f32;
var<uniform> n_steps: u32;
var<uniform> escape_radius: f32;
var stars: texture_2d<f32>;

// What a ray brings back: the disk's light, the share of it an analyzer passes, and the starlight.
struct Sample {
    light: vec3<f32>,
    analyzed: vec3<f32>,
    sky: vec3<f32>,
}

// The camera's eye, at its distance from the hole along its own backward direction.
fn eye() -> Antivector {
    return (orientation >> mv.z.dual()) * distance;
}

// The camera at rest: a unit along the lab's time measured in proper time at the eye.
fn at_rest() -> Antivector {
    return time / proper_squared(field(eye()), time).square_root();
}

// The radius inside every photon orbit, where no ray turns back out.
fn capture_radius() -> f32 {
    return 1.05 * (mass + sqrt(mass * mass - spin.squared()));
}

// The static camera's rest frame carried into the local Kerr–Schild frame.
fn camera_boost(geometry: Field) -> Rotor {
    let carried_time = gauge_inverse(geometry)(time);
    let observer = carried_time / (-carried_time.squared()).square_root();
    return (1 - observer * time).normalized();
}

// The camera's own right and up across a ray, as a map from the screen's two axes.
fn camera_screen(position: Antivector, momentum: Vector) -> ScreenFrame {
    let geometry = field(position);
    let boost = camera_boost(geometry);
    let heading = boost.reverse() >> local(geometry, momentum);
    let direction = space(heading) / -(mv.t & heading);
    // The turn carrying the camera's forward onto the direction it looks to see the ray: the reflection in the
    // forward, then in the direction halfway between the two. It turns the camera's right and up with it.
    let forward = orientation >> camera_forward;
    let halfway = direction + forward;
    let turn = halfway * forward / halfway.squared().square_root();
    return (boost * turn * orientation) >> (camera_right * (mv.x & Screen) + camera_up * (mv.y & Screen));
}

// The starlight a ray escaping along a direction brings: the sky's image at its longitude and latitude.
fn starlight(direction: Antivector) -> vec3<f32> {
    let longitude = atan2(mv.y & direction, mv.x & direction);
    let latitude = asin((mv.z & direction) / space(direction).squared().square_root());
    let place = vec2<f32>(longitude / (2 * pi) + 0.5, 0.5 - latitude / pi);
    return textureSampleLevel(stars, filtered, place, 0.0).rgb;
}

// The light along the ray through a place on the image, followed back from the camera.
fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_view;
    // Past-directed light from the pixel, looking along the camera's -z.
    var direction = orientation >> (camera_forward + camera_right * across.x + camera_up * across.y);
    direction = direction / direction.squared().square_root();
    let start = eye();
    let geometry = field(start);
    let heading = camera_boost(geometry) >> (time - direction);
    let future_momentum = -gauge_inverse(geometry)(heading).dual();
    let momentum = -future_momentum / (future_momentum & time);
    var ray = Ray(start, momentum, camera_screen(start, momentum));
    let inner = inner_radius();
    let capture = capture_radius();

    var radius = field(ray.position).radius;
    var remaining = 1.0;
    var disk = vec3<f32>(0.0);
    var through = vec3<f32>(0.0);
    var stars_seen = vec3<f32>(0.0);
    for (var step = 0u; step < n_steps; step++) {
        let end = runge_kutta(ray, step_fraction * radius);
        let end_radius = field(end.position).radius;
        // Where the step crosses the disk's plane, between its edges, the disk takes a share of the light.
        if ((normal & ray.position) * (normal & end.position) <= 0) {
            let crossing = ray + (end - ray) * (-(normal & ray.position) / (normal & (end.position - ray.position)));
            let crossing_radius = field(crossing.position).radius;
            if (crossing_radius >= inner && crossing_radius <= outer_radius) {
                let weight = remaining * disk_opacity(crossing, crossing_radius);
                let emitted = weight * disk_light(crossing, crossing_radius);
                disk += emitted;
                through += emitted * (analyzer & disk_coherency(crossing)(analyzer));
                remaining -= weight;
            }
        }
        if (end_radius > escape_radius) {
            let geometry = field(end.position);
            stars_seen = remaining * starlight(gauge(geometry)(local(geometry, end.momentum)));
            break;
        }
        if (end_radius < capture) {
            break;
        }
        ray = end;
        radius = end_radius;
    }
    return Sample(disk, through, stars_seen);
}
"""


def shader() -> Shader:
    """The camera's shader, compiled, for pictures."""
    return Shader(geometry + rays + disk + camera + supersampled(), core.ga, types, constants)


def live_shader() -> Shader:
    """The camera's shader for tracing live, one sample a pixel: resolution, not samples, is what a live view spends."""
    return Shader(geometry + rays + disk + camera + sampled(1), core.ga, types, constants)


def scene() -> dict:
    """The values the shader reads for the still: hole, camera, disk, tables and sky. The shader finds the rest from
    them: the camera's eye and rest frame, and the disk's inner edge and temperatures."""
    spin = mv.xy.dual() * scenarios.camera_spin                                 # [] Spin
    orientation = (mv.zx * (-scenarios.camera_inclination / 2)).exp()           # [] Rotor
    return dict(
        spin=spin, mass=scenarios.mass, orientation=orientation, distance=scenarios.camera_distance,
        outer_radius=scenarios.disk_outer_radius, inner_depth=scenarios.inner_depth,
        polarization_degree=scenarios.polarization_degree, clump_contrast=render.clump_contrast,
        elapsed=0.0, flow_time=scenarios.flow_time, fade=0.0, analyzer=mv.x,
        hottest=hottest, temperature_scale=scenarios.disk_temperature_scale,
        blackbodies=render.blackbody_table(hottest, n_table_entries), modes=render.wave_modes(),
        half_view=scenarios.half_view,
        step_fraction=scenarios.camera_step_fraction, n_steps=scenarios.n_camera_steps,
        escape_radius=scenarios.escape_radius,
        stars=render.star_texture(scenarios.sky_width, scenarios.sky_height, scenarios.sky_seed))


def image_shape() -> tuple[int, int]:
    """The traced image, rows first."""
    return scenarios.image_height, scenarios.image_width


def disk_flow(tracer: Shader, values: dict) -> Iterator[dict]:
    """The traced light at each frame of one loop of the disk's orbit: its texture carried by the gas,
    turned from this loop's waves into the previous loop's to close the loop."""
    for index, elapsed in enumerate(np.linspace(0, scenarios.flow_time, scenarios.n_flow_frames, endpoint=False)):
        yield tracer(image_shape(), **values | dict(elapsed=elapsed, fade=np.pi / 2 * index / scenarios.n_flow_frames))


def polarizer(tracer: Shader, values: dict) -> Iterator[dict]:
    """The traced light through each analyzer of a polarizer turning in front of the camera."""
    for analyzer in (mv.xy * (-scenarios.polarizer_angles / 2)).exp() >> mv.x:
        yield tracer(image_shape(), **values | dict(analyzer=analyzer))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure

    tracer = shader()
    values = scene()
    still = tracer(image_shape(), **values)
    save_figure(render.draw_frame(render.develop(still["image"]["light"], still["image"]["sky"], scenarios.exposure_temperature)),
                "kerr_gpu_camera")
    save_animation((render.develop(traced["image"]["light"], traced["image"]["sky"], scenarios.exposure_temperature)
                    for traced in disk_flow(tracer, values)), "kerr_gpu_disk", scenarios.duration_ms)
    # The exposure is doubled for the half of unpolarized light a polarizer stops.
    save_animation((render.develop(2 * traced["image"]["analyzed"], traced["image"]["sky"], scenarios.exposure_temperature)
                    for traced in polarizer(tracer, values)), "kerr_gpu_polarizer", scenarios.duration_ms)

    # --- checks
    # Every pixel's light is finite and none is negative.
    for traced in still["image"].values():
        assert np.all(np.isfinite(traced)) and np.all(traced >= 0)


if __name__ == "__main__":
    main()
