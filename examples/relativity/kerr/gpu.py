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
from examples.wgsl import Shader, supersampled

types = dict(Vector=core.Vector, Bivector=core.Bivector, Antivector=core.Antivector, Spin=core.Spin,
             Rotor=core.Rotor, Screen=core.Screen, ScreenFrame=core.ScreenFrame, Coherency=core.Coherency,
             GaugeField=core.GaugeField)
mv = core.mv
# Each pixel averages the light of this many samples a side, before exposure.
supersampling = 2
n_table_entries = 256
# The colour table's hottest temperature, past any the camera sees.
hottest = 3 * scenarios.disk_temperature_scale                                  # K


# --- math -----------------------------------------------------------------------------
geometry = """
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
    // The volume the position spans with the spin plane, and the position's height off that plane.
    let volume = spatial.anticommutator(spin);
    let height = volume ^ spin;
    let spin_squared = -spin.squared();
    let offset = spatial.squared() - spin_squared;
    let discriminant = (offset.squared() - 4 * volume.squared()).square_root();
    let radius_squared = (offset + discriminant) / 2;
    let radius = radius_squared.square_root();
    // Paired with a small step, how far the step changes the oblate radius.
    let radius_gradient = (radius_squared * spatial - height).dual() / (radius * discriminant);
    // The hole drags the outgoing direction against its turning.
    let direction = (radius * spatial - spatial.commutator(spin) - height / radius) / (radius_squared + spin_squared);
    let lever = spatial - 2 * radius * direction + height / radius_squared;
    let direction_gradient = (
        lever * (radius_gradient & Antivector)
        + radius * space - space.commutator(spin)
        - (space.anticommutator(spin) ^ spin) / radius
    ) / (radius_squared + spin_squared);
    let denominator = radius_squared.squared() - volume.squared();
    let profile = mass * radius**3 / denominator;
    let profile_gradient = profile * (3 * radius_gradient / radius
        - (4 * radius**3 * radius_gradient - 2 * height.dual()) / denominator);
    let null_curl = ((radius_gradient ^ lever.dual()) - 2 * spin) / (radius_squared + spin_squared);
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

var<uniform> at_rest: Antivector;
var<uniform> inner_radius: f32;
var<uniform> outer_radius: f32;
var<uniform> inner_depth: f32;
var<uniform> polarization_degree: f32;
var<uniform> clump_contrast: f32;
var<uniform> elapsed: f32;
var<uniform> flow_time: f32;
var<uniform> fade: f32;
var<uniform> hottest: f32;
// The rest temperature at evenly spaced radii across the disk, and blackbody light up to the hottest.
var<storage, read> temperatures: array<f32>;
var blackbodies: texture_2d<f32>;
var filtered: sampler;
// The disk's random waves: whole turns around it, waves per unit radius across it, and phase.
var<storage, read> modes: array<vec4<f32>>;

fn orbital_speed(radius: f32) -> f32 {
    let alignment = -spin.scalar_product(orbit);
    return mass.square_root() / (radius * radius.square_root() + alignment * mass.square_root());
}

// The unit velocity of the gas orbiting at an event: a turned point moves by its commutator with the plane.
fn orbiting_gas(position: Antivector, radius: f32) -> Antivector {
    let flow = time + orbital_speed(radius) * space(position).commutator(orbit);
    return flow / proper_squared(field(position), flow).square_root();
}

// The share of a ray's light the disk absorbs where it crosses: its depth over the slant the gas sees.
fn disk_opacity(crossing: Ray, radius: f32) -> f32 {
    let fraction = (radius - inner_radius) / (outer_radius - inner_radius);
    let depth = inner_depth * (1 - fraction) * (1 - fraction);
    let gas = orbiting_gas(crossing.position, radius);
    let slant = abs((crossing.momentum & gas) / (crossing.momentum & normal.dual()));
    return 1 - exp(-depth * slant);
}

// The disk's random waves where the gas seen at `position` was a time `before` ago.
fn waves(position: Antivector, radius: f32, before: f32) -> f32 {
    let angle = orbital_speed(radius) * (elapsed - before + (mv.t & position));
    let carried = (orbit * (-angle / 2)).exp() >> position;
    let around = atan2(mv.y & carried, mv.x & carried);
    var total = 0.0;
    for (var index = 0u; index < arrayLength(&modes); index++) {
        let mode = modes[index];
        total += sin(mode.x * around + mode.y * radius + mode.z);
    }
    return total / sqrt(f32(arrayLength(&modes)) / 2);
}

fn rest_temperature(radius: f32) -> f32 {
    let place = (radius - inner_radius) / (outer_radius - inner_radius) * f32(arrayLength(&temperatures) - 1);
    let index = min(u32(place), arrayLength(&temperatures) - 2);
    return mix(temperatures[index], temperatures[index + 1], place - f32(index));
}

// Linear light the disk emits towards the camera: a blackbody at the shifted temperature, textured.
fn disk_light(crossing: Ray, radius: f32) -> vec3<f32> {
    let gas = orbiting_gas(crossing.position, radius);
    // A blackbody seen at a shifted frequency is a blackbody at the shifted temperature.
    let shift = (crossing.momentum & at_rest) / (crossing.momentum & gas);
    // The table's entries sit at its texel centres, from zero to the hottest temperature.
    let entries = f32(textureDimensions(blackbodies).x);
    let entry = clamp(shift * rest_temperature(radius) / hottest, 0.0, 1.0) * (entries - 1) + 0.5;
    let colour = textureSampleLevel(blackbodies, filtered, vec2<f32>(entry / entries, 0.5), 0.0).rgb;
    let rings = 0.90 + 0.045 * sin(11 * radius) + 0.025 * sin(19 * radius);
    let clumps = cos(fade) * waves(crossing.position, radius, 0.0) + sin(fade) * waves(crossing.position, radius, flow_time);
    return colour * rings * exp(clump_contrast * clumps);
}

// The disk's emission on the screen: partly polarized square to the magnetic field circling the normal.
fn disk_coherency(crossing: Ray) -> Coherency {
    let heading = local(field(crossing.position), crossing.momentum);
    let direction = space(heading) / (mv.t & heading);
    let meridian = space(crossing.position).dual() ^ normal;
    let polarization_axis = meridian.commutator(direction) / meridian.norm();
    let polarization = crossing.screen.adjoint()(polarization_axis);
    let polarized_fraction = polarization_degree * polarization.squared();
    return (1 - polarized_fraction) / 2 * Screen + polarization_degree * polarization * (polarization | Screen);
}
"""

camera = """
var<uniform> eye: Antivector;
var<uniform> orientation: Rotor;
var<uniform> analyzer: Screen;
var<uniform> half_view: f32;
var<uniform> step_fraction: f32;
var<uniform> n_steps: u32;
var<uniform> escape_radius: f32;
var<uniform> capture_radius: f32;
var stars: texture_2d<f32>;

// What a ray brings back: the disk's light, the share of it an analyzer passes, and the starlight.
struct Sample {
    light: vec3<f32>,
    analyzed: vec3<f32>,
    sky: vec3<f32>,
}

const pi = 3.141592653589793;

// A future null momentum along a unit spatial direction of the local frame, carried back and lowered.
fn initial_momentum(position: Antivector, direction: Antivector) -> Vector {
    let momentum = -gauge_inverse(field(position))(time + direction).dual();
    return momentum / (momentum & time);
}

// The camera's own right and up across a ray, as a map from the screen's two axes.
fn camera_screen(position: Antivector, momentum: Vector) -> ScreenFrame {
    let heading = local(field(position), momentum);
    let direction = space(heading) / -(mv.t & heading);
    var right = orientation >> mv.y.dual();
    right = right - direction * (direction | right);
    right = right / right.squared().square_root();
    var up = orientation >> -mv.x.dual();
    up = up - direction * (direction | up) - right * (right | up);
    up = up / up.squared().square_root();
    return right * (mv.x & Screen) + up * (mv.y & Screen);
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
    var direction = orientation >> (-mv.z.dual() + mv.y.dual() * across.x - mv.x.dual() * across.y);
    direction = direction / direction.squared().square_root();
    let momentum = -initial_momentum(eye, -direction);
    var ray = Ray(eye, momentum, camera_screen(eye, momentum));

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
            if (crossing_radius >= inner_radius && crossing_radius <= outer_radius) {
                let weight = remaining * disk_opacity(crossing, crossing_radius);
                let emitted = weight * disk_light(crossing, crossing_radius);
                disk += emitted;
                through += emitted * (analyzer | disk_coherency(crossing)(analyzer));
                remaining -= weight;
            }
        }
        if (end_radius > escape_radius) {
            let geometry = field(end.position);
            stars_seen = remaining * starlight(gauge(geometry)(local(geometry, end.momentum)));
            break;
        }
        if (end_radius < capture_radius) {
            break;
        }
        ray = end;
        radius = end_radius;
    }
    return Sample(disk, through, stars_seen);
}
"""


def shader() -> Shader:
    """The camera's shader, compiled."""
    return Shader(geometry + rays + disk + camera + supersampled(supersampling), core.ga, types)


def scene() -> dict:
    """The values the shader reads for the still: hole, camera, disk, tables and sky."""
    spin = mv.xy * scenarios.camera_spin                                        # [] Spin
    orientation = (mv.zx * (-scenarios.camera_inclination / 2)).exp()           # [] Rotor
    eye = (orientation >> mv.z.dual()) * scenarios.camera_distance              # [] Antivector
    # The camera at rest, a unit along the lab's time measured in proper time at the eye.
    at_rest = core.time / core.proper_squared(core.field(eye, spin, scenarios.mass), core.time).square_root()  # [] Antivector
    inner_radius = core.innermost_stable_orbit(mv.xy, spin, scenarios.mass)   # [] Scalar
    radii = inner_radius + (scenarios.disk_outer_radius - inner_radius) * np.linspace(0, 1, n_table_entries)  # [entries] Scalar
    return dict(
        spin=spin, mass=scenarios.mass, eye=eye, orientation=orientation, at_rest=at_rest,
        inner_radius=inner_radius, outer_radius=scenarios.disk_outer_radius, inner_depth=scenarios.inner_depth,
        polarization_degree=scenarios.polarization_degree, clump_contrast=render.clump_contrast,
        elapsed=0.0, flow_time=scenarios.flow_time, fade=0.0, analyzer=mv.x.dual(),
        hottest=hottest, temperatures=core.disk_temperature(radii, mv.xy, spin, scenarios.mass, scenarios.disk_temperature_scale),
        blackbodies=render.blackbody_table(hottest, n_table_entries), modes=render.wave_modes(),
        half_view=scenarios.half_view,
        step_fraction=scenarios.camera_step_fraction, n_steps=scenarios.n_camera_steps,
        escape_radius=scenarios.escape_radius, capture_radius=scenarios.capture_radius,
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
    for analyzer in (mv.xy * (-scenarios.polarizer_angles / 2)).exp() >> mv.x.dual():
        yield tracer(image_shape(), **values | dict(analyzer=analyzer))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure

    tracer = shader()
    values = scene()
    still = tracer(image_shape(), **values)
    save_figure(render.draw_frame(render.develop(still["image"]["light"], still["image"]["sky"], scenarios.exposure_temperature)),
                "kerr_gpu_camera")
    save_animation([render.develop(traced["image"]["light"], traced["image"]["sky"], scenarios.exposure_temperature)
                    for traced in disk_flow(tracer, values)], "kerr_gpu_disk", scenarios.duration_ms, scale=scenarios.web_scale)
    # The exposure is doubled for the half of unpolarized light a polarizer stops.
    save_animation([render.develop(2 * traced["image"]["analyzed"], traced["image"]["sky"], scenarios.exposure_temperature)
                    for traced in polarizer(tracer, values)], "kerr_gpu_polarizer", scenarios.duration_ms, scale=scenarios.web_scale)

    # --- checks
    # Every pixel's light is finite and none is negative.
    for traced in still["image"].values():
        assert np.all(np.isfinite(traced)) and np.all(traced >= 0)


if __name__ == "__main__":
    main()
