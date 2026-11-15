"""Scenes of Dupin cyclides on the 3-sphere: symmetric tori, vortex flows, lopsided cyclides, and spindles.

Run from the repository root:
    python -m examples.quadrics.cyclides.scenarios
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate, stack
from examples.quadrics.cyclides.core import (
    Direction, Motor, Point, Quadric, Scalar, Upright, antipode, camera_view_motor, chart_infinity,
    chart_origin, cone_quadric, cylinder_quadrics, flat_camera_motor, linked_vortex_states, mv,
    origin, placement_motors, ray_bend, ray_linear, sensor_directions, surface_trace, vortex_states,
)

panel_shape = (180, 240)
linked_shape = (315, 420)
pixels = sensor_directions(panel_shape, np.radians(90))                        # [...] Direction-
n_frames = 54
frame_duration_ms = 40

cylinder_tube = 0.3
torus_tubes = np.array([0.15, 0.5, 0.9])
torus_strengths = np.array([1.55, 2.0, 2.4])
dupin_tubes = np.array([0.25, 0.3, 0.2])
dupin_strengths = np.array([1.5, 1.2, 1.8])
dupin_leans = np.array([0.7, 0.8, 0.5])
dupin_ahead = np.array([1.2, 1.3, 1.1])
dupin_tilts = np.array([-0.7, 0.6, 1.0])
cone_opening = 0.35
spindle_leans = np.array([0.0, 0.7])
spindle_shots = [
    (0, [-0.186, -0.692, 2.658, 0.181, -0.282, -0.225], 1.10, -0.021, 0.061),
    (0, [1.124, 0.598, 0.155, 0.889, -0.164, -0.741], 1.05, -0.051, -0.062),
    (0, [0.464, 0.073, 0.536, -2.263, 0.817, -0.768], 1.05, -0.001, 0.060),
    (1, [-0.257, 0.18, 0.46, -0.999, -1.384, -0.004], 0.85, 0.137, -0.377),
    (1, [-0.511, -0.64, -0.64, 1.096, -1.168, -0.477], 0.85, -0.297, -0.358),
]
still_scene_names = [
    "cyclide", "peanut", "elliptic_ring", "split_ring",
    "pinched", "six_families", "hyperboloid", "two_lobed",
]
vortex_scene_names = ["linked_tori", "pinched_vortex"]


# --- math -----------------------------------------------------------------------------
def placed_tori() -> Quadric:
    """Three spherical cylinders dilated toward `z` and placed in front of the eye form symmetric
    tori (`[3] Sphere+ <- Point-`)."""
    dilations = (mv.ze * (torus_strengths / 2)).exp()                          # [3] Motor+
    views = placement_motors(mv.z.dual(), 1.1, 0.8)                            # [] Motor+
    poses = views * dilations                                                  # [3] Motor+
    tubes = cylinder_quadrics(torus_tubes)                                     # [3] Sphere+ <- Point-
    return poses >> tubes(poses << Point)


def torus_traces() -> tuple[Scalar, Scalar]:
    """Symmetric tori traced from the eye return headlight cosines and hit angles
    (`[3, ...] Scalar+`, `[3, ...] Scalar+`)."""
    return surface_trace(placed_tori(), pixels)


def tilted_vortex_states(n_steps: int) -> Iterator[tuple[Scalar, Scalar]]:
    """Yields `(facings, angles)` (`[1, ...] Scalar+`) as the middle torus rolls once around its
    core circle tilted by `0.4` rad in `yw`."""
    dilation = (mv.ze * (2.0 / 2)).exp()                                       # [] Motor+
    pose = placement_motors(mv.z.dual(), 1.1, 0.8) * dilation                  # [] Motor+
    base_cylinder = cylinder_quadrics(0.5)                                     # [] Sphere+ <- Point-
    torus_surface = pose >> base_cylinder(pose << Point)                       # [] Sphere+ <- Point-
    tilt = (mv.yw * 0.2).exp()                                                 # [] Motor+
    circle_motor = pose * tilt                                                 # [] Motor+
    circle = circle_motor >> mv.zw                                             # [] Bivector+
    return vortex_states(torus_surface, circle, pixels, n_steps)


def placed_dupin_cyclides() -> Quadric:
    """Leaning the dilation aim off `z` toward `x` compresses one side of each cylinder into a
    lopsided Dupin cyclide (`[3] Sphere+ <- Point-`)."""
    aim_turns = (mv.zx * (-dupin_leans / 2)).exp()                             # [3] Motor+
    aims = aim_turns >> mv.z.dual()                                            # [3] Point-
    dilations = ((mv.e.dual() & aims).dual() * (dupin_strengths / 2)).exp()    # [3] Motor+
    tubes = cylinder_quadrics(dupin_tubes)                                     # [3] Sphere+ <- Point-
    lopsided = dilations >> tubes(dilations << Point)                          # [3] Sphere+ <- Point-
    views = placement_motors(aims, dupin_ahead, dupin_tilts)                   # [3] Motor+
    return views >> lopsided(views << Point)


def dupin_traces() -> tuple[Scalar, Scalar]:
    """Lopsided Dupin cyclides traced from the eye return headlight cosines and hit angles
    (`[3, ...] Scalar+`, `[3, ...] Scalar+`)."""
    return surface_trace(placed_dupin_cyclides(), pixels)


def spindle_traces() -> tuple[Scalar, Scalar]:
    """Dilating a two-vertex spherical cone straight and leaning yields five spindle-cyclide views
    with unsigned headlight cosines (`[5, ...] Scalar+`, `[5, ...] Scalar+`)."""
    aim_turns = (mv.xy * (-spindle_leans / 2)).exp()                           # [2] Motor+
    aims = aim_turns >> mv.x.dual()                                            # [2] Point-
    dilations = ((mv.e.dual() & aims).dual() * (3.0 / 2)).exp()                # [2] Motor+
    cone = cone_quadric(cone_opening)                                          # [] Sphere+ <- Point-
    spindles = dilations >> cone(dilations << Point)                           # [2] Sphere+ <- Point-
    midpoints = -aims                                                          # [2] Point-

    views = stack([camera_view_motor(midpoints[i], angles, ahead, yaw, pitch)
                   for i, angles, ahead, yaw, pitch in spindle_shots])         # [5] Motor+
    surfaces = stack([spindles[i] for i, *_ in spindle_shots])                 # [5] Sphere+ <- Point-
    placed_spindles = views >> surfaces(views << Point)                        # [5] Sphere+ <- Point-
    facings, angles = surface_trace(placed_spindles, pixels)
    return facings.abs(), angles


# --- flat-chart scenes ----------------------------------------------------------------
def chart_torus(radius: float, tube: float) -> Quadric:
    """A flat-chart torus of major `radius` and minor `tube` in the `xy` plane forms a polarity map
    (`[] Sphere+ <- Point-`)."""
    sphere = chart_origin - chart_infinity * ((radius * radius + tube * tube) / 2)
    return sphere * (sphere & Point) + radius * radius * (
        mv.z * (mv.z & Point) - tube * tube * chart_infinity * (chart_infinity & Point)
    )


def cyclide_scene():
    """Off-centre sphere inversion of a flat-chart torus produces an unequal-width Dupin cyclide."""
    surface = chart_torus(1.4, 0.5)
    shift = ((antipode & (mv.x.dual() * 3.2)).dual() * (-1 / 2)).exp()
    shifted_sphere = shift >> (chart_origin - chart_infinity * 4.5)
    inversion = shifted_sphere.normalized()
    turn_xy = (mv.xy * 0.12).exp()
    turn_yz = (mv.yz * -0.22).exp()
    pose = turn_xy * turn_yz * inversion
    return (), (pose >> surface(pose << Point),)


def peanut_scene():
    """Perturbing a point-sphere quadric along `y` and `z` produces a peanut-shaped surface."""
    inner_radius, outer_radius = 1.25, 1.31
    sphere = chart_origin - chart_infinity * (inner_radius ** 2 / 2)
    surface = sphere * (sphere & Point) + inner_radius ** 2 * (
        mv.y * (mv.y & Point) + mv.z * (mv.z & Point)
    ) - (outer_radius ** 4 / 4) * chart_infinity * (chart_infinity & Point)
    pose = (mv.xy * 0.10).exp()
    return (), (pose >> surface(pose << Point),)


def elliptic_ring_scene():
    """Adding a weak `y`-plane dyad squeezes a flat-chart torus into an elliptic ring."""
    surface = chart_torus(1.4, 0.5) + (0.10 * 1.4 ** 2) * mv.y * (mv.y & Point)
    pose = (mv.yz * -0.18).exp()
    return (), (pose >> surface(pose << Point),)


def split_ring_scene():
    """Adding a stronger `y`-plane dyad pinches a flat-chart torus into two lobes."""
    surface = chart_torus(1.4, 0.5) + (0.32 * 1.4 ** 2) * mv.y * (mv.y & Point)
    pose = (mv.yz * -0.18).exp()
    return (), (pose >> surface(pose << Point),)


def pinched_quadric() -> Quadric:
    """Inverting a one-sheet hyperboloid in the unit sphere closes its ends at one singular point
    (`[] Sphere+ <- Point-`)."""
    surface = (mv.z * (mv.z & Point) - mv.x * (mv.x & Point)
               - mv.y * (mv.y & Point) + chart_infinity * (chart_infinity & Point))
    inversion = (chart_origin - chart_infinity).normalized()
    return inversion >> surface(inversion << Point)


def pinched_scene():
    """Tilting `pinched_quadric` exposes its singular pinch point."""
    surface = pinched_quadric()
    pose = (mv.yz * -0.18).exp()
    return (), (pose >> surface(pose << Point),)


def sphere_family(weights: np.ndarray) -> Quadric:
    """Weighted dyads of three coordinate planes and the unit sphere form a diagonal sphere-model
    quadric (`[] Sphere+ <- Point-`)."""
    spheres = stack([mv.x, mv.y, mv.z, chart_origin - chart_infinity * 0.5])  # [4] Sphere+
    return (spheres * (spheres & Point) * weights).sum(axis=0)


def six_families_scene():
    """Weights `[-2, -1, 1, 2]` on `sphere_family` produce a four-handle cyclide."""
    surface = sphere_family(np.array([-2, -1, 1, 2]))
    pose = (mv.xy * 0.25 + mv.yz * -0.15).exp()
    return (), (pose >> surface(pose << Point),)


def inverted_hyperboloid_pair():
    """Inverting an elliptic hyperboloid in an exterior sphere closes its ends into a conical tip."""
    surface = (mv.z * (mv.z & Point) - mv.x * (mv.x & Point)
               - mv.y * (mv.y & Point) + chart_infinity * (chart_infinity & Point))
    surface = surface - (1 / 0.65 ** 2 - 1) * mv.y * (mv.y & Point)
    shift = ((antipode & (mv.x.dual() * 1.7)).dual() * (-1 / 2)).exp()
    shifted_sphere = shift >> (chart_origin - chart_infinity)
    inversion = shifted_sphere.normalized()
    return -(inversion >> surface(inversion << Point)), inversion


def hyperboloid_scene():
    """Resting view of the exterior-inverted one-sheet hyperboloid."""
    surface, _ = inverted_hyperboloid_pair()
    return (), (surface,)


def two_lobed_scene():
    """Weights `[-3, -0.25, 1, 1]` on `sphere_family` produce a two-lobed surface."""
    surface = sphere_family(np.array([-3, -0.25, 1, 1]))
    pose = (mv.xy * 0.12 + mv.yz * -0.12).exp()
    return (), (pose >> surface(pose << Point),)


def linked_vortex_scene():
    """An inverted waist circle threads the opening of the exterior-inverted hyperboloid."""
    surface, inversion = inverted_hyperboloid_pair()
    radius = 2.5
    waist_sphere = chart_origin - chart_infinity * (radius * radius / 2)
    inverted_circle = inversion >> (mv.z ^ waist_sphere)
    circle = inverted_circle.normalized()
    inverted_sphere = inversion >> waist_sphere
    unit_sphere = inverted_sphere / -(inverted_sphere | chart_infinity)
    radius_squared = unit_sphere.squared()
    tube = 0.02
    ring_sphere = unit_sphere - chart_infinity * (tube * tube / 2)
    ring = ring_sphere * (ring_sphere & Point) + radius_squared * (
        mv.z * (mv.z & Point) - tube * tube * chart_infinity * (chart_infinity & Point)
    )
    return ((surface, circle),), (ring,)


def linked_tori_scene():
    """A shifted, tilted torus turns around a linked horizontal ring of matching radius."""
    radius, offset = 1.4, 0.45
    surface = chart_torus(radius, 0.16)
    shift = ((antipode & (mv.x.dual() * offset)).dual() * (-1 / 2)).exp()
    tilt = (mv.yz * (np.arcsin(offset / radius) / 2)).exp()
    pose = shift * tilt
    circle = (mv.z ^ (chart_origin - chart_infinity * (radius * radius / 2))).normalized()
    return ((pose >> surface(pose << Point), circle),), (chart_torus(radius, 0.02),)


def pinched_vortex_scene():
    """A shifted, tilted pinched cyclide flows around a linked horizontal ring."""
    surface = pinched_quadric()
    shift = ((antipode & (mv.x.dual() * 1.8)).dual() * (-1 / 2)).exp()
    tilt = (mv.xz * (np.pi / 4)).exp()
    pose = shift * tilt
    radius = 1.4
    circle = (mv.z ^ (chart_origin - chart_infinity * (radius * radius / 2))).normalized()
    return ((pose >> surface(pose << Point), circle),), (chart_torus(radius, 0.02),)


flat_scenes = {
    "cyclide": (cyclide_scene, [0.2, -8.5, 5.8], [-0.8, 0, 0], 43, panel_shape),
    "peanut": (peanut_scene, [0.2, -7, 4.5], [0, 0, 0], 40, panel_shape),
    "elliptic_ring": (elliptic_ring_scene, [0.2, -7, 4.5], [0, 0, 0], 40, panel_shape),
    "split_ring": (split_ring_scene, [0.2, -7, 4.5], [0, 0, 0], 40, panel_shape),
    "pinched": (pinched_scene, [0.2, -7, 4.5], [0, 0, 0], 40, panel_shape),
    "six_families": (six_families_scene, [0.6, -9, 6.5], [0, 0, 0], 40, panel_shape),
    "hyperboloid": (hyperboloid_scene, [10, -2, 1.8], [0.6, 0, 0], 40, panel_shape),
    "two_lobed": (two_lobed_scene, [0.5, -12, 8], [0, 0, 0], 40, panel_shape),
    "linked_vortex": (linked_vortex_scene, [2.8, -9, 4.8], [1, 0, 0], 44, linked_shape),
    "linked_tori": (linked_tori_scene, [0.5, -7, 4.8], [0, 0, 0], 40, panel_shape),
    "pinched_vortex": (pinched_vortex_scene, [0.5, -10, 7], [0.5, 0, 0], 44, panel_shape),
}


def flat_scene_states(name: str, n_steps: int) -> Iterator[tuple[Scalar, Scalar]]:
    """Yields `(facings, angles)` (`[..., ...] Scalar+`) as moving bodies in `flat_scenes[name]`
    complete one loop over `n_steps` frames while resting bodies trace once."""
    scene_builder, position, target, degrees, image_shape = flat_scenes[name]
    moving_bodies, resting_bodies = scene_builder()
    camera = flat_camera_motor(mv(Upright, position).dual(), mv(Upright, target).dual())
    scene_pixels = sensor_directions(image_shape, np.radians(degrees))
    if len(moving_bodies) == 1 and len(resting_bodies) == 1:
        (cyclide_surface, circle), = moving_bodies
        ring_surface, = resting_bodies
        return linked_vortex_states(cyclide_surface, ring_surface, circle, camera, scene_pixels, n_steps)
    def body_states() -> Iterator[tuple[Scalar, Scalar]]:
        rest_traces = [surface_trace((camera >> surface(camera << Point)).reshape(1), scene_pixels)
                       for surface in resting_bodies]
        for phase in np.linspace(0.0, 2 * np.pi, n_steps, endpoint=False):
            motions = [camera * (circle * (phase / 2)).exp() for _, circle in moving_bodies]
            frame_traces = [surface_trace((motion >> surface(motion << Point)).reshape(1), scene_pixels)
                            for motion, (surface, _) in zip(motions, moving_bodies)] + rest_traces
            yield (concatenate([facing for facing, _ in frame_traces]).abs(),
                   concatenate([angle for _, angle in frame_traces]))
    return body_states()


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quadrics.cyclides import render

    tori_facings, tori_angles = torus_traces()
    save_figure(render.facing_figure(tori_facings, tori_angles, panel_shape), "cyclides_tori")
    vortex_records = []
    def passing_vortex() -> Iterator[tuple[Scalar, Scalar]]:
        for state in tilted_vortex_states(n_frames):
            vortex_records.append(state)
            yield state
    save_animation(render.facing_animation(panel_shape, passing_vortex()),
                   "cyclides_vortex", frame_duration_ms)
    for name, (scene_builder, *_, image_shape) in flat_scenes.items():
        moving_bodies, _ = scene_builder()
        states = flat_scene_states(name, n_frames if moving_bodies else 1)
        palette = render.linked_palette if name == "linked_vortex" else render.body_palette
        save_animation(render.scene_animation(image_shape, palette, states),
                       f"cyclides_{name}", frame_duration_ms)
    dupin_facings, dupin_angles = dupin_traces()
    save_figure(render.facing_figure(dupin_facings, dupin_angles, panel_shape), "cyclides_dupin")
    spindle_facings, spindle_angles = spindle_traces()
    save_figure(render.facing_figure(spindle_facings, spindle_angles, panel_shape), "cyclides_spindles")

    # checks
    sample_pixels = pixels[::20000]                                            # [...] Direction-
    cylinder_form = Point & cylinder_quadrics(cylinder_tube)                   # [] Scalar+ <- (Point-, Point-)
    dupin_form = Point & placed_dupin_cyclides()[0]                            # [] Scalar+ <- (Point-, Point-)
    cylinder_outer = (cylinder_form(ray_bend, ray_bend)(sample_pixels, sample_pixels, sample_pixels, sample_pixels)
                      - cylinder_form(origin, origin))                         # [...] Scalar+
    cylinder_odd = ((4 * cylinder_form(ray_linear, ray_bend))(sample_pixels, sample_pixels, sample_pixels)
                    + (4 * cylinder_form(origin, ray_linear))(sample_pixels))  # [...] Scalar+
    dupin_outer = (dupin_form(ray_bend, ray_bend)(sample_pixels, sample_pixels, sample_pixels, sample_pixels)
                   - dupin_form(origin, origin))                               # [...] Scalar+
    assert (cylinder_outer.abs() < 1e-10).all()
    assert (cylinder_odd.abs() < 1e-10).all()
    assert (dupin_outer.abs() > 1e-3).any()
    assert tori_angles.isfinite().any()
    assert spindle_angles.isfinite().any()
    assert vortex_records[0][1].isfinite().any()


if __name__ == "__main__":
    main()
