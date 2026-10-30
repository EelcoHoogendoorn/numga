"""Drawing for the symmetry example, on the GPU: heat-flow ellipsoids, the cubic lattice and its responses, and
rotary powers on the sphere of directions.

The ellipsoids and the arrows within them, the lattice's sites and bonds are solids traced by `solids`, the
stiffness surface a mesh drawn by `meshes`; the sphere of directions has a shader of its own, which evaluates the
rotary power on each pixel's direction, `rotary(direction, direction)`, and colours the sphere by it.
"""

from __future__ import annotations

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.mechanics.symmetry.core import Conductivity, Direction, Field, Gyration, Point, Rotor, ga, mv
from examples.util.wgsl import Shader, fitted, meshes, show, solids, supersampled

driving_rgb, flow_rgb = np.array(to_rgb("#262b32")), np.array(to_rgb("#d44e1a"))
surface_colour = np.array([0.36, 0.60, 0.80])
# The heat-flow ellipsoid's share of what lies behind it hidden, so that the flow within it shows; the arrows' shaft
# widths, head lengths and head widths in pixels, the driving field's and the flow's.
surface_opacity = 0.45
driving_arrow, flow_arrow = (1.6, 7.0, 5.0), (2.0, 8.0, 6.0)
stiffness_colour = np.array([0.88, 0.52, 0.30])
site_colours = {"centre": [0.20, 0.22, 0.26], "axial": [0.88, 0.45, 0.22], "diagonal": [0.30, 0.56, 0.75], "corner": [0.78, 0.78, 0.80]}
background = np.array([1.0, 1.0, 1.0])
# The view's elevation and azimuth, the margin around what it shows as a fraction of its larger side, and the
# directions an ellipsoid is framed by.
elevation, azimuth = np.radians(23.0), np.radians(35.0)
margin = 0.08
n_framing = 400
# The lattice's site and bond radii, in lattice spacings.
site_radius, bond_radius = 0.13, 0.035

sphere_source = """
// A unit sphere of directions, seen along parallel rays by a camera turned by `turn`, coloured by the rotary
// power along each direction against the largest, through a colour table, and lit by the lamp.
var<uniform> rotary: Rotary;
var<uniform> limit: f32;
var<uniform> turn: Rotor;
var<uniform> half_height: f32;
var<uniform> lamp: Field;
var<uniform> background: vec3<f32>;
var<uniform> ambient: f32;
var<uniform> gloss: f32;
var<uniform> shininess: f32;
var colours: texture_2d<f32>;
var filtered: sampler;

alias Sample = vec3<f32>;

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_height;
    let foot = (frame * vec3<f32>(across, 0.0)).sum();
    let squared = foot.scalar_norm_squared();
    if (squared > 1) {
        return background;
    }
    // The direction under the pixel: the sphere's point in the camera's frame, turned out.
    let direction = turn >> (foot + mv.z.dual() * sqrt(1 - squared));
    let power = rotary(direction, direction) / limit;
    let entries = f32(textureDimensions(colours).x);
    let colour = textureSampleLevel(colours, filtered, vec2<f32>(((power + 1) / 2 * (entries - 1) + 0.5) / entries, 0.5), 0.0).rgb;
    // The sphere's normal is the direction itself; the glint where it bisects the lamp and the viewer.
    let light = max(direction & lamp, 0.0);
    let halfway = (lamp + (turn >> mv.z)).normalized();
    let glint = pow(max(direction & halfway, 0.0), shininess);
    return colour * (ambient + (1 - ambient) * light) + gloss * glint;
}
"""


# --- plumbing -------------------------------------------------------------------------
def spiral(count: int) -> Rotor:
    """Turns that carry x to `count` directions spread evenly over the sphere, on a golden spiral: up toward z to
    heights evenly apart, then about z by the golden angle from one to the next."""
    latitude = np.arcsin(1 - 2 * (np.arange(count) + 0.5) / count)
    # The golden angle on from one direction to the next, as an angle within one turn.
    longitude = np.pi * (3 - np.sqrt(5)) * np.arange(count) % (2 * np.pi)
    return (mv.xy * (-longitude / 2)).exp() * (mv.xz * (-latitude / 2)).exp()     # [directions] Rotor+


def image_of_sphere(conductivity: Conductivity) -> tuple[solids.Quadric, Point]:
    """The solid ellipsoid a conductivity makes of the unit ball of driving fields, the flows whose driving field,
    `conductivity.inverse()`, is at most of unit length, and points on its surface to frame it by: the flows of unit
    fields."""
    # The unit ball's dual quadric, carried by the conductivity: the ball's metric, the dual, between the
    # conductivity and its adjugate.
    image = conductivity(conductivity.adjugate().dual())                      # [] Direction- <- Field+
    quadric = solids.solid_of_dual_quadric(solids.context.include_from_algebra(image))   # [] Plane+ <- Point-
    # The flows of unit fields, the scene's points they reach from its origin.
    return quadric, solids.origin + solids.context.include_from_algebra(conductivity(spiral(n_framing) >> mv.x))


def figure(halves: list[np.ndarray], height: float) -> tuple[plt.Figure, list[plt.Axes]]:
    """A row of panels with the given half widths and heights, each as wide as its aspect asks, filling the
    figure's height, with a thin gap between them."""
    widths = [height * half[0] / half[1] for half in halves]
    fig = plt.figure(figsize=(sum(widths) * 1.02, height), dpi=120)
    grid = fig.add_gridspec(1, len(halves), width_ratios=widths, left=0, right=1, bottom=0, top=1, wspace=0.02)
    return fig, [fig.add_subplot(cell) for cell in grid]


def draw_conduction(conductivities: Conductivity, driving: Field, fluxes: Direction) -> plt.Figure:
    """For the candidate and each group's average, the image of the unit ball of driving fields, a heat-flow
    ellipsoid, with one driving field, dark, and the heat flow it drives, orange."""
    # The unit driving field is drawn along its direction at the largest gain of any panel's conductivity, the scale
    # the flows are drawn at.
    gains = conductivities.svdvals().reshape(-1)
    largest = gains.max()
    # The arrows' tips, the scene's points the field's direction and the flows reach from its origin.
    driving_tip = solids.origin + solids.context.include_from_algebra(driving.dual() * largest)   # [] Point-
    flows = solids.origin + solids.context.include_from_algebra(fluxes)                 # [panels] Point-
    ellipsoids = [image_of_sphere(conductivity) for conductivity in conductivities]
    view = solids.view([framed for _, framed in ellipsoids] + [driving_tip, flows], 0.0, elevation, azimuth, margin, 1)
    fig, axes = figure([view.half] * conductivities.shape[0], 1.1)
    # The panels' rooms alike, one view fitted to the first holds them all.
    view = fitted(view, axes[0])
    for ax, (ellipsoid, _), flow in zip(axes, ellipsoids, flows):
        # Both arrows from the centre, the flow's within the ellipsoid, seen through its surface.
        arrows = [(part[None], colour) for tip, colour, (width, head_length, head_width) in
                  ((driving_tip, driving_rgb, driving_arrow), (flow, flow_rgb, flow_arrow))
                  for part in solids.arrows(solids.origin, tip, width / 2 * view.pixel, head_length * view.pixel,
                                            head_width / 2 * view.pixel)]
        show(ax, solids.draw(arrows, view, background, inks=[], translucent=[(ellipsoid[None, None], surface_colour,
                                                                     surface_opacity)], meshes=[]), view)
    return fig


def lattice(centre: Point, axial: Point, diagonal: Point, corners: Point) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The lattice's sites as balls, the centre dark, its axial neighbours orange, its diagonal ones blue, the
    corners pale, and the bonds from the centre to its neighbours as rods."""
    return [(solids.ball(centre, site_radius)[:, None], site_colours["centre"]),
            (solids.ball(axial, site_radius)[:, None], site_colours["axial"]),
            (solids.ball(diagonal, site_radius)[:, None], site_colours["diagonal"]),
            (solids.ball(corners, site_radius)[:, None], site_colours["corner"]),
            (solids.rods(centre.broadcast_to(axial.shape), axial, bond_radius), site_colours["axial"]),
            (solids.rods(centre.broadcast_to(diagonal.shape), diagonal, bond_radius), site_colours["diagonal"])]


def stiffness_mesh(stiffness: Direction) -> tuple[Point, np.ndarray]:
    """The stiffness along every direction as a radius, sampled on a latitude-longitude grid, as a mesh."""
    vertices = solids.origin + solids.context.include_from_algebra(stiffness)           # [latitudes, longitudes] Point-
    rows, columns = vertices.shape
    index = np.arange(rows * columns).reshape(rows, columns)
    corner = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()
    faces = np.concatenate([np.stack([corner[0], corner[1], corner[2]], -1), np.stack([corner[0], corner[2], corner[3]], -1)])
    return vertices.reshape(-1), faces


def draw_crystal(centre: Point, axial: Point, diagonal: Point, corners: Point, conductivity: Conductivity,
                 stiffness: Direction) -> plt.Figure:
    """The lattice, its sites by their shells about the centre, with its bonds; the lattice's conduction, the image of
    the unit ball of fields, a sphere; and its stiffness along every direction as a radius, a cube with rounded
    faces."""
    parts = lattice(centre, axial, diagonal, corners)
    lattice_view = solids.view([centre, axial, diagonal, corners], site_radius, elevation, azimuth, margin, 1)
    ellipsoid, framed = image_of_sphere(conductivity)
    conduction_view = solids.view([framed], 0.0, elevation, azimuth, margin, 1)
    points, faces = stiffness_mesh(stiffness)
    stiffness_view = solids.view([points], 0.0, elevation, azimuth, margin, 1)
    fig, axes = figure([lattice_view.half, conduction_view.half, stiffness_view.half], 2.5)
    lattice_view, conduction_view, stiffness_view = (fitted(view, ax) for view, ax in
                                                     zip((lattice_view, conduction_view, stiffness_view), axes))
    show(axes[0], solids.draw(parts, lattice_view, background, inks=[], translucent=[], meshes=[]), lattice_view)
    show(axes[1], solids.draw([(ellipsoid[None, None], surface_colour)], conduction_view, background, inks=[],
                              translucent=[], meshes=[]), conduction_view)
    show(axes[2], meshes.draw(points, faces, stiffness_colour, meshes.smooth(points, meshes.Mesh(len(points), faces)), stiffness_view, background,
                              face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss),
         stiffness_view)
    return fig


@cache
def sphere_shader(rotary: object) -> Shader:
    """The sphere's shader, for rotary powers of a type: a scalar of two directions."""
    return Shader(supersampled() + sphere_source, ga, dict(Rotary=rotary, Rotor=ga.gatype.rotor(), Field=Field, Direction=Direction), dict(frame=stack((mv.x, mv.y, mv.z)).dual()))


def draw_optical_activity(rotary: Gyration) -> plt.Figure:
    """For the candidate and each group's average, the sphere of light directions coloured by the rotary
    power along each, red for one sense and blue for the other, shaded by the lamp."""
    turn = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()
    toward_lamp = turn >> (mv.x * 0.4 + mv.y * 0.7 + mv.z * 0.6)
    directions = spiral(n_framing) >> mv.x.dual()                             # [directions] Direction-
    limit = np.abs(rotary[:, None](directions, directions).to_array()).max()
    # The sphere about the origin, seen as the other views see it.
    view = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, 1 + margin), 1)
    colours = plt.get_cmap("RdBu_r")(np.linspace(0, 1, 256))[None, :, :3]
    fig, axes = figure([view.half] * rotary.shape[0], 2.6)
    view = fitted(view, axes[0])
    for ax, response in zip(axes, rotary):
        image = sphere_shader(rotary.gatype)(view.shape, rotary=response, limit=limit, turn=turn, half_height=view.half[1],
                                lamp=toward_lamp.normalized(), background=background,
                                ambient=solids.ambient, gloss=solids.gloss, shininess=solids.shininess, colours=colours)["image"]
        show(ax, np.clip(image, 0, 1), view)
    return fig
