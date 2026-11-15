"""GPU rendering for material symmetry: heat-flow ellipsoids, cubic lattice shells and stiffness surfaces, and optical-activity spheres.

Ray-traced solids draw the translucent conductivity ellipsoids, internal field and flux arrows, and simple-cubic
lattice shells, while smooth triangle shading draws the directional stiffness surface and a custom sphere shader
colours every light direction by its signed rotary power `gyration(direction, direction)`.
"""

from __future__ import annotations

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.mechanics.symmetry.core import Conductivity, Direction, Field, Gyration, Rotor, ga, mv
from numga.wgsl import Shader, supersampled
from numga.topology import Topology
from numga.render import meshes, simplices, solids
from examples.util.panels import fitted, show

Point = solids.Point                                                            # Point-

driving_rgb, flow_rgb = np.array(to_rgb("#262b32")), np.array(to_rgb("#d44e1a"))
surface_colour = np.array([0.36, 0.60, 0.80])
# Translucent ellipsoid opacity and pixel widths/chevron lengths for the driving-field and heat-flow arrows.
surface_opacity = 0.45
driving_arrow, flow_arrow = (1.6, 6.4), (2.0, 8.0)
stiffness_colour = np.array([0.88, 0.52, 0.30])
site_colours = {"centre": [0.20, 0.22, 0.26], "axial": [0.88, 0.45, 0.22], "diagonal": [0.30, 0.56, 0.75], "corner": [0.78, 0.78, 0.80]}
background = np.array([1.0, 1.0, 1.0])
elevation, azimuth = np.radians(23.0), np.radians(35.0)
margin = 0.08
n_framing = 400
site_radius, bond_radius = 0.13, 0.035

sphere_source = '''
// Unit sphere of light directions under orthographic rays, coloured by normalized rotary power and shaded by a lamp.
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
    let direction = turn >> (foot + mv.z.dual() * sqrt(1 - squared));
    let power = rotary(direction, direction) * sense / limit;
    let entries = f32(textureDimensions(colours).x);
    let colour = textureSampleLevel(colours, filtered, vec2<f32>(((power + 1) / 2 * (entries - 1) + 0.5) / entries, 0.5), 0.0).rgb;
    let light = max(direction & lamp, 0.0);
    let halfway = (lamp + (turn >> mv.z)).normalized();
    let glint = pow(max(direction & halfway, 0.0), shininess);
    return colour * (ambient + (1 - ambient) * light) + gloss * glint;
}
'''


# --- plumbing -------------------------------------------------------------------------
def spiral_turns(n_directions: int) -> Rotor:
    """Golden-spiral rotations (`[n_directions] Rotor+`) spread the first basis axis evenly over the unit sphere."""
    latitude = np.arcsin(1 - 2 * (np.arange(n_directions) + 0.5) / n_directions)
    longitude = np.pi * (3 - np.sqrt(5)) * np.arange(n_directions) % (2 * np.pi)
    return (mv.xy * (-longitude / 2)).exp() * (mv.xz * (-latitude / 2)).exp()   # [n_directions] Rotor+


def image_of_sphere(conductivity: Conductivity) -> tuple[solids.Quadric, Point]:
    """The dual quadric of `conductivity` maps the unit ball of driving fields to a solid heat-flow ellipsoid and framing points."""
    # Carrying the unit ball's Euclidean duality metric through `conductivity` and its adjugate forms the dual ellipsoid.
    ellipsoid_image = conductivity(conductivity.adjugate().dual())              # [] Direction- <- Field+
    quadric = solids.solid_of_dual_quadric(solids.context.include_from_algebra(ellipsoid_image))  # [] Plane+ <- Point-
    sample_fields = spiral_turns(n_framing) >> mv.x                             # [n_framing] Field+
    framing_points = solids.origin + solids.context.include_from_algebra(conductivity(sample_fields))  # [n_framing] Point-
    return quadric, framing_points


def panel_row(halves: list[np.ndarray], height: float) -> tuple[plt.Figure, list[plt.Axes]]:
    """A single-row figure of height `height` sizes each panel width to match its camera aspect ratio."""
    widths = [height * half[0] / half[1] for half in halves]
    fig = plt.figure(figsize=(sum(widths) * 1.02, height), dpi=120)
    grid = fig.add_gridspec(1, len(halves), width_ratios=widths, left=0, right=1, bottom=0, top=1, wspace=0.02)
    return fig, [fig.add_subplot(panel_spec) for panel_spec in grid]


def conduction_figure(conductivities: Conductivity, driving: Field, fluxes: Direction) -> plt.Figure:
    """Each panel shows the translucent heat-flow ellipsoid of `conductivities`, the dark driving arrow, and the orange heat-flow arrow."""
    gains = conductivities.svdvals().reshape(-1)
    largest_gain = gains.max()
    driving_tip = solids.origin + solids.context.include_from_algebra(driving.dual() * largest_gain)  # [] Point-
    flow_tips = solids.origin + solids.context.include_from_algebra(fluxes)     # [...] Point-
    ellipsoids = [image_of_sphere(conductivity) for conductivity in conductivities]
    view = solids.view([framed for _, framed in ellipsoids] + [driving_tip, flow_tips], 0.0, elevation, azimuth, margin, 1)
    fig, axes = panel_row([view.half] * conductivities.shape[0], 1.1)
    view = fitted(view, axes[0])
    for ax, (ellipsoid, _), flow_tip in zip(axes, ellipsoids, flow_tips):
        arrows = [
            (
                simplices.tubes(
                    stack([solids.origin, tip]),
                    Topology.chains([2]),
                    colour,
                    width,
                    view,
                    gloss=solids.gloss,
                    chevrons=np.ones(1),
                    chevron=chevron,
                ),
                1.0,
            )
            for tip, colour, (width, chevron) in ((driving_tip, driving_rgb, driving_arrow), (flow_tip, flow_rgb, flow_arrow))
        ]
        show(
            ax,
            solids.draw([], view, background, inks=[], translucent=[(ellipsoid[None, None], surface_colour, surface_opacity)], meshes=arrows),
            view,
        )
    return fig


def lattice_solids(centre: Point, axial: Point, diagonal: Point, corners: Point) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Coloured balls at the four neighbour shells and rods along the axial and face-diagonal bonds form the lattice scene."""
    return [
        (solids.ball(centre, site_radius)[:, None], site_colours["centre"]),
        (solids.ball(axial, site_radius)[:, None], site_colours["axial"]),
        (solids.ball(diagonal, site_radius)[:, None], site_colours["diagonal"]),
        (solids.ball(corners, site_radius)[:, None], site_colours["corner"]),
        (solids.rods(centre.broadcast_to(axial.shape), axial, bond_radius), site_colours["axial"]),
        (solids.rods(centre.broadcast_to(diagonal.shape), diagonal, bond_radius), site_colours["diagonal"]),
    ]


def stiffness_triangulation(stiffness: Direction) -> tuple[Point, np.ndarray]:
    """Radial `stiffness` offsets sampled on a latitude-longitude grid triangulate into surface vertices and triangle indices."""
    vertices = solids.origin + solids.context.include_from_algebra(stiffness)   # [...] Point-
    n_rows, n_columns = vertices.shape
    index = np.arange(n_rows * n_columns).reshape(n_rows, n_columns)
    quad_corners = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()
    faces = np.concatenate([
        np.stack([quad_corners[0], quad_corners[1], quad_corners[2]], -1),
        np.stack([quad_corners[0], quad_corners[2], quad_corners[3]], -1),
    ])
    return vertices.reshape(-1), faces


def crystal_figure(
    centre: Point,
    axial: Point,
    diagonal: Point,
    corners: Point,
    conductivity: Conductivity,
    stiffness: Direction,
) -> plt.Figure:
    """Three panels display the simple-cubic bond shells, the spherical conduction surface, and the cubic stiffness surface."""
    centre_pts = solids.context.include_from_algebra(centre)                    # [...] Point-
    axial_pts = solids.context.include_from_algebra(axial)                      # [...] Point-
    diagonal_pts = solids.context.include_from_algebra(diagonal)                # [...] Point-
    corner_pts = solids.context.include_from_algebra(corners)                   # [...] Point-
    parts = lattice_solids(centre_pts, axial_pts, diagonal_pts, corner_pts)
    lattice_view = solids.view([centre_pts, axial_pts, diagonal_pts, corner_pts], site_radius, elevation, azimuth, margin, 1)
    ellipsoid, framed = image_of_sphere(conductivity)
    conduction_view = solids.view([framed], 0.0, elevation, azimuth, margin, 1)
    points, faces = stiffness_triangulation(stiffness)
    stiffness_view = solids.view([points], 0.0, elevation, azimuth, margin, 1)
    fig, axes = panel_row([lattice_view.half, conduction_view.half, stiffness_view.half], 2.5)
    lattice_view, conduction_view, stiffness_view = (
        fitted(view, ax) for view, ax in zip((lattice_view, conduction_view, stiffness_view), axes)
    )
    show(axes[0], solids.draw(parts, lattice_view, background, inks=[], translucent=[], meshes=[]), lattice_view)
    show(axes[1], solids.draw([(ellipsoid[None, None], surface_colour)], conduction_view, background, inks=[], translucent=[], meshes=[]), conduction_view)
    show(
        axes[2],
        meshes.draw(
            points,
            faces,
            stiffness_colour,
            meshes.smooth(points, Topology(len(points), faces)),
            stiffness_view,
            background,
            face_values=np.zeros(4),
            colour=meshes.plain,
            gloss=solids.gloss,
        ),
        stiffness_view,
    )
    return fig


@cache
def sphere_shader(rotary_type: object) -> Shader:
    """Compiled WGSL shader evaluating a `rotary_type` gyration map across the visible hemisphere of directions."""
    return Shader(
        supersampled() + sphere_source,
        ga,
        dict(Rotary=rotary_type, Rotor=Rotor, Field=Field, Direction=Direction),
        dict(frame=stack((mv.x, mv.y, mv.z)).dual(), sense=mv.x.dual() & mv.y.dual() & mv.z.dual()),
    )


def optical_activity_figure(gyrations: Gyration) -> plt.Figure:
    """Each panel colours the unit sphere of light directions by signed rotary power, red for one sense and blue for the opposite sense."""
    turn = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()  # [] Rotor+
    toward_lamp = turn >> (mv.x * 0.4 + mv.y * 0.7 + mv.z * 0.6)                # [] Field+
    directions = spiral_turns(n_framing) >> mv.x.dual()                         # [n_framing] Direction-
    limit = np.abs(gyrations[:, None](directions, directions).to_array()).max()
    view = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, 1 + margin), 1)
    colours = plt.get_cmap("RdBu_r")(np.linspace(0, 1, 256))[None, :, :3]
    fig, axes = panel_row([view.half] * gyrations.shape[0], 2.6)
    view = fitted(view, axes[0])
    for ax, response in zip(axes, gyrations):
        image = sphere_shader(gyrations.gatype)(
            view.shape,
            rotary=response,
            limit=limit,
            turn=turn,
            half_height=view.half[1],
            lamp=toward_lamp.normalized(),
            background=background,
            ambient=solids.ambient,
            gloss=solids.gloss,
            shininess=solids.shininess,
            colours=colours,
        )["image"]
        show(ax, np.clip(image, 0, 1), view)
    return fig
