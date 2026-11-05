"""Drawing for the dynamo, two ways. Seen from above, the core alone: its field lines and its flow's streamlines, with
chevrons pointing downstream, inside the core's surface as a faint translucent sphere. Cut away: the ball cut open by a
vertical plane through its centre, the cut face streaked along the flow within it, the back half of the planet's surface
behind it, and the field lines of the removed half standing in front of it as tubes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import numpy as np
from matplotlib.colors import to_rgb
from scipy.spatial import cKDTree

from numga import Algebra, concatenate, stack
from numga.extensor import Extensor
from examples.electromagnetism.dynamo import core
from examples.util import shapes
from examples.util.topology import Topology
from examples.util import charts
from examples.util.wgsl import flow, image_figure, meshes, simplices, solids

mv = core.mv
# The cut face's colours from no field to the strongest, the planet's surface, the field lines, and the background.
weak_colour, strong_colour = np.array(to_rgb("#1b1b3a")), np.array(to_rgb("#f59f00"))
surface_colour = np.array(to_rgb("#dee2e6"))
line_colour = np.array(to_rgb("#f1f3f5"))
stream_colour = np.array(to_rgb("#15aabf"))
background = np.ones(3)
# The lines' own picture: the core alone on a dark ground, seen from above, its elevation and azimuth in degrees, its
# pixels across, and the one radius its field lines and streamlines are seeded at; the chevrons on the streamlines,
# their length in pixels and the steps between them.
dark_background = np.array(to_rgb("#1b1b3a"))
above_degrees = np.array([89.0, -30.0])
n_lines_pixels = 256
lines_seed_radii = np.array([0.6])
# The core's surface around the lines, its colour and opacity.
core_colour, core_opacity = np.array(to_rgb("#dee2e6")), 0.14
chevron_length = 5.0
chevron_steps = 15
# The camera's elevation and azimuth in degrees, the reach framed around each point, the margin and the pixels across.
view_degrees = np.array([18.0, -30.0])
surface_reach = 0.04
margin = 0.06
n_pixels = 320
# The pixels across each of the modes' panels.
n_mode_pixels = 160
# The cut's turn about z from the plane of x and z, in degrees, so that it faces the camera.
cut_degrees = -120.0
# The cut face's rings and sectors, and the planet's surface's subdivision levels.
n_rings, n_sectors = 24, 96
n_surface_levels = 3
# The seeds of the field lines and of the streamlines, the points of a subdivided icosahedron at each radius in the core
# in front of the cut; the traced steps either way from each and their length; the share of a field's strongest below
# which a line slows to a stop; the tetrahedra a field is smoothed over and the smoothing's reach; and the tubes' width
# in pixels.
n_seed_levels = 0
seed_radii = np.array([0.45, 0.75])
stream_seed_radii = np.array([0.45, 0.75])
n_trace_steps, trace_step = 60, 0.025
softening = 1e-2
n_neighbours, smoothing_reach = 16, 0.18
line_width = 1.6
# The flow's streaks on the cut face: the percentile of the first frame's speeds within the cut whose streaks run their
# full length in pixels, the spacing of their seeds in pixels, and the share of each streak that lies upstream of its
# seed, so that it fades downstream.
reference_percentile = 95
streak_pixels = 160.0
grain_pixels = 3.0
upstream_share = 0.9
# The growth curve's colour and its axes' colour.
energy_colour = "#e8590c"
flow_colour = "#15aabf"
axis_colour = "0.8"
# An intrinsic unit: it reads an extrinsic element as the intrinsic one of the same grade.
unit = mv.antipseudoscalar([1.0])
# The cut's turn, and the plane of x and z turned with it: the cut, which reads a point's depth toward the camera.
turn = (mv.xy * (-np.radians(cut_degrees) / 2)).exp()                             # [] Rotor+
cut = turn >> mv.y                                                                # [] Plane+
Across = core.Plane.derive.within(Algebra("x+z+"))                                # Vector+
# A direction's part within the cut, its part along the cut's normal taken away.
in_cut = core.Point - cut.dual() * (cut & core.Point)                             # [] Point- <- Point-


# --- plumbing -------------------------------------------------------------------------
def camera_view(points: solids.Point, pixels: int, degrees: np.ndarray) -> solids.View:
    """A square camera view framing the points from an elevation and azimuth in degrees."""
    elevation, azimuth = np.radians(degrees[0]), np.radians(degrees[1] + 90.0)
    return solids.view([points], surface_reach, elevation, azimuth, margin, pixels).squared()


def line_tubes(lines: core.Point, inside: np.ndarray, colour: np.ndarray, frame: solids.View,
               chevron: float) -> meshes.Samples:
    """Lines `[lines, steps] Point-` as tubes where both ends of a step lie inside `[lines * steps]`, a chevron
    `chevron` pixels long ending every so many steps, pointing on along the line."""
    n_lines, n_points = lines.shape
    chains = Topology.chains([n_points] * n_lines)
    kept = np.all(inside[chains.elements[1]], axis=-1)                            # [edges]
    chevrons = (chains.elements[1][:, 0] % n_points % chevron_steps == chevron_steps - 1).astype(float)   # [edges]
    return simplices.tubes(drawn(lines.reshape(-1)), chains.subset(kept, 1), colour, line_width, frame, solids.gloss,
                           chevrons=chevrons[kept], chevron=chevron)


def drawn(points: core.Point) -> solids.Point:
    """Points of space as the renderer's points."""
    return solids.origin + solids.context.include_from_algebra(points.batch())


def nearest(points: core.Point, centres: core.Point) -> np.ndarray:
    """`[..., neighbours]`: the indices of the tetrahedra nearest each point, a search on their read-out places."""
    places = (stack((mv.x, mv.y, mv.z)) & centres[..., None]).to_array()         # [tets, 3]
    queries = (stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array()         # [..., 3]
    return cKDTree(places).query(queries, n_neighbours)[1]


def smoothed(points: core.Point, centres: core.Point, fields: Extensor) -> Extensor:
    """The field at points `[...]`, of any type, the fields of the nearest tetrahedra averaged with weights falling off
    with distance."""
    near = nearest(points, centres)                                               # [..., neighbours]
    offsets = points[..., None] - centres[near]                                   # [..., neighbours] Point-
    weights = (-offsets.antidot(offsets) / smoothing_reach**2).exp()              # [..., neighbours] Scalar+
    return (weights * fields[near]).sum(axis=-1) / weights.sum(axis=-1)


def field_lines(centres: core.Point, fields: core.Point, seeds: core.Point) -> core.Point:
    """The field lines through the seeds, `[lines, 2 * steps + 1] Point-`, traced both ways by Runge–Kutta steps of
    equal length along the smoothed field's direction, slowing to a stop where the field fades, as a flow does at the
    fluid's edge."""
    senses = np.array([-1.0, 1.0])[:, None]                                       # [senses, 1]
    floor = (softening * fields.antinorm().max())**2

    def direction(points: core.Point) -> core.Point:
        field = smoothed(points, centres, fields) * senses                        # [senses, lines] Point-
        return field / (field.antidot(field) + floor).square_root()

    points = stack((seeds, seeds))                                                # [senses, lines] Point-
    traced = [points]
    for _ in range(n_trace_steps):
        k1 = direction(points)
        k2 = direction(points + k1 * (trace_step / 2))
        k3 = direction(points + k2 * (trace_step / 2))
        k4 = direction(points + k3 * trace_step)
        points = points + (k1 + 2 * k2 + 2 * k3 + k4) * (trace_step / 6)
        traced.append(points)
    traced = stack(traced, axis=-1)                                               # [senses, lines, steps + 1] Point-
    # Back from the far end to the seed, then forward.
    return concatenate((traced[0, :, :0:-1], traced[1]), axis=-1)


class Cutaway:
    """The fixed parts of the cutaway: the cut face, the back half of the planet's surface, and the seeds."""

    def __init__(self, ball: core.Ball, radius: float, pixels: int):
        self.centres = ball.centres.batch()                                       # [tets] Point-
        disk, coordinates = shapes.disk(radius, n_rings, n_sectors)
        self.disk = disk
        self.disk_points = turn >> mv(Across, coordinates).dual()                 # [disk vertices] Point-
        self.nearest_tets = nearest(self.disk_points, self.centres)[:, 0]         # [disk vertices]
        sphere, coordinates = shapes.sphere(n_surface_levels)
        surface_points = mv(core.Plane, radius * coordinates).dual()             # [surface vertices] Point-
        behind = np.all(((cut & surface_points) < 0)[sphere.elements[2]], axis=-1)
        self.surface, self.surface_points = sphere.subset(behind, -1), surface_points
        _, coordinates = shapes.sphere(n_seed_levels)
        seeds = mv(core.Plane, (seed_radii[:, None, None] * coordinates).reshape(-1, 3)).dual()   # [seeds] Point-
        self.seeds = seeds[(cut & seeds) > 0.1]                                   # [lines] Point-
        stream_seeds = mv(core.Plane, (stream_seed_radii[:, None, None] * coordinates).reshape(-1, 3)).dual()   # [seeds] Point-
        self.stream_seeds = stream_seeds[(cut & stream_seeds) > 0.1]              # [streams] Point-
        self.frame = camera_view(drawn(self.surface_points), pixels, view_degrees)
        self.radius = radius

    def tubes(self, lines: core.Point, colour: np.ndarray) -> meshes.Samples:
        """Lines `[lines, steps] Point-` as tubes, kept within the removed half of the planet, in front of the cut."""
        places = lines.reshape(-1)                                                # [lines * steps] Point-
        inside = (places.antinorm() < self.radius) & ((cut & places) > 0)         # [lines * steps]
        return line_tubes(lines, inside, colour, self.frame, 0.0)

    def face(self, fields: core.Point) -> meshes.Samples:
        """The cut face coloured by the strength of a field of directions `Point-[tets]`, as a share of its strongest
        across the face, and the back half of the planet's surface behind it."""
        strength = smoothed(self.disk_points, self.centres, fields).antinorm().to_array()   # [disk vertices]
        levels = strength / strength.max()
        return meshes.samples(drawn(self.disk_points), self.disk.elements[2], levels[self.disk.elements[2]][..., None],
                              meshes.flat(drawn(self.disk_points), self.disk.elements[2]), self.frame,
                              face_values=np.zeros(1), colour=strength_shader, gloss=0.0)

    def bare_face(self) -> meshes.Samples:
        """The cut face in one colour."""
        return meshes.samples(drawn(self.disk_points), self.disk.elements[2], weak_colour,
                              meshes.flat(drawn(self.disk_points), self.disk.elements[2]), self.frame,
                              face_values=np.zeros(1), colour=meshes.plain, gloss=0.0)

    def shell(self) -> meshes.Samples:
        """The back half of the planet's surface."""
        return meshes.samples(drawn(self.surface_points), self.surface.elements[2], surface_colour,
                              meshes.smooth(drawn(self.surface_points), self.surface), self.frame,
                              face_values=np.zeros(1), colour=meshes.plain, gloss=solids.gloss)

    def within(self, velocity: core.Point) -> core.Point:
        """The flow within the cut at the face's vertices, `Point-[disk vertices]`, and whether each lies in the fluid,
        its nearest tetrahedron moving."""
        moving = velocity.batch()[self.nearest_tets].antinorm() > 0               # [disk vertices]
        return in_cut(smoothed(self.disk_points, self.centres, velocity.batch())), moving

    def picture(self, magnetic: core.Magnetic, velocity: core.Point, reference_speed: float) -> np.ndarray:
        """The cutaway of one field and the flow carrying it: the cut face streaked along the flow within it, and the
        field lines in front."""
        # Each tetrahedron's field line direction: the magnetic plane swept through time, read as a direction.
        fields = (magnetic.batch() ^ mv.t) * unit                                 # [tets] Point-
        solid = solids.draw([], self.frame, background, inks=[], translucent=[],
                            meshes=[(self.bare_face(), 1.0), (self.shell(), 1.0)]) / 255
        # The flow within the cut, inked on the fluid's part of the face as streaks that fade downstream.
        within, moving = self.within(velocity)                                    # [disk vertices] Point-, [disk vertices]
        fluid = self.disk.subset(np.all(moving[self.disk.elements[2]], axis=-1), -1)
        streaks = flow.texture(drawn(self.disk_points), fluid, solids.context.include_from_algebra(within),
                               self.frame, np.ones(3), reference_speed, flow.surface_grain, grain_pixels,
                               upstream_share, streak_pixels, np.ones(3))["coverage"]   # [rows, columns, 1]
        streaked = solid * (1 - streaks) + stream_colour * streaks
        # The field lines over it, their colour weighted by their coverage.
        lines, _ = solids.layer([], self.frame, [], [], [(self.tubes(field_lines(self.centres, fields, self.seeds),
                                                                      line_colour), 1.0)])   # [rows, columns, 4]
        picture = streaked * (1 - lines[..., 3:]) + lines[..., :3]
        return np.round(np.clip(picture, 0, 1) * 255).astype(np.uint8)

    def mode(self, velocity: core.Point) -> np.ndarray:
        """The cutaway of one flow, the cut face coloured by its speed."""
        streams = self.tubes(field_lines(self.centres, velocity.batch(), self.stream_seeds), stream_colour)
        return solids.draw([], self.frame, background, inks=[], translucent=[],
                           meshes=[(self.face(velocity.batch()), 1.0), (self.shell(), 1.0), (streams, 1.0)])


class Lines:
    """The field lines and the flow's streamlines in the core alone, seen from a direction on a dark ground, the
    streamlines with chevrons pointing downstream."""

    def __init__(self, ball: core.Ball, pixels: int, degrees: np.ndarray):
        self.centres = ball.centres.batch()                                       # [tets] Point-
        _, coordinates = shapes.sphere(n_seed_levels)
        self.seeds = mv(core.Plane, (lines_seed_radii[:, None, None] * coordinates).reshape(-1, 3)).dual()   # [lines] Point-
        self.stream_seeds = self.seeds                                            # [streams] Point-
        _, coordinates = shapes.sphere(n_surface_levels)
        self.frame = camera_view(drawn(mv(core.Plane, coordinates).dual()), pixels, degrees)
        # The core's surface, the unit sphere about the centre.
        self.surface = solids.ball(solids.origin, 1.0)[None, None]

    def picture(self, magnetic: core.Magnetic, velocity: core.Point) -> np.ndarray:
        """The field lines and the streamlines of one state, within the core."""
        # Each tetrahedron's field line direction: the magnetic plane swept through time, read as a direction.
        fields = (magnetic.batch() ^ mv.t) * unit                                 # [tets] Point-
        lines = field_lines(self.centres, fields, self.seeds)                     # [lines, steps] Point-
        streams = field_lines(self.centres, velocity.batch(), self.stream_seeds)  # [streams, steps] Point-
        within = [(places.reshape(-1).antinorm() < 1) for places in (lines, streams)]
        return solids.draw([], self.frame, dark_background, inks=[],
                           translucent=[(self.surface, core_colour, core_opacity)], meshes=[
                               (line_tubes(lines, within[0], line_colour, self.frame, 0.0), 1.0),
                               (line_tubes(streams, within[1], stream_colour, self.frame, chevron_length), 1.0)])


def animate_lines(ball: core.Ball, states: Iterable[tuple[core.Magnetic, core.Point]]) -> Iterator[np.ndarray]:
    """The field lines and streamlines at every frame as it comes, seen from above."""
    lines = Lines(ball, n_lines_pixels, above_degrees)
    for magnetic, velocity in states:
        yield lines.picture(magnetic, velocity)


strength_shader = meshes.Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    return vec4<f32>(mix(weak_colour, strong_colour, pow(clamp(values.x, 0.0, 1.0), 0.8)), 1.0);
}
""", dict(weak_colour=weak_colour, strong_colour=strong_colour))


def animate(ball: core.Ball, radius: float, states: Iterable[tuple[core.Magnetic, core.Point]]) -> Iterator[np.ndarray]:
    """The cutaway at every frame as it comes, of each field and the flow carrying it."""
    cutaway = Cutaway(ball, radius, n_pixels)
    states = iter(states)
    first = next(states)
    within, moving = cutaway.within(first[1])
    reference_speed = float(np.percentile(within.antinorm().to_array()[moving], reference_percentile))
    for magnetic, velocity in chain((first,), states):
        yield cutaway.picture(magnetic, velocity, reference_speed)


def draw_modes(ball: core.Ball, radius: float, velocities: core.Point) -> "plt.Figure":
    """The cutaway of each of a few flow modes `[shown] Point-[tets]` side by side, the cut face coloured by its speed
    and its streamlines standing in front."""
    cutaway = Cutaway(ball, radius, n_mode_pixels)
    return image_figure(np.concatenate([cutaway.mode(velocity) for velocity in velocities], axis=1))


@charts.style()
def draw_energies(times: np.ndarray, kinetic: core.Scalar, magnetic: core.Amount) -> "plt.Figure":
    """The flow's kinetic energy and the field's magnetic energy over time on a logarithmic scale: the field grows as
    in the kinematic dynamo until its force on the current brakes the flow, and the two settle."""
    figure, axis = charts.subplots(300, 200, layout="constrained")
    axis.semilogy(times, kinetic.to_array(), color=flow_colour, label="flow")
    axis.semilogy(times, (magnetic & mv.scalar([1.0])).to_array(), color=energy_colour, label="field")
    axis.set(xlim=(times[0], times[-1]), xticks=[times[0], times[-1]], xticklabels=[f"{times[0]:g}", f"{times[-1]:g}"],
             xlabel="time", ylabel="energy")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(axis_colour)
    return figure

