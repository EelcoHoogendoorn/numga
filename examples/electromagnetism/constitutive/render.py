"""Drawing the waves media carry: the fields along their way, the residual against trial speeds, the
speeds by heading, and the drag of moving glass."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.image import AxesImage

from numga import stack

from examples.util import charts
from examples.util.animation import capture
from examples.electromagnetism.constitutive import core
from examples.util.wgsl import fitted, show, solids, strokes
from examples.util.wgsl.scene import Scene

Modes = tuple[np.ndarray, core.Bivector]
mv, sm = core.mv, solids.mv
# The observer sees the point where its time line meets the spatial slice.
space_slice = mv.t ^ (core.Antivector & core.time)
# Electric planes meet the spatial slice in their direction; magnetic planes lie within it.
electric_direction = core.Bivector.dual_inverse() ^ mv.t
magnetic_field = (core.Bivector ^ mv.t) * mv.t.inverse()
# A third turn cycles z to x, x to y and y to z in the drawing.
to_scene = (-(sm.xy + sm.yz + sm.zx) * (np.pi / (3 * np.sqrt(3)))).exp()
# The direction of the waves' travel, drawn along the first axis, z.
travel = mv.z.dual()                                                            # [] Antivector-
electric_colour, magnetic_colour = np.array(to_rgb("crimson")), np.array(to_rgb("dodgerblue"))
axis_colour = np.full(3, 0.5)
# How much of what lies behind it each surface of the axis hides, its near side and its far side.
axis_opacity = 0.37
# The plots' faint cross through the origin, and the names of its lines and axes.
cross_colour, label_colour = "0.85", "0.4"
# The reach of the cross on polar axes, as a share of the fastest wave's.
cross_reach = 1.1
# The samples along the axis, the stations with arrows and discs, and the times of a period that the camera frames.
n_samples, n_stations, n_framing = 200, 21, 16
# The length of the axis drawn, and the waves' angular frequency.
z_max, omega = 4.0 * np.pi, 1.0
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin, a share of the larger side,
# and the figure's size in inches at its dots per inch.
elevation, azimuth = np.radians(20.0), np.radians(-35.0)
margin, figure_size, figure_dpi = 0.06, (18.3, 3.8), 43
# The widths of the field curves, the axis and the arrows' shafts, in pixels; the arrows' heads, as a share of the
# arrow's length, and their radius, as a share of the head's length.
curve_width, axis_width, arrow_width = 1.44, 1.08, 1.08
head_share, head_spread = 0.18, 0.35
# The magnetic discs, at every other station: their radius at a unit of field, in the drawing's units, their area
# growing with the field's strength as a patch's does; their thickness, as a share of their radius; and how much of
# what lies behind them they hide.
disc_scale, disc_flatness, disc_opacity = 0.41, 0.1, 0.55
# The arcs of the discs' sense of turning, just outside the rim, as a share of the disc's radius: where they start
# around it from the direction of travel, how far they run, and their heads, as angles around it; and the samples
# along each.
arc_reach = 1.2
arc_start, arc_span, arc_head = np.radians(30.0), np.radians(240.0), np.radians(35.0)
n_arc = 16


# --- plumbing -------------------------------------------------------------------------
def _zero_axes(ax: plt.Axes, across: str, up: str) -> None:
    """Axes of a quantity without tick marks or numbers: the quantity from its zero, the axis across at that zero,
    both named by what runs along them."""
    ax.set_ylim(bottom=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(across, color=label_colour)
    ax.set_ylabel(up, color=label_colour)


def _cross(ax: plt.Axes, across: str, up: str) -> None:
    """A plane without scales: a faint cross through the origin, its lines named at their ends."""
    ax.axhline(0, color=cross_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.axvline(0, color=cross_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.text(1, 0, across, transform=ax.get_yaxis_transform(), ha="right", va="top", color=label_colour)
    ax.text(0, 1, f" {up}", transform=ax.get_xaxis_transform(), ha="left", va="top", color=label_colour)
    ax.set_axis_off()


def waves(modes: Modes, taus: np.ndarray, distance: np.ndarray) -> core.Bivector:
    """The field of the modes' travelling waves at each distance along z and each time, `[..., samples] Bivector+`,
    for times `[...]`, each mode's electric field of unit strength."""
    speeds, fields = modes
    # Each mode scaled to unit strength of the electric field an observer at rest reads, the field met with its time.
    units = fields / (fields & core.time).norm()                                # [modes] Bivector+
    phases = np.cos(omega * (distance[:, None] / speeds - taus[..., None, None]))  # [..., samples, modes]
    return (units * phases).sum(axis=-1)


def wave_scene(ax: plt.Axes, modes: Modes) -> Scene:
    """The axis, seen by a camera that frames every time of a period alike, as many pixels across as the axes' room
    holds, for the fields at any time to be drawn along it."""
    distance = np.linspace(0, z_max, n_samples)
    axis = solids.origin + (to_scene >> solids.context.include_from_algebra(space_slice(travel * distance)))
    taus = np.linspace(0, 2 * np.pi / omega, n_framing, endpoint=False)
    framed = waves(modes, taus, distance)                                       # [framing, samples] Bivector+
    electric = to_scene >> solids.context.include_from_algebra(electric_direction(framed))      # [framing, samples] Point-
    view = fitted(solids.view([axis + electric], 0.0, elevation, azimuth, margin, 1), ax)
    radius = axis_width / 2 * view.pixel
    line = solids.rods(axis[0], axis[-1], radius)[None]                         # [1, 3] Plane+ <- Point-
    return Scene(view, np.ones(3), [], [], inks=[], translucent=[(line, axis_colour, axis_opacity)])


def magnetic_discs(stations: solids.Point, fields: core.Bivector,
                   radius: float) -> tuple[solids.Quadric, solids.Point, solids.Quadric, solids.Quadric]:
    """The magnetic fields `[stations] Bivector+` at stations `[stations] Point-`, each as a flat disc in its own
    plane, its area growing with the field's strength, `[stations, 1]`; and the sense it turns in, as an arc around
    the rim `[stations, arc]` ending in an arrow along the rim, its shaft of a radius and its head nearly all of it
    `[stations]` each."""
    # The plane of the drawing the field lies in, through the origin, its size the field's strength.
    magnetic = to_scene >> solids.context.include_from_algebra(magnetic_field(fields))        # [stations] Bivector+
    planes = (solids.origin & magnetic.dual()) * solids.flip                  # [stations] Plane+
    radii = disc_scale * planes.norm().square_root()                            # [stations] Scalar+
    centres = solids.translator(stations - solids.origin)                       # [stations] Motor+
    # A ball flattened across each plane, scaled by its radius squared, so that a field of no strength is a point.
    unit_planes = planes.normalized()                                           # [stations] Plane+
    flattening = (1 / disc_flatness**2 - 1) * unit_planes * (unit_planes & solids.Point)   # [stations] Plane+ <- Point-
    rounded = (solids.axes * (solids.axes & solids.Point)).sum(axis=-1)          # [] Plane+ <- Point-
    lenses = rounded + flattening - radii**2 * sm.w * (sm.w & solids.Point)     # [stations] Plane+ <- Point-
    # The line through the origin square to each plane, about which its rim turns in the field's sense.
    pivots = (solids.origin & solids.normal_direction(unit_planes)) * solids.flip   # [stations] Line+
    angles = arc_start + np.append(np.linspace(0, arc_span, n_arc), arc_span + arc_head)   # [arc + 1]
    turns = (pivots[:, None] * (-angles / 2)).exp()                             # [stations, arc + 1] Motor+
    # Each arc from where the plane holds the direction of travel, as every magnetic plane of the wave does.
    heading = to_scene >> solids.context.include_from_algebra(space_slice(travel))             # [] Point-
    rim_starts = solids.origin + heading * (arc_reach * radii)                  # [stations] Point-
    rims = (centres[:, None] * turns) >> rim_starts[:, None]                    # [stations, arc + 1] Point-
    # The head spans the chord from the arc's end to the last point, so it shrinks with the disc.
    chords = (rims[:, -2] & rims[:, -1]).norm()                                # [stations] Scalar+
    shafts, heads = solids.arrows(rims[:, -2], rims[:, -1], radius, 0.95 * chords, head_spread * chords)
    return solids.placed(lenses, centres)[:, None], rims[:, :-1], shafts, heads


def wave_image(scene: Scene, modes: Modes, tau: float) -> np.ndarray:
    """The travelling electric and magnetic fields at one time, along the scene's axis: the curve the electric
    arrows' tips trace and the arrows at stations along it, and at each station the magnetic field as a disc in its
    own plane, with an arc around its rim for the sense it turns in."""
    view = scene.view
    distance = np.linspace(0, z_max, n_samples)
    axis = solids.origin + (to_scene >> solids.context.include_from_algebra(space_slice(travel * distance)))
    field = waves(modes, np.array(tau), distance)                               # [samples] Bivector+
    electric = to_scene >> solids.context.include_from_algebra(electric_direction(field))       # [samples] Point-

    # Arrows from the axis at evenly spaced stations, their heads a share of their length, so weak fields stay slender.
    station_idx = np.linspace(0, n_samples - 1, n_stations, dtype=int)
    stations = axis[station_idx]                                                # [stations] Point-
    head_lengths = head_share * solids.context.include_from_algebra((field[station_idx] & core.time).norm())
    shafts, heads = solids.arrows(stations, stations + electric[station_idx], arrow_width / 2 * view.pixel,
                                  head_lengths, head_spread * head_lengths)
    discs, arcs, turning_shafts, turning_heads = magnetic_discs(stations[::2], field[station_idx[::2]],
                                                                arrow_width / 2 * view.pixel)

    curves = strokes.polylines([(axis + electric, electric_colour)]
                               + [(arc, magnetic_colour) for arc in arcs], curve_width, view)
    parts = [(shafts, electric_colour), (heads, electric_colour), (turning_shafts, magnetic_colour),
             (turning_heads, magnetic_colour)]
    return scene.draw(parts, inks=[], translucent=[(discs, magnetic_colour, disc_opacity)], meshes=[(curves, 1.0)])


def draw_wave_propagation(ax: plt.Axes, modes: Modes) -> AxesImage:
    """Draw the travelling electric and magnetic fields in a medium along the propagation axis z, at the start of a
    period; the image shown."""
    scene = wave_scene(ax, modes)
    return show(ax, wave_image(scene, modes, 0.0), scene.view)


def draw_wave_comparison_figure(glass_modes: Modes, crystal_modes: Modes) -> plt.Figure:
    """The waves in isotropic glass, on the left, beside those in the birefringent crystal."""
    fig = plt.figure(figsize=figure_size, dpi=figure_dpi)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0)
    draw_wave_propagation(fig.add_subplot(1, 2, 1), glass_modes)
    draw_wave_propagation(fig.add_subplot(1, 2, 2), crystal_modes)
    return fig


def animate_wave_comparison(glass_modes: Modes, crystal_modes: Modes, n_frames: int) -> Iterator[np.ndarray]:
    """Frames of the travelling electric and magnetic fields through one period, in isotropic glass beside the
    birefringent crystal."""
    fig = plt.figure(figsize=figure_size, dpi=figure_dpi)
    glass_axes = fig.add_subplot(1, 2, 1)
    crystal_axes = fig.add_subplot(1, 2, 2)
    # One layout for every frame, so the axes stay put while the waves move.
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0)
    # Each medium's axis and camera set up once, and its image, for every frame to set.
    panels = []
    for ax, modes in ((glass_axes, glass_modes), (crystal_axes, crystal_modes)):
        scene = wave_scene(ax, modes)
        panels.append((show(ax, np.full((*scene.view.shape, 3), 255, dtype=np.uint8), scene.view), scene, modes))

    for tau in np.linspace(0, 2.0 * np.pi / omega, n_frames, endpoint=False):
        for shown, scene, modes in panels:
            shown.set_data(wave_image(scene, modes, float(tau)))
        yield capture(fig)

    plt.close(fig)


def draw_dispersion(ax: plt.Axes, speeds: np.ndarray, residual: core.Scalar, expected: np.ndarray) -> None:
    """Each medium's residual against the trial speeds `[speeds, media]`, falling to zero at its expected
    speeds `[media, 2]`, marked in its colour."""
    lines = ax.plot(speeds, residual.to_array())
    for line, marks in zip(lines, expected):
        for speed in marks:
            ax.axvline(speed, color=line.get_color(), linestyle=":", alpha=0.8)

    _zero_axes(ax, "speed along $z$", "residual")


def draw_polarizations(
    ax: plt.Axes,
    modes: Modes,
) -> None:
    """Draw the electric field of the slow and the fast mode of the birefringent crystal as arrows in the xy plane,
    each of unit length."""
    _, fields = modes
    # Each mode's field at unit strength of its electric field, read through the planes x and y sweep in time.
    units = fields / (fields & core.time).norm()                                # [modes] Bivector+
    readers = stack((mv.x, mv.y)).dual() & core.time                            # [2] Antibivector-
    arrows = (units[:, None] & readers).to_array()                              # [modes, 2]
    colors = ("crimson", "dodgerblue")
    ax.quiver(np.zeros(len(arrows)), np.zeros(len(arrows)), *arrows.T,
              angles="xy", scale_units="xy", scale=1, color=colors, width=0.015)
    for (horizontal, vertical), color in zip(arrows, colors):
        ax.plot([-horizontal, horizontal], [-vertical, vertical], color=color, linestyle=":", alpha=0.6)

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_aspect("equal")
    _cross(ax, "$x$", "$y$")


def draw_fresnel_surface_polar(
    ax: plt.Axes,
    angles: np.ndarray,
    residual: core.Scalar,
    speeds: np.ndarray,
    threshold: float,
) -> None:
    """The speeds of the waves against their heading in the xz plane, one colour per medium: the dips of
    each medium's residual along speed `[speeds, headings, media]`, the n-th dip at each heading on the
    n-th sheet."""
    for dips, color in zip(np.moveaxis(core.dips(residual, threshold), -1, 0), plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        # Each dip's place among the dips at its heading, counting from the slowest.
        sheets = np.cumsum(dips, axis=0) * dips                                # [speeds - 2, headings]
        for sheet in range(1, sheets.max() + 1):
            heading_idx, speed_idx = np.nonzero((sheets == sheet).T)
            ax.plot(angles[heading_idx], speeds[1:-1][speed_idx], color=color)

    # Zero radians along +z (North), angles clockwise: +x along East.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    # A faint cross through the origin, past the fastest wave, its lines named at their ends by the directions
    # they run along.
    radius = cross_reach * ax.get_rmax()
    for heading in np.arange(4) * np.pi / 2:
        ax.plot([heading, heading], [0, radius], color=cross_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.set_rmax(radius)
    ax.text(0, radius, "$z$", ha="left", va="top", color=label_colour)
    ax.text(np.pi / 2, radius, "$x$", ha="right", va="bottom", color=label_colour)


def draw_fresnel_drag_curves(
    ax: plt.Axes,
    fractions: np.ndarray,
    found: np.ndarray,
    curve: np.ndarray,
    relativistic: np.ndarray,
    first_order: np.ndarray,
) -> None:
    """The speeds of light found in the moving glass, with it in red and against it in blue `[2, drags]`,
    over the relativistic sum, solid, and Fresnel's first-order drag, dashed `[2, curve]`."""
    for speeds, exact, approximate, color, marker in zip(found, relativistic, first_order, "rb", "os"):
        ax.plot(fractions, speeds, color=color, marker=marker, linestyle="none")
        ax.plot(curve, exact, color=color)
        ax.plot(curve, approximate, color=color, linestyle="--", alpha=0.7)

    _zero_axes(ax, "speed of the glass", "speed of light")


@charts.style()
def draw_dispersion_figure(speeds: np.ndarray, residual: core.Scalar, expected: np.ndarray) -> plt.Figure:
    """Each medium's residual against the trial speeds."""
    # Half the column: a few smooth curves, named axes and no numbers.
    fig, ax = charts.subplots(400, 280)
    draw_dispersion(ax, speeds, residual, expected)
    fig.tight_layout()
    return fig


@charts.style()
def draw_polarizations_figure(modes: Modes) -> plt.Figure:
    """The electric fields of the crystal's two modes, as arrows in the xy plane."""
    # Two arrows on a cross: a third of the column.
    fig, ax = charts.subplots(260, 260)
    draw_polarizations(ax, modes)
    fig.tight_layout()
    return fig


@charts.style()
def draw_fresnel_surface_figure(angles: np.ndarray, residual: core.Scalar, speeds: np.ndarray,
                                threshold: float) -> plt.Figure:
    """The speeds of the waves against their heading, on polar axes."""
    # A few closed sheets on a cross: a third of the column.
    fig, ax = charts.subplots(300, 300, subplot_kw={"projection": "polar"})
    draw_fresnel_surface_polar(ax, angles, residual, speeds, threshold)
    fig.tight_layout()
    return fig


@charts.style()
def draw_fresnel_drag_figure(
    fractions: np.ndarray,
    found: np.ndarray,
    curve: np.ndarray,
    relativistic: np.ndarray,
    first_order: np.ndarray,
) -> plt.Figure:
    """The speeds of light in the moving glass, against the speed of the glass."""
    # Half the column: two curve pairs with their samples, named axes and no numbers.
    fig, ax = charts.subplots(400, 280)
    draw_fresnel_drag_curves(ax, fractions, found, curve, relativistic, first_order)
    fig.tight_layout()
    return fig
