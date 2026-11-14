"""Drawing the waves media carry: the fields along their way, the residual against trial speeds, the
speeds by heading, and the drag of moving glass."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.image import AxesImage

from numga import concatenate, stack
from numga.render import Camera, Scene, simplices, solids
from numga.topology import Topology

from examples.electromagnetism.constitutive import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import fitted, offset_panel, overlay, show

Modes = tuple[np.ndarray, core.Field]
mv, sm = core.mv, solids.mv
# The observer sees the point where its time line meets the spatial slice.
space_slice = core.t ^ (core.Direction & core.time)
# Electric planes meet the spatial slice in their direction; magnetic planes lie within it.
electric_direction = core.Field.dual_inverse() ^ core.t
magnetic_field = (core.Field ^ core.t) * core.t.inverse()
# A third turn cycles z to x, x to y and y to z in the drawing.
to_scene = (-(sm.xy + sm.yz + sm.zx) * (np.pi / (3 * np.sqrt(3)))).exp()
# The direction of the waves' travel, drawn along the first axis, z.
travel = core.z.dual()                                                          # [] Antivector-
electric_colour, magnetic_colour = np.array(to_rgb("crimson")), np.array(to_rgb("dodgerblue"))
axis_colour = np.full(3, 0.5)
# Opacity of each surface of the axis rod.
axis_opacity = 0.37
# Faint cross through the origin, and axis label colour.
cross_colour, label_colour = "0.85", "0.4"
# Polar cross radius as a multiple of the fastest wave speed.
cross_reach = 1.1
# Samples along the propagation axis, arrow stations, and framing times across one period.
n_samples, n_stations, n_framing = 200, 21, 16
# Length of the propagation axis drawn.
z_max = 4.0 * np.pi
# View elevation and azimuth in radians (azimuth zero looks along +y), margin, and figure dimensions.
elevation, azimuth = np.radians(20.0), np.radians(-35.0)
margin, figure_size, figure_dpi = 0.06, (18.3, 3.8), 43
# Pixel widths of field curves, axis rod, and arrows, plus arrow chevron length.
curve_width, axis_width, arrow_width, arrow_chevron = 1.44, 1.08, 1.08, 4.3
# Magnetic discs at every other station: unit-field radius, thickness-to-radius ratio, and opacity.
disc_scale, disc_flatness, disc_opacity = 0.41, 0.1, 0.55
# Rim sense arcs just outside each magnetic disc.
arc_reach = 1.2
arc_start, arc_span = np.radians(30.0), np.radians(240.0)
n_arc = 16
# Polarization arrow stroke widths, chevron length, and dotted guide parameters.
polarization_width, polarization_chevron = 3.5, 14.0
polarization_dots, polarization_opacity = (1.5, 2.5), 0.6


# --- plumbing -------------------------------------------------------------------------
def zero_axes(ax: plt.Axes, horizontal_label: str, vertical_label: str) -> None:
    """Quantity axes run from zero without tick marks or numbers, labelled by `horizontal_label` and `vertical_label`."""
    ax.set_ylim(bottom=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(horizontal_label, color=label_colour)
    ax.set_ylabel(vertical_label, color=label_colour)


def origin_cross(ax: plt.Axes, camera: Camera, horizontal_label: str, vertical_label: str) -> np.ndarray:
    """A faint reference cross through the origin spans the panel along `camera`'s readout directions (`[rows, columns, 4]`)."""
    ax.text(1, 0, horizontal_label, transform=ax.get_yaxis_transform(), ha="right", va="top", color=label_colour)
    ax.text(0, 1, f" {vertical_label}", transform=ax.get_xaxis_transform(), ha="left", va="top", color=label_colour)
    ax.set_axis_off()
    (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
    horizontal, vertical = (plane.dual() / (plane & plane.dual()) for plane in (camera.across, camera.up))
    ends = stack((left * horizontal, right * horizontal, bottom * vertical, top * vertical))  # [4] Bivector+
    return simplices.strokes(ends, Topology.chains([2, 2]), np.array(to_rgb(cross_colour)), charts.FRAME_PX,
                             camera, cap=simplices.Cap.BUTT)


def wave_scene(ax: plt.Axes, modes: Modes, distances: np.ndarray) -> Scene:
    """A camera fitted across `n_framing` snapshots of one period frames the propagation axis inside `ax`."""
    speeds, fields = modes
    axis = solids.origin + (to_scene >> solids.context.include_from_algebra(space_slice(travel * distances)))
    framing_fields = stack(tuple(core.travelling_fields(speeds, fields, distances, n_framing)))  # [n_framing, n_samples] Bivector+
    electric = to_scene >> solids.context.include_from_algebra(electric_direction(framing_fields))  # [n_framing, n_samples] Point-
    view = fitted(solids.view([axis + electric], 0.0, elevation, azimuth, margin, 1), ax)
    radius = axis_width / 2 * view.pixel
    line = solids.rods(axis[0], axis[-1], radius)[None]                         # [1, 3] Plane+ <- Point-
    return Scene(view, np.ones(3), [], [], inks=[], translucent=[(line, axis_colour, axis_opacity)])


def magnetic_discs(stations: solids.Point, fields: core.Field) -> tuple[solids.Quadric, solids.Point]:
    """Magnetic fields (`[...] Bivector+`) at `stations` (`[...] Point-`) form flat quadric discs (`[..., 1] Plane+ <- Point-`) with sense arcs (`[..., n_arc] Point-`)."""
    # Embedding the magnetic field gives the line through the origin it turns about, weighted by field strength.
    magnetic = to_scene >> solids.context.include_from_algebra(magnetic_field(fields))  # [...] Line+
    radii = disc_scale * magnetic.norm().square_root()                          # [...] Scalar+
    centres = solids.translator(stations - solids.origin)                       # [...] Motor+
    pivots = magnetic.normalized()                                              # [...] Line+
    planes = pivots | solids.origin                                             # [...] Plane+
    # Flatten a unit ball across each plane and scale by radius squared so zero field collapses to a point.
    flattening = (1 / disc_flatness**2 - 1) * planes * (planes & solids.Point)  # [...] Plane+ <- Point-
    rounded = (solids.axes * (solids.axes & solids.Point)).sum(axis=-1)         # [] Plane+ <- Point-
    lenses = rounded + flattening - radii**2 * sm.w * (sm.w & solids.Point)     # [...] Plane+ <- Point-
    angles = arc_start + np.linspace(0, arc_span, n_arc)                        # [n_arc]
    turns = (pivots[:, None] * (-angles / 2)).exp()                             # [..., n_arc] Motor+
    heading = to_scene >> solids.context.include_from_algebra(space_slice(travel))  # [] Point-
    rim_starts = solids.origin + heading * (arc_reach * radii)                  # [...] Point-
    rims = (centres[:, None] * turns) >> rim_starts[:, None]                    # [..., n_arc] Point-
    return solids.placed(lenses, centres)[:, None], rims


def wave_image(scene: Scene, field: core.Field, distances: np.ndarray) -> np.ndarray:
    """Renders one snapshot (`[rows, columns, 3]`) of `field` (`[n_samples] Bivector+`) along `distances`."""
    view = scene.view
    axis = solids.origin + (to_scene >> solids.context.include_from_algebra(space_slice(travel * distances)))
    electric = to_scene >> solids.context.include_from_algebra(electric_direction(field))  # [n_samples] Point-

    station_idx = np.linspace(0, n_samples - 1, n_stations, dtype=int)
    stations = axis[station_idx]                                                # [n_stations] Point-
    tips = stations + electric[station_idx]                                     # [n_stations] Point-
    arrows = simplices.tubes(stack((stations, tips), axis=-1).reshape(-1), Topology.chains([2] * n_stations),
                             electric_colour, arrow_width, view, gloss=solids.gloss, chevrons=np.ones(n_stations),
                             chevron=arrow_chevron)
    discs, arcs = magnetic_discs(stations[::2], field[station_idx[::2]])

    n_discs = arcs.shape[0]
    curve_points = concatenate([axis + electric, arcs.reshape(-1)])             # [n_samples + n_discs * n_arc] Point-
    chains = Topology.chains([n_samples] + [n_arc] * n_discs)
    colours = np.repeat([electric_colour, magnetic_colour], [n_samples - 1, n_discs * (n_arc - 1)], axis=0)
    arc_ends = np.zeros((n_discs, n_arc - 1))
    arc_ends[:, -1] = 1.0
    chevrons = np.concatenate([np.zeros(n_samples - 1), arc_ends.reshape(-1)])
    curves = simplices.tubes(curve_points, chains, colours, curve_width, view, gloss=solids.gloss, chevrons=chevrons,
                             chevron=arrow_chevron)
    return scene.draw([], inks=[], translucent=[(discs, magnetic_colour, disc_opacity)],
                      meshes=[(curves, 1.0), (arrows, 1.0)])


def wave_propagation_panel(ax: plt.Axes, modes: Modes, distances: np.ndarray) -> AxesImage:
    """Displays the initial wave field (`tau = 0`) of `modes` along `distances` inside `ax`."""
    speeds, fields = modes
    scene = wave_scene(ax, modes, distances)
    initial_field = next(core.travelling_fields(speeds, fields, distances, 1))  # [n_samples] Bivector+
    return show(ax, wave_image(scene, initial_field, distances), scene.view)


def wave_comparison_figure(glass_modes: Modes, crystal_modes: Modes, distances: np.ndarray) -> plt.Figure:
    """Side-by-side 3D comparison places isotropic glass on the left beside the birefringent crystal on the right."""
    fig = plt.figure(figsize=figure_size, dpi=figure_dpi)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0)
    wave_propagation_panel(fig.add_subplot(1, 2, 1), glass_modes, distances)
    wave_propagation_panel(fig.add_subplot(1, 2, 2), crystal_modes, distances)
    return fig


def wave_comparison_frames(
    glass_modes: Modes,
    crystal_modes: Modes,
    distances: np.ndarray,
    state_pairs: Iterable[tuple[core.Field, core.Field]],
) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames (`[rows, columns, 3]`) for each `(glass_field, crystal_field)` snapshot in `state_pairs`."""
    fig = plt.figure(figsize=figure_size, dpi=figure_dpi)
    glass_axes = fig.add_subplot(1, 2, 1)
    crystal_axes = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0)

    glass_scene = wave_scene(glass_axes, glass_modes, distances)
    crystal_scene = wave_scene(crystal_axes, crystal_modes, distances)
    glass_shown = show(glass_axes, np.full((*glass_scene.view.shape, 3), 255, dtype=np.uint8), glass_scene.view)
    crystal_shown = show(crystal_axes, np.full((*crystal_scene.view.shape, 3), 255, dtype=np.uint8), crystal_scene.view)

    for glass_field, crystal_field in state_pairs:
        glass_shown.set_data(wave_image(glass_scene, glass_field, distances))
        crystal_shown.set_data(wave_image(crystal_scene, crystal_field, distances))
        yield capture(fig)

    plt.close(fig)


def dispersion_panel(ax: plt.Axes, speeds: np.ndarray, residual: core.Scalar, expected: np.ndarray) -> None:
    """Plots each medium's wave residual (`[n_speeds, n_media] Scalar+`) against trial `speeds` with dotted expected speeds."""
    lines = ax.plot(speeds, residual.to_array())
    for line, marks in zip(lines, expected):
        for speed in marks:
            ax.axvline(speed, color=line.get_color(), linestyle=":", alpha=0.8)

    zero_axes(ax, "speed along z", "residual")


def polarizations_panel(ax: plt.Axes, modes: Modes) -> None:
    """Draws the unit electric-field polarizations of the crystal's two modes in the xy plane."""
    _, fields = modes
    units = fields / (fields & core.time).norm()                                # [2] Bivector+
    n_modes = len(units)
    colours = np.stack((electric_colour, magnetic_colour))                      # [n_modes, 3]
    pairs = Topology.chains([2] * n_modes)

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.figure.tight_layout()
    # Pair the field bivectors directly against the swept x-t and y-t planes.
    plane = offset_panel(ax, *core.swept_planes[:2])
    overlay(ax, [
        origin_cross(ax, plane, "x", "y"),
        simplices.strokes(stack((-units, units), axis=-1).reshape(-1), pairs, colours, charts.LINE_PX, plane,
                          dashes=polarization_dots, cap=simplices.Cap.ROUND, opacity=polarization_opacity),
        simplices.strokes(stack((0 * units, units), axis=-1).reshape(-1), pairs, colours, polarization_width, plane,
                          chevrons=np.ones(n_modes), chevron=polarization_chevron)])


def fresnel_surface_panel(
    ax: plt.Axes,
    angles: np.ndarray,
    residual: core.Scalar,
    speeds: np.ndarray,
    threshold: float,
) -> None:
    """Plots the wave-speed sheets in the xz plane from local minima of `residual` (`[n_speeds, n_headings, n_cases] Scalar+`)."""
    for minima, color in zip(np.moveaxis(core.local_minima(residual, threshold), -1, 0), plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        sheets = np.cumsum(minima, axis=0) * minima                             # [..., ...]
        for sheet in range(1, sheets.max() + 1):
            heading_idx, speed_idx = np.nonzero((sheets == sheet).T)
            ax.plot(angles[heading_idx], speeds[1:-1][speed_idx], color=color)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    radius = cross_reach * ax.get_rmax()
    for heading in np.arange(4) * np.pi / 2:
        ax.plot([heading, heading], [0, radius], color=cross_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.set_rmax(radius)
    ax.text(0, radius, "z", ha="left", va="top", color=label_colour)
    ax.text(np.pi / 2, radius, "x", ha="right", va="bottom", color=label_colour)


def fresnel_drag_panel(
    ax: plt.Axes,
    fractions: np.ndarray,
    found_speeds: np.ndarray,
    curve: np.ndarray,
    relativistic: np.ndarray,
    first_order: np.ndarray,
) -> None:
    """Plots co-propagating (red) and counter-propagating (blue) light speeds over exact relativistic and first-order Fresnel curves."""
    for speeds, exact, approximate, color, marker in zip(found_speeds, relativistic, first_order, "rb", "os"):
        ax.plot(fractions, speeds, color=color, marker=marker, linestyle="none")
        ax.plot(curve, exact, color=color)
        ax.plot(curve, approximate, color=color, linestyle="--", alpha=0.7)

    zero_axes(ax, "speed of the glass", "speed of light")


@charts.style()
def dispersion_figure(speeds: np.ndarray, residual: core.Scalar, expected: np.ndarray) -> plt.Figure:
    """Wave-map residuals across trial speeds along z fall to zero at each medium's phase speeds."""
    fig, ax = charts.subplots(400, 280)
    dispersion_panel(ax, speeds, residual, expected)
    fig.tight_layout()
    return fig


@charts.style()
def polarizations_figure(modes: Modes) -> plt.Figure:
    """Unit electric-field arrows show the orthogonal polarizations of the crystal's two modes."""
    fig, ax = charts.subplots(260, 260)
    polarizations_panel(ax, modes)
    return fig


@charts.style()
def fresnel_surface_figure(
    angles: np.ndarray,
    residual: core.Scalar,
    speeds: np.ndarray,
    threshold: float,
) -> plt.Figure:
    """Polar phase-speed sheets trace how propagation speed varies with heading across the xz plane."""
    fig, ax = charts.subplots(300, 300, subplot_kw={"projection": "polar"})
    fresnel_surface_panel(ax, angles, residual, speeds, threshold)
    fig.tight_layout()
    return fig


@charts.style()
def fresnel_drag_figure(
    fractions: np.ndarray,
    found_speeds: np.ndarray,
    curve: np.ndarray,
    relativistic: np.ndarray,
    first_order: np.ndarray,
) -> plt.Figure:
    """Sampled light speeds in moving glass follow relativistic velocity addition past first-order Fresnel drag."""
    fig, ax = charts.subplots(400, 280)
    fresnel_drag_panel(ax, fractions, found_speeds, curve, relativistic, first_order)
    fig.tight_layout()
    return fig
