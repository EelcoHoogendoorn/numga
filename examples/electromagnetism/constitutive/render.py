"""Drawing the waves media carry: the fields along their way, the residual against trial speeds, the
speeds by heading, and the drag of moving glass."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.image import AxesImage

from examples.animation import capture
from examples.electromagnetism.constitutive import core
from examples.wgsl import meshes, show, solids

Modes = tuple[np.ndarray, core.Bivector]
sm = solids.mv
# The drawing's frame: the wave's z across it, x into it and y up; a cyclic order of the axes, so turned, not mirrored.
DRAWN = core.STA.subspace("z x y")
ELECTRIC, MAGNETIC, AXIS_COLOUR = np.array(to_rgb("crimson")), np.array(to_rgb("dodgerblue")), np.full(3, 0.5)
# How much of what lies behind it each surface of the axis hides, its near side and its far side.
AXIS_OPACITY = 0.37
# The plots' faint cross through the origin, and the names of its lines and axes.
CROSS_COLOUR, LABEL_COLOUR = "0.85", "0.4"
# The reach of the cross on polar axes, as a share of the fastest wave's.
CROSS_REACH = 1.1
# The samples along the axis, the stations with arrows, and the times of a period that the camera frames.
N_SAMPLES, N_STATIONS, N_FRAMING = 200, 21, 16
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin, a share of the larger side;
# the pixels across each panel, and the figure's size in inches.
ELEVATION, AZIMUTH = np.radians(20.0), np.radians(-35.0)
MARGIN, N_PIXELS, FIGURE_SIZE = 0.06, 840, (14, 3.8)
# The widths of the field curves, the axis and the arrows' shafts, in pixels; the arrows' heads, as a share of the
# arrow's length, and their radius, as a share of the head's length.
CURVE_WIDTH, AXIS_WIDTH, ARROW_WIDTH = 4.0, 3.0, 3.0
HEAD_SHARE, HEAD_SPREAD = 0.18, 0.35


def _zero_axes(ax: plt.Axes, across: str, up: str) -> None:
    """Axes of a quantity without tick marks or numbers: the quantity from its zero, the axis across at that zero,
    both named by what runs along them."""
    ax.set_ylim(bottom=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(across, color=LABEL_COLOUR)
    ax.set_ylabel(up, color=LABEL_COLOUR)


def _cross(ax: plt.Axes, across: str, up: str) -> None:
    """A plane without scales: a faint cross through the origin, its lines named at their ends."""
    ax.axhline(0, color=CROSS_COLOUR, linewidth=0.7, zorder=0)
    ax.axvline(0, color=CROSS_COLOUR, linewidth=0.7, zorder=0)
    ax.text(1, 0, across, transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=11, color=LABEL_COLOUR)
    ax.text(0, 1, f" {up}", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=11, color=LABEL_COLOUR)
    ax.set_axis_off()


def field_arrows(modes: Modes, taus: np.ndarray, distance: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray]:
    """The electric and magnetic fields an observer at rest reads as arrows, at each distance along z and each
    time `[..., samples, 3]`, for times `[...]`: in the drawing's frame, the wave's z across it, x into it and y up."""
    speeds, fields = modes
    fields = fields / (fields | core.t).squared().square_root()
    phase = omega * (distance[:, None] / speeds - taus[..., None, None])        # [..., samples, modes]
    wave = (fields * np.cos(phase)).sum(axis=-1)                               # [..., samples] Bivector
    return (wave | core.t).cast(DRAWN).kernel, (wave.dual() | core.t).cast(DRAWN).kernel


def wave_image(modes: Modes, tau: float, z_max: float, omega: float) -> tuple[np.ndarray, solids.View]:
    """The travelling E and B fields at one time: the curves their arrows' tips trace along the axis, the arrows at
    stations along it, and the axis; seen by a camera that frames every time of a period alike."""
    distance = np.linspace(0, z_max, N_SAMPLES)
    axis = distance[:, None] * np.eye(3)[0]                                     # [samples, 3]
    reached = field_arrows(modes, np.linspace(0, 2 * np.pi / omega, N_FRAMING, endpoint=False), distance, omega)
    view = solids.view([sm.point(axis + arrows) for arrows in reached], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    electric, magnetic = field_arrows(modes, np.array(tau), distance, omega)

    curves = meshes.tube_samples([(axis + electric, ELECTRIC), (axis + magnetic, MAGNETIC)], CURVE_WIDTH, view)
    line = solids.rods(sm.point(axis[0]), sm.point(axis[-1]), AXIS_WIDTH / 2 * view.pixel)[None]   # [1, 3] Quadric
    # Arrows from the axis at evenly spaced stations, their heads a share of their length, so weak fields stay slender.
    station_idx = np.linspace(0, N_SAMPLES - 1, N_STATIONS, dtype=int)
    stations = axis[station_idx]
    parts = []
    for arrows, colour in ((electric, ELECTRIC), (magnetic, MAGNETIC)):
        tips = arrows[station_idx]
        head_lengths = HEAD_SHARE * np.linalg.norm(tips, axis=-1)
        shafts, heads = solids.arrows(sm.point(stations), sm.point(stations + tips), ARROW_WIDTH / 2 * view.pixel,
                                      head_lengths, HEAD_SPREAD * head_lengths)
        parts += [(shafts, colour), (heads, colour)]
    return solids.draw(parts, view, np.ones(3), translucent=[(line, AXIS_COLOUR, AXIS_OPACITY)],
                       meshes=[(curves, 1.0)]), view


def draw_wave_propagation(
    ax: plt.Axes,
    modes: Modes,
    tau: float = 0.0,
    z_max: float = 4.0 * np.pi,
    omega: float = 1.0,
) -> AxesImage:
    """Draw the travelling E and B field vectors in the medium along the propagation axis z, at one time; the
    image, for an animation to set its data."""
    return show(ax, *wave_image(modes, tau, z_max, omega))


def draw_wave_comparison_3d(
    fig: plt.Figure,
    glass_modes: Modes,
    crystal_modes: Modes,
    z_max: float = 4.0 * np.pi,
) -> tuple[plt.Axes, plt.Axes]:
    """Render side-by-side views comparing isotropic vs birefringent medium."""
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    # Isotropic glass on the left, birefringent crystal on the right.
    draw_wave_propagation(ax1, glass_modes, z_max=z_max)
    draw_wave_propagation(ax2, crystal_modes, z_max=z_max)
    return ax1, ax2


def draw_wave_comparison_figure(
    glass_modes: Modes,
    crystal_modes: Modes,
    z_max: float = 4.0 * np.pi,
) -> plt.Figure:
    """Render standalone figure with side-by-side views comparing isotropic vs birefringent medium."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=120)
    draw_wave_comparison_3d(fig, glass_modes, crystal_modes, z_max=z_max)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0)
    return fig


def animate_wave_comparison(
    glass_modes: Modes,
    crystal_modes: Modes,
    n_frames: int,
    z_max: float = 4.0 * np.pi,
    omega: float = 1.0,
) -> list[np.ndarray]:
    """Frames of the travelling E and B field vectors through one period, in isotropic glass beside
    the birefringent crystal."""
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=80)
    glass_axes = fig.add_subplot(1, 2, 1)
    crystal_axes = fig.add_subplot(1, 2, 2)
    # One layout for every frame, so the axes stay put while the waves move.
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0)
    panels = [(draw_wave_propagation(ax, modes, z_max=z_max, omega=omega), modes)
              for ax, modes in ((glass_axes, glass_modes), (crystal_axes, crystal_modes))]

    frames = []
    for tau in np.linspace(0, 2.0 * np.pi / omega, n_frames, endpoint=False):
        for shown, modes in panels:
            shown.set_data(wave_image(modes, float(tau), z_max, omega)[0])
        frames.append(capture(fig))

    plt.close(fig)
    return frames


def draw_dispersion(ax: plt.Axes, speeds: np.ndarray, residual: core.Scalar, expected: np.ndarray) -> None:
    """Each medium's residual against the trial speeds `[speeds, media]`, falling to zero at its expected
    speeds `[media, 2]`, marked in its colour."""
    lines = ax.plot(speeds, residual.to_array(), linewidth=1.5)
    for line, marks in zip(lines, expected):
        for speed in marks:
            ax.axvline(speed, color=line.get_color(), linestyle=":", linewidth=1.2, alpha=0.8)

    _zero_axes(ax, "speed along $z$", "residual")


def draw_polarizations(
    ax: plt.Axes,
    modes: Modes,
) -> None:
    """Draw 2D transverse polarization arrows in the xy plane for the slow and fast birefringent modes."""
    _, fields = modes
    electric = fields | core.t
    electric = electric / electric.squared().square_root()
    arrows = electric.cast(core.STA.subspace("x y")).kernel
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
    for dips, color in zip(np.moveaxis(core.dips(speeds, residual, threshold), -1, 0), plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        # Each dip's place among the dips at its heading, counting from the slowest.
        sheets = np.cumsum(dips, axis=0) * dips                                # [speeds - 2, headings]
        for sheet in range(1, sheets.max() + 1):
            heading_idx, speed_idx = np.nonzero((sheets == sheet).T)
            ax.plot(angles[heading_idx], speeds[1:-1][speed_idx], color=color, linewidth=1.6)

    # Zero radians along +z (North), angles clockwise: +x along East.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    # A faint cross through the origin, past the fastest wave, its lines named at their ends by the directions
    # they run along.
    reach = CROSS_REACH * ax.get_rmax()
    for heading in np.arange(4) * np.pi / 2:
        ax.plot([heading, heading], [0, reach], color=CROSS_COLOUR, linewidth=0.7, zorder=0)
    ax.set_rmax(reach)
    ax.text(0, reach, "$z$", ha="left", va="top", fontsize=11, color=LABEL_COLOUR)
    ax.text(np.pi / 2, reach, "$x$", ha="right", va="bottom", fontsize=11, color=LABEL_COLOUR)


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
        ax.plot(fractions, speeds, color=color, marker=marker, linestyle="none", markersize=5)
        ax.plot(curve, exact, color=color, linewidth=1.5)
        ax.plot(curve, approximate, color=color, linestyle="--", linewidth=1.1, alpha=0.7)

    _zero_axes(ax, "speed of the glass", "speed of light")


def draw_dispersion_figure(speeds: np.ndarray, residual: core.Scalar, expected: np.ndarray) -> plt.Figure:
    """Each medium's residual against the trial speeds."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    draw_dispersion(ax, speeds, residual, expected)
    plt.tight_layout()
    return fig


def draw_polarizations_figure(modes: Modes) -> plt.Figure:
    """Render standalone figure for 2D transverse polarization eigenmode quivers."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    draw_polarizations(ax, modes)
    plt.tight_layout()
    return fig


def draw_fresnel_surface_figure(angles: np.ndarray, residual: core.Scalar, speeds: np.ndarray,
                                threshold: float) -> plt.Figure:
    """The speeds of the waves against their heading, on polar axes."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120, subplot_kw={"projection": "polar"})
    draw_fresnel_surface_polar(ax, angles, residual, speeds, threshold)
    plt.tight_layout()
    return fig


def draw_fresnel_drag_figure(
    fractions: np.ndarray,
    found: np.ndarray,
    curve: np.ndarray,
    relativistic: np.ndarray,
    first_order: np.ndarray,
) -> plt.Figure:
    """The speeds of light in the moving glass, against the speed of the glass."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    draw_fresnel_drag_curves(ax, fractions, found, curve, relativistic, first_order)
    plt.tight_layout()
    return fig
