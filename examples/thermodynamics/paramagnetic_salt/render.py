"""Planar camera projections and logarithmic charts render adiabatic demagnetization of a paramagnetic crystal slab
alongside its temperature-entropy diagram.

Magnetic field and spin magnetization live as planes and patches in the slab's own spatial plane; pairing
`magnet_plane & spins` distinguishes aligned dots from opposed rings at interior vertices, while `field & spin` scales
the green wash across the magnet gap. Spatial planes `x` and `y` with point weight `w` project the slab and its
retracting bath and load bars, while logarithmic temperature colours tint the triangles and `s & states` with
`potentials & entropy` place states onto the temperature-entropy chart."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga import dim, stack
from numga.render import contours, drawing, simplices
from numga.render.camera import Camera, preimage
from numga.topology import Topology
from examples.thermodynamics.paramagnetic_salt import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import image_figure

mv = core.mv
# Logarithmic temperature colour limits and white-blend fraction keeping spin markers legible.
temperature_colours = colormaps["RdYlBu_r"]
coldest, warmest, paleness = 3e-4, 5.0, 0.5
# Magnet gap bounds in slab widths, green wash colour, peak wash opacity, and outline stroke width in pixels.
gap_low, gap_high = np.array([-0.3, -0.22]), np.array([2.3, 1.22])
magnet_colour, magnet_depth, magnet_outline = np.array(to_rgb("#3a9a6b")), 0.6, 1.5
# Slab border and triangle stroke widths, reservoir bar geometry, and open-switch separation in pixels.
outline_colour, outline_width, edge_width = np.array(to_rgb("#3d4148")), 1.5, 1.0
bar_width, bar_outline, bar_gap = 5.0, 1.5, 7.0
# Spin marker colour, white centre colour for opposed rings, outer radius, and ring rim width in pixels.
spin_colour, opposed_colour, spin_radius, spin_rim = np.array(to_rgb("#2b2140")), np.ones(3), 4.5, 1.7
# Planar camera bounds in slab widths and pixel scale (`pixel/length`).
scene_low = np.array([-0.36, -0.28]) * mv.scalar(1.0, unit=dim.length)
scene_high = np.array([2.36, 1.28]) * mv.scalar(1.0, unit=dim.length)
pixels_per_width = mv.scalar(110.0, unit=dim.pixel / dim.length)
# Companion temperature-entropy chart dimensions, curve strokes, state marker style, and axis ranges.
chart_width, ideal_colour, field_line_px = 280, "0.6", 2.5
state_dot, state_colour, state_rim = 7.0, "#1d2228", "white"
chart_entropy_range, chart_temperature_range = (0.0, 0.76), (3e-4, 6.0)
# Equation-of-state curve colour and shared logarithmic temperature tick labels.
profile_colour = "#1d2228"
named_temperatures, temperature_names = [1e-3, 1e-2, 1e-1, 1.0], ["1 mK", "10 mK", "100 mK", "1 K"]
# Standalone ideal-cycle and equation-of-state chart styling and pixel dimensions.
cycle_colour, line_colour, name_colour = "#1d2228", "0.75", "0.45"
name_offset = (-4, 0)
cycle_px, line_px = 2.0, 1.0
entropy_range, temperature_range = (0.0, 0.76), (4e-4, 2.4)
cycle_size, state_size = (300, 240), (300, 210)


# --- plumbing -------------------------------------------------------------------------
def temperature_rgb(temperatures: np.ndarray) -> np.ndarray:
    """Logarithmic scaling between `coldest` and `warmest` maps dimensionless `temperatures` (`[...]`) onto RGB colours
    (`[..., 3]`)."""
    log_fraction = np.log(temperatures / coldest) / np.log(warmest / coldest)
    return np.asarray(temperature_colours(np.clip(log_fraction, 0, 1)))[..., :3]


def white_composite(picture: np.ndarray) -> np.ndarray:
    """Alpha blending composites premultiplied RGBA `picture` (`[rows, columns, 4]`) over a white background as 8-bit
    RGB bytes (`[rows, columns, 3]`)."""
    rgb = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def scene_points(x_coords: np.ndarray, y_coords: np.ndarray) -> core.Point:
    """Planar coordinates `x_coords` and `y_coords` (`[...]`) lift to unit-weight projective points (`[...] Point-, 1 | length`)."""
    return (mv(core.Plane, np.stack([x_coords, y_coords], axis=-1)) * dim.length + mv.w).dual()


def scene_camera() -> Camera:
    """Planar pairing with `pixels_per_width * x`, `pixels_per_width * y`, and point weight `w` frames the crystal slab
    and surrounding magnet gap into `dim.pixel`."""
    across, up = pixels_per_width * mv.x, pixels_per_width * mv.y             # Plane+, pixel/length
    low_px, high_px = pixels_per_width * scene_low, pixels_per_width * scene_high   # [2] Scalar+, pixel
    return Camera(across, up, 0 * across, mv.w, 0.0, (low_px + high_px) / 2, (high_px - low_px) / 2)


def ends(topology: Topology, points: core.Point) -> tuple[list[np.ndarray], list[Topology]]:
    """Minimum and maximum `x` pairings on `points` (`[V] Point-`) select the left and right boundary vertex arrays
    and border edge topologies of `topology`."""
    x_coords = (mv.x & points).to_array().ravel()                             # [V]
    end_vertices = [np.flatnonzero(x_coords == x_coords.min()), np.flatnonzero(x_coords == x_coords.max())]
    border = topology.border
    return end_vertices, [border.subset(np.isin(border.elements[1], vertices).all(axis=-1), 1)
                          for vertices in end_vertices]


def reservoir_bars(points: core.Point, end_edges: list[Topology], is_linked: np.ndarray, colours: np.ndarray,
                   camera: Camera) -> list:
    """Outlined boundary strokes along `end_edges` render the reservoir bars in `colours` (`[2, 3]`), flush against
    `points` where `is_linked` (`[2]`) holds and offset by `bar_gap` when open."""
    _, step_across, _, _ = preimage(camera, core.Point)                       # [] Point-, length/pixel
    outward_steps = stack((-step_across, step_across))                        # [2] Point-, length/pixel
    bar_offsets = (bar_width / 2 + bar_outline + bar_gap * ~is_linked) * mv.scalar(1.0, unit=dim.pixel)   # [2] Scalar+, pixel
    # Draw an outer stroke first so each temperature-tinted bar stands clear of the white background.
    return [layer for edges, step, offset, colour in zip(end_edges, outward_steps, bar_offsets, colours)
            for layer in (simplices.strokes(points + offset * step, edges, outline_colour,
                                            bar_width + 2 * bar_outline, camera),
                          simplices.strokes(points + offset * step, edges, colour, bar_width, camera))]


def scene(topology: Topology, points: core.Point, end_vertices: list[np.ndarray], end_edges: list[Topology],
          sites: np.ndarray, reservoir_colours: np.ndarray, highest_field: core.Scalar,
          salt_potentials: core.Potential, field: core.Magnetic, links: core.Scalar,
          spins: core.Spin, camera: Camera) -> np.ndarray:
    """Composited triangle tints, gap wash, reservoir bars, and interior spin markers render one RGB frame of the slab
    from `salt_potentials`, `field`, `links`, and `spins`."""
    V, E, F = topology.n_elements
    temperatures = ((salt_potentials & core.entropy) / dim.temperature).batch().to_array().ravel()   # [V]
    is_linked = (links > 0).ravel()                                           # [V]
    is_opposed = ((core.magnet_plane & spins) < 0).ravel()[sites]             # [V_interior]
    # Square-root scaling keeps millitesla fields visible in the wash while leaving zero field white.
    depth = magnet_depth * np.sqrt(float(((field & core.spin) / highest_field).to_array()))
    gap_corners = scene_points(np.array([gap_low[0], gap_high[0], gap_high[0], gap_low[0]]),
                               np.array([gap_low[1], gap_low[1], gap_high[1], gap_high[1]]))   # [4] Point-
    return white_composite(drawing.composited([
        contours.picture(gap_corners, Topology(4, [[0, 1, 2], [0, 2, 3]]), np.zeros(4),
                         1 - depth * (1 - magnet_colour), camera),
        simplices.strokes(gap_corners, Topology.loops([4]), magnet_colour, magnet_outline, camera),
        contours.picture(points, topology, np.zeros(V), paleness + (1 - paleness) * temperature_rgb(temperatures),
                         camera),
        simplices.strokes(points, topology.border, outline_colour, outline_width, camera),
        *reservoir_bars(points, end_edges, np.array([is_linked[vertices].any() for vertices in end_vertices]),
                        reservoir_colours, camera),
        simplices.markers(points[sites], spin_colour, spin_radius, camera),
        simplices.markers(points[sites][is_opposed], opposed_colour, spin_radius - spin_rim, camera)]))


def chart(height: int, corner_entropies: core.Scalar, corner_temperatures: core.Scalar,
          line_states: core.State, line_potentials: core.Potential):
    """Logarithmic temperature and entropy-per-spin axes of height `height` pixels hold the static reference isofields,
    ideal cycle rectangle, and live field-curve and state-dot artists."""
    corner_s = (corner_entropies / (core.one_spin * core.boltzmann)).to_array()   # [4]
    corner_t = (corner_temperatures / dim.temperature).to_array()             # [4]
    figure = plt.figure(figsize=(chart_width / charts.DPI, height / charts.DPI), dpi=charts.DPI)
    ax = figure.add_axes((0.2, 0.27, 0.75, 0.69))
    for states, salt_potentials in zip(line_states, line_potentials):
        ax.plot(*chart_readings(states, salt_potentials, core.one_spin)[::-1], color=line_colour)
    ax.plot(np.append(corner_t, corner_t[0]), np.append(corner_s, corner_s[0]), color=ideal_colour)
    ax.set_xscale("log")
    ax.set_xticks(named_temperatures, temperature_names)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([0, np.log(2)], ["0", "ln 2"])
    ax.set(xlim=chart_temperature_range, ylim=chart_entropy_range, xlabel="temperature", ylabel="entropy per spin")
    ax.spines[["top", "right"]].set_visible(False)
    field_line, = ax.plot([], [], color=magnet_colour, linewidth=charts.points(field_line_px))
    dots, = ax.plot([], [], "o", color=state_colour, markersize=charts.points(state_dot), markeredgecolor=state_rim,
                    markeredgewidth=charts.points(1.0))
    return figure, field_line, dots


def chart_readings(states: core.State, salt_potentials: core.Potential,
                   spins: core.Scalar) -> tuple[np.ndarray, np.ndarray]:
    """Pairing extensive `states` with `s` divided by `spins * boltzmann` and `salt_potentials` with `entropy` extracts
    intensive entropy per spin and temperature arrays (`[...]`)."""
    return (((mv.s & states) / (spins * core.boltzmann)).batch().to_array().reshape(-1),
            ((salt_potentials & core.entropy) / dim.temperature).batch().to_array().reshape(-1))


# --- drawing --------------------------------------------------------------------------
@charts.style()
def draw_equation_of_state(states: core.State, salt_potentials: core.Potential,
                           moment: core.Scalar) -> Figure:
    """Magnetization shares from single-spin `states` (`[...]`) against `field / temperature` from `salt_potentials`
    (`[...]`) trace the spin-half saturation curve alongside the dashed Curie-law slope."""
    shares = ((core.magnet_plane & states) / (core.one_spin * moment)).to_array()   # [...]
    ratios = (((salt_potentials & core.spin) / (salt_potentials & core.entropy)) / (dim.field / dim.temperature)).to_array()   # [...]
    curie_field = float(((core.boltzmann / moment) / (dim.field / dim.temperature)).to_array())
    figure, ax = charts.subplots(*state_size, layout="constrained")
    ax.plot(ratios, shares, color=profile_colour)
    ax.plot([0, curie_field], [0, 1], "--", color="0.55")
    ax.set(xlim=(0, ratios.max()), ylim=(0, 1.05), xlabel="field / temperature (T/K)", ylabel="magnetization")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1], ["0", "saturation"])
    ax.spines[["top", "right"]].set_visible(False)
    return figure


@charts.style()
def draw_cycle(corner_entropies: core.Scalar, corner_temperatures: core.Scalar,
               line_states: core.State, line_potentials: core.Potential) -> Figure:
    """Logarithmic temperature-entropy axes display the ideal refrigeration cycle through `corner_entropies` and
    `corner_temperatures` (`[4]`) over labelled constant-field curves."""
    corner_s = (corner_entropies / (core.one_spin * core.boltzmann)).to_array()   # [4]
    corner_t = (corner_temperatures / dim.temperature).to_array()             # [4]
    line_s = ((mv.s & line_states) / (core.one_spin * core.boltzmann)).to_array()
    line_t = ((line_potentials & core.entropy) / dim.temperature).to_array()
    line_fields = ((line_potentials[:, -1] & core.spin) / dim.field).to_array()   # [2]
    figure, ax = charts.subplots(*cycle_size, layout="constrained")
    for entropies, temperatures, field in zip(line_s, line_t, line_fields):
        ax.plot(entropies, temperatures, color=line_colour, linewidth=charts.points(line_px))
        name = f"{field:g} T" if field >= 0.1 else f"{1000 * field:g} mT"
        ax.annotate(name, (entropies[-1], temperatures[-1]), textcoords="offset points",
                    xytext=name_offset, ha="right", va="center", color=name_colour)
    ax.plot(np.append(corner_s, corner_s[0]), np.append(corner_t, corner_t[0]), color=cycle_colour,
            linewidth=charts.points(cycle_px))
    ax.set_yscale("log")
    ax.set_xticks([np.log(2)], ["ln 2"])
    ax.set_yticks(named_temperatures, temperature_names)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set(xlim=entropy_range, ylim=temperature_range, xlabel="entropy per spin", ylabel="temperature")
    ax.spines[["top", "right"]].set_visible(False)
    return figure


def draw_crystal(topology: Topology, vertices: core.Point, sites: np.ndarray, bath_temperature: core.Scalar,
                 load_temperature: core.Scalar) -> Figure:
    """Resting triangle fills and edges at `vertices` (`Point-[V]`) show the slab with interior spin `sites`
    (`[V_interior]`) and open bath and load bars."""
    V, E, F = topology.n_elements
    points = vertices.batch()                                                 # [V] Point-
    _, end_edges = ends(topology, points)
    camera = scene_camera()
    reservoir_temps = (stack((bath_temperature, load_temperature)) / dim.temperature).to_array()
    return image_figure(white_composite(drawing.composited([
        contours.picture(points, topology, np.zeros(V), np.full(3, 0.88), camera),
        simplices.strokes(points, topology, outline_colour, edge_width, camera),
        *reservoir_bars(points, end_edges, np.zeros(2, dtype=bool), temperature_rgb(reservoir_temps), camera),
        simplices.markers(points[sites], spin_colour, 3.0, camera)])))


def animate_crystal(topology: Topology, vertices: core.Point, cell_spins: core.Scalar, sites: np.ndarray,
                    bath_temperature: core.Scalar, load_temperature: core.Scalar,
                    highest_field: core.Scalar, corner_entropies: core.Scalar,
                    corner_temperatures: core.Scalar, line_states: core.State,
                    line_potentials: core.Potential,
                    frames: Iterable[tuple[core.State, core.Potential, core.Magnetic,
                                           core.Scalar, core.Spin, core.State,
                                           core.Potential]]) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames combining the rendered crystal scene with the live temperature-entropy chart
    across `frames`."""
    points = vertices.batch()                                                 # [V] Point-
    end_vertices, end_edges = ends(topology, points)
    reservoir_colours = temperature_rgb((stack((bath_temperature, load_temperature)) / dim.temperature).to_array())   # [2, 3]
    camera = scene_camera()
    y_coords = ((mv.y / dim.length) & points).to_array().ravel()              # [V]
    # Track vertices along the horizontal row nearest the slab's centreline on the temperature-entropy chart.
    middle_row = np.flatnonzero(np.isclose(y_coords, y_coords[np.argmin(np.abs(y_coords - (y_coords.min() + y_coords.max()) / 2))]))
    with charts.style():
        figure, field_line, dots = chart(camera.shape[0], corner_entropies, corner_temperatures, line_states,
                                         line_potentials)
        for states, salt_potentials, field, links, spins, field_states, field_potentials in frames:
            field_line.set_data(*chart_readings(field_states, field_potentials, core.one_spin)[::-1])
            entropies, temperatures = chart_readings(states, salt_potentials, cell_spins)
            dots.set_data(temperatures[middle_row], entropies[middle_row])
            yield np.concatenate([scene(topology, points, end_vertices, end_edges, sites, reservoir_colours,
                                        highest_field, salt_potentials, field, links, spins, camera),
                                  capture(figure)], axis=1)
        plt.close(figure)
