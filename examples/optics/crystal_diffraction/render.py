"""GPU overlays draw crystal atoms and cyan reciprocal-lattice rings alongside logarithmic
momentum-transfer brightness maps in both spacetime and spatial algebras."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.image import AxesImage

from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay, pinned
from numga import Extensor, stack
from numga.render import Camera, drawing, read, simplices
from numga.topology import Topology

position_limit = 7.8
brightness_floor = 1e-4
# Margin around and between square panels in pixels.
gap = 6
# Marker radius, ring radius and stroke width in pixels, ring vertex count, and animation stroke scale.
atom_radius = 1.6
atom_colour = to_rgb("#0d9488")
ring_radius, ring_width = 4.9, 1.4
ring_colour = to_rgb("#67e8f9")
n_ring_vertices = 24
cross_colour = to_rgb("#cbd5e1")
movie_scale = 1.5


# --- plumbing -------------------------------------------------------------------------
def panel_grid(
    n_rows: int, n_columns: int, n_pixels: int
) -> tuple[plt.Figure, list[list[Axes]]]:
    """Square axes of `n_pixels` pixels separated by `gap`-pixel margins tile a figure in `n_rows`
    rows by `n_columns` columns."""
    span = n_pixels + gap
    width, height = n_columns * span + gap, n_rows * span + gap
    figure = plt.figure(figsize=(width / charts.DPI, height / charts.DPI), dpi=charts.DPI)
    axes = [
        [
            figure.add_axes(
                (
                    (gap + column * span) / width,
                    (gap + (n_rows - 1 - row) * span) / height,
                    n_pixels / width,
                    n_pixels / height,
                )
            )
            for column in range(n_columns)
        ]
        for row in range(n_rows)
    ]
    return figure, axes


def atom_camera(ax: Axes, positions: Extensor) -> Camera:
    """Labels the x and y axes on `ax` and configures an offset camera reading atom positions
    against the x and y planes."""
    ax.text(position_limit, 0, "x", ha="right", va="bottom", color="#94a3b8")
    ax.text(0, position_limit, " y", ha="left", va="top", color="#94a3b8")
    ax.set(xlim=(-position_limit, position_limit), ylim=(-position_limit, position_limit))
    ax.set_axis_off()
    mv = positions.context.multivector
    return offset_panel(ax, mv.x, mv.y)


def atom_layer(positions: Extensor, radius: float, camera: Camera) -> np.ndarray:
    """Filled markers of `radius` pixels mark `positions` (`[n_atoms]`) over faint origin crosshairs
    on `camera`."""
    horizontal_axis, vertical_axis = (
        plane.dual() / (plane & plane.dual()) for plane in (camera.across, camera.up)
    )
    ends = position_limit * stack(
        (-horizontal_axis, horizontal_axis, -vertical_axis, vertical_axis)
    )                                                                        # [4] Antivector-
    return drawing.composited(
        [
            simplices.strokes(
                ends,
                Topology.chains([2, 2]),
                cross_colour,
                charts.FRAME_PX,
                camera,
                cap=simplices.Cap.BUTT,
            ),
            simplices.markers(positions, atom_colour, radius, camera),
        ]
    )


def reciprocal_rings(reciprocal: Extensor, scale: float, camera: Camera) -> np.ndarray:
    """Polygonal rings scaled by `scale` outline each wavevector in `reciprocal`
    (`[n_reflections]`) on `camera`."""
    n_reflections = len(reciprocal)
    horizontal_step, vertical_step = (
        plane.dual() / read(plane, plane.dual()) for plane in (camera.across, camera.up)
    )
    angles = np.linspace(0, 2 * np.pi, n_ring_vertices, endpoint=False)
    rotors = ((horizontal_step ^ vertical_step) * (-angles / 2)).exp()       # [n_ring_vertices] Rotor+
    circle = (rotors >> horizontal_step) * (scale * ring_radius * camera.pixel)  # [n_ring_vertices] Vector+
    ring_points = (reciprocal[:, None] + circle).reshape(-1)                 # [n_reflections * n_ring_vertices] Vector+
    return simplices.strokes(
        ring_points,
        Topology.loops([n_ring_vertices] * n_reflections),
        ring_colour,
        scale * ring_width,
        camera,
    )


def diffraction_panel(
    ax: Axes,
    reciprocal: Extensor,
    transfers: Extensor,
    brightness: Extensor,
    scale: float,
) -> tuple[AxesImage, AxesImage, Camera]:
    """Logarithmic `brightness` over `transfers` (`[n_pixels, n_pixels]`) underlays cyan
    `reciprocal` rings on `ax`, yielding the brightness image, ring overlay, and panel camera."""
    mv = transfers.context.multivector
    n_samples = transfers.shape[0] * transfers.shape[1]
    readings = transfers.reshape(-1) & stack((mv.x.dual(), mv.y.dual()))[:, None]  # [2, n_samples] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)                 # [2] Scalar+, [2] Scalar+
    half_step = (high - low) / (2 * (np.array(transfers.shape[::-1]) - 1))   # [2] Scalar+
    extent = tuple(stack((low - half_step, high + half_step), axis=-1).reshape(-1).to_array())
    # Rows increase upward in y whereas image rows start from the top.
    image = pinned(ax, brightness.to_array()[::-1], extent)
    image.set(cmap="magma", norm=LogNorm(vmin=brightness_floor, vmax=1))
    ax.set(xlim=extent[:2], ylim=extent[2:])
    camera = offset_panel(ax, mv.x.dual(), mv.y.dual())
    return image, overlay(ax, [reciprocal_rings(reciprocal, scale, camera)]), camera


@charts.style()
def comparison_figure(
    positions: Extensor,
    reciprocal: Extensor,
    transfers: Extensor,
    brightness: Extensor,
) -> plt.Figure:
    """Two rows of `n_cases` panels place each deformed atom layer above its logarithmic diffraction
    pattern and carried reciprocal rings."""
    n_cases = positions.shape[0]
    figure, (top, bottom) = panel_grid(2, n_cases, transfers.shape[-1])
    for case in range(n_cases):
        overlay(top[case], [atom_layer(positions[case], atom_radius, atom_camera(top[case], positions[case]))])
        diffraction_panel(bottom[case], reciprocal[case], transfers, brightness[case], 1.0)
    return figure


def diffraction_animation(
    frames: Iterable[tuple[Extensor, Extensor, Extensor]], transfers: Extensor
) -> Iterator[np.ndarray]:
    """Yields RGB frames showing the deforming atom layer beside its live diffraction pattern and
    reciprocal-lattice rings."""
    frames = iter(frames)
    with charts.style():
        figure, ((left, right),) = panel_grid(1, 2, transfers.shape[-1])
        positions, reciprocal, brightness = next(frames)
        camera = atom_camera(left, positions)
        atoms_seen = overlay(left, [atom_layer(positions, movie_scale * atom_radius, camera)])
        image, rings_seen, ring_camera = diffraction_panel(
            right, reciprocal, transfers, brightness, movie_scale
        )
        yield capture(figure)
        for positions, reciprocal, brightness in frames:
            atoms_seen.set_data(atom_layer(positions, movie_scale * atom_radius, camera))
            image.set_data(brightness.to_array()[::-1])
            rings_seen.set_data(drawing.composited([reciprocal_rings(reciprocal, movie_scale, ring_camera)]))
            yield capture(figure)
        plt.close(figure)
