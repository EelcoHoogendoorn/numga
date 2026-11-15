"""Rendering for magnetic skyrmions tiles the periodic triangular lattice across a rectangular viewport, colours out-of-plane spin components in quantized bands, overlays in-plane spin components as flow streaks, and plots crystallization transients.

Because triangles straddling the torus wrap have vertices at opposite ends of the fundamental rhombus, each triangle is reconstructed locally from its first vertex and two edge steps and then translated across all nine neighbour copies so the rectangular frame clips cleanly without torn faces. Camera planes pair directly with site points, while pairing spins with the film plane and perpendicular coordinate planes reads out the out-of-plane colour scalar and in-plane streak directions.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from examples.util import charts
from examples.util.panels import image_figure
from numga import stack
from numga.extensor import Extensor
from numga.render import contours, flow, simplices
from numga.render.camera import Camera
from numga.render.drawing import composited
from numga.topology import Topology

# Quantized coolwarm colour bands keep animated GIF palettes compact.
field_colormap = colormaps["coolwarm"]
n_bands = 13
n_animation_colours = 32
n_pixels = 320
n_panel_pixels, panel_gap = 260, 8
streak_ink, full_streak, streak_length = np.array(to_rgb("#26262e")), 1.0, 14.0
corner_reach = 7.0
bond_colour, bond_width, site_radius = np.array(to_rgb("#5b6470")), 1.5, 3.0
wrap_colour, wrap_width, wrap_dashes = np.array(to_rgb("#c0392b")), 1.5, (6.0, 4.0)


def white_composite(picture: np.ndarray) -> np.ndarray:
    """Alpha compositing blends an RGBA image (`[n_rows, n_columns, 4]`) onto a white background as RGB bytes (`[n_rows, n_columns, 3]`)."""
    composite = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(composite, 0, 1) * 255).astype(np.uint8)


def lattice_shifts(periods: Extensor) -> Extensor:
    """Nine periodic translation offsets (`[9] Point-`) span one period backward, zero, and one period forward along both `periods` (`[2] Point-`)."""
    steps = np.array([-1, 0, 1])
    return (periods[0] * steps[:, None] + periods[1] * steps[None, :]).reshape(-1)   # [9] Point-


def tiled_triangles(topology: Topology, vertices: Extensor, bonds: Extensor, periods: Extensor) -> tuple[Extensor, Topology]:
    """Local edge reconstructions translated across all nine periodic copies (`[27 * F] Point-`) keep triangles straddling the torus wrap whole."""
    V, E, F = topology.n_elements
    sides = topology.boundary[2, 1] * bonds                                   # [3] Point-[F]
    first_corners = (topology.incidence[2, 0][0] * vertices).batch()          # [F] Point-
    # Walking along the incident sides places the second and third corners next to the first without wrap jumps.
    face_corners = stack((first_corners, first_corners + sides[2].batch(), first_corners - sides[1].batch()), axis=-1)   # [F, 3] Point-
    places = (face_corners[None] + lattice_shifts(periods)[:, None, None]).reshape(-1)   # [27 * F] Point-
    n_places = places.shape[0]
    return places, Topology(n_places, np.arange(n_places).reshape(-1, 3))


def film_camera(periods: Extensor, low: np.ndarray, high: np.ndarray, n_view_pixels: int) -> Camera:
    """A planar camera `n_view_pixels` wide frames the rectangle from `low` to `high` (`[2]`) across the horizontal and vertical coordinate planes."""
    mv = periods.context.multivector
    return Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, (low + high) / 2, (high - low) / 2, n_view_pixels)


def period_bounds(periods: Extensor) -> np.ndarray:
    """Coordinate-plane pairings with `periods` (`[2] Point-`) give the horizontal and vertical extent (`[2]`) of the tiled viewport."""
    mv = periods.context.multivector
    return np.abs((stack((mv.x, mv.y))[None] & periods[:, None]).to_array()).max(axis=0)


def spin_layers(topology: Topology, spins: Extensor, places: Extensor, tile_topology: Topology, view: Camera) -> list:
    """Quantized out-of-plane colour bands and in-plane flow streaks depict `spins` (`Spin-[V]`) on the tiled triangles."""
    V, E, F = topology.n_elements
    mv = spins.context.multivector
    corner_spins = topology.incidence[2, 0] * spins                           # [3] Spin-[F]
    face_spins = stack([corner_spins[index].batch() for index in range(3)], axis=-1)   # [F, 3] Spin-
    n_places = places.shape[0]
    n_copies = n_places // (3 * F)
    tile_spins = face_spins[None].broadcast_to((n_copies, F, 3)).reshape(-1)  # [n_places] Spin-
    field_components = np.asarray((mv.xy & tile_spins).to_array())
    # Pairing with yz and zx reads the spin's in-plane precession components as horizontal and vertical film steps.
    in_plane_readings = stack((mv.yz, mv.zx)) & tile_spins[:, None]           # [n_places, 2] Scalar+
    in_plane_steps = (stack((mv.x.dual(), mv.y.dual())) * in_plane_readings).sum(axis=-1)   # [n_places] Point-
    colours = field_colormap((field_components + 1) / 2)[:, :3]
    band_centres = (np.arange(n_bands) + 0.5) / n_bands                       # [n_bands]
    bands = contours.Bands(-1.0, 2 / n_bands, field_colormap(band_centres))
    return [
        contours.picture(places, tile_topology, field_components, colours, view, bands=(bands,)),
        flow.layer(places, tile_topology, in_plane_steps, view, full_streak, streak_ink, length=streak_length),
    ]


def lattice_figure(topology: Topology, vertices: Extensor, bonds: Extensor, periods: Extensor) -> Figure:
    """A close-up around the lattice origin shows tiled bonds, sites, and the dashed fundamental periods where opposite edges wrap."""
    V, E, F = topology.n_elements
    view = film_camera(periods, -np.full(2, corner_reach) / 4, np.full(2, corner_reach), n_pixels)
    tail_sites = (topology.incidence[1, 0][0] * vertices).batch()             # [E] Point-
    bond_ends = stack((tail_sites, tail_sites + bonds.batch()), axis=-1)      # [E, 2] Point-
    shifts = lattice_shifts(periods)                                          # [9] Point-
    bond_strokes = (bond_ends[None] + shifts[:, None, None]).reshape(-1)      # [18 * E] Point-
    tiled_sites = (vertices.batch()[None] + shifts[:, None]).reshape(-1)      # [9 * V] Point-
    wrap_strokes = stack((-periods, periods), axis=-1).reshape(-1)            # [4] Point-
    n_bond_chains, n_wrap_chains = bond_strokes.shape[0] // 2, wrap_strokes.shape[0] // 2
    return image_figure(white_composite(composited([
        simplices.strokes(bond_strokes, Topology.chains([2] * n_bond_chains), bond_colour, bond_width, view),
        simplices.markers(tiled_sites, bond_colour, site_radius, view),
        simplices.strokes(wrap_strokes, Topology.chains([2] * n_wrap_chains), wrap_colour, wrap_width, view, dashes=wrap_dashes),
    ])))


def spin_frames(
    topology: Topology,
    vertices: Extensor,
    bonds: Extensor,
    periods: Extensor,
    states: Iterable[Extensor],
    n_view_pixels: int = n_pixels,
) -> Iterator[np.ndarray]:
    """Yields RGB frames of the tiled spin texture as each state (`Spin-[V]`) arrives from `states`."""
    places, tile_topology = tiled_triangles(topology, vertices, bonds, periods)
    view = film_camera(periods, np.zeros(2), period_bounds(periods), n_view_pixels)
    for spins in states:
        yield white_composite(composited(spin_layers(topology, spins, places, tile_topology, view)))


def comparison_frames(
    topology: Topology,
    vertices: Extensor,
    bonds: Extensor,
    periods: Extensor,
    state_tuples: Iterable[tuple[Extensor, ...]],
) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames separated by white gaps for each tuple of spin states (`Spin-[V]`) in `state_tuples`."""
    places, tile_topology = tiled_triangles(topology, vertices, bonds, periods)
    view = film_camera(periods, np.zeros(2), period_bounds(periods), n_panel_pixels)
    for case_spins in state_tuples:
        panels = [white_composite(composited(spin_layers(topology, spins, places, tile_topology, view))) for spins in case_spins]
        spacer = np.full((panels[0].shape[0], panel_gap, 3), 255, dtype=np.uint8)
        yield np.concatenate([part for panel in panels for part in (spacer, panel)][1:], axis=1)


@charts.style()
def transient_figure(times: np.ndarray, charges: Extensor, energies: Extensor, changes: Extensor) -> Figure:
    """Three stacked panels plot total topological charge (`[n_frames] Scalar+`), excess energy per site, and frame-to-frame spin change (`[n_frames - 1] Scalar+`) against `times`."""
    charge_values = np.asarray(charges.to_array())
    energy_values = np.asarray(energies.to_array())
    change_values = np.asarray(changes.to_array())
    figure, (charge_axis, energy_axis, change_axis) = charts.subplots(480, 480, nrows=3, sharex=True, layout="constrained")
    charge_axis.plot(times, charge_values, color="#2e5c8a")
    charge_axis.axhline(0.0, color="0.3", linewidth=charts.FRAME_WIDTH)
    charge_axis.set(ylabel="charge", ylim=(charge_values[len(charge_values) // 10:].min() - 2, 2))
    energy_axis.semilogy(times[:-1], energy_values[:-1] - energy_values[-1], color="#b03a2e")
    energy_axis.set(ylabel="energy above final")
    change_axis.semilogy(times[1:], change_values, color="#3d7f4a")
    change_axis.set(xlabel="time", ylabel="change per frame", xlim=(0, times[-1]))
    for axis in (charge_axis, energy_axis, change_axis):
        axis.spines[["top", "right"]].set_visible(False)
    return figure
