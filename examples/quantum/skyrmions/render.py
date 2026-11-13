"""Drawing for magnetic skyrmions: the spin texture over the lattice, each spin's part along the field as colour, grey
where the spin lies in the film, warm where it points with the field and cool against it, with its part in the film
drawn over the colour as streaks along it, alone or several side by side; a close-up of the lattice's corner; and a
preview chart of a run's transient.

The lattice repeats over the rhombus its periods span, and the picture is the rectangle as wide and as high as the
periods reach: the triangles are drawn apart, each at its corners' true places from one corner along its sides, and
again a period on and a period back along each period, so that the copies tile the rectangle and the triangles the
wrap cuts are whole. A site is read by the planes x and y, a
spin's part along the field by the film's plane, and its part in the film by the planes yz and zx, and set out as a step
in the film for the streaks.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga import stack
from numga.extensor import Extensor
from examples.util import charts
from examples.util.panels import image_figure
from numga.render import contours, flow, simplices
from numga.render.camera import Camera
from numga.render.drawing import composited
from numga.topology import Topology

# The colours of a spin's part along the field, from against it through lying in the film to along it, in bands of
# flat colour, which keep the animations' files small.
along_field_map = colormaps["coolwarm"]
n_bands = 13
# The colours an animation's palette holds.
n_animation_colours = 32
# The picture's pixels across, and a panel's of several side by side, with the white gap between them; the streaks' ink,
# the part in the film at which a streak reaches its full length, and that length in pixels.
n_pixels = 320
n_panel_pixels, gap = 260, 8
streak_ink, full_streak, streak_length = np.array(to_rgb("#26262e")), 1.0, 14.0
# The close-up of the lattice's corner: how far it reaches from the corner, in lattice spacings, the bonds' colour and
# width and the sites' radius in pixels, and the wrap's colour, width and dashes in pixels.
corner_reach = 7.0
bond_colour, bond_width, site_radius = np.array(to_rgb("#5b6470")), 1.5, 3.0
wrap_colour, wrap_width, wrap_dashes = np.array(to_rgb("#c0392b")), 1.5, (6.0, 4.0)


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def repeats(periods: Extensor) -> Extensor:
    """The steps to the lattice's copies about it, a period on, none and a period back along each period
    `[copies] Point-`."""
    shifts = np.array([-1, 0, 1])
    return (periods[0] * shifts[:, None] + periods[1] * shifts[None, :]).reshape(-1)


def tiled(topology: Topology, vertices: Extensor, bonds: Extensor, periods: Extensor) -> tuple[Extensor, Topology]:
    """The lattice's triangles drawn apart, each at its corners' places from its first corner along its sides, in
    every copy about the lattice: their corners `[copies faces corners] Point-`, and their topology, each triangle on
    its own three corners."""
    sides = topology.boundary[2, 1] * bonds                                   # [corners] Point-[faces]
    first = (topology.incidence[2, 0][0] * vertices).batch()                  # [faces] Point-
    # The second corner one side on from the first, and the third one side back.
    corners = stack((first, first + sides[2].batch(), first - sides[1].batch()), axis=-1)   # [faces, corners] Point-
    places = (corners[None] + repeats(periods)[:, None, None]).reshape(-1)    # [copies faces corners] Point-
    n_places = places.shape[0]
    return places, Topology(n_places, np.arange(n_places).reshape(-1, 3))


def camera(periods: Extensor, low: np.ndarray, high: np.ndarray, pixels: int) -> Camera:
    """The camera on the film from `low` to `high` across and up `[2]`, read by the planes x and y, each place a step of
    weight one from the origin, `pixels` across."""
    mv = periods.context.multivector
    return Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, (low + high) / 2, (high - low) / 2, pixels)


def rectangle(periods: Extensor) -> np.ndarray:
    """The far corner across and up `[2]` of the rectangle the periods reach from the origin, as wide as the furthest
    reaches across and as high as the furthest up."""
    mv = periods.context.multivector
    return np.abs((stack((mv.x, mv.y))[None] & periods[:, None]).to_array()).max(axis=0)


def layers(topology: Topology, spins: Extensor, places: Extensor, soup: Topology, view: Camera) -> list:
    """The spins `Spin-[vertices]` over the tiled triangles at these places `[copies faces corners] Point-`: their part
    along the field as colour, and their part in the film as streaks."""
    mv = spins.context.multivector
    corners = topology.incidence[2, 0] * spins                                # [corners] Spin-[faces]
    per_face = stack([corners[index].batch() for index in range(3)], axis=-1) # [faces, corners] Spin-
    n_copies = places.shape[0] // per_face.shape[0] // 3
    tile_spins = per_face[None].broadcast_to((n_copies,) + per_face.shape).reshape(-1)   # [copies faces corners] Spin-
    along_field = np.asarray((mv.xy & tile_spins).to_array())
    # A spin's part in the film, read by the planes yz and zx, set out as a step across and up the film.
    readings = stack((mv.yz, mv.zx)) & tile_spins[:, None]                    # [places, 2] Scalar+
    steps = (stack((mv.x.dual(), mv.y.dual())) * readings).sum(axis=-1)       # [places] Point-
    colours = along_field_map((along_field + 1) / 2)[:, :3]
    middles = (np.arange(n_bands) + 0.5) / n_bands                            # [bands]
    bands = contours.Bands(-1.0, 2 / n_bands, along_field_map(middles))
    return [contours.picture(places, soup, along_field, colours, view, bands=(bands,)),
            flow.layer(places, soup, steps, view, full_streak, streak_ink, length=streak_length)]


def draw_lattice(topology: Topology, vertices: Extensor, bonds: Extensor, periods: Extensor) -> Figure:
    """The lattice's corner close up: its bonds, each from its tail along its step, and its sites, in every copy about
    the lattice, and its periods through the corner, where the lattice wraps, dashed."""
    view = camera(periods, -np.full(2, corner_reach) / 4, np.full(2, corner_reach), n_pixels)
    tails = (topology.incidence[1, 0][0] * vertices).batch()                  # [edges] Point-
    ends = stack((tails, tails + bonds.batch()), axis=-1)                     # [edges, ends] Point-
    shifts = repeats(periods)                                                 # [copies] Point-
    strokes = (ends[None] + shifts[:, None, None]).reshape(-1)                # [copies edges ends] Point-
    sites = (vertices.batch()[None] + shifts[:, None]).reshape(-1)            # [copies vertices] Point-
    # The lattice's periods through the corner, from a period back to a period on along each.
    wraps = stack((-periods, periods), axis=-1).reshape(-1)                   # [sides ends] Point-
    n_strokes, n_wraps = strokes.shape[0] // 2, wraps.shape[0] // 2
    return image_figure(on_white(composited([
        simplices.strokes(strokes, Topology.chains([2] * n_strokes), bond_colour, bond_width, view),
        simplices.markers(sites, bond_colour, site_radius, view),
        simplices.strokes(wraps, Topology.chains([2] * n_wraps), wrap_colour, wrap_width, view, dashes=wrap_dashes)])))


def animate(topology: Topology, vertices: Extensor, bonds: Extensor, periods: Extensor,
            states: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """The spin texture over the lattice at every state of the spins `Spin-[vertices]` as it comes."""
    places, soup = tiled(topology, vertices, bonds, periods)
    view = camera(periods, np.zeros(2), rectangle(periods), n_pixels)
    for spins in states:
        yield on_white(composited(layers(topology, spins, places, soup, view)))


def animate_together(topology: Topology, vertices: Extensor, bonds: Extensor, periods: Extensor,
                     states: Iterable[tuple[Extensor, ...]]) -> Iterator[np.ndarray]:
    """The spin textures of several cases side by side over the lattice, a white gap between them, at every tuple of
    their states `Spin-[vertices]` as it comes."""
    places, soup = tiled(topology, vertices, bonds, periods)
    view = camera(periods, np.zeros(2), rectangle(periods), n_panel_pixels)
    for cases in states:
        panels = [on_white(composited(layers(topology, spins, places, soup, view))) for spins in cases]
        spacer = np.full((panels[0].shape[0], gap, 3), 255, dtype=np.uint8)
        yield np.concatenate([part for panel in panels for part in (spacer, panel)][1:], axis=1)


@charts.style()
def draw_transient(times: np.ndarray, charges: np.ndarray, energies: np.ndarray, changes: np.ndarray) -> Figure:
    """The texture's charge, its energy per site above the last frame's, and the spins' change from each frame to the
    next `[frames - 1]`, the latter two on logarithmic scales, against time `[frames]`, one above the other."""
    figure, (charge_axis, energy_axis, change_axis) = charts.subplots(480, 480, nrows=3, sharex=True,
                                                                      layout="constrained")
    charge_axis.plot(times, charges, color="#2e5c8a")
    charge_axis.axhline(0.0, color="0.3", linewidth=charts.FRAME_WIDTH)
    charge_axis.set(ylabel="charge", ylim=(charges[len(charges) // 10:].min() - 2, 2))
    energy_axis.semilogy(times[:-1], energies[:-1] - energies[-1], color="#b03a2e")
    energy_axis.set(ylabel="energy above the last")
    change_axis.semilogy(times[1:], changes, color="#3d7f4a")
    change_axis.set(xlabel="time", ylabel="change a frame", xlim=(0, times[-1]))
    for axis in (charge_axis, energy_axis, change_axis):
        axis.spines[["top", "right"]].set_visible(False)
    return figure
