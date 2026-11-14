"""GPU rendering of stereographically projected isoclinic circle families and torus-knot orbits."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from examples.math.spin_groups import core
from examples.util.panels import fitted, show
from numga.render import Scene, simplices, solids
from numga.render.drawing import Draw
from numga.topology import Topology

background = np.ones(3)
# Shared camera elevation and azimuth in radians, framing margin, and per-panel pixel width.
elevation, azimuth = np.radians(24.0), np.radians(35.0)
margin = 0.14
frame_pixels = 300
# Stroke widths in pixels for the static figure and faint animation guides, plus rider ball radius and count.
circle_width, knot_width, faint_width, faint_opacity = 1.1, 1.8, 1.1, 0.45
rider_radius = 2.2
n_riders = 6


def colours(shape: tuple[int, int]) -> np.ndarray:
    """Each orbit in `[n_heights, n_azimuths]` takes its hue from the starting azimuth and its value
    from the starting height (`[n_heights, n_azimuths, 3]`)."""
    hue = np.broadcast_to(np.linspace(0.0, 1.0, shape[1], endpoint=False), shape)
    value = np.broadcast_to(np.linspace(0.95, 0.55, shape[0])[:, None], shape)
    return hsv_to_rgb(np.stack([hue, np.full(shape, 0.75), value], axis=-1))


def view(pixels: int, families: list[solids.Point]) -> solids.View:
    """One camera `pixels` wide frames all three orbit families (`families`) at once."""
    return solids.view(families, 0.0, elevation, azimuth, margin, pixels)


def orbit_samples(orbits: solids.Point, width: float, camera: solids.View) -> Draw:
    """Each sampled orbit in `orbits` (`[n_heights, n_azimuths, n_steps + 1] Point-`) renders as a
    tube chain of pixel diameter `width` coloured by its starting point."""
    n_orbits, n_points = int(np.prod(orbits.shape[:-1])), orbits.shape[-1]
    chains = Topology.chains([n_points] * n_orbits)
    V, E = chains.n_elements
    shades = np.repeat(colours(orbits.shape[:2]).reshape(-1, 3), n_points - 1, axis=0)   # [E, 3]
    return simplices.tubes(orbits.reshape(-1), chains, shades, width, camera, gloss=solids.gloss)


def dots(positions: solids.Point, rgb: np.ndarray, camera: solids.View) -> tuple[solids.Quadric, np.ndarray]:
    """Each point in `positions` (`[n_dots] Point-`) carries an opaque sphere of colour `rgb`
    (`[n_dots, 3]`)."""
    return solids.ball(positions, rider_radius * camera.pixel)[:, None], rgb


def projectively_embedded(families: tuple[core.Euclidean, ...]) -> list[solids.Point]:
    """Stereographically projected orbits sit orthogonal to `w`; slicing out `w` places them around
    the three-dimensional scene origin (`Point-`)."""
    w = core.mv.w
    return [
        solids.origin + solids.context.include_from_algebra(w ^ (orbits & w.dual()))
        for orbits in families
    ]


def draw_flows(left_orbits: core.Euclidean, right_orbits: core.Euclidean,
               knotted_orbits: core.Euclidean) -> plt.Figure:
    """Draws the two opposite isoclinic circle families alongside the `(2, 3)` trefoil-knot orbits across
    three side-by-side panels."""
    families = projectively_embedded((left_orbits, right_orbits, knotted_orbits))
    figure = plt.figure(figsize=(9, 3.1))
    axes = [figure.add_axes((index / len(families), 0.0, 1 / len(families), 1.0)) for index in range(len(families))]
    panel = fitted(view(1, families), axes[0])
    for ax, orbits, width in zip(axes, families, (circle_width, circle_width, knot_width)):
        show(ax, solids.draw([], panel, background, inks=[], translucent=[],
                             meshes=[(orbit_samples(orbits, width, panel), 1.0)]), panel)
    return figure


def animate_flows(left_orbits: core.Euclidean, right_orbits: core.Euclidean,
                  knotted_orbits: core.Euclidean, frame_steps: Iterable[int] | Iterator[int]) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames carrying points along both isoclinic flows and evenly spaced
    riders along each trefoil knot at each step index in `frame_steps`."""
    families = projectively_embedded((left_orbits, right_orbits, knotted_orbits))
    camera = view(frame_pixels, families)
    panels = [
        Scene(camera, background, [], [(orbit_samples(orbits, faint_width, camera), faint_opacity)],
              inks=[], translucent=[])
        for orbits in families
    ]
    n_steps = families[0].shape[2] - 1
    rider_offsets = np.arange(n_riders) * n_steps // n_riders
    knots = families[2]
    for step in frame_steps:
        riding = [
            (orbits[:, :, step].reshape(-1), colours(orbits.shape[:2]).reshape(-1, 3))
            for orbits in families[:2]
        ]
        riding.append((
            knots[:, :, (step + rider_offsets) % n_steps].reshape(-1),
            np.repeat(colours(knots.shape[:2]).reshape(-1, 3), len(rider_offsets), axis=0),
        ))
        yield np.concatenate([
            panel.draw([dots(*points, camera)], inks=[], translucent=[], meshes=[])
            for panel, points in zip(panels, riding)
        ], axis=1)
