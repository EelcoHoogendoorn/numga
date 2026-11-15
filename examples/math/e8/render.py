"""Renders Coxeter-plane projections of the E8 root system and animations of its four simultaneous plane rotations."""

from collections import deque
from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga.extensor import stack
from numga.render import drawing, simplices
from numga.topology import Topology

from examples.math.e8.core import Vector
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay

background = "#101219"
orbit_colours = np.array([
    [0.98, 0.72, 0.27],
    [0.96, 0.39, 0.32],
    [0.83, 0.40, 0.72],
    [0.60, 0.48, 0.92],
    [0.32, 0.63, 0.95],
    [0.24, 0.80, 0.82],
    [0.35, 0.82, 0.59],
    [0.76, 0.84, 0.40],
])
# Stroke widths and marker radii in pixels, together with layer opacities.
edge_width, orbit_width, trail_width = 0.5, 0.4, 1.1
root_radius, faint_root_radius, rider_radius = 2.4, 0.6, 1.5
edge_opacity, orbit_opacity, faint_root_opacity = 0.17, 0.26, 0.35
n_circle_points = 120


def draw_roots(roots: Vector, horizontal: Vector, vertical: Vector) -> plt.Figure:
    """Projects `roots` (`[n_steps, n_orbits] Vector+`) onto the plane of `horizontal` and `vertical` with nearest-neighbour edges."""
    n_steps, n_orbits = roots.shape
    n_roots = n_steps * n_orbits
    flat_roots = roots.reshape((-1,))                                         # [n_roots] Vector+
    distances = (flat_roots[:, None] - flat_roots[None, :]).scalar_norm_squared()  # [n_roots, n_roots] Scalar+
    nonzero_distances = distances[distances > 1e-8]                           # [...] Scalar+
    min_distance = nonzero_distances.min()                                    # [] Scalar+
    first_indices, second_indices = np.nonzero(
        np.triu((distances - min_distance).abs() <= (1e-8 + 1e-5 * min_distance.abs()), k=1)
    )
    # Contraction onto the unit plane bivector gives the projected radial reach of each root.
    reach = (roots | (horizontal ^ vertical)).norm().reshape(-1)              # [n_roots] Scalar+
    colours = np.tile(orbit_colours[:n_orbits], (n_steps, 1))
    edge_colours = (colours[first_indices] + colours[second_indices]) / 2

    figure, ax = plt.subplots(figsize=(7, 7), facecolor=background)
    radius = float((reach.max() * 1.05).to_array())
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect="equal")
    ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1)
    # Pairing each root vector against the duals of the two frame vectors reads its orthogonal plane coordinates.
    camera = offset_panel(ax, horizontal.dual(), vertical.dual())
    neighbours = Topology(n_roots, np.stack([first_indices, second_indices], axis=-1))
    overlay(ax, [
        simplices.strokes(flat_roots, neighbours, edge_colours, edge_width, camera, opacity=edge_opacity),
        simplices.markers(flat_roots, colours, root_radius, camera),
    ])
    plt.close(figure)
    return figure


def planes_animation(
    roots: Vector,
    horizontal: Vector,
    vertical: Vector,
    n_trail: int,
    turning_roots: Iterable[Vector],
) -> Iterator[np.ndarray]:
    """Yields frames showing the orbit riders turning simultaneously across all four invariant planes with `n_trail`-step trails."""
    n_steps, n_orbits = roots.shape
    n_planes = len(horizontal)
    planes = horizontal ^ vertical                                            # [n_planes] Bivector+
    orbit_radii = (roots[:, None, :] | planes[None, :, None]).norm()          # [n_steps, n_planes, n_orbits] Scalar+
    radius = float((orbit_radii.max() * 1.08).to_array())

    fade = np.linspace(0.12, 0.85, n_trail)[:, None]                          # [n_trail, 1]
    background_colour = np.array(to_rgb(background))
    palette = orbit_colours[:n_orbits]
    trail_colours = (background_colour + fade * (palette[:, None] - background_colour)).reshape(-1, 3)
    trail_chains = Topology.chains([n_trail + 1] * n_orbits)
    orbit_loops = Topology.loops([n_circle_points] * n_orbits)
    seeds = roots[0]                                                          # [n_orbits] Vector+
    half_angles = np.linspace(0.0, -np.pi, n_circle_points, endpoint=False)   # [n_circle_points]

    figure, axes = plt.subplots(1, n_planes, figsize=(4.4, 1.14), facecolor=background)
    for ax in axes:
        ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.025)

    horizontal_directions = horizontal.dual()                                 # [n_planes] Antivector-
    vertical_directions = vertical.dual()                                     # [n_planes] Antivector-
    cameras = [
        offset_panel(ax, horizontal_directions[index], vertical_directions[index])
        for index, ax in enumerate(axes)
    ]
    backdrops = []
    for index, camera in enumerate(cameras):
        circle_rotors = (planes[index] * half_angles).exp()                   # [n_circle_points] Rotor+
        circles = circle_rotors[None, :] >> seeds[:, None]                    # [n_orbits, n_circle_points] Vector+
        backdrops.append([
            simplices.strokes(
                circles.reshape(-1),
                orbit_loops,
                np.repeat(palette, n_circle_points, axis=0),
                orbit_width,
                camera,
                opacity=orbit_opacity,
            ),
            simplices.markers(
                roots.reshape(-1),
                np.tile(palette, (n_steps, 1)),
                faint_root_radius,
                camera,
                opacity=faint_root_opacity,
            ),
        ])
    images = [overlay(ax, backdrop) for ax, backdrop in zip(axes, backdrops)]

    stream = iter(turning_roots)
    first_roots = next(stream, None)
    try:
        if first_roots is None:
            return
        trail_window = deque([first_roots] * n_trail, maxlen=n_trail + 1)
        for frame_roots in chain((first_roots,), stream):
            trail_window.append(frame_roots)
            trails = stack(tuple(trail_window), axis=-1)                      # [n_orbits, n_trail + 1] Vector+
            for image, camera, backdrop in zip(images, cameras, backdrops):
                image.set_data(drawing.composited(backdrop + [
                    simplices.strokes(trails.reshape(-1), trail_chains, trail_colours, trail_width, camera),
                    simplices.markers(frame_roots, palette, rider_radius, camera),
                ]))
            yield capture(figure)
    finally:
        plt.close(figure)
