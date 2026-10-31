"""Coxeter projections of the E8 roots and their simultaneous planar rotations."""

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga.extensor import stack

from examples.util.animation import capture
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import panel
from examples.math.e8.core import Vector

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
# In pixels: the widths of the roots' edges, of the orbits and of the trails, and the radii of the roots and of the
# riders on their orbits; and how much the edges, the orbits and the roots behind the riders cover.
edge_width, orbit_width, trail_width = 0.5, 0.4, 1.1
root_radius, faint_root_radius, rider_radius = 2.4, 0.6, 1.5
edge_opacity, orbit_opacity, faint_root_opacity = 0.3, 0.26, 0.35
# The points along each orbit's circle.
n_around = 120


def draw_roots(roots: Vector, horizontal: Vector, vertical: Vector) -> plt.Figure:
    """The roots in the Coxeter plane, coloured by orbit, with nearest neighbours joined."""
    flat = roots.reshape((-1,))
    distances = (flat[:, None] - flat[None, :]).scalar_norm_squared()
    distinct = distances[distances > 1e-8]
    nearest = distinct.min()
    first, second = np.nonzero(np.triu((distances - nearest).abs() <= (1e-8 + 1e-5 * nearest.abs()), k=1))
    # A root's part in the drawing plane is as long as its contraction onto the plane's unit bivector.
    reach = (roots | (horizontal ^ vertical)).norm().reshape(-1)             # [steps * orbits] Scalar+
    colours = np.tile(orbit_colours, (roots.shape[0], 1))
    edge_colours = (colours[first] + colours[second]) / 2

    figure, ax = plt.subplots(figsize=(7, 7), facecolor=background)
    radius = float((reach.max() * 1.05).to_array())
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect="equal")
    ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1)
    # The camera reads roots, vectors, as they are, along the drawing plane's two directions: `root & horizontal.dual()`
    # is `horizontal | root`.
    camera = panel(ax, horizontal.dual(), vertical.dual())
    neighbours = simplices.Mesh(len(flat), np.stack([first, second], axis=-1))
    simplices.overlay(ax, [
        simplices.strokes(flat, neighbours, edge_colours, edge_width, camera, opacity=edge_opacity),
        simplices.markers(flat, colours, root_radius, camera)])
    plt.close(figure)
    return figure


def animate_planes(turning_roots: Iterable[Vector], roots: Vector, horizontal: Vector,
                   vertical: Vector) -> Iterator[np.ndarray]:
    """Eight coloured roots moving in each invariant plane, with trails of equal duration."""
    turning = stack(list(turning_roots), axis=-1)                              # [orbits, frames] Vector+
    n_orbits, n_frames = turning.shape
    n_planes = len(horizontal)
    # The radius of each root's orbit in each plane: the length of its contraction onto the plane's unit bivector.
    planes = horizontal ^ vertical                                            # [planes] Bivector+
    orbit_radii = (roots[:, None, :] | planes[None, :, None]).norm().to_array()   # [steps, planes, orbits]
    radius = orbit_radii.max() * 1.08
    # Each trail runs through the riders' places over the share of the loop it covers, one Coxeter step of thirty,
    # fading from the background into the rider's colour.
    n_trail = round(n_frames / 30)
    history = np.arange(-n_trail, 1)                                           # [trail + 1]
    fade = np.linspace(0.12, 0.85, n_trail)[:, None]                           # [trail, 1]
    background_colour = np.array(to_rgb(background))
    trail_colours = (background_colour + fade * (orbit_colours[:, None] - background_colour)).reshape(-1, 3)
    trail_chains = simplices.Mesh.chains([n_trail + 1] * n_orbits)
    # A full turn of each plane in small steps, which carries every root's part in it round its circle.
    orbit_chains = simplices.Mesh.chains([n_around + 1] * n_orbits)
    seeds = turning[:, 0]                                                      # [orbits] Vector+

    figure, axes = plt.subplots(1, n_planes, figsize=(4.4, 1.14), facecolor=background)
    for ax in axes:
        ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.025)
    # Each plane's camera reads roots as they are, along its two directions.
    across, up = horizontal.dual(), vertical.dual()                             # [planes] Antivector-
    cameras = [panel(ax, across[index], up[index]) for index, ax in enumerate(axes)]
    backdrops = []
    for index, camera in enumerate(cameras):
        step = (planes[index] * (-np.pi / n_around)).exp()                     # [] Rotor+
        turned = [seeds]
        for _ in range(n_around):
            turned.append(step >> turned[-1])
        circles = stack(turned, axis=-1)                                       # [orbits, around + 1] Vector+
        backdrops.append([
            simplices.strokes(circles.reshape(-1), orbit_chains, np.repeat(orbit_colours, n_around, axis=0),
                              orbit_width, camera, opacity=orbit_opacity),
            simplices.markers(roots.reshape(-1), np.tile(orbit_colours, (roots.shape[0], 1)),
                              faint_root_radius, camera, opacity=faint_root_opacity)])
    images = [simplices.overlay(ax, backdrop) for ax, backdrop in zip(axes, backdrops)]

    for frame in range(n_frames):
        trails = turning[:, (frame + history) % n_frames]                     # [orbits, trail + 1] Vector+
        for image, camera, backdrop in zip(images, cameras, backdrops):
            image.set_data(simplices.composited(backdrop + [
                simplices.strokes(trails.reshape(-1), trail_chains, trail_colours, trail_width, camera),
                simplices.markers(turning[:, frame], orbit_colours, rider_radius, camera)]))
        yield capture(figure)
    plt.close(figure)
