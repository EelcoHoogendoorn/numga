"""GPU tube rendering frames Zitterbewegung charge trajectories and spatial spin arrows in the
observer's rest slice."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from numga.render import simplices, solids
from numga.topology import Topology
from examples.relativity.dirac import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import fitted, show

# Spatial slice projection reading where an observer time line meets the rest slice.
space_slice = core.mv.t ^ (core.Antivector & core.time)

colours = ("#2e86c1", "#7d3c98", "#c0392b")
background = np.ones(3)
# Camera elevation and azimuth in radians (azimuth zero looking along +y) and framing margin.
elevation, azimuth = np.radians(12.0), np.radians(30.0)
margin = 0.1
# Display length of unit spin axes in Compton wavelengths, tube stroke widths in pixels, and chevron length.
spin_length = 0.4
path_width, arrow_width = 1.5, 1.5
chevron_length = 4.0
static_width_px, static_height_px = 260, 350
animation_width_px, animation_height_px = 240, 300


def spin_offset(psi: core.Spinor) -> core.Antivector:
    """Scales the observer-sliced spatial spin axis of `psi` (`[...] Spinor+`) to display length
    `spin_length` (`[...] Antivector-`)."""
    axis = core.spin_axis(psi)                                                  # [...] Antivector-
    return spin_length * axis / axis.antinorm()                                 # [...] Antivector-


def camera_framing(ax, places, tips) -> solids.View:
    """Fits one orthographic `solids.View` around `places` and `tips` inside the pixel budget of
    `ax`."""
    return fitted(solids.view([places.reshape(-1), tips.reshape(-1)], 0.0, elevation, azimuth, margin, 1), ax)


def scene_picture(places, tip_ends, view: solids.View) -> np.ndarray:
    """Renders `[n_mixtures, n_steps]` path tubes and `[n_mixtures]` terminal spin arrows into an
    `[rows, columns, 3]` RGB image."""
    n_mixtures, n_steps = places.shape
    rgb = np.array([to_rgb(colour) for colour in colours[:n_mixtures]])
    starts = places[:, -1]                                                      # [n_mixtures] Point-
    trails = places.reshape(-1)                                                 # [n_mixtures * n_steps] Point-
    lines = simplices.tubes(
        trails, Topology.chains([n_steps] * n_mixtures), np.repeat(rgb, n_steps - 1, axis=0),
        path_width, view, solids.gloss,
    )
    arrows = simplices.tubes(
        stack((starts, tip_ends), axis=-1).reshape(-1), Topology.chains([2] * n_mixtures), rgb,
        arrow_width, view, solids.gloss, chevrons=np.ones(n_mixtures),
        chevron=chevron_length * arrow_width,
    )
    return solids.draw([], view, background, inks=[], translucent=[], meshes=[(lines, 1.0), (arrows, 1.0)])


@charts.style()
def paths_figure(states: Iterable[tuple[core.Antivector, core.Spinor]]) -> plt.Figure:
    """Renders completed Zitterbewegung trajectories and terminal spin arrows from `states` framed
    across every streamed trail and spin-arrow tip."""
    collected = list(states)
    final_paths, final_spinors = collected[-1]
    frame_spinors = stack([psi for _, psi in collected], axis=-1)               # [..., ...] Spinor+
    places = solids.origin + solids.context.include_from_algebra(space_slice(final_paths))
    stride = max(1, final_paths.shape[-1] // len(collected))
    sampled_places = places[:, stride - 1 :: stride]
    all_tips = sampled_places + solids.context.include_from_algebra(space_slice(spin_offset(frame_spinors)))
    tip_ends = places[:, -1] + solids.context.include_from_algebra(space_slice(spin_offset(final_spinors)))
    figure = plt.figure(figsize=(static_width_px / charts.DPI, static_height_px / charts.DPI))
    ax = figure.add_axes((0, 0, 1, 1))
    view = camera_framing(ax, places, all_tips)
    show(ax, scene_picture(places, tip_ends, view), view)
    return figure


def paths_animation(states: Iterable[tuple[core.Antivector, core.Spinor]]) -> Iterator[np.ndarray]:
    """Yields `[rows, columns, 3]` RGB frames as `(trail, spinor)` `states` trace out inside a
    stationary camera framed across the full trajectory."""
    collected = list(states)
    final_paths = collected[-1][0]                                              # [..., ...] Antivector-
    frame_spinors = stack([psi for _, psi in collected], axis=-1)               # [..., ...] Spinor+
    all_places = solids.origin + solids.context.include_from_algebra(space_slice(final_paths))
    stride = max(1, final_paths.shape[-1] // len(collected))
    sampled_places = all_places[:, stride - 1 :: stride]
    all_tips = sampled_places + solids.context.include_from_algebra(space_slice(spin_offset(frame_spinors)))
    with charts.style():
        figure = plt.figure(figsize=(animation_width_px / charts.DPI, animation_height_px / charts.DPI))
        ax = figure.add_axes((0, 0, 1, 1))
        view = camera_framing(ax, all_places, all_tips)
        shown = None
        for trail, psi in collected:
            places = solids.origin + solids.context.include_from_algebra(space_slice(trail))
            tip_ends = places[:, -1] + solids.context.include_from_algebra(space_slice(spin_offset(psi)))
            image = scene_picture(places, tip_ends, view)
            if shown is None:
                shown = show(ax, image, view)
            else:
                shown.set_data(image)
            yield capture(figure)
        plt.close(figure)
