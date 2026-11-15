"""The E8 roots appear as the mass rings of affine Toda theory, and its three-mode couplings appear as root triangles."""

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from numga.render import simplices
from numga.topology import Topology

from examples.math.e8.render import background, draw_roots, orbit_colours
from examples.quantum.affine_toda.core import Root
from examples.util.panels import offset_panel, overlay

# The samples around each orbit's ring.
n_ring_samples = 120
# In pixels: ring width and orbit dot radius, arrow width and tip chevron length, closure gap width and dashes, and start dot radius.
ring_width, ring_opacity, orbit_dot, orbit_opacity = 1.0, 0.22, 1.4, 0.4
arrow_width, chevron_length = 2.5, 10.0
gap_width, gap_dashes = 1.8, (5.0, 5.0)
start_dot = 2.4
white = np.ones(3)


def triangles_figure(triangles: Root, roots: Root, species: np.ndarray,
                     horizontal: Root, vertical: Root) -> plt.Figure:
    """Each species triple appears as three roots joined head to tail in the Coxeter plane, with any closure gap dashed."""
    paths = concatenate((0 * triangles[:, :1], triangles.cumsum(axis=1)), axis=1)  # [..., 4] Root+
    # Each orbit's radius in the Coxeter plane is the norm of a root's projection onto it.
    coxeter_plane = horizontal ^ vertical                                       # [] Bivector+
    radii = (roots[0] | coxeter_plane).scalar_norm_squared().square_root()      # [8] Scalar+
    # Turn the horizontal vector around a closed loop in the Coxeter plane to trace a unit ring.
    turns = (coxeter_plane * (-np.linspace(0, 2 * np.pi, n_ring_samples, endpoint=False) / 2)).exp()  # [n_ring_samples] Rotor+
    ring = turns >> horizontal                                                  # [n_ring_samples] Root+
    # Frame the view comfortably past the farthest path corner or participating orbit.
    drawing_plane = stack((horizontal, vertical))                               # [2] Root+
    reach = max((drawing_plane | paths[..., None]).abs().max().to_array(),
                (drawing_plane | roots[:, species, None]).abs().max().to_array()) * 1.16
    n_cases, n_corners = paths.shape
    n_steps = roots.shape[0]
    figure, axes = plt.subplots(1, n_cases, figsize=(9, 3), facecolor=background, squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.04)
    for index, ax in enumerate(axes):
        # Read roots directly by pairing with the hyperplanes complementary to the drawing plane's two vectors.
        camera = offset_panel(ax, horizontal.dual(), vertical.dual())
        particles = np.unique(species[index])
        n_particles = len(particles)
        rings = (radii[particles, None] * ring[None, :]).reshape(-1)            # [n_particles * n_ring_samples] Root+
        orbits = roots[:, particles].reshape(-1)                                # [n_steps * n_particles] Root+
        corners = paths[index]                                                  # [n_corners] Root+
        overlay(ax, [
            simplices.strokes(rings, Topology.loops([n_ring_samples] * n_particles),
                              np.repeat(orbit_colours[particles], n_ring_samples, axis=0), ring_width, camera,
                              opacity=ring_opacity),
            simplices.markers(orbits, np.tile(orbit_colours[particles], (n_steps, 1)), orbit_dot, camera,
                              opacity=orbit_opacity),
            # Draw each root as an arrow from the corner before it, with a chevron at its tip.
            simplices.strokes(corners, Topology.chains([n_corners]), orbit_colours[species[index]],
                              arrow_width, camera, chevrons=np.ones(n_corners - 1), chevron=chevron_length,
                              cap=simplices.Cap.BUTT),
            # A closed path leaves a zero-length dashed segment.
            simplices.strokes(corners[[0, -1]], Topology.chains([2]), white, gap_width, camera,
                              dashes=gap_dashes, cap=simplices.Cap.BUTT),
            simplices.markers(corners[:1], white, start_dot, camera),
        ])
    plt.close(figure)
    return figure
