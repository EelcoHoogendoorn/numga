"""The E8 roots are drawn as the mass rings of affine Toda theory, and its three-mode couplings as root triangles."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.math.e8.render import background, orbit_colours, draw_roots
from examples.quantum.affine_toda.core import Root
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import panel

# The samples around each orbit's ring.
n_ring_samples = 120
# In pixels: the rings' width and the orbits' dots' radius, both faint; the arrows' width and their heads' length; the
# closure gap's width and its dashes; the start's dot's radius.
ring_width, ring_opacity, orbit_dot, orbit_opacity = 1.0, 0.22, 1.4, 0.4
arrow_width, arrow_head = 2.5, 10.0
gap_width, gap_dashes = 1.8, (5.0, 5.0)
start_dot = 2.4
white = np.ones(3)


def draw_triangles(paths: Root, roots: Root, species: np.ndarray,
                   horizontal: Root, vertical: Root) -> plt.Figure:
    """Each case is drawn as its three roots joined head to tail, with any closure gap dashed."""
    # Each orbit's radius in the Coxeter plane: the size of a root's part in it.
    coxeter_plane = horizontal ^ vertical                                       # [] Bivector+
    radii = (roots[0] | coxeter_plane).scalar_norm_squared().square_root()     # [species] Scalar+
    # A unit ring in the Coxeter plane, turned from the horizontal around a closed loop.
    turns = (coxeter_plane * (-np.linspace(0, 2 * np.pi, n_ring_samples + 1) / 2)).exp()   # [samples + 1] Rotor+
    ring = turns >> horizontal                                                 # [samples + 1] Root+
    # The view reaches past the farthest vertex or orbit across or up.
    drawing_plane = stack((horizontal, vertical))                               # [2] Root+
    reach = max((drawing_plane | paths[..., None]).abs().max().to_array(),
                (drawing_plane | roots[:, species, None]).abs().max().to_array()) * 1.16
    n_cases, n_corners = paths.shape
    figure, axes = plt.subplots(1, n_cases, figsize=(9, 3), facecolor=background)
    for index, ax in enumerate(axes):
        ax.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.04)
    for index, ax in enumerate(axes):
        # Roots are read as they are, by the hyperplanes that complement the drawing plane's two vectors:
        # `root & horizontal.dual()` is `horizontal | root`.
        camera = panel(ax, horizontal.dual(), vertical.dual())
        particles = np.unique(species[index])
        rings = (radii[particles, None] * ring[None, :]).reshape(-1)            # [particles * (samples + 1)] Root+
        orbits = roots[:, particles].reshape(-1)                                # [steps * particles] Root+
        n_steps = roots.shape[0]
        corners = paths[index]                                                  # [corners] Root+
        simplices.overlay(ax, [
            simplices.strokes(rings, simplices.Mesh.chains([n_ring_samples + 1] * len(particles)),
                              np.repeat(orbit_colours[particles], n_ring_samples, axis=0), ring_width, camera,
                              opacity=ring_opacity),
            simplices.markers(orbits, np.tile(orbit_colours[particles], (n_steps, 1)), orbit_dot, camera,
                              opacity=orbit_opacity),
            simplices.strokes(corners, simplices.Mesh.chains([n_corners]), orbit_colours[species[index]],
                              arrow_width, camera, heads=np.ones(n_corners - 1), head=arrow_head,
                              cap=simplices.Cap.BUTT),
            # A closed path has a zero-length dashed segment.
            simplices.strokes(corners[[0, -1]], simplices.Mesh.chains([2]), white, gap_width, camera,
                              dashes=gap_dashes, cap=simplices.Cap.BUTT),
            simplices.markers(corners[:1], white, start_dot, camera),
        ])
    plt.close(figure)
    return figure
