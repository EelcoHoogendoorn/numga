"""The E8 roots are drawn as the mass rings of affine Toda theory, and its three-mode couplings as root triangles."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from examples.math.e8.render import BACKGROUND, COLOURS, coordinates, draw_roots
from examples.quantum.affine_toda.core import Root


def draw_triangles(paths: Root, roots: Root, species: np.ndarray,
                   horizontal: Root, vertical: Root) -> plt.Figure:
    """Each case is drawn as its three roots joined head to tail, with any closure gap dashed."""
    vertices = coordinates(paths, horizontal, vertical)
    orbits = coordinates(roots, horizontal, vertical)
    selected_orbits = orbits[:, species]
    limit = max(np.abs(vertices).max(), np.abs(selected_orbits).max()) * 1.16
    n_cases = len(species)
    figure, axes = plt.subplots(1, n_cases, figsize=(9, 3), facecolor=BACKGROUND)
    for index, ax in enumerate(axes):
        for particle in np.unique(species[index]):
            points = orbits[:, particle]
            radius = np.linalg.norm(points[0])
            colour = COLOURS[particle]
            ax.add_patch(Circle((0, 0), radius, fill=False, color=colour, alpha=0.22, linewidth=0.55))
            ax.scatter(*points.T, color=colour, s=4, alpha=0.4, linewidths=0)
        for start, end, particle in zip(vertices[index, :-1], vertices[index, 1:], species[index]):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                        linewidth=2, color=COLOURS[particle], shrinkA=0, shrinkB=0))
        # A closed path has a zero-length dashed segment.
        ax.plot(*vertices[index, [0, -1]].T, color="white", linestyle=(0, (3, 3)), linewidth=1.3)
        ax.scatter(*vertices[index, 0], color="white", s=12, zorder=4)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.04)
    plt.close(figure)
    return figure
