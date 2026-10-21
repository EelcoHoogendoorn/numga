"""Mass readouts and root triangles for affine E8 Toda theory."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from examples.math.e8.render import BACKGROUND, COLOURS, coordinates, draw_roots
from examples.quantum.affine_toda.core import Root, Scalar


def mass_table(masses: Scalar, radii: Scalar) -> str:
    """Mass ratios beside independently measured Coxeter-ring radii."""
    mass_ratios = (masses / masses[0]).to_array()
    radius_ratios = (radii / radii[0]).to_array()
    lines = ["species    mass / lightest    radius / smallest"]
    lines.extend(f"{index + 1:7d}{mass:18.6f}{radius:21.6f}"
                 for index, (mass, radius) in enumerate(zip(mass_ratios, radius_ratios)))
    return "\n".join(lines)


def measured_mass_table(predicted: Scalar, measurements: np.ndarray) -> str:
    """Calculated excitation energies beside measured peak positions, in meV."""
    predictions = predicted.to_array()
    lines = ["mode    E8 prediction (meV)    measured peak (meV)"]
    lines.extend(f"{index + 1:4d}{prediction:23.3f}{measurement:23.2f}"
                 for index, (prediction, measurement) in enumerate(zip(predictions, measurements)))
    return "\n".join(lines)


def coupling_table(species: np.ndarray, couplings: Scalar, closure: Scalar) -> str:
    """Three-mode coupling magnitudes and the shortest full-space root closure."""
    strengths = couplings.abs().to_array()
    gaps = closure.square_root().to_array()
    lines = ["species       |coupling|    root closure"]
    for triple, strength, gap in zip(species, strengths, gaps):
        label = ",".join(str(index + 1) for index in triple)
        lines.append(f"{label:12s}{strength:11.6f}{gap:16.6f}")
    return "\n".join(lines)


def area_table(couplings: Scalar, predicted: Scalar) -> str:
    """The cubic form evaluated on modes beside the coupling predicted by triangle area."""
    values = couplings.abs().to_array()
    areas = predicted.to_array()
    lines = ["from cubic form    from triangle area"]
    lines.extend(f"{value:15.6f}{area:22.6f}" for value, area in zip(values, areas))
    return "\n".join(lines)


def draw_triangles(paths: Root, roots: Root, species: np.ndarray,
                   horizontal: Root, vertical: Root) -> plt.Figure:
    """Three root vectors joined head to tail, with any closure gap drawn dashed."""
    vertices = coordinates(paths, horizontal, vertical)
    orbits = coordinates(roots, horizontal, vertical)
    selected_orbits = orbits[:, species]
    limit = max(np.abs(vertices).max(), np.abs(selected_orbits).max()) * 1.16
    n_cases = len(species)
    figure, axes = plt.subplots(1, n_cases, figsize=(10.5, 3.5), facecolor=BACKGROUND)
    for index, ax in enumerate(axes):
        for particle in np.unique(species[index]):
            points = orbits[:, particle]
            radius = np.linalg.norm(points[0])
            colour = COLOURS[particle]
            ax.add_patch(Circle((0, 0), radius, fill=False, color=colour, alpha=0.22, linewidth=0.65))
            ax.scatter(*points.T, color=colour, s=5, alpha=0.4, linewidths=0)
        for start, end, particle in zip(vertices[index, :-1], vertices[index, 1:], species[index]):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                        linewidth=2.3, color=COLOURS[particle], shrinkA=0, shrinkB=0))
        # A closed path has a zero-length dashed segment.
        ax.plot(*vertices[index, [0, -1]].T, color="white", linestyle=(0, (3, 3)), linewidth=1.5)
        ax.scatter(*vertices[index, 0], color="white", s=16, zorder=4)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1, wspace=0.04)
    plt.close(figure)
    return figure
