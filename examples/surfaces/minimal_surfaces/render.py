"""Lit surface sheets, domain parameter curves, and normal pins display minimal surfaces and their conjugates."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from numga.sparse import SparseExtensor, spdiag
from examples.surfaces.minimal_surfaces import core
from numga.topology import Topology
from numga.render.drawing import Draw
from numga.render import meshes, simplices, solids
from examples.util.panels import image_figure

mv = core.mv
background = np.ones(3)
surface_colour = np.array([0.90, 0.93, 0.96])
ink = np.array([0.16, 0.23, 0.30])
coral = np.array([0.91, 0.24, 0.16])
teal = np.array([0.02, 0.65, 0.55])
blue = np.array([0.12, 0.40, 0.88])
gold = np.array([0.99, 0.67, 0.10])
violet = np.array([0.57, 0.27, 0.84])
meridian_colours = np.stack((coral, gold, teal, blue, violet, [0.90, 0.20, 0.57]))
n_pixels = 300
n_curve_samples = 80
n_rings = 4
n_spokes = len(meridian_colours)
# Concentric ring radii as fractions of the planar disk radius.
ring_shares = np.linspace(0.22, 0.86, n_rings)
# Reach behind and ahead of each pinned vertex along its outward normal, plus pin-head radius.
pin_depth, pin_reach, pin_head_radius = 0.16, 0.30, 0.048
# Multiplying an intrinsic normal patch by the unit volume and taking the dual gives the outward normal direction.
unit_volume = mv.xyz.dual()                                                  # [] Antipseudoscalar-
plateau_view = np.array([28.0, -38.0])
conjugate_view = np.array([30.0, -36.0])
bonnet_view = np.array([24.0, -36.0])


# --- plumbing -------------------------------------------------------------------------
def square_view(view_degrees: np.ndarray, surfaces: list[core.Point], spare: float) -> solids.View:
    """A square orthographic view frames the supplied surfaces."""
    elevation, azimuth = np.radians(view_degrees + [0, 90])
    places = [solids.origin + solids.context.include_from_algebra(surface.batch()) for surface in surfaces]
    return solids.view(places, spare, elevation, azimuth, 0.06, n_pixels).squared()


def sheet(topology: core.Topology, points: solids.Point, corner_colours: np.ndarray, view: solids.View) -> Draw:
    """Smooth vertex normals light a triangulated surface with corner colours (`[F, 3, 3]`)."""
    return meshes.samples(points, topology.elements[2], corner_colours, meshes.smooth(points, topology), view,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)


def boundary_lines(topology: core.Topology, points: solids.Point, width: float, view: solids.View) -> Draw:
    """Crisp outlines mark the boundary wire of a surface."""
    return simplices.tubes(points, topology.border, ink, width, view, solids.gloss)


def disk_paths(rim: core.Point) -> core.Point:
    """Concentric rings and radial spokes sample a planar disk domain (`[n_rings + n_spokes, n_curve_samples] Point-`) from its rim vertices (`Point-[V_rim]`)."""
    radius = rim.norm().sites.mean()                                         # [] Scalar+
    turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_curve_samples, endpoint=False) / 2)).exp()  # [n_curve_samples] Rotor+
    rings = (radius * ring_shares)[:, None] * (turns >> mv.x.dual())         # [n_rings, n_curve_samples] Point-

    spoke_angles = np.linspace(-np.pi, np.pi, n_spokes, endpoint=False)
    spoke_planes = (mv.xy * (-spoke_angles / 2)).exp() >> mv.x               # [n_spokes] Plane+
    rim_points = rim.batch()                                                 # [V_rim] Point-
    # Pairing each spoke's radial plane with the unit rim directions selects the rim vertex closest in angle.
    alignments = spoke_planes[:, None] & rim_points.normalized()[None]       # [n_spokes, V_rim] Scalar+
    spoke_ends = rim_points[alignments.argmax(axis=-1)]                      # [n_spokes] Point-
    spokes = np.linspace(0, 1, n_curve_samples) * spoke_ends[:, None]        # [n_spokes, n_curve_samples] Point-
    return concatenate((rings, spokes))


def path_sampler(topology: core.Topology, flat_vertices: core.Point, paths: core.Point) -> SparseExtensor:
    """Piecewise-linear interpolation lifts planar domain paths (`[n_curves, n_samples] Point-`) onto any surface over `topology`."""
    n_curves, n_samples = paths.shape
    corners = topology.incidence[2, 0]                                       # [3] Scalar+[F <- V]
    domain_corners = corners * flat_vertices                                 # [3] Point-[F]
    sides = topology.boundary[2, 1] * (topology.d[0] * flat_vertices)        # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[F]
    gradients = (sides | patches.inverse()) / 2                              # [3] Plane+[F]
    # Three affine corner weights always sum to one, so their magnitudes sum to one only inside the enclosing triangle.
    weights_everywhere = 1 + (gradients.batch()[:, None, None]
                              & (paths[..., None] - domain_corners.batch()[:, None, None]))  # [3, n_curves, n_samples, F] Scalar+
    enclosing_faces = topology.reading(weights_everywhere.abs().sum(axis=0).argmin(axis=-1), 2)  # [n_curves] Scalar+[n_samples <- F]
    weights = 1 + ((enclosing_faces[None] * gradients[:, None])
                   & (paths.field() - enclosing_faces[None] * domain_corners[:, None]))  # [3, n_curves] Scalar+[n_samples]
    return (spdiag(weights) * (enclosing_faces[None] * corners[:, None])).sum(axis=0)  # [n_curves] Scalar+[n_samples <- V]


def surface_with_curves(
    topology: core.Topology, sampler: SparseExtensor, surface: core.Point, corner_colours: np.ndarray, view: solids.View,
) -> np.ndarray:
    """A lit surface sheet displays its boundary wire and lifted domain curves."""
    curves = solids.origin + solids.context.include_from_algebra((sampler * surface).batch())  # [..., n_curve_samples] Point-
    n_curves, n_samples = curves.shape
    # Draw each ring as a closed teal loop and each radial spoke in its own meridian colour.
    ring_strokes = simplices.tubes(curves[:n_rings].reshape(-1), Topology.loops([n_samples] * n_rings),
                                   teal, 2.0, view, solids.gloss)
    spoke_strokes = simplices.tubes(curves[n_rings:].reshape(-1),
                                    Topology.chains([n_samples] * (n_curves - n_rings)),
                                    np.repeat(meridian_colours, n_samples - 1, axis=0), 2.0, view, solids.gloss)
    points = solids.origin + solids.context.include_from_algebra(surface.batch())  # [V] Point-
    base = sheet(topology, points, corner_colours, view)
    rim = boundary_lines(topology, points, 2.2, view)
    return solids.draw([], view, background, inks=[], translucent=[],
                       meshes=[(base, 1.0), (ring_strokes, 1.0), (spoke_strokes, 1.0), (rim, 1.0)])


def plateau_animation(
    topology: core.Topology, flat_vertices: core.Point,
    states: Iterable[tuple[core.Point, core.Scalar, core.Curvature]],
) -> Iterator[np.ndarray]:
    """Yields frames as the domed surface relaxes onto the four-arch minimal soap film while its mean-curvature tint fades."""
    states = list(states)
    # Centre every frame on the relaxed film's mean point so the held boundary wire stays stationary.
    centre = states[-1][0].sites.mean()                                      # [] Point-
    surfaces = [vertices - centre for vertices, _, _ in states]
    view = square_view(plateau_view, surfaces, 0.12)
    sampler = path_sampler(topology, flat_vertices, disk_paths(topology.reading(topology.exterior[0], 0) * flat_vertices))
    # Anchor full coral near the top of the initial dome's curvature distribution.
    scale = max(float(np.percentile(states[0][2].norm().batch().to_array(), 95)), 1e-6)
    for surface, (_, _, curvature) in zip(surfaces, states):
        # Face-corner curvature magnitudes set the coral tint across each triangle.
        corner_curvatures = (topology.incidence[2, 0] * curvature.norm()).batch().to_array().T  # [F, 3]
        intensity = np.clip(corner_curvatures / scale, 0.0, 1.0)[..., None]
        corner_colours = (1 - intensity) * surface_colour + intensity * coral
        yield surface_with_curves(topology, sampler, surface, corner_colours, view)


def conjugate_pair(
    topology: core.Topology, flat_vertices: core.Point, vertices: core.Point, conjugate: core.Point,
) -> np.ndarray:
    """Side-by-side panels pair a minimal surface with its harmonic conjugate and matched domain curves."""
    surfaces = [vertices - vertices.sites.mean(), conjugate - conjugate.sites.mean()]
    view = square_view(conjugate_view, surfaces, 0.12)
    sampler = path_sampler(topology, flat_vertices, disk_paths(topology.reading(topology.exterior[0], 0) * flat_vertices))
    left, right = (surface_with_curves(topology, sampler, surface, surface_colour, view) for surface in surfaces)
    return np.concatenate((left, np.full((n_pixels, 8, 3), 255, dtype=np.uint8), right), axis=1)


def conjugates_figure(
    plateau_topology: core.Topology, plateau_flat: core.Point,
    plateau_vertices: core.Point, plateau_conjugate: core.Point,
    enneper_topology: core.Topology, enneper_flat: core.Point,
    enneper_vertices: core.Point, enneper_conjugate: core.Point,
) -> plt.Figure:
    """Two rows compare the relaxed four-arch film and Enneper's surface with their recovered harmonic conjugates."""
    top_row = conjugate_pair(plateau_topology, plateau_flat, plateau_vertices, plateau_conjugate)
    bottom_row = conjugate_pair(enneper_topology, enneper_flat, enneper_vertices, enneper_conjugate)
    h_gutter = np.full((8, top_row.shape[1], 3), 255, dtype=np.uint8)
    return image_figure(np.concatenate((top_row, h_gutter, bottom_row), axis=0))


def bonnet_animation(
    topology: core.Topology, rings: np.ndarray, meridians: np.ndarray,
    surfaces: Iterable[tuple[core.Point, core.Patch]],
) -> Iterator[np.ndarray]:
    """Yields frames as the slit catenoid uncurls isometrically into the helicoid with fixed Gauss normal pins along its rings (`[n_heights, n_azimuths]`) and meridians (`[n_azimuths, n_heights]`)."""
    frames = [(surface - surface.sites.mean(), normals) for surface, normals in surfaces]
    view = square_view(bonnet_view, [surface for surface, _ in frames], 0.14)

    n_heights, n_azimuths = len(rings), len(meridians)
    ring_reading = topology.reading(rings[np.linspace(2, n_heights - 3, 5, dtype=int)], 0)  # [5] Scalar+[n_azimuths <- V]
    meridian_reading = topology.reading(meridians[np.linspace(3, n_azimuths - 4, 6, dtype=int)], 0)  # [6] Scalar+[n_heights <- V]
    # Place normal pins at a coarse grid of ring-meridian crossings.
    pin_rings = rings[np.linspace(3, n_heights - 4, 3, dtype=int)]
    pin_reading = topology.reading(pin_rings[:, np.linspace(5, n_azimuths - 6, 4, dtype=int)].reshape(-1), 0)  # Scalar+[12 <- V]

    for surface, normals in frames:
        ring_curves = solids.origin + solids.context.include_from_algebra((ring_reading * surface).batch())  # [5, n_azimuths] Point-
        meridian_curves = solids.origin + solids.context.include_from_algebra((meridian_reading * surface).batch())  # [6, n_heights] Point-
        # Draw each ring in teal and each meridian in coral as edge chains.
        lengths = [n_azimuths] * len(ring_curves) + [n_heights] * len(meridian_curves)
        colours = np.repeat(np.stack((teal, coral)),
                            [len(ring_curves) * (n_azimuths - 1), len(meridian_curves) * (n_heights - 1)], axis=0)
        curve_strokes = simplices.tubes(concatenate((ring_curves.reshape(-1), meridian_curves.reshape(-1))),
                                        Topology.chains(lengths), colours, 2.0, view, solids.gloss)
        points = solids.origin + solids.context.include_from_algebra(surface.batch())  # [V] Point-
        base = sheet(topology, points, surface_colour, view)
        rim = boundary_lines(topology, points, 1.8, view)

        # Outward normal pins at the crossings show that Gauss normals keep fixed directions in space.
        pin_points = pin_reading * surface                                   # Point-[12]
        normal_directions = (pin_reading * normals * unit_volume).dual()     # Point-[12]
        roots = solids.origin + solids.context.include_from_algebra((pin_points - pin_depth * normal_directions).batch())  # [12] Point-
        tips = solids.origin + solids.context.include_from_algebra((pin_points + pin_reach * normal_directions).batch())   # [12] Point-
        stems = simplices.tubes(stack((roots, tips), axis=-1).reshape(-1), Topology.chains([2] * len(roots)),
                                gold, 2.2, view, solids.gloss)
        heads = solids.ball(tips, pin_head_radius)[:, None]
        yield solids.draw([(heads, gold)], view, background, inks=[], translucent=[],
                          meshes=[(base, 1.0), (curve_strokes, 1.0), (rim, 1.0), (stems, 1.0)])
