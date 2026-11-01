"""The sphere below the Hopf circles seen from outside, ray traced on the GPU: an icosphere shaded smooth, coloured by
how much the circles above it stretch, red where they lengthen and blue where they shorten, blended across its faces;
the decagons' base points as balls in their guardrails' hues; triangle gauges lying on the sphere, their bonds arcs
coloured the same way by their strains; and the light's paths across it."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from numga import stack
from examples.relativity.hopf_projection import core
from examples.relativity.hopf_projection.scenarios import flat
from examples.util import charts
from examples.util.wgsl import meshes, simplices, solids

N_PIXELS = 300
HALF = 1.15
ELEVATION, AZIMUTH = 0.45, 2.75
view = solids.View(solids.aim(solids.origin, ELEVATION, AZIMUTH), np.full(2, HALF), N_PIXELS)
background = np.ones(3)
STRAIN_MAP = "coolwarm"
# The balls at the decagons' base points and the gauges' atoms, their radii in pixels, and the widths of the gauges'
# bonds and of the light's paths, in pixels; atoms, bonds and light lie on the sphere, raised by half the bonds' or
# the light's width.
DOT_RADIUS, ATOM_RADIUS = 4.5, 2.5
BOND_WIDTH, LIGHT_WIDTH = 3.0, 1.4
ATOM_INK = np.array([0.25, 0.25, 0.25])
LIGHT_INK = np.array([0.1, 0.1, 0.1])


def seen(steps: core.Point) -> flat.Point:
    """Unit steps of the three-sphere's origin, `[n] Point-`, as points of the algebra of space, read by pairing with
    the planes across x, y and z: a stopgap until the steps embed in the renderer's space."""
    coordinates = (stack((core.three_sphere.mv.x, core.three_sphere.mv.y, core.three_sphere.mv.z), axis=-1)
                   & steps[..., None]).to_array()                               # [n, 3]
    return flat.mv.vector(coordinates).dual()


def lying(steps: core.Point, height: float) -> solids.Point:
    """Points of the base `[...] Point-` in the scene, raised above the sphere by `height` pixels."""
    return solids.origin + solids.context.include_from_algebra(seen(steps)) * (1 + height * view.pixel)


def on_screen(steps: core.Point) -> tuple[np.ndarray, np.ndarray]:
    """Points of the base `[...] Point-` on the screen, in pixels across and up from its middle, `[..., 2]`, and
    whether each faces the viewer, `[...]`: in front of the plane through the sphere's middle across the view, which
    the camera looks down along its own -z."""
    places = lying(steps, 0.0)                                                  # [...] Point-
    seen_by_camera = view.camera << places                                      # [...] Point-
    depths = ((solids.mv.z & seen_by_camera) / (solids.mv.w & seen_by_camera)).to_array()
    return view.across(places) / view.pixel, depths > 0


def inks(stretches: core.Scalar, saturating: float) -> np.ndarray:
    """The colours of stretches `[n] Scalar+`, blue shortened and red lengthened, saturating at `saturating`."""
    return plt.get_cmap(STRAIN_MAP)(0.5 + 0.5 * np.clip(stretches.to_array().reshape(-1) / saturating, -1, 1))[:, :3]


def base_view(sphere_mesh: flat.Mesh, sphere_points: flat.Point, stretches: core.Scalar, decagon_bases: core.Point,
              atoms: core.Point, arcs: core.Point, strains: core.Scalar, light: core.Point, saturating: float
              ) -> np.ndarray:
    """The base from outside: the icosphere coloured by how much the circles above its points stretch,
    `Scalar+[points]`, the decagons' base points `[decagons] Point-` in hues around the decagons, the gauges' atoms
    `[atoms] Point-` and their bonds' arcs `[bonds, arc points] Point-` coloured by their strains `[bonds] Scalar+`,
    and the light's paths `[rays, steps + 1] Point-`, all on the base."""
    pixel = view.pixel

    points = solids.origin + solids.context.include_from_algebra(sphere_points.batch())   # [points] Point-
    faces = sphere_mesh.faces
    normals = meshes.smooth(points, meshes.Mesh(points.shape[0], faces))
    surface = meshes.samples(points, faces, inks(stretches, saturating)[faces], normals, view, np.zeros(3),
                             meshes.plain, solids.gloss)
    n_bonds, n_arc_points = arcs.shape
    bonds = simplices.tubes(lying(arcs.reshape(-1), BOND_WIDTH / 2), simplices.Mesh.chains([n_arc_points] * n_bonds),
                            np.repeat(inks(strains, saturating), n_arc_points - 1, axis=0), BOND_WIDTH, view,
                            solids.gloss)
    n_rays, n_path_points = light.shape
    rays = simplices.tubes(lying(light.reshape(-1), LIGHT_WIDTH / 2), simplices.Mesh.chains([n_path_points] * n_rays),
                           LIGHT_INK, LIGHT_WIDTH, view, solids.gloss)
    hues = np.arange(decagon_bases.shape[0]) / decagon_bases.shape[0]
    dot_inks = hsv_to_rgb(np.stack([hues, np.full_like(hues, 0.75), np.full_like(hues, 0.95)], axis=-1))
    parts = [(solids.ball(lying(decagon_bases, 0.0), DOT_RADIUS * pixel)[:, None], dot_inks),
             (solids.ball(lying(atoms, BOND_WIDTH / 2), ATOM_RADIUS * pixel)[:, None],
              np.broadcast_to(ATOM_INK, (atoms.shape[0], 3)))]
    return solids.draw(parts, view, background, inks=[], translucent=[],
                       meshes=[(surface, 1.0), (bonds, 1.0), (rays, 1.0)])


# --- the gauges' and the light's response -----------------------------------------------------------------------------
response_width, response_height = 520, 230


@charts.style()
def response(strains: core.Scalar, light: core.Point, undisturbed: core.Point, target_strain: float,
             target_sways: tuple[float, float]) -> plt.Figure:
    """The gauges' and the light's response over a swing, against what they are tuned to show: each bond's strain
    from one end of its swing to the other, from its strains at each frame `[frames, bonds] Scalar+`, largest first,
    against `target_strain`; and each ray's largest sway on the screen from its undisturbed path, among its points
    that face the viewer, from its paths at each frame `[frames, rays, steps + 1] Point-` and undisturbed `[rays,
    steps + 1] Point-`, as a share of the picture, against the band `target_sways`."""
    figure, (gauges, rays) = charts.subplots(response_width, response_height, ncols=2)
    readings = strains.to_array().reshape(strains.shape)                       # [frames, bonds]
    swings = np.sort(readings.max(axis=0) - readings.min(axis=0))[::-1]        # [bonds]
    gauges.bar(np.arange(len(swings)), swings, width=1.0, color="#d0453b")
    gauges.axhline(target_strain, color="0.3", linewidth=charts.FRAME_WIDTH, linestyle=(0, (3, 2)))
    gauges.set_xticks([])
    gauges.set_yticks([0, target_strain, round(swings.max(), 2)])
    gauges.set_xlabel("bonds, largest first")
    gauges.set_ylabel("bond strain, peak to peak")
    disturbed, facing = on_screen(light)                                        # [frames, rays, steps + 1, 2], [frames, rays, steps + 1]
    calm, _ = on_screen(undisturbed)                                            # [rays, steps + 1, 2]
    sways = np.linalg.norm(disturbed - calm[None], axis=-1) * facing / N_PIXELS   # [frames, rays, steps + 1]
    largest = sways.max(axis=(0, 2))                                            # [rays]
    rays.axhspan(*target_sways, color="0.9")
    rays.bar(np.arange(len(largest)), largest, width=0.7, color="#3b6fb6")
    rays.set_xticks([])
    rays.set_yticks([0, *target_sways, round(largest.max(), 2)])
    rays.set_xlabel("rays, across the bundle")
    rays.set_ylabel("ray sway, share of picture")
    for ax in (gauges, rays):
        ax.axhline(0, color="0.6", linewidth=charts.FRAME_WIDTH)
        ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure
