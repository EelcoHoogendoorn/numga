"""Sampling, population setup, broad-phase selection and rendering for S³ quadric physics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import save_gif
from examples.sketches.spherical_raytracer import DualQuadric, Point, ScreenPoint, mv, origin, render
from examples.sketches.s3_quadric_physics import Bodies, body, cap_quadric, overlap, placed


def filled(Q: DualQuadric, mass: np.ndarray, count: int, rng: np.random.Generator) -> tuple[Point, np.ndarray]:
    """Mass points filling the inside of any quadric, where its form is negative. In the form's
    eigenbasis the inside is where the negative block outweighs the positive one, so a point is a
    core direction in the negative block, an extent direction in the positive block, both uniform
    on their spheres, and the angle between them, up to the angle where the blocks balance; the
    angle is drawn uniformly and weighted by the sphere's measure, cosᵏ⁻¹ sin³⁻ᵏ for k core axes,
    so the weighted points are uniform in the inside. Batched over quadrics of one signature."""
    values, principal = (Point & Q.inverse()(mv.rotor() >> Point)).eigh()   # the negative block comes first
    values = values.kernel[..., 0]
    k = int((values < 0.0).sum(axis=-1).ravel()[0])                # core axes, the same across the batch
    core = rng.normal(size=values.shape[:-1] + (count, k))
    extent = rng.normal(size=values.shape[:-1] + (count, 4 - k))
    core, extent = core / np.linalg.norm(core, axis=-1, keepdims=True), extent / np.linalg.norm(extent, axis=-1, keepdims=True)
    balance = np.arctan(np.sqrt(np.einsum("...c,...nc->...n", -values[..., :k], core**2) / np.einsum("...e,...ne->...n", values[..., k:], extent**2)))
    angle = rng.uniform(size=balance.shape) * balance
    weight = np.cos(angle) ** (k - 1) * np.sin(angle) ** (3 - k) * balance
    coordinates = np.concatenate([np.cos(angle)[..., None] * core, np.sin(angle)[..., None] * extent], axis=-1)
    return (principal[..., None, :] * coordinates).sum(axis=-1), weight * mass[..., None] / weight.sum(axis=-1, keepdims=True)



def candidates_near(bodies: Bodies) -> tuple[np.ndarray, np.ndarray]:
    """Broad phase: the pairs whose poles are closer than their reaches summed."""
    poles = bodies.motor >> origin
    apart = np.arccos(np.clip(np.abs((poles.reshape(-1, 1) | poles).kernel[..., 0]), 0.0, 1.0))
    return np.nonzero(np.triu(apart < bodies.reach[:, None] + bodies.reach[None, :], 1))



def population(rng: np.random.Generator, count: int, candidates: int, sizes: tuple[float, float], *seeded: Bodies, drift: float = 0.6) -> Bodies:
    """The seeded bodies plus `count` random caps out of `candidates` draws: half-widths log-uniform
    per axis between the sizes, so needles, discs and blobs; placed uniformly on the 3-sphere; kept
    where they overlap neither the seeded bodies nor any kept before them (one batched overlap test
    over the pairs within reach, then a greedy pass over its boolean matrix). Tennis-racket rates:
    spin about the intermediate axis, the middle half-width, whose bivector is the plane it is
    normal to (axis x -> yz, y -> zx, z -> xy), plus a nudge and a drift across the sphere."""
    half_widths = 10 ** rng.uniform(np.log10(sizes[0]), np.log10(sizes[1]), size=(candidates, 3))
    rate = rng.normal(size=(candidates, 6)) * np.array([0.3, 0.3, 0.3, drift, drift, drift])
    rate[np.arange(candidates), np.argsort(half_widths, axis=-1)[:, 1]] = rng.choice([-5.0, 5.0], size=candidates)
    place, spin = mv.antivector(rng.normal(size=(candidates, 4))), mv.bivector(rng.normal(size=(candidates, 6)) * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    drawn = body(plt.get_cmap("hsv")(rng.permutation(candidates) / candidates)[:, :3], cap_quadric(half_widths), placed(place, spin), mv.bivector(rate), np.prod(half_widths, axis=-1) * 200.0, rng)
    everyone = Bodies.join(*seeded, drawn)
    given = len(everyone.reach) - candidates
    i, j = candidates_near(everyone)
    relative = everyone.motor[i].inverse() * everyone.motor[j]
    overlapping = np.zeros((len(everyone.reach),) * 2, dtype=bool)
    overlapping[i, j] = overlapping[j, i] = overlap(everyone.C[i], relative >> everyone.C[j](relative << Point))[0] < 0.0
    keep = list(range(given))
    for candidate in range(given, len(everyone.reach)):
        if not overlapping[candidate, keep].any():
            keep.append(candidate)
    assert len(keep) >= given + count, f"only {len(keep) - given} admissible non-overlapping candidates"
    keep = np.array(keep[:given + count])
    return Bodies(everyone.color[keep], everyone.motor[keep], everyone.momentum[keep], everyone.Q[keep], everyone.C[keep], everyone.I_inv[keep], everyone.reach[keep])



def render_states(states, colors, chart: ScreenPoint, shape: tuple[int, int], supersample: int) -> list[np.ndarray]:
    """Rasterize the geometric states yielded by the simulation."""
    return [(render(eye, world, surfaces, colors, light, chart, shape, supersample) * 255).astype(np.uint8)
            for eye, light, world, surfaces in states]



def draw_simulation(frames_out, body_count: int, plot_path: str, animation_path: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.imshow(frames_out[-1], interpolation="nearest"); ax.axis("off"); ax.set_title(f"{body_count} caps on the 3-sphere after {len(frames_out)} frames")
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    if animation_path:
        save_gif(frames_out, animation_path, duration_ms=50)

    return fig
