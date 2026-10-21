"""A rattleback with a curved underside and flat deck, viewed above its ground."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from numga import stack
from numga.algebras import PGA3D
from examples import instantiate
from examples.raster import view_frame

core = instantiate("examples.mechanics.rattleback.core", PGA3D)
mv = core.mv
# The planes across the axes, which read points and directions by pairing, and the axes' directions.
mirrors = stack((mv.x, mv.y, mv.z))
directions = mirrors.dual()

n_pixels = 500
n_supersamples = 2
VIEW = (25, -90)
SETUP_VIEW = (58, -68)
LIGHT = np.array([-0.3, -0.6, 0.85])
BODY = np.array([0.19, 0.52, 0.62])
NOSE = np.array([0.95, 0.48, 0.18])
DECK = np.array([0.91, 0.81, 0.60])
GROUND = np.array([0.95, 0.94, 0.91])
MASS = np.array([0.81, 0.34, 0.12])
AMBIENT = 0.38
SHADOW = 0.22
NOSE_START = 0.65
STRIPE_HALF_WIDTH = 0.06
GRID_SPACING = 2
GRID_WIDTH = 0.018
GRID_CONTRAST = 0.07


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """Cartesian positions at unit weight: each point paired with the planes across the axes, over its
    pairing with the plane at infinity."""
    return ((mirrors & points[..., None]) / (core.infinity & points[..., None])).to_array()


def matrix(outline: core.Stretch) -> np.ndarray:
    """The stretch as a matrix on the last two axes: each axis's direction stretched, read by the planes."""
    return (mirrors[:, None] & outline[..., None, None](directions[None, :])).to_array()


def placements(motors: core.Motor, outline: core.Stretch) -> tuple[np.ndarray, np.ndarray]:
    """The ellipsoid's centres and maps from its unit sphere, for ray casting."""
    return coordinates(motors >> core.origin), matrix(motors >> outline)


def camera(centres: np.ndarray, transforms: np.ndarray, view: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parallel camera rays and a fixed viewport enclosing the complete motion."""
    frame = view_frame(*view)
    viewed = centres @ frame.T
    radii = np.linalg.norm(np.einsum("ij,...jk->...ik", frame, transforms), axis=-1)
    # The directional shadow belongs to the scene bounds as well as the body.
    shadow_centres = centres - centres[..., 2:] * LIGHT / LIGHT[2]
    shadow_transforms = transforms - LIGHT[:, None] * transforms[..., 2:3, :] / LIGHT[2]
    shadow_viewed = shadow_centres @ frame.T
    shadow_radii = np.linalg.norm(np.einsum("ij,...jk->...ik", frame, shadow_transforms), axis=-1)
    low = np.minimum(viewed - radii, shadow_viewed - shadow_radii).reshape(-1, 3).min(axis=0)
    high = np.maximum(viewed + radii, shadow_viewed + shadow_radii).reshape(-1, 3).max(axis=0)
    middle = (low + high) / 2
    extent = 0.55 * (high - low)[:2]
    n_height = round(n_pixels * extent[1] / extent[0])
    extent[1] = extent[0] * n_height / n_pixels
    n_samples = n_pixels * n_supersamples
    n_rows = n_height * n_supersamples
    horizontal = ((np.arange(n_samples) + 0.5) / n_samples * 2 - 1) * extent[0]
    vertical = ((np.arange(n_rows) + 0.5) / n_rows * 2 - 1) * extent[1]
    origin = middle @ frame + frame[2] * (2 * extent.max())
    rays = origin + horizontal[None, :, None] * frame[0] - vertical[:, None, None] * frame[1]
    return rays, -frame[2], frame, middle, extent


def intersection(origins: np.ndarray, heading: np.ndarray, centre: np.ndarray,
                 inverse: np.ndarray, deck_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The ray's entry into the ellipsoid below its flat deck."""
    start = (origins - centre) @ inverse.T
    direction = heading @ inverse.T
    quadratic = np.dot(direction, direction)
    linear = start @ direction
    constant = np.einsum("...i,...i->...", start, start) - 1
    discriminant = linear * linear - quadratic * constant
    root = np.sqrt(np.maximum(discriminant, 0))
    entering = (-linear - root) / quadratic
    leaving = (-linear + root) / quadratic
    # Clip the interval inside the ellipsoid to the halfspace below the deck.
    with np.errstate(divide="ignore", invalid="ignore"):
        deck_distance = (deck_fraction - start[..., 2]) / direction[2]
    distance = np.where(direction[2] < 0, np.maximum(entering, deck_distance), entering)
    leaving = np.where(direction[2] > 0, np.minimum(leaving, deck_distance), leaving)
    hit = (discriminant >= 0) & (distance >= 0) & (distance <= leaving)
    hit &= (direction[2] != 0) | (start[..., 2] <= deck_fraction)
    deck = (direction[2] < 0) & (distance == deck_distance)
    local = start + distance[..., None] * direction
    return hit, distance, local, deck


def picture(centre: np.ndarray, transform: np.ndarray, rays: np.ndarray,
            heading: np.ndarray, deck_fraction: float) -> np.ndarray:
    """The hull and its shadow, with a striped deck and an orange nose."""
    inverse = np.linalg.inv(transform)
    lamp = LIGHT / np.linalg.norm(LIGHT)
    ground_distance = -rays[..., 2] / heading[2]
    ground = rays + ground_distance[..., None] * heading
    shadow, _, _, _ = intersection(ground, lamp, centre, inverse, deck_fraction)
    grid_distance = np.abs(ground[..., :2] / GRID_SPACING - np.rint(ground[..., :2] / GRID_SPACING)) * GRID_SPACING
    grid = np.clip(1 - grid_distance.min(axis=-1) / GRID_WIDTH, 0, 1)
    colour = GROUND * (1 - GRID_CONTRAST * grid[..., None]) * (1 - SHADOW * shadow[..., None])

    hit, distance, local, deck = intersection(rays, heading, centre, inverse, deck_fraction)
    hit &= distance < ground_distance
    # Pulling a sphere normal back through the stretch gives the hull's surface normal.
    local_normal = np.where(deck[hit, None], [0, 0, 1], local[hit])
    normal = local_normal @ inverse
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    diffuse = AMBIENT + (1 - AMBIENT) * np.maximum(normal @ lamp, 0)
    halfway = lamp - heading
    halfway /= np.linalg.norm(halfway)
    highlight = 0.13 * np.maximum(normal @ halfway, 0) ** 32
    paint = np.where((local[hit, 0] > NOSE_START)[:, None], NOSE, BODY)
    deck_paint = np.where((np.abs(local[hit, 1]) < STRIPE_HALF_WIDTH)[:, None], BODY, DECK)
    paint = np.where(deck[hit, None], deck_paint, paint)
    colour[hit] = np.clip(paint * diffuse[:, None] + highlight[:, None], 0, 1)

    image = np.rint(colour * 255).astype(np.uint8)
    n_height = image.shape[0] // n_supersamples
    return np.asarray(Image.fromarray(image).resize((n_pixels, n_height), Image.Resampling.BOX))


def frames(motors: core.Motor, outline: core.Stretch, deck_fraction: float) -> list[np.ndarray]:
    """The rolling body, viewed from a fixed camera."""
    centres, transforms = placements(motors, outline)
    rays, heading, _, _, _ = camera(centres, transforms, VIEW)
    return [picture(centre, transform, rays, heading, deck_fraction) for centre, transform in zip(centres, transforms)]


def image_figure(image: np.ndarray) -> plt.Figure:
    """An image at its pixel size, without axes or margins."""
    figure, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
    ax.imshow(image)
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure


def draw_setup(outline: core.Stretch, mass_points: core.Point, deck_fraction: float) -> plt.Figure:
    """The six masses inside a translucent hull, viewed across their skewed axes."""
    transform = matrix(outline)
    centre = np.array([0, 0, np.linalg.norm(transform[2])])
    rays, heading, frame, middle, extent = camera(centre, transform, SETUP_VIEW)
    image = picture(centre, transform, rays, heading, deck_fraction)
    image = np.rint(0.23 * image + 0.77 * 255).astype(np.uint8)
    figure = image_figure(image)
    ax, = figure.axes

    positions = (coordinates(mass_points) + centre) @ frame.T - middle
    n_height = image.shape[0]
    image_size = np.array([n_pixels, n_height])
    projected = positions[:, :2] / (2 * extent) * image_size
    projected[:, 0] += n_pixels / 2
    projected[:, 1] = n_height / 2 - projected[:, 1]
    origin = (centre @ frame.T - middle)[:2] / (2 * extent) * image_size
    origin = np.array([origin[0] + n_pixels / 2, n_height / 2 - origin[1]])
    for endpoint in projected:
        ax.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]], color=BODY, linewidth=1.4)
    order = np.argsort(positions[:, 2])
    ax.scatter(*projected[order].T, s=105, color=MASS, edgecolors="white", linewidths=0.8)
    return figure
