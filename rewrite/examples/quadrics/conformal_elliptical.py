"""Conformal and spherical raycasting geometry examples.

Renders planar intersections on the unit 2-sphere using geometric algebra,
demonstrating rigid and conformal transformations.
"""

from __future__ import annotations

import numpy as np

from numga.algebra import Algebra
from numga import NumpyContext
from examples import PLOT_DIR


ga = Algebra("x+y+z+")
Vector = ga.gatype.vector()


def setup_rays(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Set up parallel rays impinging on the unit 2-sphere."""
    ones = np.ones((n, n))
    x = np.linspace(-1.0, +1.0, n)[:, None] * ones
    y = np.linspace(-1.0, +1.0, n)[None, :] * ones
    z = np.sqrt(np.maximum(1.0 - x**2 - y**2, 0.0))
    mask = (x**2 + y**2) > 1.0
    p = np.stack([x, y, z], axis=-1)
    return p, mask


def render(rays: Vector, planes: Vector, scale: float = 512.0) -> np.ndarray:
    """Render planes intersecting surface of a sphere with antialiasing."""
    v = (rays[:, :, None] | planes).kernel[..., 0]
    q = np.prod(np.tanh(v * scale), axis=-1)
    return (q + 1.0) / 2.0


def image_downsample(img: np.ndarray, bin_size: int = 2) -> np.ndarray:
    """Downsample image by bin_size x bin_size average pooling."""
    input_size = img.shape[0]
    output_size = input_size // bin_size
    return img.reshape((output_size, bin_size, output_size, bin_size)).mean((1, 3))


def octahedral_planes(context: NumpyContext) -> Vector:
    """All 13 planes of the octahedral symmetry group."""
    planes = np.array(np.meshgrid(*[[1, 0, -1]] * 3)).reshape(3, -1).T[:13]
    return context.multivector.vector(planes).normalized()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cga = NumpyContext(ga)
    x = cga.multivector.vector([1.0, 0.0, 0.0])
    y = cga.multivector.vector([0.0, 1.0, 0.0])
    z = cga.multivector.vector([0.0, 0.0, 1.0])

    rays, mask = setup_rays(512)
    rays = cga.multivector.vector(rays)
    planes = octahedral_planes(cga)

    render2 = lambda p: image_downsample((render(rays, p) + 1.0) * (1.0 - mask))
    b = -(x.wedge(z)) + (y.wedge(z))

    # Vectorized motor batch across all angles:
    alphas = cga.multivector.scalar(np.linspace(0, np.pi, 100)[:, None])
    motors = (b.normalized() * alphas).exp()
    rot_planes = motors[:, None].sandwich(planes)  # Shape: (100, 13)

    frame = render2(planes)
    plt.imshow(frame, cmap="gray")
    plt.axis("off")
    plt.title("Octahedral Planes on Unit 2-Sphere")
    plt.savefig(PLOT_DIR / "sphere_rigid.png", bbox_inches="tight", dpi=150)
    print(f"Saved {PLOT_DIR / 'sphere_rigid.png'}")
    plt.show()
