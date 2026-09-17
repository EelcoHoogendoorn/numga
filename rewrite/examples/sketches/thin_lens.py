"""Gaussian optics in PGA2D: a thin lens is a linear map on lines, so it is an extensor.

A ray is a line, not a height and a slope at some reference plane. A thin lens of focal
length f at the origin sends the line a x + b y + c w to (a - c/f) x + b y + c w, which is
Line - (Line ∨ origin) x / f with the line left open. A lens elsewhere is that map conjugated
by a translator, and a system of lenses is a composition. There is no free-space propagation
matrix, because lines are already global; ABCD matrices are what you get when you insist on
describing a line by where it crosses one particular plane.
"""

from __future__ import annotations

from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D

from examples import PLOT_DIR
from examples.animation import capture, save_gif

ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
Line = ga.gatype.vector()
P = ga.subspace.antivector()
Point = ga.gatype.antivector()
LineMap = ga.gatype((Line, Line))
Motor = ga.gatype.rotor()


# --- plumbing -------------------------------------------------------------------------
def xy(points: Point) -> np.ndarray:
    k = points.cast(P).kernel
    return k[..., :2] / k[..., 2:]


def draw_rays(ax, rays: Line, start: Line, stop: Line, color: str) -> None:
    """Draw ray segments between two planes (vertical lines)."""
    a, b = xy(rays ^ start), xy(rays ^ stop)
    for p, q in zip(a, b):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=0.7)


def travel(rays: Line, handedness: np.ndarray) -> np.ndarray:
    """Unit direction of travel along each ray: its normal turned a quarter turn, times the handedness."""
    n = rays.kernel[:, :2] / np.linalg.norm(rays.kernel[:, :2], axis=-1, keepdims=True)
    return handedness[:, None] * np.stack([-n[:, 1], n[:, 0]], axis=-1)


def heading(rays: Line, start: Line, stop: Line) -> np.ndarray:
    """The handedness (±1 per ray) for which travel() points from the start plane toward the stop plane."""
    step = xy(rays ^ stop) - xy(rays ^ start)
    return np.sign(np.einsum("ij,ij->i", step, travel(rays, np.ones(len(step)))))


def draw_onward(ax, rays: Line, start: Line, handedness: np.ndarray, length: float, color: str) -> None:
    """Draw each ray from its point on the start plane a fixed length along its direction of travel."""
    a = xy(rays ^ start)
    for p, q in zip(a, a + length * travel(rays, handedness)):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=0.7)


def draw_scene(ax, subject: Point, planes: list[Line], legs: list[Line], train: LineMap, picture: Point) -> None:
    """Draw the bundle leg by leg between the element planes, then onward along its direction of travel.

    A line has no direction of travel, so the last leg follows a handedness that flips once per
    orientation-reversing element, which is what a mirror is: the sign of the train's determinant.
    """
    ax.cla()
    starts = [subject & mv.wx] + planes
    for stage, (rays, start, plane) in enumerate(zip(legs, starts, planes)):
        draw_rays(ax, rays, start, plane, f"C{stage}")
        draw_plane(ax, plane, 0.8)
    handedness = heading(legs[0], starts[0], planes[0]) * np.sign(train.det().kernel.item())
    draw_onward(ax, legs[-1], planes[-1], handedness, 2.5, f"C{len(planes)}")
    ax.scatter(*xy(subject), color="C0", zorder=3); ax.scatter(*xy(picture), color=f"C{len(planes)}", zorder=3)
    ax.set_xlim(-1.3, 4.7); ax.set_ylim(-1.5, 3.2); ax.set_aspect("equal")


def draw_plane(ax, plane: Line, half_height: float) -> None:
    """Draw an element's plane between the lines y = ±half_height."""
    a, b = xy(plane ^ (mv.y + mv.w * half_height)), xy(plane ^ (mv.y - mv.w * half_height))
    ax.plot([a[0], b[0]], [a[1], b[1]], color="gray")


def draw_lenses(rays, obj, plane_1, plane_2, out, image, parallel, lens_1, focused, focus, plot_path) -> plt.Figure:
    beyond = (mv.wx * -0.25).exp()                                  # half a unit further along x
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=120, sharex=True)
    draw_rays(axes[0], rays, obj & mv.wx, plane_1, "tab:orange")
    draw_rays(axes[0], out, plane_1, beyond >> (image & mv.wx), "tab:blue")
    draw_plane(axes[0], plane_1, 1.0); axes[0].set_title("one thin lens")
    draw_rays(axes[1], parallel, (mv.wx * 0.5).exp() >> plane_1, plane_1, "tab:orange")
    draw_rays(axes[1], lens_1(parallel), plane_1, plane_2, "tab:green")
    draw_rays(axes[1], focused, plane_2, beyond >> (focus & mv.wx), "tab:blue")
    draw_plane(axes[1], plane_1, 1.0); draw_plane(axes[1], plane_2, 1.0); axes[1].set_title("two lenses")
    for ax in axes:
        ax.set_aspect("equal")
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")

    return fig


def draw_train(states, subject: Point, animation_path: str) -> None:
    """Draw the optical train from its completed geometric states."""
    if animation_path:
        anim = plt.figure(figsize=(7, 5), dpi=100)
        ax = anim.add_subplot()
        frames_out = []
        for planes, legs, train, picture in states:
            draw_scene(ax, subject, planes, legs, train, picture)
            frames_out.append(capture(anim))
        plt.close(anim)
        save_gif(frames_out, animation_path, duration_ms=60)


# --- math -----------------------------------------------------------------------------
def main(
    plot_path: str = str(PLOT_DIR / "sketch_thin_lens.png"),
    animation_path: str = str(PLOT_DIR / "sketch_thin_lens.gif"),
) -> plt.Figure:
    origin = mv.xy
    # Each lens is a plane (a vertical line) and a focal length; nothing else is stored.
    plane_1, focal_1 = mv.x - mv.w * 1.5, 1.0
    plane_2, focal_2 = mv.x - mv.w * 2.1, 0.5

    # 1. A thin lens at the origin is a linear map on lines: it shears a line's x coefficient
    #    by the line's incidence with the origin over the focal length.
    shear_1 = Line - (Line & origin) * (mv.x / focal_1)
    shear_2 = Line - (Line & origin) * (mv.x / focal_2)

    # A lens anywhere else is that map conjugated by the translator onto its plane, generated
    # by the ideal point along y scaled by the plane's offset from the origin.
    shift_1 = (mv.wx * (plane_1 & origin) * 0.5).exp()
    shift_2 = (mv.wx * (plane_2 & origin) * 0.5).exp()
    lens_1 = shift_1 >> shear_1(shift_1 << Line)
    lens_2 = shift_2 >> shear_2(shift_2 << Line)

    # Rays from an object point through a pupil of points on the lens plane. Every
    # transformed ray passes through one point: the image, the meet of any two of them.
    obj = (mv.x + mv.w * 2.0) ^ (mv.y - mv.w * 0.5)                # object at (-2, 0.5)
    pupil = plane_1 ^ (mv.y - mv.w * np.linspace(-0.8, 0.8, 9))
    rays = obj & pupil
    out = lens_1(rays)
    image = out[0] ^ out[1]
    print(f"image at {np.round(xy(image), 4)}")

    # 2. Two lenses. The system is a composition of maps. Parallel rays are joins with the
    #    ideal point along x; where they meet after the system is the back focal point.
    system = lens_2(lens_1)
    parallel = mv.yw & pupil
    focused = system(parallel)
    focus = focused[0] ^ focused[1]
    print(f"two-lens focus at {np.round(xy(focus), 4)}")
    print("system map on lines:\n", np.round(system.kernel, 3))

    # 3. An optical train in the abstract. Every element is a map on lines defined in one home
    #    plane, x = 0: a thin lens shears a line by its incidence with the centre, a thin prism
    #    by its incidence with an ideal point (the same slope change for every height), and a
    #    flat mirror is the sandwich by the home plane. A motor per element places it, and the train is the
    #    composition of the placed elements, one map on lines that acts on the whole bundle.
    #    A lens still images when tilted, because the ideal thin lens is a collineation and
    #    concurrent rays stay concurrent. The animation slides and tilts the first lens and
    #    rocks the mirror; the image is the meet of two output rays.
    home: Line = mv.x
    elements: tuple[LineMap, ...] = (
        Line - (Line & origin) * (home / 1.0),                      # thin lens, f = 1
        Line - (Line & mv.wx) * home * 0.15,                        # thin prism, slope change 0.15
        Line - (Line & origin) * (home / -2.0),                     # thin lens, f = -2
        home.normalized() >> Line,                                  # flat mirror in the home plane
    )
    identity: LineMap = mv.rotor() >> Line
    subject: Point = (mv.x + mv.w * 1.0) ^ (mv.y - mv.w * 0.5)      # object at (-1, 0.5)
    fan: Line = subject & ((mv.x - mv.w * 1.0) ^ (mv.y - mv.w * np.linspace(-0.6, 0.6, 7)))

    def scenes():
        """Per frame: the element planes, the bundle after each leg, and the composed train."""
        for t in np.linspace(0.0, 2 * np.pi, 72, endpoint=False):
            motors = (
                (mv.wx * (-(1.0 + 0.3 * np.sin(t)) / 2)).exp() * (mv.xy * (0.3 * np.sin(2 * t) / 2)).exp(),   # slides and tilts
                (mv.wx * (-1.9 / 2)).exp(),                                                                  # fixed
                (mv.wx * (-2.2 / 2)).exp(),                                                                  # fixed
                (mv.wx * (-3.2 / 2)).exp() * (mv.xy * ((np.pi / 4 + 0.1 * np.cos(t)) / 2)).exp(),            # rocks about its pivot
            )
            rays, planes, legs, train = fan, [], [fan], identity
            for motor, element in zip(motors, elements):
                placed: LineMap = motor >> element(motor << Line)  # the element conjugated into place
                rays: Line = placed(rays)                          # the bundle after this element
                train = placed(train)                              # the train so far, as one map
                planes.append(motor >> home)                       # where the element sits, for drawing
                legs.append(rays)
            back = train(fan)
            picture = back[0] ^ back[-1]
            yield planes, legs, train, picture

    states = list(scenes())
    fig = draw_lenses(rays, obj, plane_1, plane_2, out, image, parallel, lens_1, focused, focus, plot_path)
    draw_train(states, subject, animation_path)

    # --- checks: kernel-level assertions, deliberately outside the demonstration ----------
    # Every transformed ray passes through the image. A point's incidence with the lens plane
    # over its incidence with the line at infinity (its weight) is its signed distance, which
    # gives the thin lens equation and, for two lenses, Gullstrand's back focal distance.
    np.testing.assert_allclose((out ^ image).kernel, 0.0, atol=1e-12)
    d_obj = -(plane_1 & obj) / (mv.w & obj)
    d_img = (plane_1 & image) / (mv.w & image)
    np.testing.assert_allclose((1 / d_obj + 1 / d_img).kernel, 1 / focal_1, atol=1e-12)
    gap = (plane_1 - plane_2) & origin
    f_eff = 1 / (1 / focal_1 + 1 / focal_2 - gap / (focal_1 * focal_2))
    back_focal = f_eff * (focal_1 - gap) / focal_1
    np.testing.assert_allclose(((plane_2 & focus) / (mv.w & focus)).kernel, back_focal.kernel, atol=1e-12)
    np.testing.assert_allclose((mv.y ^ focus).kernel, 0.0, atol=1e-12)
    for planes, legs, train, picture in states:
        np.testing.assert_allclose((train(legs[0]) ^ picture).kernel, 0.0, atol=1e-10)
        np.testing.assert_allclose((train(legs[0]) - legs[-1]).kernel, 0.0, atol=1e-10)
    return fig


if __name__ == "__main__":
    main()
