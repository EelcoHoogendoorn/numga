"""Which responses survive averaging over an object's rotation symmetries?

heat_conduction(): a positive conductivity and a plot of its heat-flow ellipsoids.
flywheel(): a flywheel with three arms, assembled by adding their rotated inertia maps.
crystal_lattice(): the same cubic bond orbits give isotropic conduction and anisotropic elasticity.
main() runs all three physical examples.

Run from rewrite/ with PYTHONPATH=src:. python -m examples.sketches.symmetry.
Crystal-symmetry motivation: https://dictionary.iucr.org/Neumann%27s_principle
"""

from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from numga import Algebra, Extensor, NumpyContext
from numga.algebras import PGA3D
from examples import PLOT_DIR

# Conductivity acts on Euclidean driving-field and heat-flow vectors.
ga = Algebra("x+y+z+")
ctx = NumpyContext(ga)
mv = ctx.multivector
Vector = ga.gatype.vector()
Rotor = ga.gatype.rotor()

# Mechanics uses projective points and a momentum response to rigid-body motion.
pga_mv = NumpyContext(PGA3D).multivector
Point = PGA3D.gatype.antivector()
Bivector = PGA3D.gatype.bivector()
AntiBivector = PGA3D.gatype.antibivector()
Scalar = PGA3D.gatype.scalar()


# --- plumbing -------------------------------------------------------------------------
def closure(generators: list[Rotor]) -> Rotor:
    """All rotations reachable by composing the given ones, each rotor counted once (g and -g are the same rotation)."""
    def key(rotor: Rotor) -> tuple[float, ...]:
        k = np.round(rotor.kernel, 6) + 0.0
        return tuple(k if k[np.argmax(np.abs(k) > 1e-9)] > 0 else -k)

    identity = mv.rotor()
    elements, seen, frontier = [identity], {key(identity)}, [identity]
    while frontier:
        fresh = [g * h for g in frontier for h in generators]
        frontier = []
        for element in fresh:
            if key(element) not in seen:
                seen.add(key(element)); elements.append(element); frontier.append(element)
    return Extensor.stack(elements)


def directions() -> Vector:
    """A sphere of unit driving fields, sampled for drawing."""
    longitude = np.linspace(0, 2 * np.pi, 65)[None, :]
    latitude = np.linspace(-np.pi / 2, np.pi / 2, 33)[:, None]
    return (mv.x * (np.cos(latitude) * np.cos(longitude))
            + mv.y * (np.cos(latitude) * np.sin(longitude))
            + mv.z * np.broadcast_to(np.sin(latitude), (33, 65)))


def arm_samples() -> Point:
    """Equal-mass samples of a thin rectangular flywheel arm attached to a massless hub."""
    translations = (pga_mv.xw * np.linspace(0.25, 2.2, 24)[:, None]
                    + pga_mv.yw * np.linspace(-0.14, 0.14, 5)[None, :]) / 2
    return translations.exp() >> pga_mv.zyx


def lattice_samples() -> tuple[Point, Point, Point]:
    """PGA sites of a simple-cubic lattice, with the central site's two neighbour shells."""
    i, j, k = np.indices((3, 3, 3)) - 1
    sites = pga_mv.zyx + pga_mv.yzw * i + pga_mv.zxw * j + pga_mv.xyw * k
    shell = i * i + j * j + k * k
    return sites.reshape(-1), sites[shell == 1], sites[shell == 2]


def save(fig: plt.Figure, name: str) -> plt.Figure:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    fig.savefig(path)
    print(f"Figure saved to {path}")
    return fig


def draw(labels: list[str], surfaces: Vector, driving: Vector, fluxes: Vector) -> plt.Figure:
    """Draw the image of the unit input sphere, and one common input/output pair."""
    coordinates = surfaces.cast(Vector.output_subspace).kernel
    flow = fluxes.cast(Vector.output_subspace).kernel
    direction = driving.cast(Vector.output_subspace).kernel
    radius = np.linalg.norm(coordinates, axis=-1).max() * 1.05
    arrow = direction * radius * 0.85
    angles = np.degrees(np.arccos(np.clip(flow @ direction / np.linalg.norm(flow, axis=-1), -1, 1)))

    fig = plt.figure(figsize=(15, 5.6), dpi=140)
    fig.suptitle("What heat flow does crystal symmetry allow?", fontsize=19, y=0.97)
    fig.text(0.5, 0.895, "Each surface is K applied to every unit driving field: a heat-flow ellipsoid.",
             ha="center", fontsize=12, color="#46505a")
    for i, (label, surface, flux, angle) in enumerate(zip(labels, coordinates, flow, angles)):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d", computed_zorder=False)
        ax.plot_surface(*np.moveaxis(surface, -1, 0), color="#438ab0", alpha=0.4,
                        linewidth=0.15, edgecolor="#356781", rstride=2, cstride=2)
        ax.quiver(0, 0, 0, *arrow, color="#262b32", linewidth=2, arrow_length_ratio=0.12, zorder=3)
        ax.quiver(0, 0, 0, *flux, color="#d44e1a", linewidth=3, arrow_length_ratio=0.16, zorder=4)
        ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius),
               xlabel="flow x", ylabel="flow y", zlabel="flow z")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=25, azim=-120)
        ax.set_xticks([-4, 0, 4])
        ax.set_yticks([-4, 0, 4])
        ax.set_zticks([-4, 0, 4])
        ax.tick_params(labelsize=8, pad=0)
        ax.set_title(label, fontsize=11, pad=7)
        ax.text2D(0.5, -0.16, f"Flow deflection: {angle:.1f}°", transform=ax.transAxes,
                  ha="center", fontsize=11, color="#a53c14")
    fig.legend([Line2D([], [], color="#262b32", lw=2), Line2D([], [], color="#d44e1a", lw=3)],
               ["Same driving direction (arrow length arbitrary)", "Resulting heat flow"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.06))
    fig.text(0.5, 0.025, "Symmetry constrains the response: 6 → 3 → 2 → 1 independent components.",
             ha="center", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.975, bottom=0.22, top=0.78, wspace=0.03)
    return save(fig, "symmetry_conductivity")


def draw_flywheel(arms: Point, angles: np.ndarray, moments: Scalar) -> plt.Figure:
    """Show the point masses and moment of inertia versus the in-plane axis direction."""
    homogeneous = arms.cast(Point.output_subspace).kernel
    coordinates = homogeneous[..., :3] / homogeneous[..., 3:]
    curves = moments.kernel[..., 0]
    fig = plt.figure(figsize=(11, 5.6), dpi=140)
    fig.suptitle("Threefold shape, axially symmetric inertia", fontsize=18, y=0.97)
    ax = fig.add_subplot(1, 2, 1)
    for points, color in zip(coordinates, ("#d44e1a", "#438ab0", "#438ab0")):
        points = points.reshape(-1, 3)
        ax.scatter(points[:, 0], points[:, 1], s=12, color=color)
    ax.scatter([0], [0], s=30, color="#262b32")
    ax.axhline(0, color="#adb4bb", lw=0.7, zorder=0)
    ax.axvline(0, color="#adb4bb", lw=0.7, zorder=0)
    ax.set(aspect="equal", xlabel="x", ylabel="y", title="Three unit-mass arms, 120° apart")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(1, 2, 2, projection="polar")
    for curve, color, label in zip(curves, ("#d44e1a", "#438ab0"), ("One arm", "Whole flywheel")):
        ax.plot(angles, curve, color=color, lw=2.5, label=label)
    ax.set_title("Moment about an axis in the wheel's plane", pad=24)
    ax.set_rlabel_position(65)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.text(0.5, 0.025, "Polar angle = axis direction; radius = moment of inertia. All axes pass through the hub.",
             ha="center", fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.79, bottom=0.2, wspace=0.32)
    return save(fig, "symmetry_flywheel")


def draw_crystal(sites: Point, axial: Point, diagonal: Point,
                 conduction: Vector, stiffness: Vector) -> plt.Figure:
    """Plot the lattice bonds and radial directional responses, each in its own units."""
    fig = plt.figure(figsize=(15, 5.6), dpi=140)
    fig.suptitle("One cubic symmetry group, two different kinds of response", fontsize=18, y=0.97)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    homogeneous = sites.cast(Point.output_subspace).kernel
    coordinates = homogeneous[..., :3] / homogeneous[..., 3:]
    ax.scatter(*coordinates.T, s=30, color="#7e8b98", alpha=0.6)
    for neighbours, color, label in ((axial, "#d44e1a", "6 axial neighbours"),
                                      (diagonal, "#438ab0", "12 face-diagonal neighbours")):
        homogeneous = neighbours.cast(Point.output_subspace).kernel
        coordinates = homogeneous[..., :3] / homogeneous[..., 3:]
        for end in coordinates:
            ax.plot([0, end[0]], [0, end[1]], [0, end[2]], color=color, lw=1.5)
        ax.scatter(*coordinates.T, s=45, color=color, label=label)
    ax.scatter([0], [0], [0], s=70, color="#262b32")
    ax.set(title="Two seed bonds generate both families", xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=23, azim=-55)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), frameon=False, fontsize=10)

    for panel, surface, color, title, caption in (
        (2, conduction, "#438ab0", "Conductivity: isotropic", "Radius = d · K(d) = 2/3 in every direction"),
        (3, stiffness, "#d44e1a", "Elastic stiffness: cubic anisotropy", "Radius = C(d, d, d, d)\n1/2 on axes; 1/3 on body diagonals"),
    ):
        coordinates = surface.cast(Vector.output_subspace).kernel
        limit = np.linalg.norm(coordinates, axis=-1).max() * 1.05
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        ax.plot_surface(*np.moveaxis(coordinates, -1, 0), color=color,
                        linewidth=0.15, edgecolor="#48505a", rstride=1, cstride=1, alpha=0.85)
        ax.set(title=title, xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
               xlabel="x", ylabel="y", zlabel="z", xticks=[], yticks=[], zticks=[])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=23, azim=-55)
        ax.text2D(0.5, -0.14, caption, transform=ax.transAxes, ha="center", fontsize=10)
    fig.text(0.5, 0.035, "Averaging over the same 24 cube rotations makes conductivity isotropic while preserving cubic elastic anisotropy.",
             ha="center", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.22, wspace=0.05)
    return save(fig, "symmetry_crystal_lattice")


# --- math -----------------------------------------------------------------------------
def heat_conduction() -> plt.Figure:
    """Which heat-conduction responses are compatible with a crystal's symmetry?

    Conductivity maps the negative temperature gradient to heat flow. In an anisotropic
    material these vectors need not be parallel. Start with a positive conductivity,
    rotate the whole response through each symmetry, and average. This Reynolds
    projection gives the closest invariant response in the Frobenius norm, preserving
    positive definiteness. Symmetry constrains the response, not its numerical gains.

    Mapping unit driving fields gives an ellipsoid of heat-flow vectors. Half turns
    align its axes; quarter turns about z make a spheroid; cube rotations make a sphere.
    This isotropy follows for rank-two conductivity; rank-four cubic elasticity can
    still be anisotropic.
    """
    # Inputs: a positive conductivity in a tilted principal frame, and candidate symmetries.
    pose = (mv.xy * 0.3 + mv.yz * 0.16).exp()
    axes = pose >> Extensor.stack((mv.x, mv.y, mv.z))
    gains = mv.scalar([[4.5], [1.5], [0.6]])
    driving = (mv.x + mv.y + mv.z).normalized()
    sphere = directions()
    cube = closure([(mv.xy * (np.pi / 4)).exp(), (mv.yz * (np.pi / 4)).exp()])
    groups = {
        "Half-turns about x and z\n3 independent components": closure([(mv.xy * (np.pi / 2)).exp(), (mv.yz * (np.pi / 2)).exp()]),
        "Quarter-turns about z\n2 independent components": closure([(mv.xy * (np.pi / 4)).exp()]),
        "Rotations of a cube\n1 independent component": cube,
    }

    # Each principal axis measures one component of the driving field and contributes
    # heat flow along that axis. Positive gains make this a passive conductivity.
    conductivity = (axes * (axes | Vector) * gains).sum(axis=0)
    responses = [conductivity]
    for group in groups.values():
        # Pull the input into each rotated frame, apply K, and rotate the output back.
        # The mean is unchanged by any rotation in the group: those rotations merely
        # permute the terms. This projects a measured response onto the allowed ones.
        invariant = (group >> Vector)(conductivity(group << Vector)).mean(axis=0)
        responses.append(invariant)

    # Half turns eliminate off-diagonal coupling; quarter turns also equate x and y.
    # For this rank-two response, fourfold symmetry already implies axial symmetry.
    # Cube rotations equate all three axes, so heat always flows along the driving field.
    responses = Extensor.stack(responses)
    surfaces = responses[:, None, None](sphere[None, :, :])
    fluxes = responses(driving)

    return draw(["Measured candidate\n6 independent components", *groups], surfaces, driving, fluxes)


def flywheel() -> plt.Figure:
    """Assemble a flywheel by adding rotated copies of one arm's inertia.

    Each arm has unit mass. Threefold symmetry puts the center of mass at the hub
    and makes all transverse moments equal, even though the shape has only three arms.
    Rotation about z still has a different moment: for this thin planar object it is
    twice the transverse moment. The plotted axes pass through the hub, not each arm's
    individual center of mass.

    Mass locations are PGA points (antivectors). Inertia maps bivector rigid-body
    velocities to antibivector momenta, including translation as well as rotation.
    """
    arm = arm_samples()
    rotations = (pga_mv.xy * (np.arange(3) * np.pi / 3)).exp()
    angles = np.linspace(0, 2 * np.pi, 181)
    probes = (pga_mv.xy * (-angles / 2)).exp() >> pga_mv.yz

    # The commutator gives each point's velocity under the open rigid-motion slot.
    # Join point and velocity to get momentum, then average for a unit-mass arm.
    arm_inertia = (arm & arm.commutator(Bivector)).mean()

    # Rotate that entire response and add the three arms. Unlike conductivity's mean,
    # this sum assembles actual masses: the finished flywheel has mass three.
    inertia = (rotations >> AntiBivector)(arm_inertia(rotations << Bivector)).sum(axis=0)
    arms = rotations[:, None, None] >> arm[None, :, :]

    # Unit bivectors describe rotation about axes in the wheel's plane. The regressive
    # pairing of motion and momentum gives the moment about each chosen axis.
    arm_moments = probes & arm_inertia(probes)
    wheel_moments = probes & inertia(probes)
    return draw_flywheel(arms, angles, Extensor.stack((arm_moments, wheel_moments)))


def crystal_lattice() -> plt.Figure:
    """Average microscopic bond responses over a crystal's cubic point group.

    A simple-cubic lattice has axial and face-diagonal bond families. Starting with
    x and normalized(x+y), the 24 cube rotations generate every direction in each
    family. Every member has the same weight, so another cube rotation just permutes
    the terms in the average. This is the microscopic meaning of the group projection.

    Give the two families equal total response weights. For the elastic model, take
    unstressed central springs with lattice spacing 1, axial stiffness 1/3, and
    face-diagonal stiffness 1/12. Bond counts and squared lengths give weight 1 to
    each family. Diagonal springs provide shear stiffness. This particular model has
    C12 = C44; a general cubic crystal need not have that extra relation.

    A unit bond n measures a thermal gradient through n·g. For imposed strain
    strain(v) = s*d*(d·v), its fractional extension is s*(n·d)^2. Squaring to get
    spring energy gives four factors of n·d, hence a scalar form with four vector
    slots. The plotted stiffness is C(d,d,d,d), with energy density s^2*C/2.
    It holds transverse strain fixed; it is not the relaxed Young's modulus.

    Cubic elasticity background: https://www.ctcms.nist.gov/oof/oof1/Manual/node152.html
    """
    seeds = Extensor.stack((mv.x, (mv.x + mv.y).normalized()))
    cube = closure([(mv.xy * (np.pi / 4)).exp(), (mv.yz * (np.pi / 4)).exp()])
    sites, axial, diagonal = lattice_samples()
    sphere = directions()

    # Batch axes are [24 rotations, 2 bond families]. These are free bond directions;
    # the actual lattice sites above are projective points.
    bonds = cube[:, None] >> seeds[None, :]
    projection = bonds | Vector                         # Scalar <- Vector

    # Average over the group, then add the two families. No direction is privileged
    # within either orbit. Conductivity measures the gradient and returns flow along n.
    conductivity = (bonds * projection).mean(axis=0).sum(axis=0)

    # Four independent slots encode the spring stiffness. The same group average
    # leaves cubic structure here, although it made the vector response isotropic.
    elasticity = (projection * projection * projection * projection).mean(axis=0).sum(axis=0)

    # Bind a unit direction into every slot to probe imposed uniaxial strain.
    # The response values become radii, making isotropy versus cubic lobes visible.
    conduction_surface = sphere * (sphere | conductivity(sphere))
    stiffness_surface = sphere * elasticity(sphere, sphere, sphere, sphere)
    return draw_crystal(sites, axial, diagonal, conduction_surface, stiffness_surface)


def main() -> None:
    heat_conduction()
    flywheel()
    crystal_lattice()


if __name__ == "__main__":
    main()
    plt.show()
