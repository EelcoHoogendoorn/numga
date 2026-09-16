"""A curvature map with only zero eigenvalues, and the tides it produces.

In a vacuum plane wave, curvature is a nonzero bivector-to-bivector extensor
whose composition with itself vanishes. Its image consists of null bivectors
containing the wave's propagation direction; that image lies in its kernel.
An indefinite metric allows a self-adjoint map to have this structure. All
six eigenvalues vanish, although the map and its physical effects do not.

Fix an observer. Wedge a neighbouring particle's separation with the observer's
velocity to form a spacetime ribbon, apply curvature, then let the resulting
bivector act on that velocity. Leaving separation open gives the tidal map:
separation -> relative acceleration. This observer binding is a composition of
different maps, not a similarity transformation. Its spatial eigenvalues can
therefore be positive, negative and zero even though curvature is nilpotent.

The upper panels depict two applications of the SAME local curvature map at
ONE event. They do not depict successive time steps. The lower panels evolve
freely falling particles in the central observer's local inertial frame, using
the tidal maps of plus, cross and circularly polarized wave packets. Coloured
beads distinguish a rotating deformation pattern from rigid rotation.

This is first-order geodesic deviation for a weak wave and a detector much
smaller than its wavelength. Acceleration acts on each bead's unperturbed
separation. A smooth strain packet and its first derivative vanish at both
ends, so the initially stationary ring returns to rest to first order. No
spring, damping or endpoint reset is applied. Only the displayed displacement
and acceleration are magnified; nonlinear memory is outside this model.
Units have c = 1; the detector radius is 0.01 and the carrier wavelength is 2.

The full curvature is assembled from null-bivector dyads. A rotor constructs
the cross polarization from plus; quarter-cycle phase separation gives the
circular case. Geometry stays in GA, with coefficient work confined to the
waveform, numerical integration and plot read-out helpers.

References:
  Coley & Hervik, "Higher dimensional bivectors and classification of the Weyl
  operator", CQG 27 (2010) 015002, Sec. 4.5 and Appendix B (type N):
  https://arxiv.org/pdf/0909.1160#page=11
  Tong, General Relativity, Sec. 5.2.2, "Bobbing on the Waves":
  https://www.damtp.cam.ac.uk/user/tong/gr/grhtml/S5.html#S5.S2.SS2

Pass animation_path to main to also save the wave packet as a GIF.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples import PLOT_DIR
from examples.relativity.curvature_plumbing import (
    STA,
    Bivector,
    Vector,
    detector_ring,
    draw_curvature,
    integrate_acceleration,
    mv,
    plot_data,
    save_animation,
    t,
    wave_packet,
    x,
    y,
    z,
)

# Map types (GATypes) read output <= inputs:
Curvature = STA.gatype((Bivector, Bivector))       # curvature bivector <= area bivector
Tidal = STA.gatype((Vector, Vector))               # relative acceleration <= separation


def main(plot_path: str = str(PLOT_DIR / "curvature.png"), animation_path: str = "") -> plt.Figure:
    """Construct wave curvature, bind the observer and integrate the detector response."""

    # -----------------------------------------------------------------------
    # 1. Curvature from null dyads
    # -----------------------------------------------------------------------
    # For a wave along the null direction k, the null bivectors k ∧ x and k ∧ y are mutually
    # orthogonal, so their dyads compose to zero and the map is nilpotent. Opposite weights
    # cancel the Ricci contraction: this is vacuum curvature with all eigenvalues zero. The
    # cross polarization is the plus polarization conjugated by an eighth-turn rotor.
    k = t + z
    nx, ny = k.wedge(x), k.wedge(y)
    plus: Curvature = nx * (nx | Bivector) - ny * (ny | Bivector)
    rotor = (mv.xy * (np.pi / 8)).exp()
    cross: Curvature = rotor >> plus(rotor << Bivector)
    np.testing.assert_allclose(plus(plus).kernel, 0.0, atol=1e-14)
    assert np.abs(plus.kernel).max() > 0.0

    # -----------------------------------------------------------------------
    # 2. A wave packet in three polarizations
    # -----------------------------------------------------------------------
    # The weak-wave curvature is R_{0i0j} = -½ ḧ_ij, so the strain's second derivative
    # scales the unit maps; the circular case adds the cross polarization a quarter cycle
    # behind. The result is one curvature batch over (time, polarization).
    time = np.linspace(0.0, 6.0, 1201)
    _, second = wave_packet(time)
    cosine, sine = second[:, 0], second[:, 1]
    plus_wave, cross_wave = plus * cosine, cross * cosine
    waves: Curvature = -0.5 * stack((plus_wave, cross_wave, plus_wave + cross * sine), axis=1)

    # -----------------------------------------------------------------------
    # 3. The tidal map: bind the observer twice, leave the separation open
    # -----------------------------------------------------------------------
    # Wedge the separation with the observer's velocity into a spacetime ribbon, apply the
    # curvature, and let the resulting bivector act on that velocity. With the separation
    # open this is a map from separation to relative acceleration. It composes different
    # maps rather than conjugating one, which is why its eigenvalues can be nonzero even
    # though the curvature's are all zero.
    response: Tidal = waves(t.wedge(Vector)).commutator(t)

    # -----------------------------------------------------------------------
    # 4. Integrate the detector
    # -----------------------------------------------------------------------
    reference: Vector = detector_ring() * 0.01
    acceleration: Vector = response[:, :, None](reference)
    displacement: Vector = integrate_acceleration(time, acceleration)

    # -----------------------------------------------------------------------
    # 5. Draw
    # -----------------------------------------------------------------------
    data = plot_data(time, reference, displacement, acceleration, plus, amplification=4000)
    figure = draw_curvature(data, plot_path)
    print(f"Figure saved to {plot_path}")
    if animation_path:
        save_animation(data, animation_path)
        print(f"Animation saved to {animation_path}")
    return figure


if __name__ == "__main__":
    main()
    plt.show()
