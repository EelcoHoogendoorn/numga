"""The Wu experiment, its mirror image, and the mirrored apparatus as nature runs it."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.orientation.wu_experiment import core
from examples.orientation.wu_experiment.core import Direction, Mirror, Scalar, Spin, mv

# The decay's asymmetry: minus one for cobalt-60, times the electrons' speed over light's, taken as 0.6.
decay_asymmetry = -0.6
# The directions around the plane holding the spin's axis, the last closing the loop on the first.
n_directions = 181


# --- math -----------------------------------------------------------------------------
def experiment() -> tuple[Spin, Direction, Scalar, core.Asymmetry, Mirror]:
    """The emission around the apparatus `[cases, directions]`, the coil's spin for each case, and the
    directions: the apparatus; its mirror image, coil and emission alike; and the mirrored apparatus
    under the decay law as it is. Also the decay's asymmetry and the mirror.

    The current circulates counterclockwise seen from above, and the electrons leave mostly downward,
    against the spin. In the mirror, held upright beside the apparatus, the current circulates the other
    way and the electrons still leave mostly downward: in the mirror image they leave along the spin.
    Built for real, the mirrored apparatus sends them upward, against its spin, as the original does.
    """
    # Measured against the lab's handedness, the basis volume x ^ y ^ z.
    asymmetry = mv(core.Asymmetry, [decay_asymmetry])                             # [] Asymmetry-
    # The patch the coil's current circulates around, counterclockwise seen from above.
    spin = mv.xy.dual()                                                           # [] Spin-
    # The plane x = 0, holding the coil's axis.
    mirror = mv.x                                                                 # [] Mirror+
    angles = np.linspace(0, 2 * np.pi, n_directions)
    # Directions turned from z toward x, around the plane that holds the spin's axis.
    directions = (mv.zx * (-angles / 2)).exp() >> mv.z.dual()                     # [directions] Direction-

    # The rate along each direction departs from that of unpolarized nuclei by the asymmetry times the spin's
    # pairing with the direction.
    original = 1 + asymmetry * (spin & directions)                                # [directions] Scalar+
    # The mirror image: what the mirror shows along each direction is what the apparatus sends along its image.
    image = 1 + asymmetry * (spin & (mirror >> directions))                       # [directions] Scalar+
    # The mirrored apparatus, built for real: its coil is the mirror image, the decay law the same.
    built = 1 + asymmetry * ((mirror >> spin) & directions)                       # [directions] Scalar+
    spins = stack([spin, mirror >> spin, mirror >> spin])                         # [cases] Spin-
    return spins, directions, stack([original, image, built]), asymmetry, mirror


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.orientation.wu_experiment import render

    spins, directions, rates, asymmetry, mirror = experiment()
    save_figure(render.draw_experiment(spins, directions, rates), "wu_experiment")

    # checks
    spin = spins[0]
    _, image, built = rates
    # With the asymmetry mirrored along, the mirrored apparatus would give the mirror image.
    mirrored_law_rates = 1 + (mirror >> asymmetry) * ((mirror >> spin) & directions)  # [directions] Scalar+
    np.testing.assert_allclose((mirrored_law_rates - image).to_array(), 0, atol=1e-12)
    # As it is, the mirrored apparatus sends electrons where the mirror image does not.
    np.testing.assert_allclose((built + image - 2).to_array(), 0, atol=1e-12)
    # The original sends them mostly against its spin, downward; the mirrored apparatus upward.
    down_and_up = stack([-mv.z.dual(), mv.z.dual()])                              # [2] Direction-
    original_down, original_up = (1 + asymmetry * (spin & down_and_up)).to_array()
    built_down, built_up = (1 + asymmetry * ((mirror >> spin) & down_and_up)).to_array()
    assert original_down > original_up and built_up > built_down


if __name__ == "__main__":
    main()
