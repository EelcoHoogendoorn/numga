"""The mathematics of every example imports without the plotting stack."""

import subprocess
import sys

PROBE = """
from numga.algebra import Algebra
from numga.algebras import PGA2D, Spherical3D
from examples import instantiate
import sys
import examples.electromagnetism.constitutive.core
import examples.electromagnetism.second_harmonic.core
import examples.electromagnetism.second_harmonic.scenarios
import examples.geometry.belief_propagation.scenarios
import examples.geometry.cyclides.core
import examples.geometry.epipolar.core
import examples.geometry.fitting.core
import examples.geometry.fitting.scenarios
import examples.geometry.kalman.core
import examples.geometry.kalman.scenarios
import examples.geometry.pose_diffusion.core
import examples.geometry.pose_diffusion.scenarios
import examples.geometry.projection.core
import examples.geometry.qem.core
import examples.geometry.registration.core
import examples.geometry.registration.scenarios
import examples.geometry.scenegraph.core
import examples.geometry.surface_curvature.core
import examples.math.hopf.core
import examples.math.hopf.scenarios
import examples.math.klein_quadric.core
import examples.math.klein_quadric.scenarios
import examples.math.pascal.core
import examples.math.pascal.scenarios
import examples.math.poncelet.core
import examples.math.poncelet.scenarios
import examples.math.spin_groups.core
import examples.math.spin_groups.scenarios
import examples.mechanics.crystal_waves.core
import examples.mechanics.crystal_waves.scenarios
import examples.mechanics.manipulability.core
import examples.mechanics.modes.core
import examples.mechanics.robot_arm.core
import examples.mechanics.spinning_top.core
import examples.mechanics.spinning_top.scenarios
import examples.mechanics.symmetry.core
import examples.mechanics.xpbd.core
import examples.optics.lens_camera.core
import examples.optics.thin_lens.core
import examples.quantum.graphene.core
import examples.quantum.graphene.scenarios
import examples.quantum.magnetic_resonance.core
import examples.quantum.magnetic_resonance.scenarios
import examples.quantum.process_tomography.core
import examples.quantum.process_tomography.scenarios
import examples.quantum.two_spins.core
import examples.quantum.two_spins.scenarios
import examples.quadrics.cayley_klein.core
import examples.quadrics.cga_spherical_quadrics.core
import examples.quadrics.conformal_elliptical.core
import examples.quadrics.gaussian.core
import examples.quadrics.quadric_collision.core
import examples.quadrics.quadrics.core
import examples.quadrics.s3_raytracer.core
import examples.quadrics.spherical_quadrics.core
import examples.relativity.curvature.core
import examples.relativity.dirac.core
import examples.relativity.dirac.scenarios
import examples.relativity.gravitational_lensing.core
import examples.relativity.gravitational_lensing.scenarios
import examples.relativity.impulse.core
instantiate('examples.geometry.belief_propagation.core', PGA2D)
instantiate('examples.geometry.multiview.core', PGA2D)
instantiate('examples.mechanics.tennis_racket.core', Algebra.from_pqr(3, 0, 0))
instantiate('examples.quadrics.elliptic_physics.core', Spherical3D)
print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])
"""


def test_mathematics_does_not_import_plotting():
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
