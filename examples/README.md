# Examples

| Topic | Examples |
| --- | --- |
| Geometry | [projection](geometry/projection/), [scenegraph](geometry/scenegraph/), [fitting](geometry/fitting/), [registration](geometry/registration/), [quadric error metrics (QEM)](geometry/qem/), [epipolar geometry & 3D reconstruction](geometry/epipolar/), [multi-view cone reconstruction & bundle adjustment](geometry/multiview/), [camera pose by gradient descent (JAX)](geometry/camera_fit.py), [Kalman pose filter](geometry/kalman/), [skinning](geometry/skinning/), [curvature of quadric surfaces](geometry/surface_curvature/), [uncertainty of a held pose](geometry/pose_diffusion/), [the most likely poses of a lap, by belief propagation](geometry/belief_propagation/) |
| Quadrics | [Projective quadrics](quadrics/quadrics/), [Gaussian fit and 1σ quadric](quadrics/gaussian/), [spherical quadrics](quadrics/spherical_quadrics/), [CGA spherical quadrics](quadrics/cga_spherical_quadrics/), [sphere rendering](quadrics/conformal_elliptical/), [collision](quadrics/quadric_collision/), [Cayley–Klein geometry](quadrics/cayley_klein/), [dynamics on the 2- and 3-sphere](quadrics/elliptic_physics/), [rendering on the 3-sphere](quadrics/s3_raytracer/) |
| Mechanics | [Inertia](mechanics/inertia.py), [simplex inertia](mechanics/simplex.py), [stiffness and normal modes](mechanics/modes/), [tennis racket instability](mechanics/tennis_racket/), [Lie integrators](mechanics/lie_integrators.py), [rigid body chains (XPBD)](mechanics/xpbd/), [robot arm kinematics](mechanics/robot_arm/), [velocity and force ellipsoids of an arm](mechanics/manipulability/), [elastic waves and phonon focusing in crystals](mechanics/crystal_waves/), [a spinning top on a bowl](mechanics/spinning_top/) |
| Relativity | [Distributed impulses, the ladder paradox and Bell's spaceships](relativity/impulse/), [curvature and gravitational waves](relativity/curvature/), [the Dirac electron as a map](relativity/dirac/), [gravitational lensing: one star, several images](relativity/gravitational_lensing/) |
| Quantum | [magnetic resonance: nutation, line shapes and spin echoes](quantum/magnetic_resonance/), [graphene: Dirac cones, pseudospin and the Berry phase](quantum/graphene/), [two spins: exchange, entanglement and Bell's inequality](quantum/two_spins/), [quantum process tomography: learn a noisy gate](quantum/process_tomography/) |
| Electromagnetism | [Maxwell maps](electromagnetism/maxwell.py), [constitutive maps](electromagnetism/constitutive/), [second-harmonic generation: a crystal that doubles the frequency](electromagnetism/second_harmonic/) |
| Optics | [Thin lenses as maps on lines](optics/thin_lens/), [zoom camera with depth of field](optics/lens_camera/) |
| Symmetry | [Heat conduction, a flywheel and a crystal lattice](mechanics/symmetry/) |
| Conformal geometry | [Ray tracing Dupin cyclides on the 3-sphere](geometry/cyclides/) |
| Mathematics | [the Hopf fibration: spinors sorted by direction](math/hopf/), [the spin groups of every signature up to six dimensions](math/spin_groups/), [a six-dimensional rotation as three independent turns](math/invariant_decomposition.py), [the two lines meeting four lines, and the Klein quadric](math/klein_quadric/), [the conic through five points, and Pascal's theorem](math/pascal/), [Poncelet's porism: polygons between two conics](math/poncelet/) |
| Benchmarks | [Motor map timings](../benchmarks/motor_map.py) |

Run an example from the repository root with `python -m examples.<topic>.<name>.scenarios`.

Notebooks sit next to their examples. Open one in Jupyter from inside the repository, or use the Open in Colab badge at its top.
