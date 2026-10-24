# Metric and Exterior Structure in the Examples

This audit covers the **145 entries** in the [example catalog](../examples/README.md), excluding the benchmark. Kerr's metric and gauge notebooks count separately because the catalog lists them separately; the four reflection notebooks count together because the catalog groups them. It is a snapshot of the working checkout on **24 October 2026**, based on HEAD `0dd92ce` with local changes present.

| Classification | Examples | Share |
| --- | ---: | ---: |
| Clearly use both | **134** | **92.4%** |
| Exterior-only mathematical core | **7** | **4.8%** |
| Metric-only mathematical core | **1** | **0.7%** |
| Boundary cases described below | **3** | **2.1%** |
| Total | **145** | **100%** |

There are **eight clear one-sided cores** under the counting rule below. Including the three boundary cases in the combined category gives 137/145, or 94.5%. The comparison uses the rounded statement **roughly 90–95% of the examples in this collection use both sides of the algebra**. This leaves room for classification choices and changes to the collection while retaining the practical finding: the combined case is overwhelmingly usual here.

## Counting rule

The unit is the mathematical example, including calculations in `scenarios.py` or a notebook and relevant imported mathematical helpers. Code that only positions a drawing, samples initial data, rescales homogeneous representatives or checks an answer does not determine its classification. Rendering is inspected where it contains part of the advertised construction. A short kernel is not taken to stand for the whole example: the Wu emission formula is a dot product, but the example also compares the reflection of a spin plane with that of an intrinsic direction.

The metric side includes lengths, angles, norms and metric contractions, as well as physical metrics supplied as bilinear forms or constitutive maps. The exterior side includes joins, meets, alternating products, incidence and complementary pairings, oriented areas and volumes, outermorphisms and determinants. General rotor, spinor and Clifford products use the combined algebra even when no explicit wedge appears. Scalar multiplication and scalar squares do not by themselves count as metric use. A method named `dual`, `norm` or `inverse` is checked for what it does in the calculation rather than classified from its name.

Sandwiches count as use of the combined geometric product. In a vector reflection, the first product contains scalar and bivector parts; the second returns the result to the vector space. An equivalent dot-product formula describes the same final map, but the sandwich construction uses both parts of the algebra. The induced-dipole and E-series examples are therefore counted as using both. The remaining boundaries concern complementary scalar evaluation and geometric construction outside the main solver.

The source review used the catalog, mathematical introductions and operation sites across all 145 entries, then followed scenario calculations and helpers where the primary file did not settle the classification. The one-sided and boundary candidates received a closer review. Each row below records the construction supporting its classification; this is a manual mathematical classification, not a tally of `^`, `&` and `|` tokens.

## The one-sided cores

| Example | Classification | Why |
| --- | --- | --- |
| [Rational Bézier cone](../examples/geometry/bezier/core.py) | Exterior only | Weighted points, their control-triangle joins and a conic polarity. The unit sphere is a display of projective rays. |
| [Projective Loop subdivision](../examples/surfaces/projective_loop/core.py) | Exterior only | The subdivision itself is linear averaging; the larger example adds projective collineations and homogeneous readouts. The metric gap is a comparison diagnostic. |
| [Area transport](../examples/mechanics/area_transport/core.py) | Exterior only | Oriented patches, the outermorphism, determinant and flux pullback. The scalar norm in `cube` only extracts an orientation sign. |
| [Klein quadric](../examples/math/klein_quadric/core.py) | Exterior only | Incidence determines the transversals and ruled quadric. SVD, unit-norm lines and a nearest-point span choose numerical representatives; they do not supply the geometry of the answer. |
| [Pascal's theorem](../examples/math/pascal/core.py) | Exterior only | Conic pencils and joins/meets. A rotor supplies sampled headings for the illustration. |
| [Poncelet's porism](../examples/math/poncelet/core.py) | Exterior only | Conic incidence and tangent iteration. The implementation rescales directions by a norm and places ellipses by motors; those choices leave the projective construction unchanged. |
| [Eight-conic theorem](../examples/math/eight_conics/core.py) | Exterior only | Contact-line dyads complete the conic configuration. Euclidean line parameterization locates the displayed contact endpoints. |
| [Hanging chain](../examples/mechanics/hanging_chain/hanging_chain.ipynb) | Metric only | Spring lengths and gravitational dot products define the energy, gradient and Hessian. The initial arc happens to be sampled with a rotor. |

These are classifications of the mathematical core, not claims that the files contain no operation from the other side. Projective Loop is especially small: its subdivision kernel alone needs neither geometric product; its projective comparison supplies the exterior content counted here.

## The three boundary cases

| Example | One-sided reading | Combined reading |
| --- | --- | --- |
| [Gaussian fit](../examples/quadrics/gaussian/core.py) | Moments, covariance and Mahalanobis distance are a metric/statistical construction. | The formulation constructs a projective polarity and evaluates it through complementary plane–point pairings. There is no further join/meet construction. |
| [Cube-axis planes](../examples/quadrics/conformal_elliptical/core.py) | The core merely normalizes thirteen supplied vectors. | Its [drawing](../examples/quadrics/conformal_elliptical/render.py) uses plane–point incidence on the sphere; its checks additionally use rotors. |
| [Isospectral drums](../examples/math/isospectral_drums/scenarios.py) | Transplantation and vibration are scalar-field linear algebra with a metric Laplacian. | The [mesh construction](../examples/math/isospectral_drums/numerics.py) uses oriented two-dimensional determinants for winding and collinearity when identifying the boundary. |

Cayley–Klein geometry shows why metric structure also counts when supplied separately. Its core can avoid the built-in inner product, yet its absolute conic supplies distances and perpendicularity. It is counted as combining projective and metric geometry. Metric structure does not disappear when it moves into an extensor.

## Complete index

**B** means both; **E** exterior-only core; **M** metric-only core; **?** a boundary case. The source link opens the primary mathematical file or the catalog's notebook group. Scenario calculations are included where the reason names a construction performed there.

| # | Example/source | Class | Construction used for the classification |
| ---: | --- | :---: | --- |
| 1 | [geometry/projection](../examples/geometry/projection/core.py) | B | Join/meet cameras and epipolar incidence; motors place and relate the camera frames. |
| 2 | [geometry/scenegraph](../examples/geometry/scenegraph/core.py) | B | Lens incidence and ray intersections; articulated motors and rigid frame transformations. |
| 3 | [geometry/fitting](../examples/geometry/fitting/core.py) | B | Join residuals for fitting; squared residuals and metric normalization select the fit. |
| 4 | [geometry/registration](../examples/geometry/registration/core.py) | B | Motor and rotor alignment; scalar products and residual norms define its objective. |
| 5 | [geometry/three_point_pose](../examples/geometry/three_point_pose/core.py) | B | Lines and planes joined from markers; geometric ratios and normalized motors recover pose. |
| 6 | [geometry/bezier](../examples/geometry/bezier/core.py) | E | Control-triangle joins and conic polarity; no distance or angle enters the curve construction. |
| 7 | [geometry/sundial](../examples/geometry/sundial/core.py) | B | Shadow incidence; celestial turns and the angle-dependent seasonal cone. |
| 8 | [geometry/origami](../examples/geometry/origami/core.py) | B | Crease intersections and polygon incidence; angle bisectors and rigid fold motors. |
| 9 | [geometry/rubiks_cube](../examples/geometry/rubiks_cube/core.py) | B | Layer incidence and oriented quarter-turn rotors; the rotor product supplies their metric action. |
| 10 | [estimation/camera_fit](../examples/estimation/camera_fit.py) | B | Projective camera joins and meets; motor pose and metric reprojection error. |
| 11 | [estimation/kalman](../examples/estimation/kalman/core.py) | B | Twist/readout pairing and covariance; motor innovations and rigid pose propagation. |
| 12 | [estimation/epipolar](../examples/estimation/epipolar/core.py) | B | Coplanarity of sight rays; metric triangulation residuals and motor pose updates. |
| 13 | [estimation/multiview](../examples/estimation/multiview/core.py) | B | Projective sight cones and pullbacks; their uncertainty forms and rigid camera motion. |
| 14 | [estimation/odometry](../examples/estimation/odometry/core.py) | B | Twist/load pairings; motor-relative errors and quadratic uncertainty forms. |
| 15 | [estimation/pose_diffusion](../examples/estimation/pose_diffusion/core.py) | B | Complementary covariance pairings; turning dynamics and point motion through commutators. |
| 16 | [surfaces/qem](../examples/surfaces/qem/core.py) | B | Faces joined from mesh vertices; unit-plane distances define the error form. |
| 17 | [surfaces/skinning](../examples/surfaces/skinning/core.py) | B | Mixed-grade motor interpolation and normalization; metric radius preservation in the skin. |
| 18 | [surfaces/surface_curvature](../examples/surfaces/surface_curvature/core.py) | B | Quadric polarity and tangent restriction; principal curvature compares the second form with the metric. |
| 19 | [surfaces/arap](../examples/surfaces/arap/core.py) | B | Rotor-valued local fits; cotangent mesh geometry and metric edge mismatch. |
| 20 | [surfaces/projective_loop](../examples/surfaces/projective_loop/core.py) | E | Weighted linear subdivision and a projective collineation; distance appears only in the comparison diagnostic. |
| 21 | [surfaces/motor_loop](../examples/surfaces/motor_loop/core.py) | B | Averaging mixed-grade motors and returning to the motor group by normalization. |
| 22 | [surfaces/weighted_dual](../examples/surfaces/weighted_dual/core.py) | B | Hull faces and reciprocal incidence; spherical polarity and edge lengths. |
| 23 | [surfaces/koebe](../examples/surfaces/koebe/core.py) | B | Circle wedges and face incidence; tangency inner products and the sphere's metric. |
| 24 | [surfaces/reciprocal](../examples/surfaces/reciprocal/core.py) | B | Reciprocal frames and face incidence; slope jumps, perpendicularity and bar lengths. |
| 25 | [surfaces/spin_transformations](../examples/surfaces/spin_transformations/core.py) | B | Quaternion multiplication in the Dirac operator; oriented mesh geometry and area/energy weights. |
| 26 | [surfaces/dirac_waves](../examples/surfaces/dirac_waves/core.py) | B | Geometric multiplication of quaternion fields; the imported Dirac operator uses mesh edges and areas. |
| 27 | [surfaces/connections](../examples/surfaces/connections/core.py) | B | Rotor transport and its plane; metric normalization, surface angles and holonomy. |
| 28 | [surfaces/developability](../examples/surfaces/developability/core.py) | B | Oriented face joins; lengths, corner angles and the normal-spread quadratic form. |
| 29 | [surfaces/k_surfaces](../examples/surfaces/k_surfaces/core.py) | B | Reflected tangent planes and the commutator giving edge displacements; metric reflection normalization. |
| 30 | [quadrics/quadrics](../examples/quadrics/quadrics/core.py) | B | Quadric polarity and contact incidence; unit normals and motor transport of the whole ellipsoid. |
| 31 | [quadrics/gaussian](../examples/quadrics/gaussian/core.py) | ? | Mahalanobis metric and homogeneous covariance; the only exterior operation is complementary scalar evaluation. |
| 32 | [quadrics/spherical_quadrics](../examples/quadrics/spherical_quadrics/core.py) | B | Spherical conic polarity; metric geodesic distances and rotor motion. |
| 33 | [quadrics/cga_spherical_quadrics](../examples/quadrics/cga_spherical_quadrics/core.py) | B | Wedges of circles and quadric incidence; conformal metric and rotor flow. |
| 34 | [quadrics/conformal_elliptical](../examples/quadrics/conformal_elliptical/core.py) | ? | Core construction normalizes supplied vectors; plane/point incidence is in the spherical drawing, and rotors in checks. |
| 35 | [quadrics/quadric_collision](../examples/quadrics/quadric_collision/core.py) | B | Tangency and the determinant of a quadric pencil; normal distances and motor placement. |
| 36 | [quadrics/cayley_klein](../examples/quadrics/cayley_klein/core.py) | B | Projective joins, meets and cross ratios; the absolute conic explicitly supplies the metric. |
| 37 | [quadrics/elliptic_physics](../examples/quadrics/elliptic_physics/core.py) | B | Oriented inertia and contact pairings; spherical metric, commutators and motor dynamics. |
| 38 | [quadrics/s3_raytracer](../examples/quadrics/s3_raytracer/core.py) | B | Central projection and conic incidence; spherical normalization and geodesic depth. |
| 39 | [quadrics/cyclides](../examples/quadrics/cyclides/core.py) | B | Conformal point/sphere incidence and rotation planes; null metric and spherical ray travel. |
| 40 | [quadrics/csg_raytracer](../examples/quadrics/csg_raytracer/core.py) | B | Quadric and half-space incidence; unit geometry, motor placement and ray distances. |
| 41 | [quadrics/glass_cyclide](../examples/quadrics/glass_cyclide/core.py) | B | Conformal surface incidence; metric sphere inversion, surface normals and refraction. |
| 42 | [mechanics/inertia](../examples/mechanics/inertia.py) | B | Bivector inertia and complementary pairings; metric principal frames and motor recovery. |
| 43 | [mechanics/simplex](../examples/mechanics/simplex.py) | B | Moment pairings and angular inertia; the commutator maps a rotation plane to each point's velocity. |
| 44 | [mechanics/modes](../examples/mechanics/modes/core.py) | B | Spring lines and twist/load incidence; unit spring lengths and inertia from rotational motion. |
| 45 | [mechanics/tennis_racket](../examples/mechanics/tennis_racket/core.py) | B | Bivector rates and rotor evolution; inertia and kinetic-energy norms. |
| 46 | [mechanics/lie_integrators](../examples/mechanics/lie_integrators.py) | B | Mixed-grade motor evolution and bivector commutators; inertia and normalization. |
| 47 | [mechanics/xpbd](../examples/mechanics/xpbd/core.py) | B | Join of joint anchors; line lengths, inverse inertia and motor corrections. |
| 48 | [mechanics/modal_xpbd](../examples/mechanics/modal_xpbd/core.py) | B | Twist/load pairing and rotational motion; spring distances, modal stiffness and inertia. |
| 49 | [mechanics/robot_arm](../examples/mechanics/robot_arm/core.py) | B | Joint axes met from planes and load incidence; exponential rigid motions and their commutators. |
| 50 | [mechanics/manipulability](../examples/mechanics/manipulability/core.py) | B | Axes joined with the gripper; rotational velocity and metric reciprocal force/velocity ellipsoids. |
| 51 | [mechanics/spinning_top](../examples/mechanics/spinning_top/core.py) | B | Contact polarity and force lines; distances, friction and rigid motor evolution. |
| 52 | [mechanics/riccati/riccati](../examples/mechanics/riccati/riccati.ipynb) | B | Thruster lines and twist/load pairings; mobility and quadratic state/effort measures. |
| 53 | [mechanics/rolling_ball](../examples/mechanics/rolling_ball/rolling_ball.ipynb) | B | The oriented rolling plane and loop area; radius-dependent rotor transport. |
| 54 | [mechanics/tricycle](../examples/mechanics/tricycle/core.py) | B | Axle intersections give the turning centre; normalized axes and rigid motion generated by it. |
| 55 | [mechanics/hanging_chain](../examples/mechanics/hanging_chain/hanging_chain.ipynb) | M | Spring lengths, gravitational dot products and the Newton solve; the rotor only generates the initial arc. |
| 56 | [mechanics/kepler](../examples/mechanics/kepler/core.py) | B | Mixed-grade position spinor and angular momentum; distances and the metric Kepler Hamiltonian. |
| 57 | [mechanics/foucault](../examples/mechanics/foucault/core.py) | B | Rotation planes and transported directions; metric projection onto the local floor. |
| 58 | [mechanics/tides](../examples/mechanics/tides/core.py) | B | The geometric derivative explicitly separates curl and divergence; the physical metric converts force to acceleration. |
| 59 | [mechanics/rattleback](../examples/mechanics/rattleback/core.py) | B | Contact/load pairings and rolling generators; shape support lengths, kinetic energy and motor evolution. |
| 60 | [mechanics/stress](../examples/mechanics/stress/core.py) | B | Traction/face pairing and rotated strain maps; metric constitutive law and principal stresses. |
| 61 | [mechanics/area_transport](../examples/mechanics/area_transport/core.py) | E | Outermorphism on oriented patches, determinant and flux pullback; no physical metric is needed. |
| 62 | [mechanics/composite_strip](../examples/mechanics/composite_strip/core.py) | B | Interior-product constitutive response; the predicted twist is applied as a rotor to produce the deformed strip. |
| 63 | [mechanics/crystal_waves](../examples/mechanics/crystal_waves/core.py) | B | Exterior term in the shear construction and face flux; metric wave-speed eigenproblem. |
| 64 | [mechanics/bending](../examples/mechanics/bending/core.py) | B | Fibre/strain incidence and section geometry; wall lengths, material response and finite turning of sections. |
| 65 | [mechanics/elastica](../examples/mechanics/elastica/core.py) | B | Oriented loads and bivector rod strains; constitutive stiffness and motor transport of sections. |
| 66 | [mechanics/slip_systems](../examples/mechanics/slip_systems/core.py) | B | Slip directions from plane meets; metric resolved shear and the crystal's rotation. |
| 67 | [mechanics/earthquake](../examples/mechanics/earthquake/core.py) | B | Fault couples and the exterior curl of the moment; metric acceleration and rotated fault geometry. |
| 68 | [mechanics/freeform](../examples/mechanics/freeform/core.py) | B | Oriented reference volume and cofactor; metric Neo-Hookean stress and elastic energy. |
| 69 | [mechanics/odd_elasticity](../examples/mechanics/odd_elasticity/core.py) | B | Force/displacement pairings and the actuation plane; metric bond extension and transverse response. |
| 70 | [mechanics/shells](../examples/mechanics/shells/core.py) | B | Oriented face and hinge joins; edge lengths, dihedral angles and bending energy. |
| 71 | [fluids/wing](../examples/fluids/wing/core.py) | B | Stream-function/exterior derivative and circulation; metric potential derivative and conformal flow. |
| 72 | [fluids/vortices](../examples/fluids/vortices/core.py) | B | Bivector circulation and the geometric derivative; inverse-square distance and commutator-driven flow. |
| 73 | [fluids/jeffery](../examples/fluids/jeffery/core.py) | B | Bivector tumbling and shape/readout pairings; sphere metric in the ellipsoid's shape law. |
| 74 | [fluids/floating](../examples/fluids/floating/core.py) | B | Wetted oriented faces and force lines; hydrostatic depth, normals and rigid tilts. |
| 75 | [fluids/kirchhoff](../examples/fluids/kirchhoff/core.py) | B | Twist/load pairing and bivector dynamics; shape-dependent added inertia and kinetic energy. |
| 76 | [fluids/smoke_rings](../examples/fluids/smoke_rings/core.py) | B | Filament turning plane and commutators; segment lengths, curvature and inverse-distance flow. |
| 77 | [fluids/sphere_vortices](../examples/fluids/sphere_vortices/core.py) | B | Joined vortex and field-point geometry; spherical distance and rotor transport. |
| 78 | [fluids/stokes_swimmer](../examples/fluids/stokes_swimmer/core.py) | B | Force/torque pairings and body motion; distance-dependent Stokes mobility. |
| 79 | [relativity/impulse](../examples/relativity/impulse/core.py) | B | Spacetime incidence and boost planes; proper lengths and Lorentz transformations. |
| 80 | [relativity/curvature](../examples/relativity/curvature/core.py) | B | Curvature on spacetime areas; metric tidal eigenproblem and force-to-acceleration conversion. |
| 81 | [relativity/dirac](../examples/relativity/dirac/core.py) | B | Mixed-grade Dirac spinors and geometric multiplication; spacetime metric and probability/current normalization. |
| 82 | [relativity/spinors](../examples/relativity/spinors/core.py) | B | Dirac/Weyl/Majorana spinors and their Clifford actions; spacetime current and probability pairing. |
| 83 | [relativity/gravitational_lensing](../examples/relativity/gravitational_lensing/core.py) | B | Area determinant of the lens map; inverse-distance deflection and its metric reflection derivative. |
| 84 | [relativity/kerr/kerr](../examples/relativity/kerr/kerr.ipynb) | B | Rotation/spin planes and complementary momentum pairing; flat reference metric plus physical gauge field. |
| 85 | [relativity/kerr/kerr_metric](../examples/relativity/kerr/kerr_metric.ipynb) | B | Spin geometry and complementary metric evaluation; the physical Kerr metric and Levi-Civita transport. |
| 86 | [relativity/ligo](../examples/relativity/ligo/core.py) | B | Polarization planes and rotor action; metric strain measured along the detector arms. |
| 87 | [relativity/twistors](../examples/relativity/twistors/core.py) | B | Twistor Clifford actions and ray incidence; conformal null metric and Lorentz geometry. |
| 88 | [relativity/terrell](../examples/relativity/terrell/core.py) | B | Spacetime incidence and boost action; null travel times and physical lengths. |
| 89 | [relativity/thomas_precession](../examples/relativity/thomas_precession/core.py) | B | Boost/rotation planes and their composition; velocity hyperboloid and proper-time metric. |
| 90 | [quantum/magnetic_resonance](../examples/quantum/magnetic_resonance/core.py) | B | Pauli geometric products and commutators; spin-state norm and measurement pairing. |
| 91 | [quantum/graphene](../examples/quantum/graphene/core.py) | B | Spinor multiplication and Berry transport; state norm and momentum/bond pairing. |
| 92 | [quantum/kane_mele](../examples/quantum/kane_mele/core.py) | B | Oriented two-bond spin-orbit plane; spinor dynamics and probability norms. |
| 93 | [quantum/two_spins](../examples/quantum/two_spins/core.py) | B | Products of spin planes and the two-spin correlator; state norm and observable pairing. |
| 94 | [quantum/process_tomography](../examples/quantum/process_tomography/core.py) | B | Rotor and Kraus sandwiches; scalar measurement pairings and channel reconstruction. |
| 95 | [quantum/stabilizer](../examples/quantum/stabilizer/core.py) | B | Clifford blade actions on the code spinor; error probabilities and the state pairing. |
| 96 | [quantum/hubbard_dimer](../examples/quantum/hubbard_dimer/core.py) | B | Exterior many-electron states and hopping lift; state inner product and probabilities. |
| 97 | [quantum/hubbard_ring](../examples/quantum/hubbard_ring/core.py) | B | Exterior four-electron states and hopping lift; state inner product and energy/probability readings. |
| 98 | [quantum/kitaev_wire](../examples/quantum/kitaev_wire/core.py) | B | Majorana generator as a bivector action; mode norms and overlaps. |
| 99 | [quantum/superconductivity](../examples/quantum/superconductivity/core.py) | B | Bivector pseudospin dynamics and commutators; spin normalization and energy response. |
| 100 | [quantum/neutrino_isospin](../examples/quantum/neutrino_isospin/core.py) | B | Flavor-plane commutator and rotor mixing; probabilities and relaxation. |
| 101 | [quantum/affine_toda](../examples/quantum/affine_toda/core.py) | B | Root-triangle wedge areas are compared with cubic couplings; metric mass eigenproblem and Coxeter radii. |
| 102 | [quantum/particle_charges](../examples/quantum/particle_charges/core.py) | B | Spin(10) Clifford actions on spinors; state pairing and charge-sector projections. |
| 103 | [electromagnetism/maxwell](../examples/electromagnetism/maxwell.py) | B | Field/excitation incidence and stress-energy construction; physical metric in the constitutive response. |
| 104 | [electromagnetism/constitutive](../examples/electromagnetism/constitutive/core.py) | B | Exterior Maxwell constraints and field planes; metric material response and moving frames. |
| 105 | [electromagnetism/moving_charge](../examples/electromagnetism/moving_charge/core.py) | B | Exterior derivative gives the field and conservation; metric source distances and Lorentz boosts. |
| 106 | [electromagnetism/fresnel](../examples/electromagnetism/fresnel/core.py) | B | Exterior wave constraints and outermorphism detecting degeneracy; material metric and wave speed. |
| 107 | [electromagnetism/induced_dipoles](../examples/electromagnetism/induced_dipoles/induced_dipoles.ipynb) | B | Dipole coupling uses a bivector half-turn sandwich and metric separation lengths. |
| 108 | [electromagnetism/electrostriction](../examples/electromagnetism/electrostriction/core.py) | B | Dipoles sweep spacetime areas and pair with fields; metric separations and elastic/electrostatic energy. |
| 109 | [electromagnetism/eddy_brake](../examples/electromagnetism/eddy_brake/core.py) | B | Oriented spacetime sweeps and flux; conduction metric, distances and rotational dynamics. |
| 110 | [electromagnetism/second_harmonic](../examples/electromagnetism/second_harmonic/core.py) | B | Bond directions sweep spacetime planes; polarization response, normalized axes and rotated crystal maps. |
| 111 | [electromagnetism/magnetic_bottle](../examples/electromagnetism/magnetic_bottle/core.py) | B | Magnetic bivector and Lorentz-force pairing; particle proper speed and coil-distance field law. |
| 112 | [optics/thin_lens](../examples/optics/thin_lens/core.py) | B | Ray/element incidence and lens maps; mirror action and motor transport of the optical train. |
| 113 | [optics/lens_camera](../examples/optics/lens_camera/core.py) | B | Projective lens maps and aperture cones; round aperture metric and flat/spherical placement. |
| 114 | [optics/coating](../examples/optics/coating/core.py) | B | Exterior Maxwell boundary constraints; metric field projections and material propagation. |
| 115 | [optics/photoelasticity](../examples/optics/photoelasticity/core.py) | B | Electric planes and polarization pairings; metric elastic stress and optical response. |
| 116 | [optics/crystal_diffraction](../examples/optics/crystal_diffraction/core.py) | B | Reciprocal phase pairing; geometric phase amplitude and rotor deformation of the crystal. |
| 117 | [optics/black_sandwich](../examples/optics/black_sandwich/core.py) | B | An axis swept through time reads the field; metric dual supplies the polarizer's output field. |
| 118 | [optics/rainbow](../examples/optics/rainbow/core.py) | B | A rotor in the incidence plane carries successive bounces; metric Snell/Fresnel laws. |
| 119 | [optics/specular_caustics](../examples/optics/specular_caustics/core.py) | B | Ray-map determinant gives oriented area/folds; metric normals and Snell refraction. |
| 120 | [mechanics/symmetry](../examples/mechanics/symmetry/core.py) | B | Rotation/reflective actions on response maps; conductivity and elastic metric structures. |
| 121 | [orientation/conformal_circle](../examples/orientation/conformal_circle/core.py) | B | Joined points versus intersected surfaces; conformal null metric and reflection action. |
| 122 | [orientation/mechanical_work](../examples/orientation/mechanical_work/core.py) | B | Force line from joining handle and force; metric rigid turning and velocity commutator. |
| 123 | [orientation/mirror_energy](../examples/orientation/mirror_energy/core.py) | B | Oriented angular momentum and reflected inertia; kinetic energy and spherical metric. |
| 124 | [orientation/normals](../examples/orientation/normals/core.py) | B | Joined versus intersected line/plane orientation; normalized normals and metric reflection action. |
| 125 | [orientation/optical_rotation](../examples/orientation/optical_rotation/core.py) | B | Spacetime polarization planes and their oriented generators; metric rotor propagation. |
| 126 | [orientation/wu_experiment](../examples/orientation/wu_experiment/core.py) | B | Emission is an inner-product law; the example also reflects the spin plane and intrinsic directions differently. |
| 127 | [math/hopf](../examples/math/hopf/core.py) | B | Spinor action and linked-fibre volume; unit-spin metric and parallel-transport phase. |
| 128 | [math/spin_groups](../examples/math/spin_groups/core.py) | B | Clifford products, bivectors and spin groups; signatures and their invariant metric form. |
| 129 | [math/invariant_decomposition](../examples/math/invariant_decomposition.py) | B | Wedge powers resolve rotation planes; squared norms and metric eigenspaces give their rates. |
| 130 | [math/klein_quadric](../examples/math/klein_quadric/core.py) | E | Klein wedge form, ruled quadric and incidences determine the answer; metric choices select representatives and a numerical nullspace basis. |
| 131 | [math/pascal](../examples/math/pascal/core.py) | E | Conic pencil and Pascal joins/meets; rotor headings only sample the diagram. |
| 132 | [math/poncelet](../examples/math/poncelet/core.py) | E | Conic incidence, tangency and projective iteration; norm rescaling and ellipse placement do not change the porism. |
| 133 | [math/reflections](../examples/math/reflections) | B | Composed mirror products produce rotors, edges and spinor polytopes; mirror angles and signatures supply the metric. |
| 134 | [math/fractals](../examples/math/fractals/fractals.ipynb) | B | Complex/quaternion geometric multiplication; boundedness uses a norm, and quaternion slices turn. |
| 135 | [math/octonions](../examples/math/octonions/core.py) | B | The octonion product uses a trivector and Clifford multiplication; its norm and G2 action use the metric. |
| 136 | [math/extensor_representations](../examples/math/extensor_representations/core.py) | B | Faithful Clifford actions on spinor ideals; their defining metric and non-scalar products. |
| 137 | [math/doubled_algebra](../examples/math/doubled_algebra/core.py) | B | Exterior creation and contraction explicitly realized together in a split-signature Clifford algebra. |
| 138 | [math/e8](../examples/math/e8/core.py) | B | Coxeter rotor and wedge-defined invariant planes; root lengths and orthogonal eigenspaces. |
| 139 | [math/e_series](../examples/math/e_series/core.py) | B | Reflection sandwiches combine scalar and bivector parts; metric adjoints and eigenspaces resolve the invariant planes. |
| 140 | [math/eight_conics](../examples/math/eight_conics/core.py) | E | Contact-line dyads and conic completion; Euclidean intersections only locate the displayed contact endpoints. |
| 141 | [math/g2_rolling](../examples/math/g2_rolling/core.py) | B | Split-octonion products and rolling planes; null metric, sphere lengths and rotor attitude. |
| 142 | [math/isospectral_drums](../examples/math/isospectral_drums/core.py) | ? | Scalar-field transplantation and a metric Laplacian; exterior determinants occur in mesh winding/boundary construction. |
| 143 | [math/penrose](../examples/math/penrose/core.py) | B | Oriented cuts and projected lattice faces; orthogonal projection and tile areas. |
| 144 | [math/spherical_synchronization](../examples/math/spherical_synchronization/core.py) | B | Conformal rotor evolution and circle incidence; spherical null metric and commutator attraction. |
| 145 | [math/triality](../examples/math/triality/core.py) | B | Octonion/Clifford products and bivector derivations; norm-preserving rotations and inner products. |
