# Examples

## Geometry

- [Projection](geometry/projection/) — Pinhole cameras, shadows and stereo correspondences use the same join-and-meet construction.
- [Scenegraph](geometry/scenegraph/) — Composed maps carry an articulated scene through its camera and onto the image plane.
- [Fitting](geometry/fitting/) — An eigenproblem fits points, lines and planes to noisy samples.
- [Registration](geometry/registration/) — Rotor and motor eigenproblems align corresponding point clouds.
- [Pose from three markers](geometry/three_point_pose/) — Three point, line and plane matches determine an object's rigid pose.
- [Rational Bézier curves](geometry/bezier/) — Weighted control points trace a conic and its cone of rays.
- [Sundials](geometry/sundial/) — The sun's motion determines hour lines and seasonal shadow paths on a horizontal dial.
- [Origami](geometry/origami/) — Crease intersections and rigid folds turn a sheet of paper into a Kabuto helmet.
- [Rubik's cube](geometry/rubiks_cube/) — The same quarter-turn maps move, draw and solve the cube's stickers.

## Estimation

- [Camera pose fitting](estimation/camera_fit.py) — JAX differentiates through a motor and camera map to recover pose from image measurements.
- [Kalman pose filter](estimation/kalman/) — A moving body's pose and uncertainty are updated from motion and measurement estimates.
- [Epipolar reconstruction](estimation/epipolar/) — Corresponding sight rays recover two cameras' relative pose and the points they observe.
- [Multi-view reconstruction](estimation/multiview/) — Sight cones with image uncertainty determine landmarks and camera poses together.
- [Odometry](estimation/odometry/) — Relative pose measurements and loop closure determine a robot's most likely trajectory and its uncertainty.
- [Station keeping](estimation/pose_diffusion/) — A covariance describes how far gusts push a vessel from its controlled position and heading.

## Surfaces

- [Quadric error metrics](surfaces/qem/) — Summed face quadrics place edge contractions, from a single ridge to progressive simplification of the bunny.
- [Skinning](surfaces/skinning/) — Blending motors and blending their maps produce different deformations of a jointed surface.
- [Surface curvature](surfaces/surface_curvature/) — A quadric's polarity determines its tangent planes, principal curvatures and curvature directions.
- [As-rigid-as-possible deformation](surfaces/arap/) — Moving a few vertices bends a mesh while keeping each neighbourhood close to a rigid rotation.
- [Projective Loop subdivision](surfaces/projective_loop/) — Subdivision averages homogeneous points whose weights control how strongly they pull the surface.
- [Motor Loop subdivision](surfaces/motor_loop/) — Subdivision averages and normalizes motors to refine a field of rigid frames.
- [Weighted dual meshes](surfaces/weighted_dual/) — A convex polyhedron and its polar dual produce paired meshes on the sphere.
- [Koebe polyhedra](surfaces/koebe/) — Circle tangencies construct polyhedra whose edges all touch the unit sphere.
- [Lifts and reciprocal diagrams](surfaces/reciprocal/) — Heights on a planar mesh determine crease stresses and a reciprocal diagram.
- [Spin transformations](surfaces/spin_transformations/) — A quaternion field deforms a surface to change its mean curvature while preserving angles.
- [Spinor waves](surfaces/dirac_waves/) — A sparse Dirac map propagates a spinor field over a triangulated sphere.
- [Surface connections](surfaces/connections/) — Parallel transport constructs tangent fields, their singularities and stripe patterns on curved surfaces.
- [Developable surfaces](surfaces/developability/) — A mesh flows towards patches that can be flattened without stretching.
- [Discrete K-surfaces](surfaces/k_surfaces/) — Compatible reflections grow saddle surfaces and their Bäcklund partners.
- [Whitney forms](surfaces/whitney_forms/) — Corner hat functions and edge one-forms interpolate vertex and edge cochains across triangles and pull material conductivity back to the mesh.
- [Regge calculus](surfaces/regge_calculus/) — Edge-length readings define triangle correlations, corner turns reveal intrinsic curvature, and an adjugate assembles the conduction response.
- [Dirac–Kähler mixed-grade fields](surfaces/dirac_kahler/) — A geometric derivative couples wave components on an annulus; coloured reliefs show their propagation and conversion at the boundary.
- [Rotor Whitney forms](surfaces/rotor_whitney/) — Interpolated rotors split turning from chord stretch; a disk wraps a director sphere, and coloured ribbons reveal a relaxing frame field.

## Quadrics

- [Projective quadrics](quadrics/quadrics/) — Plane-to-point maps describe ellipsoids, their tangent planes and their rigid motions.
- [Gaussian fitting](quadrics/gaussian/) — Homogeneous point moments determine a Gaussian and its one-standard-deviation quadric.
- [Spherical quadrics](quadrics/spherical_quadrics/) — A quadratic cone cuts the sphere into a conic described by points or tangent great circles.
- [Conformal spherical quadrics](quadrics/cga_spherical_quadrics/) — Conformal motions deform spherical regions defined by quadric maps.
- [Sphere rendering](quadrics/conformal_elliptical/) — Planes across the cube's rotation axes draw a symmetric pattern of great circles on the sphere.
- [Quadric collision](quadrics/quadric_collision/) — Blends of dual ellipsoids determine separation, contact and overlap.
- [Cayley–Klein geometry](quadrics/cayley_klein/) — A chosen absolute quadric supplies distances, angles and perpendiculars to projective geometry.
- [Spherical rigid-body dynamics](quadrics/elliptic_physics/) — The same construction moves and collides quadric bodies on the two- and three-sphere.
- [Rendering on the three-sphere](quadrics/s3_raytracer/) — Projected quadrics give the visible outlines of bodies in spherical space.
- [Cyclides on the three-sphere](quadrics/cyclides/) — Great-circle rays intersect Dupin cyclides in the conformal model of spherical space.
- [Solids cut from quadrics](quadrics/csg_raytracer/) — Intersections and differences of quadrics form solids for ray tracing.
- [Glass cyclide](quadrics/glass_cyclide/) — Reflection and refraction reveal a glass Dupin cyclide represented by a conformal quadric.

## Mechanics

- [Inertia](mechanics/inertia.py) — Two eigenproblems recover a body's principal frame and its pose motor from its inertia.
- [Simplex moments](mechanics/simplex.py) — Weighted points integrate the second moments and inertia of simplices in arbitrary dimensions.
- [Stiffness and normal modes](mechanics/modes/) — Stiffness and inertia maps determine the vibration modes of a suspended plate.
- [Tennis racket instability](mechanics/tennis_racket/) — A freely spinning body tumbles when its rotation starts near the intermediate principal axis.
- [Lie integrators](mechanics/lie_integrators.py) — Rigid-body time integration evolves pose motors and motion rates under applied loads.
- [Rigid-body chains](mechanics/xpbd/) — Position-based constraints keep a moving chain's rigid links joined at their anchors.
- [Flexible bodies, rigidly joined](mechanics/modal_xpbd/) — Low vibration modes let connected girders bend while their point constraints stay closed.
- [Robot arm kinematics](mechanics/robot_arm/) — Joint motors and carried axes determine gripper motion, joint torques and target tracking.
- [Arm manipulability](mechanics/manipulability/) — Joint axes determine the gripper's velocity and force ellipsoids as the arm changes pose.
- [Spinning top](mechanics/spinning_top/) — A top built from ellipsoids moves in contact with a concave bowl.
- [Riccati feedback](mechanics/riccati/riccati.ipynb) — Optimal feedback docks a vessel with three thrusters while balancing accuracy and effort.
- [Rolling ball](mechanics/rolling_ball/) — A ball's orientation changes as it rolls around a closed path on a table.
- [Tricycle](mechanics/tricycle/) — Wheel constraints determine steering motion in the plane and in space.
- [Hanging chain](mechanics/hanging_chain/) — Newton's method balances the forces on a chain suspended between two pins.
- [Kepler motion](mechanics/kepler/) — A scaled position spinor regularizes close passages in a Kepler orbit.
- [Foucault's pendulum](mechanics/foucault/) — Earth's rotation turns a pendulum's swing plane relative to its floor.
- [Tides](mechanics/tides/) — Soft gravitational masses produce a tidal map that stretches a cluster of stars.
- [Rattleback](mechanics/rattleback/) — An asymmetric rolling body exchanges rocking and spin, reversing rotation in one sense.

## Elasticity and materials

- [Stress in an elastic cube](mechanics/stress/) — Strain and stress maps determine face tractions, principal directions and Mohr's circle.
- [Area transport](mechanics/area_transport/) — A deformation carries edges, oriented areas and volumes through a complete flattening.
- [Composite strip](mechanics/composite_strip/) — Fibre directions couple axial extension to twist in a laminated strip.
- [Piezoelectric bimorph](electromagnetism/piezoelectric/piezoelectric.ipynb) — One constitutive coupling bends a layered continuum under voltage and generates an electrical signal under load.
- [Crystal waves](mechanics/crystal_waves/) — Crystal stiffness determines elastic wave speeds, polarizations and energy focusing.
- [Seismic waves in a small planet](mechanics/whitney_elasticity/) — Constant-strain triangles carry P, S and surface waves through a solid shell around a liquid core, which casts a shadow for the S waves.
- [Beam bending](mechanics/bending/) — Loads on thin-walled cantilevers produce bending and torsion according to the cross-section's geometry.
- [Elastica](mechanics/elastica/) — End loads bend elastic rods into buckled shapes, curls and helices.
- [Crystal slip](mechanics/slip_systems/) — Applied tension selects slip systems and turns a metal crystal as it yields.
- [Earthquake beachballs](mechanics/earthquake/) — Fault slip determines a moment map, seismic radiation and its beachball diagram.
- [FreeForm elasticity](mechanics/freeform/) — Blending a few affine motions deforms an elastic solid without a simulation mesh.
- [Odd elasticity](mechanics/odd_elasticity/) — Active bonds exert sideways forces under extension and exchange work over deformation cycles.
- [Discrete shells](mechanics/shells/) — Edge stretching and hinge bending determine the deformation of thin triangle meshes.

## Fluids and floating bodies

- [Joukowski wing](fluids/wing/) — A potential flow bends around a wing with vanishing divergence and curl outside it.
- [Leapfrogging vortices](fluids/vortices/) — Planar vortices move one another through the velocity fields they induce.
- [Jeffery orbits](fluids/jeffery/) — Rigid ellipsoids tumble in a shear flow according to their shape and orientation.
- [Hydrostatic stability](fluids/floating/) — Water pressure determines restoring loads on floating bodies from small tilts to capsizing.
- [Kirchhoff's equations](fluids/kirchhoff/) — Added fluid inertia changes the motion of an ellipsoid through still ideal fluid.
- [Smoke rings](fluids/smoke_rings/) — Vortex filaments evolve under their own curvature and the flow induced by neighbouring rings.
- [Vortices on the sphere](fluids/sphere_vortices/) — Point vortices carry circulation and induce motion on a spherical surface.
- [Swimming without inertia](fluids/stokes_swimmer/) — Cyclic shape changes propel a swimmer while fluid forces and torques balance at every instant.
- [Stokes flow on a mesh](fluids/dec_stokes/) — Incidence and diagonal Hodge maps determine pressure and flux through a channel around an obstacle.
- [Projective Whitney fields](fluids/projective_sphere_flow/) — Homogeneous corner weights vary rational interpolation and the polar dual while diffusion stays on a fixed sphere; the same quotient works in the affine plane.
- [Laplacian eigenfunction fluids](fluids/eigen_fluids/) — Whitney gradients and precomputed mode interactions evolve vortices on the Stanford bunny and shear layers on a sphere.

## Relativity

- [Radar ranging](relativity/radar/radar.ipynb) — Echo times recover a moving reflector's range and velocity in projective spacetime.
- [Satellite navigation](relativity/gps/gps.ipynb) — Light-cone correlations recover a receiver's position and clock error, while timing readouts determine its uncertainty.
- [Distributed impulses](relativity/impulse/) — Continuous worldlines expose the strains behind the ladder paradox and Bell's spaceships.
- [Gravitational-wave curvature](relativity/curvature/) — A spacetime curvature map determines the tidal deformation caused by a passing wave.
- [Dirac electron](relativity/dirac/) — A spacetime spinor carries the electron's density, current and spin through its action as a map.
- [Dirac, Weyl and Majorana spinors](relativity/spinors/) — Three kinds of particle state occupy one real spacetime spinor space.
- [Gravitational lensing](relativity/gravitational_lensing/) — Point masses bend sightlines to produce multiple images of a source.
- [Kerr ray tracing in GTG](relativity/kerr/kerr.ipynb) — Gauge theory gravity transports light and polarization past a rotating black hole.
- [Kerr ray tracing with a metric](relativity/kerr/kerr_metric.ipynb) — The metric and connection describe light and polarization in the same black-hole scene.
- [LIGO antenna patterns](relativity/ligo/) — Detector arms respond differently to gravitational waves arriving from different directions and polarizations.
- [Twistors and linked light](relativity/twistors/) — Twistors describe light rays and a twisting family of rays filling spacetime.
- [Terrell rotation](relativity/terrell/) — Light travel times determine photographs of bodies passing a camera near the speed of light.
- [Thomas precession](relativity/thomas_precession/) — Successive boosts turn a particle's frame and contribute to the spin motion measured in muon experiments.
- [Spacetime Whitney waves](relativity/spacetime_whitney/) — Whitney interpolation on a spacetime mesh follows a scalar pulse reflecting from an advancing mirror.

## Quantum

- [Magnetic resonance](quantum/magnetic_resonance/) — Driven and relaxing spins produce nutation, resonance line shapes and spin echoes.
- [Graphene](quantum/graphene/) — Electron hopping produces Dirac cones, pseudospin textures and a Berry phase around each cone.
- [Graphene edge states](quantum/kane_mele/) — Spin-orbit coupling produces boundary states in a finite graphene flake.
- [Two spins](quantum/two_spins/) — Exchange interactions entangle a pair of spins and change their Bell correlations.
- [Quantum process tomography](quantum/process_tomography/) — Prepared states and measured probabilities reconstruct a noisy qubit gate.
- [Stabilizer code](quantum/stabilizer/) — Four physical qubits encode two logical qubits while detecting single-qubit bit and phase errors.
- [Hubbard dimer](quantum/hubbard_dimer/) — Two electrons hop between two sites while paying an energy cost for sharing a site.
- [Hubbard ring](quantum/hubbard_ring/) — Exterior products enforce exclusion and exchange signs for four electrons on a four-site ring.
- [Kitaev wire](quantum/kitaev_wire/) — A moving gate transports Majorana modes at the boundaries of a topological segment.
- [Superconducting condensate](quantum/superconductivity/) — Coupled Anderson pseudospins produce oscillations of a superconducting gap.
- [Neutrino flavour](quantum/neutrino_isospin/) — Coherent evolution and decoherence change neutrino flavour probabilities during propagation.
- [Affine E8 Toda theory](quantum/affine_toda/) — Derivatives of a root-based potential determine masses and three-mode interactions.
- [Spin(10) particle charges](quantum/particle_charges/) — Gauge actions on a spinor distinguish its particle sectors and their charges.

## Electromagnetism

- [Constitutive maps](electromagnetism/constitutive/) — A material response maps electromagnetic fields to their excitations in spacetime algebra.
- [Moving charges](electromagnetism/moving_charge/) — Derivatives of a moving charge's potential determine its electromagnetic field.
- [Fresnel wave surfaces](electromagnetism/fresnel/) — A crystal's constitutive map determines the directions and speeds of light propagation.
- [Induced dipoles](electromagnetism/induced_dipoles/) — Atoms polarize in response to an applied field and to one another's dipoles.
- [Electrostriction](electromagnetism/electrostriction/) — An elastic lattice deforms while its induced dipoles reach electrical and mechanical equilibrium.
- [Eddy-current braking](electromagnetism/eddy_brake/) — Directional conductivity determines currents and braking torque in a rotating disc.
- [Second-harmonic generation](electromagnetism/second_harmonic/) — A crystal's quadratic field response generates light at twice the driving frequency.
- [Magnetic bottle](electromagnetism/magnetic_bottle/) — Charged particles spiral, bounce or escape in the field between two coils.
- [Magnetic flux advection](electromagnetism/lie_advection/) — Swept edge patches and Cartan's decomposition transport magnetic circulation and conserve total flux across a swirling conductor.
- [Force-free Beltrami fields](electromagnetism/beltrami_fields/) — Tetrahedral Whitney forms carry magnetic energy and helicity; selective modal decay at fixed helicity reveals a force-free state inside a torus.

## Optics

- [Thin lenses](optics/thin_lens/) — Maps on rays compose into optical trains and determine their imaging properties.
- [Lens camera](optics/lens_camera/) — A lens train carries aperture cones to image cones to produce zoom and depth of field.
- [Optical coatings](optics/coating/) — Constitutive maps determine coupled polarization propagation through layered materials.
- [Photoelasticity](optics/photoelasticity/) — Polarized light reveals the stress around a circular hole in a loaded plate.
- [Crystal diffraction](optics/crystal_diffraction/) — Deformation carries atoms and diffraction peaks through paired maps that preserve phase.
- [Black polarization sandwiches](optics/black_sandwich/) — A wave plate between crossed polarizers produces a nonzero optical map whose square vanishes.
- [Rainbow](optics/rainbow/) — Refraction and internal reflection in a raindrop concentrate sunlight into rainbow angles.
- [Specular caustics](optics/specular_caustics/) — Refraction through a rippled surface concentrates light on a flat receiver.

## Thermodynamics

- [Constrained heat capacity](thermodynamics/state_space/) — Constraints on a gas's state determine how much its temperature rises under heating.
- [Carnot cycle](thermodynamics/heat_cycles/) — A gas exchanges heat and work around a cycle between hot and cold reservoirs.
- [Legendre transformation](thermodynamics/contact_geometry/) — Contact geometry changes the controlled variable from entropy to temperature while preserving equilibrium relations.
- [Entropy production](thermodynamics/entropy_production/) — A mobility map transfers heat between two bodies while increasing their total entropy.
- [Damping as heating](thermodynamics/generic/) — Reversible motion and friction exchange mechanical and thermal energy in one GENERIC system.
- [Critical fluctuations](thermodynamics/ruppeiner/) — A fluid's thermodynamic metric and its curvature describe fluctuations near the critical point.
- [Air conditioning](thermodynamics/property_charts/) — Cooling, condensation and reheating trace a path across the psychrometric chart.
- [Boiling](thermodynamics/phase_change/) — A common-tangent construction determines coexisting phases and boiling at fixed pressure.

## Symmetry

- [Material symmetry](mechanics/symmetry/) — Averaging over symmetry groups restricts heat conduction, crystal elasticity and optical activity.

## Orientation

- [A circle constructed twice](orientation/conformal_circle/) — Ordered points and intersecting surfaces give a conformal circle distinct orientations under reflection.
- [Mechanical work in a mirror](orientation/mechanical_work/) — A crank's load and motion pair to the same work in a reflected experiment.
- [Energy in a mirror](orientation/mirror_energy/) — Reflected motion and mass distributions retain the same positive kinetic energy.
- [Surface normals](orientation/normals/) — Chosen sides and vertex winding determine different behaviours of normals under reflection.
- [Optical rotation](orientation/optical_rotation/) — Round trips distinguish polarization rotation by an optically active medium from rotation by a magnetic field.
- [Wu experiment](orientation/wu_experiment/) — Electron emission from polarized cobalt-60 distinguishes an experiment from its mirror image.

## Mathematics

- [Hopf fibration](math/hopf/) — Spinors with the same spin direction form linked circles on the three-sphere.
- [Spin groups](math/spin_groups/) — Rotors of every signature up to six dimensions are compared inside one algebra.
- [Invariant decomposition](math/invariant_decomposition/) — A six-dimensional rotation separates into three commuting plane rotations.
- [Klein quadric](math/klein_quadric/) — The quadric of lines determines the two lines meeting four given lines in space.
- [Pascal's theorem](math/pascal/) — Five points determine a quadric whose inscribed hexagons exhibit Pascal's collinearity.
- [Poncelet's porism](math/poncelet/) — Tangent steps between two quadrics close into polygons independently of their starting point.
- [Reflections](math/reflections/) — Composed mirrors generate kaleidoscopes, polyhedra and tilings in three signatures.
- [Fractals](math/fractals/) — Iteration in even subalgebras produces escape-time fractal images.
- [Octonions](math/octonions/) — An eight-component product gives a concrete construction of its exceptional symmetry group G2.
- [Extensor representations](math/extensor_representations/) — Blade charts realize faithful Clifford actions on their smallest real state spaces.
- [Doubled algebra](math/doubled_algebra/) — Products in Cl(2,2) represent wedge, contraction and every linear map on planar multivectors.
- [E8 reflections](math/e8/) — Coxeter orbits generated by eight mirrors produce all 240 roots of E8.
- [The E series](math/e_series/) — Nested sets of mirrors generate the roots and rotation planes of E6, E7 and E8.
- [Eight-conic theorem](math/eight_conics/) — Compatible contacts between seven quadrics determine an eighth quadric in three equivalent ways.
- [Octonionic rolling](math/g2_rolling/) — Split-octonion multiplication plans a ball's rolling motion over a sphere three times its radius.
- [Isospectral drums](math/isospectral_drums/) — Sparse maps transplant vibration modes between differently shaped drums with the same frequencies.
- [Penrose tilings](math/penrose/) — A plane cutting a five-dimensional integer lattice produces a nonperiodic rhombus tiling.
- [Spherical synchronization](math/spherical_synchronization/) — Interacting directions on a sphere evolve through one common Möbius transformation.
- [Spin(8) triality](math/triality/) — Octonion multiplication links three rotations whose common subgroup is G2.

## Benchmarks

- [Motor map timings](../benchmarks/motor_map.py) — Benchmarks compare ways of applying motor sandwiches and their compiled maps.

Run an example from the repository root with `python -m examples.<topic>.<name>.scenarios`.

Notebooks sit next to their examples. Open one in Jupyter from inside the repository, or use the Open in Colab badge at its top.
