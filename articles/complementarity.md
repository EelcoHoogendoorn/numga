# Complementarity

Quantities that move with a space and quantities that measure or act on them transform in different ways, yet combine into invariant physical readings. This document sets out a single complementary form for that pairing, providing a unifying perspective on dual constructions across a wide range of fields.

## The ansatz

A geometric relationship or a physical measurement that evaluates to an ordinary scalar takes one of two forms:

```python
scalar = extrinsic & intrinsic                # [] Scalar+
scalar = a & duality(a)                       # [] Scalar+
```

The first form pairs two quantities of opposite [orientation](orientation.md): an extrinsic element (`+`), oriented in the space around it like the sides of a plane or a turn about a line, and an intrinsic element (`-`), oriented within itself like a weight at a point or a direction along a line. Through [the regressive product](regressive_product.md), `&` pairs an extrinsic quantity with an intrinsic one of complementary grade, k and n − k in a space of n dimensions. A plane tests a point this way, `plane & point`, a wavevector reads the phase a wave gains across a displacement, `wavevector & displacement`, and a motion takes power from a force, `rate & forque`.

The second form is the quadratic case, in which a quantity is paired with its own image under a map. The term ***duality map*** here means a [linear extensor](multiple_lenses.md) from one kind of element to the complementary kind, intrinsic to extrinsic or the reverse; `duality` takes `a` into its complementary space so the two can pair. Placing the extrinsic factor first writes this as `a & duality(a)` when `a` is extrinsic and `duality(a) & a` when `a` is intrinsic. A body's inertia takes a rate of motion to a momentum, and `rate & inertia(rate)` is twice the kinetic energy. A medium's susceptibility takes a field to the response it induces, and `field & susceptibility(field)` pairs the two. A quadric surface takes an intrinsic point to an extrinsic plane, and `quadric(point) & point` tests whether the point lies on the surface. Each is again an extrinsic quantity paired with an intrinsic one.

Which of the two kinds a quantity is follows from how it is built and from what it does ([Appendix: representations](#appendix-representations)). The mirrors of a space, its hyperplanes, are extrinsic, and so are the rates, twists and fields built from them by the outer product `^`, which generate continuous motion. What those transformations carry is intrinsic: points, the lines and patches joined from points by `&`, and extensive quantities such as mass and entropy.

## Claims

A reading written in either form has four properties. The first is that the pairing involves no metric. The regressive product of complementary grades depends on the grades and on the unit of volume alone, so one expression serves a Euclidean algebra, a projective algebra, whose metric is degenerate, a spacetime, whose metric is indefinite, and a space of thermodynamic states, which has no metric at all. Where a metric does belong to a problem, it enters explicitly as a duality map of its own, `metric(a) & a`: as a constitutive law, as a reference quadric shape, or as the comparison form in an eigenproblem.

The second is that the expression does not depend on the dimension. In a space of another dimension the two factors have other grades, and with the extrinsic factor first the line that pairs them is unchanged, with no sign in it that varies with the dimension.

The third is that a mirror leaves the reading as it is. A reflection commutes with every product, `mirror >> (a & b) == (mirror >> a) & (mirror >> b)`, and an extrinsic factor paired with an intrinsic one gives an extrinsic scalar, which a reflection does not change.

The fourth is that every linear map transfers across the pairing, including a map that has no inverse. Where a map `T` carries the intrinsic factor forward, its adjugate pulls the extrinsic factor back, `measurement & T(variation) == T.adjugate()(measurement) & variation`, whether or not `T` preserves lengths; when `T` is invertible, solving through `T.adjugate()` carries the extrinsic factor forward as well.

## A spinning wheel

A spinning wheel shows two of these properties, the dimension and the mirror, on quantities from a first course in mechanics. The wheel spins about its hub while a push at its rim drives it, as in a `numga` [spinning wheel](../examples/orientation/spinning_wheel/spinning_wheel.ipynb) example. Its rate of turning is modelled as the extrinsic plane it turns in, while the mass positions and the push are intrinsic ([Points, motions and momenta](#points-motions-and-momenta)). Joining each mass with its velocity sums the wheel's inertia, and pairing the rate with the momentum and with the torque gives the kinetic energy and the power:

```python
inertia = points.regressive(points.commutator(Rate) * masses).sum()   # [] Antibivector- <- Bivector+
momentum = inertia(rate)                                              # [] Antibivector-
energy = (rate & momentum) / 2                                        # [] Scalar+: 13.5
torque = handle & push                                                # [] Antibivector-
power = rate & torque                                                 # [] Scalar+: 4.5
```

The momentum joins where each mass is with where it is going, and the torque joins where the push acts with the push. Both are intrinsic, the rate is extrinsic, and the energy and the power are its pairings with them.

The same five lines serve a wheel in the plane and a wheel in space. In the plane the momentum is one number, 9. For the same wheel tilted in space it has three components, of size 9. The energy and the power are 13.5 and 4.5 in both.

The wheel's mirror image is built by the same lines from the mirror images of its ingredients, `mirror >> points`, `mirror >> rate` and `mirror >> push`. Its momentum and its torque come out reversed, since it turns the other way, and its energy and its power are again 13.5 and 4.5.

<p align="center"><img src="../plots/spinning_wheel_plane.gif" alt="A wheel of three masses turning under a push at its rim, beside its mirror image turning the other way" width="420" /></p>

*Figure: a wheel of three masses, left, and its mirror image, right, with the mirror as the dashed line. Coloured arrows are the masses' velocities, and the dark arrow is the push.*

In the vector notation of a first course in mechanics, the wheel is treated once for the plane and once for space. In the plane the speed of turning, the angular momentum and the torque are signed numbers, counted positive counterclockwise: $L = I\omega$, $K = \tfrac12 I\omega^2$, $\tau = rF$, $P = \tau\omega$. In space they are vectors along the axle, pointed by the right-hand rule: $\mathbf v = \boldsymbol\omega\times\mathbf r$, $\mathbf L = \sum_i \mathbf r_i\times m_i\mathbf v_i$, $\boldsymbol\tau = \mathbf r\times\mathbf F$, $P = \boldsymbol\tau\cdot\boldsymbol\omega$. Under a mirror $M$, a position and a push go to $M\mathbf r$ and $M\mathbf F$, while $\boldsymbol\omega$, $\mathbf L$ and $\boldsymbol\tau$ transform by $\det(M)\,M$ times themselves under the axial-vector reflection rule [[1, §52-5]](#references). All three give the numbers above; projective rigid-body mechanics and the dual-algebra distinction between axes and spears are developed in [[2, §§2.2, 6]](#references), [[3, Ch. 9]](#references), [[4, §2.7]](#references) and [[5, §8.6.1]](#references).

## Wave phase under a deformation

A plane wave advances its phase from crest to crest, and the phase gained across a displacement counts how many wavefronts that displacement crosses, showing the other two properties: freedom from a metric and transport under a linear deformation. In [crystal diffraction](../examples/optics/crystal_diffraction/crystal_diffraction.ipynb) and [graphene](../examples/quantum/graphene/graphene.ipynb) examples, the wavevector is modelled as the extrinsic plane of its wavefronts and the displacement between two atoms as intrinsic, so their pairing gives the phase directly and the deformation's adjugate transports the wavevector:

```python
phase = wavevector & displacement                                        # [] Scalar+: 2.0
moved_displacement = deformation(displacement)                           # [] Antivector-
moved_wavevector = deformation.adjugate().solve(wavevector)              # [] Vector+
moved_phase = moved_wavevector & moved_displacement                      # [] Scalar+: 2.0
```

The pairing `wavevector & displacement` uses no metric, so the same line counts phase across the three bonds of a honeycomb sheet in the plane, across a crystal in space, and across an event in spacetime, where time squares to minus one. Under any linear map `deformation` on displacements, its adjugate pulls a wavevector back so that `deformation.adjugate()(wavevector) & displacement == wavevector & deformation(displacement)`, even when `deformation` flattens a volume onto a plane and has no inverse. When an invertible stretch, shear and turn deforms a crystal, solving forward through `deformation.adjugate()` carries each reciprocal-lattice wavevector so that its phase at every displaced atom stays what it was: doubling the spacing along x halves the wavevector component along x.

<p align="center"><img src="../plots/crystal_diffraction.png" alt="A square crystal layer, stretched along x, and stretched, sheared and turned, above the scattered intensity and reciprocal lattice peaks of each" width="318" /></p>

*Figure: a square layer of atoms (top left), stretched along x (top centre), and stretched, sheared and turned (top right), above the scattered intensity of each; circles mark the reciprocal wavevectors carried by solving through the deformation's adjugate.*

In the vector notation of solid-state physics and wave optics, the phase is written as the Euclidean dot product $\delta\phi = \mathbf{k}\cdot\delta\mathbf{x}$, and kept unchanged under an invertible lattice deformation by defining the reciprocal lattice $\mathbf{b}_i \cdot \mathbf{a}_j = 2\pi\delta_{ij}$ [[6, §2.2.2]](#references), where the reciprocal lattice is distinguished from the direct lattice by its inverse-length dimensions while both are written as three-component vectors. In three dimensions that reciprocal basis is commonly written with the cross and triple products $\mathbf{b}_1 = 2\pi(\mathbf{a}_2\times\mathbf{a}_3)/[\mathbf{a}_1\cdot(\mathbf{a}_2\times\mathbf{a}_3)]$, which use three-dimensional vector products and introduce the Euclidean metric twice, once in $\mathbf{a}_2\times\mathbf{a}_3$ and once in $\mathbf{k}\cdot\delta\mathbf{x}$, before the two factors cancel in the phase.

## One perspective across languages

The vector notation stores the wheel's rates and momenta, and the crystal's displacements and wavevectors, as vectors of one space. In its own setting that gives the right numbers: in orthonormal three-dimensional Euclidean coordinates under proper rotations, the Euclidean metric has the identity matrix, rotations equal their own inverse transpose (`F == F.adjugate().inverse()`), and transformations have determinant +1. In that setting an intrinsic quantity and its extrinsic counterpart transform with identical coordinates, so storing both as vectors of one space produces the same numbers. Leaving that setting separates the transformation laws of the two, whether through a non-rigid stretch, a camera projection that has no inverse, a reflection, an indefinite or degenerate metric, a state space whose variables share no unit, or a change of dimension. With each quantity's grade and orientation chosen from how it is built and how it acts ([Appendix: representations](#appendix-representations)), its type records which side of the pairing it belongs to from the start.

Each discipline that works outside that setting has built its own formalism for the distinction between a quantity and its complementary counterpart: covariant and contravariant vectors, reciprocal lattices, $V/V^*$ and $\Sigma/\Sigma^{-1}$ pair dual vector spaces without distinguishing intrinsic and extrinsic orientation; untwisted and twisted differential forms track grade and orientation on fields; screw theory ($\mathfrak{se}(3)/\mathfrak{se}(3)^*$) pairs rigid-body twists with wrenches on a six-dimensional space, and Lie–Poisson mechanics ($\mathfrak{g}/\mathfrak{g}^*$) pairs symmetry generators with dual momenta on the Lie algebra of a configuration group; axial vectors, inverse-transpose normals and cofactor matrices supply reflection signs and area transport to three-dimensional vector coordinates. In a graded algebra with orientation on every subspace and typed duality maps, `&` expresses those pairings within one algebraic representation. Each row below pairs a domain's established dual construction with its complementary reading here and links to its section in the [domains appendix](#appendix-domains):

| Domain | Established dual pair | Complementary reading |
| --- | --- | --- |
| [Surface normals and area](#surface-normals-and-area-transport) | Cutting plane and area element | `flux & areas(patch)` |
| [Projective cameras](#projective-cameras-and-multiview-geometry) | Projective space and dual space | `cones(point) & point` |
| [Projective quadrics](#projective-quadrics-and-non-euclidean-distance) | Point locus and tangent envelope | `quadric(point) & point` |
| [Rigid-body mechanics](#rigid-body-mechanics-and-inertia) | Twists and wrenches | `rate & inertia(rate)` |
| [Continuum elasticity](#continuum-stress-strain-and-fault-slip) | Covariant strain and contravariant stress | `face & stress(face)` |
| [Simplicial topologies](#simplicial-topologies-and-finite-elements) | Primal and dual cochains | `gradient & direction` |
| [Fluid vorticity](#fluid-vorticity-and-circulation) | Vorticity 2-form and surface patch | `vorticity & patch` |
| [Wave phase and diffraction](#wave-phase-and-crystal-diffraction) | Reciprocal and direct lattices | `wavevector & displacement` |
| [Electrodynamics](#electrodynamics-and-crystal-optics) | Faraday 2-form and excitation | `field & excitation` |
| [Spin and chiral media](#spin-precession-and-chiral-media) | Magnetic field and spin moment | `field & spin` |
| [Gravitational waves](#gravitational-waves-and-spacetime-curvature) | Metric strain and arm separation | `strain(arm) & arm` |
| [Symplectic geometry](#state-changes-and-conjugate-potentials) | Conjugate potentials and state variations | `potential & change` |
| [Information geometry](#uncertainty-principal-components-and-gradient-descent) | Readout differential and covariance | `readout & covariance(readout)` |
| [Thermodynamics](#thermodynamics-and-heat-transport) | Intensive affinities and state variations | `entropy_gradient & mobility(entropy_gradient)` |
| [Quantum measurements](#quantum-states-and-measurements) | POVM effects and density operators | `effect & state` |

## Conclusion

Within geometric algebra, projective geometric algebra made this complementary structure common practice, and it did so out of necessity, because its pseudoscalar squares to zero. Gunn's axes and spears [[5]](#references), Dorst's projective dual algebras with intrinsic and extrinsic orientations [[7]](#references) and the antiproduct of Lengyel's Projective Geometric Algebra [[8]](#references) agree, where the argument of this document is concerned, that a degenerate algebra needs elements of complementary grades side by side, and the regressive product `&` that pairs them beside the outer product `^`. The form set out here carries that shared structure from projective geometry into physics, continuum mechanics and estimation, in Euclidean and spacetime algebras as much as in degenerate ones: scalar observables are complementary pairings, `extrinsic & intrinsic`, and physical laws, uncertainties and metrics are duality maps between complementary types, `a & duality(a)`.

## Companion documents

- [Extensors through multiple lenses](multiple_lenses.md): multilinear maps between multivector spaces, constructed from geometric-algebra products.
- [Why orientation](orientation.md): the mirror test and how intrinsic and extrinsic orientations arise from joins and meets.
- [The regressive product](regressive_product.md): the regressive product from its metric-free volume definition to geometric joins, partial contractions and linear maps.
- [Extensors, linear algebra, tensors and forms](frameworks.md): how linear algebra, tensors, exterior forms and specialized algebras sit inside geometric algebra and extensors.
- [Extensors and common practice](common_practice.md): comparisons with conventional matrix and tensor formulations in applied problems.
- [Discrete differential geometric algebra](ddga.md): calculus and material pullbacks on a simplicial topology from the simplex gradient.
- [Geometry and quantities](thermodynamics.md): extensive material quantities joined to spacetime in one algebra, and the coupled transport that follows.
- `numga` documentation: [Orientation](../docs/orientation.md), the type constructors and propagation rules for `+` and `-`.

## Appendix: domains

This appendix applies the ansatz to a range of domains in geometry, mechanics and physics. Each section states the domain's readings as `extrinsic & intrinsic`, sets them beside the formulation the domain itself uses, and links `numga` examples that run:

- Geometry
  - [Surface normals and area transport](#surface-normals-and-area-transport)
  - [Projective cameras and multiview geometry](#projective-cameras-and-multiview-geometry)
  - [Projective quadrics and non-Euclidean distance](#projective-quadrics-and-non-euclidean-distance)
- Mechanics
  - [Rigid-body mechanics and inertia](#rigid-body-mechanics-and-inertia)
  - [Continuum stress, strain and fault slip](#continuum-stress-strain-and-fault-slip)
  - [Simplicial topologies and finite elements](#simplicial-topologies-and-finite-elements)
  - [Fluid vorticity and circulation](#fluid-vorticity-and-circulation)
- Fields
  - [Wave phase and crystal diffraction](#wave-phase-and-crystal-diffraction)
  - [Electrodynamics and crystal optics](#electrodynamics-and-crystal-optics)
  - [Spin precession and chiral media](#spin-precession-and-chiral-media)
  - [Gravitational waves and spacetime curvature](#gravitational-waves-and-spacetime-curvature)
- State spaces
  - [State changes and conjugate potentials](#state-changes-and-conjugate-potentials)
  - [Uncertainty, principal components and gradient descent](#uncertainty-principal-components-and-gradient-descent)
  - [Thermodynamics and heat transport](#thermodynamics-and-heat-transport)
  - [Quantum states and measurements](#quantum-states-and-measurements)

### Geometry

#### Surface normals and area transport

A surface's positive side, oriented across the surface, and a surface patch joined from edges, oriented within the surface, respond differently to reflections and deformations, as [surface normals](../examples/orientation/normals/normals.ipynb), [area transport](../examples/mechanics/area_transport/area_transport.ipynb) and [freeform elasticity](../examples/mechanics/freeform/freeform.ipynb) examples develop. Under a linear deformation, an intrinsic patch is carried forward by the deformation's outermorphism, while an extrinsic flux density reading that patch is pulled backward by the adjugate of the area map:

```python
areas = deformation.outermorphism(Antibivector)  # [] Antibivector- <- Antibivector-
flux_pullback = areas.adjugate()                 # [] Bivector+ <- Bivector+
preserved_flux = flux & areas(patch)             # [] Scalar+ (equals flux_pullback(flux) & patch)
```

A flux reads a carried patch just as its pullback `flux_pullback(flux)` reads the original patch, a vertical mirror keeps a horizontal triangle's upper side up while reversing its corner winding, and both `areas` and `flux_pullback` remain defined when `deformation` flattens a volume onto a plane and has no inverse.

In the matrix and cofactor notation of continuum mechanics, oriented area elements are transported by the determinant-weighted relation $\mathbf{n}\,\mathrm{d}a = (\det F)F^{-\mathsf T}\mathbf{N}\,\mathrm{d}A = \operatorname{cof}(F)\mathbf{N}\,\mathrm{d}A$ [[9, §1.7]](#references), whereas a cutting plane covector is carried forward by the inverse transpose $F^{-\mathsf T}$ and pulled back by the transpose $F^{\mathsf T}$. When both the cutting plane and the area patch are represented in $\mathbb{R}^3$ by a single normal vector $\mathbf{N}$ built from the cross product $\mathrm{d}\mathbf{X}_1 \times \mathrm{d}\mathbf{X}_2$, that vector transforms by $F^{-\mathsf T}$ when tracking a surface's outward side, which is undefined when $\det F = 0$ and reverses the side under a reflection, and by $\operatorname{cof}(F)$ when transporting flux across a patch.

#### Projective cameras and multiview geometry

A pinhole camera is a projective map from scene points to sensor points that collapses depth onto the sensor plane, and each pixel's measurement uncertainty is a polarity on that sensor. Because the camera loses the depth dimension, it has no inverse. Its adjugate still pulls polar planes back through the pinhole, pulling each pixel disc back into a sight cone on scene points in [multiview reconstruction](../examples/estimation/multiview/multiview_reconstruction.ipynb), [projective cameras and shadows](../examples/geometry/projection/projection.ipynb), [lens camera](../examples/optics/lens_camera/lens_camera.ipynb) and [rendering on the three-sphere](../examples/quadrics/s3_raytracer/s3_raytracer.ipynb) examples:

```python
back_projection = camera.adjugate()              # [] Vector+ <- Vector+
cones = back_projection(discs(camera))           # [] Vector+ <- Antivector-
scene_cost = cones(Point) & Point                # [] Scalar+ <- (Antivector-, Antivector-)
```

Both `discs` and `cones` take points to their polar hyperplanes, and triangulation and camera alignment take the same form in any dimension and any metric signature (Euclidean, elliptic or hyperbolic) without inverting the projection or dividing by homogeneous coordinates.

In the homogeneous matrix notation of projective computer vision, points and planes sit in dual projective spaces $\mathbb{P}^3$ and $(\mathbb{P}^3)^*$, and an image point back-projects through a camera centre to a ray in space [[10, §9.1]](#references). In that matrix representation, quadric pullbacks $P^{\mathsf T}DP$, skew-symmetric $4\times 4$ Plücker line matrices $L = \mathbf{A}\mathbf{B}^{\mathsf T} - \mathbf{B}\mathbf{A}^{\mathsf T}$ with their dual matrices $L^*$, and inhomogeneous $3\times 3$ rotation blocks with translation vectors are written as separate matrix constructions rather than elements and maps of one algebra.

#### Projective quadrics and non-Euclidean distance

A quadric locus is defined by a symmetric duality map that assigns to each point a hyperplane, and a point lies on the surface when `quadric(point) & point == 0`. When that symmetric map is nondegenerate it is a polarity: each surface point's image is its tangent plane, and the inverse polarity describes the same surface as an envelope of tangent planes. Choosing an absolute quadric supplies distances and angles to projective space in [projective quadrics](../examples/quadrics/quadrics/quadrics.ipynb), [Cayley–Klein geometry](../examples/quadrics/cayley_klein/cayley_klein.ipynb), [quadric collision](../examples/quadrics/quadric_collision/quadric_collision.ipynb), [Poncelet's porism](../examples/math/poncelet/poncelet.ipynb) and [Pascal's theorem](../examples/math/pascal/pascal.ipynb) examples:

```python
locus_test = quadric(point) & point              # [] Scalar+: 0 on the surface
tangent_test = plane & envelope(plane)           # [] Scalar+: 0 when tangent
```

Under a projective transformation `T` on points, point loci pull back as `T.adjugate()(quadric(T))` while dual envelopes push forward as `T(envelope(T.adjugate()))`, keeping incidence and contact invariant without a background metric. In that [Cayley–Klein geometry](../examples/quadrics/cayley_klein/cayley_klein.ipynb) example, supplying different symmetric absolute maps to the same `absolute(Point) & Point` pairing constructs elliptic, Euclidean, hyperbolic and relativistic metrics from one incidence expression.

In the homogeneous matrix notation of classical projective geometry, maps of $\mathbb{P}^n$ to itself are collineations and maps between $\mathbb{P}^n$ and its dual $(\mathbb{P}^n)^*$ are correlations, with nondegenerate correlations of period two called polarities [[11]](#references); point loci read as $x^{\mathsf T}Ax = 0$, nondegenerate tangent envelopes read as $u^{\mathsf T}Bu = 0$, and Euclidean perpendicularity is induced by an elliptic polarity in the hyperplane at infinity [[12]](#references). In unoriented vector-based geometric algebra, points and planes are instead both placed in grade 1 and paired with the metric inner product.

### Mechanics

#### Rigid-body mechanics and inertia

A rigid body's rate of motion is an extrinsic bivector carrying turns and slides together in projective space, and it induces the velocity `points.commutator(Rate)` at each mass point. Joining each mass point with its mass-weighted velocity sums the inertia, a duality map from rates to momenta, and pairing the rate with the momentum gives the kinetic energy in [energy in a mirror](../examples/orientation/mirror_energy/mirror_energy.ipynb), [mechanical work in a mirror](../examples/orientation/mechanical_work/mechanical_work.ipynb), [spinning top](../examples/mechanics/spinning_top/spinning_top.ipynb), [robot arm kinematics](../examples/mechanics/robot_arm/robot_arm.ipynb), [stiffness and normal modes](../examples/mechanics/modes/modes.ipynb) and [flexible bodies, rigidly joined](../examples/mechanics/modal_xpbd/modal_xpbd.ipynb) examples:

```python
points = mv.point([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])             # [points] Antivector-
inertia = points.regressive(points.commutator(Rate) * masses).sum()  # [] Antibivector- <- Bivector+
rate = mv.xy + mv.yw                                               # [] Bivector+
momentum = inertia(rate)                                           # [] Antibivector-

kinetic_energy = (rate & momentum) / 2                             # [] Scalar+: 2.0
mirrored_energy = ((mirror >> rate) & (mirror >> momentum)) / 2    # [] Scalar+: 2.0
```

The same lines give the kinetic energy in the projective plane, in projective three-space and in curved elliptic space, whereas `(rate | momentum) / 2` does not produce an ordinary scalar and `(rate | momentum.dual()) / 2` flips the sign of the rotational part while dropping the translational energy (`-1.0` instead of `2.0`). Reflecting `rate` and `momentum` with `>>` preserves the positive kinetic energy (`2.0`), just as rebuilding the inertia from the mirrored points does, whereas reflecting `rate` and the already-built `momentum` with a uniform grade-involuted sandwich `mirror * x.involute() * mirror.inverse()` flips the sign of `momentum` relative to the mirrored body and yields `-2.0`.

In three-dimensional vector notation, angular velocity and angular momentum are axial vectors whose mirror behaviour is distinguished from displacements [[1, §52-5]](#references), transforming with the determinant sign $\mathbf{L} \mapsto (\det R)R\mathbf{L}$ and paired through the rotational inertia tensor $I_{ij} = \sum m(r^2\delta_{ij} - r_i r_j)$; that tensor is restricted to three dimensions, separates translation from rotation, and carries the subtracted metric trace $r^2\delta_{ij}$ because planes of rotation are stored as normal axes. In projective geometric algebra, the dual inertia between velocity states and momentum spears and the metric-free regressive pairing for kinetic energy and power are formulated on dual Grassmann algebras across Cayley–Klein spaces [[2, §6]](#references), [[3, Ch. 9]](#references), in single-algebra plane-based form [[4, §2.7]](#references), and in degenerate-metric duality comparisons [[5, §8.6.1]](#references). Without orientation on the types, applying the single grade-involuted sandwich above to both `rate` and `momentum` flips the sign of `momentum` relative to the mirrored body and produces the negative kinetic energy `-2.0` checked in that [energy in a mirror](../examples/orientation/mirror_energy/mirror_energy.ipynb) example.

#### Continuum stress, strain and fault slip

In a deformable solid, linear strain is the change to the metric, a duality map from displacements to planes, while Cauchy stress maps a cut plane to the traction force across it, with the normal traction read by `faces & stress(faces)`. In an [earthquake beachballs](../examples/mechanics/earthquake/earthquake.ipynb) example's model of fault slip, a shear slip is constructed as a symmetric map built directly from the fault plane and the auxiliary plane across the slip, whose rotational curl vanishes without a metric and whose P-wave first motion along a ray direction is `moment(directions) & directions`. Together with [stress in an elastic cube](../examples/mechanics/stress/stress.ipynb), [crystal waves](../examples/mechanics/crystal_waves/crystal_waves.ipynb) and [photoelasticity](../examples/optics/photoelasticity/photoelasticity.ipynb) examples, these notebooks build the maps directly:

```python
normal_traction = faces & stress(faces)                              # [faces] Scalar+
values, principal_planes = (Vector & stress(Vector)).eigh(Vector | Vector)  # [3] Scalar+, [3] Vector+

moment = auxiliaries * (faults & Antivector) + faults * (auxiliaries & Antivector)  # [] Vector+ <- Antivector-
curl = (Vector ^ moment(Antivector)).trace(1, 2)                     # [] Bivector+: 0
p_wave = moment(directions) & directions                             # [rays] Scalar+
```

Building the fault-slip map `moment` in that model, checking that its curl vanishes, and reading `p_wave` or `normal_traction` use no metric; the metric enters in the isotropic Hooke law linking `strain` to `stress` and as the comparison form `.eigh(Vector | Vector)` when finding principal stress planes.

In the tensor notation of continuum elasticity, strain is the covariant pullback change of the metric $g_{ij}$ [[9, §1.3]](#references), and Cauchy stress $\sigma^{ij}$ maps a cut-plane covector $n_j$ to a traction vector $t^i = \sigma^{ij}n_j$ with stress power $\sigma : d$ in the local energy balance [[13, §2.3]](#references), while principal stresses solve the metric pencil $\det(\sigma^{ij} - \lambda g^{ij}) = 0$. Engineering elasticity in orthonormal Cartesian coordinates lowers all indices to $\varepsilon_{ij}$ and $\sigma_{ij}$, and writes the curl in three dimensions with the Levi-Civita density $\varepsilon^{ijk}\partial_j u_k$.

#### Simplicial topologies and finite elements

On a simplicial topology, a potential's simplex gradient is the extrinsic plane whose pairing with each edge direction equals the coboundary difference across that edge ([Discrete differential geometric algebra](ddga.md)). On a top-dimensional or projective simplex, dividing each corner's opposing boundary facet by its pairing with that corner gives the corner simplex gradient directly from `&` and scalar division, without an inner product or metric inverse: `(b & c) / (a & b & c)` on a projective triangle, and `(b & c & d) / ((b & c & d) & a)` on a projective tetrahedron. Wedging two corner simplex gradients gives an alternating edge element `(gradient_a ^ gradient_b) & (point & step)`, and pulling a material's conductivity back through the assembled `gradient` via `.adjugate()` forms the discrete conductance operator in [Whitney forms](../examples/surfaces/whitney_forms/whitney_forms.ipynb), [eddy-current braking](../examples/electromagnetism/eddy_brake/eddy_brake.ipynb), [projective sphere flow](../examples/fluids/projective_sphere_flow/projective_sphere_flow.ipynb), [Stokes flow on a triangulated surface](../examples/fluids/dec_stokes/dec_stokes.ipynb) and [force-free Beltrami fields](../examples/electromagnetism/beltrami_fields/beltrami_fields.ipynb) examples:

```python
gradient_a = (b & c) / (a & b & c)                                # [] Vector+
edge_reading = (gradient_a ^ gradient_b) & (point & step)         # [] Scalar+

gradient = (spdiag(gradients) * topology.incidence[2, 0]).sum(axis=0) * Scalar  # Vector+[faces] <- Scalar+[vertices]
conductance = gradient.adjugate()(spdiag(areas * conductivity)(gradient))       # Antiscalar-[vertices] <- Scalar+[vertices]
```

Because the volume join of a top-dimensional or projective simplex is a single intrinsic top-grade component, dividing the opposing boundary join by its pairing with the corner uses only `&` and scalar division in any dimension; restricting a projective gradient to the tangent hyperplane of an embedded surface or choosing the isotropic conductivity `Vector.dual()` is where a metric enters.

In the exterior-calculus notations of discrete complexes, finite element exterior calculus [[14, §§1.3, 5]](#references) separates the metric-free exterior derivative $d$ from inner-product mass matrices on Whitney basis forms $\lambda_i\,\mathrm{d}\lambda_j - \lambda_j\,\mathrm{d}\lambda_i$. Discrete exterior calculus [[15, §6, Definition 6.1]](#references) instead defines a diagonal Hodge star $\star_k$ between primal simplices and circumcentric dual cells from the ratio of dual to primal cell measures, coupling the metric operator to a dual complex whose circumcentres can lie outside obtuse simplices.

#### Fluid vorticity and circulation

In an ideal fluid, velocity is an intrinsic tangent displacement rate while vorticity is an extrinsic bivector whose pairing with an intrinsic surface patch reads Kelvin's circulation around the patch boundary, `vorticity & patch`. On a curved triangulated surface, meeting each mode's simplex gradient with a triangle's patch gives its area-weighted tangent flow across the triangle, and pairing another mode's gradient with that flow gives the Lie–Poisson advection coupling on each face in [Laplacian eigenfunction fluids](../examples/fluids/eigen_fluids/eigen_fluids.ipynb), [leapfrogging vortices](../examples/fluids/vortices/vortices.ipynb), [smoke rings](../examples/fluids/smoke_rings/smoke_rings.ipynb) and [Kirchhoff's equations](../examples/fluids/kirchhoff/kirchhoff.ipynb) examples:

```python
circulation = vorticity & patch                                    # [] Scalar+
flows = face_gradients ^ patches                                   # [modes] Antivector-[faces]
jacobians = face_gradients[:, None] & flows[None, :]               # [modes, modes] Scalar+[faces]
```

Forming `flows = face_gradients ^ patches` and pairing `face_gradients & flows` uses neither a surface normal vector nor a metric; the metric enters only in the Laplace–Beltrami eigenproblem that links vorticity to its streamfunction.

In the exterior-calculus notation of geometric hydrodynamics, vorticity is an exact 2-form $\omega = \mathrm{d}u^\flat$ whose integral over an advected surface patch gives Kelvin's circulation $\Gamma = \int_\Sigma \omega$, derived by pullback in Euler–Poincaré continuum mechanics [[16, §6, Theorem 6.2]](#references), while two-dimensional Lie–Poisson advection on a closed surface weights the Jacobian wedge $\mathrm{d}\psi \wedge \mathrm{d}\zeta$ by the vorticity field under appropriate boundary conditions. In three-dimensional vector calculus on a curved surface, vorticity is written as the axial vector $\boldsymbol{\omega} = \nabla\times\mathbf{u}$, circulation as $\Gamma = \iint \boldsymbol{\omega}\cdot\mathbf{n}\,\mathrm{d}A$, and the surface Jacobian as $\{\psi, \zeta\} = (\nabla\psi \times \nabla\zeta)\cdot\mathbf{n}$, introducing metric gradients $\nabla\psi$, $\nabla\zeta$ and a unit normal $\mathbf{n}$ that the exterior pairing `face_gradients & flows` above does not use.

### Fields

#### Wave phase and crystal diffraction

As shown in [Wave phase under a deformation](#wave-phase-under-a-deformation), an extrinsic wavevector reads an intrinsic displacement into a phase change without a metric, `wavevector & displacement`, in [crystal diffraction](../examples/optics/crystal_diffraction/crystal_diffraction.ipynb), [graphene](../examples/quantum/graphene/graphene.ipynb) and [specular caustics](../examples/optics/specular_caustics/specular_caustics.ipynb) examples:

```python
displacement = mv.vector([1.0, 2.0, 3.0]).dual()               # [] Antivector-
wavevector = mv.vector([2.0, 0.0, 0.0])                        # [] Vector+
stretch = Antivector + mv.x.dual() * (mv.x & Antivector)       # [] Antivector- <- Antivector-

moved_displacement = stretch(displacement)                     # [] Antivector-
moved_wavevector = stretch.adjugate().solve(wavevector)        # [] Vector+
preserved_phase = moved_wavevector & moved_displacement        # [] Scalar+: 2.0
```

Solving through `stretch.adjugate()` halves the wavevector component along the stretched axis as the displacement and wavefront spacing double, keeping every Bragg peak's phase at every atom unchanged in the plane, in space and in spacetime alike.

In the vector notation of solid-state physics and wave optics, the phase is written as the Euclidean dot product of three-component vectors, $\delta\phi = \mathbf{k}\cdot\delta\mathbf{x}$, and preserved under an invertible lattice deformation by defining the reciprocal lattice $\mathbf{b}_i \cdot \mathbf{a}_j = 2\pi\delta_{ij}$ [[6, §2.2.2]](#references), where reciprocal and direct lattice vectors are distinguished by their inverse-length and length units while sharing three-component vector notation. In three dimensions that reciprocal basis is commonly written from the cross and triple products $\mathbf{b}_1 = 2\pi(\mathbf{a}_2\times\mathbf{a}_3)/[\mathbf{a}_1\cdot(\mathbf{a}_2\times\mathbf{a}_3)]$, which use three-dimensional vector products and introduce the Euclidean metric twice, once in $\mathbf{a}_2\times\mathbf{a}_3$ and once in $\mathbf{k}\cdot\delta\mathbf{x}$, before the two factors cancel in the phase.

#### Electrodynamics and crystal optics

In spacetime (`x+ y+ z+ t-`), the electromagnetic field is an extrinsic bivector, read across the intrinsic spacetime plane `bond & time_direction` that a spatial direction sweeps along an observer's clock ([Electric and magnetic fields](#electric-and-magnetic-fields)). Its excitation is an intrinsic antibivector, and the vacuum or a crystal medium is a duality map between the two, the constitutive map, in [constitutive media](../examples/electromagnetism/constitutive/constitutive.ipynb), [Fresnel wave surfaces](../examples/electromagnetism/fresnel/fresnel.ipynb), [second-harmonic generation](../examples/electromagnetism/second_harmonic/second_harmonic.ipynb) and [piezoelectric bimorph](../examples/electromagnetism/piezoelectric/piezoelectric.ipynb) examples:

```python
swept_plane = bond & time_direction              # [] Antibivector-
field = mv.xt * 2.0 + mv.xy                      # [] Bivector+
reading = field & swept_plane                    # [] Scalar+: 2.0

vacuum = Bivector.dual()                         # [] Antibivector- <- Bivector+
excitation = vacuum(field)                       # [] Antibivector-
lagrangian = (field & excitation) / 2            # [] Scalar+
```

Evaluating `field & swept_plane` and taking coboundary derivatives in Maxwell's conservation laws use no metric; the spacetime metric enters explicitly in the vacuum's constitutive map, `vacuum = Bivector.dual()`, the algebra's own duality map, or is replaced there by a crystal's dielectric and magnetic response map.

In the tensor and differential-form notations of relativistic and premetric electrodynamics, the metric-free conservation laws $\mathrm{d}F = 0$ and $\mathrm{d}H = J$ keep the Faraday 2-form $F$ distinct from the twisted excitation 2-form $H$, while the constitutive relation $H = \kappa(F)$ of the vacuum or crystal medium introduces the metric or material response [[17, §§2, 6]](#references). Spacetime geometric algebra splits a single spacetime bivector into an observer's relative electric and magnetic bivectors through the observer's timelike axis [[18]](#references), whereas three-dimensional Gaussian vector notation identifies $\mathbf{D} = \mathbf{E}$ and $\mathbf{H} = \mathbf{B}$ in vacuum and standard exterior-form treatments place $F \in \Omega^2(M)$ and $H \in \Omega^2_-(M)$ in separate form bundles rather than one geometric algebra with Lorentz rotors.

#### Spin precession and chiral media

A spin or magnetic moment is an angular momentum, the intrinsic patch of the plane it circulates in, whereas the magnetic field that turns it is an extrinsic rate ([Spin](#spin)). Their Zeeman interaction energy is the complementary pairing `field & spin`, and the Larmor precession rate is `spin.commutator(field)`. When two intrinsic quantities pair directly, such as a nuclear spin and an emitted electron's direction or a light wave's polarization plane and propagation direction, their pairing `spin & direction` is a handed intrinsic scalar, which multiplies a handed coupling constant such as weak parity violation or optical rotatory power to produce an ordinary scalar rate, in [magnetic resonance](../examples/quantum/magnetic_resonance/magnetic_resonance.ipynb), [Wu experiment](../examples/orientation/wu_experiment/wu_experiment.ipynb), [optical rotation](../examples/orientation/optical_rotation/optical_rotation.ipynb), [two spins](../examples/quantum/two_spins/two_spins.ipynb), [material symmetry](../examples/mechanics/symmetry/symmetry.ipynb) and [superconductivity](../examples/quantum/superconductivity/superconductivity.ipynb) examples:

```python
spin = mv.yz.dual() + mv.xy.dual() * 0.5            # [] Antibivector-
field = mv.xy * 2.0                                 # [] Bivector+
zeeman_energy = field & spin                        # [] Scalar+: 1.0
precession = spin.commutator(field)                 # [] Antibivector-

emission_rate = 1 + asymmetry * (spin & direction)  # [] Scalar+ (Scalar- * Scalar-)
```

Under a volume-preserving shear that tilts x toward z in this model, the spin's spanning directions move forward and the field's planes pull back, leaving `field & spin` unchanged (`1.0`), whereas typing spin as a rate and taking the metric dot product `field.dot(spin)` changes the energy (`1.25`). Under a mirror, `field & spin`, `asymmetry * (spin & direction)` and `spin.commutator(field)` all transform without axial-vector sign rules.

In the three-dimensional vector notation of quantum mechanics and magnetic resonance, the Zeeman coupling reads as $H = -\boldsymbol{\mu}\cdot\mathbf{B}$ and Bloch precession as $\dot{\mathbf{s}} = \gamma\,\mathbf{s}\times\mathbf{B}$, with axial-vector parity assigned to both $\mathbf{s}$ and $\mathbf{B}$ so their dot and cross products survive spatial inversion, and pseudoscalar sign rules attached to weak-decay $\mathbf{J}\cdot\mathbf{p}$ couplings whose experimental parity violation was demonstrated in beta decay [[19, Part II]](#references). Spacetime geometric algebra identifies the spin plane and its bivector generator directly in spacetime [[18, §VII.D]](#references). When spin and magnetic field are instead modelled as grade-1 vectors of an unoriented three-dimensional spatial algebra and reflected by a uniform vector sandwich while dualizing the field with the spatial pseudoscalar $I = \sigma_1\sigma_2\sigma_3$ in $(I\mathbf{B})\times\mathbf{s}$, the mirrored precession reverses because every spatial reflection negates $I$, as a [spatial-vector magnetic resonance twin](../examples/quantum/magnetic_resonance/magnetic_resonance_space.ipynb) example demonstrates side by side.

#### Gravitational waves and spacetime curvature

In general relativity, the Riemann curvature is a duality map from an intrinsic spacetime loop area to an extrinsic rotation-and-boost generator. Restricting both sides by an observer's clock direction yields the spatial tidal map, which in weak-field linearized gravity has the same type as the gravitational-wave metric strain: the tidal map is minus the second time derivative of `strain` with no metric conversion between them, an interferometer arm short compared with the gravitational wavelength reads the passing wave as `(strain(arm) & arm) / 2`, and a pair of nearby test masses reads the tidal stretch as `tidal(separation) & separation`. On a simplicial topology, symmetric products of each simplex's corner gradients build its metric directly from the squared edge lengths, in [gravitational-wave curvature](../examples/relativity/curvature/curvature.ipynb), [LIGO antenna patterns](../examples/relativity/ligo/ligo.ipynb), [gravitational waves of a closed universe](../examples/relativity/gravitational_waves/waves.ipynb), [Hopf-projected gravitational waves](../examples/relativity/gravitational_waves/hopf_projection.ipynb), [Regge calculus](../examples/surfaces/regge_calculus/regge_calculus.ipynb) and [satellite navigation](../examples/relativity/gps/gps.ipynb) examples:

```python
arm_reading = (strain(arm) & arm) / 2                                         # [] Scalar+
tidal_reading = tidal(separation) & separation                                # [] Scalar+
basis = -(first * (second & Antivector) + second * (first & Antivector)) / 2  # [] Vector+ <- Antivector-
kinetic_energy = (rates & masses(rates)) / 2                                  # [] Scalar+
```

Along each edge of a simplex, its tail and head corner gradients pair to −1 and +1 while the remaining corner gradients vanish, so their symmetric product `basis` reads that edge's squared length alone in any dimension. In that [gravitational waves of a closed universe](../examples/relativity/gravitational_waves/waves.ipynb) example on the 600-cell's tetrahedral three-sphere in four-dimensional space (`x+ y+ z+ w+`), each spherical tetrahedron's inverse cone metric pairs its face planes into dihedral cosines and edge deficit angles, while the inertia `masses` and the linearized Regge `stiffness` are duality maps from edge-length rates and perturbations to volume-integrated momenta and forces, pulled back through vertex displacements to project out pure vertex motions.

In tensor notation, geodesic deviation reads as $\ddot{\xi}^i = -R^i{}_{0j0}\xi^j$ with the mixed Riemann tensor and linearized long-wavelength detector response reads as $\Delta L/L = \frac{1}{2}h_{ij}^{\mathrm{TT}}n^i n^j$ [[20, §3]](#references), while lattice general relativity concentrates curvature on codimension-two hinges through deficit angles determined by simplicial edge lengths [[21, §II]](#references). Raising the first index on $R^i{}_{0j0} = g^{ik}R_{k0j0}$ folds in the background metric $g^{ik}$ so that the mixed tensor maps displacements to accelerations, whereas the unraised tidal components $R_{i0j0}$ and the linearized strain rate $-\frac{1}{2}\ddot{h}_{ij}^{\mathrm{TT}}$ are both covariant $(0,2)$ forms on separations.

### State spaces

#### State changes and conjugate potentials

In [rubber band](../examples/thermodynamics/rubber_band/rubber_band.ipynb) and [constrained heat capacity](../examples/thermodynamics/state_space/state_space.ipynb) examples, an equilibrium state combines intrinsic quantities such as spatial position or volume with entropy, while its conjugate potential combines minus the mechanical force or pressure with the temperature. Pairing the potential with a state variation reads the first-order change of stored energy (`potential & change`), summed across vertices when the state is a field on a topology. Allowing states and potentials to vary independently doubles the space, and on two variations `(state_a, potential_a)` and `(state_b, potential_b)` the cross-pairing `potential_b & state_a` splits into an alternating difference and a symmetric average:

```python
work = potential & change                                            # [] Scalar+
omega = (potential_b & state_a) - (potential_a & state_b)            # [] Scalar+
symmetric = ((potential_b & state_a) + (potential_a & state_b)) / 2  # [] Scalar+
potential_a, potential_b = material(state_a), material(state_b)      # [] Vector+, [] Vector+
curvature = material(state_a) & state_b                              # [] Scalar+
```

The alternating difference `omega` is the canonical symplectic two-form on the combined state–potential space, while the average `symmetric` is a split-signature bilinear form with equally many positive and negative directions.[^symmetric-affine] Both are nondegenerate because every nonzero variation pairs nontrivially with a complementary one, `omega` is closed because its coefficients are constant in affine state and potential coordinates, and neither form uses a metric on the state space, including along the null entropy direction.

An energy law selects one potential for each state, and its derivative is the full thermodynamic response `material` coupling spatial deformation and entropy in the band. Because `material` is the second derivative of one energy, it equals its own adjugate (`material == material.adjugate()`), making the two cross-pairings equal along the half-dimensional constitutive graph where potential variations follow `material(state)`.

Restricting the two ambient forms to that constitutive graph separates reciprocity from stability. The difference `omega` vanishes on the graph, making it a Lagrangian submanifold and giving the Maxwell relation between stretch heating and thermal tension. The average `symmetric` restricts on the same graph to the energy Hessian `curvature`, which is positive semidefinite for a convex energy up to neutral motions such as rigid translations of the band.

In the exterior-calculus notation of symplectic geometry, cotangent coordinates $(x^i, \xi_i)$ on $T^*X$ carry the tautological Liouville one-form $\alpha = \sum_i \xi_i\,\mathrm{d}x^i$ and the canonical symplectic two-form $\omega = -\mathrm{d}\alpha = \sum_i \mathrm{d}x_i \wedge \mathrm{d}\xi_i$, which evaluates on linear tangent–cotangent pairs as $\Omega_0(u \oplus \alpha, v \oplus \beta) = \beta(u) - \alpha(v)$ [[22, Homework 1, §§2.2–2.3]](#references). The graph of a one-form $\mu$ on $X$ has half the dimension of $T^*X$ and is a Lagrangian submanifold ($i^*\omega = 0$) if and only if $\mathrm{d}\mu = 0$ [[22, §3.2]](#references), which holds whenever $\mu = \mathrm{d}U$ is the differential of an internal energy $U$, while the symmetric cross-form $\frac{1}{2}\sum_i (\mathrm{d}x^i \odot \mathrm{d}\xi_i)$ pulls back on that graph to the Hessian $\frac{\partial^2 U}{\partial x^i \partial x^j}\,\mathrm{d}x^i \odot \mathrm{d}x^j$.

[^symmetric-affine]: Extending `symmetric` off the critical points onto a general curved state manifold requires a choice of connection to compare potentials at nearby states, whereas the canonical symplectic two-form is defined without a connection.

#### Uncertainty, principal components and gradient descent

In estimation and statistical optics, a covariance is a duality map from a linear readout to the state variation associated with it, while a precision map (or, in a parametric statistical model, the Fisher information) maps a state variation back to a readout, whichever side of the orientation the variation has. For a point, variations are intrinsic and linear readouts are planes, as they are for the polarization coherency on a screen across a light beam; for a rigid pose, a motor perturbation is an extrinsic twist and its linear readout is an intrinsic line, and pushing the pose covariance forward through the linearized map `shift` from twists to point displacements gives the first-order position covariance in [Kalman pose filter](../examples/estimation/kalman/kalman.ipynb), [Gaussian fitting](../examples/quadrics/gaussian/gaussian.ipynb), [odometry](../examples/estimation/odometry/odometry.ipynb), [station keeping](../examples/estimation/pose_diffusion/station_keeping.ipynb) and [Kerr black-hole ray tracing](../examples/relativity/kerr/kerr.ipynb) examples:

```python
point_variance = plane & point_covariance(plane)             # [] Scalar+
error_cost = information(variation) & variation              # [] Scalar+
transmitted_power = analyzer & coherency(analyzer)           # [] Scalar+
pose_variance = pose_covariance(line) & line                 # [] Scalar+
moved_covariance = shift(pose_covariance(shift.adjugate()))  # [] Antivector- <- Vector+
```

Under any linear or linearized map `shift` on variations, `covariance` and `information` transform in opposite ways: variations push forward by `shift` and readouts pull back by `shift.adjugate()`, so `shift(pose_covariance(shift.adjugate()))` carries pose uncertainty onto first-order point uncertainty with no metric involved, while taking principal components of a covariance requires an explicit comparison metric form.

In the tensor notation of information geometry and state estimation, a Riemannian or Fisher information metric $g_{ij}$ is a covariant $(0,2)$ form mapping tangent variations to cotangent differentials, so the natural gradient converts a covariant differential $\partial_j f$ into a contravariant step $g^{ij}\partial_j f$ through the inverse metric [[23, §4.1.2]](#references), while covariance $\Sigma^{ij}$ is a contravariant $(2,0)$ form on differentials. For nonlinear camera measurements in robotics, state-estimation treatments propagate homogeneous point and pose covariances to first order by linearizing the camera model into Jacobian blocks $F P F^{\mathsf T}$ [[24, §7.3.5]](#references).

#### Thermodynamics and heat transport

In a thermodynamic state space with null basis squares (`a0 b0` or `t0 v0`), extensive variables such as energy, entropy and volume have incommensurable units and no Euclidean inner product exists between them. With extensive state variations intrinsic and conjugate intensive affinities extrinsic as in the equilibrium state–potential space above, an Onsager dissipative mobility maps the entropy differential to an energy-conserving heat transfer (`energy_gradient & transfer == 0`), while the entropy Hessian with respect to those extensive coordinates defines a Ruppeiner fluctuation metric in [entropy production](../examples/thermodynamics/entropy_production/entropy_production.ipynb), [critical fluctuations](../examples/thermodynamics/ruppeiner/ruppeiner.ipynb), [constrained heat capacity](../examples/thermodynamics/state_space/state_space.ipynb) and [damping as heating](../examples/thermodynamics/generic/generic.ipynb) examples:

```python
mobility = transfer * (Vector & transfer) * conductance * temp_a * temp_b  # [] Antivector- <- Vector+
entropy_production = entropy_gradient & mobility(entropy_gradient)         # [] Scalar+
fluctuation_cost = metric(variation) & variation                           # [] Scalar+
```

Even though the algebra's built-in metric is identically zero, `entropy_gradient & mobility(entropy_gradient)` and `metric(variation) & variation` evaluate entropy production and fluctuation curvature directly, while keeping the singular energy-conserving `mobility` distinct in type from the fluctuation `metric`.

Spacetime keeps the same pairing, with positions, velocities and fluxes intrinsic and fields and drives extrinsic, so extensive quantities can join spacetime in one algebra ([Geometry and quantities](thermodynamics.md)). In a [magnetotransport](../examples/thermodynamics/magnetotransport/magnetotransport.ipynb) example on a planar conductor, electrons and holes experience the magnetic field and the electrochemical and thermal drives through one commutator, `(magnetic + drive).commutator(parcel + valence * velocity)`, so the steady `response` maps the drive on charge and entropy to the flux they carry, the dissipation `drive & response(drive)` is a positive scalar every observer reads alike, and reversing the magnetic field in that model turns `response` into `response.adjugate()`.

In the differential notation of classical and nonequilibrium thermodynamics, extensive differentials $\mathrm{d}x^i = (\mathrm{d}U, \mathrm{d}V, \mathrm{d}N)$ pair with conjugate intensive affinities $X_i = \partial S/\partial x^i = (1/T, p/T, -\mu/T)$, irreversible fluxes in the absence of magnetic fields follow the symmetric mobility $J^i = L^{ij}X_j$ with bilinear entropy production $\dot{S} = X_i L^{ij} X_j$ [[25, §1]](#references), and equilibrium fluctuation geometry is defined in the natural extensive coordinates by the Hessian line element $g_{ij} = -\partial^2 S/\partial x^i\partial x^j$, which does not keep a pure Hessian form under arbitrary coordinate changes [[26, §3.1]](#references). In unoriented vector-based geometric algebra, pairing vectors through a non-degenerate inner product would assign commensurable squared lengths to energy, volume and particle number.

#### Quantum states and measurements

A qubit's state is determined by the outcome probabilities of measurements on it, and mixing two states mixes their probabilities linearly. With a state modelled as a projective point of unit weight (`w & state == 1` for the trace) and a measurement outcome as a plane ([States and effects](#states-and-effects)), Born's rule is `effect & state`. A noisy quantum channel maps prepared states forward in the Schrödinger picture while its adjugate pulls measurement effects backward in the Heisenberg picture in [quantum process tomography in the projective Bloch ball](../examples/quantum/process_tomography/process_tomography.ipynb) and [Bloch-vector companion](../examples/quantum/process_tomography/process_tomography_bloch.ipynb) examples:

```python
probability = effect & device(preparation)         # [] Scalar+
pulled_effect = device.adjugate()(effect)          # [] Vector+
```

Reconstructing a noisy channel from four tetrahedral preparations and four detector planes uses only joins, meets and `&`: the point where three outcomes' planes meet is dual to the fourth outcome, and the plane through three prepared states is dual to the fourth preparation.

In the density-matrix, Pauli-transfer and operational notations of quantum information, general measurements and open-system dynamics are written with density operators $\rho$, POVM effects $E_b$, the Born trace pairing $p = \operatorname{tr}[E\,\Lambda(\rho)] = \langle\!\langle E \mid R_\Lambda \mid \rho \rangle\!\rangle$, and the $4\times 4$ Pauli transfer matrix $R_\Lambda$ acting on homogeneous Pauli coordinates $(1, \mathbf{r})$ [[27, §§2.1.3, 2.2]](#references), while convex state mixtures and linear probability functionals are axiomatized operationally in generalized probabilistic theories [[28, §II.A]](#references). The Hilbert–Schmidt inner product $\operatorname{tr}(A^\dagger B)$ identifies the space of Hermitian operators with its own dual so that states $\rho$ and effects $E_b$ share one matrix type and the full adjoint channel $\Lambda^\dagger(E_b)$ retains non-unital shifts on the identity component; only when the unit trace is separated from the three-dimensional Bloch vector in $p = \frac{1}{2}(1 + \mathbf{e}\cdot\mathbf{r})$ so a non-unital channel reads $\mathbf{r} \mapsto A\mathbf{r} + \mathbf{t}$ does a truncated three-component transpose $A^{\mathsf T}$ omit the shift $\mathbf{e}\cdot\mathbf{t}$.

## Appendix: representations

Modelling a physical or geometric system in an algebra fixes three representation choices for each quantity: which axes the algebra includes, which grade the quantity occupies, and which orientation it carries. Those choices decide whether related parts of a quantity transform together, such as an electric and a magnetic field under a boost, and whether physical laws stay unchanged under reflections, translations and deformations:

- [Space, grade and orientation](#space-grade-and-orientation)
- [Points, motions and momenta](#points-motions-and-momenta)
- [Electric and magnetic fields](#electric-and-magnetic-fields)
- [Spin](#spin)
- [Cutting planes and surface windings](#cutting-planes-and-surface-windings)
- [Handedness in the medium](#handedness-in-the-medium)
- [States and effects](#states-and-effects)
- [Summary](#summary)

### Space, grade and orientation

The first choice is which axes the algebra includes rather than packs away. Including time in spacetime (`x+ y+ z+ t-`) puts electric and magnetic fields in one space so boosts mix them within a single type, and including a projective weight (`w0`) puts points and statistical states in homogeneous spaces where translations and state mixing act linearly. Dropping those axes into three-dimensional Euclidean space and using the spatial pseudoscalar x∧y∧z to pack two vector spaces together conflates a spatial reflection with inversion of the omitted axis, because every mirror negates x∧y∧z.

The second choice is which grade represents each object as the ambient dimension varies. Hyperplanes sit at grade 1 and generators of continuous motion sit at grade 2 in the plane, in three-space and in spacetime alike, whereas points and objects joined from points by the regressive product `&` sit at fixed codimension (grades n − 1, n − 2, n − 3 in an n-dimensional algebra), shifting grade with the ambient space while keeping one geometric role. Dualizing planes or patches into grade-1 normal vectors relies on grade 2 and grade 1 sharing three components in three-dimensional space.

The third choice is [orientation](orientation.md), which separates elements that share both an algebra and a grade, following projective duality treatments [[5]](#references), [[7]](#references): hyperplanes and their outer products `^` are extrinsic (`+`), while points, joins of points `&`, and handed material constants are intrinsic (`-`). Under proper rotations in three-dimensional Euclidean space, packed and unpacked representations have matching component counts and give the same numbers ([One perspective across languages](#one-perspective-across-languages)); reflections, boosts, translations, planar slices and non-rigid deformations separate them:

```python
mirror >> f(a, b) == f(mirror >> a, mirror >> b)                  # [] same output type
```

### Points, motions and momenta

A point is where hyperplanes meet, a velocity is how a point moves, and a momentum is the line a mass moves along, weighted by its mass and speed. All three live in projective space and are built from points, so points and velocities are intrinsic antivectors, at grade n − 1, and the join of a point with its velocity is an intrinsic antibivector, at grade n − 2. A rigid motion rate combines turns and translations as an extrinsic bivector, at grade 2, and pairing the rate with the momentum reads twice the kinetic energy in [energy in a mirror](../examples/orientation/mirror_energy/mirror_energy.ipynb) and [mechanical work in a mirror](../examples/orientation/mechanical_work/mechanical_work.ipynb) examples:

```python
point = mv.point([2, 0, 0])                                       # [] Antivector-
rate = mv.xy                                                      # [] Bivector+
momentum = point & point.commutator(rate)                         # [] Antibivector-: 2*zx + 4*wz
energy = (rate & momentum) / 2                                    # [] Scalar+: 2

mirror = mv.x                                                     # [] Vector+
mirrored = mirror >> momentum                                     # [] Antibivector-: 2*zx - 4*wz
rebuilt = (mirror >> point) & (mirror >> point).commutator(mirror >> rate)  # [] Antibivector-: 2*zx - 4*wz
```

The same lines hold in the projective plane, in projective three-space, on the sphere and in spacetime; only the ambient dimension changes. When positions and velocities are written as grade-1 Euclidean vectors and wedged as `r ^ v`, momentum acquires the type of a rate and energy is read through the metric inner product, which drops translational energy in projective space because `w | w == 0`. Reflecting `rate` and the already-built `momentum` with the single grade-involuted sandwich `mirror * x.involute() * mirror.inverse()`, as if both were extrinsic, flips the sign of `momentum` relative to the mirrored body and yields −2 instead of 2 in that [energy in a mirror](../examples/orientation/mirror_energy/mirror_energy.ipynb) example.

In three-dimensional vector notation, torque $\boldsymbol{\tau} = \mathbf{r} \times \mathbf{F}$ and angular velocity $\boldsymbol{\omega}$ are axial vectors reflected with an explicit determinant sign, as distinguished from polar displacements [[1, §52-5]](#references), while the Euclidean geometric algebra of three-space writes positions and forces at grade 1 and angular momentum at grade 2 around a chosen origin. In projective geometric algebra, dual Grassmann models formulate rigid-body dynamics with axes and spears paired by the regressive product across Cayley–Klein spaces [[2, §§2.2, 6]](#references), [[3, Ch. 9]](#references), [[5, §8.6.1]](#references), single-algebra plane-based treatments map rate bivectors to dual momentum $(d-1)$-vectors paired by join [[4, §2.7]](#references), projective orientation duality places intrinsic and extrinsic orientations on the types [[7]](#references), and point-based projective geometric algebra places points at grade 1 and pairs the geometric product with an antiproduct [[8]](#references).

### Electric and magnetic fields

An electric field accelerates charges at rest along a spatial direction, sweeping a plane along the observer's time; a magnetic field turns moving charges in a plane of space. Both generate a charge's motion in spacetime (`x+ y+ z+ t-`), so both inhabit one extrinsic bivector space, and a spatial direction swept along the observer's clock, `direction & time`, reads the electric part across that worldsheet patch:

```python
time = -mv.t.dual()                                               # [] Antivector-
field = mv.xt * 2 + mv.xy                                         # [] Bivector+
electric_reading = field & (mv.x.dual() & time)                   # [] Scalar+: 2
horizontal_mirror = mv.z >> field                                 # [] Bivector+: xy + 2*xt
vertical_mirror = mv.x >> field                                   # [] Bivector+: -xy - 2*xt
```

Lorentz boosts mix electric and magnetic components within that one type, and a planar problem drops a spatial axis rather than time: in an [eddy-current braking](../examples/electromagnetism/eddy_brake/eddy_brake.ipynb) example the algebra is `x+ y+ t-`, where the magnetic field sits on x∧y and the motional electric field sits on the temporal planes x∧t and y∧t, whereas two-dimensional space `x+ y+` alone has only one bivector component and leaves nowhere for an in-plane electric field. The same spacetime bivector type carries [induced dipoles](../examples/electromagnetism/induced_dipoles/induced_dipoles.ipynb), [photoelasticity](../examples/optics/photoelasticity/photoelasticity.ipynb), [black polarization sandwiches](../examples/optics/black_sandwich/black_sandwich.ipynb) and [Thomas precession](../examples/relativity/thomas_precession/thomas_precession.ipynb) examples. Packing two spatial vectors into three-space with the spatial pseudoscalar x∧y∧z reverses the magnetic plane relative to the electric vector when rebuilt from inputs mirrored in z, because x∧y∧z negates under the mirror.

In three-dimensional vector notation, $\mathbf{E}$ is a polar vector and $\mathbf{B}$ is an axial vector, each with its own reflection rule and coupled under boosts by separate transformation equations. In the paravector notation of the algebra of physical space, the field is written as $F = \mathbf{E} + i c\mathbf{B}$ with the spatial pseudoscalar dualizing the rotation plane [[29, §IV.B]](#references); if a spatial mirror is represented by the ungraded sandwich $\sigma_z F \sigma_z$, its spacetime counterpart $\gamma_z\gamma_0 F \gamma_0\gamma_z$ combines that spatial reflection with time reversal unless parity conjugation is specified separately. Spacetime algebra represents the electromagnetic field as a single spacetime bivector and an observer's relative vectors $\sigma_k = \gamma_k\gamma_0$ as temporal bivectors [[18]](#references).

### Spin

A spin or magnetic moment is an angular momentum, the intrinsic patch of the plane its current or phase circulates in, so like every angular momentum it is an intrinsic antibivector, at grade n − 2: grade 1 in three-dimensional space, and grade 2 in four-dimensional spacetime. An isospin or pseudospin has the same representation in its state space. The magnetic field that turns it is an extrinsic plane of turning, the Zeeman energy is `field & spin`, and Larmor precession is `spin.commutator(field)`, in [magnetic resonance](../examples/quantum/magnetic_resonance/magnetic_resonance.ipynb), [superconductivity](../examples/quantum/superconductivity/superconductivity.ipynb), [neutrino oscillations](../examples/quantum/neutrino_isospin/neutrino_isospin.ipynb), [graphene](../examples/quantum/graphene/graphene.ipynb), [Thomas precession](../examples/relativity/thomas_precession/thomas_precession.ipynb) and [Kerr black-hole ray tracing](../examples/relativity/kerr/kerr.ipynb) examples:

```python
spin = mv.yz.dual() + mv.xy.dual() * 0.5                          # [] Antibivector-: x + 0.5*z
field = mv.xy * 2                                                 # [] Bivector+: 2*xy
energy = field & spin                                             # [] Scalar+: 1
mirrored_rate = mv.z >> spin.commutator(field)                    # [] Antibivector-: -2*y
rebuilt_rate = (mv.z >> spin).commutator(mv.z >> field)           # [] Antibivector-: -2*y

sheared_spin = (mv.x.dual() + 0.5 * mv.z.dual()) & mv.y.dual()    # [] Antibivector-
sheared_rate = (mv.x + 0.5 * mv.z) ^ mv.y                         # [] Bivector+
preserved_energy = mv.xy & sheared_spin                           # [] Scalar+: 1
metric_energy = sheared_rate.dot(sheared_rate)                    # [] Scalar+: 1.25
```

If spin is instead typed as an extrinsic rate in this model, it transforms like the field that turns it: under a volume-preserving shear that tilts x toward z, the directions spanning `sheared_spin` move forward while the field's plane x∧y pulls back unchanged, keeping the pairing equal to 1, whereas the metric dot product `sheared_rate.dot(sheared_rate)` changes to 1.25. When spin and field are represented as grade-1 vectors of an unoriented three-space algebra, reflected by a uniform vector sandwich and coupled through the spatial pseudoscalar x∧y∧z, precession rebuilt in a mirror runs backwards because every mirror negates x∧y∧z; twin notebooks for [magnetic resonance](../examples/quantum/magnetic_resonance/magnetic_resonance_space.ipynb) and [two spins](../examples/quantum/two_spins/two_spins_space.ipynb) run that spatial-vector prescription beside the oriented one.

In three-dimensional vector notation, the Bloch equations write precession with a cross product, $\dot{\mathbf{s}} = \mathbf{B} \times \mathbf{s}$, assigning axial-vector parity to both $\mathbf{s}$ and $\mathbf{B}$ so their cross product survives inversion while weak-decay spin–momentum couplings take an explicit parity-violating pseudoscalar sign [[19, Part II]](#references). Spacetime algebra identifies the spin plane directly as a spacetime bivector generator alongside the observer's relative basis $\sigma_k = \gamma_k\gamma_0$ [[18, §VII.D]](#references), whereas an unoriented spatial Pauli algebra that replaces the unit imaginary by the spatial pseudoscalar $I = \sigma_1\sigma_2\sigma_3$ in $(I\mathbf{B})\times\mathbf{s}$ leaves extrinsic rotation rates and intrinsic angular-momentum patches in one unoriented type.

### Cutting planes and surface windings

A plane that cuts space has two sides, oriented across the plane, whereas an ordered triangle has a vertex winding, oriented within the plane. In three-dimensional projective space (`w0 x+ y+ z+`), joining three points produces an intrinsic antitrivector at grade 4 − 3 = 1, the same grade and basis blade z as the horizontal cutting plane, as a [surface normals](../examples/orientation/normals/normals.ipynb) example shows. A vertical mirror keeps the upper side up and reverses the triangle's winding:

```python
corners = mv.point([[0, 0, 0], [1, 0, 0], [0, 1, 0]])             # [3] Antivector-
side = mv.z                                                       # [] Vector+: z
winding = corners[0] & corners[1] & corners[2]                    # [] Antitrivector- (Vector-): z
mirrored_side = mv.x >> side                                      # [] Vector+: z
mirrored_winding = mv.x >> winding                                # [] Antitrivector- (Vector-): -z
```

Grade alone cannot tell `side` and `winding` apart here: the unoriented grade-involuted sandwich `mirror * winding.involute() * mirror.inverse()` treats `winding` as a mirror plane and leaves it at +z, even though the reflected corners run the opposite way round.

In the matrix notation of continuum mechanics and computer graphics, oriented area elements transform by the cofactor multiple $(\det F)F^{-\mathsf T}$ [[9, §1.7]](#references), whereas plane covectors are carried forward by the inverse transpose $F^{-\mathsf T}$ and pulled back by the transpose $F^{\mathsf T}$. When both a cutting plane and a triangle's area normal $\mathbf{n} = (\mathbf{p}_1 - \mathbf{p}_0) \times (\mathbf{p}_2 - \mathbf{p}_0)$ are stored as one three-vector, that vector cannot keep the outward side while reversing the corner winding under a reflection, so graphics pipelines transform shading normals by $F^{-\mathsf T}$ and adopt the convention of reversing triangle index order under negative-determinant transforms.

### Handedness in the medium

Some physical processes distinguish left from right: cobalt-60 beta decay emits electrons preferentially opposite the nuclear spin, and chiral media rotate a light beam's polarization in one sense around its travel direction. Modelled here as an intrinsic antipseudoscalar (an intrinsic scalar at grade 0), a chiral material constant or weak coupling carries that handedness itself while laboratory fields stay extrinsic. In an [optical rotation, out and back](../examples/orientation/optical_rotation/optical_rotation.ipynb) example's model, light travelling along a spatial direction in spacetime sweeps the intrinsic plane `direction & time`, and multiplying by `rotary_power` yields an extrinsic rotation rate that reverses on the return leg and cancels out and back, whereas a magnetic field turns the light in proportion to its flux `magnetic & cross_section` across the beam's intrinsic cross-section, doubling on the return trip:

```python
sweep = mv.z.dual() & (-mv.t.dual())                              # [] Antibivector-: xy
rotary_power = mv.antipseudoscalar([2])                           # [] Antipseudoscalar- (Scalar-): 2
round_trip_chiral = sweep * rotary_power + (-sweep) * rotary_power  # [] Bivector+: 0

cross_section = mv.xy.dual()                                      # [] Antibivector-: zt
magnetic = mv.xy * 3                                              # [] Bivector+: 3*xy
turn = cross_section.dual_inverse() * (magnetic & cross_section)  # [] Bivector+: 3*xy
round_trip_faraday = turn + turn                                  # [] Bivector+: 6*xy
```

If `rotary_power` were an ordinary extrinsic scalar, `sweep * 2` would stay intrinsic rather than generating a rotation. Averaged over a crystal's point group in a [material symmetry](../examples/mechanics/symmetry/symmetry.ipynb) example, a handed gyration map vanishes in every group containing a mirror or an inversion centre while thermal conductivity survives; in a [Wu experiment](../examples/orientation/wu_experiment/wu_experiment.ipynb) example, pairing two intrinsic quantities `spin & direction` produces an intrinsic scalar that multiplies the weak-decay `asymmetry` into an ordinary scalar emission rate `1 + asymmetry * (spin & direction)`.

In the tensor and vector notations of weak-interaction physics and electrodynamics, weak parity violation in beta decay breaks mirror symmetry on spin–momentum correlations [[19, Part II]](#references), and electromagnetic media enter through constitutive maps between untwisted field 2-forms and twisted excitation 2-forms [[17, §§2, 6]](#references), with parity-odd couplings written as ordinary numbers accompanied by separate reflection sign rules under $\mathbf{x} \mapsto -\mathbf{x}$.

### States and effects

A qubit's state fixes the outcome probabilities of every measurement on it, and a statistical mixture of states mixes those probabilities linearly. Because an affine function on three-dimensional space is a projective plane, including a projective weight w represents a qubit state as an intrinsic point of unit weight and a measurement outcome as an extrinsic plane, paired by `effect & state`, in a [quantum process tomography in the projective Bloch ball](../examples/quantum/process_tomography/process_tomography.ipynb) example:

```python
centre = mv.point([0, 0, 0])                                      # [] Antivector-
north = mv.point([0, 0, 1])                                       # [] Antivector-
state = 0.4 * centre + 0.6 * north                                # [] Antivector-
effect = (mv.w + mv.z) / 2                                        # [] Vector+: spin up along z
probability = effect & state                                      # [] Scalar+: 0.8
trace = mv.w & state                                              # [] Scalar+: 1
relax = Point + 0.1 * (north * (mv.w & Point) - Point)            # [] Antivector- <- Antivector-
```

A quantum channel is then a linear map on points, whether it rotates the Bloch ball coherently or pulls states toward `north` under amplitude damping. When the projective weight is dropped in favour of a three-dimensional Bloch vector, the unit trace sits apart from the vector and relaxation becomes an affine shift alongside a linear map.

In the operator, Pauli-transfer and operational notations of quantum information, a qubit state is written as the density matrix $\rho = \tfrac12(\mathbb 1 + \mathbf r \cdot \boldsymbol\sigma)$ and a channel on homogeneous Pauli coordinates $(1, \mathbf r)$ as the $4\times 4$ Pauli transfer matrix, which retains amplitude damping's affine shift in a single matrix product while truncating the trace coordinate turns the channel on Bloch vectors into $\mathbf r \mapsto A\mathbf r + \mathbf t$ [[27, §§2.1.3, 2.2]](#references). In generalized probabilistic theories [[28, §II.A]](#references), states form a convex set of normalized preparation vectors, physical transformations are linear maps preserving convex mixtures, and measurement outcomes act as linear probability functionals.

### Summary

Each row sets the space, grade and orientation modelled in this document beside the packed or three-dimensional representation commonly used in its place:

| Quantity | Algebra and type | Common alternative |
| --- | --- | --- |
| Position, velocity, force | Projective or spatial `Antivector-` | Euclidean grade-1 vector |
| Rate of motion | Projective or spatial `Bivector+` | Axial three-vector |
| Momentum, forque | Projective or spatial `Antibivector-` | Axial three-vector, or dualized bivector |
| Electromagnetic field | Spacetime `Bivector+` | Polar and axial three-vectors, or spatial paravector |
| Spin, magnetic moment | Space or spacetime `Antibivector-` | Axial three-vector, or unoriented bivector |
| Cutting plane | Projective `Vector+` | Surface normal three-vector |
| Surface patch, triangle winding | Space `Antibivector-`, projective `Antitrivector-` | Cross-product normal three-vector |
| Rotary power, weak asymmetry | Intrinsic `Antipseudoscalar-` | Ordinary scalar with parity sign table |
| Qubit state | Projective `Antivector-` of unit weight | Density matrix, or Bloch vector plus trace |
| Measurement effect | Projective `Vector+` | Positive operator |

## References

1. [*The Feynman Lectures on Physics*, Vol. I, Chapter 52: Symmetry in Physical Laws](https://www.feynmanlectures.caltech.edu/I_52.html) (Richard P. Feynman, Robert B. Leighton and Matthew Sands, 1963).
2. [*On the Homogeneous Model of Euclidean Geometry*](https://arxiv.org/pdf/1101.4542) (Charles Gunn, in *Guide to Geometric Algebra in Practice*, Leo Dorst and Joan Lasenby, eds., Springer, 2011; extended preprint of AGACSE 2010).
3. [*Geometry, Kinematics, and Rigid Body Mechanics in Cayley–Klein Geometries*](https://depositonce.tu-berlin.de/bitstreams/5a02e6fc-d9c3-4e6f-a5ab-f4a970e81209/download) (Charles Gunn, Doctoral Dissertation, Technische Universität Berlin, 2011; DOI `10.14279/depositonce-3058`).
4. [*May the Forque Be with You: Dynamics in PGA*](https://bivector.net/PGAdyn.pdf) (Leo Dorst and Steven De Keninck, 2022).
5. [*A Bit Better: Variants of Duality in Geometric Algebras with Degenerate Metrics*](https://arxiv.org/pdf/2206.02459) (Charles Gunn, 2022).
6. [*Solid State Physics* lecture notes](https://davidtong.org/pdfs/teaching/solid-state-physics/solidstate.pdf) (David Tong, University of Cambridge, 2017).
7. [*Projective Duality Encodes Complementary Orientations in Geometric Algebras*](https://pure.uva.nl/ws/files/189200508/Projective_duality_encodes_complementary_orientations_in_geometric_algebras.pdf) (Leo Dorst, *Mathematical Methods in the Applied Sciences*, 2024).
8. [*Projective Geometric Algebra*](https://projectivegeometricalgebra.org/) (Eric Lengyel).
9. [*Mathematical Foundations of Elasticity*, Chapter 1: Geometry and Kinematics of Bodies](https://authors.library.caltech.edu/records/s9jhk-sn323/files/MFE_Chapter_1.pdf) (Jerrold E. Marsden and Thomas J. R. Hughes, 1983; Dover ed., 1994).
10. [*Multiple View Geometry in Computer Vision*, Chapter 9: Epipolar Geometry and the Fundamental Matrix](https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf) (Richard Hartley and Andrew Zisserman, 2nd ed., 2004).
11. [*Introduction to Projective Geometry*, Chapter 7: Polarities](https://www.voutsadakis.com/TEACH/LECTURES/PROJECTIVE/Chapter7.pdf) (George Voutsadakis, lecture notes).
12. [*Introduction to Projective Geometry*, Chapter 11: Parallelism](https://voutsadakis.com/TEACH/LECTURES/PROJECTIVE/Chapter11.pdf) (George Voutsadakis, lecture notes).
13. [*Mathematical Foundations of Elasticity*, Chapter 2: Balance Principles](https://authors.library.caltech.edu/records/s9jhk-sn323/files/MFE_Chapter_2.pdf) (Jerrold E. Marsden and Thomas J. R. Hughes, 1983; Dover ed., 1994).
14. [*Finite element exterior calculus: from Hodge theory to numerical stability*](https://arxiv.org/pdf/0906.4325) (Douglas N. Arnold, Richard S. Falk and Ragnar Winther, *Bulletin of the American Mathematical Society*, 2010).
15. [*Discrete Exterior Calculus*](https://arxiv.org/pdf/math/0508341) (Mathieu Desbrun, Anil N. Hirani, Melvin Leok and Jerrold E. Marsden, 2005).
16. [*The Euler–Poincaré Equations and Semidirect Products with Applications to Continuum Theories*](https://arxiv.org/pdf/chao-dyn/9801015) (Darryl D. Holm, Jerrold E. Marsden and Tudor S. Ratiu, *Advances in Mathematics*, 1998).
17. [*Recent developments in premetric classical electrodynamics*](https://arxiv.org/pdf/physics/0610221) (Friedrich W. Hehl, Yakov Itin and Yuri N. Obukhov, 2006).
18. [*Spacetime Physics with Geometric Algebra*](https://davidhestenes.net/geocalc/pdf/SpacetimePhysics.pdf) (David Hestenes, *American Journal of Physics*, 2003).
19. [*The law of parity conservation and other symmetry laws of physics*](https://www.nobelprize.org/uploads/2018/06/yang-lecture.pdf) (Chen Ning Yang, Nobel Lecture, 1957).
20. [*The basics of gravitational wave theory*](https://arxiv.org/pdf/gr-qc/0501041) (Éanna É. Flanagan and Scott A. Hughes, *New Journal of Physics*, 2005).
21. [*Regge calculus: a unique tool for numerical relativity*](https://arxiv.org/pdf/gr-qc/0408006) (Adrian P. Gentle, *General Relativity and Gravitation*, 2002; arXiv, 2004).
22. [*Lectures on Symplectic Geometry*](https://people.math.ethz.ch/~acannas/Papers/lsg.pdf) (Ana Cannas da Silva, Springer, 2001; corrected lecture notes, 2006).
23. [*An elementary introduction to information geometry*](https://arxiv.org/pdf/1808.08271) (Frank Nielsen, *Entropy*, 2020).
24. [*State Estimation for Robotics*](https://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf) (Timothy D. Barfoot, 2017).
25. [*Reciprocal Relations in Irreversible Processes. I.*](https://users.jyu.fi/~veapaja/Statistical_Physics_B/onsager_reciprocal.relations.PhysRev.37.405.1931.pdf) (Lars Onsager, *Physical Review*, 1931).
26. [*Thermodynamic curvature and phase transitions in Kerr–Newman black holes*](https://arxiv.org/pdf/0802.1326) (George Ruppeiner, *Physical Review D*, 2008).
27. [*Introduction to Quantum Gate Set Tomography*](https://arxiv.org/pdf/1509.02921) (Erik Nielsen, John King Gamble, Kenneth Rudinger, Travis Scholten, Kevin Young and Robin Blume-Kohout, *Quantum*, 2021).
28. [*Information processing in generalized probabilistic theories*](https://arxiv.org/pdf/quant-ph/0508211) (Jonathan Barrett, *Physical Review A*, 2007).
29. [*Relativity in Introductory Physics*](https://arxiv.org/pdf/physics/0406158) (William E. Baylis, *Canadian Journal of Physics*, 2004).
