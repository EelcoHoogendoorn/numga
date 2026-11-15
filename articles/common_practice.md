# Extensors and common practice

This document compares common approaches to applied problems with the constructions that arise in geometric algebra and extensors. Each case describes how a problem is usually formulated, develops the geometric construction, and examines the practical consequences for modelling and computation: what becomes simpler, which choices and conventions become explicit, and where sign or factor mistakes disappear.

## Maps and numerical methods

These cases concern linear and multilinear maps in estimation, mechanics and geometry, where typing inputs, outputs and pairings removes auxiliary coordinate conventions.

### Scene reconstruction and camera alignment

Scene reconstruction estimates landmark positions and camera poses from image measurements. A common formulation minimizes weighted squared image errors, forming Gauss–Newton normal equations for the update and eliminating the landmark steps by a Schur complement into a reduced camera system ([modern synthesis of bundle adjustment](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Triggs00.pdf), §§4.3, 6.1).

The [multiview example](../examples/estimation/multiview/multiview_reconstruction.ipynb) represents each image measurement by a quadratic cost on the sensor and pulls that cost back through the camera into a sight cone. Because the cone's polar plane vanishes on the sight line, it is tempting to treat that plane as a residual vector and form normal equations from its coefficients, squaring a cost that is already quadratic. Instead, camera motion enters the cone's quadratic cost directly:

```python
motion = -Twist.commutator(seen)                                  # [landmarks, cameras] Antivector- <- Bivector+
curvature = (cones(motion) & motion).sum(axis=0)                  # [cameras] Scalar+ <- (Bivector+, Bivector+)
slope = (cones(seen) & motion).sum(axis=0)                        # [cameras] Scalar+ <- Bivector+
step = curvature.lstsq(-slope, rcond=1e-4)                        # [cameras] Bivector+
```

Evaluating the cone directly preserves its intended measurement weighting instead of squaring it and introducing an arbitrary coefficient norm on planes, leaving the local curvature and slope ready for the linear solve. In the joint problem, stepping cameras by twists (`Bivector+`) and landmarks by spatial directions (`Antivector-`) keeps each landmark block three-dimensional and nonsingular, so eliminating the landmarks in the Schur complement uses a direct solve rather than a pseudoinverse over homogeneous coordinates.

In matrix notation, a cone cost $x^{\mathsf T}Qx$ composed with a point-motion Jacobian $M$ gives the Gauss–Newton Hessian $M^{\mathsf T}QM$, whereas treating the polar plane $Qx$ as a Euclidean residual produces $M^{\mathsf T}Q^{\mathsf T}QM$ with a squared copy of $Q$.

### Motion and force in robot control

A robot arm often moves freely in some directions while pressing against a constraint in others. Splitting those tasks by Euclidean orthogonality of six motion coordinates mixes angular and linear units and changes with the chosen origin and length scale. Motion and force instead occupy dual spaces paired into mechanical power, where zero-power reciprocity relates compatible motions and constraints ([reciprocal pairing of twists and wrenches](https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-complete.pdf), Ch. 2, §5) and the [dual motion and force spaces of spatial-vector algebra](https://bleyer.org/files/A%20Beginner%27s%20Guide%20to%206-D%20Vectors%20-%20Feathersone%20%28IEEE%2C%202010%29.pdf) carry no inner product on motion coordinates alone.

In projective geometric algebra, a motion rate is an extrinsic twist (`Bivector+`) and a force system is an intrinsic forque (`Antibivector-`), paired into power by `twist & forque`. In the [odometry example](../examples/estimation/odometry/odometry.ipynb), the same type split separates a pose correction from the differential of its cost: information maps twists to forques (`Antibivector- <- Bivector+`), covariance maps forques back to twists (`Bivector+ <- Antibivector-`), and preconditioned conjugate gradients pairs those two spaces while preconditioning by solving the information map, so no coordinate dot product silently weights angles against lengths.

### Surface normals under deformation

A surface normal can record a chosen outward side of a plane or encode the area and vertex winding of a patch. Computer graphics commonly stores both as three-component vectors. Under a regular linear deformation, preserving tangent orthogonality transforms a surface normal by the inverse transpose ([inverse-transpose transformation of normals](https://pbr-book.org/4ed/Geometry_and_Transformations/Applying_Transformations), §3.10.3), whereas transporting an oriented patch multiplies that inverse transpose by the Jacobian determinant ([transport of oriented area elements](https://authors.library.caltech.edu/records/s9jhk-sn323/files/MFE_Chapter_1.pdf?download=1), §1.7).

The [surface-normals example](../examples/orientation/normals/normals.ipynb) separates the two by orientation: a cutting plane is extrinsic (`Vector+`) and a triangle's joined patch is intrinsic (`Antitrivector-`), so reflecting the scene carries both the outward side and the corner winding consistently even when their coefficients acquire opposite signs. Under a general linear deformation between intrinsic point spaces, the outermorphism carries patches forward while the [adjugate](complementarity.md#claims) pulls plane conditions backward across the regressive pairing:

```python
input_plane = deformation.adjugate()(output_plane)                # [] Vector+
```

This pullback remains defined when the deformation is singular; carrying an input plane forward while preserving its readings requires solving through the adjugate, recovering the inverse transpose only when the deformation is invertible. Separating the two maps prevents a backward plane pullback from being applied as a forward normal transform, and leaves the cofactor outermorphism available when volume collapses while surface patches survive.

### Area elements and stress measures

Deforming a continuum changes both the area patch on which a traction acts and the directions in which the resulting force is expressed. Continuum mechanics records those choices in the Cauchy, first Piola–Kirchhoff and second Piola–Kirchhoff stress tensors, whose component formulas combine the deformation gradient, its inverse and its Jacobian determinant.

The [area-transport example](../examples/mechanics/area_transport/area_transport.ipynb) transports an oriented patch forward by applying the deformation's outermorphism to its joined edges, and pulls a flux bivector backward through the adjugate of that area map so both measure the same scalar flux:

```python
areas = deformation.outermorphism(Patch)                          # [] Antibivector- <- Antibivector-
flux_pullback = areas.adjugate()                                  # [] Bivector+ <- Bivector+
```

As algebraic maps, both `areas` and `flux_pullback` remain defined when a deformation becomes singular, extending cofactor area transport beyond the regular regime where a physical Cauchy stress is defined. For a regular deformation, stress conversions follow the same composition: composing a Cauchy traction map (from current patches to current forces) with `areas` gives the first Piola–Kirchhoff map on reference patches, and pulling the force output back through the inverse deformation gives the second Piola–Kirchhoff map, with each factor's domain and codomain explicit.

In the matrix notation of continuum mechanics, for a regular orientation-preserving deformation $F$ with Jacobian determinant $J$, deformed and reference area vectors satisfy $n\,da=J F^{-\mathsf T}N\,dA$ ([Piola transformation](https://authors.library.caltech.edu/records/s9jhk-sn323/files/MFE_Chapter_1.pdf?download=1), §1.7), and for Cauchy traction $t=\sigma n$ the first and second Piola–Kirchhoff stresses read $P=J\sigma F^{-\mathsf T}$ and $S=F^{-1}P$ ([definitions of the Piola–Kirchhoff stresses](https://authors.library.caltech.edu/records/s9jhk-sn323/files/MFE_Chapter_2.pdf?download=1), §2.2).

### Vibration modes

In numerical mechanics, a stiffness or inertia matrix written between dual coordinate conventions can fail to look symmetric on its raw array entries, tempting implementations either to average the matrix with its transpose, which changes the operator, or to form normal equations, squaring its singular values and condition number. When symmetry belongs to an energy pairing between complementary spaces, pairing the map exposes that symmetry directly.

A vibration mode is a motion rate for which elastic and inertial loads stay proportional. The [normal-modes example](../examples/mechanics/modes/modes.ipynb) sums spring stiffnesses and point-mass inertias as maps from twists to forques (`Antibivector- <- Bivector+`), and pairs both with a second twist to form symmetric quadratic energy forms:

```python
pe_form = Twist & stiffness                                       # [] Scalar+ <- (Bivector+, Bivector+)
ke_form = Twist & inertia                                         # [] Scalar+ <- (Bivector+, Bivector+)
values, modes = pe_form.eigh(ke_form)                             # [modes] Scalar+, [modes] Bivector+
```

Symmetry follows directly from the spring and mass dyads, and under the assumption of a positive-definite inertia form, that form supplies the physical inner product between modes without symmetrizing arrays or squaring operators, returning squared angular frequencies in `values` (with zero eigenvalues for unconstrained rigid-body motions).

In matrix notation, modal analysis solves $Ku=\omega^2Mu$ for a real symmetric stiffness matrix $K$ (which may have rigid-motion zero modes when unconstrained) under the assumption of a symmetric positive-definite mass matrix $M$, yielding real eigenvalues and $M$-orthogonal modes ([generalized structural eigenvalue problem](https://www-users.cse.umn.edu/~saad/eig_book_2ndEd.pdf), Ch. 9–10, §9.2.6).

### Fitting a rotation

Fitting a rigid rotation between corresponding vectors via the Kabsch algorithm takes a singular-value decomposition of their cross-covariance matrix and branches on the factor determinants to rule out improper reflections ([algebraic Kabsch–Umeyama derivation](https://nvlpubs.nist.gov/nistpubs/jres/124/jres.124.028.pdf)), as applied per cell in local surface deformation ([as-rigid-as-possible surface modeling](https://igl.ethz.ch/projects/ARAP/arap_web.pdf), §2.1).

In the [as-rigid-as-possible deformation example](../examples/surfaces/arap/arap.ipynb), each vertex instead maximizes edge alignment directly as a quadratic form on four-dimensional Euclidean rotors (`Scalar+ + Bivector+`), reproducing the [unit-quaternion solution of absolute orientation](https://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf) directly from the sandwich product rather than a hand-assembled four-by-four matrix:

```python
alignment = (weights * edges.dot(Rotor >> rest_edges)).sum(axis=0)  # [] Scalar+ <- (Rotor+, Rotor+)
values, rotors = alignment.eigh()                                   # [modes] Scalar+, [modes] Rotor+
rotation = rotors[values.argmax()].normalized()                     # [] Rotor+
```

Because every normalized eigenvector in `Rotor+` is a proper rotation, reflections never enter the search space. Extending `Rotor` to projective or spherical motors (`Rotor+`) in the [point-cloud registration example](../examples/geometry/registration/registration.ipynb) eliminates separate centroid subtraction as well: `target * Motor - Motor * source` (`[points] Antivector- <- Rotor+`) is linear in the motor, so minimizing its summed bulk-and-weight residual against the motor's bulk norm solves for rotation and translation together without choosing an origin.

### Material laws in moving frames

Moving a dielectric medium mixes the electric and magnetic parts of both the field and the induced excitation, producing coupled three-vector formulas in laboratory coordinates ([Minkowski constitutive relations for moving media](https://arxiv.org/pdf/1204.0114), §5.1, Eq. 22). Under those rest-frame response assumptions, the moving medium evaluates its rest-frame law on the field boosted into its rest frame and boosts the resulting excitation back to the laboratory.

The [moving-media constitutive example](../examples/electromagnetism/constitutive/constitutive.ipynb) writes that conjugation directly on the spacetime bivector map:

```python
moving_medium = boost >> rest_medium(boost << Bivector)           # [] Antibivector- <- Bivector+
```

The same pattern moves a rigid-body inertia between body and world frames, while applying a transformation only on the output or only on the input produces distinct maps, as [Extensors through multiple lenses](multiple_lenses.md#2-the-geometric-lens) explains; a Lorentz boost likewise transforms the [moving charge's field gradient](../examples/electromagnetism/moving_charge/moving_charge.ipynb) (`Bivector+ <- Antivector-`).

In matrix notation, transporting a linear map across frames takes the form $A_{\mathrm{out}}TA_{\mathrm{in}}^{-1}$, where $A_{\mathrm{in}}$ and $A_{\mathrm{out}}$ carry the frame representations of the input and output spaces.

## Tensor representations

These cases concern quantities that tensor formulations pack into coordinate arrays or compressed matrices; [Extensors, linear algebra, tensors and forms](frameworks.md) sets out how the underlying frameworks relate.

### Inertia, vorticity and curvature

Three-dimensional vector mechanics represents an angular velocity by an axial arrow along its rotation axis so a three-by-three inertia matrix can map it to an angular-momentum arrow, or assembles angular and linear parts into a six-by-six generalized inertia matrix ([rigid-body inertia construction](https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-complete.pdf), Ch. 4, §2.2), relying on grade 2 and grade 1 sharing dimension three. The [spinning-top example](../examples/mechanics/spinning_top/spinning_top.ipynb) instead builds inertia directly on motion rates (`Bivector+`) by joining each mass point with its induced velocity:

```python
inertia = points.regressive(points.commutator(Rate) * masses).sum()  # [] Antibivector- <- Bivector+
```

The same line includes linear momentum alongside angular momentum in projective space and carries unchanged to the plane with one rotation component and four-space with six. Fluid vorticity and torsion-free spacetime curvature follow the same pattern: the [vortices example](../examples/fluids/vortices/vortices.ipynb) takes the geometric derivative of the velocity field to obtain divergence and the vorticity plane together without a three-dimensional curl, and the [gravitational-wave curvature example](../examples/relativity/curvature/curvature.ipynb) represents Riemann curvature as a map from spacetime area patches (`Antibivector-`) to rotation planes (`Bivector+`), symmetric under the spacetime pairing rather than as a raw Euclidean six-by-six array ([bivector treatment of curvature](https://arxiv.org/pdf/gr-qc/0405033), §5).

In three-dimensional vector and tensor notation, rotational inertia reads $I_{ij}=\sum m(r^2\delta_{ij}-r_i r_j)$, fluid vorticity is the axial curl $\boldsymbol{\omega} = \nabla \times \mathbf{v}$ (twice the antisymmetric continuum spin rate $\frac{1}{2}(\partial_i v_j - \partial_j v_i)$), and torsion-free Riemann curvature is the four-index tensor $R^{\mu}{}_{\nu\alpha\beta}$, antisymmetric within each index pair and symmetric under pair exchange once the first index is lowered with the spacetime metric.

### Elasticity and compressed tensor notation

Computational elasticity commonly packs symmetric strain and stress tensors into six-component columns and stiffness into a six-by-six matrix. Engineering Voigt notation doubles shear strains while leaving shear stresses unscaled, whereas Kelvin (or Mandel) notation scales both by square-root-of-two factors, so stress and strain arrays must use compatible shear factors to preserve their elastic-energy pairing ([Voigt and Kelvin matrix representations](https://d-nb.info/1263215041/34), §§2–3).

The [crystal-waves example](../examples/mechanics/crystal_waves/crystal_waves.ipynb) keeps stiffness as a multilinear extensor taking a face normal, a displacement and a wavevector to a traction, and binds the wave propagation heading to both plane arguments to form the acoustic Christoffel operator directly:

```python
christoffel = crystal(heading, Antivector, heading)               # [] Vector+ <- Antivector-
wave_form = christoffel(Antivector) & Antivector                  # [] Scalar+ <- (Antivector-, Antivector-)
length_form = Antivector.dot(Antivector)                          # [] Scalar+ <- (Antivector-, Antivector-)
values, polarizations = wave_form.eigh(length_form)               # [modes] Scalar+, [modes] Antivector-
```

The eigenvalues `values` are density times squared phase speed and `polarizations` are the wave displacement directions, obtained directly from the material law without six-vector packing or shear-factor conventions.

### Stress index conventions

Specifying a component of stress requires two geometric arguments, the plane of the cut face and the direction in which traction is read. Under balance of moment of momentum, the Cauchy stress is symmetric while the first Piola–Kirchhoff stress has distinct current and reference arguments and is generally unsymmetric ([definitions and balance laws of the Piola–Kirchhoff stresses](https://authors.library.caltech.edu/records/s9jhk-sn323/files/MFE_Chapter_2.pdf?download=1), §2.2), so a transposition mistake hidden by Cauchy symmetry changes the first Piola–Kirchhoff stress as soon as one argument is pulled back to the reference configuration. Typing stress as `Antivector- <- Vector+` declares the cut plane as the input and the traction force as the output once at construction, so application, area pullback and frame changes follow the type rather than an index-order convention remembered on the side.

### Electromagnetic constitutive tensors

Macroscopic electromagnetism commonly relates electric field, magnetic induction, electric displacement and magnetic field strength through separate permittivity, permeability and magnetoelectric matrices, whereas the [premetric formulation of classical electrodynamics](https://arxiv.org/pdf/physics/0610221) separates the untwisted field two-form from the twisted excitation two-form and places all metric and material structure into a constitutive law between them.

For a local linear medium, the [constitutive example](../examples/electromagnetism/constitutive/constitutive.ipynb) types that response as `Antibivector- <- Bivector+` in four-dimensional spacetime: both spaces are six-dimensional, their orientations separate field from excitation, and one linear map composes with boosts, media boundaries and dispersion solves while the exterior Maxwell equations stay untouched.

## Representations in geometric algebra

These cases concern the [multivector representations](complementarity.md#appendix-representations) themselves, and apply even in geometric-algebra calculations that form no higher-arity extensors.

### Spatial vectors

Spatial-vector dynamics, as presented in the [six-dimensional spatial-vector tutorial](https://bleyer.org/files/A%20Beginner%27s%20Guide%20to%206-D%20Vectors%20-%20Feathersone%20%28IEEE%2C%202010%29.pdf), packages rigid-body velocities and forces into dual six-dimensional motion and force spaces, introducing distinct six-by-six Plücker transformation matrices and distinct motion and force cross-product matrices so power stays a scalar pairing. Projective geometric algebra represents twists as `Bivector+` and forques as `Antibivector-` within the same algebra that holds points, planes and lines: one motor sandwich `>>` transforms both, one `.commutator()` differentiates both, and one regressive product `&` pairs them into power.

### Blending rigid motions

Linear blend skinning averages bone transformation matrices entry by entry, which ceases to be rigid and pinches twisting joints, whereas dual-quaternion skinning forms a weighted sum of unit dual quaternions representing finite rigid transformations and normalizes the result after choosing compatible antipodal signs ([dual quaternion linear blending](https://users.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf), §§3.2–3.3). Unit dual quaternions represent rigid transformations only, so non-rigid scale and shear require a separate treatment. The [skeletal-skinning example](../examples/surfaces/skinning/skinning.ipynb) writes the same motor blend alongside matrix skinning in projective three-space:

```python
motor_skin = (root + weight * (tip - root)).normalized() >> vertices                  # [vertices] Antivector-
matrix_skin = ((root >> Point) + weight * ((tip >> Point) - (root >> Point)))(vertices)  # [vertices] Antivector-
```

With the double-cover sign of `tip` aligned to `root` beforehand so antipodal representatives of the same pose do not cancel, the first line applies a normalized rigid motor at every vertex, while the second averages the induced point maps and reproduces linear blend skinning's joint pinching.

### Electromagnetic fields under a boost

Vector electrodynamics transforms electric and magnetic fields under a Lorentz boost through separate parallel and transverse three-vector formulas ([Lorentz transformations of the fields](https://www.feynmanlectures.caltech.edu/II_26.html), §26-3, Table 26-4), whereas tensor electrodynamics transforms an antisymmetric four-by-four field matrix. Spacetime algebra represents the electromagnetic field as a single spacetime bivector boosted by a rotor and split into electric and magnetic parts only relative to a chosen observer's velocity ([spacetime bivector electrodynamics](https://davidhestenes.net/geocalc/pdf/SpacetimePhysics.pdf), §§III–IV, VI), which the [moving-charge example](../examples/electromagnetism/moving_charge/moving_charge.ipynb) types as `Bivector+` and boosts directly from a Coulomb rest field by `boost >> field`.

### Composition of boosts

Two successive Lorentz boosts along non-collinear directions compose into a pure boost accompanied by a spatial Wigner rotation whose angle is not specified by the resultant velocity alone; in the low-speed limit, accumulating those rotations along an accelerated worldline supplies the Thomas correction that halves the naive nonrelativistic spin–orbit term ([Wigner rotation and Thomas precession](https://arxiv.org/pdf/gr-qc/0501070), §§VII.C–VII.D). In the [Thomas-precession example](../examples/relativity/thomas_precession/thomas_precession.ipynb), multiplying boost rotors in spacetime algebra retains both the worldline velocity and the frame rotation in one product, and factoring out the pure boost back to the laboratory frame reads off the accumulated spatial rotation directly.

### Reflections and pseudovectors

Three-dimensional vector calculus distinguishes polar and axial vectors under mirror reflection ([polar and axial vectors](https://www.feynmanlectures.caltech.edu/I_52.html), §52-5), multiplying an orthogonal coordinate transformation by its determinant for angular momenta, torques and magnetic fields ([determinant transformation rule](https://cns.gatech.edu/~predrag/courses/PHYS-4421-04/lautrup/7.7/space.pdf), §2.10) and keeping that rule alongside the coordinates. Carrying intrinsic (`-`) and extrinsic (`+`) orientations on the algebraic type, as [Why orientation](orientation.md) develops, lets one reflection operator `mirror >> x` negate a rotation plane lying in a mirror while preserving a directed momentum line lying in the same mirror and reversing a triangle's winding while keeping its cutting plane's upward side.

In three-dimensional Cartesian vector notation, an orthogonal transformation $M$ carries polar vector coordinates by $M$ and axial vector coordinates by $\det(M)\,M$.

### Quaternion conventions

Quaternion formulas differ in standard-versus-flipped multiplication order, scalar-versus-vector component order and active-versus-passive or world-versus-body frame sense, requiring identification and translation procedures such as those in the [analysis of flipped quaternion multiplication](https://arxiv.org/pdf/1801.07478). In geometric algebra, the geometric product fixes bivector multiplication directly from the basis planes x∧y, y∧z and z∧x without a separate multiplication convention, and `rotor >> value` versus `rotor << value` writes the two sandwich directions explicitly, while frame sense, perturbation side and external interface ordering remain explicit modelling choices.
