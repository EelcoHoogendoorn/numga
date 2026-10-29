# Extensors and Common Practice

This document compares common approaches to applied problems with the constructions that arise naturally in geometric algebra and extensors. Each case describes how the problem is usually formulated, develops the geometric construction, and examines the practical consequences for modelling and computation. The comparisons focus on what becomes simpler, which choices and conventions become explicit, and where mistakes can arise. Together, they show how a choice of mathematical language shapes the work of building and solving a model.

The [framework comparison](frameworks.md) develops the relation to linear algebra, tensors and forms. The snippets use the [extensor syntax](extensor_syntax.md), [conventions](conventions.md) and [orientation rules](orientation.md) documented separately.

## Maps and numerical methods

### Scene reconstruction and camera alignment

Scene reconstruction estimates landmark positions and camera poses from image measurements. A common formulation minimizes weighted squared image errors, using Gauss–Newton and the normal equations to compute an update. A Schur complement eliminates the landmark steps before solving for the camera steps. [Ceres](https://ceres-solver.readthedocs.io/latest/nnls_solving.html) develops these methods for camera alignment.

The [multiview example](../examples/estimation/multiview/core.py) represents each image measurement by a quadratic cost on the sensor. Pulling that cost through the camera gives a sight cone. Its polar plane vanishes on the sight line, making it tempting to treat the plane as a residual and form normal equations from its coefficients. But the cone already supplies a quadratic cost and its weighting. Camera motion can enter that cost directly:

```python
motion = -Twist.commutator(seen)                         # Point <- Twist
curvature = (cones(motion) & motion).sum(axis=0)         # Scalar <- (Twist, Twist)
slope = (cones(seen) & motion).sum(axis=0)               # Scalar <- Twist
step = curvature.lstsq(-slope, rcond=1e-4)               # Twist
```

Squaring the polar-plane residual applies the cone's weighting twice and introduces a coefficient norm on planes. Using the cone value directly preserves the intended weighting and avoids building a second least-squares objective around an already quadratic cost. The curvature and slope are geometric expressions ready for the numerical solve.

The joint solve also benefits from choosing the unknowns geometrically. Cameras step by twists and landmarks by directions, so the landmark blocks contain no redundant homogeneous scale. With the landmarks constrained by the views, `point_curvature[:, None].solve(cross_curvature)` gives their response to a camera step by a direct solve, avoiding a pseudoinverse over homogeneous point coordinates. Schur elimination then folds those responses into the camera problem. Depth weighting relates the cone costs to pixel errors, and fixed cameras anchor the scene.

In matrix notation, a cone cost $x^{\mathsf T}Qx$ and a point-motion map $M$ give the local quadratic form $M^{\mathsf T}QM$. Treating $Qx$ as a Euclidean residual instead gives $M^{\mathsf T}Q^{\mathsf T}QM$. The extra copy of the cone comes from the choice of objective. The geometric construction makes that choice visible.[^multiview-history]

### Motion and force in robot control

A robot can move freely in some directions while pressing against a constraint in others. Splitting these tasks by Euclidean orthogonality of six motion coordinates mixes angular and linear quantities and depends on how the coordinates are chosen. Duffy's [critique of orthogonal complements in hybrid control](https://onlinelibrary.wiley.com/doi/10.1002/rob.4620070202) concerns this identification of motion and force spaces. Their physically meaningful relation is the power a force system supplies to a motion.

In numga a rate is a twist and a force system a forque, with complementary grades and opposite orientations. `twist & forque` gives power. In the [odometry example](../examples/estimation/odometry/core.py), the same distinction separates a pose correction from the differential of its cost: information takes twists to forques, and covariance runs in the opposite direction. Conjugate gradients pairs those types and preconditions by solving the information map.

The types carry the distinction through the products, compositions and solves. An inertia or information form supplies a meaningful weighting of motions; a coefficient dot product would silently choose its own weighting of angular and linear coordinates.

### Surface normals under deformation

A surface normal can describe a chosen positive side or encode the area and winding of a patch. Graphics commonly stores either as three coefficients. Under an invertible linear deformation, a side covector uses the inverse transpose; an area normal uses the cofactor matrix. They differ by the determinant, including its sign under a reflection. Reed develops the distinction through [areas](https://www.reedbeta.com/blog/normals-inverse-transpose-part-1/) and [dual spaces](https://www.reedbeta.com/blog/normals-inverse-transpose-part-2/).

The [normals example](../examples/orientation/normals/core.py) gives the side and winding different orientations. Reflecting a triangle carries its chosen side and its ordered vertices consistently, even when their normal coefficients acquire different signs. For a general deformation, the outermorphism carries the patch forward. The [adjugate](tutorial/extensors/10-the-adjugate.md) is the dual map on complementary spaces, defined by `T.adjugate()(c) & x == c & T(x)`. It carries a plane condition backward:

```python
input_plane = deformation.adjugate()(output_plane)       # Plane
```

This pullback exists even for a singular deformation. In dual bases it is represented by a transpose; carrying an input plane forward while preserving its readings requires solving through it, which gives the inverse transpose when the deformation is invertible. The distinction prevents a backward measurement map from being used as a forward normal transform. It also leaves the cofactor available when a volume collapses but some surface areas survive.

### Area elements and stress measures

A deformation changes both the area on which a load acts and the directions in which that load is expressed. Continuum mechanics records these choices through the Cauchy and Piola–Kirchhoff stress measures. Their component formulas combine the deformation gradient, its inverse and its determinant.

The [area transport example](../examples/mechanics/area_transport/core.py) carries an oriented patch by moving its spanning edges together. A flux density is carried backward through that area map so that it measures the same flux:

```python
areas = deformation.outermorphism(Patch)                 # Patch <- Patch
flux_pullback = areas.adjugate()                          # Flux <- Flux
```

Both maps remain defined when the deformation is singular. Stress conversion follows the same geometry: composing a map from current patches to forces with `areas` makes it act on reference patches. Expressing its output in reference directions requires a further inverse deformation. Each factor has a role on an input or an output, which makes the placement of the factors visible.

In continuum-mechanics matrix notation, Nanson's relation is $n\,da=J F^{-\mathsf T}N\,dA$. With traction written as $t=\sigma n$, the first and second Piola–Kirchhoff stresses are $P=J\sigma F^{-\mathsf T}$ and $S=F^{-1}P$. The cofactor extends to singular $F$; the expressions requiring $F^{-1}$ do not. [Traction and stress](https://archive.nptel.ac.in/content/storage2/courses/105106049/lecnotes/mainch4.html) develops these measures and their index conventions.

### Vibration modes

A common frustration in numerical geometry is a matrix that fails to display the symmetry expected from the physical model. Averaging it with its transpose is a tempting repair; forming normal equations is another familiar route to a symmetric system. The first changes the operator, and the second squares its singular values and condition number. When the symmetry belongs to a pairing between different spaces, the useful step is to recover that pairing.

A vibration mode is a motion for which elastic and inertial loads are proportional. The [modes example](../examples/mechanics/modes/core.py) constructs stiffness from springs and inertia from masses. Both take twists to forques, so their coefficient arrays need not display the symmetry of the energies. Pairing each output with a twist exposes it:

```python
pe_form = Twist & stiffness                              # Scalar <- (Twist, Twist)
ke_form = Twist & inertia                                # Scalar <- (Twist, Twist)
values, modes = pe_form.eigh(ke_form)
```

The pairing expresses the original maps as energy forms, preserving their information. Their symmetry follows from the spring and mass constructions, and inertia supplies the physical comparison between modes. The geometry therefore leads directly to a symmetric generalized eigenproblem, without an artificial symmetrization or a normal-equation product.

In structural-mechanics notation the problem is $Ku=\omega^2Mu$, as in [finite-element modal analysis](https://docs.frontistr.com/en/theory/solution_methods/03_eigen.html). The eigenvalues above are squared angular frequencies, and the inertia form must be positive definite on the motions being solved for.

### Fitting a rotation

Rigid registration aligns corresponding points after subtracting their centroids. The Kabsch method takes an SVD of the cross-covariance and includes a determinant correction to select a proper rotation. That correction is easy to mishandle: omitting it permits a reflection, and putting the sign in the wrong place can spoil the fit. The [reflection fix in MapClosures](https://github.com/PRBonn/MapClosures/pull/118) gives a concrete implementation example; the [Kabsch–Umeyama derivation](https://doi.org/10.6028/jres.124.028) states the constrained problem.

The [registration example](../examples/geometry/registration/core.py) instead maximizes alignment as a quadratic form on the four-dimensional Euclidean rotor space:

```python
alignment = target.scalar_product(Rotor >> source).sum(axis=0)
values, rotors = alignment.eigh()
rotation = rotors[values.argmax()].normalized()
```

The normalized solution is a proper rotation, so a reflection cannot be selected. This is the quaternion eigenproblem of [Horn](https://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf), whose four-by-four matrix is usually supplied as a separate coefficient formula. Here the alignment expression constructs that form. The objective and the solution live in rotation space, and the determinant correction disappears from the implementation.

### Material laws in moving frames

A moving dielectric mixes the electric and magnetic components of both the field and the material response. Component formulations express that mixing through coupled transformation formulas. The underlying operation is to evaluate the rest-frame material on the rest-frame field and carry its response back.

The [constitutive example](../examples/electromagnetism/constitutive/core.py) expresses that operation directly:

```python
moving_medium = boost >> rest_medium(boost << Bivector)  # Antibivector <- Bivector
```

The same composition moves an inertia between body and space frames. Applying a motion to the output alone, to the input alone, or to both produces different maps, as the [moving-maps tutorial](tutorial/extensors/08-moving-maps.md) shows. The expression records which frame supplies each argument, reducing the need to reconstruct that information from matrix order. A boost also carries the [moving charge's field gradient](../examples/electromagnetism/moving_charge/core.py), whose type is `Bivector <- Antivector`.

In matrix notation a map moved by a change of frame has the form $A_{\mathrm{out}}TA_{\mathrm{in}}^{-1}$. The matrices on the two sides represent the actions on their respective spaces; they need not be the same matrix.

## Tensor representations

### Inertia, vorticity and curvature

Three-dimensional vector mechanics describes a rotation by an arrow along its axis. An inertia matrix then maps angular velocity to angular momentum. The representation is compact, but the identification of a rotation plane with its normal depends on three-dimensional space and its orientation.

The [spinning top](../examples/mechanics/spinning_top/core.py) builds inertia from each mass point and the velocity induced there by a rate:

```python
inertia = points.regressive(points.commutator(Rate) * masses).sum()  # Forque <- Rate
```

The same construction applies to rotational and translational motion in projective geometry. Changing dimension changes the types and their coefficient counts while preserving the construction. In vector-mechanics notation the rotational part reads $I_{ij}=\sum m(r^2\delta_{ij}-r_i r_j)$; the terms arise when the action on rotation planes is expressed through axis vectors.

Vorticity has a similar distinction. Vector calculus uses the curl, whose magnitude is twice a fluid element's angular speed in rigid rotation. The [vortices example](../examples/fluids/vortices/core.py) keeps the velocity gradient as a map on displacements and obtains its divergence and vorticity together through the geometric derivative. The factor relating vorticity to angular speed remains part of the mathematics; the rotation plane is represented directly.

For curvature, tensor indices express antisymmetry in each pair. The [gravitational-wave example](../examples/relativity/curvature/core.py) makes those pairs the input and output spaces, `Bivector <- Antibivector`: a spacetime area produces a plane of rotation. Pair symmetry and the Bianchi identity remain conditions on curvature. Binding an observer into the map gives the tidal response without expanding the paired indices.

### Elasticity and compressed tensor notation

Elasticity often compresses symmetric strain and stress tensors into six coordinates and the stiffness into a six-by-six matrix. Engineering Voigt notation doubles the shear strains; Mandel notation uses square-root-of-two factors to preserve the tensor inner product in an orthonormal compressed basis. Mixing the encodings gives incorrect energies and spectra. [Helnwein](https://repositum.tuwien.at/handle/20.500.12708/174886) explains the covariant and contravariant coordinates behind these representations.

The [crystal-wave example](../examples/mechanics/crystal_waves/core.py) constructs stiffness as a map taking a face, a displacement and a wavevector to a traction. Binding the propagation heading gives the response whose modes are the wave polarizations:

```python
christoffel = crystal(heading, Antivector, heading)       # Vector <- Antivector
wave_form = christoffel(Antivector) & Antivector          # Scalar <- (Antivector, Antivector)
length_form = Antivector.dual().dot(Antivector.dual())    # Scalar <- (Antivector, Antivector)
values, polarizations = wave_form.eigh(length_form)
```

The eigenvalues are density times squared wave speed. Binding arguments gives the wave problem directly from the material law, with displacement and traction kept distinct. The geometric products determine the contractions, and the calculation bypasses the six-coordinate strain encoding and its shear-factor conventions.

### Stress index conventions

A stress component needs two directions: the face on which the traction acts and the component of the traction being read. Texts differ on which receives the first index, as the [NPTEL stress notes](https://archive.nptel.ac.in/content/storage2/courses/105106049/lecnotes/mainch4.html) explain. Symmetric Cauchy stress can conceal a transposition mistake that becomes visible for first Piola–Kirchhoff stress.

An extensor declares its face argument and force output. Application, composition and frame changes follow that declaration. Converting an external coefficient array still requires knowing its convention, but the convention need not be carried as an informal index rule through the subsequent calculation.

### Electromagnetic constitutive tensors

Electromagnetism commonly separates electric field, magnetic induction, electric displacement and magnetic field strength. Their material relations may be assembled into block matrices. Premetric electrodynamics instead starts from field strength and excitation as distinct two-forms, the excitation twisted, and places the metric and material response in a constitutive map. [Hehl, Itin and Obukhov](https://arxiv.org/abs/physics/0610221) develop this formulation.

The [constitutive example](../examples/electromagnetism/constitutive/core.py) follows that separation with a map of type `Antibivector <- Bivector`. Both spaces have six coefficients in spacetime, while their orientations distinguish field from excitation. The same map can be composed with a boost, applied to a field, or inserted into the wave equations.

The practical comparison is with both traditions. The four-field notation exposes an observer's electric and magnetic split; the premetric formulation exposes the constitutive law; the extensor formulation makes that law a geometric map that participates in the same products as the field. A change of medium changes the response map while the exterior field equations keep their form.

## Representations in geometric algebra

These cases concern the geometric objects themselves. Their benefits also apply to GA calculations that do not construct extensors.

### Spatial vectors

[Featherstone's spatial vector algebra](https://royfeatherstone.org/teaching/2008/notes.pdf) packages rigid-body motion and force into separate six-dimensional spaces. Reciprocal bases make their power pairing a coefficient dot product. Motion and force transform differently, and their cross-product actions have separate matrix representations.

Projective GA represents a twist by an extrinsic bivector and a forque by an intrinsic antibivector. One commutator acts on both, one motor action moves both, and the regressive product gives their power pairing. Those same operations act on the points, planes and rays used elsewhere in a robot's model. Separate cross-product matrices, transformation rules and pairing matrices become instances of the algebra's products.

### Blending rigid motions

Linear blend skinning averages transformation matrices. The average need not be rigid, producing the familiar loss of volume at a twisting joint. [Dual quaternion skinning](https://users.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf) blends representations of rigid motions and normalizes the result.

The [skinning example](../examples/surfaces/skinning/core.py) writes both constructions in the same algebra:

```python
motor_skin = (root + weight * (tip - root)).normalized() >> vertices
matrix_skin = ((root >> Point) + weight * ((tip >> Point) - (root >> Point)))(vertices)
```

The first blends the motors; the second blends their point maps. Normalizing the motor blend gives a rigid motion at each vertex and avoids the collapse caused by averaging rotation matrices. Compatible motor signs must be chosen before blending, since a motor and its negative describe the same motion but can cancel in a sum.

### Electromagnetic fields under a boost

Three-vector formulas transform the electric and magnetic fields by splitting each into parts parallel and perpendicular to the relative velocity. A Lorentz-tensor formulation transforms the antisymmetric field tensor instead. Both describe the mixing seen by a moving observer.

Spacetime GA represents the field by one bivector. `boost >> field` transforms it, and the observer supplies the electric and magnetic split when a measurement needs it. The [moving-charge example](../examples/electromagnetism/moving_charge/core.py) uses this action to obtain moving fields from rest-frame constructions. The calculation avoids maintaining separate transformation formulas for the two spatial fields.

### Composition of boosts

Successive boosts in different directions include a spatial rotation. A calculation that tracks only the final velocity discards that rotation. Its effect on spin was the correction in [Thomas's account of the spinning electron](https://www.nature.com/articles/117514a0).

The [Thomas-precession example](../examples/relativity/thomas_precession/core.py) carries a spin by a product of boost rotors. The product keeps the frame's orientation as well as its velocity, so the accumulated rotation follows from composing the boosts. A final pure boost brings the velocity back to the observer's frame and exposes that rotation. The same geometric product handles the boosts, their composition and their action on the spin.

### Reflections and pseudovectors

Vector notation assigns a determinant sign to axial vectors so that angular momentum and magnetic quantities transform correctly under reflection. A separate rule is needed whenever an oriented plane is represented by its normal arrow.

The [orientation examples](why_orientation.md) retain the geometry and its orientation on the type. A rotation plane, a directed line and a surface's positive side can therefore respond differently to the same mirror. The reflection operation propagates those differences through the calculation. This moves the sign bookkeeping into the algebra's operations and permits the result to be checked by reflecting the complete scene.

### Quaternion conventions

Quaternion formulas differ in multiplication order, component order and the direction of the represented frame change. [Sommer and colleagues](https://arxiv.org/abs/1801.07478) document the confusion between Hamilton multiplication and its flipped alternative, and give procedures for identifying and translating conventions.

In numga the geometric product fixes composition, and `rotor >> value` and `rotor << value` distinguish the forward and reverse actions for a normalized rotor. Named blades also identify the planes used to construct it. Exchanging quaternion coefficients with another library still requires a convention conversion; within the geometric calculation, the action and its direction are explicit.

[^multiview-history]: The multiview example underwent this reformulation in [commit `d0b41af`](https://github.com/EelcoHoogendoorn/numga/commit/d0b41af9f270900e609898a54823f367f601ba4e). The earlier implementation squared polar-plane residuals and used a pseudoinverse on homogeneous landmark blocks; the revision worked directly with cone costs and landmark displacements.
