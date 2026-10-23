# Definition

An extensor is a multilinear map from multivectors to a multivector: an expression whose arguments are bound later. On the blackboard one moves freely between a specific vector and the space of all vectors; extensors bring that to code. The definition is Hestenes and Sobczyk's [[ca-to-gc](#ref-ca-to-gc)], for whom extensors are what tensors become when their arguments are multivectors. Built from the algebra's products, they transform as the geometry they are built from.

# Motivation

Inertia, stiffness and material responses are maps we construct, combine, transform and solve with. Extensors make them first-class, written with the same products as the geometry they relate.

Extensors include matrices and tensors. A matrix is an extensor that takes a vector and returns a vector, and a tensor is an extensor that takes only vectors. Extensors take any kind of multivector, such as a plane, a line or a rotor, and can be built from any product of the algebra.

# Companion documents

* [`multiple_lenses.md`](multiple_lenses.md): the definition read three ways, as partial application, as objects and transformations in one algebra, and as multilinear maps.
* [`extensor_syntax.md`](extensor_syntax.md): the syntax and the extension methods, as a reference sheet.
* [`internals.md`](internals.md): what the library builds from an expression, and what runs when values are supplied.
* [`examples/README.md`](../examples/README.md): the complete index of examples.

# Examples

The repository holds 138 worked examples, each with a notebook, indexed in [`examples/README.md`](../examples/README.md); the gallery below shows favourites. In the code, capitalized names stand for whole spaces of multivectors and lower case names for specific ones, so `v ^ V` is a vector wedged with every vector, a map from vectors to bivectors.

---

## 1. A Scenegraph, a Robot Arm and a Camera, as One Map

**Notebook**: [`examples/geometry/scenegraph/scenegraph.ipynb`](../examples/geometry/scenegraph/scenegraph.ipynb)

![A robot arm in 3D, and its picture on a camera sensor](../plots/scenegraph.gif)

```python
lens = Line - (center & (Line ^ plane)) / focal_length                                            # [] Line <- Line
rear_lens = rear_placement >> thin_lens(origin, front_plane, focal_rear)(rear_placement << Line)   # [] Line <- Line
to_sensor = Line ^ sensor_plane                                                                   # [] Point <- Line
camera = to_sensor(rear_lens(front_lens(incoming_rays)))                                          # [] Point <- Point
local_to_pixel = world_to_pixel(bodies_to_world)                                                  # [bodies] Point <- Point
```

* A thin lens is a map from rays to rays, `Line - (center & (Line ^ plane)) / focal_length`.
* A motor moves a lens as it moves any map, `rear_placement >> lens(rear_placement << Line)`.
* Calling a map on a map composes them, so each part gets one map from points to pixels.

---

## 2. Light Past a Rotating Black Hole

**Notebook**: [`examples/relativity/kerr/kerr.ipynb`](../examples/relativity/kerr/kerr.ipynb)

<p align="center"><img src="../plots/kerr.gif" alt="The hot disk of gas around a rotating black hole, bent around its shadow as it orbits" width="400" /></p>

```python
dyad = geometry.profile * geometry.null * (geometry.null_plane & Antivector)   # [...] Antivector <- Antivector
gauge = Antivector + dyad                                                     # [...] Antivector <- Antivector
gauge_inverse = Antivector - dyad                                             # [...] Antivector <- Antivector
momentum = -gauge_inverse(local_null).dual()                                  # [...] Vector
overlap_gradient = geometry.null_gradient.adjugate()(momentum)                # [...] Vector
```

* The geometry is a gauge field, a map from local directions to Kerr's, not a metric.
* That map is the identity plus a null dyad; the dyad squares to zero, so the inverse subtracts it.
* The force on a ray is an adjugate, `geometry.null_gradient.adjugate()(momentum)`.

---

## 3. Multi-View Reconstruction and Camera Alignment

**Notebook**: [`examples/estimation/multiview/multiview_reconstruction.ipynb`](../examples/estimation/multiview/multiview_reconstruction.ipynb)

<p align="center"><img src="../plots/multiview_convergence.gif" alt="Three cameras aligned step by step, with their sight cones and the splats they fuse into" width="390" /></p>

```python
back_projection = camera.adjugate()                        # [] Plane <- Plane
cones = back_projection(discs(camera))                     # [landmarks, cameras] Plane <- Point
splats = (poses >> cones(poses << Point)).sum(axis=-1)     # [landmarks] Plane <- Point
motion = -Twist.commutator(seen)                           # [landmarks, cameras] Point <- Twist
curvature = (cones(motion) & motion).sum(axis=0)           # [cameras] Scalar <- (Twist, Twist)
```

* A camera has no inverse, but its adjugate maps a pixel's uncertainty disc to a cone of sight.
* Views combine by adding cones, `(poses >> cones(poses << Point)).sum(axis=-1)`.
* The cost's curvature over camera motions is `cones(motion) & motion`; no Jacobian is formed.

---

## 4. Potential Flow past a Joukowski Wing

**Notebook**: [`examples/fluids/wing/wing.ipynb`](../examples/fluids/wing/wing.ipynb)

<p align="center"><img src="../plots/wing.gif" alt="A wing pitching up and back in a steady stream, its pressure, streamlines and lift" width="400" /></p>

```python
velocity = stream - radius_squared * (inverse * stream * inverse) - swirl * inverse / (2 * np.pi)   # [...] Point
change = -(inverse * Point * inverse)                                                               # [...] Point <- Point
gradient = -radius_squared * (change * stream * inverse + inverse * stream * change) - swirl * change / (2 * np.pi)
derivatives = (Point * gradient(Point)).contract(1, 2)                                              # [rings, angles] Even
potential_gradient = (velocity | Point)(jacobian.inverse())                                         # [...] Scalar <- Point
```

* The geometric derivative `(Point * gradient(Point)).contract(1, 2)` holds divergence and vorticity in one even multivector, and ideal flow makes it vanish.
* The velocity's gradient is written out as a map, `change = -(inverse * Point * inverse)`.
* Summed around a loop, `velocity * step` gives circulation and flux in one even multivector.

---

## 5. Edge States of a Graphene Flake

**Notebook**: [`examples/quantum/kane_mele/kane_mele.ipynb`](../examples/quantum/kane_mele/kane_mele.ipynb)

<p align="center"><img src="../plots/kane_mele.gif" alt="An electron's two spins running round the edge of a honeycomb flake in opposite directions" width="400" /></p>

```python
energy = hamiltonian(flake, spin_orbit, 0.0)                      # [atoms, atoms] Up
values, states = (energy * sector).eigh(unit * sector, count)     # [count] Scalar, [count] sector[atoms]
turned = weights * (mv.xy * (-energies * time)).exp()             # [states] Even
spin = (states * turned).sum(axis=0) >> mv.z                      # Vector[atoms]
```

* Spin conservation is a type: `energy * Up` is the Hamiltonian on one spin sector.
* Each sector is solved on its own, `(energy * sector).eigh(unit * sector, count)`.
* A state's spin density is a sandwich, `state >> mv.z`.

---

## 6. Twistors and Linked Light

**Notebook**: [`examples/relativity/twistors/twistors.ipynb`](../examples/relativity/twistors/twistors.ipynb)

![The linked electric field lines of a pulse of light, carried along straight light rays](../plots/twistor_hopfion.gif)

```python
representations = state_readout(Full * state_embedding)    # [] Twistor <- (Full, Twistor)
family = representations(points[0] * points[1])            # [] Twistor <- Twistor
states, _, _ = family.svd()
planes = ray_readout(twistors, twistors)                    # [cases] Bivector
```

* One extensor, `Twistor <- (Full, Twistor)`, holds the action of every multivector on twistors.
* The twistors two events both annihilate come from one SVD of `representations(points[0] * points[1])`.
* A twistor paired with itself, `ray_readout(twistor, twistor)`, is its light ray.

---

## 7. The Rattleback

**Notebook**: [`examples/mechanics/rattleback/rattleback.ipynb`](../examples/mechanics/rattleback/rattleback.ipynb)

<p align="center"><img src="../plots/rattleback.gif" alt="A wooden rattleback spun the wrong way, rocking, stopping and spinning back" width="420" /></p>

```python
height = (normal & shape(normal)).square_root()                        # Scalar
support = -shape(normal) / height                                      # Direction
rolling = ((centre + support) * origin.inverse() / 2).exp() >> Spin   # Rate <- Spin
pullback = rolling.adjugate()                                          # AngularMomentum <- Forque
effective = pullback(inertia(rolling))                                 # AngularMomentum <- Spin
```

* Rolling is a spin carried to the contact point, a map `Rate <- Spin`.
* Its adjugate removes the contact reaction: `pullback(force) & spin == force & rolling(spin)`.
* The constrained inertia is the free one pulled back, `pullback(inertia(rolling))`, with no Lagrange multiplier.

---

## 8. Bending and Twisting of Thin-Walled Beams

**Notebook**: [`examples/mechanics/bending/bending.ipynb`](../examples/mechanics/bending/bending.ipynb)

![A cantilever loaded off its principal axes, bending sideways as well as down](../plots/bending.gif)

```python
fibres = beam_direction & samples(walls, thickness)                                                    # [walls, 2] Resultant
pulling = (modulus * fibres * (Twist & fibres)).sum(axis=(-2, -1))                                      # [] Resultant <- Twist
twisted = twisting(walls, thickness, shear_modulus) * couple_about_beam * (couple_about_beam & Twist)   # [] Forque <- Twist
stiffness = pulling + twisted + sliding                                                                 # [] Forque <- Twist
strains = stiffness.solve(-loads_at_stations(loads, stations))                                          # [..., pieces] Twist
```

* In a projective algebra a section's stretch and curvatures are one twist, its force and moments one forque.
* The stiffness is one map `Forque <- Twist`, a sum of fibre dyads `modulus * fibres * (Twist & fibres)`.
* Bending sideways under an off-axis load comes out of that map.

---

## 9. Quantum Process Tomography in the Projective Bloch Ball

**Notebook**: [`examples/quantum/process_tomography/process_tomography_projective.ipynb`](../examples/quantum/process_tomography/process_tomography_projective.ipynb)

![Noisy quantum channels squashing and turning the ball of states, and the channels recovered from measurements](../plots/tomography.gif)

```python
effects = (mv.w + analyzers) / 4                                                                      # [outcomes] Vector
dephasing = Point + phase_flip * ((mv.xy >> Point) - Point)                                           # [channels] Point <- Point
device = rotations >> damping(dephasing)                                                              # [channels] Point <- Point
probabilities = effects & outputs[..., None]                                                          # [channels, preparations, outcomes] Scalar
meets = effects[other_idx[:, 0]] ^ effects[other_idx[:, 1]] ^ effects[other_idx[:, 2]]                # [outcomes] Antivector
joins = preparations[other_idx[:, 0]] & preparations[other_idx[:, 1]] & preparations[other_idx[:, 2]]   # [preparations] Vector
learned_channel = (recovered_outputs * (preparation_dual & Point)).sum(axis=-1)                       # [channels] Point <- Point
```

* A qubit's state is a point of the Bloch ball, a detector outcome a plane, and a probability their pairing, `effect & state`.
* A channel is a map from points to points; dephasing mixes a state with its half turn, `Point + phase_flip * ((mv.xy >> Point) - Point)`.
* The reciprocal frames are meets and joins: an outcome's dual is where the other three planes meet, a preparation's dual the plane through the other three. No scalar product enters.

---

## 10. Euler Elastica

**Notebook**: [`examples/mechanics/elastica/elastica.ipynb`](../examples/mechanics/elastica/elastica.ipynb)

![A coiled telephone cord pulled out by its handset and let go, sagging, stretching and springing back](../plots/telephone_cord.gif)

```python
turns = couples * (couples & Twist) * turning                                          # [3] Forque <- Twist
shifts = force_axes * (force_axes & Twist) * shifting                                  # [3] Forque <- Twist
stiff = (turns + shifts).sum(axis=0)                                                   # [] Forque <- Twist
deviation = (rest.reverse() * frames[..., :-1].reverse() * frames[..., 1:]).log() * 2   # [..., joints] Twist
load = stiff(deviation / step)                                                         # [..., joints] Forque
load = load - deviation.commutator(load) / 2 + deviation.commutator(deviation.commutator(load)) / 12
```

* A joint's strain is the logarithm of its neighbours' relative motor, a twist.
* The stiffness is a sum of dyads, `couples * (couples & Twist) * turning` plus the shifts.
* The load reaches the frames through the derivative of the exponential, two commutator terms.

---

## 11. Black Polarization Sandwiches

**Notebook**: [`examples/optics/black_sandwich/black_sandwich.ipynb`](../examples/optics/black_sandwich/black_sandwich.ipynb)

![Light through two identical stacks of a turning wave plate between crossed polarizers: the first stack passes light, the second stays dark](../plots/sandwich.gif)

```python
def polarizer(axis: Direction) -> OpticalMap:
    return field_along(axis) * (Field & (axis & time))           # [...] Field <- Field

plate = 2 * polarizer(plate_axes) - Field                        # [frames] Field <- Field
sandwich = exit(plate(entrance))                                 # [frames] Field <- Field
double_stack = sandwich(sandwich)                                # [frames] Field <- Field
```

* A polarizer reads the field by pairing it with the plane its axis sweeps in time, `Field & (axis & time)`.
* A half-wave plate is built from a polarizer, `2 * polarizer(axis) - Field`.
* One stack passes light, yet `sandwich(sandwich)` is zero for every plate: the stack is a square root of zero.

---

## 12. Gravitational-Wave Antenna Patterns

**Notebook**: [`examples/relativity/ligo/ligo.ipynb`](../examples/relativity/ligo/ligo.ipynb)

![Each detector's sensitivity over the sky, turning as the wave's polarization turns](../plots/ligo.gif)

```python
plus = east * (east & Point) - north * (north & Point)    # [...] Plane <- Point
cross = east * (north & Point) + north * (east & Point)   # [...] Plane <- Point
stretches = (strain[..., None](arms) & arms) / 2          # [..., 2] Scalar
```

* A wave's strain is a map from points to planes, `east * (east & Point) - north * (north & Point)`.
* An arm reads it as `strain(arm) & arm`.
* Turning the polarization turns every antenna pattern twice as fast: spin two.

---

## 13. Moving Charges

**Notebook**: [`examples/electromagnetism/moving_charge/moving_charge.ipynb`](../examples/electromagnetism/moving_charge/moving_charge.ipynb)

<p align="center"><img src="../plots/radiating_charge_fast.gif" alt="A charge circling fast, its field spiralling outward" width="420" /></p>

```python
at_rest = weight * mv.t * (separations | Antivector)                  # [...] Vector <- Antivector
gradients = charges.boost >> at_rest(charges.boost << Antivector)     # [...] Vector <- Antivector
field = (Vector ^ gradients(Antivector)).trace(1, 2)                  # [...] Bivector
```

* A moving charge is its gradient at rest, moved by its boost: `boost >> at_rest(boost << Antivector)`.
* The field is the summed gradients wedged with `Vector` and traced.
* The metric enters only through the vacuum, the map from field to excitation.

---

## 14. Magnetic Resonance

**Notebook**: [`examples/quantum/magnetic_resonance/magnetic_resonance_space.ipynb`](../examples/quantum/magnetic_resonance/magnetic_resonance_space.ipynb)

![Spins fanning out in an uneven field and refocusing into an echo](../plots/resonance_echo.gif)

```python
back = process.reverse().symmetric_reverse_product()           # [] State
relaxation = (process >> State) - back.anticommutator(State)   # [] State <- State
echo = waiting(turn(waiting(tip)))                             # [delays, spins] State <- State
sample_echo = echo.mean(axis=-1)                               # [delays] State <- State
```

* Relaxation is a map on states written as its formula, a sandwich minus an anticommutator.
* A pulse sequence is a composition: `waiting(turn(waiting(tip)))` is the spin echo.
* Averaged over spins, `echo.mean(axis=-1)`, the sample's echo is one map.

---

## 15. As Rigid As Possible

**Notebook**: [`examples/surfaces/arap/arap.ipynb`](../examples/surfaces/arap/arap.ipynb)

<p align="center"><img src="../plots/arap_bending.gif" alt="A bar held at one end and bent a quarter turn up, keeping its shape as it bends" width="360" /></p>

```python
edges = T10 * vertices                                                         # Point[E]
forms = ~A10 * H1 * edges.dot(Even >> rest)                                    # Scalar[V] <- (Even, Even)
rotors = forms.batch().eigh()[1][..., -1].field()                              # Even[V]
turned_rest = (A10 * (rotors >> Point))(rest)                                  # Point[E]
vertices = handle_laplacian.solve(~T10 * H1 * turned_rest + P * targets)       # Point[V]
```

* How well a rotor fits a vertex's edges is a form on rotors; the best rotor is its top eigenvector.
* `rotors >> Point` turns rotors into maps, which carry the rest edges.
* `~T10 * H1 * T10` is the cotangent Laplacian, a map between fields.

---

## 16. Jeffery Orbits

**Notebook**: [`examples/fluids/jeffery/jeffery.ipynb`](../examples/fluids/jeffery/jeffery.ipynb)

![Ellipsoids tumbling in a shear flow, their tips tracing closed orbits, and an unbalanced one wandering](../plots/jeffery.gif)

```python
sphere = Plane.dual()                                                                            # [] Point <- Plane
shape = stretch(sphere(stretch.adjugate()))                                                      # [...] Point <- Plane
flow_rate = gradient(shape) - shape(gradient.adjugate())                                         # [...] Point <- Plane
turning_rate = Point.commutator(Bivector)(shape) + shape(Plane.commutator(Bivector))             # [...] Point <- (Plane, Bivector)
spin = (Point & turning_rate(Plane)).trace(1, 2).solve((Point & flow_rate(Plane)).trace(1, 2))   # [...] Bivector
```

* A body is its shape, a map from planes to points, `stretch(sphere(stretch.adjugate()))`.
* The flow acts on the shape from both sides, `gradient(shape) - shape(gradient.adjugate())`.
* The body's spin is one solve, matching turning rate to flow rate.

---

## 17. Slip in Metal Crystals

**Notebook**: [`examples/mechanics/slip_systems/slip_systems.ipynb`](../examples/mechanics/slip_systems/slip_systems.ipynb)

<p align="center"><img src="../plots/slip_systems_00.gif" alt="Pull directions in a crystal drifting as it slips, coloured by the slip system that leads" width="420" /></p>

```python
directions = (planes[:, None] ^ planes[others]).normalized()                 # [planes, directions] Line
tension = axis * (Plane & axis)                                               # [...] Direction <- Plane
factors = directions.dot(tension(axis[..., None, None])(planes[:, None]))    # [..., planes, directions] Scalar
moves = amounts * directions * (planes[:, None] & axis[..., None, None])     # [..., planes, directions] Direction
axis = (axis.commutator(move) / 2).exp() >> axis                              # [...] Direction
```

* A pull is a stress, a map from planes to tractions, `axis * (Plane & axis)`.
* Slip directions are where dense planes meet, `plane ^ other`.
* Slip turns the pulled axis by a rotor, `(axis.commutator(move) / 2).exp() >> axis`.

---

## 18. Vibration Modes of a Suspended Plate

**Notebook**: [`examples/mechanics/modes/modes.ipynb`](../examples/mechanics/modes/modes.ipynb)

![The three vibration modes of the coupled suspension](../plots/modes.gif)

```python
extension = Twist & lines                                        # [springs] Scalar <- Twist
stiffness = (lines * extension * spring_constants).sum(axis=0)   # [] Forque <- Twist
velocities = body.commutator(Twist)                              # [corners] Point <- Twist
inertia = ((body & velocities) * masses).sum(axis=0)             # [] Forque <- Twist
values, modes = (Twist & stiffness).eigh(Twist & inertia)        # [modes] Scalar, [modes] Twist
```

* Stiffness and inertia are sums over springs and mass points, maps `Forque <- Twist` with no origin to choose.
* `(Twist & stiffness).eigh(Twist & inertia)` solves one energy form against the other; the modes are twists.

---

## 19. The Most Likely Trajectory from Odometry

**Notebook**: [`examples/estimation/odometry/odometry.ipynb`](../examples/estimation/odometry/odometry.ipynb)

![A robot's lap: dead reckoning drifts open, while the most likely path settles onto the true one and its uncertainty ellipses shrink](../plots/odometry.gif)

```python
weights = noises.inverse()                                            # [readings] Forque <- Twist
step = (aligned / (pushed & direction).sum(axis=-1))[..., None]       # [..., 1] Scalar
correction = correction + direction * step                            # [..., poses] Twist
residual = residual - pushed * step                                   # [..., poses] Forque
```

* A covariance is a map from forques to twists; its inverse weighs motion errors.
* Conjugate gradients pair residual forques with correction twists by `&`; no inner product on motions is needed.

---

## 20. Area Transport through a Collapse

**Notebook**: [`examples/mechanics/area_transport/area_transport.ipynb`](../examples/mechanics/area_transport/area_transport.ipynb)

![A cube flattened while its layers slide, its faces' areas and its volume measured throughout](../plots/area_transport.gif)

```python
areas = deformation.outermorphism(Patch)   # [frames] Patch <- Patch
volume = deformation.det()                 # [frames] Scalar
adjugate = areas.adjugate()                # [frames] Flux <- Flux
```

* One point map gives patches by its outermorphism and volumes by its determinant.
* The adjugate carries flux back with no inverse, `adjugate(flux) & patch == flux & areas(patch)`, through the collapse to zero volume.

# References

* <a id="ref-ca-to-gc"></a>**[ca-to-gc]** D. Hestenes and G. Sobczyk, *Clifford Algebra to Geometric Calculus: A Unified Language for Mathematics and Physics*, Reidel, 1984. Extensors are defined in Section 3-10, "Tensors"; extensor fields and their differentials in Section 4-1. [Link](https://math.mit.edu/~dunkel/Teach/18.S996_2022S/books/Hestenes-Sobczyk1984_Book_CliffordAlgebraToGeometricCalc.pdf)
