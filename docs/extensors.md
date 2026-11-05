# Definition

In mathematical terms, an extensor is a multilinear map from multivectors to a multivector. Tensors are a special case of extensors, restricted to inner products of 1-vectors.

In programming terms, extensors let us build maps directly from geometric expressions. On the blackboard one moves freely between a specific vector and the space of all vectors; extensor syntax brings that flexibility to code.

Geometrically, objects and transformations belong to the same algebra. A lens is both an object we can move and a map that acts on rays. An extensor built from geometry takes its meaning from that construction and transforms as its geometry does.

# Motivation

Geometric relationships deserve to be first-class objects alongside the objects they relate. Inertia, stiffness and material responses are maps we construct, combine, transform and solve with. Extensors keep those relationships in the same algebra as their ingredients.

# Companion documents

* [`multiple_lenses.md`](multiple_lenses.md): the three perspectives developed in detail, from maps built from expressions to their geometric meaning and multilinear structure.
* [`extensor_syntax.md`](extensor_syntax.md): the syntax and the extension methods, as a reference sheet.
* [`internals.md`](internals.md): what the library builds from an expression, and what runs when values are supplied.
* [Complete index of 155 worked examples](../examples/README.md).

# Examples

In the selected examples below, lowercase names denote values and capitalized names denote spaces: `v ^ Vector` is a map from vectors to bivectors.

---

## 1. Multi-View Reconstruction and Camera Alignment

**Notebook**: [`examples/estimation/multiview/multiview_reconstruction.ipynb`](../examples/estimation/multiview/multiview_reconstruction.ipynb)

<p align="center"><img src="../plots/multiview_convergence.gif" alt="Three cameras aligned step by step, with their sight cones and the splats they fuse into" width="300" /></p>

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

## 2. Light Past a Rotating Black Hole

**Notebook**: [`examples/relativity/kerr/kerr.ipynb`](../examples/relativity/kerr/kerr.ipynb)

**Explore it live**: [the black hole](https://eelcohoogendoorn.github.io/numga/examples/relativity/kerr/kerr_webgpu.html)

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

## 3. Snowflake Growth

**Notebook**: [`examples/thermodynamics/snowflake/snowflake.ipynb`](../examples/thermodynamics/snowflake/snowflake.ipynb)

<p align="center"><img src="../plots/snowflake.gif" alt="A seed of ice in undercooled water growing into a six-armed dendrite, its latent heat glowing into the cold water around it" width="300" /></p>

```python
gradients = (volume & ice & heat & sides) / double_areas                              # [corners] Plane+[faces]
heat_drives = -(gradients ^ mv.h)                                                     # [corners] Drive+[faces]
freezing = ice + latent_heat * heat                                                   # [] Point-
energy = mv.h - latent_heat * mv.i                                                    # [] Plane+
entropy_slope = mv.h / absolute_temperatures - latent_heat / absolute_melting * mv.i  # Plane+[vertices]
push = absolute_melting * cells * (entropy_slope & freezing) * held_slope             # Scalar-[vertices]
```

* Freezing is one direction among the amounts the water holds, ice gained and its latent heat released, and the two laws of thermodynamics are two planes reading it: `energy & freezing == 0`, so freezing makes no energy, and `entropy_slope & freezing` is the entropy it produces, which drives the ice.
* A corner's gradient is the facing side joined with every amount, over the triangle joined with them, so it reads place and nothing a parcel holds; wedged with heat it is the drive that conducts the latent heat away from the growing tips.

---

## 4. A Rotating Bose–Einstein Condensate

**Notebook**: [`examples/quantum/condensate/condensate.ipynb`](../examples/quantum/condensate/condensate.ipynb)

<p align="center"><img src="../plots/condensate.gif" alt="A rotating condensate in a harmonic trap settling into a lattice of quantized vortices" width="300" /></p>

```python
drags = (corners.mean(axis=0) * vertices).commutator(mv.xy)                       # Point-[F]
along = (areas / 3)[..., None, None] * (drags[..., None, None] & gradients[..., None, :])
stepping = (spdiag((cells / step + cells * potential) * Wave) + assembled(kinetic * Wave) +
            assembled(rotation * along * (Wave * mv.ab)))                         # (Wave+ <- Wave+)[V <- V]
turns = (ends[..., 0] * ends[..., 1]).normalized().log()                          # Turn+[E]
```

* Two turnings in two planes, kept apart: the trap drags each point a quarter turn in its plane, `point.commutator(mv.xy)`, and the wave turns in its own, `wave * mv.ab`; the rotation's energy pairs the drag with the wave's gradient.
* A vortex is a triangle round which the wave turns once: the logarithms of the rotors between neighbouring waves, summed round its edges, counted against Feynman's rule.

---

## 5. Potential Flow past a Joukowski Wing

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
* Summed around a loop, `velocity * step` gives circulation and flux in one even multivector.

---

## 6. A Dynamo in a Planet's Core

**Notebook**: [`examples/electromagnetism/dynamo/dynamo.ipynb`](../examples/electromagnetism/dynamo/dynamo.ipynb)

<p align="center"><img src="../plots/dynamo_lines.gif" alt="A planet's liquid core seen from above, its field lines in white wound up by the stirred flow, whose streamlines run in cyan inside the core's faint surface" width="256" /></p>

```python
motional = spdiag((velocities & Magnetic) ^ mv.t)(ball.curl)     # (Field+ <- Scalar+)[tets <- edges]
conductance = ball.electric.adjugate()(current(ball.electric))   # (Amount- <- Scalar+)[edges <- edges]
drive = ball.electric.adjugate()(current(motional))              # (Amount- <- Scalar+)[edges <- edges]
kinetic = circulating.adjugate()(momentum(circulating))          # (Amount- <- Scalar+)[edges <- edges]
```

* The moving fluid reads the magnetic plane as an electric field, `(velocity & magnetic) ^ mv.t`: the plane met with the velocity, swept through time.
* The field's force on the fluid is that motional map's adjugate applied to the current, so the energy the field gains is the energy the flow loses.
* The field is the curl of a potential on the edges: it has no sources, however it grows.

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

## 8. Gravitational Waves of a Closed Universe

**Notebook**: [`examples/relativity/gravitational_waves/waves.ipynb`](../examples/relativity/gravitational_waves/waves.ipynb)

**Explore it live**: [the gravitational wave](https://eelcohoogendoorn.github.io/numga/examples/relativity/gravitational_waves/gravitational_waves_webgpu.html)

<p align="center"><img src="../plots/gravitational_waves_quadrupole.gif" alt="A gravitational wave travelling round a closed universe, its stretch drawn on a sphere, rung by strain gauges and crossed by light" width="300" /></p>

```python
ramps = faces / (faces & corners)                                                     # [corners] Plane+[cells]
cone = radial * (radial & Point) + cone_map * squared_lengths                         # (Plane+ <- Point-)[cells]
pairings = ramps[:, None] & cone.batch().inverse().field()(ramps[None, :])            # [corners, corners] Scalar+[cells]
restoring = onto_waves(masses.solve(stiffness(strains)))                              # Scalar+[edges]
strains = (cell_frames.inverse() >> (cone_map * changes)(frame_steps)).cast(Strain)   # (Plane+ <- Point-)[cells]
```

* Regge calculus on curved tetrahedra: each tetrahedron's metric is a map from points to planes, `cone`, and the angles between its faces are its ramps paired through that map's inverse.
* The kinetic term is the inertia of space, `masses`, a map from the rates of the squared lengths to their momenta.
* The closed universe carries a frame everywhere at once. Read in it, a wave is a strain field, and a stretch is the strain read against the metric it changes, `strain(step) & step`.

---

## 9. Edge States of a Graphene Flake

**Notebook**: [`examples/quantum/kane_mele/kane_mele.ipynb`](../examples/quantum/kane_mele/kane_mele.ipynb)

<p align="center"><img src="../plots/kane_mele.gif" alt="An electron's two spins running round the edge of a honeycomb flake in opposite directions" width="400" /></p>

```python
turns = (near - flake.between).commutator(flake.between - far).normalized()   # [pairs] Bivector+
energy = hamiltonian(flake, spin_orbit, 0.0)                                  # Up+[atoms <- atoms]
values, states = (energy * sector).eigh(unit * sector, count)                 # [count] Scalar+, [count] Even+[atoms]
amplitudes = weights * (mv.xy * (-energies * time)).exp()                     # [states] Even+
spins = (states * amplitudes).sum(axis=0) >> up_spin                          # Spin-[atoms]
```

* Spin conservation is a type: `energy * Up` is the Hamiltonian on one spin sector.
* The spin-orbit hop couples by a bivector, the plane a second-neighbour path turns in, the commutator of its two steps.
* A state evolves by a turn on its right at the rate of its energy, `(mv.xy * (-energies * time)).exp()`, with no imaginary unit.

---

## 10. Beltrami Fields

**Notebook**: [`examples/electromagnetism/beltrami_fields/beltrami_fields.ipynb`](../examples/electromagnetism/beltrami_fields/beltrami_fields.ipynb)

<p align="center"><img src="../plots/beltrami.gif" alt="Tangled magnetic field lines in a torus, seen from the side and from above, settling into evenly twisted windings" width="548" /></p>

```python
curls = 2 * (corner_gradients[tails] ^ corner_gradients[heads])     # [tet_edges] Field+[tets]
curl = (spdiag(curls * Scalar) * mesh.oriented[3, 1]).sum(axis=0)   # (Field+ <- Scalar+)[tets <- edges]
vacuum = spdiag(prisms.norm() * Field.dual())                       # (Patch- <- Field+)[tets <- tets]
stiffness = curl.adjugate()(vacuum(curl))                           # (Amount- <- Scalar+)[edges <- edges]
volume_weights = spdiag(Plane ^ volumes)                            # (Patch- <- Plane+)[tets <- tets]
raw_helicity = curl.adjugate()(volume_weights(whitney1))            # (Amount- <- Scalar+)[edges <- edges]
helicity = (raw_helicity + raw_helicity.adjugate()) / 2             # (Amount- <- Scalar+)[edges <- edges]
```

* The curl of an edge field is a wedge of two corner gradients, `2 * (corner_gradients[tails] ^ corner_gradients[heads])`, with no cross product.
* Energy and helicity are both pulled back through the curl's adjugate; the energy alone uses the metric, `Field.dual()`, the helicity only a wedge and a signed volume.

---

## 11. A Scenegraph, a Robot Arm and a Camera, as One Map

**Notebook**: [`examples/geometry/scenegraph/scenegraph.ipynb`](../examples/geometry/scenegraph/scenegraph.ipynb)

<p align="center"><img src="../plots/scenegraph.gif" alt="A robot arm in 3D, and its picture on a camera sensor" width="598" /></p>

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

## 12. As Rigid As Possible

**Notebook**: [`examples/surfaces/arap/arap.ipynb`](../examples/surfaces/arap/arap.ipynb)

<p align="center"><img src="../plots/arap_bending.gif" alt="A bar held at one end and bent a quarter turn up, keeping its shape as it bends" width="400" /></p>

```python
edges = T10 * vertices                                                         # Point[E]
forms = ~A10 * H1 * edges.dot(Even >> rest)                                    # Scalar[V] <- (Even, Even)
rotors = forms.batch().eigh()[1][..., -1].field()                              # Even[V]
turned_rest = (A10 * (rotors >> Point))(rest)                                  # Point[E]
vertices = handle_laplacian.solve(~T10 * H1 * turned_rest + P * targets)       # Point[V]
```

* How well a rotor fits a vertex's edges is a form on rotors; the best rotor is its top eigenvector.
* `rotors >> Point` turns rotors into maps, which carry the rest edges.

---

## 13. Bending and Twisting of Thin-Walled Beams

**Notebook**: [`examples/mechanics/bending/bending.ipynb`](../examples/mechanics/bending/bending.ipynb)

<p align="center"><img src="../plots/bending.gif" alt="A cantilever loaded off its principal axes, bending sideways as well as down" width="576" /></p>

```python
fibres = beam_direction & samples(walls, thickness)                                                    # [walls, 2] Resultant
pulling = (modulus * fibres * (Twist & fibres)).sum(axis=(-2, -1))                                      # [] Resultant <- Twist
twisted = twisting(walls, thickness, shear_modulus) * couple_about_beam * (couple_about_beam & Twist)   # [] Forque <- Twist
stiffness = pulling + twisted + sliding                                                                 # [] Forque <- Twist
strains = stiffness.solve(-loads_at_stations(loads, stations))                                          # [..., pieces] Twist
```

* A section's stretch and curvatures are one twist, its force and moments one forque.
* The stiffness is one map `Forque <- Twist`, a sum of fibre dyads `modulus * fibres * (Twist & fibres)`.

---

## 14. Twistors and Linked Light

**Notebook**: [`examples/relativity/twistors/twistors.ipynb`](../examples/relativity/twistors/twistors.ipynb)

<p align="center"><img src="../plots/twistor_hopfion.gif" alt="The linked electric field lines of a pulse of light, carried along straight light rays" width="330" /></p>

```python
representations = state_readout(Full * state_embedding)    # [] Twistor <- (Full, Twistor)
family = representations(points[0] * points[1])            # [] Twistor <- Twistor
states, _, _ = family.svd()
planes = ray_readout(twistors, twistors)                    # [cases] Bivector
```

* One extensor, `Twistor <- (Full, Twistor)`, holds the action of every multivector on twistors.
* A twistor paired with itself, `ray_readout(twistor, twistor)`, is its light ray.

---

## 15. The Meissner Effect

**Notebook**: [`examples/electromagnetism/superconductor/superconductor.ipynb`](../examples/electromagnetism/superconductor/superconductor.ipynb)

<p align="center"><img src="../plots/superconductor.gif" alt="A rod cooled through its superconducting transition expelling the magnetic field it sat in, beside a perfect conductor freezing it in" width="406" /></p>

```python
fields = gradients ^ mv.z                                                                      # [corners] Field+[faces]
local = (fields[..., :, None] & Field.dual()(fields[..., None, :])) * areas[..., None, None]   # [corners, corners] Scalar+[faces]
londons = spdiag(cells * inside * share / penetration ** 2)                                    # Scalar+[vertices <- vertices]
currents = ring_gradients ^ mv.xy.dual()                                                       # [corners] Current-[faces]
kinetic = penetration ** 2 * (in_ring_areas * (currents & inertia(currents))).sites.sum()     # [candidates, fields] Scalar+
```

* London's two equations are two parts of one closure in spacetime, `gradient ^ (momentum + charge * potential) == 0`: its part through time makes a perfect conductor, its part in space expels the field.
* Around a ring the same closure counts whole turns, the fluxoid, so a ring cooled in a field keeps none, one or two flux quanta.

---

## 16. Superfluid Helium: Heat That Travels as a Wave

**Notebook**: [`examples/fluids/superfluid/superfluid.ipynb`](../examples/fluids/superfluid/superfluid.ipynb)

<p align="center"><img src="../plots/superfluid.gif" alt="A pulse of heat in superfluid helium travelling out as a wave of temperature" width="320" /></p>

```python
heating = mv.s - entropies * mv.m                                             # [temperatures] Potential+
stiffnesses = (first_sounds ** 2 / densities) * (mv.m * (mv.m & Amount)) + (
    temperatures / (heat_capacities * densities)) * (heating * (heating & Amount))   # [temperatures] Potential+ <- Amount-
potentials = stiffness(amounts)                                       # Potential+[vertices]
drive = -(gradients ^ (corners * potentials[..., None])).sum(axis=-1)   # Drive+[faces]
velocities = velocities + dt * freedom * drive.commutator(liquid_parcels)   # [components] Point-[faces]
flux = velocities.regressive(liquid_parcels).sum(axis=-1)            # Flux-[faces]
```

* One temperature gradient, two equal and opposite pushes: `drive.commutator(parcels)` sends the normal fluid away from the heat and the superfluid toward it, and the heat moves on as a wave, second sound.
* Energy conserved by construction: what `gradient ^ flux` takes from the cells is what the drive gives the flow.
* The liquid's stiffness is two dyads, compression `mv.m * (mv.m & Amount)` and heating `heating * (heating & Amount)`, so it is symmetric by construction, as Maxwell's relations require.

---

## 17. Charge and Heat in a Magnetic Field

**Notebook**: [`examples/thermodynamics/magnetotransport/magnetotransport.ipynb`](../examples/thermodynamics/magnetotransport/magnetotransport.ipynb)

<p align="center"><img src="../plots/magnetotransport.gif" alt="A bar and a ring warmed at one contact in a magnetic field, as their carriers run from electrons only to holes only, a voltage building across the bar and a current circling the ring where the two kinds balance" width="375" /></p>

```python
parcels = valences * (time + entropies_per_charge * entropy)             # [kinds] Parcel-
push = Drive.commutator(parcels)                                          # [kinds] Spatial- <- Drive+
hall = 1 - valences * mobilities * magnetic[..., None]                   # [..., kinds] Even+
drifting = hall.inverse() * (mobilities * push)                           # [..., kinds] Spatial- <- Drive+
carrying = (densities * drifting.regressive(parcels)).sum(axis=-1)       # [...] Flux- <- Drive+
```

* `hall.inverse() * (mobilities * push)`: a scalar plus a bivector multiplying the push map turns every push through the carrier's Hall angle at once, a rotor acting on a whole map.
* Reversing the field reverses each carrier's rotor, `hall.reverse()`, and turns the conductor's response into its adjugate, `responses.adjugate()`: Onsager–Casimir reciprocity as one identity between maps.
* A carrier is one parcel, `valences * (time + entropies_per_charge * entropy)`, so Seebeck's voltage and Peltier's heat read the same entropy per charge, and Kelvin's relation holds by construction.

---

## 18. Laplacian Eigenfunction Fluids

**Notebook**: [`examples/fluids/eigen_fluids/eigen_fluids.ipynb`](../examples/fluids/eigen_fluids/eigen_fluids.ipynb)

<p align="center"><img src="../plots/eigen_fluids.gif" alt="Vortices of opposite sense carrying one another across the surface of the Stanford bunny, with the streamlines of their flow" width="300" /></p>

```python
gradient = (spdiag(corner_gradients * Scalar) * mesh.incidence[2, 0]).sum(axis=0)   # (Plane+ <- Scalar+)[faces <- vertices]
stiffness = gradient.adjugate()(spdiag(areas * Plane.dual())(gradient))            # (Amount- <- Scalar+)[vertices <- vertices]
eigenvalues, eigenfields = stiffness.eigh(mass, n_modes + 1)                       # [modes + 1] Scalar+, [modes + 1] Scalar+[vertices]
flows = face_gradients ^ patches                                                   # [modes] Point-[faces]
jacobians = face_gradients[:, None] & flows[None, :]                               # [modes, modes] Scalar+[faces]
structure = structure_constants(mesh, patches, gradient, modes)                    # (Scalar+ <- (Scalar+, Scalar+))[modes <- (modes, modes)]
advection = structure(stream, vorticity)                                           # Scalar+[modes]
```

* The flow along a stream function's level lines is its gradient joined with the tangent patch, `gradient ^ patch`, with no normal vector and no quarter turn.
* The Jacobian of two fields is `gradient & (gradient ^ patch)`, with no metric, so all of advection is one map of two arguments over the modes, `structure(stream, vorticity)`.

---

## 19. Quantum Process Tomography in the Projective Bloch Ball

**Notebook**: [`examples/quantum/process_tomography/process_tomography.ipynb`](../examples/quantum/process_tomography/process_tomography.ipynb)

<p align="center"><img src="../plots/tomography.gif" alt="Noisy quantum channels squashing and turning the ball of states, and the channels recovered from measurements" width="720" /></p>

```python
effects = (mv.w + analyzers) / 4                                                                      # [outcomes] Effect+
dephasing = Point + phase_flip * ((mv.xy >> Point) - Point)                                           # [channels] Point- <- Point-
device = rotations >> damping(dephasing)                                                              # [channels] Point- <- Point-
probabilities = effects & outputs[..., None]                                                          # [channels, preparations, outcomes] Scalar+
meets = effects[other_idx[:, 0]] ^ effects[other_idx[:, 1]] ^ effects[other_idx[:, 2]]                # [outcomes] Antivector+
joins = preparations[other_idx[:, 0]] & preparations[other_idx[:, 1]] & preparations[other_idx[:, 2]]   # [preparations] Vector-
learned_channel = (recovered_outputs * (preparation_dual & Point)).sum(axis=-1)                       # [channels] Point- <- Point-
```

* A qubit's state is a point of the Bloch ball, a detector outcome a plane, and a probability their pairing, `effect & state`.
* A channel is a map from points to points, `Point- <- Point-`.
* The reciprocal frames are meets and joins: an outcome's dual is where the other three planes meet, a preparation's dual the plane through the other three. No scalar product enters.

---

## 20. Euler Elastica

**Notebook**: [`examples/mechanics/elastica/elastica.ipynb`](../examples/mechanics/elastica/elastica.ipynb)

<p align="center"><img src="../plots/telephone_cord.gif" alt="A coiled telephone cord pulled out by its handset and let go, sagging, stretching and springing back" width="600" /></p>

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

## 21. Piezoelectric Bimorph

**Notebook**: [`examples/electromagnetism/piezoelectric/piezoelectric.ipynb`](../examples/electromagnetism/piezoelectric/piezoelectric.ipynb)

<p align="center"><img src="../plots/piezoelectric.gif" alt="A clamped strip of two bonded piezoelectric layers bending up and down as the voltage across its electrodes swings, one layer stretching while the other shortens" width="525" /></p>

```python
stretches = (directions & Point) * (Plane & directions.dual())          # [2] Scalar+ <- (Point-, Plane+)
polarized = polarization(Point, gradients) * poling.batch()            # [corners, faces] Point- <- Point-
mixed_cells = -(polarized[:, None] & time).adjugate()(electric[None])  # [corners, corners, faces] Plane+ <- Scalar+
elastic = movement.adjugate()(mechanical(movement))                    # (Plane+ <- Point-)[free <- free]
mixed = movement.adjugate()(coupling(voltage))                         # (Plane+ <- Scalar+)[free <- voltages]
```

* The material is built from its planes' readings of stretch on a displacement and its gradient, `(directions & Point) * (Plane & directions.dual())`, with no six-by-six table.
* The converse effect is the adjugate of the same coupling map, `coupling.adjugate()(motion)`: there is no second constant to keep consistent with the first.

---

## 22. Black Polarization Sandwiches

**Notebook**: [`examples/optics/black_sandwich/black_sandwich.ipynb`](../examples/optics/black_sandwich/black_sandwich.ipynb)

<p align="center"><img src="../plots/sandwich.gif" alt="Light through two identical stacks of a turning wave plate between crossed polarizers: the first stack passes light, the second stays dark" width="720" /></p>

```python
field_along = (Direction & time).dual()                          # [] Field <- Direction
polarizer = field_along * ((Direction & time) & Field)           # [] Field <- (Direction, Direction, Field)

plate = 2 * polarizer(plate_axes, plate_axes) - Field            # [frames] Field <- Field
sandwich = exit(plate(entrance))                                 # [frames] Field <- Field
double_stack = sandwich(sandwich)                                # [frames] Field <- Field
```

* A polarizer reads the field by pairing it with the plane its axis sweeps in time, `Field & (axis & time)`.
* A half-wave plate is built from a polarizer, `2 * polarizer(axis, axis) - Field`.
* One stack passes light, yet `sandwich(sandwich)` is zero for every plate: the stack is a square root of zero.

---

## 23. Gravitational-Wave Antenna Patterns

**Notebook**: [`examples/relativity/ligo/ligo.ipynb`](../examples/relativity/ligo/ligo.ipynb)

<p align="center"><img src="../plots/ligo.gif" alt="Each detector's sensitivity over the sky, turning as the wave's polarization turns" width="720" /></p>

```python
plus = east * (east & Point) - north * (north & Point)                       # [...] Plane+ <- Point-
cross = east * (north & Point) + north * (east & Point)                      # [...] Plane+ <- Point-
stretches = (strain[..., None](arms) & arms) / 2                             # [..., 2] Scalar+
rings = ring + strain_shown * overhead[:, None](ring).dual()                 # [frames, masses] Point-
```

* A wave's strain is a map from points to planes, `east * (east & Point) - north * (north & Point)`.
* An arm reads its stretch by pairing, `strain(arm) & arm`, with no metric; the metric enters only to turn the plane a strain hands a mass into its motion, `.dual()`.

---

## 24. Moving Charges

**Notebook**: [`examples/electromagnetism/moving_charge/moving_charge.ipynb`](../examples/electromagnetism/moving_charge/moving_charge.ipynb)

<p align="center"><img src="../plots/radiating_charge_fast.gif" alt="A charge circling fast, its field spiralling outward" width="300" /></p>

```python
at_rest = weight * mv.t * (separations | Antivector)                  # [...] Vector+ <- Antivector-
gradients = charges.boost >> at_rest(charges.boost << Antivector)     # [...] Vector+ <- Antivector-
field = (Vector ^ gradients(Antivector)).trace(1, 2)                  # [...] Bivector+
```

* A moving charge is its gradient at rest, moved by its boost: `boost >> at_rest(boost << Antivector)`.
* The metric enters only through the vacuum, the map from field to excitation.

---

## 25. Magnetic Resonance

**Notebook**: [`examples/quantum/magnetic_resonance/magnetic_resonance_space.ipynb`](../examples/quantum/magnetic_resonance/magnetic_resonance_space.ipynb)

<p align="center"><img src="../plots/resonance_echo.gif" alt="Spins fanning out in an uneven field and refocusing into an echo" width="603" /></p>

```python
back = process.reverse().symmetric_reverse_product()           # [] State
relaxation = (process >> State) - back.anticommutator(State)   # [] State <- State
echo = waiting(turn(waiting(tip)))                             # [delays, spins] State <- State
sample_echo = echo.mean(axis=-1)                               # [delays] State <- State
```

* Relaxation is a map on states written as its formula, a sandwich minus an anticommutator.
* A pulse sequence is a composition: `waiting(turn(waiting(tip)))` is the spin echo.

---

## 26. Jeffery Orbits

**Notebook**: [`examples/fluids/jeffery/jeffery.ipynb`](../examples/fluids/jeffery/jeffery.ipynb)

<p align="center"><img src="../plots/jeffery.gif" alt="Ellipsoids tumbling in a shear flow, their tips tracing closed orbits, and an unbalanced one wandering" width="720" /></p>

```python
sphere = Plane.dual()                                                                            # [] Point <- Plane
shape = stretch(sphere(stretch.adjugate()))                                                      # [...] Point <- Plane
flow_rate = gradient(shape) - shape(gradient.adjugate())                                         # [...] Point <- Plane
turning_rate = Point.commutator(Bivector)(shape) + shape(Plane.commutator(Bivector))             # [...] Point <- (Plane, Bivector)
spin = (Point & turning_rate(Plane)).trace(1, 2).solve((Point & flow_rate(Plane)).trace(1, 2))   # [...] Bivector
```

* A body is its shape, a map from planes to points, `stretch(sphere(stretch.adjugate()))`.
* The flow acts on the shape from both sides, `gradient(shape) - shape(gradient.adjugate())`.

---

## 27. Slip in Metal Crystals

**Notebook**: [`examples/mechanics/slip_systems/slip_systems.ipynb`](../examples/mechanics/slip_systems/slip_systems.ipynb)

<p align="center"><img src="../plots/slip_systems.gif" alt="Pull directions in a crystal drifting as it slips, coloured by the slip system that leads" width="420" /></p>

```python
winding = (atoms[1] - atoms[0]) & (atoms[2] - atoms[0]) & (atoms[3] - atoms[0])   # [] Scalar-
planes = ((directions[:, 0] & directions[:, 1]) * winding).normalized()          # [planes] Plane+
forces = axis.dual_inverse()[..., None] * (planes & axis[..., None])            # [..., planes] Force+
factors = forces[..., None] & directions                                        # [..., planes, directions] Scalar+
moves = amounts * directions * (planes[:, None] & axis[..., None, None])       # [..., planes, directions] Direction-
move = moves.sum(axis=-1).sum(axis=-1)                                          # [...] Direction-
axis = (axis.commutator(move) / 2).exp() >> axis                                # [...] Direction-
```

* A pull puts on each plane the force along its axis, `axis.dual_inverse() * (planes & axis)`, and the Schmid factor is that force read on a slip direction, `forces & directions`.
* A slip plane is the join of two slip directions, its side given by the winding of the atoms.

---

## 28. Boiling at Fixed Pressure

**Notebook**: [`examples/thermodynamics/phase_change/phase_change.ipynb`](../examples/thermodynamics/phase_change/phase_change.ipynb)

<p align="center"><img src="../plots/boiling.gif" alt="A fluid heated at fixed pressure, its path drawn across the dome of coexistence on the pressure–volume chart and along the flat stretch of boiling on the temperature–enthalpy chart" width="660" /></p>

```python
liquid_volume = mv.l & State                                                                       # [] Scalar+ <- State-
vapour_volume = mv.g & State                                                                       # [] Scalar+ <- State-
liquid, vapour = liquid_volume(state), vapour_volume(state)                                        # [...] Scalar+, Scalar+
pressure_gap = pressure(liquid, temperature) - pressure(vapour, temperature)                       # [...] Scalar+
chemical_gap = chemical_potential(liquid, temperature) - chemical_potential(vapour, temperature)   # [...] Scalar+
state = state - derivative(gap)(state).solve(gap(state))                                           # [...] State-
```

* The state space is an exterior algebra with no metric, and the volumes are read by the regressive product alone, `mv.l & State`.
* The derivative of the mismatch is a map `State- <- State-`, and a Newton step solves it.
* Coexistence is a common tangent of the Helmholtz energy, and a chemical potential is that tangent's intercept at zero volume.

---

## 29. Vibration Modes of a Suspended Plate

**Notebook**: [`examples/mechanics/modes/modes.ipynb`](../examples/mechanics/modes/modes.ipynb)

<p align="center"><img src="../plots/modes.gif" alt="The three vibration modes of the coupled suspension" width="603" /></p>

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

## 30. The Most Likely Trajectory from Odometry

**Notebook**: [`examples/estimation/odometry/odometry.ipynb`](../examples/estimation/odometry/odometry.ipynb)

<p align="center"><img src="../plots/odometry.gif" alt="A robot's lap: dead reckoning drifts open, while the most likely path settles onto the true one and its uncertainty ellipses shrink" width="500" /></p>

```python
weights = noises.inverse()                                            # [readings] Forque <- Twist
step = (aligned / (pushed & direction).sum(axis=-1))[..., None]       # [..., 1] Scalar
correction = correction + direction * step                            # [..., poses] Twist
residual = residual - pushed * step                                   # [..., poses] Forque
```

* A covariance is a map from forques to twists; its inverse weighs motion errors.
* Conjugate gradients pair residual forques with correction twists by `&`; no inner product on motions is needed.

---

## 31. Area Transport through a Collapse

**Notebook**: [`examples/mechanics/area_transport/area_transport.ipynb`](../examples/mechanics/area_transport/area_transport.ipynb)

<p align="center"><img src="../plots/area_transport.gif" alt="A cube flattened while its layers slide, its faces' areas and its volume measured throughout" width="600" /></p>

```python
areas = deformation.outermorphism(Patch)   # [frames] Patch <- Patch
volume = deformation.det()                 # [frames] Scalar
adjugate = areas.adjugate()                # [frames] Flux <- Flux
```

* One point map gives patches by its outermorphism and volumes by its determinant.
* The adjugate carries flux back with no inverse, `adjugate(flux) & patch == flux & areas(patch)`, through the collapse to zero volume.

---

## 32. Satellite Navigation

**Notebook**: [`examples/relativity/gps/gps.ipynb`](../examples/relativity/gps/gps.ipynb)

<p align="center"><img src="../plots/gps.gif" alt="Six satellites spreading across the sky above a receiver at Earth's pole, their signals' paths to it, and the ellipsoid of its fitted position's uncertainty, long while they bunch and shrinking as they spread" width="320" /></p>

```python
cones = emitted | (emitted & Event)                                # [signals] Slice+ <- Event-
difference = cones[1:] - cones[0]                                  # [signals - 1] Slice+ <- Event-
axis = radicals[0] ^ radicals[1] ^ radicals[2]                     # [] Antibivector+
tangent = cones(receiver[..., None])                               # [signals] Slice+
information = (weight * readout * (readout & Step)).sum(axis=-1)   # [] Readout+ <- Step-
covariance = information.inverse()                                 # [] Step- <- Readout+
```

* A light cone is a map from events to planes, `emitted | (emitted & Event)`, and `event & cone(event)` is the interval. Two cones differ by a pair of planes, so three radical planes meet in a line that a cone cuts at the reception.
* `cone(receiver)` is the tangent plane that reads a step of the reception. Positive dyads of these readings sum to the information map, whose inverse is one covariance over place and time, so the clock's uncertainty widens the position's.

# References

* <a id="ref-ca-to-gc"></a>**[ca-to-gc]** D. Hestenes and G. Sobczyk, *Clifford Algebra to Geometric Calculus: A Unified Language for Mathematics and Physics*, Reidel, 1984. Extensors are defined in Section 3-10, "Tensors"; extensor fields and their differentials in Section 4-1. [Link](https://math.mit.edu/~dunkel/Teach/18.S996_2022S/books/Hestenes-Sobczyk1984_Book_CliffordAlgebraToGeometricCalc.pdf)
