# Extensor cheatsheet

Capitalized names are open GA types; lowercase names are supplied geometry or parameters. Mechanics and optics use projective GA; electromagnetism and curvature use spacetime GA.

## Mass points → inertia → response

Sum each point's momentum under an open motion to construct the body's inertia.

```python
velocity = points.commutator(Bivector)
inertia = (points & velocity * masses).sum()

rate = inertia.inverse()(momentum)
kinetic_energy = (rate & momentum) * 0.5
```

## Spring geometry → vibration modes

Construct stiffness and inertia from geometry, then pair them with an open twist to obtain energy forms.

```python
lines = (anchors & attachments).normalized()
extension = Twist & lines
stiffness = (lines * extension * spring_constants).sum()

inertia = (mass_points & mass_points.commutator(Twist) * masses).sum()
values, modes = (Twist & stiffness).eigh(Twist & inertia)
```

## Point correspondences → best rotation

In Euclidean 3D, leaving both rotor occurrences open makes the alignment objective a quadratic form.

```python
rotated = Rotor.sandwich(source)
alignment = target.scalar_product(rotated).sum()

values, rotors = alignment.eigh()
rotation = rotors[-1].normalized()
```

## Two lenses → a camera

The rear lens is a motor-posed copy of the front lens. Two points define the ray; the sensor meets its image.

```python
front = Line - (origin & (Line ^ lens_plane)) / focal_length
rear = placement >> front(placement << Line)
train = rear(front)

camera = train(Point & Point) ^ sensor_plane
image = camera(subject, aperture_point)
```

## Gravitational wave → tidal acceleration

A null wavevector and transverse orthonormal spatial axes construct plus-polarized curvature. A unit timelike observer extracts its tidal map.

```python
ribbon_x = wavevector.wedge(x)
ribbon_y = wavevector.wedge(y)
curvature = amplitude * (
    ribbon_x * (ribbon_x | Bivector)
    - ribbon_y * (ribbon_y | Bivector)
)

tidal = curvature(observer.wedge(Vector)).commutator(observer)
acceleration = tidal(separation)
```

## Moving anisotropic material → wave polarization

The concrete observer `obs` stays fixed; `x`, `y`, and `z` are orthonormal spatial material axes in its rest frame. At an allowed wavevector, a zero singular value identifies a physical polarization.

```python
obs, x, y, z = mv.t, mv.x, mv.y, mv.z
electric = Bivector.commutator(obs).wedge(obs)
magnetic = Bivector - electric

permittivity = -(eps_x * x * (x | Vector)
              + eps_y * y * (y | Vector)
              + eps_z * z * (z | Vector))
material = permittivity(Bivector.commutator(obs)).wedge(obs)
material = material + magnetic / permeability

boost = ((z ^ obs) * (rapidity / 2)).exp()
moving_material = boost >> material(boost << Bivector)

field = wavevector.wedge(Spatial)
wave_equation = wavevector.commutator(moving_material(field))
_, singular_values, polarizations = wave_equation.svd()
polarization = polarizations[-1]
```
