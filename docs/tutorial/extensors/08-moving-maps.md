# 8. Moving maps

A map built from geometric elements can be moved by rotating those elements. For example, a map built from two vectors, and the same construction with both vectors rotated:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
rotor = ((x ^ y) * (-pi / 4)).exp()
A = Vector | (a ^ b)
moved = Vector | ((rotor >> a) ^ (rotor >> b))
```

A rotated input gives the rotated output:

```python
c = 7 * x + 8 * y + 10 * z
moved(rotor >> c) - (rotor >> A(c))
```

```text
  x  y  z
  0  0  0
```

Rotating an input back into the frame of $A$, applying $A$ there, and rotating its output forward gives the same moved map:

```python
(rotor >> A(rotor << Vector)) - moved
```

```text
   x  y  z
x  0  0  0
y  0  0  0
z  0  0  0
```

The map `rotor >> Vector` rotates vectors. Moving $A$ is therefore a composition of three maps:

```python
rotation = rotor >> Vector
rotation(A(rotation.inverse())) - moved
```

```text
   x  y  z
x  0  0  0
y  0  0  0
z  0  0  0
```

Each rotation acts on the grade of its argument. For a map from vectors to bivectors, the input rotates as a vector and the output as a bivector:

```python
area = a ^ Vector
(rotor >> Bivector)(area(rotor << Vector)) - ((rotor >> a) ^ Vector)
```

```text
    x  y  z
yz  0  0  0
zx  0  0  0
xy  0  0  0
```

Rotations and the maps they move are built in the same way: a map both acts on inputs and moves by composing rotations with its input and output.

In matrix notation, for a map from vectors to vectors, rotating both sides reads as $R A R^{-1}$. Rotating only the output reads as $R A$, and rotating only the input back as $A R^{-1}$.

Example: [rigid bodies on a sphere](../../../examples/quadrics/elliptic_physics/s2_physics.ipynb), where every body's shape, a map between points and planes, is placed by its motor.
