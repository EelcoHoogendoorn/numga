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

This relation also determines how to move a map whose construction is unknown. Rotating an input back puts it in the frame of $A$. Applying $A$ there and rotating its output forward gives the moved map. For the map of the example:

```python
(rotor >> A(rotor << Vector)) - moved
```

```text
   x  y  z
x  0  0  0
y  0  0  0
z  0  0  0
```

The forward and backward rotations themselves have open slots, so moving the map is a composition of three maps:

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

Each rotation acts on the grade of the slot it occupies. For a map from vectors to bivectors, the input rotates as a vector and the output as a bivector:

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

The rotor supplies the rotation of each grade directly. Composing those rotations with the input and output of a map gives its action in the rotated frame.

In matrix notation, for a map from vectors to vectors, rotating both sides reads as $R A R^{-1}$. Rotating only the output reads as $R A$, and rotating only the input back as $A R^{-1}$.
