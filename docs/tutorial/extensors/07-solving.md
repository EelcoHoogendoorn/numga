# 7. Solving

Binding goes from the axis to the output, and solving goes the other way, finding the value in the axis that gives a known output.

For example, binding a vector $b$ into the axis of a map `A` gives an output $w$:

```python
b = 4 * x + 5 * y + 6 * z
A = Vector + (Vector | (x ^ y))
w = A(b)
```

Solving `A` for $w$ gives back $b$:

```python
A.solve(w) - b
```

```text
  x  y  z
  0  0  0
```

An output in the image of the map determines a unique input when no two inputs give the same output.

Solving against the type `Vector`, in place of a given output, leaves the output open. The result is a map from outputs back to the axis, the solve for every output at once: the inverse of the map.

For the map `A` of the example, solving against `Vector` gives `A.inverse()`:

```python
A.solve(Vector) - A.inverse()
```

```text
   x  y  z
x  0  0  0
y  0  0  0
z  0  0  0
```

Binding $w$ into it gives $b$:

```python
A.solve(Vector)(w) - b
```

```text
  x  y  z
  0  0  0
```

The inverse is the solve precomputed for every output: binding an output into it replaces solving for that output.

Solving against a map in place of a value keeps the axis of that map open. The result takes the input of that map to the value in the axis of the solved map.

For example, with a second map `B`, from bivectors to vectors, solving `A` against the composition `A(B)` gives back `B`, with its bivector axis open. Subtracting `B` leaves the zero map:

```python
B = b | Bivector
A.solve(A(B)) - B
```

```text
   yz  zx  xy
x   0   0   0
y   0   0   0
z   0   0   0
```

In matrix notation, solving reads as the linear system $A v = w$, solved for $v$. With $w$ left open, the solve is the inverse matrix $A^{-1}$; against a matrix $B$, it is $A^{-1} B$, with the input of $B$ left open as the input of the result.

Example: [induced dipoles in a cluster of atoms](../../../examples/electromagnetism/induced_dipoles/induced_dipoles.ipynb), where the dipoles of all the atoms, each driven by the fields of the others, follow from one solve.
