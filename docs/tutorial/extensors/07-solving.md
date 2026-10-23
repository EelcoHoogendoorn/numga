# 7. Solving

Applying a map gives the output for a known input. Solving finds the input that gives a known output.

For example, applying `A` to a vector $b$ gives an output $w$:

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

The inverse maps each output back to its input. Solving `A` against the identity map `Vector` constructs this inverse.

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

The solve also accepts a map: `A.solve(B)` maps each argument of `B` to the input of `A` whose output equals `B` applied to that argument.

For example, with a second map `B`, from bivectors to vectors, `A.solve(A(B))` recovers `B`:

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

In matrix notation, solving reads as the linear system $A v = w$, solved for $v$. The inverse is $A^{-1}$, and solving against a matrix $B$ gives $A^{-1} B$.

Example: [induced dipoles in a cluster of atoms](../../../examples/electromagnetism/induced_dipoles/induced_dipoles.ipynb), where the dipoles of all the atoms, each driven by the fields of the others, follow from one solve.
