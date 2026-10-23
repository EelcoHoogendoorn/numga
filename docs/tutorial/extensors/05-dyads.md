# 5. Dyads

A linear map is determined by its outputs on a basis. The scalar map `x | Vector` picks out the coefficient of $x$; multiplying it by a chosen output assigns that output to $x$. For example:

```python
a = 1 * x + 2 * y + 3 * z
a * (x | Vector)
```

```text
   x  y  z
x  1  0  0
y  2  0  0
z  3  0  0
```

The map sends $x$ to $a$, and $y$ and $z$ to zero. Adding contributions for $y$ and $z$ assigns their outputs independently:

```python
b = 4 * x + 5 * y + 6 * z
c = 7 * x + 8 * y + 10 * z
A = a * (x | Vector) + b * (y | Vector) + c * (z | Vector)
A
```

```text
   x  y   z
x  1  4   7
y  2  5   8
z  3  6  10
```

The map `A` sends $x$ to $a$, $y$ to $b$ and $z$ to $c$. Linearity determines its output for every other input. For example:

```python
A(2 * x - y + z) - (2 * a - b + c)
```

```text
  x  y  z
  0  0  0
```

The outputs $a$, $b$ and $c$ can be chosen freely, so this construction can give any linear map on vectors. They can also have any grade: multiplying linear scalar measurements by multivectors and adding the resulting maps builds every linear map from vectors to multivectors.

Each term in the sum has one fixed output, scaled by a scalar product with the input. Such a one-direction map is a dyad: every output lies along its fixed output element, and sums of dyads build any linear map. The scalar factor can come from the inner product or from the regressive product `&` with a complementary grade.

In matrix notation, the first dyad is $a e_x^T$, where $e_x^T$ extracts the first input coordinate. The sum $A$ has the columns $a$, $b$ and $c$.

Examples: [tidal forces on a satellite](../../../examples/mechanics/tides/tides.ipynb), where a body's mass distribution is a sum of dyads of its point positions; [bending of a beam](../../../examples/mechanics/bending/bending.ipynb), where each fibre contributes a dyad `fibre * (Strain & fibre)` and their sum is the beam's stiffness; and [the cone of a rational Bézier curve](../../../examples/geometry/bezier/bezier.ipynb), where the form of a conic is a sum of dyads of plane normals.
