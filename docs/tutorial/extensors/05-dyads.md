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

The outputs $a$, $b$ and $c$ can be chosen freely, so this construction can give any linear map on vectors. They can also have any grade: opening slots, multiplying scalar outputs by multivectors and adding maps are enough to build every linear map from vectors to multivectors.

Each contribution has one fixed output, multiplied by a scalar-valued linear function of the input. A nonzero map of this form has rank one and is called a dyad. Every output of a dyad lies along its fixed output element. The inner product is one way to supply the scalar function; the regressive product with a complementary grade is another.

The construction extends to any finite-dimensional multivector input space: each basis coefficient is a scalar-valued linear function of the input. Assigning an output to each basis element and adding those dyads constructs the whole map.

Sums of dyads also describe the second moment of a collection of vectors, as in the [tides example](../../../examples/mechanics/tides/core.py).

In matrix notation, the first dyad is $a e_x^T$, where $e_x^T$ extracts the first input coordinate. The sum $A$ has the columns $a$, $b$ and $c$.
