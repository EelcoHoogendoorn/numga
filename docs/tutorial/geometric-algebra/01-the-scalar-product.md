# 1. The scalar product of vectors

A product of two vectors has more than one possible definition. The scalar product is one choice, and it keeps very little: a single number, which many vectors share with a given vector, so the product cannot be undone. Other choices keep more. The geometric product is one of them, and it differs from the scalar product in a single rule.

In three dimensions, a vector is a sum of three basis vectors, each with a coefficient:

$$a = a_1 x + a_2 y + a_3 z$$

Multiplying two such vectors term by term gives a grid of nine terms, one for each pair of basis vectors:

|           | $b_1 x$           | $b_2 y$           | $b_3 z$           |
| --------- | ----------------- | ----------------- | ----------------- |
| $a_1 x$   | $a_1 b_1\, x x$   | $a_1 b_2\, x y$   | $a_1 b_3\, x z$   |
| $a_2 y$   | $a_2 b_1\, y x$   | $a_2 b_2\, y y$   | $a_2 b_3\, y z$   |
| $a_3 z$   | $a_3 b_1\, z x$   | $a_3 b_2\, z y$   | $a_3 b_3\, z z$   |

The scalar product of the two keeps only the diagonal of the grid:

$$a \cdot b = a_1 b_1 + a_2 b_2 + a_3 b_3$$

The scalar product of a vector with itself is the sum of the squares of its coefficients, its squared length by Pythagoras. The diagonal terms keep their coefficients and lose their basis vectors, and every term off the diagonal drops out. That does not follow from the expansion. It comes from two rules about basis vectors, the defining choices of the scalar product of vectors:

$$e_i e_i = 1, \qquad e_i e_j = e_j e_i = 0 \quad \text{for } i \neq j$$

The first keeps the diagonal and turns each of its terms into a number; the second drops every term off it:

|       | $x$ | $y$ | $z$ |
| ----- | --- | --- | --- |
| $x$   | 1   | 0   | 0   |
| $y$   | 0   | 1   | 0   |
| $z$   | 0   | 0   | 1   |

The same in numga:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
a.scalar_product(b)
```

```text
32
```
