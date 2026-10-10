# 7. Grades and multivectors

The grade of a basis element is the number of basis vectors in it. The basis elements of three-dimensional space, sorted by grade:

| grade | basis elements      | count |
| ----- | ------------------- | ----- |
| 0     | $1$                 | 1     |
| 1     | $x$, $y$, $z$       | 3     |
| 2     | $xy$, $xz$, $yz$    | 3     |
| 3     | $xyz$               | 1     |

Each count is the number of ways to choose that many of the three basis vectors. In $n$ dimensions there are as many basis elements of grade $k$ as there are ways to choose $k$ of the $n$ basis vectors, and together they number $2^n$:

$$\binom{n}{0} + \binom{n}{1} + \dots + \binom{n}{n} = 2^n$$

A product of three vectors works out as follows:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
d = 7 * x + 8 * y + 10 * z
a * b * d
```

```text
140 x + 247 y + 386 z - 3 xyz
```

It has a part of grade 1 and a part of grade 3, where the product of two vectors has parts of grade 0 and 2. A product of an even number of vectors has only even grades, and a product of an odd number only odd ones.

A general element of the algebra is a sum of parts of every grade: a scalar, a vector, a bivector and a trivector. It is called a multivector.

Selecting a grade keeps the part of a multivector of that grade and discards the rest:

```python
(a * b * d).select_grade(3)
```

```text
-3 xyz
```
