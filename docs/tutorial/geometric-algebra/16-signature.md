# 16. Signature

The multiplication rules of the geometric product as presented so far set the square of each basis vector to one. That is a choice as well, and not the only useful one: the square can instead be $-1$ or $0$, with the rule for two different basis vectors unchanged:

$$e_i e_i = s_i \in \{1, -1, 0\}, \qquad e_i e_j + e_j e_i = 0 \quad \text{for } i \neq j$$

The list of squares is the signature of the algebra. With $x$ squaring to $1$, $y$ to $-1$ and $z$ to $0$, written `x+y-z0` in numga:

```python
y * y
```

```text
-1
```

```python
z * z
```

```text
0
```

```python
x * y + y * x
```

```text
0
```

Two different basis vectors still anticommute. The squares of two bivectors:

```python
(x * y) * (x * y)
```

```text
1
```

```python
(x * z) * (x * z)
```

```text
0
```

A bivector can square to $1$ or to $0$, depending on the squares of its basis vectors. The square of the vector $x + y$:

```python
(x + y) * (x + y)
```

```text
0
```

A vector that is not zero can square to zero. The grades, the $2^n$ basis elements and their multiplication table carry over unchanged; only the signs and zeros in the table change with the signature. Three signatures recur. All squares $1$ is the Euclidean case of this tutorial. One square opposite in sign to the others gives the algebra of spacetime, used in the [relativity examples](../../../examples/relativity/). One square $0$ gives projective geometry, used in the [geometry examples](../../../examples/geometry/).
