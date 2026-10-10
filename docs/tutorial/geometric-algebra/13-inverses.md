# 13. Inverses

The row for $xy$ in the multiplication table contains every basis blade once, up to sign. Multiplying a multivector by $xy$ rearranges its coefficients:

```python
blade = x * y
value = 2 + x + 3 * y + 4 * (y * z)
blade * value
```

```text
3 x - y + 2 xy + 4 xz
```

Every coefficient is still present. The same holds for every row and column of the multiplication table: multiplying by a basis blade permutes the basis elements and changes some signs. The multiplication can be undone.

Multiplying by $xy$ twice gives a sign change, since its square is:

```python
blade * blade
```

```text
-1
```

Changing the sign of the second factor gives:

```python
blade * (-blade)
```

```text
1
```

The product in the opposite order is also one. The element $-xy$ is therefore the inverse of $xy$. An inverse multiplies its element to one on either side:

$$B\, B^{-1} = B^{-1} B = 1$$

```python
blade.inverse()
```

```text
-xy
```

Multiplying by the inverse recovers the entire input multivector:

```python
blade.inverse() * (blade * value)
```

```text
2 + x + 3 y + 4 yz
```

The diagonal of the multiplication table gives the inverse of every basis blade. A basis blade that squares to $1$ is its own inverse; one that squares to $-1$ has its negative as its inverse.
