# 15. The regressive product

The regressive product of two elements $A$ and $B$ is the dual of the outer product of their duals, with the pseudoscalar $I = xyz$ of three-dimensional space:

$$A \vee B = \big((A I) \wedge (B I)\big)\, I^{-1}$$

For $xy$ and $yz$, worked out by this definition:

```python
I = x * y * z
((x * y) * I ^ (y * z) * I) * I.inverse()
```

```text
y
```

numga writes the regressive product as `&`:

```python
(x * y) & (y * z)
```

```text
y
```

The regressive product of $xy$ and $yz$ is $y$, the basis vector the two share. The same for the bivectors $a \wedge b$ and $b \wedge d$:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
d = 7 * x + 8 * y + 10 * z
(a ^ b) & (b ^ d)
```

```text
-12 x - 15 y - 18 z
```

The result is $-3 b$, a multiple of $b$, the vector the two bivectors share.
