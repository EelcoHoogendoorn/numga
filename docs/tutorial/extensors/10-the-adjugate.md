# 10. The adjugate

A phase measurement on deformed positions can be read as a measurement on the positions before deformation. The adjugate carries the wavevector backward, preserving that reading for every position. Positions are intrinsic antivectors and wavevectors extrinsic vectors; their regressive product gives the phase:

```python
from numga import NumpyContext
from numga.algebras import VGA3D

ga = VGA3D
mv = NumpyContext(ga).multivector
Position = ga.gatype.antivector()                 # intrinsic
Wavevector = ga.gatype.vector()                  # extrinsic

displacement = mv.antivector([1, 2, 3])
deformation_axis = mv.vector([4, 5, 6])
deformation = Position + displacement * (deformation_axis & Position)
wavevector = mv.z

# Find the wavevector reading the same phase on every undeformed position.
input_wavevector = (Wavevector & Position).solve(wavevector & deformation)
input_wavevector
```

```text
   x   y   z
  12  15  19
```

The pairing `Wavevector & Position` identifies a wavevector with its phase measurement. Solving through this pairing expresses the composed measurement as a wavevector on the input side. The two measurements agree:

```python
(input_wavevector & Position) - (wavevector & deformation)
```

```text
  yz  zx  xy
1  0   0   0
```

In particular, `wavevector & deformation(position) == 0` says that the image position lies on the zero-phase plane. The same condition on its input is `input_wavevector & position == 0`. A plane condition on the outputs has become a plane condition on the inputs.

The adjugate of the deformation maps each output wavevector to the input wavevector with the same phase reading:

```python
adjugate = (Wavevector & Position).solve(Wavevector & deformation)
adjugate - deformation.adjugate()
```

```text
   x  y  z
x  0  0  0
y  0  0  0
z  0  0  0
```

The construction gives `adjugate(wavevector) & position == wavevector & deformation(position)` for every position and wavevector.

For other input and output grades, the adjugate uses the same pairing with complementary grades and opposite orientations. It runs from complements of outputs to complements of inputs. Because `&` pairs every basis blade with its complement without using the inner product, the adjugate exists even when basis vectors square to zero and even when the deformation itself cannot be inverted.

In the notation of projective geometric algebra, which writes the regressive product as $\vee$, the identity reads as $\operatorname{adj} T(k) \vee p = k \vee T(p)$.

In matrix notation, with positions in the basis $yz$, $zx$, $xy$ and wavevectors in the paired basis $x$, $y$, $z$, the coefficients of this adjugate read as $T^\mathsf{T}$.

Example: [multi-view reconstruction](../../../examples/estimation/multiview/multiview_reconstruction.ipynb), where a camera's adjugate carries a pixel's disc of uncertainty back into a cone of sight.

Example: [crystal diffraction](../../../examples/optics/crystal_diffraction/crystal_diffraction.ipynb), where solving through the deformation's adjugate carries wavevectors forward while preserving their phases on the moving atoms.
