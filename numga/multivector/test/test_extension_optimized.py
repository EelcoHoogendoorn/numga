"""Test ability to extend dynamically dispatched methods"""

import numpy as np
import numpy.testing as npt
from numga.backend.numpy.context import NumpyContext as Context


def test_normalize_extended():
	ga = Context('x+y+z+w0')
	m = ga.multivector.motor(np.random.normal(size=(20, 8)))
	print()
	print(m)
	print(m.norm())
	n1 = m.normalized()

	print()
	print(n1)
	print(n1.norm())

	from numga.multivector import extension_optimized
	print()
	n2 = m.normalized()
	print(n2)
	print(n2.norm())

	npt.assert_allclose(n1.values, n2.values, atol=1e-12)
