"""Test ability to extend dynamically dispatched methods"""

import numpy as np
import numpy.testing as npt
from numga.backend.numpy.context import NumpyContext as Context
from numga.multivector.test.util import *


def test_normalize():
	ga = Context('x+y+z+w0')
	m = random_non_motor(ga, (10,))
	print()
	print(m)
	print(m.norm())
	n1 = m.normalized()

	print()
	print(n1)
	print(n1.norm())

	from numga.multivector.extension import optimized
	print()
	n2 = m.normalized()
	print(n2)
	print(n2.norm())

	assert_close(n1, n2)


def test_exp():
	np.random.seed(0)
	from time import time
	from numga.multivector.extension import optimized
	from numga.multivector.extension.logexp import exp_bisect
	from numga.backend.numpy.operator import NumpyEinsumOperator as Operator

	ga = Context('x+y+z+w0', otype=Operator)
	B = ga.subspace.bivector()

	subspaces = [B, B.degenerate(), B.nondegenerate()]
	for s in subspaces:
		print(s)
		b = random_subspace(ga, s, (10, 10, ))
		b = random_subspace(ga, s, ( ))
		t = time()
		m1 = exp_bisect(b)
		print(time() - t)
		t = time()
		m2 = b.exp()
		print(time() - t)

		assert_close(m1, m2, atol=1e-8)
