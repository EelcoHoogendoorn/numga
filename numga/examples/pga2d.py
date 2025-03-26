"""Use pga to compose motions

why not just use 3x3 affine?

same amount of code, eliminates dependency
not really using pga features atm
"""

from numga.backend.numpy.context import NumpyContext
import numpy as np

pga = NumpyContext('w0x+y+')
x = pga.multivector.x
y = pga.multivector.y
w = pga.multivector.w
xy = pga.multivector.xy
wx = pga.multivector.wx
wy = pga.multivector.wy


origin = pga.multivector.yx

identity = pga.multivector.scalar()


def plane(nx, ny, d):
	return x*nx + y*ny - w*d

def point(px, py):
	return translator(px, py) >> origin
	return -(x*px+y*py+w).dual()	# minus here so we rotate from x to y
	# return py * wx - px * wy + xy

def rotor(angle):
	ha = angle/-2
	return xy * np.sin(ha) + np.cos(ha)


def translator(tx, ty):
	return 1 + wx*tx/-2 + wy*ty/-2


from numga.multivector.multivector import AbstractMultiVector as mv
@mv.exp.register(
	lambda s:
	s.inside.bivector() and s.algebra.description.n_dimensions <= 3,
	position=1
)
def exponentiate_bivector_numpy(b: "BiVector") -> "Motor":
	"""optimized 2d pga bivector exponential"""
	av = b.norm().values[..., 0]
	cv = np.cos(av)
	sv = np.where(av > 1e-20, np.sin(av) / av, 1)
	return b * sv + cv



# # god this is so janky... need to add custom subspace ordering to numga to get rid of these signs
signs = 1 - (np.arange(9).reshape(3,3)%2)*2
assert pga.subspace.antivector().named_str == 'wx,wy,xy'
def as_matrix(motor):
	return motor.select.motor().sandwich(pga.subspace.antivector()).kernel
def transform_points(motor, p):
	"""Optimized sandwich implementation for point transformation, eliminating intermediaries"""
	m = (as_matrix(motor) * signs)[::-1,::-1]
	return p.dot(m[1:, 1:]) + m[0:1, 1:]


def test_pga():
	from pygeartrain.core.profiles import epi_hypo_gear
	from pygeartrain.core.profiles import Profile, circle

	gear = Profile.concat([epi_hypo_gear(3, 5, 0.5, 100), circle(0.1)])
	m = translator(1, 2) * rotor(0.1)

	print(origin)
	print(origin.exp())
	# return
	print(point(1, 0))
	print(translator(1,0) >> point(0,0))
	# return
	p = point(-10, 1)
	print(p)
	p = plane(0, 1, 1) ^ plane(1, 0, -10)
	print(p)

	q=(point(0,0)*0.1).exp().sandwich(point(1,0))

	print(q.dual(), point(0,0))
	# FIXME why is y negative? xy rotates negatively; why?
	# return
	# p = xy
	print(p)
	m = p.exp()
	print(m)
	print(m * ~m)
	import matplotlib.pyplot as plt
	fix, ax = plt.subplots()

	for i in range(10):
		(gear>> (p*i/10).exp()).plot(ax=ax)
	# (gear<< (p*0.2).exp()).plot(ax=ax)
	plt.show()

