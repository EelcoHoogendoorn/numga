"""Simple 2d pga utility function examples
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
def exponentiate_bivector(b: "BiVector") -> "Motor":
	"""optimized 2d pga bivector exponential"""
	av = b.norm().values[..., 0]
	cv = pga.cos(av)
	sv = pga.sinc(av/np.pi)
	return b * sv + cv



def as_matrix(motor):
	return motor.select.motor().sandwich(pga.subspace.antivector()).kernel
# # god this is so janky... need to add custom subspace ordering to numga to get rid of these signs
signs = 1 - (np.arange(9).reshape(3,3)%2)*2
assert pga.subspace.antivector().named_str == 'wx,wy,xy'
def transform_points_optimized(motor, p):
	"""Optimized sandwich implementation for point transformation,
	encoded as [n,2] arrays without homogenous coord, eliminating intermediaries"""
	m = (as_matrix(motor) * signs)[::-1,::-1]
	return p.dot(m[1:, 1:]) + m[0:1, 1:]
def transform_points(motor, p):
	q = x * p[..., 0] + y * p[..., 1] + w
	q = (motor >> q.dual()).dual()
	return q.values[..., 1:]




class PGA2d:

	def __init__(self, ctx):
		self.ctx = ctx
		pass



def test_pga():
	# create polygon
	a = np.linspace(0, np.pi*2, 7, endpoint=True)
	poly = np.array([np.cos(a), np.sin(a)]).T
	# create random point
	p = point(-10, 1)
	# rotate around this point in a number of steps
	motors = (p * np.linspace(0, 1, 10)).exp()

	import matplotlib.pyplot as plt
	fix, ax = plt.subplots()
	for m in motors:
		ax.plot(*transform_points_optimized(m, poly).T)
	plt.axis('equal')
	plt.show()
