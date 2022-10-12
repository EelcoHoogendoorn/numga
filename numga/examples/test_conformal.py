"""Explore working with CGA in numga"""

import numpy as np

from numga.backend.numpy.context import NumpyContext
from numga.examples.conformal import Conformal, Conformalize


def plot_point(plt, cga, p):
	assert p.subspace.inside.vector()
	plt.scatter(*cga.point_position(cga.normalize(p)).values.T)


def plot_circle(plt, cga, circle):
	assert circle.subspace.inside.antivector()
	circle = cga.normalize(circle.dual())
	# this is pretty elegant; center is point at infinity reflected in the circle
	center = cga.normalize(circle >> cga.ni)
	radius = circle.norm()

	ang = np.linspace(0, 2 * np.pi, 100, endpoint=True)
	cx, cy = np.cos(ang), np.sin(ang)
	r = radius.values
	px, py = cga.point_position(center).values.T
	if isinstance(px, float):
		plt.plot(cx * r + px, cy * r + py)
	else:
		for xx, yy, rr in zip(px, py, r):
			plt.plot(cx * rr + xx, cy * rr + yy)


def test_circle_reflect():
	# construct 2d cga
	cga = Conformalize(NumpyContext('x+y+'))

	# construct a point grid
	points = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
	P = cga.embed_point(np.array(points).T.reshape(-1, 2), r=0.1)
	# construct a circle
	# C = cga.embed_point((-1.5, -1.5), r=2).dual()
	C = cga.embed_point((0.1, 0.2), r=2).dual()

	Q = C >> P

	import matplotlib.pyplot as plt
	plot_circle(plt, cga, P.dual())
	plot_circle(plt, cga, Q.dual())
	plot_circle(plt, cga, C)
	# plot_point(plt, cga, cga.project(Q, C.dual()))
	plt.axis('equal')
	plt.show()


def test_circle_fit():
	# construct 2d cga
	cga = Conformal(NumpyContext(Conformal.algebra('x+y+')))

	assert np.allclose(cga.ni.norm().values, 0)
	assert np.allclose(cga.no.norm().values, 0)
	assert np.allclose(((cga.N * cga.N)).values, 1)

	# create some points
	a, b, c = cga.embed_point((-1.5, -0.5)), cga.embed_point((1, -0.5)), cga.embed_point((0, 1.5))

	assert np.allclose(b.inner(cga.ni).values, -1)
	assert np.allclose((b*b).values, 0)
	assert np.allclose(cga.point_position(b).values, (1, -0.5))

	# construct a circle as the wedge of three points
	C = a ^ b ^ c
	# direct construction of a circle
	D = cga.embed_point((-1, -1), r=1).dual()
	# get intersection set of two circles
	T = C & D
	# split them out into two points
	l, r = cga.split_points(T)

	print(D.norm_squared())


	import matplotlib.pyplot as plt

	plot_circle(plt, cga, C)
	plot_circle(plt, cga, D)

	plot_point(plt, cga, a)
	plot_point(plt, cga, b)
	plot_point(plt, cga, c)

	plot_point(plt, cga, l)
	plot_point(plt, cga, r)

	plt.axis('equal')
	plt.show()
