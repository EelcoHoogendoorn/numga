import numpy as np

from numga.backend.numpy.context import NumpyContext
from numga.examples.conformal_elliptical import setup_rays, image_downsample, render


def test_rigid():
	cga = NumpyContext('x+y+z+')
	x,y,z = cga.multivector.basis()

	rays, mask = setup_rays(512)
	rays = cga.multivector.vector(rays)

	planes = np.array(np.meshgrid(*[[1, 0, -1]]*3)).reshape(3, -1).T[:13]
	planes = cga.multivector.vector(planes)
	planes = planes.normalized() #+ n

	render2 = lambda planes: image_downsample((render(rays, planes) + 1) * (1 - mask))
	b = -(x^z)+(y^z)
	# b = ((x + y).normalized()).dual() & ((- x - y).normalized()).dual()

	motors = [(b.normalized()*a).exp() for a in np.linspace(0, np.pi, 100)]

	from numga.examples.physics.render import write_animation_simulation
	write_animation_simulation(
		[m >> planes for m in motors],
		render2,
		'sphere_rigid.gif'
	)


def test_conformal():
	cga = NumpyContext('x+y+z+n-')
	x,y,z,n = cga.multivector.basis()

	sphere = cga.subspace.vector().difference(cga.subspace.n)
	def make_scalar(s):
		return cga.multivector.scalar(np.array(s)[:, None])
	def make_vector(v):
		return cga.multivector.multivector(values=v, subspace=sphere)
	def make_plane(s, d=0):
		return (s.normalized() + d*n)
	def make_point(s):
		"""norm-0 antivector"""
		return make_plane(s, 1).dual()


	# set up rays
	rays, mask = setup_rays(512)
	rays = make_vector(rays) + n

	# set up scene
	planes = np.array(np.meshgrid(*[[1, 0, -1]]*3)).reshape(3, -1).T[:13]	# all planes of the octahedral symmetry group
	# planes = planes[(planes!=0).sum(axis=1)==3]
	planes = make_vector(planes).normalized()


	render2 = lambda planes: image_downsample((render(rays, planes.normalized()) + 1) * (1 - mask))

	l, r = make_point(z + x + y), make_point(z - x - y)
	if True:
		filename = 'rotation'
		# l, r = make_point( + x + y), make_point( - x - y)
		b = l & r	# join of antivector points; rotation transform
		b = b.normalized()
	if False:
		filename = 'dipole'
		# bireflection in two infinitimal circles; dipole transform
		# note intersecting line lies outside the unit sphere
		b = (l.dual() ^ r.dual())
	if False:
		# note intersecting line lies outside the unit sphere, at infinity
		filename = 'zoom'
		l, r = make_plane(y, -0.5), make_plane(y, -0.6)	# concentric bireflection; zoom
		b = l ^ r


	print(b.norm())
	print(b.symmetric_reverse_product())
	print(b.dual().symmetric_reverse_product())
	print(b.squared())

	# return

	a = make_scalar(np.linspace(-np.pi/2, np.pi/2, 100, endpoint=False))
	motors = (b*a).exp()
	# print(motors.norm())

	if False:
		filename = 'mirror'
		v = make_vector([1, 1, 1])
		d = make_scalar(np.linspace(.999 , -.999, 100, endpoint=True))
		motors = make_plane(v, d)

	from numga.examples.physics.render import write_animation_simulation
	write_animation_simulation(
		# [m >> planes for m in motors],
		motors[:, None] >> planes,
		render2,
		# 'sphere_conformal_zoom_y_slow_q20.gif',
		f'sphere_conformal_{filename}.gif',
		fps=20,
		q=33,
	)
