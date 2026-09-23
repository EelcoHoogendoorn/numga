"""

"""

import numpy as np

def setup_rays(n=200):
	"""Set up parralel rays impinging on the unit 2-sphere"""
	ones = np.ones((n, n))
	x = np.linspace(-1, +1, n)[:, None] * ones
	y = np.linspace(-1, +1, n)[None, :] * ones
	z = np.sqrt(1 - x**2-y**2)
	mask = np.isnan(z)
	z = np.nan_to_num(z, nan=0)
	p = np.array([x, y, z])
	p = np.moveaxis(p, 0, 2)
	p = np.array(p)
	mask = np.array(mask)
	return p, mask


def render(rays, planes, scale=512):
	"""render planes intersecting surface of a sphere with AA"""
	v = rays[:,:, None].inner(planes).values
	q = np.prod(np.tanh(v*scale), axis=(-1, -2))
	return (q + 1) / 2

def image_downsample(img, bin_size = 2):
	input_size = img.shape[0]
	output_size = input_size // bin_size
	return img.reshape((output_size, bin_size, output_size, bin_size)).mean((1, 3))
