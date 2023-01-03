"""setup.py for NumGA.

Install for development:

  pip install -e .
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name="numga",
    version="0.0.1",
    description=("A geometric algebra library for JAX/numpy."),
    author="Eelco Hoogendoorn",
    author_email="hoogendoorn.eelco@gmail.com",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/eelcohoogendoorn/numga",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    # scripts=["bin/learn"],
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
    ],
    extras_require={
        "develop": ["pytest"],
    },
    classifiers=[
        "Development Status :: 4 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="geometric algebra GA jax numpy",
)
