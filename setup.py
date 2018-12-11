import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("pyxhmm", ["src/hmm.pyx"], include_dirs=[np.get_include()]),
]
setup(
    name="My hello app",
    ext_modules=cythonize(extensions),
)
