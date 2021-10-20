######################################################
#
# PyRAI2MD 2 setup file for compling fssh cython library
#
# Author Jingbai Li
# Oct 11 2021
#
######################################################

from setuptools import setup
from Cython.Build import cythonize
import numpy as np
setup(
    ext_modules = cythonize("./PyRAI2MD/Dynamics/Propagators/fssh.pyx",compiler_directives={'language_level' : "3"}),
    include_dirs=[np.get_include()],
    package_dir={'cython_fssh': ''},

)
