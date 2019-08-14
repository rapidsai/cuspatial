# Copyright (c) 2018, NVIDIA CORPORATION.

import os
import sysconfig
from distutils.sysconfig import get_python_lib

import numpy as np
import versioneer
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

install_requires = ["numba", "cython"]
cython_files = ["cuspatial/bindings/**/*.pyx"]

extensions = [
    Extension(
        "*",
        sources=cython_files,
        include_dirs=[
            "../../cpp/include/cuspatial",
            os.environ['CONDA_PREFIX']+"/include/cudf",
            os.path.dirname(sysconfig.get_path("include")),
            np.get_include()
        ],
        library_dirs=[get_python_lib()],
        libraries=["cudf","cuspatial"],
        language="c++",
        extra_compile_args=["-std=c++14"]
    )
]

setup(
    name="cuspatial",
    version=versioneer.get_version(),
    description="cuSpstial - GPU-Accelerated Spatial and Trajectory Data Management and Analytics Library",
    url="https://gitlab-master.nvidia.com/jiantingz/cuspatial",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    # Include the separately-compiled shared library
    setup_requires=["cython"],
    ext_modules=cythonize(extensions),
    packages=find_packages(include=["cuspatial", "cuspatial.*"]),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    zip_safe=False,
)
