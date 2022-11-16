# Copyright (c) 2018-2022, NVIDIA CORPORATION.
import versioneer
from setuptools import find_packages
from skbuild import setup

setup(
    name="cuspatial",
    version=versioneer.get_version(),
    description=(
        "cuSpatial: GPU-Accelerated Spatial and Trajectory Data Management and"
        " Analytics Library"
    ),
    url="https://github.com/rapidsai/cuspatial",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(include=["cuspatial", "cuspatial.*"]),
    package_data={"cuspatial._lib": ["*.pxd"]},
    cmdclass=versioneer.get_cmdclass(),
    install_requires=["numba"],
    zip_safe=False,
)
