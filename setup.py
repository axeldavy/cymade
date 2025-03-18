from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import os
import sys

# Define Cython extensions
extensions = [
    Extension(
        "cymade.atomic",
        ["cymade/atomic.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "cymade.threadpool",
        ["cymade/threadpool.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
]

setup(
    name="cymade",
    version="0.0.1",
    description="A Python package for cymade",
    author="Axel Davy",
    url="https://github.com/axeldavy/cymade",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data={
        "cymade": ["*.pxd"],
    },
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
        },
    ),
    install_requires=[
        "Cython >= 3.0",
    ],
    setup_requires=[
        "Cython >= 3.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,  # Cython modules are not zip safe
)