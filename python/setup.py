#!/usr/bin/env python

from __future__ import print_function, division

from distutils.ccompiler import new_compiler as _new_compiler, LinkError, CompileError
from distutils.command.clean import clean, log
from distutils.core import Command
from distutils.errors import DistutilsExecError
from distutils.msvccompiler import MSVCCompiler
from setuptools import setup, find_packages, Extension, Distribution
from subprocess import Popen, PIPE, call, check_output
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import sys, os, pathlib

from os.path import join, exists,dirname
from os import environ


cmake_builddir = pathlib.Path('.').resolve().parent
print(cmake_builddir.as_posix())
cmake_libobj= cmake_builddir.joinpath('build').joinpath('src').joinpath('librmwarp.a')
print(cmake_libobj.as_posix())
assert cmake_libobj.exists()

extra_objects = [cmake_libobj.as_posix()]
include_dirs  = [ pathlib.Path('.').resolve().parent.joinpath('include')]
setup(
    name='rmwrap',
    version='0.0.0',
    author="gabriel d. karpman",
    license='MIT',
    description='cython binding for librmwrap reassignment-method time stretch',
    classifiers=['Programming Language :: Python :: 2.7',
                'Operating System :: MacOS :: MacOS X',
                'Operating System :: Microsoft :: Windows',
                'Operating System :: POSIX :: BSD :: FreeBSD',
                'Operating System :: POSIX :: Linux',
                'Intended Audience :: Developers'],
    cmdclass = { 'build_ext':build_ext},
    packages=find_packages(exclude=['build*']),
    ext_modules = cythonize([Extension("_rmwarp", ["rmwarp/_rmwarp.pyx"],
        include_dirs=include_dirs
        extra_objects=extra_objects
        ,compiler_directives={
        "embedsignature":True,
        "always_allow_kwords":True,
        "cdivision_warnings":True,
        "cdivision":True,
        "infer_types":True,
        "boundscheck":False,
        "overflowcheck":False,
        "wraparound":False})
    ])
)
