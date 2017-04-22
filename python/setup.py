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

this_dir = pathlib.Path(__file__).resolve().parent

cmake_rootdir = this_dir.parent
print(cmake_rootdir.as_posix())
cmake_builddir = cmake_rootdir.joinpath('build')
print(cmake_builddir.as_posix())
cmake_libobj= cmake_builddir.joinpath('lib','librmwarp.a')
print(cmake_libobj.as_posix())
assert cmake_libobj.exists()

ext_srcdir = this_dir.joinpath('rmwarp')
ext_srcs   = list(ext_srcdir.glob('*.pyx'))
ext_incdir = cmake_rootdir.joinpath('rmwarp')
print(ext_incdir.as_posix())
ext_modules = cythonize([Extension(
    src.relative_to(this_dir).with_suffix('').as_posix().replace('/','.')
  , [src.as_posix()]
  , language='c++'
  , include_dirs=[ext_incdir.as_posix(),cmake_rootdir.as_posix()]
  , libraries=['rmwarp']
  , extra_compile_args=['-std=gnu++14','-g','-ggdb','-march=native']
  , library_dirs=[cmake_libobj.parent.as_posix()]
  ) for src in ext_srcs]
  , compiler_directives={
     "embedsignature":True
    ,"always_allow_kwords":True
    ,"cdivision_warnings":True
    ,"cdivision":True
    ,"infer_types":True
    ,"boundscheck":False
    ,"overflowcheck":False
    ,"wraparound":True}
  , nthreads=6)

include_dirs = cmake_rootdir.joinpath('rmwarp')

extra_objects = [cmake_libobj.as_posix()]
setup(
    name='rmwarp',
    version='0.0.0',
    author="gabriel d. karpman",
    license='MIT',
    description='cython binding for librmwarp reassignment-method time stretch',
    classifiers=['Programming Language :: Python :: 2.7',
                'Operating System :: MacOS :: MacOS X',
                'Operating System :: Microsoft :: Windows',
                'Operating System :: POSIX :: BSD :: FreeBSD',
                'Operating System :: POSIX :: Linux',
                'Intended Audience :: Developers'],
    cmdclass = { 'build_ext':build_ext},
    packages=find_packages(exclude=['build*']),
    ext_modules = ext_modules,
    zip_safe=False
)
