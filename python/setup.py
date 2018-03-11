#!/usr/bin/env python

from __future__ import print_function, division

from distutils.ccompiler import new_compiler as _new_compiler, LinkError, CompileError
from distutils.command.clean import clean, log
from distutils.core import Command
from distutils.errors import DistutilsExecError
from setuptools import setup, find_packages, Extension, Distribution
from subprocess import Popen, PIPE, call, check_output
from Cython.Distutils import build_ext
from Cython.Build import cythonize

#Distribution(dict(setup_requires='pythran'))

import sys, os, pathlib

from os.path import join, exists,dirname
from os import environ

this_dir = pathlib.Path(__file__).resolve().parent

cmake_rootdir = this_dir.parent
print(cmake_rootdir.as_posix())
cmake_builddir = cmake_rootdir.joinpath('build')
print(cmake_builddir.as_posix())
cmake_libobj= cmake_builddir.joinpath('lib','librmwarp.so')
print(cmake_libobj.as_posix())
assert(cmake_libobj.exists())

ext_srcdir = this_dir.joinpath('rmwarp')
ext_srcs   = list(ext_srcdir.glob('*.pyx'))
ext_incdir = cmake_rootdir.joinpath('rmwarp')
print(ext_incdir.as_posix())

def get_pythran_config():
    try:
        proc = Popen(['pythran-config','--cflags','--libs'],stdout=PIPE,stderr=PIPE)
    except OSError:
        print('pkg-config is required!')
        exit(1)
    raw_config, err = proc.communicate()
    if proc.wait():
        return
    config = dict()
    for chunk in raw_config.decode('utf8').strip().split():
        if chunk.startswith('-I'):
            config.setdefault('include_dirs',[]).append(chunk[2:])
        elif chunk.startswith('-L'):
            config.setdefault('library_dirs',[]).append(chunk[2:])
        elif chunk.startswith('-l'):
            lib = chunk[2:]
            if not 'python3.6' in lib:
                config.setdefault('libraries',[]).append(lib)
        elif chunk.startswith('-D'):
            name = chunk[2:].split('=')[0]
            config.setdefault('define_macros',[]).append((name,None))
    return config

config = dict()#get_pythran_config()
config.setdefault('libraries',[]).extend(['rmwarp','-pthread'])
config.setdefault('include_dirs',[]).extend([ext_incdir.as_posix(),cmake_rootdir.as_posix()])
config.setdefault('extra_compile_args',[]).extend(['-std=gnu++14','-shared','-O3','-Ofast','-g','-ggdb'])
config.setdefault('library_dirs',[]).extend([cmake_libobj.parent.as_posix()])
config['language'] = 'c++'
ext_modules = cythonize([Extension(
    src.relative_to(this_dir).with_suffix('').as_posix().replace('/','.')
  , [src.as_posix()]
#  , language='c++'
  , **config
#  , include_dirs=[ext_incdir.as_posix(),cmake_rootdir.as_posix()]
#  , libraries=['rmwarp']
#  , extra_compile_args=['-std=gnu++14']
#  , define_macros = [('NPY_NO_DEPRECATED_API','NPY_1_4_API_VERSION')]
#  , library_dirs=[cmake_libobj.parent.as_posix()]
  ) for src in ext_srcs]
  , compiler_directives={
     "embedsignature":True
    ,"cdivision_warnings":False
    ,"cdivision":True
    ,'always_allow_keywords':True
    ,'linetrace':True
    ,"language_level":3
    ,"infer_types":True
    ,"boundscheck":True
    ,"overflowcheck":False
    ,"wraparound":True
#    ,'np_pythran':True
        }
  , nthreads=6)

#for ext in ext_modules:
#    for k,v in get_pythran_config().items():
#        a = getattr(ext,k, [])
#        setattr(ext,k,v + a)

include_dirs = cmake_rootdir.joinpath('rmwarp')

extra_objects = [cmake_libobj.as_posix()]
setup(
    name='rmwarp',
    version='0.0.1',
    author="gabriel d. karpman",
    license='MIT',
    description='cython binding for librmwarp reassignment-method time stretch',
    classifiers=['Programming Language :: Python :: 2.7',
                'Operating System :: MacOS :: MacOS X',
                'Operating System :: Microsoft :: Windows',
                'Operating System :: POSIX :: BSD :: FreeBSD',
                'Operating System :: POSIX :: Linux',
                'Intended Audience :: Developers'],
    setup_requires=[
        'cython>=0.x',
        'pythran',
    ],
    cmdclass = { 'build_ext':build_ext},
    packages=find_packages(exclude=['build*']),
    package_data = {
        'rmwarp':['*.pxd'],

    },
    ext_modules = ext_modules,
    zip_safe=False
)
