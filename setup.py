# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #
from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension
from pathlib import Path

#######################################
# Prepare list of compiled extensions #
#######################################

extensions = [
    Extension(
        "xtrack.sequence.parser",
        sources=[
            "xtrack/sequence/parser.py",
            "xtrack/sequence/lexer.c",
            "xtrack/sequence/parser_tab.c",
        ],
        language="c",
    )
]

#########
# Setup #
#########

version_file = Path(__file__).parent / 'xtrack/_version.py'
dd = {}
with open(version_file.absolute(), 'r') as fp:
    exec(fp.read(), dd)
__version__ = dd['__version__']

setup(
    name='xtrack',
    version=__version__,
    description='Tracking library for particle accelerators',
    long_description='Tracking library for particle accelerators',
    url='https://xsuite.readthedocs.io/',
    author='G. Iadarola et al.',
    license='Apache 2.0',
    download_url="https://pypi.python.org/pypi/xtrack",
    project_urls={
            "Bug Tracker": "https://github.com/xsuite/xsuite/issues",
            "Documentation": 'https://xsuite.readthedocs.io/',
            "Source Code": "https://github.com/xsuite/xtrack",
        },
    packages=find_packages(),
    ext_modules=cythonize(extensions, gdb_debug=False),
    include_package_data=True,
    install_requires=[
        'numpy>=1.0',
        "pandas>=2.0",
        'scipy',
        'tqdm',
        'xobjects',
        'xdeps'
        ],
    extras_require={
        'tests': ['cpymad', 'nafflib', 'PyHEADTAIL', 'pytest', 'pytest-mock'],
        },
    )
