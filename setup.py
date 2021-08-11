from setuptools import setup, find_packages, Extension

#######################################
# Prepare list of compiled extensions #
#######################################

extensions = []


#########
# Setup #
#########

setup(
    name='xtrack',
    version='0.1.0',
    description='Tracking library for particle accelerators',
    url='https://github.com/xsuite/xtrack',
    author='Riccard De Maria, Giovanni Iadarola',
    packages=find_packages(),
    ext_modules = extensions,
    install_requires=[
        'numpy>=1.0',
        'xobjects>=0.0.4',
        'xline'
        ]
    )
