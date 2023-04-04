# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xobjects
import xpart
import xdeps
import xtrack
import xfields
import xmask

print("\n-------- Paths: --------")

print('xobjects: ', xobjects.__path__)
print('xpart:    ', xpart.__path__)
print('xdeps:    ', xdeps.__path__)
print('xtrack:   ', xtrack.__path__)
print('xfields:  ', xfields.__path__)
print('xmask     ', xmask.__path__)


try:
    from git import Repo
    from pathlib import Path

    print("\n-------- Git branches: --------")
    repo_xobjects = Repo(Path(xobjects.__path__[0]).parent)
    repo_xpart = Repo(Path(xpart.__path__[0]).parent)
    repo_xdeps = Repo(Path(xdeps.__path__[0]).parent)
    repo_xtrack = Repo(Path(xtrack.__path__[0]).parent)
    repo_xfields = Repo(Path(xfields.__path__[0]).parent)
    repo_xmask = Repo(Path(xmask.__path__[0]).parent)

    print('xobjects is on branch: ', repo_xobjects.active_branch)
    print('xpart is on branch:    ', repo_xpart.active_branch)
    print('xdeps is on branch:    ', repo_xdeps.active_branch)
    print('xtrack is on branch:   ', repo_xtrack.active_branch)
    print('xfields is on branch:  ', repo_xfields.active_branch)
    print('xmask is on branch:    ', repo_xmask.active_branch)

except Exception as err:
    print('No git info because of the following exception: ', err)

print("\n-------- Enabled test contexts: --------")
for cc in xobjects.context.get_test_contexts():
        print(cc)
