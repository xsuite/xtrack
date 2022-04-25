import xobjects
import xpart
import xdeps
import xtrack
import xfields

print("\n-------- Paths: --------")

print('xobjects:\t', xobjects.__path__)
print('xpart:\t', xpart.__path__)
print('xdeps:\t', xdeps.__path__)
print('xtrack\t: ', xtrack.__path__)
print('xfields\t: ', xfields.__path__)


try:
    from git import Repo
    from pathlib import Path

    print("\n-------- Git branches: --------")
    repo_xobjects = Repo(Path(xobjects.__path__[0]).parent)
    repo_xpart = Repo(Path(xpart.__path__[0]).parent)
    repo_xdeps = Repo(Path(xdeps.__path__[0]).parent)
    repo_xtrack = Repo(Path(xtrack.__path__[0]).parent)
    repo_xfields = Repo(Path(xfields.__path__[0]).parent)

    print('Xobjects is on branch: ', repo_xobjects.active_branch)
    print('Xpart is on branch: ', repo_xpart.active_branch)
    print('Xdeps is on branch: ', repo_xdeps.active_branch)
    print('Xtrack is on branch: ', repo_xtrack.active_branch)
    print('Xfields is on branch: ', repo_xfields.active_branch)

except Exception as err:
    print('No git info because of the following exception: ', err)
