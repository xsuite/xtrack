import json
import io
from pathlib import Path
import gzip
from xobjects import JEncoder


def dump(data, file, indent=1):
    if isinstance(file, io.IOBase):
        fh, close = file, False
    elif (isinstance(file, str) and file.endswith(".gz")) or (
        isinstance(file, Path) and file.suffix == ".gz"
    ):
        fh, close = gzip.open(file, "wt"), True
    else:
        fh, close = open(file, "w"), True

    json.dump(data, fh, indent=indent, cls=JEncoder)

    if close:
        fh.close()


def load(file):
    if isinstance(file, io.IOBase):
        fh, close = file, False
    elif (isinstance(file, str) and file.endswith(".gz")) or (
        isinstance(file, Path) and file.suffix == ".gz"
    ):
        fh, close = gzip.open(file, "rt"), True
    else:
        fh, close = open(file, "r"), True

    data = json.load(fh)

    if close:
        fh.close()

    return data
