import json
import io
from pathlib import Path
import gzip

import numpy as np

from xobjects import JEncoder


_COMPLEX_SCALAR_TAG = "__complex__"
_COMPLEX_ARRAY_TAG = "__complex_ndarray__"


class _XtrackJSONEncoder(JEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {_COMPLEX_SCALAR_TAG: [obj.real, obj.imag]}

        if isinstance(obj, np.ndarray) and np.issubdtype(
            obj.dtype, np.complexfloating
        ):
            return {
                _COMPLEX_ARRAY_TAG: {
                    "dtype": str(obj.dtype),
                    "shape": obj.shape,
                    "real": obj.real.tolist(),
                    "imag": obj.imag.tolist(),
                }
            }

        return super().default(obj)


def _complex_object_hook(obj):
    if _COMPLEX_SCALAR_TAG in obj:
        real, imag = obj[_COMPLEX_SCALAR_TAG]
        return complex(real, imag)

    if _COMPLEX_ARRAY_TAG in obj:
        metadata = obj[_COMPLEX_ARRAY_TAG]
        real = np.array(metadata["real"], dtype=float)
        imag = np.array(metadata["imag"], dtype=float)
        data = real + 1j * imag
        dtype = metadata.get("dtype")
        if dtype is not None:
            data = data.astype(np.dtype(dtype))
        shape = metadata.get("shape")
        if shape is not None:
            data = np.array(data).reshape(tuple(shape))
        return data

    return obj


def dump(data, file, indent=1):
    if isinstance(file, io.IOBase):
        fh, close = file, False
    elif (isinstance(file, str) and file.endswith(".gz")) or (
        isinstance(file, Path) and file.suffix == ".gz"
    ):
        fh, close = gzip.open(file, "wt"), True
    else:
        fh, close = open(file, "w"), True

    json.dump(data, fh, indent=indent, cls=_XtrackJSONEncoder)

    if close:
        fh.close()


def load(file=None, string=None):

    if string is not None:
        assert file is None, "Cannot specify both file and string"
        data = json.loads(string, object_hook=_complex_object_hook)
        return data

    if file is None:
        raise ValueError("Must specify either file or string")

    if isinstance(file, io.IOBase):
        fh, close = file, False
    elif (isinstance(file, str) and file.endswith(".gz")) or (
        isinstance(file, Path) and file.suffix == ".gz"
    ):
        fh, close = gzip.open(file, "rt"), True
    else:
        fh, close = open(file, "r"), True

    data = json.load(fh, object_hook=_complex_object_hook)

    if close:
        fh.close()

    return data
