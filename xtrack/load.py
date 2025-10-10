# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal, Optional

import xtrack as xt


_SUPPORTED_FORMATS = {'json', 'madx', 'python', 'csv', 'hdf5', 'tfs'}


def _resolve_table_instance(table: xt.Table):
    table_class = getattr(table, '_data', {}).get('__class__')
    if not isinstance(table_class, str):
        return table

    cls = getattr(xt, table_class, None)
    if cls is None or cls is xt.Table:
        return table

    out = cls(data=table._data, col_names=table._col_names)

    return out

def _guess_format_from_path(path: str) -> Optional[str]:
    lower = path.lower()
    if lower.endswith(('.json', '.json.gz')):
        return 'json'
    if lower.endswith(('.seq', '.madx')):
        return 'madx'
    if lower.endswith('.py'):
        return 'python'
    if lower.endswith(('.h5', '.hdf5')):
        return 'hdf5'
    if lower.endswith('.csv'):
        return 'csv'
    if lower.endswith('.tfs'):
        return 'tfs'
    return None


def load(
        file=None,
        string=None,
        format: Literal['json', 'madx', 'python', 'csv', 'hdf5', 'tfs'] = None,
        timeout=5.0,
        reverse_lines=None,
):
    if isinstance(file, Path):
        file = str(file)

    if (file is None) == (string is None):
        raise ValueError('Must specify either file or string, but not both')

    if string is not None and format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f'Format must be specified to be one of {_SUPPORTED_FORMATS} when using string input'
        )

    if format is None and file is not None and isinstance(file, str):
        format = _guess_format_from_path(file)

    if format is None:
        raise ValueError('format could not be determined, please specify it explicitly')

    if reverse_lines and format != 'madx':
        raise ValueError('`reverse_lines` is only supported for madx input.')

    if file and isinstance(file, str) and (file.startswith('http://') or file.startswith('https://')):
        binary = format == 'hdf5'
        string = xt.general.read_url(file, timeout=timeout, binary=binary)
        file = None

    if format == 'json':
        payload = xt.json.load(file=file, string=string)
        cls_name = payload.pop('__class__', None)
        if cls_name is not None:
            cls = getattr(xt, cls_name, None)
            if cls is None:
                raise ValueError(f'Unknown class {cls_name!r} in json data')
            return cls.from_dict(payload)
        if 'lines' in payload:
            return xt.Environment.from_dict(payload)
        if 'element_names' in payload or 'line' in payload:
            if 'line' in payload:
                payload = payload['line']
            return xt.Line.from_dict(payload)
        raise ValueError('Cannot determine class from json data')

    if format == 'madx':
        return xt.load_madx_lattice(file=file, string=string, reverse_lines=reverse_lines)

    if format == 'python':
        if string is not None:
            raise NotImplementedError('Loading from string not implemented for python format')
        env = xt.Environment()
        env.call(file)
        return env

    if format == 'csv':
        if string is not None:
            text = string.decode() if isinstance(string, bytes) else string
            buffer = io.StringIO(text)
            base_table = xt.Table.from_csv(buffer)
        else:
            if hasattr(file, 'seek'):
                file.seek(0)
            base_table = xt.Table.from_csv(file)
        return _resolve_table_instance(base_table)

    if format == 'hdf5':
        if string is not None:
            if not isinstance(string, (bytes, bytearray)):
                raise TypeError('HDF5 string input must be bytes-like')
            buffer = io.BytesIO(string)
            base_table = xt.Table.from_hdf5(buffer)
        else:
            if hasattr(file, 'seek'):
                file.seek(0)
            base_table = xt.Table.from_hdf5(file)
        return _resolve_table_instance(base_table)

    if format == 'tfs':
        if string is not None:
            text = string.decode() if isinstance(string, bytes) else string
            buffer = io.StringIO(text)
            base_table = xt.Table.from_tfs(buffer)
        else:
            if hasattr(file, 'seek'):
                file.seek(0)
            base_table = xt.Table.from_tfs(file)
        return _resolve_table_instance(base_table)

    raise ValueError(f'Unsupported format {format!r}')
