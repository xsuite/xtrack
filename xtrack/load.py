# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #
import xtrack as xt
from typing import Literal
from pathlib import Path


def load(
        file=None,
        string=None,
        format: Literal['json', 'madx', 'python'] = None,
        timeout=5.,
):
    if isinstance(file, Path):
        file = str(file)

    if (file is None) == (string is None):
        raise ValueError('Must specify either file or string, but not both')

    FORMATS = {'json', 'madx', 'python'}
    if string and format not in FORMATS:
        raise ValueError(f'Format must be specified to be one of {FORMATS} when '
                         f'using string input')

    if format is None and file is not None:
        if file.endswith('.json') or file.endswith('.json.gz'):
            format = 'json'
        elif file.endswith('.seq') or file.endswith('.madx'):
            format = 'madx'
        elif file.endswith('.py'):
            format = 'python'

    if file and (file.startswith('http://') or file.startswith('https://')):
        string = xt.general.read_url(file, timeout=timeout)
        file = None

    if format == 'json':
        ddd = xt.json.load(file=file, string=string)
        if '__class__' in ddd:
            cls_name = ddd.pop('__class__')
            cls = getattr(xt, cls_name)
            return cls.from_dict(ddd)
        elif 'lines' in ddd: # is environment
            return xt.Environment.from_dict(ddd)
        elif 'element_names' in ddd:
            return xt.Line.from_dict(ddd)
        else:
            raise ValueError('Cannot determine class from json data')
    elif format == 'madx':
        return xt.load_madx_lattice(file=file, string=string)
    elif format == 'python':
        if string is not None:
            raise NotImplementedError('Loading from string not implemented for python format')
        env = xt.Environment()
        env.call(file)
        return env
