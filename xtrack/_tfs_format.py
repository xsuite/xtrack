# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Versioned TFS rendering, independent of the legacy table serializer.

The defaults and rendering rules below belong to specification version "1"
and the explicitly listed v1 preset identifiers. Changing them requires a new
version, not just a change to the parser.
"""

from collections.abc import Mapping
import json
import math
import numbers
import os
from pathlib import Path
import re
import tomllib

import numpy as np


_UNSET = object()
_V1_VERSIONS = frozenset({
    '1', 'optics_orbit_v1', 'optics_orbit_strengths_v1', 'madx_twiss_v1',
})
_TOKENS = {'string': '%s', 'float': '%le', 'int': '%d',
           'bool': '%b', 'complex': '%lz'}
_COMPATIBLE_TYPES = {
    'string': {'string'}, 'bool': {'bool'}, 'int': {'int'},
    'float': {'int', 'float'}, 'complex': {'int', 'float', 'complex'},
}
_FLOAT_FORMAT = re.compile(r'[+ -]?#?0?[0-9]*(?:\.[0-9]+)?[eEfFgG]\Z')
_INT_FORMAT = re.compile(r'[+ -]?0?[0-9]*d\Z')


def _unique_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate format specification key: {key!r}')
        result[key] = value
    return result


def _check_keys(value, allowed, required, where):
    if not isinstance(value, Mapping):
        raise TypeError(f'{where} must be a mapping')
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f'Unknown key(s) in {where}: {sorted(unknown, key=str)}')
    missing = required - value.keys()
    if missing:
        raise ValueError(f'Missing key(s) in {where}: {sorted(missing)}')


def _width(value, where):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f'{where} must be a non-negative integer')
    if value < 0:
        raise ValueError(f'{where} must be a non-negative integer')
    return int(value)


def _name(value, where, output=False):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{where} must be a non-empty string')
    if output and (any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value)
                   or any(ch in value for ch in '\"\'\\')):
        raise ValueError(f'{where} cannot contain whitespace, quotes or backslashes')
    return value


def _case(value, case):
    return value if case == 'preserve' else getattr(value, case)()


def load_format_spec(spec):
    """Load, validate and copy a mapping or a JSON/TOML specification file."""
    if isinstance(spec, (str, os.PathLike)):
        path = Path(spec)
        if path.suffix.lower() == '.json':
            with path.open(encoding='utf-8') as fh:
                spec = json.load(fh, object_pairs_hook=_unique_json_keys)
        elif path.suffix.lower() == '.toml':
            with path.open('rb') as fh:
                spec = tomllib.load(fh)
        else:
            raise ValueError('Format specification files must be .json or .toml')

    _check_keys(spec, {'version', 'case', 'float_format', 'columns', 'scalars',
                       'default_column_width', 'numeric_column_width'},
                {'version', 'columns', 'scalars'}, 'format_spec')
    if not isinstance(spec['version'], str):
        raise TypeError('format_spec.version must be a string')
    if spec['version'] not in _V1_VERSIONS:
        raise ValueError(f'Unsupported TFS format specification version: {spec["version"]!r}')
    case = spec.get('case', 'preserve')
    if case not in ('preserve', 'upper', 'lower'):
        raise ValueError('format_spec.case must be "preserve", "upper" or "lower"')
    float_format = spec.get('float_format', '.16e')
    if not isinstance(float_format, str) or not _FLOAT_FORMAT.fullmatch(float_format):
        raise ValueError('float_format must be a Python e/E/f/F/g/G format, e.g. ".16e"')
    result = {
        'version': spec['version'], 'case': case, 'float_format': float_format,
        'default_column_width': _width(spec.get('default_column_width', 0),
                                       'default_column_width'),
        'numeric_column_width': _width(spec.get('numeric_column_width', 16),
                                       'numeric_column_width'),
    }
    for section in ('columns', 'scalars'):
        fields = spec[section]
        if not isinstance(fields, (list, tuple)):
            raise TypeError(f'format_spec.{section} must be an ordered list')
        result[section] = []
        names = set()
        allowed = {'name', 'source_name', 'type', 'format'}
        if section == 'columns':
            allowed.add('width')
        else:
            allowed.add('value')
        for index, field in enumerate(fields):
            where = f'format_spec.{section}[{index}]'
            _check_keys(field, allowed, {'name'}, where)
            field = dict(field)
            name = _name(field['name'], f'{where}.name', output=True)
            if 'value' in field:
                if 'source_name' in field:
                    raise ValueError(f'{where} cannot specify both value and source_name')
                _value_type(field['value'])
            else:
                field['source_name'] = _name(field.get('source_name', name),
                                             f'{where}.source_name')
            output_name = _case(name, case)
            if output_name in names:
                raise ValueError(f'Duplicate output name in {section}: {output_name!r}')
            names.add(output_name)
            if 'type' in field:
                if not isinstance(field['type'], str) or field['type'] not in _TOKENS:
                    raise ValueError(f'{where}.type must be one of {tuple(_TOKENS)}')
            if 'format' in field:
                fmt = field['format']
                if not isinstance(fmt, str) or not (
                        _FLOAT_FORMAT.fullmatch(fmt) or _INT_FORMAT.fullmatch(fmt)):
                    raise ValueError(f'Invalid numeric format in {where}: {fmt!r}')
                if 'type' in field:
                    _check_format_type(fmt, field['type'], where)
                elif 'value' in field:
                    _check_format_type(fmt, _value_type(field['value']), where)
            if 'width' in field:
                field['width'] = _width(field['width'], f'{where}.width')
            result[section].append(field)
    return result


def _check_format_type(fmt, kind, where):
    pattern = (_FLOAT_FORMAT if kind in ('float', 'complex') else
               _INT_FORMAT if kind == 'int' else None)
    if pattern is None or not pattern.fullmatch(fmt):
        raise ValueError(f'Format {fmt!r} is incompatible with {kind} in {where}')


def _value_type(value):
    if isinstance(value, (str, np.str_)):
        return 'string'
    if isinstance(value, (bool, np.bool_)):
        return 'bool'
    if isinstance(value, numbers.Integral):
        return 'int'
    if isinstance(value, numbers.Real):
        return 'float'
    if isinstance(value, numbers.Complex):
        return 'complex'
    raise TypeError(f'Unsupported TFS value of type {type(value).__name__}')


def _infer_type(values, where):
    dtype_kind = values.dtype.kind
    if dtype_kind in 'biufcU':
        return {'b': 'bool', 'i': 'int', 'u': 'int', 'f': 'float',
                'c': 'complex', 'U': 'string'}[dtype_kind]
    kinds = {_value_type(value) for value in values}
    if len(kinds) == 1:
        return kinds.pop()
    if kinds and kinds <= {'int', 'float', 'complex'}:
        return 'complex' if 'complex' in kinds else 'float'
    raise TypeError(f'Cannot infer a single TFS type for {where}; specify type explicitly')


def _format_float(value, fmt):
    value = float(value)
    # Fixed spellings independent of case, precision and locale.
    if math.isnan(value):
        return 'nan'
    if math.isinf(value):
        return '-inf' if value < 0 else 'inf'
    return format(value, fmt)


def _format_value(value, kind, fmt, case):
    actual = _value_type(value)
    if actual not in _COMPATIBLE_TYPES[kind]:
        raise TypeError(f'Expected {kind}, got {actual}')
    if kind == 'string':
        value = _case(str(value), case)
        # A deliberately small, portable quoting convention. Do not emit
        # escapes that TFS readers interpret differently in headers and cells.
        if any(ch in value for ch in '\"\\') or any(
                ord(ch) < 32 or ord(ch) == 127 or ch in '\u2028\u2029' for ch in value):
            raise ValueError('TFS strings cannot contain double quotes, backslashes or control characters')
        return f'"{value}"'
    if kind == 'bool':
        return '1' if value else '0'
    if kind == 'int':
        return format(int(value), fmt or 'd')
    if kind == 'float':
        return _format_float(value, fmt)
    value = complex(value)
    sign = '-' if math.copysign(1, value.imag) < 0 else '+'
    # Strip width padding inside complex tokens; column padding is separate.
    real = _format_float(value.real, fmt).strip()
    imag = _format_float(abs(value.imag), fmt).strip().lstrip('+')
    return f'{real}{sign}{imag}i'


def _prepare_field(table, field, section, spec):
    where = f'{section} field {field["name"]!r}'
    if 'value' in field:
        value = field['value']
        is_column = False
    else:
        source = field['source_name']
        where += f' (source {source!r})'
        if source not in table._data:
            raise KeyError(f'Missing {where}')
        is_column = source in table._col_names
        if is_column != (section == 'columns'):
            raise ValueError(f'{where} is a {"column" if is_column else "scalar"}, '
                             f'not a {"column" if section == "columns" else "scalar"}')
        value = table._data[source]
    if is_column:
        values = np.asarray(value)
        if values.ndim != 1:
            raise ValueError(f'{where} must be one-dimensional')
        kind = field.get('type') or _infer_type(values, where)
    else:
        # Keep the original scalar type instead of coercing it through NumPy.
        values = [value]
        kind = field.get('type') or _value_type(value)
    fmt = field.get('format')
    if fmt is not None:
        _check_format_type(fmt, kind, where)
    elif kind in ('float', 'complex'):
        fmt = spec['float_format']
    cells = []
    for index, value in enumerate(values):
        try:
            cells.append(_format_value(value, kind, fmt, spec['case']))
        except (ValueError, TypeError, OverflowError) as exc:
            raise type(exc)(f'{where}, value {index}: {exc}') from exc
    return _case(field['name'], spec['case']), _TOKENS[kind], cells


def write_tfs(table, file, format_spec):
    """Validate and render all data before touching the output destination."""
    spec = load_format_spec(format_spec)
    headers = [_prepare_field(table, field, 'scalars', spec)
               for field in spec['scalars']]
    columns = [_prepare_field(table, field, 'columns', spec)
               for field in spec['columns']]
    row_count = len(columns[0][2]) if columns else 0
    if any(len(cells) != row_count for _, _, cells in columns):
        raise ValueError('All selected column arrays must have the same length')

    lines = []
    name_width = max((len(name) for name, _, _ in headers), default=0)
    token_width = max((len(token) for _, token, _ in headers), default=0)
    for name, token, cells in headers:
        lines.append(f'@ {name:<{name_width}} {token:<{token_width}} {cells[0]}')

    widths = []
    for field, (name, token, cells) in zip(spec['columns'], columns):
        widths.append(max(
            len(name), len(token), max(map(len, cells), default=0),
            field.get('width', 0), spec['default_column_width'],
            spec['numeric_column_width'] if token != '%s' else 0))

    def align(values):
        return ' '.join(value.ljust(width) if token == '%s' else value.rjust(width)
                        for value, width, (_, token, _) in zip(values, widths, columns)).rstrip()

    if columns:
        lines.append('* ' + align([name for name, _, _ in columns]))
        lines.append('$ ' + align([token for _, token, _ in columns]))
        for row in zip(*(cells for _, _, cells in columns)):
            lines.append(' ' + align(row))
    content = ''.join(line + '\n' for line in lines)
    if isinstance(file, (str, os.PathLike)):
        with open(file, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(content)
    else:
        file.write(content)
