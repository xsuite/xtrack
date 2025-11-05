# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import csv
import io
import json
import math
import numbers
import os
import shlex
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from xdeps import Table as _XdepsTable
import xtrack as xt

from . import json as json_utils

_PARTICLES_CLS: Optional[type] = None


def _get_particles_cls():
    global _PARTICLES_CLS
    try:
        import xtrack as xt  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency during import
        _PARTICLES_CLS = None
    else:
        _PARTICLES_CLS = getattr(xt, 'Particles', None)

    return _PARTICLES_CLS


def _prepare_header_source(source):
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        with open(path, 'r') as fh:
            content = fh.read()
        return content, source

    if isinstance(source, io.StringIO):
        return source.getvalue(), source

    if isinstance(source, io.IOBase):
        original_pos = None
        if source.seekable():
            original_pos = source.tell()
            source.seek(0)
            content = source.read()
            source.seek(original_pos)
        else:
            content = source.read()

        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')

        if source.seekable():
            return content, source

        return content, io.StringIO(content)

    if hasattr(source, 'read'):
        content = source.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
        return content, io.StringIO(content)

    raise TypeError(
        f"Unsupported TFS input type {type(source)!r} for header parsing"
    )


def _strip_outer_quotes(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _convert_header_value(type_token, raw_value):
    value = raw_value.strip()
    if value.lower() == 'null':
        return None

    token_letters = ''.join(ch for ch in type_token.lower() if ch.isalpha())

    if token_letters.endswith('b'):
        lower = value.lower()
        if lower in ('true', 't', 'yes', 'y', '1'):
            return True
        if lower in ('false', 'f', 'no', 'n', '0'):
            return False
        try:
            return bool(int(value))
        except ValueError:
            try:
                return bool(float(value))
            except ValueError:
                return value

    if token_letters.endswith(('d', 'i')):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return value

    if token_letters.endswith(('e', 'f', 'g')):
        try:
            return float(value)
        except ValueError:
            return value

    return _strip_outer_quotes(value)


def _parse_headers(text):
    headers = {}
    for line in text.splitlines():
        if not line.startswith('@'):
            continue
        stripped = line[1:].lstrip()
        if not stripped:
            continue
        parts = stripped.split(None, 2)
        if len(parts) != 3:
            continue
        name, type_token, raw_value = parts
        headers[name] = _convert_header_value(type_token, raw_value)
    return headers


class Table(_XdepsTable):
    """Extension of :class:`xdeps.Table` with export/import helpers."""

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_name_selection(names):
        if names is None:
            return None
        if isinstance(names, str):
            return {names}
        try:
            return set(names)
        except TypeError as exc:  # pragma: no cover - defensive programming
            raise TypeError(
                f"Selection must be iterable of names or a single string, got {names!r}"
            ) from exc

    @classmethod
    def _resolve_name_selection(cls, all_names: Iterable[str], *,
                                 include=None, exclude=None,
                                 missing='error', kind='item'):
        if missing not in ('error', 'ignore'):
            raise ValueError(
                f"Invalid missing policy {missing!r} (expected 'error' or 'ignore')"
            )

        include_set = cls._normalize_name_selection(include)
        exclude_set = cls._normalize_name_selection(exclude)
        available = set(all_names)

        if include_set is not None:
            missing_include = include_set - available
            if missing == 'error' and missing_include:
                raise KeyError(
                    f"Unknown {kind}(s) in include selection: {sorted(missing_include)}"
                )
            include_set &= available

        if exclude_set:
            missing_exclude = exclude_set - available
            if missing == 'error' and missing_exclude:
                raise KeyError(
                    f"Unknown {kind}(s) in exclude selection: {sorted(missing_exclude)}"
                )
            exclude_set &= available

        selected = []
        for name in all_names:
            if include_set is not None and name not in include_set:
                continue
            if exclude_set and name in exclude_set:
                continue
            selected.append(name)

        return selected

    # ------------------------------------------------------------------
    # Attribute (de-)serialisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize_attr_value(value):
        """Return a serialization-friendly version of an attribute value."""
        particles_cls = _get_particles_cls()
        if particles_cls is not None and isinstance(value, particles_cls):
            out = value.to_dict()
            out['__class__'] = 'Particles'
            return out
        return value

    @staticmethod
    def _deserialize_attr_value(value):
        """Rebuild runtime objects from serialized attribute data."""
        if not isinstance(value, dict):
            return value
        if value.get('__class__', None) != 'Particles':
            return value

        value.pop('__class__', None)
        particles_cls = _get_particles_cls()
        if particles_cls is None:
            return value

        return particles_cls.from_dict(value)

    # ------------------------------------------------------------------
    # Generic dictionary export/import
    # ------------------------------------------------------------------
    @classmethod
    def _split_include_exclude(cls, include, exclude, column_order,
                               attr_order, missing):
        """Split unified include/exclude selectors into column and attr sets."""

        column_names = set(column_order)
        attr_names = set(attr_order)
        all_names = column_names | attr_names

        include_set = cls._normalize_name_selection(include)
        if include_set is not None:
            unknown = include_set - all_names
            if unknown and missing == 'error':
                raise KeyError(
                    f"Unknown name(s) in include selection: {sorted(unknown)}"
                )
            include_set &= all_names

        exclude_set = cls._normalize_name_selection(exclude)
        if exclude_set:
            unknown = exclude_set - all_names
            if unknown and missing == 'error':
                raise KeyError(
                    f"Unknown name(s) in exclude selection: {sorted(unknown)}"
                )
            exclude_set &= all_names

        include_cols = include_set & column_names if include_set is not None else None
        include_attrs = include_set & attr_names if include_set is not None else None

        exclude_cols = exclude_set & column_names if exclude_set else None
        exclude_attrs = exclude_set & attr_names if exclude_set else None

        return include_cols, include_attrs, exclude_cols, exclude_attrs

    def _extra_metadata(self) -> Dict[str, Any]:
        """Return metadata fields always attached to serialized tables."""
        class_name = self.__class__.__name__
        return {
            '__class__': class_name,
            'xtrack_version': xt.__version__,
        }

    @classmethod
    def _strip_extra_metadata(cls, payload: Dict[str, Any]) -> None:
        """Remove helper metadata keys prior to table reconstruction."""
        payload.pop('__class__', None)
        payload.pop('xtrack_version', None)

    def to_dict(self, *, include=None, exclude=None,
                missing='error', include_meta=True):
        """Serialize the table to a dictionary, applying optional filters."""

        column_order = list(self._col_names)
        raw_attrs = {kk: vv for kk, vv in self._data.items() if kk not in column_order}
        raw_attrs.pop('_action', None)
        raw_attrs.pop('_col_names', None)
        attr_order = list(raw_attrs.keys())

        meta_filterable_keys = {
            'dropped_columns', 'dropped_attrs',
            '__class__', 'xtrack_version'
        }

        raw_include_set = self._normalize_name_selection(include)
        raw_exclude_set = self._normalize_name_selection(exclude)

        if raw_include_set is not None:
            include_meta = {name.lower() for name in raw_include_set
                            if name.lower() in meta_filterable_keys}
            include_for_split = {name for name in raw_include_set
                                 if name.lower() not in meta_filterable_keys}
            include_arg = include_for_split if include_for_split else None
            if not include_meta:
                include_meta = None
        else:
            include_meta = None
            include_arg = None

        if raw_exclude_set:
            exclude_meta = {name.lower() for name in raw_exclude_set
                            if name.lower() in meta_filterable_keys}
            exclude_for_split = {name for name in raw_exclude_set
                                 if name.lower() not in meta_filterable_keys}
            exclude_arg = exclude_for_split
        else:
            exclude_meta = set()
            exclude_arg = None

        include_cols, include_attrs, exclude_cols, exclude_attrs = (
            self._split_include_exclude(include_arg, exclude_arg,
                                        column_order, attr_order, missing)
        )

        selected_columns = self._resolve_name_selection(
            column_order, include=include_cols, exclude=exclude_cols,
            missing=missing, kind='column')

        selected_attrs = self._resolve_name_selection(
            attr_order, include=include_attrs, exclude=exclude_attrs,
            missing=missing, kind='attribute')
        if include_attrs is None and exclude_attrs is None:
            selected_attrs = attr_order
        if include_attrs is None and exclude_attrs is None:
            selected_attrs = attr_order

        out = {
            'columns': {name: self._data[name] for name in selected_columns},
            'attrs': {
                name: self._serialize_attr_value(raw_attrs[name])
                for name in selected_attrs
            },
        }

        if include_meta:
            dropped_columns = [name for name in column_order if name not in selected_columns]
            dropped_attrs = [name for name in attr_order if name not in selected_attrs]
            meta = {}
            if dropped_columns:
                meta['dropped_columns'] = dropped_columns
            if dropped_attrs:
                meta['dropped_attrs'] = dropped_attrs
            if meta:
                out['meta'] = meta

        extra = self._extra_metadata()
        if extra:
            out.update(extra)
            if include_meta:
                out.setdefault('meta', {}).update({k: v for k, v in extra.items() if k not in out['meta']})

        return out

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]):
        """Construct a table from its dictionary representation."""

        payload = dict(dct)
        table_class_name = payload.get('__class__')
        xtrack_version = payload.get('xtrack_version')
        cls._strip_extra_metadata(payload)

        columns_src = dict(payload['columns'])
        attrs_src = dict(payload.get('attrs', {}))

        converted_columns = {}
        for name, value in columns_src.items():
            if not isinstance(value, np.ndarray):
                converted_columns[name] = np.array(value)
            else:
                converted_columns[name] = value

        converted_attrs = {}
        for name, value in attrs_src.items():
            converted_attrs[name] = cls._deserialize_attr_value(value)

        data = converted_columns | converted_attrs
        instance = cls(data=data, col_names=list(columns_src.keys()))

        if table_class_name is not None:
            instance._data['__class__'] = table_class_name
        if xtrack_version is not None:
            instance._data['xtrack_version'] = xtrack_version
        return instance

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------
    def to_json(self, file, indent=1, **kwargs):
        """Dump the table to JSON using the xtrack JSON utilities."""
        json_utils.dump(self.to_dict(**kwargs), file, indent=indent)

    @classmethod
    def from_json(cls, file):
        """Load a serialized table from a JSON file or file-like object."""
        if isinstance(file, io.IOBase):
            dct = json.load(file)
        else:
            with open(file, 'r') as fid:
                dct = json.load(fid)
        return cls.from_dict(dct)

    # ------------------------------------------------------------------
    # HDF5 helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_hdf5_target(file, mode, group):
        """Return the HDF5 target object and owning file handle."""
        try:
            import h5py  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "h5py is required for Table HDF5 serialization"
            ) from exc

        close_file = False
        if isinstance(file, (str, os.PathLike)):
            h5file = h5py.File(file, mode)
            close_file = True
            base = h5file
        elif hasattr(file, 'read') or hasattr(file, 'write'):
            if hasattr(file, 'seek'):
                file.seek(0)
            h5file = h5py.File(file, mode)
            close_file = True
            base = h5file
        elif isinstance(file, h5py.File):
            h5file = file
            base = h5file
        elif isinstance(file, h5py.Group):
            h5file = None
            base = file
        else:  # pragma: no cover - defensive programming
            raise TypeError(
                "Unsupported file type for HDF5 serialization: "
                f"{type(file)!r}"
            )

        if group is None:
            target = base
        else:
            target = base.require_group(group)

        return target, h5file, close_file

    @staticmethod
    def _needs_json_serialization(column_array):
        """Determine whether column values require JSON encoding."""
        for val in column_array:
            if isinstance(val, np.ndarray):
                if val.ndim > 0:
                    return True
            elif isinstance(val, (list, tuple, dict)):
                return True
        return False

    @staticmethod
    def _serialize_json_value(value):
        """Serialize complex objects to a JSON string payload."""
        buffer = io.StringIO()
        if isinstance(value, np.ndarray):
            json_utils.dump(value.tolist(), buffer, indent=None)
        else:
            json_utils.dump(value, buffer, indent=None)
        return buffer.getvalue()

    @staticmethod
    def _serialize_csv_value(value):
        """Convert a value into a CSV-friendly scalar."""
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            return 'null'
        if isinstance(value, float) or isinstance(value, np.floating):
            if math.isnan(value):
                return 'nan'
        if isinstance(value, complex):
            return str(value)
        return value

    @staticmethod
    def _cast_csv_column(values, dtype_str):
        """Convert CSV column strings back into numpy arrays."""
        if dtype_str is None:
            return np.array(values)

        np_dtype = np.dtype(dtype_str)
        kind = np_dtype.kind

        if kind in ('U', 'S'):
            return np.array(values, dtype=str)

        if kind == 'O':
            return np.array(values, dtype=object)

        if kind == 'b':
            converted = []
            for val in values:
                if isinstance(val, str):
                    val_lower = val.lower()
                    if val_lower in ('true', '1', 'yes'):
                        converted.append(True)
                    elif val_lower in ('false', '0', 'no', ''):
                        converted.append(False)
                    else:
                        raise ValueError(f"Cannot parse boolean value {val!r}")
                else:
                    converted.append(bool(val))
            return np.array(converted, dtype=bool)

        if kind == 'c':
            converted = []
            for val in values:
                if val in ('', None) or (isinstance(val, str) and val.lower() in ('nan', 'null')):
                    converted.append(complex(np.nan, np.nan))
                else:
                    converted.append(complex(val))
            return np.array(converted, dtype=np_dtype)

        if kind in ('i', 'u'):
            converted = []
            has_missing = False
            for val in values:
                if val in ('', None) or (isinstance(val, str) and val.lower() in ('nan', 'null')):
                    has_missing = True
                    converted.append(np.nan)
                else:
                    converted.append(int(float(val)))
            if has_missing:
                return np.array(converted, dtype=float)
            return np.array(converted, dtype=np_dtype)

        if kind == 'f':
            converted = [np.nan if (val in ('', None) or (isinstance(val, str) and val.lower() in ('nan', 'null')))
                         else float(val) for val in values]
            return np.array(converted, dtype=np_dtype)

        return np.array(values)

    @staticmethod
    def _determine_tfs_column_type(values):
        arr = np.asarray(values)
        kind = arr.dtype.kind
        if kind in 'f':
            return '%le', 'float'
        if kind in 'i':
            return '%d', 'int'
        if kind in 'u':
            return '%d', 'int'
        if kind == 'b':
            return '%b', 'bool'
        if kind == 'c':
            return '%lz', 'complex'

        arr_obj = np.asarray(values, dtype=object).ravel()
        non_null = []
        for val in arr_obj:
            if val is None:
                continue
            if isinstance(val, (float, np.floating)) and math.isnan(val):
                continue
            if isinstance(val, str) and val == '':
                continue
            non_null.append(val)

        if not non_null:
            return '%le', 'float'

        if all(isinstance(val, (bool, np.bool_)) for val in non_null):
            return '%b', 'bool'

        all_number_like = True
        treat_as_float = False
        treat_as_complex = False
        for val in non_null:
            if isinstance(val, (bool, np.bool_)):
                continue
            if isinstance(val, numbers.Real):
                fval = float(val)
                if math.isnan(fval) or math.isinf(fval):
                    treat_as_float = True
                elif not math.isclose(fval, round(fval), rel_tol=0.0, abs_tol=1e-12):
                    treat_as_float = True
                if isinstance(val, (float, np.floating)):
                    treat_as_float = True
            elif isinstance(val, numbers.Complex):
                treat_as_complex = True
            else:
                all_number_like = False
                break

        if treat_as_complex:
            return '%lz', 'complex'

        if all_number_like:
            if treat_as_float:
                return '%le', 'float'
            return '%d', 'int'

        return '%s', 'string'

    @staticmethod
    def _tfs_type_token(values):
        token, _ = Table._determine_tfs_column_type(values)
        return token

    @staticmethod
    def _format_tfs_header_value(value):
        if value is None:
            return '%s', 'null'
        if isinstance(value, (bool, np.bool_)):
            return '%b', '1' if bool(value) else '0'
        if isinstance(value, (int, np.integer)):
            return '%d', str(int(value))
        if isinstance(value, (float, np.floating)):
            return '%le', f"{float(value):.16g}"
        if isinstance(value, (complex, np.complexfloating)):
            real = float(value.real)
            imag = float(value.imag)
            real_part = f"{real:.16g}"
            imag_part = f"{abs(imag):.16g}"
            sign = '+' if imag >= 0 else '-'
            return '%lz', f"{real_part}{sign}{imag_part}i"
        if isinstance(value, str):
            if ' ' in value:
                return '%s', f'"{value}"'
            return '%s', value
        if isinstance(value, np.ndarray):
            serialized = Table._serialize_json_value(value)
            return '%s', f'"{serialized}"'
        if isinstance(value, (list, dict)):
            serialized = Table._serialize_json_value(value)
            return '%s', f'"{serialized}"'
        serialized = Table._serialize_json_value(value)
        return '%s', f'"{serialized}"'

    @staticmethod
    def _parse_tfs_value(token, value):
        token = token.lower()
        if isinstance(value, str) and value.lower() == 'null':
            return None
        if token in ('%le', '%lf', '%e', '%f'):
            return float(value)
        if token in ('%d', '%hd', '%ld', '%i', '%u'):
            return int(float(value))
        if token == '%b':
            return bool(int(value))
        if token == '%lz':
            if isinstance(value, str):
                stripped = value.strip().replace(' ', '')
                if stripped.endswith(('i', 'I')):
                    converted = stripped[:-1] + 'j'
                    try:
                        return complex(converted)
                    except ValueError:
                        pass
                try:
                    return complex(value)
                except ValueError:
                    return complex(stripped)
            return complex(value)
        if token == '%s':
            if isinstance(value, str) and len(value) >= 2 and value[0] == value[-1] == '"':
                return value[1:-1]
            return value
        return value

    def to_hdf5(self, file, *, include=None, exclude=None,
                missing='error', include_meta=True, group=None):
        """Persist the table into an HDF5 file or group."""

        target, h5file, close_file = self._resolve_hdf5_target(
            file, mode='w', group=group)

        try:
            column_order = list(self._col_names)
            raw_attrs = {kk: vv for kk, vv in self._data.items()
                         if kk not in column_order}
            raw_attrs.pop('_action', None)
            raw_attrs.pop('_col_names', None)
            attr_order = list(raw_attrs.keys())

            include_cols, include_attrs, exclude_cols, exclude_attrs = (
                self._split_include_exclude(include, exclude,
                                            column_order, attr_order, missing)
            )

            selected_columns = self._resolve_name_selection(
                column_order, include=include_cols, exclude=exclude_cols,
                missing=missing, kind='column')

            selected_attrs = self._resolve_name_selection(
                attr_order, include=include_attrs, exclude=exclude_attrs,
                missing=missing, kind='attribute')

            data_attrs = {name: self._serialize_attr_value(raw_attrs[name])
                          for name in selected_attrs}

            meta_data = {}
            if include_meta:
                dropped_columns = [name for name in column_order
                                   if name not in selected_columns]
                dropped_attrs = [name for name in attr_order
                                 if name not in selected_attrs]
                if dropped_columns:
                    meta_data['dropped_columns'] = dropped_columns
                if dropped_attrs:
                    meta_data['dropped_attrs'] = dropped_attrs

            dtype_info = {
                name: np.asarray(self._data[name]).dtype.str
                for name in selected_columns
            }
            meta_data['column_dtypes'] = dtype_info

            extra = self._extra_metadata()
            if extra:
                meta_data.update(extra)

            column_serialization = {}

            import h5py
            string_dtype = h5py.string_dtype(encoding='utf-8')

            for key in ['columns', 'attrs', 'meta', 'payload']:
                if key in target:
                    del target[key]

            columns_grp = target.create_group('columns')
            columns_grp.attrs['order'] = np.array(selected_columns, dtype='S')

            for name in selected_columns:
                raw_array = np.asarray(self._data[name], dtype=object)
                if raw_array.ndim == 0:
                    raw_array = np.array([raw_array])

                if self._needs_json_serialization(raw_array):
                    column_serialization[name] = 'json'
                    serialized = [self._serialize_json_value(val)
                                  for val in raw_array]
                    columns_grp.create_dataset(name, data=serialized,
                                               dtype=string_dtype)
                else:
                    array = np.asarray(self._data[name])
                    if array.dtype == object:
                        str_values = [self._serialize_csv_value(val)
                                      for val in array]
                        columns_grp.create_dataset(name, data=str_values,
                                                   dtype=string_dtype)
                    elif array.dtype.kind in ('U', 'S'):
                        columns_grp.create_dataset(name, data=array.astype('U'),
                                                   dtype=string_dtype)
                    else:
                        columns_grp.create_dataset(name, data=array)

            if column_serialization:
                meta_data['column_serialization'] = column_serialization

            attrs_serialization = {}

            attrs_grp = target.create_group('attrs') if data_attrs else None
            if attrs_grp is not None:
                attrs_grp.attrs['order'] = np.array(selected_attrs, dtype='S')
                for name, value in data_attrs.items():
                    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
                        attrs_grp.create_dataset(name, data=value)
                    elif isinstance(value, (str, np.str_)):
                        attrs_grp.create_dataset(name, data=np.array(value, dtype='S'), dtype=string_dtype)
                    elif isinstance(value, np.ndarray):
                        attrs_grp.create_dataset(name, data=value)
                    else:
                        serialized = self._serialize_json_value(value)
                        attrs_serialization[name] = 'json'
                        attrs_grp.create_dataset(name, data=serialized, dtype=string_dtype)

            if attrs_serialization:
                meta_data['attrs_serialization'] = attrs_serialization

            if meta_data:
                meta_grp = target.create_group('meta')
                for name, value in meta_data.items():
                    buffer = io.StringIO()
                    json_utils.dump(value, buffer, indent=None)
                    meta_grp.create_dataset(name, data=buffer.getvalue(),
                                             dtype=string_dtype)
        finally:
            if close_file and h5file is not None:
                h5file.close()

    @classmethod
    def from_hdf5(cls, file, *, group=None):
        """Load a table from an HDF5 file or group."""

        if group is None:
            import h5py

            with h5py.File(file, 'r') as h5:
                group_candidates = [
                    name for name, obj in h5.items()
                    if isinstance(obj, h5py.Group)
                ]
                if len(group_candidates) != 1:
                    raise ValueError(
                        'HDF5 file must contain exactly one group when `group` is not provided'
                    )
                group = group_candidates[0]

        target, h5file, close_file = cls._resolve_hdf5_target(
            file, mode='r', group=group)

        try:
            if 'columns' not in target:
                raise KeyError("HDF5 group does not contain 'columns' subgroup")

            import h5py

            meta_data = {}
            if 'meta' in target:
                meta_grp = target['meta']
                for name in meta_grp.keys():
                    ds = meta_grp[name]
                    serialized = ds.asstr()[()] if isinstance(ds, h5py.Dataset) else ds[()]
                    if isinstance(serialized, np.ndarray) and serialized.shape == ():
                        serialized = serialized.item()
                    if isinstance(serialized, bytes):
                        serialized = serialized.decode('utf-8')
                    meta_data[name] = json_utils.load(string=serialized)

            dtype_info = meta_data.get('column_dtypes', {}) if isinstance(meta_data, dict) else {}
            column_serialization = meta_data.get('column_serialization', {}) if isinstance(meta_data, dict) else {}

            columns_grp = target['columns']
            if 'order' in columns_grp.attrs:
                order_raw = columns_grp.attrs['order']
                column_order = [
                    item.decode('utf-8') if isinstance(item, (bytes, bytearray))
                    else str(item)
                    for item in order_raw
                ]
            else:
                column_order = list(columns_grp.keys())

            columns_data = {}
            for name in column_order:
                ds = columns_grp[name]
                if isinstance(ds, h5py.Dataset) and ds.dtype.kind in ('S', 'O'):
                    values = ds.asstr()[()]
                else:
                    values = ds[()]

                if isinstance(values, np.ndarray) and values.shape == ():
                    values = values.item()

                if column_serialization.get(name) == 'json':
                    if isinstance(values, str):
                        raw_list = [values]
                    elif isinstance(values, np.ndarray):
                        raw_list = values.tolist()
                    else:
                        raw_list = list(values)

                    parsed = []
                    for item in raw_list:
                        if item in ('', None):
                            parsed.append(None)
                        else:
                            obj = json_utils.load(string=item)
                            if isinstance(obj, list):
                                parsed.append(np.array(obj))
                            else:
                                parsed.append(obj)
                    columns_data[name] = np.array(parsed)
                else:
                    if isinstance(values, np.ndarray) and values.dtype.kind not in ('S', 'O'):
                        columns_data[name] = values
                    else:
                        if isinstance(values, str):
                            raw_list = [values]
                        elif isinstance(values, np.ndarray):
                            raw_list = values.tolist()
                        else:
                            raw_list = list(values)
                        columns_data[name] = cls._cast_csv_column(raw_list, dtype_info.get(name))

            attrs_serialization = meta_data.get('attrs_serialization', {}) if isinstance(meta_data, dict) else {}

            attrs_data = {}
            if 'attrs' in target:
                attrs_grp = target['attrs']
                if 'order' in attrs_grp.attrs:
                    attr_order_raw = attrs_grp.attrs['order']
                    attr_order = [
                        item.decode('utf-8') if isinstance(item, (bytes, bytearray))
                        else str(item)
                        for item in attr_order_raw
                    ]
                else:
                    attr_order = list(attrs_grp.keys())

                for name in attr_order:
                    ds = attrs_grp[name]
                    if attrs_serialization.get(name) == 'json':
                        serialized = ds.asstr()[()]
                        attrs_data[name] = json_utils.load(string=serialized)
                        continue

                    value = ds[()]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    if isinstance(value, np.ndarray) and value.shape == ():
                        value = value.item()
                    attrs_data[name] = value

            data = columns_data | attrs_data
            if isinstance(meta_data, dict) and meta_data:
                table_class_name = meta_data.get('__class__')
                xtrack_version = meta_data.get('xtrack_version')
                attrs_data['__class__'] = table_class_name
                attrs_data['xtrack_version'] = xtrack_version

            return cls.from_dict({'columns': columns_data, 'attrs': attrs_data})
        finally:
            if close_file and h5file is not None:
                h5file.close()

    def to_csv(self, file, *, include=None, exclude=None,
               missing='error', include_meta=True):
        """Write the table to CSV, embedding metadata as comments."""

        column_order = list(self._col_names)
        raw_attrs = {kk: vv for kk, vv in self._data.items()
                     if kk not in column_order}
        raw_attrs.pop('_action', None)
        raw_attrs.pop('_col_names', None)
        attr_order = list(raw_attrs.keys())

        include_cols, include_attrs, exclude_cols, exclude_attrs = (
            self._split_include_exclude(include, exclude,
                                        column_order, attr_order, missing)
        )

        selected_columns = self._resolve_name_selection(
            column_order, include=include_cols, exclude=exclude_cols,
            missing=missing, kind='column')

        selected_attrs = self._resolve_name_selection(
            attr_order, include=include_attrs, exclude=exclude_attrs,
            missing=missing, kind='attribute')

        data_attrs = {name: self._serialize_attr_value(raw_attrs[name])
                      for name in selected_attrs}

        meta_data = {}
        if include_meta:
            dropped_columns = [name for name in column_order
                               if name not in selected_columns]
            dropped_attrs = [name for name in attr_order
                             if name not in selected_attrs]
            if dropped_columns:
                meta_data['dropped_columns'] = dropped_columns
            if dropped_attrs:
                meta_data['dropped_attrs'] = dropped_attrs

        dtype_info = {
            name: np.asarray(self._data[name]).dtype.str
            for name in selected_columns
        }
        meta_data['column_dtypes'] = dtype_info
        column_serialization = {}

        extra = self._extra_metadata()
        if extra:
            meta_data.update(extra)

        if isinstance(file, io.IOBase):
            fh = file
            close_file = False
        else:
            fh = open(file, 'w', newline='')
            close_file = True

        try:
            column_arrays = []
            for name in selected_columns:
                array = np.asarray(self._data[name], dtype=object)
                if array.ndim == 0:
                    array = np.array([array])
                if self._needs_json_serialization(array):
                    column_serialization[name] = 'json'
                column_arrays.append(array)

            if column_serialization:
                meta_data['column_serialization'] = column_serialization

            header_tag = getattr(self, '_csv_header_tag', self.__class__.__name__)
            fh.write(f'# {header_tag}\n')

            if data_attrs:
                buffer = io.StringIO()
                json_utils.dump(data_attrs, buffer, indent=None)
                fh.write('# attrs=' + buffer.getvalue() + '\n')

            if meta_data:
                buffer = io.StringIO()
                json_utils.dump(meta_data, buffer, indent=None)
                fh.write('# meta=' + buffer.getvalue() + '\n')

            writer = csv.writer(fh)
            writer.writerow(selected_columns)

            if column_arrays:
                row_count = len(column_arrays[0])
                for array in column_arrays[1:]:
                    if len(array) != row_count:
                        raise ValueError('All column arrays must have the same length')

                for idx in range(row_count):
                    row = []
                    for name, col in zip(selected_columns, column_arrays):
                        value = col[idx]
                        if column_serialization.get(name) == 'json':
                            row.append(self._serialize_json_value(value))
                        else:
                            row.append(self._serialize_csv_value(value))
                    writer.writerow(row)
        finally:
            if close_file:
                fh.close()

    @classmethod
    def from_csv(cls, file):
        """Reconstruct a table instance from CSV data."""
        if isinstance(file, io.IOBase):
            content = file.read().splitlines()
        else:
            with open(file, 'r', newline='') as fh:
                content = fh.read().splitlines()

        attrs_payload = {}
        meta_payload = {}
        data_lines = []

        for line in content:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('#'):
                payload = stripped[1:].strip()
                if payload.lower() == cls.__name__.lower():
                    continue
                if payload.startswith('attrs='):
                    attrs_payload = json_utils.load(string=payload[6:])
                elif payload.startswith('meta='):
                    meta_payload = json_utils.load(string=payload[5:])
                continue
            data_lines.append(line)

        if not data_lines:
            raise ValueError('CSV file does not contain data rows')

        reader = csv.reader(io.StringIO('\n'.join(data_lines)))
        rows = list(reader)
        if not rows:
            raise ValueError('CSV file has no rows after comments')

        header = rows[0]
        data_rows = rows[1:]

        columns_values = {name: [] for name in header}
        for row in data_rows:
            if len(row) < len(header):
                row = row + [''] * (len(header) - len(row))
            elif len(row) > len(header):
                raise ValueError('Row has more fields than header columns')
            for name, value in zip(header, row):
                columns_values[name].append(value)

        dtype_info = meta_payload.get('column_dtypes', {}) if isinstance(meta_payload, dict) else {}
        column_serialization = meta_payload.get('column_serialization', {}) if isinstance(meta_payload, dict) else {}

        columns_data = {}
        for name in header:
            dtype_str = dtype_info.get(name)
            if column_serialization.get(name) == 'json':
                parsed_vals = []
                for val in columns_values[name]:
                    if val == '' or val is None:
                        parsed_vals.append(None)
                    else:
                        obj = json_utils.load(string=val)
                        if isinstance(obj, list):
                            parsed_vals.append(np.array(obj))
                        else:
                            parsed_vals.append(obj)
                columns_data[name] = np.array(parsed_vals)
            else:
                columns_data[name] = cls._cast_csv_column(columns_values[name], dtype_str)

        if isinstance(meta_payload, dict) and meta_payload:
            if '__class__' in meta_payload:
                attrs_payload['__class__'] = meta_payload['__class__']
            if 'xtrack_version' in meta_payload:
                attrs_payload['xtrack_version'] = meta_payload['xtrack_version']

        instance = cls.from_dict({
            'columns': columns_data,
            'attrs': attrs_payload,
        })
        return instance

    def to_tfs(self, file, *, include=None, exclude=None,
               missing='error', include_meta=True,
               default_column_width=None, float_precision=8,
               numeric_column_width=16, column_formats=None,
               column_widths=None):
        """Write the table in TFS format.

        Parameters
        ----------
        file : path-like or file-like
            Output target.
        include, exclude, missing, include_meta
            See :meth:`Table.to_dict` for details.
        default_column_width : int, optional
            Minimum column width to enforce for headers and data cells.
        float_precision : int, optional
            Significant digits used when writing floating-point values.
        numeric_column_width : int, optional
            If provided, enforces this uniform width for all numeric columns.
            When omitted, numeric columns still share a common width derived
            from the widest numeric entry.
        column_formats : Mapping[str, str], optional
            Per-column Python/C-style format specifiers (e.g. '.3f', '10.4g').
            Applied to data cells while falling back to ``float_precision``
            defaults when unspecified.
        column_widths : Mapping[str, int], optional
            Per-column minimum widths overriding the defaults. Non-numeric
            columns stay left-aligned; numeric ones keep right alignment.
        """

        if float_precision <= 0:
            raise ValueError('float_precision must be a positive integer')

        if numeric_column_width is not None and numeric_column_width <= 0:
            raise ValueError('numeric_column_width must be a positive integer')

        if column_formats is None:
            column_format_overrides: Dict[str, str] = {}
        elif isinstance(column_formats, Mapping):
            column_format_overrides = {
                str(key): str(value) for key, value in column_formats.items()
            }
        else:
            raise TypeError('column_formats must be a mapping of column names to format strings')

        if column_widths is None:
            column_width_overrides: Dict[str, int] = {}
        elif isinstance(column_widths, Mapping):
            column_width_overrides = {}
            for key, value in column_widths.items():
                if not isinstance(value, numbers.Integral):
                    raise TypeError('column_widths values must be integers')
                if value <= 0:
                    raise ValueError('column_widths values must be positive integers')
                column_width_overrides[str(key)] = int(value)
        else:
            raise TypeError('column_widths must be a mapping of column names to integer widths')

        column_order = list(self._col_names)
        raw_attrs = {kk: vv for kk, vv in self._data.items() if kk not in column_order}
        raw_attrs.pop('_action', None)
        raw_attrs.pop('_col_names', None)
        attr_order = list(raw_attrs.keys())

        meta_filterable_keys = {
            'dropped_columns', 'dropped_attrs', 'column_dtypes',
            '__class__', 'xtrack_version'
        }

        raw_include_set = self._normalize_name_selection(include)
        raw_exclude_set = self._normalize_name_selection(exclude)

        if raw_include_set is not None:
            include_meta_set = {name.lower() for name in raw_include_set
                                if name.lower() in meta_filterable_keys}
            include_for_split = {name for name in raw_include_set
                                 if name.lower() not in meta_filterable_keys}
            include_arg = include_for_split if include_for_split else None
            include_meta = include_meta_set if include_meta_set else None
        else:
            include_meta = None
            include_arg = None

        if raw_exclude_set:
            exclude_meta = {name.lower() for name in raw_exclude_set
                            if name.lower() in meta_filterable_keys}
            exclude_for_split = {name for name in raw_exclude_set
                                 if name.lower() not in meta_filterable_keys}
            exclude_arg = exclude_for_split
        else:
            exclude_meta = set()
            exclude_arg = None

        include_cols, include_attrs, exclude_cols, exclude_attrs = (
            self._split_include_exclude(include_arg, exclude_arg,
                                        column_order, attr_order, missing)
        )

        selected_columns = self._resolve_name_selection(
            column_order, include=include_cols, exclude=exclude_cols,
            missing=missing, kind='column')

        selected_attrs = self._resolve_name_selection(
            attr_order, include=include_attrs, exclude=exclude_attrs,
            missing=missing, kind='attribute')

        data_attrs = {name: self._serialize_attr_value(raw_attrs[name])
                      for name in selected_attrs}

        column_arrays = []
        column_types = []
        column_categories = []
        column_serialization = {}
        column_format_specs = []
        for name in selected_columns:
            values = self._data[name]
            array = np.asarray(values, dtype=object)
            if array.ndim == 0:
                array = np.array([array], dtype=object)
            column_arrays.append(array)
            if self._needs_json_serialization(array):
                column_serialization[name] = 'json'
                token, category = '%s', 'string'
            else:
                token, category = self._determine_tfs_column_type(values)
            column_types.append(token)
            column_categories.append(category)
            column_format_specs.append(column_format_overrides.get(name))

        meta_data = {}
        if include_meta:
            dropped_columns = [name for name in column_order
                               if name not in selected_columns]
            dropped_attrs = [name for name in attr_order
                             if name not in selected_attrs]
            if dropped_columns:
                meta_data['dropped_columns'] = dropped_columns
            if dropped_attrs:
                meta_data['dropped_attrs'] = dropped_attrs

        if column_serialization:
            meta_data['column_serialization'] = column_serialization

        attrs_serialization = {}
        extra = self._extra_metadata()
        if extra:
            meta_data.update(extra)

        attr_lines = []
        for name, value in data_attrs.items():
            dtype_token, formatted = self._format_tfs_header_value(value)
            if dtype_token == '%s' and not isinstance(value, (str, np.str_)):
                attrs_serialization[name] = 'json'
            attr_lines.append((name.upper(), dtype_token, formatted))

        if attrs_serialization:
            meta_data['attrs_serialization'] = attrs_serialization

        if include_meta is not None:
            filtered_meta = {}
            for key, value in meta_data.items():
                key_lower = key.lower()
                if key_lower in meta_filterable_keys:
                    if key_lower in include_meta:
                        filtered_meta[key] = value
                else:
                    filtered_meta[key] = value
            meta_data = filtered_meta
        elif exclude_meta:
            filtered_meta = {}
            for key, value in meta_data.items():
                key_lower = key.lower()
                if key_lower in meta_filterable_keys and key_lower in exclude_meta:
                    continue
                filtered_meta[key] = value
            meta_data = filtered_meta

        ordered_meta_keys = ['__class__', 'xtrack_version']
        meta_lines = []
        append_seen = set()
        for ordered_key in ordered_meta_keys:
            if ordered_key in meta_data:
                dtype_token, formatted = self._format_tfs_header_value(meta_data[ordered_key])
                meta_lines.append((ordered_key.upper(), dtype_token, formatted))
                append_seen.add(ordered_key)
        for key, value in meta_data.items():
            key_lower = key.lower()
            if key_lower in append_seen:
                continue
            dtype_token, formatted = self._format_tfs_header_value(value)
            meta_lines.append((key.upper(), dtype_token, formatted))

        header_entries = attr_lines + meta_lines
        if header_entries:
            name_width = max(len(name) for name, _, _ in header_entries)
            token_width = max(len(token) for _, token, _ in header_entries)
        else:
            name_width = 0
            token_width = 0

        if column_arrays:
            row_count = len(column_arrays[0])
            for array in column_arrays[1:]:
                if len(array) != row_count:
                    raise ValueError('All column arrays must have the same length')

            column_cells = []
            column_align_left = []
            for name, array, token, category, fmt_spec in zip(
                    selected_columns, column_arrays, column_types,
                    column_categories, column_format_specs):
                align_left = token.lower() == '%s'
                column_align_left.append(align_left)
                cells = []
                use_json = column_serialization.get(name) == 'json'
                for value in array:
                    if use_json:
                        if value is None:
                            cells.append('null')
                        else:
                            json_string = self._serialize_json_value(value)
                            cells.append(json.dumps(json_string))
                        continue

                    if value is None:
                        cells.append('null')
                        continue

                    if category == 'float':
                        if fmt_spec:
                            try:
                                cells.append(format(float(value), fmt_spec))
                            except (ValueError, TypeError):
                                cells.append(f"{float(value):.{float_precision}g}")
                        else:
                            cells.append(f"{float(value):.{float_precision}g}")
                        continue

                    if category == 'int':
                        try:
                            int_value = int(value)
                        except (TypeError, ValueError):
                            int_value = int(float(value))
                        if fmt_spec:
                            try:
                                cells.append(format(int_value, fmt_spec))
                            except (ValueError, TypeError):
                                cells.append(str(int_value))
                        else:
                            cells.append(str(int_value))
                        continue

                    if category == 'bool':
                        bool_int = 1 if bool(value) else 0
                        if fmt_spec:
                            try:
                                cells.append(format(bool_int, fmt_spec))
                            except (ValueError, TypeError):
                                cells.append('1' if bool(value) else '0')
                        else:
                            cells.append('1' if bool(value) else '0')
                        continue

                    if category == 'complex':
                        comp_val = complex(value)
                        if fmt_spec:
                            try:
                                real_part = format(comp_val.real, fmt_spec)
                                imag_part = format(abs(comp_val.imag), fmt_spec)
                            except (ValueError, TypeError):
                                real_part = f"{comp_val.real:.{float_precision}g}"
                                imag_part = f"{abs(comp_val.imag):.{float_precision}g}"
                        else:
                            real_part = f"{comp_val.real:.{float_precision}g}"
                            imag_part = f"{abs(comp_val.imag):.{float_precision}g}"
                        sign = '+' if comp_val.imag >= 0 else '-'
                        cells.append(f"{real_part}{sign}{imag_part}i")
                        continue

                    if category == 'string':
                        string_value = str(value)
                        if fmt_spec:
                            try:
                                string_value = format(string_value, fmt_spec)
                            except (ValueError, TypeError):
                                pass
                        if not (
                            string_value.startswith('"') and string_value.endswith('"')
                        ):
                            string_value = f'"{string_value}"'
                        cells.append(string_value)
                        continue

                    if isinstance(value, (str, np.str_)):
                        string_value = str(value)
                        if not (
                            string_value.startswith('"') and string_value.endswith('"')
                        ):
                            string_value = f'"{string_value}"'
                        cells.append(string_value)
                        continue

                    string_value = str(value)
                    if ' ' in string_value and not (
                        string_value.startswith('"') and string_value.endswith('"')
                    ):
                        string_value = f'"{string_value}"'
                    cells.append(string_value)
                column_cells.append(cells)

            column_widths = []
            column_width_override_flags = []
            for idx, name in enumerate(selected_columns):
                upper_name = name.upper()
                width_candidates = [len(upper_name), len(column_types[idx])]
                width_candidates.extend(len(val) for val in column_cells[idx])
                min_width = default_column_width or 0
                computed_width = max(width_candidates + [min_width])
                override_width = column_width_overrides.get(name)
                if override_width is not None:
                    computed_width = max(computed_width, override_width)
                if not column_align_left[idx] and numeric_column_width is not None:
                    computed_width = max(computed_width, numeric_column_width)
                column_widths.append(computed_width)
                column_width_override_flags.append(override_width is not None)

            if numeric_column_width is None:
                numeric_widths = [w for w, left, overridden in zip(
                    column_widths, column_align_left, column_width_override_flags)
                    if not left and not overridden]
                if numeric_widths:
                    uniform_width = max(numeric_widths)
                    column_widths = [
                        uniform_width if (not left and not overridden) else w
                        for w, left, overridden in zip(
                            column_widths, column_align_left, column_width_override_flags)
                    ]
        else:
            row_count = 0
            column_cells = []
            column_align_left = [token.lower() == '%s' for token in column_types]
            column_width_override_flags = []
            column_widths = []
            for idx, (name, token) in enumerate(zip(selected_columns, column_types)):
                min_width = default_column_width or 0
                upper_name = name.upper()
                base_width = max(len(upper_name), len(token), min_width)
                override_width = column_width_overrides.get(name)
                if override_width is not None:
                    base_width = max(base_width, override_width)
                if token.lower() != '%s' and numeric_column_width is not None:
                    base_width = max(base_width, numeric_column_width)
                column_widths.append(base_width)
                column_width_override_flags.append(override_width is not None)

            if numeric_column_width is None:
                numeric_widths = [w for w, token, overridden in zip(
                    column_widths, column_types, column_width_override_flags)
                    if token.lower() != '%s' and not overridden]
                if numeric_widths:
                    uniform_width = max(numeric_widths)
                    column_widths = [
                        uniform_width if (token.lower() != '%s' and not overridden) else w
                        for w, token, overridden in zip(
                            column_widths, column_types, column_width_override_flags)
                    ]

        if isinstance(file, io.IOBase):
            fh = file
            close_file = False
        else:
            fh = open(file, 'w')
            close_file = True

        try:
            for name, token, formatted in meta_lines:
                fh.write(
                    f"@ {name:<{name_width}} {token:<{token_width}} {formatted}\n"
                    if name_width and token_width else f"@ {name} {token} {formatted}\n"
                )

            for name, token, formatted in attr_lines:
                fh.write(
                    f"@ {name:<{name_width}} {token:<{token_width}} {formatted}\n"
                    if name_width and token_width else f"@ {name} {token} {formatted}\n"
                )

            if selected_columns:
                header_names = []
                header_types = []
                for idx, name in enumerate(selected_columns):
                    width = column_widths[idx]
                    upper_name = name.upper()
                    if column_align_left[idx]:
                        header_names.append(upper_name.ljust(width))
                        header_types.append(column_types[idx].ljust(width))
                    else:
                        header_names.append(upper_name.rjust(width))
                        header_types.append(column_types[idx].rjust(width))
                fh.write('* ' + ' '.join(header_names).rstrip() + '\n')
                fh.write('$ ' + ' '.join(header_types).rstrip() + '\n')

            for row_idx in range(row_count):
                row_cells = []
                for col_idx in range(len(selected_columns)):
                    value = column_cells[col_idx][row_idx]
                    width = column_widths[col_idx]
                    if column_align_left[col_idx]:
                        row_cells.append(value.ljust(width))
                    else:
                        row_cells.append(value.rjust(width))
                fh.write(' ' + ' '.join(row_cells).rstrip() + '\n')
        finally:
            if close_file:
                fh.close()

    @classmethod
    def from_tfs(cls, file):
        """Load a table from a TFS file."""
        header_text, file = _prepare_header_source(file)
        parsed_headers = _parse_headers(header_text)

        try:
            import tfs
        except ImportError:
            raise ImportError('Please install tfs-pandas to read TFS files.')

        # If I get a StringIO I need to make a temporary file (to be fixed in tfs)
        if isinstance(file, io.StringIO):
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            tfs_table = tfs.read(tmp_path)
            os.remove(tmp_path)
        else:
            tfs_table = tfs.read(file)

        data = {}
        col_names = []

        contain_capital = ["W_matrix", "R_matrix", "R_matrix_ebe", "T_rev0"]
        rename_dict = {cc.lower(): cc for cc in contain_capital}

        for cc in tfs_table.columns:
            cc_lower = cc.lower()
            if cc_lower in rename_dict:
                cc_lower = rename_dict[cc_lower]
            col_names.append(cc_lower)
            data[cc_lower] = tfs_table[cc].to_numpy()

        for kk, value in parsed_headers.items():
            kk_lower = kk.lower()
            if kk_lower in col_names:
                continue # There is a clash in legacy madx files
            if kk_lower in rename_dict:
                kk_lower = rename_dict[kk_lower]
            data[kk_lower] = value
            if data[kk_lower] == 'null':
                data[kk_lower] = None

        # Temporary solution for multidimensional arrays
        for kk in ['W_matrix', 'R_matrix_ebe']:
            if kk not in data:
                continue
            if isinstance(data[kk], str) and data[kk] == 'null':
                data[kk] = None
            if kk not in data or data[kk] is None:
                continue
            data[kk] = np.array(
                list(map(lambda ss: json_utils.load(string=ss), data[kk])))

        if 'attrs_serialization' in data:
            attrs_serialization = json_utils.load(string=data['attrs_serialization'])
            for kk, ss in attrs_serialization.items():
                if data[kk] is None:
                    continue
                assert ss == 'json'
                data[kk] = json_utils.load(string=data[kk])

        if tmad :=tfs_table.headers.get('TYPE', None):
            if tmad == 'TWISS':
                data['__class__'] = 'TwissTable'

        attr_names_set = set(data.keys()) - set(col_names)
        out = cls.from_dict({'columns': {kk: data[kk] for kk in col_names},
                                'attrs': {kk: data[kk] for kk in attr_names_set}})
        return out
