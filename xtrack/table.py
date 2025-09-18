# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import csv
import io
import json
import math
import os
from typing import Any, Dict, Iterable, Optional

import numpy as np

from xdeps import Table as _XdepsTable

from . import json as json_utils

_PARTICLES_CLS: Optional[type] = None


def _get_particles_cls():
    global _PARTICLES_CLS
    if _PARTICLES_CLS is not None:
        return _PARTICLES_CLS

    try:
        import xtrack as xt  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency during import
        _PARTICLES_CLS = None
    else:
        _PARTICLES_CLS = getattr(xt, 'Particles', None)

    return _PARTICLES_CLS


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
        particles_cls = _get_particles_cls()
        if particles_cls is not None and isinstance(value, particles_cls):
            return value.to_dict()
        return value

    @staticmethod
    def _deserialize_attr_value(value):
        if not isinstance(value, dict):
            return value
        if value.get('__class__', None) != 'Particles':
            return value

        particles_cls = _get_particles_cls()
        if particles_cls is None:
            return value

        return particles_cls.from_dict(value)

    # ------------------------------------------------------------------
    # Generic dictionary export/import
    # ------------------------------------------------------------------
    def _extra_metadata(self) -> Dict[str, Any]:
        class_name = self.__class__.__name__
        return {
            'table_class': class_name,
            '__class__': class_name,
        }

    @classmethod
    def _strip_extra_metadata(cls, payload: Dict[str, Any]) -> None:
        payload.pop('table_class', None)
        payload.pop('__class__', None)

    def to_dict(self, *, columns=None, exclude_columns=None,
                attrs=None, exclude_attrs=None, missing='error',
                include_meta=True):

        column_order = list(self._col_names)
        selected_columns = self._resolve_name_selection(
            column_order, include=columns, exclude=exclude_columns,
            missing=missing, kind='column')

        raw_attrs = {kk: vv for kk, vv in self._data.items() if kk not in column_order}
        raw_attrs.pop('_action', None)
        raw_attrs.pop('_col_names', None)
        attr_order = list(raw_attrs.keys())
        selected_attrs = self._resolve_name_selection(
            attr_order, include=attrs, exclude=exclude_attrs,
            missing=missing, kind='attribute')

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
    def from_dict(cls, dct: Dict[str, Any], *, columns=None,
                  exclude_columns=None, attrs=None,
                  exclude_attrs=None, missing='error'):

        payload = dict(dct)
        table_class_name = payload.get('table_class') or payload.get('__class__')
        cls._strip_extra_metadata(payload)

        columns_src = dict(payload['columns'])
        attrs_src = dict(payload.get('attrs', {}))

        column_order = list(columns_src.keys())
        selected_columns = cls._resolve_name_selection(
            column_order, include=columns, exclude=exclude_columns,
            missing=missing, kind='column')

        attr_order = list(attrs_src.keys())
        selected_attrs = cls._resolve_name_selection(
            attr_order, include=attrs, exclude=exclude_attrs,
            missing=missing, kind='attribute')

        converted_columns = {}
        for name in selected_columns:
            value = columns_src[name]
            if not isinstance(value, np.ndarray):
                converted_columns[name] = np.array(value)
            else:
                converted_columns[name] = value

        converted_attrs = {}
        for name in selected_attrs:
            converted_attrs[name] = cls._deserialize_attr_value(attrs_src[name])

        data = converted_columns | converted_attrs
        instance = cls(data=data, col_names=list(selected_columns))
        if table_class_name:
            instance._data['_table_class'] = table_class_name
        return instance

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------
    def to_json(self, file, indent=1, **kwargs):
        json_utils.dump(self.to_dict(**kwargs), file, indent=indent)

    @classmethod
    def from_json(cls, file):
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
        try:
            import h5py  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "h5py is required for Table HDF5 serialization"
            ) from exc

        import h5py

        close_file = False
        if isinstance(file, (str, os.PathLike)):
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
        for val in column_array:
            if isinstance(val, np.ndarray):
                if val.ndim > 0:
                    return True
            elif isinstance(val, (list, tuple, dict)):
                return True
        return False

    @staticmethod
    def _serialize_json_value(value):
        buffer = io.StringIO()
        if isinstance(value, np.ndarray):
            json_utils.dump(value.tolist(), buffer, indent=None)
        else:
            json_utils.dump(value, buffer, indent=None)
        return buffer.getvalue()

    @staticmethod
    def _serialize_csv_value(value):
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            return ''
        if isinstance(value, float) or isinstance(value, np.floating):
            if math.isnan(value):
                return 'nan'
        if isinstance(value, complex):
            return str(value)
        return value

    @staticmethod
    def _cast_csv_column(values, dtype_str):
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
                if val == '' or (isinstance(val, str) and val.lower() == 'nan'):
                    converted.append(complex(np.nan, np.nan))
                else:
                    converted.append(complex(val))
            return np.array(converted, dtype=np_dtype)

        if kind in ('i', 'u'):
            converted = []
            has_missing = False
            for val in values:
                if val == '' or (isinstance(val, str) and val.lower() == 'nan'):
                    has_missing = True
                    converted.append(np.nan)
                else:
                    converted.append(int(float(val)))
            if has_missing:
                return np.array(converted, dtype=float)
            return np.array(converted, dtype=np_dtype)

        if kind == 'f':
            converted = [np.nan if (val == '' or (isinstance(val, str) and val.lower() == 'nan'))
                         else float(val) for val in values]
            return np.array(converted, dtype=np_dtype)

        return np.array(values)

    def to_hdf5(self, file, *, columns=None, exclude_columns=None,
                attrs=None, exclude_attrs=None, missing='error',
                include_meta=True, group='table'):

        target, h5file, close_file = self._resolve_hdf5_target(
            file, mode='w', group=group)

        try:
            column_order = list(self._col_names)
            selected_columns = self._resolve_name_selection(
                column_order, include=columns, exclude=exclude_columns,
                missing=missing, kind='column')

            raw_attrs = {kk: vv for kk, vv in self._data.items()
                         if kk not in column_order}
            raw_attrs.pop('_action', None)
            raw_attrs.pop('_col_names', None)
            attr_order = list(raw_attrs.keys())
            selected_attrs = self._resolve_name_selection(
                attr_order, include=attrs, exclude=exclude_attrs,
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

            attrs_grp = target.create_group('attrs') if data_attrs else None
            if attrs_grp is not None:
                attrs_grp.attrs['order'] = np.array(selected_attrs, dtype='S')
                for name, value in data_attrs.items():
                    buffer = io.StringIO()
                    json_utils.dump(value, buffer, indent=None)
                    attrs_grp.create_dataset(name, data=buffer.getvalue(),
                                             dtype=string_dtype)

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
    def from_hdf5(cls, file, *, columns=None, exclude_columns=None,
                  attrs=None, exclude_attrs=None, missing='error',
                  group=None):

        if group is None:
            try:
                import h5py  # noqa: F401
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise ModuleNotFoundError(
                    'h5py is required for Table HDF5 deserialization'
                ) from exc

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
                    columns_data[name] = np.array(parsed, dtype=object)
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
                    if isinstance(ds, h5py.Dataset) and ds.dtype.kind in ('S', 'O'):
                        serialized = ds.asstr()[()]
                    else:
                        serialized = ds[()]
                    if isinstance(serialized, np.ndarray) and serialized.shape == ():
                        serialized = serialized.item()
                    if isinstance(serialized, bytes):
                        serialized = serialized.decode('utf-8')
                    attrs_data[name] = json_utils.load(string=serialized)

            data = {
                'columns': columns_data,
                'attrs': attrs_data,
            }
            if isinstance(meta_data, dict) and meta_data:
                data['meta'] = meta_data
                table_class_name = meta_data.get('table_class') or meta_data.get('__class__')
                if table_class_name:
                    data['table_class'] = table_class_name

            return cls.from_dict(
                data,
                columns=columns, exclude_columns=exclude_columns,
                attrs=attrs, exclude_attrs=exclude_attrs,
                missing=missing)
        finally:
            if close_file and h5file is not None:
                h5file.close()

    def to_csv(self, file, *, columns=None, exclude_columns=None,
               attrs=None, exclude_attrs=None, missing='error',
               include_meta=True):

        column_order = list(self._col_names)
        selected_columns = self._resolve_name_selection(
            column_order, include=columns, exclude=exclude_columns,
            missing=missing, kind='column')

        raw_attrs = {kk: vv for kk, vv in self._data.items()
                     if kk not in column_order}
        raw_attrs.pop('_action', None)
        raw_attrs.pop('_col_names', None)
        attr_order = list(raw_attrs.keys())
        selected_attrs = self._resolve_name_selection(
            attr_order, include=attrs, exclude=exclude_attrs,
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
    def from_csv(cls, file, *, columns=None, exclude_columns=None,
                 attrs=None, exclude_attrs=None, missing='error'):

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
                columns_data[name] = np.array(parsed_vals, dtype=object)
            else:
                columns_data[name] = cls._cast_csv_column(columns_values[name], dtype_str)

        data = {
            'columns': columns_data,
            'attrs': attrs_payload,
        }
        if isinstance(meta_payload, dict) and meta_payload:
            data['meta'] = meta_payload
            table_class_name = meta_payload.get('table_class') or meta_payload.get('__class__')
            if table_class_name:
                data['table_class'] = table_class_name

        return cls.from_dict(
            data,
            columns=columns, exclude_columns=exclude_columns,
            attrs=attrs, exclude_attrs=exclude_attrs,
            missing=missing)
